"""取り込み(インデックス作成)。

動画を時間窓に区切り、窓ごとに「複数のフレーム(+音声)」を1ベクトルにして保存します。
考え方は Google AI Edge Gallery の Video Moments Finder と同じです(Apache-2.0)。

  - 窓の長さ・フレーム数・重なり幅は設定できます。
  - 音声を使う窓は、「時刻 → 音声 → 時刻 → 画像 …」の順に並べて、1回の推論で1ベクトルにします。
  - 音声なしの窓は、画像だけを並べます。

音声あり(tav)と音声なし(frames)のベクトルは、同じ空間とは限らないので、種類(kind)を分けて
保存し、検索時に種類で絞り込めるようにしています。
"""

from __future__ import annotations

import logging
import math
import queue
import tempfile
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import documents, media, thumbs
from .config import PRESETS, Settings
from .disk import InsufficientStorage
from .disk import check as check_disk
from .embedders import SAMPLE_RATE, AudioPart, Embedder, ImagePart, Part, TextPart
from .store import Store

log = logging.getLogger(__name__)

# モデルの入力上限(トークン)。画像1枚は 280 トークン、音声は 1秒あたり 25 トークンとして見積もります。
# 実測(transformers の EmbeddingGemma2Processor): 音声は 40ms ごとに 1 トークン(=25/秒)、
# 448px の画像は 256 トークン。画像は少し多め(安全側)に見ています。
MAX_INPUT_TOKENS = 8192
IMAGE_TOKENS = 280
AUDIO_TOKENS_PER_SEC = 25
LABEL_TOKENS = 8


INGEST_KINDS = ("video", "audio", "image", "document")  # キューで処理できる取り込み元の種類


@dataclass(frozen=True)
class IngestParams:
    """取り込みの設定。取り込み元ごとに DB へ記録します(あとで再取り込みできるように)。"""

    window_sec: int
    frames_per_window: int
    overlap_sec: int
    include_audio: bool
    chunk_sec: int = 10  # 音声だけを取り込むときの区切り(秒)
    # 偽なら、元のファイル(アップロードしたもの)とサムネイルを残さない。検索結果は時刻などの情報だけになる
    store_media: bool = True
    # 真なら、前に推論した窓と映像が変わらない窓を推論せず、前の窓の終わりを延ばす(映像のみの窓だけ。is_static_window)。
    # 以前の版の取り込み元には無い項目のため、既定は偽(取り込み直しても、以前と同じ窓になる)
    skip_static: bool = False
    static_threshold: float = 4.0
    # 文書の塊の大きさ(文字数)と重なり。文書だけで使う(documents.chunk_pages)
    doc_chunk_chars: int = 800
    doc_overlap_chars: int = 100

    @classmethod
    def from_settings(cls, s: Settings, preset: str | None = None, **overrides) -> IngestParams:
        base = {
            "window_sec": s.window_sec,
            "frames_per_window": s.frames_per_window,
            "overlap_sec": s.overlap_sec,
            "include_audio": s.include_audio,
            "chunk_sec": s.audio_chunk_sec,
            "store_media": s.store_media,
            "skip_static": s.skip_static,
            "static_threshold": s.static_threshold,
            "doc_chunk_chars": s.doc_chunk_chars,
            "doc_overlap_chars": s.doc_overlap_chars,
        }
        if preset:
            if preset not in PRESETS:
                raise ValueError(f"preset は {list(PRESETS)} のいずれかにしてください: {preset!r}")
            base.update(PRESETS[preset])
        base.update({k: v for k, v in overrides.items() if v is not None})
        params = cls(**base)
        params.validate()
        return params

    def validate(self) -> None:
        if not 1 <= self.window_sec <= 60:
            raise ValueError("window_sec は 1〜60 にしてください")
        if not 1 <= self.frames_per_window <= self.window_sec:
            raise ValueError("frames_per_window は 1〜window_sec にしてください")
        if not 0 <= self.overlap_sec < self.window_sec:
            raise ValueError("overlap_sec は 0 以上 window_sec 未満にしてください")
        if not 1 <= self.chunk_sec <= 300:
            raise ValueError("chunk_sec は 1〜300 にしてください")
        if not 0.5 <= self.static_threshold <= 64:
            raise ValueError("static_threshold は 0.5〜64 にしてください")
        if not 100 <= self.doc_chunk_chars <= 4000:
            raise ValueError("doc_chunk_chars は 100〜4000 にしてください")
        if not 0 <= self.doc_overlap_chars <= self.doc_chunk_chars // 2:
            raise ValueError("doc_overlap_chars は 0〜doc_chunk_chars の半分にしてください")
        tokens = self.frames_per_window * (IMAGE_TOKENS + 2 * LABEL_TOKENS)
        if self.include_audio:
            tokens += self.window_sec * AUDIO_TOKENS_PER_SEC
        if tokens > MAX_INPUT_TOKENS:
            raise ValueError(
                f"1窓の見積もりが {tokens} トークンで、モデルの上限({MAX_INPUT_TOKENS})を超えます。"
                "フレーム数か窓の長さを減らしてください"
            )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class WindowPlan:
    start_ms: int
    end_ms: int
    sample_ms: list[int]  # フレームを取り出す時刻


def plan_windows(
    duration_ms: int, window_sec: int, frames_per_window: int, overlap_sec: int
) -> list[WindowPlan]:
    """動画を時間窓に区切り、各窓でフレームを取り出す時刻を決めます。

    窓をフレームの数で等分し、それぞれの区間の中央の時刻を使います(各フレームが区間を代表し、
    音声を区間ごとに対応させたときの位置とも揃う)。端の時刻を使う方式に比べ、隣の窓と同じフレームを
    重複して使うことがなく、動画の末尾ちょうど(フレームが無いことがある)も避けられます。
    """
    window_ms = window_sec * 1000
    step_ms = max(window_ms - overlap_sec * 1000, 1000)
    last_ok = max(duration_ms - 50, 0)
    plans: list[WindowPlan] = []
    start = 0
    while start < duration_ms:
        end = min(start + window_ms, duration_ms)
        length = end - start
        if plans and length < 500:  # 端数の短すぎる窓は作らない
            break
        times = [start + (2 * i + 1) * length // (2 * frames_per_window) for i in range(frames_per_window)]
        plans.append(WindowPlan(start, end, [min(max(t, 0), last_ok) for t in times]))
        if end >= duration_ms:
            break
        start += step_ms
    return plans


def decode_step_ms(plans: list[WindowPlan]) -> int:
    """連続復号でフレームを取り出す間隔(ms)。全ての取り出し時刻を割り切れる間隔にします。

    取り出し時刻の最大公約数を使い、100ms 未満になる場合は 100ms にします(最寄りのフレームとの差は
    最大 50ms で、検索の精度には影響しない)。間隔が広いほど、縮小と転送の量が減って速くなります。
    """
    step = 0
    for plan in plans:
        for t in plan.sample_ms:
            step = math.gcd(step, t)
    return max(step, 100) if step else 1000


def decode_select_grid(need: list[list[int]]) -> tuple[int, int]:
    """全窓で必要なフレーム番号(need)が等間隔の格子に乗る場合、(先頭オフセット, 間引き間隔)を返します。

    既定の 2 秒窓・2 フレームでは sample_ms = [500, 1500, 2500, ...] となり、
    step=500ms に対して必要な番号は 1, 3, 5, ... (offset=1, stride=2) だけです。
    ffmpeg の scale フィルタの前に select で偶数フレームを落とすことで、出力画素を完全一致させたまま
    縮小・色変換・パイプ転送の負荷を半減できます。
    """
    indices = [k for row in need for k in row]
    if not indices:
        return 0, 1
    offset = min(indices)
    stride = 0
    for k in indices:
        stride = math.gcd(stride, k - offset)
    if stride <= 1:
        return 0, 1
    return offset, stride


def audio_slice_bounds(start_ms: int, end_ms: int, count: int) -> list[tuple[int, int]]:
    """窓を、フレームの数だけ等分します(各フレームに、その前後の音声を対応させる)。"""
    if count <= 0 or end_ms <= start_ms:
        return []
    span = end_ms - start_ms
    return [
        (start_ms + i * span // count, start_ms + (i + 1) * span // count) for i in range(count)
    ]


def split_window_pcm(
    pcm: np.ndarray, start_ms: int, end_ms: int, count: int
) -> list[np.ndarray | None]:
    """窓の音声を、フレームの数に分けます。足りない末尾は None にします(引き伸ばさない)。"""
    if pcm.size == 0:
        return []
    out: list[np.ndarray | None] = []
    for s, e in audio_slice_bounds(start_ms, end_ms, count):
        lo = min(max((s - start_ms) * SAMPLE_RATE // 1000, 0), pcm.size)
        hi = min(max((e - start_ms) * SAMPLE_RATE // 1000, lo), pcm.size)
        out.append(pcm[lo:hi] if hi > lo else None)
    return out


def build_window_parts(
    frames: list[np.ndarray],
    frame_times_ms: list[int],
    audio_slices: list[np.ndarray | None] | None = None,
) -> list[Part]:
    """1窓ぶんの入力を、モデルに渡す順序で並べます。

    音声あり:  "00:00" 音声1  "00:00" 画像1  "00:02" 音声2  "00:02" 画像2 …
    音声なし:  画像1  画像2 …(時刻ラベルは付けない。Gallery と同じ)
    """
    if len(frames) != len(frame_times_ms):
        raise ValueError("frames と frame_times_ms の数が合いません")
    has_audio = bool(audio_slices) and any(a is not None and a.size > 0 for a in audio_slices)
    if not has_audio:
        return [ImagePart(f) for f in frames]
    assert audio_slices is not None
    parts: list[Part] = []
    for i, frame in enumerate(frames):
        label = media.format_time(frame_times_ms[i])
        audio = audio_slices[i] if i < len(audio_slices) else None
        if audio is not None and audio.size > 0:
            parts += [TextPart(label), AudioPart(audio)]
        parts += [TextPart(label), ImagePart(frame)]
    return parts


# 変化の無い区間の省略で使う、縮小した画像の大きさ(横 64 × 縦 36)と、判定の区画(横 16 × 縦 9。1 区画 4×4)
SIGNATURE_W, SIGNATURE_H = 64, 36
BLOCKS_W, BLOCKS_H = 16, 9
# 1 つの窓を延ばせる長さの上限(ms)。検索結果の区間が長くなりすぎないようにし、ゆっくりした変化(日の傾きなど)も
# この間隔で推論し直す
MAX_STATIC_SPAN_MS = 300_000


def frame_signature(image: np.ndarray) -> np.ndarray:
    """フレームを、変化の判定に使う 36×64 の輝度(float32)に縮めます(面積の平均。追加の依存は使わない)。"""
    gray = image.astype(np.float32).mean(axis=2) if image.ndim == 3 else image.astype(np.float32)
    h, w = gray.shape
    bh, bw = max(h // SIGNATURE_H, 1), max(w // SIGNATURE_W, 1)
    gray = gray[: bh * SIGNATURE_H, : bw * SIGNATURE_W]
    if gray.shape != (bh * SIGNATURE_H, bw * SIGNATURE_W):  # 36×64 より小さい画像(テスト用など)
        from PIL import Image

        return np.asarray(
            Image.fromarray(image).convert("L").resize((SIGNATURE_W, SIGNATURE_H), Image.BILINEAR), dtype=np.float32
        )
    return gray.reshape(SIGNATURE_H, bh, SIGNATURE_W, bw).mean(axis=(1, 3))


def block_change(a: np.ndarray, b: np.ndarray) -> float:
    """2 つの縮小画像の差を、16×9 の区画ごとの平均の差の最大値で返します(0〜255)。

    画面全体の平均ではなく区画ごとの最大を使うのは、遠くを歩く人のような小さな動きを、全体の平均で
    薄めずに捉えるためです(1 区画は元の画面の 1/144。448×252 の画像で約 28×28 ピクセル)。
    """
    diff = np.abs(a - b).reshape(BLOCKS_H, SIGNATURE_H // BLOCKS_H, BLOCKS_W, SIGNATURE_W // BLOCKS_W)
    return float(diff.mean(axis=(1, 3)).max())


def is_static_window(signatures: list[np.ndarray], reference: list[np.ndarray] | None, threshold: float) -> bool:
    """窓の全てのフレームが、前に推論した窓の対応するフレームと、どの区画でも threshold 未満の差しかないか。

    前の窓(reference)とだけ比べ、省略した窓どうしでは比べません。少しずつ変わる場面(日の傾き・ゆっくり
    動く物)で、差が積み重なっても気づけるようにするためです。フレームの数が違う窓(動画の末尾)は省略しません。
    """
    if not reference or len(signatures) != len(reference):
        return False
    return all(block_change(a, b) < threshold for a, b in zip(signatures, reference, strict=True))


def save_thumbnail(image: np.ndarray, dest: Path) -> None:
    """検索結果に表示するサムネイル(長辺 320px の JPEG)を保存します。"""
    from PIL import Image

    dest.parent.mkdir(parents=True, exist_ok=True)
    thumb = Image.fromarray(image)
    thumb.thumbnail((320, 320))
    thumb.save(dest, format="JPEG", quality=80)


_TIMER_LOCK = threading.Lock()


class StageTimer:
    """取り込みの処理時間を、段階ごとに積算します(ベンチマーク用。ジョブの timings として保存)。

    段階: probe(ffprobe)/ decode(映像の復号の待ち)/ audio(音声の変換の待ち)/ image(画像の読み込み)/
    embed(推論)/ store(DB への保存)/ thumb(サムネイルの保存)。
    復号は推論と並行して進むため、decode は「推論側が次のフレームを待った時間」です。0 に近いほど、
    推論が律速になっています。
    動画の取り込みでは、推論と保存も並行して進むため、embed は推論が動いていた時間(別のスレッドで計測し、
    並行した分は重ねて数えない)です。段階の合計は total_ms より大きくなることがあります。
    """

    def __init__(self) -> None:
        self._t0 = time.perf_counter()
        self.stages: dict[str, float] = {}
        self.extra: dict[str, object] = {}
        self._active: dict[str, tuple[int, float]] = {}  # 並行して動いている段階の数と、動き始めた時刻

    @contextmanager
    def stage(self, name: str):
        t = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, time.perf_counter() - t)

    @contextmanager
    def overlapping(self, name: str):
        """複数のスレッドで同時に動く段階(推論)の時間を、重ねて数えずに積算します(どれかが動いていた時間)。"""
        with _TIMER_LOCK:
            count, since = self._active.get(name, (0, 0.0))
            self._active[name] = (count + 1, time.perf_counter() if count == 0 else since)
        try:
            yield
        finally:
            with _TIMER_LOCK:
                count, since = self._active[name]
                self._active[name] = (count - 1, since)
                if count == 1:
                    self.stages[name] = self.stages.get(name, 0.0) + time.perf_counter() - since

    def add(self, name: str, seconds: float) -> None:
        with _TIMER_LOCK:  # 推論のスレッドからも加算するため
            self.stages[name] = self.stages.get(name, 0.0) + seconds

    def result(self, windows: int, media_ms: int) -> dict:
        total = time.perf_counter() - self._t0
        out: dict[str, object] = {
            "total_ms": round(total * 1000, 1),
            "stages_ms": {k: round(v * 1000, 1) for k, v in self.stages.items()},
            "windows": windows,
            "media_ms": media_ms,
            "per_window_ms": round(total * 1000 / windows, 1) if windows else None,
            # 実時間比: 1 秒の処理で何秒ぶんの映像・音声を取り込めたか(1 より大きければ実時間より速い)
            "realtime_factor": round(media_ms / 1000 / total, 2) if media_ms and total > 0 else None,
        }
        out.update(self.extra)
        return out


def _put_bounded(q: queue.Queue, item: object, stop: threading.Event) -> bool:
    """有界キューへ要素を投入します。バックプレッシャーで待機中でも停止要求があれば抜けます。"""
    while not stop.is_set():
        try:
            q.put(item, timeout=0.1)
            return True
        except queue.Full:
            continue
    return False


class _Lease:
    """前処理済みの入力の枠(サーバー全体で数を制限する)を 1 つ持ちます。release() は何度呼んでも 1 回だけ返します。"""

    def __init__(self, sem: threading.Semaphore) -> None:
        self._sem = sem
        self._lock = threading.Lock()
        self._held = True

    def release(self) -> None:
        with self._lock:
            if self._held:
                self._held = False
                self._sem.release()


def _acquire_lease(sem: threading.Semaphore, stop: threading.Event) -> _Lease | None:
    """枠が空くまで待ちます。停止を求められたら None を返します。"""
    while not stop.is_set():
        if sem.acquire(timeout=0.1):
            return _Lease(sem)
    return None


class _Interrupted(Exception):
    """停止の要求で、取り込みを途中で止めた。"""


class _Cancelled(Exception):
    """取り消しの要求(POST /api/jobs/{id}/cancel)で、取り込みを途中で止めた。"""


# 時間をおけば成功する可能性がある失敗(推論側に接続できない、ディスクの空きが足りない)。再試行する
_TRANSIENT = (ConnectionError, InsufficientStorage)
RETRY_BASE_SEC = 30  # 再試行までの間隔(1 回目 30 秒、以降 2 倍。最大 10 分)


class Ingestor:
    """取り込みジョブを、バックグラウンドのスレッドで処理します。

    スレッドの数は INGEST_WORKERS(既定 1)です。2 以上にすると、複数のファイルの復号・前処理・保存を同時に進めます。
    推論は Embedder の PriorityGate を通るため、GPU には今までどおり 1 回ずつ流れ、検索の優先も変わりません。
    速い GPU では、1 本の ffmpeg の復号と Python の前処理が追いつかず GPU が待つため、ファイルを並べて埋めます。
    同じ取り込み元のジョブ(取り込み直しなど)は、窓の削除と追加が混ざらないよう、同時には動かしません。
    """

    def __init__(self, settings: Settings, store: Store, embedder: Embedder) -> None:
        self._settings = settings
        self._store = store
        self._embedder = embedder
        self._queue: queue.Queue[int | None] = queue.Queue()
        self._workers = max(int(getattr(settings, "ingest_workers", 1) or 1), 1)
        self._threads: list[threading.Thread] = []
        self._stopping = threading.Event()
        self._local = threading.local()  # スレッドごとの実行中のジョブ(取り消しの判定に使う)
        self._running: set[int] = set()  # 実行中のジョブ
        self._cancel: set[int] = set()  # 取り消しを求められた実行中のジョブ
        self._busy_sources: set[int] = set()  # 実行中のジョブの取り込み元
        self._busy_cond = threading.Condition()
        # 前処理済みの入力(GPU へ渡す固定のメモリに置いたもの)の数の上限を、サーバー全体で「先読み + 2」にする。
        # 1 回分は EMBED_BATCH 件の窓(GPU の既定は 4 窓。6 秒窓なら 24 枚で約 177MB、固定のメモリでは 256MB)。
        # 推論は 1 回ずつなので、本数ぶん先読みを持っても速くならず、メモリだけが本数に比例して増えていた(#26)。
        # 1 本のときの数(先読み + 前処理中 1 + 推論中 1)と同じにし、同時数を増やしてもこれ以上は持たない
        self._prepared_limit = max(int(getattr(settings, "ingest_prefetch", 2) or 2), 1) + 2
        self._prepared_slots = threading.BoundedSemaphore(self._prepared_limit)

    # ------------------------------------------------------------------ 起動・停止
    def start(self) -> None:
        # 前回、途中で止まったジョブを、最初からやり直します(窓は削除してから作り直すので安全)。
        for job in self._store.unfinished_jobs():
            log.info("途中で止まったジョブを再開します: job=%s", job["id"])
            self._store.update_job(job["id"], status="queued", progress=0)
            self._queue.put(job["id"])
        # ベクトルを書き出す前に停止して、窓の一部が失われた取り込み元を、取り込み直す
        running = {j["source_id"] for j in self._store.unfinished_jobs()}
        for source_id in self._store.lost_sources:
            source = self._store.get_source(source_id)
            if source is None or source_id in running:
                continue
            if source["path"] and source["kind"] in INGEST_KINDS and Path(source["path"]).is_file():
                log.warning("ベクトルが失われた取り込み元を取り込み直します: %s", source["name"])
                self._store.update_source(source_id, status="queued", error=None)
                self.submit(source_id)
        for i in range(self._workers):
            name = "ingestor" if self._workers == 1 else f"ingestor-{i + 1}"
            th = threading.Thread(target=self._loop, name=name, daemon=True)
            th.start()
            self._threads.append(th)

    def stop(self) -> None:
        # 取り込み中のジョブは、窓の区切りで止めて queued に戻す(次の起動で最初からやり直す)。
        # 推論の途中の窓は、終わるまで待つ(待たずに終了すると、推論中のスレッドが残ったまま Python が終了する)
        self._stopping.set()
        for _ in self._threads:  # スレッドごとに 1 つずつ、終わりの印を受け取る
            self._queue.put(None)
        with self._busy_cond:  # 同じ取り込み元の空きを待っているスレッドを起こす
            self._busy_cond.notify_all()
        deadline = time.monotonic() + 45
        for th in self._threads:
            th.join(timeout=max(deadline - time.monotonic(), 0.1))
        if any(th.is_alive() for th in self._threads):
            log.warning("取り込みの推論が 45 秒以内に終わりませんでした")

    def _check_stop(self) -> None:
        if self._stopping.is_set():
            raise _Interrupted()
        current = getattr(self._local, "job", None)
        if current is not None and current in self._cancel:
            raise _Cancelled()

    def cancel(self, job_id: int) -> dict | None:
        """ジョブを取り消します。待っているジョブはすぐに、実行中のジョブは窓の区切りで止めます。

        実行中に取り消した場合、それまでに作った窓は残します(検索できる)。取り込み直しは reindex で行えます。
        """
        job = self._store.get_job(job_id)
        if job is None:
            return None
        if job["status"] == "queued":
            self._store.update_job(job_id, status="cancelled", finished_at=time.time(), error="取り消しました")
            self._store.update_source(job["source_id"], status="cancelled", error="取り消しました")
        elif job["status"] == "running":
            self._cancel.add(job_id)
        return self._store.get_job(job_id)

    def _handle_error(self, job_id: int, source: dict, exc: Exception) -> None:
        """失敗を記録します。一時的な失敗(推論側に接続できない・空き容量の不足)は、間隔をあけて再試行します。"""
        if isinstance(exc, _TRANSIENT):
            job = self._store.get_job(job_id) or {}
            attempts = int(job.get("attempts") or 0) + 1
            if attempts <= self._settings.job_retries:
                delay = min(RETRY_BASE_SEC * 2 ** (attempts - 1), 600)
                retries = self._settings.job_retries
                message = f"{type(exc).__name__}: {exc}({delay} 秒後に再試行します。{attempts}/{retries} 回目)"
                log.warning("取り込みを再試行します: job=%s %s", job_id, message)
                self._store.update_job(job_id, status="queued", progress=0, error=message, attempts=attempts)
                self._store.update_source(source["id"], status="queued", error=message)
                timer = threading.Timer(delay, self._queue.put, args=(job_id,))
                timer.daemon = True
                timer.start()
                return
        self._fail(job_id, source, exc)

    def submit(self, source_id: int) -> int:
        job_id = self._store.create_job(source_id)
        self._queue.put(job_id)
        return job_id

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    @property
    def workers(self) -> int:
        return self._workers

    @property
    def running(self) -> int:
        """実行中のジョブの数。"""
        return len(self._running)

    @contextmanager
    def _claim_sources(self, source_ids: list[int]):
        """取り込み元を、このスレッドだけが処理している状態にします(同じ取り込み元のジョブを同時に動かさない)。

        ほかのスレッドが同じ取り込み元を処理していれば、終わるまで待ちます。
        停止を求められたら _Interrupted を送出します。
        """
        ids = set(source_ids)
        with self._busy_cond:
            while ids & self._busy_sources:
                if self._stopping.is_set():
                    raise _Interrupted()
                self._busy_cond.wait(timeout=1.0)
            self._busy_sources |= ids
        try:
            yield
        finally:
            with self._busy_cond:
                self._busy_sources -= ids
                self._busy_cond.notify_all()

    def _source_ids(self, job_ids: list[int]) -> list[int]:
        ids = []
        for job_id in job_ids:
            job = self._store.get_job(job_id)
            if job is not None:
                ids.append(job["source_id"])
        return ids

    # ------------------------------------------------------------------ ジョブの実行
    IMAGE_BATCH = 16  # 画像はまとめて推論する(1 枚ずつより速い)

    def _loop(self) -> None:
        while True:
            job_id = self._queue.get()
            if job_id is None:
                return
            # 続けて登録された画像は、まとめて処理する(フォルダの一括取り込みなど)
            batch, rest = [job_id], []
            if self._is_image_job(job_id):
                while len(batch) < self.IMAGE_BATCH:
                    try:
                        nxt = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    if nxt is None:
                        rest.append(None)
                        break
                    (batch if self._is_image_job(nxt) else rest).append(nxt)
            try:
                if len(batch) > 1:
                    with self._claim_sources(self._source_ids(batch)):
                        self._run_image_batch(batch)
                else:
                    with self._claim_sources(self._source_ids([job_id])):
                        self._run_job(job_id)
                for other in rest:
                    if other is None:
                        return
                    with self._claim_sources(self._source_ids([other])):
                        self._run_job(other)
            except _Interrupted:  # 停止中に、同じ取り込み元の空きを待っていた(ジョブは queued のまま残る)
                return
            self._release_if_idle()

    def _release_if_idle(self) -> None:
        """取り込みが空になったら、推論で使った一時的なメモリ(固定のメモリなど)を OS へ返します。"""
        if self._queue.qsize() == 0 and not self._running and not self._stopping.is_set():
            try:
                self._embedder.release_cache()
            except Exception:  # 解放の失敗で取り込みを止めない
                log.debug("一時的なメモリを返せませんでした", exc_info=True)

    def _is_image_job(self, job_id: int) -> bool:
        job = self._store.get_job(job_id)
        source = self._store.get_source(job["source_id"]) if job else None
        return bool(source) and source["kind"] == "image"

    def _begin(self, job_id: int) -> dict | None:
        job = self._store.get_job(job_id)
        if job is None or job["status"] in ("cancelled", "done", "failed"):  # 削除・取り消し済み
            return None
        source = self._store.get_source(job["source_id"])
        if source is None:
            return None
        self._store.update_job(job_id, status="running", started_at=time.time(), error=None, timings=None)
        self._store.update_source(source["id"], status="running", error=None)
        return source

    def _discard_media(self, source: dict) -> None:
        """保存しない設定(store_media が偽)の取り込み元で、アップロードされたファイルを削除します。

        削除するのは DATA_DIR/media の下(アップロードで受け取ったもの)だけです。
        パス指定で取り込んだ利用者のファイルは消しません(配信と再取り込みの対象から外すだけ)。
        """
        if source["params"].get("store_media", True) or not source["path"]:
            return
        path = Path(source["path"])
        try:
            uploaded = path.resolve().is_relative_to((self._settings.data_dir / "media").resolve())
        except OSError:
            return
        if uploaded:
            path.unlink(missing_ok=True)
            self._store.update_source(source["id"], path=None)

    def _fail(self, job_id: int, source: dict, exc: Exception) -> None:
        log.exception("取り込みに失敗しました: job=%s source=%s", job_id, source["id"], exc_info=exc)
        message = f"{type(exc).__name__}: {exc}"
        self._store.update_job(job_id, status="failed", error=message, finished_at=time.time())
        self._store.update_source(source["id"], status="failed", error=message)
        self._discard_media(source)

    def _finish(self, job_id: int, source: dict, timings: dict) -> None:
        info = self._embedder.info()
        timings.update(accelerator=info.accelerator, backend=info.backend)
        self._store.update_job(job_id, status="done", finished_at=time.time(), timings=timings)
        self._store.update_source(source["id"], status="done")
        self._discard_media(source)
        log.info(
            "取り込みが完了しました: %s(%s 窓、%.1f 秒、実時間比 %s)",
            source["name"], timings.get("windows"), (timings.get("total_ms") or 0) / 1000,
            timings.get("realtime_factor"),
        )  # fmt: skip

    def _run_job(self, job_id: int) -> None:
        source = self._begin(job_id)
        if source is None:
            return
        self._local.job = job_id
        self._running.add(job_id)
        try:
            check_disk(self._settings.data_dir, self._settings.min_free_disk_mb)
            info = self._embedder.info()
            self._store.ensure_compat(info.model_id, info.dims)
            if source["kind"] == "video":
                timings = self._index_video(job_id, source)
            elif source["kind"] == "audio":
                timings = self._index_audio(job_id, source)
            elif source["kind"] == "image":
                timings = self._index_images([(job_id, source)])
            elif source["kind"] == "document":
                timings = self._index_document(job_id, source)
            else:
                raise ValueError(f"キューで処理できない種類です: {source['kind']}")
        except _Interrupted:
            log.info("停止のため、取り込みを中断しました(次の起動でやり直します): %s", source["name"])
            self._store.update_job(job_id, status="queued", progress=0)
            self._store.update_source(source["id"], status="queued")
            return
        except _Cancelled:
            log.info("取り込みを取り消しました: %s", source["name"])
            self._store.update_job(job_id, status="cancelled", finished_at=time.time(), error="取り消しました")
            self._store.update_source(source["id"], status="cancelled", error="取り消しました")
            self._discard_media(source)
            return
        except Exception as exc:
            self._handle_error(job_id, source, exc)
            return
        finally:
            self._local.job = None
            self._running.discard(job_id)
            self._cancel.discard(job_id)
        self._finish(job_id, source, timings)

    def _thumb_path(self, window_id: int) -> Path:
        return thumbs.path_for(self._settings.data_dir / "thumbs", window_id)

    def _index_video(self, job_id: int, source: dict) -> dict:
        """動画を窓ごとにベクトルにします(Producer-Consumer パイプライン)。

        Producer スレッド(name="decode")が ffmpeg からの連続復号・窓の組み立て・音声切り出し・
        CPU 前処理(テンソル化・GPU 向け pin_memory)までを先行実行し、有界キュー(maxsize=INGEST_PREFETCH、
        既定 2)越しに Consumer(ingestor スレッド)へ渡します。これにより窓 N の推論中に窓 N+1 の
        復号と前処理をオーバーラップさせつつ、バックプレッシャーでメモリ肥大化を防ぎます。
        """
        timer = StageTimer()
        params = IngestParams(**source["params"])
        path = source["path"]
        with timer.stage("probe"):
            info = media.probe(path)
        if not info.has_video or info.duration_ms <= 0:
            raise media.MediaError("映像トラックが見つからないか、長さが 0 です")
        self._store.update_source(source["id"], duration_ms=info.duration_ms)
        self._store.delete_windows(source["id"])  # 再取り込みのとき、古い窓を残さない

        plans = plan_windows(
            info.duration_ms, params.window_sec, params.frames_per_window, params.overlap_sec
        )
        use_audio = params.include_audio and info.has_audio
        if params.include_audio and not info.has_audio:
            log.info("音声トラックがないので、映像だけで取り込みます: %s", source["name"])
        self._store.update_job(job_id, total=len(plans), progress=0)

        step = decode_step_ms(plans)
        need = [[round(t / step) for t in plan.sample_ms] for plan in plans]
        select_offset, select_stride = decode_select_grid(need)
        hwaccel, device = media.select_hwaccel(
            self._settings.ffmpeg_hwaccel, self._settings.ffmpeg_hwaccel_device
        )
        source_size = (info.width, info.height) if info.width > 0 and info.height > 0 else None
        with timer.stage("probe"):
            reader = media.VideoFrameReader(
                path,
                step,
                self._settings.image_max_side,
                hwaccel,
                device,
                self._settings.ffmpeg_skip_frames,
                source_size=source_size,
                select_offset=select_offset,
                select_stride=select_stride,
            )
        batch = max(self._embedder.info().batch_size, 1)
        prefetch = max(self._settings.ingest_prefetch, 1)
        # 推論は別のスレッドで行い、その間に前の回の保存(DB・サムネイル)と、Producer 側の復号・前処理を進める。
        # 先に次の回を推論に出してから前の回を保存するため、推論中の回は最大 2 つ
        embed_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ingest-embed")
        inflight: tuple[list[dict], Future] | None = None
        # 変化の無い区間の省略は、映像のみの窓だけで行う(音声つきの窓は、映像が止まっていても音が変わり得るため)
        skip_static = params.skip_static and not use_audio
        with tempfile.TemporaryDirectory(prefix="mediasearch-") as tmp:
            audio = media.AudioTrack(path, tmp) if use_audio else None
            work_q: queue.Queue = queue.Queue(maxsize=prefetch)
            done = object()
            stop = threading.Event()

            def emit(ready: list[dict], audio_sec: float) -> bool:
                """窓をまとめて前処理し、有界キューへ積みます。前処理の前に、サーバー全体の枠を 1 つ取ります。"""
                lease = _acquire_lease(self._prepared_slots, stop)
                if lease is None:
                    return False
                try:
                    t_prep = time.perf_counter()
                    prepared = self._embedder.prepare_parts_batch([w["parts"] for w in ready])
                    prep_sec = time.perf_counter() - t_prep
                except BaseException:
                    lease.release()
                    raise
                if not _put_bounded(work_q, (ready, prepared, audio_sec, prep_sec, lease), stop):
                    lease.release()
                    return False
                return True

            def produce() -> None:
                it = iter(reader)
                frames: dict[int, np.ndarray] = {}
                pending = 0  # 次に組み立てる窓
                last_index = -1
                ready: list[dict] = []  # フレームが揃い、推論を待っている窓(batch 件たまったらまとめて推論する)
                batch_audio_sec = 0.0
                ref_win: dict | None = None  # 変化の判定の基準(最後に推論に回した窓)と、その縮小画像
                ref_sigs: list[np.ndarray] | None = None
                try:
                    while pending < len(plans) and not stop.is_set():
                        item = next(it, None)
                        if item is not None:
                            last_index = item[0] // step
                            frames[last_index] = item[1]
                        # 必要なフレームが揃った窓から順に前処理し、batch 件ごとに有界キューへ積む
                        while pending < len(plans) and (item is None or max(need[pending]) <= last_index):
                            if stop.is_set():
                                return
                            plan = plans[pending]
                            win, audio_sec = self._prepare_window(plan, need[pending], frames, audio)
                            batch_audio_sec += audio_sec
                            skipped = False
                            if skip_static:
                                sigs = [frame_signature(p.image) for p in win["parts"] if isinstance(p, ImagePart)]
                                if (
                                    ref_win is not None
                                    and plan.end_ms - ref_win["plan"].start_ms <= MAX_STATIC_SPAN_MS
                                    and is_static_window(sigs, ref_sigs, params.static_threshold)
                                ):
                                    # 推論せず、基準の窓の終わりを延ばす。基準の窓がまだ手元にあれば直接、
                                    # 既に推論に回していれば、取り込みのスレッドに延長を頼む(保存の前後どちらでも反映)
                                    skipped = True
                                    if any(w is ref_win for w in ready):
                                        ref_win["end_ms"] = plan.end_ms
                                        ref_win["skipped"] = ref_win.get("skipped", 0) + 1
                                    elif not _put_bounded(work_q, ("extend", ref_win, plan.end_ms), stop):
                                        return
                                else:
                                    ref_win, ref_sigs = win, sigs
                            if not skipped:
                                ready.append(win)
                            pending += 1
                            keep = min(need[pending]) if pending < len(plans) else last_index + 1
                            for k in [k for k in frames if k < keep]:
                                del frames[k]
                            if ready and (len(ready) >= batch or pending == len(plans)):
                                if not emit(ready, batch_audio_sec):
                                    return
                                ready = []
                                batch_audio_sec = 0.0
                        if item is None:
                            break
                    if ready and not stop.is_set():
                        if not emit(ready, batch_audio_sec):
                            return
                    _put_bounded(work_q, done, stop)
                except BaseException as exc:  # Consumer 側で同じ例外を送出する
                    _put_bounded(work_q, exc, stop)
                finally:
                    close = getattr(it, "close", None)
                    if close:  # 途中で止めた場合も、ffmpeg のプロセスを確実に終わらせる
                        close()

            producer = threading.Thread(target=produce, name="decode", daemon=True)
            producer.start()
            stored = 0
            skipped_static = 0
            try:
                while True:
                    self._check_stop()
                    t = time.perf_counter()
                    packet = work_q.get()
                    timer.add("decode", time.perf_counter() - t)
                    if packet is done:
                        break
                    if isinstance(packet, BaseException):
                        raise packet
                    if packet[0] == "extend":  # 変化の無い窓を省略し、基準の窓を延ばす
                        _, ref, end_ms = packet
                        if "window_id" in ref:  # 保存済み
                            self._store.extend_window(ref["window_id"], end_ms)
                        else:  # 推論中・保存前(保存のときに反映する)
                            ref["end_ms"] = end_ms
                        skipped_static += 1
                        stored += 1
                        self._store.update_job(job_id, progress=stored)
                        continue
                    ready, prepared, audio_sec, prep_sec, lease = packet
                    if audio_sec > 0:
                        timer.add("audio", audio_sec)
                    if prep_sec > 0:
                        timer.add("preprocess", prep_sec)
                    try:
                        self._check_stop()
                        check_disk(self._settings.data_dir, self._settings.min_free_disk_mb)
                        future = embed_pool.submit(self._embed_windows, ready, prepared, timer, lease)
                    except BaseException:
                        lease.release()
                        raise
                    submitted = (ready, future, lease)
                    # 推論が終わった入力(固定のメモリ)を、次の回まで持ち続けない
                    packet = prepared = None
                    if inflight is not None:
                        prev_ready, prev_fut, _ = inflight
                        inflight = submitted
                        stored += self._save_windows(source, prev_ready, prev_fut.result(), timer)
                        skipped_static += sum(w.get("skipped", 0) for w in prev_ready)
                        self._store.update_job(job_id, progress=stored)
                    else:
                        inflight = submitted
                if inflight is not None:
                    prev_ready, prev_fut, _ = inflight
                    inflight = None
                    stored += self._save_windows(source, prev_ready, prev_fut.result(), timer)
                    skipped_static += sum(w.get("skipped", 0) for w in prev_ready)
                    self._store.update_job(job_id, progress=stored)
            finally:
                stop.set()
                if inflight is not None:  # 途中で止めた場合も、推論中の回が終わるのを待ってから片付ける
                    inflight[1].cancel()
                embed_pool.shutdown(wait=True)
                if inflight is not None:  # 取り消して推論しなかった回の枠を返す(推論した回は返済済み。二重には返さない)
                    inflight[2].release()

                def discard(item: object) -> None:
                    if isinstance(item, tuple) and len(item) == 5:
                        item[4].release()

                while producer.is_alive():  # 生成側が put で止まっていれば、取り出して終わらせる
                    try:
                        discard(work_q.get(timeout=0.1))
                    except queue.Empty:
                        pass
                producer.join()
                while True:  # キューに残った回の枠を返す
                    try:
                        discard(work_q.get_nowait())
                    except queue.Empty:
                        break
                if audio is not None:
                    audio.close()
        timer.extra.update(
            decoder=reader.decoder, skip_frames=reader.skip_used, decode_step_ms=step, decoded_frames=reader.frames,
            embed_batch=batch, prefetch=prefetch, select_stride=select_stride,
        )  # fmt: skip
        if skip_static:
            timer.extra["skipped_static"] = skipped_static
        return timer.result(len(plans), info.duration_ms)

    def _prepare_window(
        self,
        plan: WindowPlan,
        need: list[int],
        frames: dict[int, np.ndarray],
        audio: media.AudioTrack | None,
    ) -> tuple[dict, float]:
        """1 窓ぶんの入力を組み立てます(推論はまだしない。Producer スレッドから呼ぶ)。"""
        picked, times = [], []
        for k, t in zip(need, plan.sample_ms, strict=True):
            if k in frames:
                picked.append(frames[k])
                times.append(t)
        if not picked and frames:  # 動画の末尾が probe の長さより短い場合は、最後のフレームで代用する
            last = max(frames)
            picked, times = [frames[last]], [plan.sample_ms[0]]
        if not picked:
            raise media.MediaError(f"{plan.start_ms}ms からの窓でフレームを1枚も取り出せません")
        slices = None
        audio_sec = 0.0
        if audio is not None:
            t0 = time.perf_counter()
            pcm = audio.slice(plan.start_ms, plan.end_ms)
            audio_sec = time.perf_counter() - t0
            slices = split_window_pcm(pcm, plan.start_ms, plan.end_ms, len(picked))
        win = {"plan": plan, "parts": build_window_parts(picked, times, slices), "thumb": picked[len(picked) // 2]}
        return win, audio_sec

    def _embed_windows(
        self, ready: list[dict], prepared, timer: StageTimer, lease: _Lease | None = None
    ) -> list[np.ndarray]:
        """前処理済みの窓をまとめて推論します(EMBED_BATCH が 1 なら 1 件ずつ推論する)。推論のスレッドで動く。

        推論が終わったら(失敗しても)、前処理済みの入力の枠を返します。
        """
        try:
            with timer.overlapping("embed"):
                if prepared is not None:
                    return self._embedder.embed_prepared_parts(prepared, high=False)
                return self._embedder.embed_parts_many([w["parts"] for w in ready], high=False)
        finally:
            if lease is not None:
                lease.release()

    def _save_windows(self, source: dict, ready: list[dict], vecs: list[np.ndarray], timer: StageTimer) -> int:
        """推論した窓を保存し、処理した窓の数を返します(DB への書き込みは、取り込みのスレッドだけで行う)。

        変化が無いため省略した窓(推論に回す前に、この窓へまとめたもの)も、処理した数に含めます。
        """
        skipped = 0
        for w, vec in zip(ready, vecs, strict=True):
            plan = w["plan"]
            kind = "tav" if any(isinstance(p, AudioPart) for p in w["parts"]) else "frames"
            skipped += w.get("skipped", 0)
            with timer.stage("store"):
                window_id = self._store.add_window(
                    source_id=source["id"],
                    start_ms=plan.start_ms,
                    end_ms=w.get("end_ms", plan.end_ms),
                    kind=kind,
                    group_id=source["group_id"],
                    abs_ts=source["start_ts"] + plan.start_ms / 1000,
                    vec=vec,
                )
            w["window_id"] = window_id  # 後から延長の依頼が来たときに使う
            if source["params"].get("store_media", True):
                with timer.stage("thumb"):
                    # 復号済みのフレームから作る(表示のたびに ffmpeg を起動しなくて済む)
                    save_thumbnail(w["thumb"], self._thumb_path(window_id))
        return len(ready) + skipped

    def _index_audio(self, job_id: int, source: dict) -> dict:
        timer = StageTimer()
        params = IngestParams(**source["params"])
        with timer.stage("probe"):
            info = media.probe(source["path"])
        if not info.has_audio or info.duration_ms <= 0:
            raise media.MediaError("音声トラックが見つからないか、長さが 0 です")
        self._store.update_source(source["id"], duration_ms=info.duration_ms)
        self._store.delete_windows(source["id"])
        chunk_ms = params.chunk_sec * 1000
        starts = list(range(0, info.duration_ms, chunk_ms))
        self._store.update_job(job_id, total=len(starts), progress=0)
        windows = 0
        with tempfile.TemporaryDirectory(prefix="mediasearch-") as tmp:
            audio = media.AudioTrack(source["path"], tmp)  # 全体を 1 回で変換し、区切りごとに切り出す
            try:
                for n, start in enumerate(starts, start=1):
                    self._check_stop()
                    end = min(start + chunk_ms, info.duration_ms)
                    if n > 1 and end - start < 500:
                        self._store.update_job(job_id, progress=n)
                        continue
                    with timer.stage("audio"):
                        pcm = audio.slice(start, end)
                    if pcm.size == 0:
                        continue
                    with timer.stage("embed"):
                        vec = self._embedder.embed_audio(pcm, high=False)
                    with timer.stage("store"):
                        self._store.add_window(
                            source_id=source["id"],
                            start_ms=start,
                            end_ms=end,
                            kind="audio",
                            group_id=source["group_id"],
                            abs_ts=source["start_ts"] + start / 1000,
                            vec=vec,
                        )
                    windows += 1
                    self._store.update_job(job_id, progress=n)
            finally:
                audio.close()
        return timer.result(windows, info.duration_ms)

    # 文書の塊を 1 回の推論にまとめる件数。推論の途中には検索が割り込めないため、まとめるほど取り込みは速いが、
    # 取り込み中の検索が待たされる。RTX 3060・800 字の塊で、16 件: 1 塊 15.6 ms・検索 p50 235 ms、
    # 4 件: 17.3 ms・67 ms、1 件: 29.5 ms・28 ms(2026-11-20 実測)。取り込みの遅れが 1 割で済む 4 にする
    DOC_EMBED_BATCH = 4

    def _index_document(self, job_id: int, source: dict) -> dict:
        """文書を塊ごとにベクトルにします(窓の種類は document。時刻の幅は 0 で、ページを window_texts に記録)。

        塊は document のプロンプトで埋め込みます(検索文は query のプロンプト)。推論は取り込みの優先度で、
        DOC_EMBED_BATCH 件ずつ行い、合間に検索が割り込めるようにします。
        """
        timer = StageTimer()
        params = IngestParams(**source["params"])
        path = Path(source["path"])
        with timer.stage("extract"):
            extracted = documents.extract(path)
        chunks = documents.chunk_pages(extracted.pages, params.doc_chunk_chars, params.doc_overlap_chars)
        if not chunks:
            raise documents.DocumentError(f"文書から文字を取り出せません: {source['name']}")
        self._store.update_source(source["id"], duration_ms=0)
        self._store.delete_windows(source["id"])
        self._store.update_job(job_id, total=len(chunks), progress=0)
        keep_text = params.store_media
        done = 0
        for i in range(0, len(chunks), self.DOC_EMBED_BATCH):
            self._check_stop()
            check_disk(self._settings.data_dir, self._settings.min_free_disk_mb)
            part = chunks[i : i + self.DOC_EMBED_BATCH]
            with timer.stage("embed"):
                vecs = self._embedder.embed_texts([c.text for c in part], kind="document")
            with timer.stage("store"):
                for c, vec in zip(part, vecs, strict=True):
                    self._store.add_window(
                        source_id=source["id"],
                        start_ms=0,
                        end_ms=0,
                        kind="document",
                        group_id=source["group_id"],
                        abs_ts=source["start_ts"],
                        vec=vec,
                        page=c.page,
                        text=c.text if keep_text else None,
                    )
            done += len(part)
            self._store.update_job(job_id, progress=done)
        timer.extra.update(
            pages=extracted.total_pages,
            chars=sum(len(p.text) for p in extracted.pages),
            truncated=extracted.truncated,
        )
        return timer.result(len(chunks), 0)

    def _run_image_batch(self, job_ids: list[int]) -> None:
        """続けて登録された複数の画像を、1 回の推論でまとめてベクトルにします。

        まとめて失敗した場合は、原因の画像を特定するため 1 枚ずつ処理し直します。
        """
        items = []
        for job_id in job_ids:
            source = self._begin(job_id)
            if source is not None:
                items.append((job_id, source))
        if not items:
            return
        ids = {job_id for job_id, _ in items}
        self._running |= ids
        try:
            self._run_image_items(items)
        finally:
            self._running -= ids

    def _run_image_items(self, items: list[tuple[int, dict]]) -> None:
        try:
            check_disk(self._settings.data_dir, self._settings.min_free_disk_mb)
            info = self._embedder.info()
            self._store.ensure_compat(info.model_id, info.dims)
            timings = self._index_images(items)
        except Exception:
            log.warning("画像のまとめての取り込みに失敗したため、1 枚ずつ処理し直します")
            for job_id, source in items:
                try:
                    timings = self._index_images([(job_id, source)])
                except Exception as exc:
                    self._handle_error(job_id, source, exc)
                else:
                    self._finish(job_id, source, timings)
            return
        for job_id, source in items:
            self._finish(job_id, source, dict(timings))

    def _index_images(self, items: list[tuple[int, dict]]) -> dict:
        """静止画を 1 枚 1 ベクトルにします(窓は 1 つ。時刻の幅は 0)。

        動画から切り出せない場合や、写真だけを検索したい場合のための取り込みです。ffmpeg は使わず、
        Pillow で読み込みます(JPEG は縮小しながら復号)。複数枚は 1 回の推論にまとめます。
        画像単体のベクトルは、映像の窓(frames)と同じく画像だけを入力にしたものなので、文字や画像のクエリで探せます。
        """
        timer = StageTimer()
        images = []
        with timer.stage("image"):
            for _, source in items:
                images.append(media.load_image(source["path"], self._settings.image_max_side))
        with timer.stage("embed"):
            vecs = self._embedder.embed_images(images, high=False)
        for (job_id, source), image, vec in zip(items, images, vecs, strict=True):
            self._store.update_source(source["id"], duration_ms=0)
            self._store.delete_windows(source["id"])
            with timer.stage("store"):
                window_id = self._store.add_window(
                    source_id=source["id"],
                    start_ms=0,
                    end_ms=0,
                    kind="image",
                    group_id=source["group_id"],
                    abs_ts=source["start_ts"],
                    vec=vec,
                )
            if source["params"].get("store_media", True):
                with timer.stage("thumb"):
                    save_thumbnail(image, self._thumb_path(window_id))
            self._store.update_job(job_id, total=1, progress=1)
        timer.extra["batch"] = len(items)
        return timer.result(len(items), 0)

    # ------------------------------------------------------------------ 加工済みフレームの取り込み
    def index_frames_now(
        self,
        *,
        frames: list[np.ndarray],
        times_ms: list[int],
        audio: np.ndarray | None,
        name: str,
        group_id: str | None,
        location: str | None,
        start_ts: float,
        thumb_dir: Path,
        store_media: bool | None = None,
    ) -> dict:
        """事前に加工済みのフレーム(と音声)から、1窓を作ります。キューを通さず、その場で処理します。"""
        info = self._embedder.info()
        self._store.ensure_compat(info.model_id, info.dims)
        end_ms = max(times_ms) + 1000
        params = IngestParams(
            window_sec=max(1, (end_ms - min(times_ms)) // 1000),
            frames_per_window=max(1, len(frames)),
            overlap_sec=0,
            include_audio=audio is not None and audio.size > 0,
            store_media=self._settings.store_media if store_media is None else store_media,
        )
        slices = None
        if audio is not None and audio.size > 0:
            slices = split_window_pcm(audio, min(times_ms), end_ms, len(frames))
        parts = build_window_parts(frames, times_ms, slices)
        vec = self._embedder.embed_parts(parts, high=False)
        source_id = self._store.add_source(
            kind="frames",
            path=None,
            name=name,
            group_id=group_id,
            location=location,
            start_ts=start_ts,
            params=params.to_dict(),
            duration_ms=end_ms,
            status="done",
        )
        kind = "tav" if any(isinstance(p, AudioPart) for p in parts) else "frames"
        window_id = self._store.add_window(
            source_id=source_id,
            start_ms=min(times_ms),
            end_ms=end_ms,
            kind=kind,
            group_id=group_id,
            abs_ts=start_ts + min(times_ms) / 1000,
            vec=vec,
        )
        if params.store_media:
            save_thumbnail(frames[0], thumbs.path_for(thumb_dir, window_id))
        return {"source_id": source_id, "window_id": window_id, "kind": kind}
