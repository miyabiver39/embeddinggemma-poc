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
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import media
from .config import PRESETS, Settings
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


@dataclass(frozen=True)
class IngestParams:
    """取り込みの設定。取り込み元ごとに DB へ記録します(あとで再取り込みできるように)。"""

    window_sec: int
    frames_per_window: int
    overlap_sec: int
    include_audio: bool
    chunk_sec: int = 10  # 音声だけを取り込むときの区切り(秒)

    @classmethod
    def from_settings(cls, s: Settings, preset: str | None = None, **overrides) -> IngestParams:
        base = {
            "window_sec": s.window_sec,
            "frames_per_window": s.frames_per_window,
            "overlap_sec": s.overlap_sec,
            "include_audio": s.include_audio,
            "chunk_sec": s.audio_chunk_sec,
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


def save_thumbnail(image: np.ndarray, dest: Path) -> None:
    """検索結果に表示するサムネイル(長辺 320px の JPEG)を保存します。"""
    from PIL import Image

    dest.parent.mkdir(parents=True, exist_ok=True)
    thumb = Image.fromarray(image)
    thumb.thumbnail((320, 320))
    thumb.save(dest, format="JPEG", quality=80)


class StageTimer:
    """取り込みの処理時間を、段階ごとに積算します(ベンチマーク用。ジョブの timings として保存)。

    段階: probe(ffprobe)/ decode(映像の復号の待ち)/ audio(音声の変換の待ち)/ image(画像の読み込み)/
    embed(推論)/ store(DB への保存)/ thumb(サムネイルの保存)。
    復号は推論と並行して進むため、decode は「推論側が次のフレームを待った時間」です。0 に近いほど、
    推論が律速になっています。
    """

    def __init__(self) -> None:
        self._t0 = time.perf_counter()
        self.stages: dict[str, float] = {}
        self.extra: dict[str, object] = {}

    @contextmanager
    def stage(self, name: str):
        t = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, time.perf_counter() - t)

    def add(self, name: str, seconds: float) -> None:
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


def _prefetch(iterable, size: int = 32):
    """別のスレッドで iterable を先読みします(ffmpeg の復号と、推論を並行させるため)。"""
    q: queue.Queue = queue.Queue(maxsize=size)
    done = object()
    stop = threading.Event()

    def run() -> None:
        it = iter(iterable)
        try:
            for item in it:
                if stop.is_set():
                    break
                q.put(item)
            q.put(done)
        except BaseException as exc:  # 呼び出し側で同じ例外を出す
            q.put(exc)
        finally:
            close = getattr(it, "close", None)
            if close:  # 途中で止めた場合も、ffmpeg のプロセスを確実に終わらせる
                close()

    th = threading.Thread(target=run, name="decode", daemon=True)
    th.start()
    try:
        while True:
            item = q.get()
            if item is done:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()
        while th.is_alive():  # 生成側が put で止まっていれば、取り出して終わらせる
            try:
                q.get(timeout=0.1)
            except queue.Empty:
                pass
        th.join()


class Ingestor:
    """取り込みジョブを、バックグラウンドのスレッドで1つずつ処理します。"""

    def __init__(self, settings: Settings, store: Store, embedder: Embedder) -> None:
        self._settings = settings
        self._store = store
        self._embedder = embedder
        self._queue: queue.Queue[int | None] = queue.Queue()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------ 起動・停止
    def start(self) -> None:
        # 前回、途中で止まったジョブを、最初からやり直します(窓は削除してから作り直すので安全)。
        for job in self._store.unfinished_jobs():
            log.info("途中で止まったジョブを再開します: job=%s", job["id"])
            self._store.update_job(job["id"], status="queued", progress=0)
            self._queue.put(job["id"])
        self._thread = threading.Thread(target=self._loop, name="ingestor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._queue.put(None)
        if self._thread:
            self._thread.join(timeout=5)

    def submit(self, source_id: int) -> int:
        job_id = self._store.create_job(source_id)
        self._queue.put(job_id)
        return job_id

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

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
            if len(batch) > 1:
                self._run_image_batch(batch)
            else:
                self._run_job(job_id)
            for other in rest:
                if other is None:
                    return
                self._run_job(other)

    def _is_image_job(self, job_id: int) -> bool:
        job = self._store.get_job(job_id)
        source = self._store.get_source(job["source_id"]) if job else None
        return bool(source) and source["kind"] == "image"

    def _begin(self, job_id: int) -> dict | None:
        job = self._store.get_job(job_id)
        if job is None:  # 取り込み元ごと削除された
            return None
        source = self._store.get_source(job["source_id"])
        if source is None:
            return None
        self._store.update_job(job_id, status="running", started_at=time.time(), error=None, timings=None)
        self._store.update_source(source["id"], status="running", error=None)
        return source

    def _fail(self, job_id: int, source: dict, exc: Exception) -> None:
        log.exception("取り込みに失敗しました: job=%s source=%s", job_id, source["id"], exc_info=exc)
        message = f"{type(exc).__name__}: {exc}"
        self._store.update_job(job_id, status="failed", error=message, finished_at=time.time())
        self._store.update_source(source["id"], status="failed", error=message)

    def _finish(self, job_id: int, source: dict, timings: dict) -> None:
        info = self._embedder.info()
        timings.update(accelerator=info.accelerator, backend=info.backend)
        self._store.update_job(job_id, status="done", finished_at=time.time(), timings=timings)
        self._store.update_source(source["id"], status="done")
        log.info(
            "取り込みが完了しました: %s(%s 窓、%.1f 秒、実時間比 %s)",
            source["name"], timings.get("windows"), (timings.get("total_ms") or 0) / 1000,
            timings.get("realtime_factor"),
        )  # fmt: skip

    def _run_job(self, job_id: int) -> None:
        source = self._begin(job_id)
        if source is None:
            return
        try:
            info = self._embedder.info()
            self._store.ensure_compat(info.model_id, info.dims)
            if source["kind"] == "video":
                timings = self._index_video(job_id, source)
            elif source["kind"] == "audio":
                timings = self._index_audio(job_id, source)
            elif source["kind"] == "image":
                timings = self._index_images([(job_id, source)])
            else:
                raise ValueError(f"キューで処理できない種類です: {source['kind']}")
        except Exception as exc:
            self._fail(job_id, source, exc)
            return
        self._finish(job_id, source, timings)

    def _thumb_path(self, window_id: int) -> Path:
        return self._settings.data_dir / "thumbs" / f"w{window_id}.jpg"

    def _index_video(self, job_id: int, source: dict) -> dict:
        """動画を窓ごとにベクトルにします。

        ffmpeg は 1 回だけ起動し、先頭から最後まで連続で復号します(VideoFrameReader)。復号は別スレッドで
        先読みし、推論と並行させます。音声も 1 回の ffmpeg でまとめて変換します(AudioTrack)。
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
        hwaccel, device = media.select_hwaccel(
            self._settings.ffmpeg_hwaccel, self._settings.ffmpeg_hwaccel_device
        )
        with timer.stage("probe"):
            reader = media.VideoFrameReader(
                path, step, self._settings.image_max_side, hwaccel, device, self._settings.ffmpeg_skip_frames
            )
        need = [[round(t / step) for t in plan.sample_ms] for plan in plans]
        with tempfile.TemporaryDirectory(prefix="mediasearch-") as tmp:
            audio = media.AudioTrack(path, tmp) if use_audio else None
            try:
                frames: dict[int, np.ndarray] = {}
                pending = 0  # 次に処理する窓
                last_index = -1
                stream = _prefetch(reader)
                while pending < len(plans):
                    t = time.perf_counter()
                    item = next(stream, None)
                    timer.add("decode", time.perf_counter() - t)
                    if item is not None:
                        last_index = item[0] // step
                        frames[last_index] = item[1]
                    # 必要なフレームが揃った窓から順に処理する(動画が probe の長さより短い場合は、手元の分で処理)
                    while pending < len(plans) and (item is None or max(need[pending]) <= last_index):
                        self._embed_window(source, plans[pending], need[pending], frames, audio, timer)
                        pending += 1
                        self._store.update_job(job_id, progress=pending)
                        keep = min(need[pending]) if pending < len(plans) else last_index + 1
                        for k in [k for k in frames if k < keep]:
                            del frames[k]
                    if item is None:
                        break
            finally:
                stream.close()
                if audio is not None:
                    audio.close()
        timer.extra.update(
            decoder=reader.decoder, skip_frames=reader.skip_used, decode_step_ms=step, decoded_frames=reader.frames
        )
        return timer.result(len(plans), info.duration_ms)

    def _embed_window(
        self,
        source: dict,
        plan: WindowPlan,
        need: list[int],
        frames: dict[int, np.ndarray],
        audio: media.AudioTrack | None,
        timer: StageTimer,
    ) -> None:
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
        if audio is not None:
            with timer.stage("audio"):
                pcm = audio.slice(plan.start_ms, plan.end_ms)
            slices = split_window_pcm(pcm, plan.start_ms, plan.end_ms, len(picked))
        parts = build_window_parts(picked, times, slices)
        with timer.stage("embed"):
            vec = self._embedder.embed_parts(parts, high=False)
        kind = "tav" if any(isinstance(p, AudioPart) for p in parts) else "frames"
        with timer.stage("store"):
            window_id = self._store.add_window(
                source_id=source["id"],
                start_ms=plan.start_ms,
                end_ms=plan.end_ms,
                kind=kind,
                group_id=source["group_id"],
                abs_ts=source["start_ts"] + plan.start_ms / 1000,
                vec=vec,
            )
        with timer.stage("thumb"):
            # 復号済みのフレームから作る(表示のたびに ffmpeg を起動しなくて済む)
            save_thumbnail(picked[len(picked) // 2], self._thumb_path(window_id))

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
        try:
            info = self._embedder.info()
            self._store.ensure_compat(info.model_id, info.dims)
            timings = self._index_images(items)
        except Exception:
            log.warning("画像のまとめての取り込みに失敗したため、1 枚ずつ処理し直します")
            for job_id, source in items:
                try:
                    timings = self._index_images([(job_id, source)])
                except Exception as exc:
                    self._fail(job_id, source, exc)
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
        save_thumbnail(frames[0], thumb_dir / f"w{window_id}.jpg")
        return {"source_id": source_id, "window_id": window_id, "kind": kind}
