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
import queue
import threading
import time
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
    """動画を時間窓に区切り、各窓でフレームを取り出す時刻を決めます。"""
    window_ms = window_sec * 1000
    step_ms = max(window_ms - overlap_sec * 1000, 1000)
    last_ok = max(duration_ms - 100, 0)  # 末尾ちょうどはフレームを取れないことがある
    plans: list[WindowPlan] = []
    start = 0
    while start < duration_ms:
        end = min(start + window_ms, duration_ms)
        length = end - start
        if plans and length < 500:  # 端数の短すぎる窓は作らない
            break
        if frames_per_window == 1:
            times = [start + length // 2]
        else:
            times = [start + i * length // (frames_per_window - 1) for i in range(frames_per_window)]
        plans.append(WindowPlan(start, end, [min(max(t, 0), last_ok) for t in times]))
        if end >= duration_ms:
            break
        start += step_ms
    return plans


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
    def _loop(self) -> None:
        while True:
            job_id = self._queue.get()
            if job_id is None:
                return
            self._run_job(job_id)

    def _run_job(self, job_id: int) -> None:
        job = self._store.get_job(job_id)
        if job is None:  # 取り込み元ごと削除された
            return
        source = self._store.get_source(job["source_id"])
        if source is None:
            return
        self._store.update_job(job_id, status="running", started_at=time.time(), error=None)
        self._store.update_source(source["id"], status="running", error=None)
        try:
            info = self._embedder.info()
            self._store.ensure_compat(info.model_id, info.dims)
            if source["kind"] == "video":
                self._index_video(job_id, source)
            elif source["kind"] == "audio":
                self._index_audio(job_id, source)
            else:
                raise ValueError(f"キューで処理できない種類です: {source['kind']}")
        except Exception as exc:
            log.exception("取り込みに失敗しました: job=%s source=%s", job_id, source["id"])
            message = f"{type(exc).__name__}: {exc}"
            self._store.update_job(job_id, status="failed", error=message, finished_at=time.time())
            self._store.update_source(source["id"], status="failed", error=message)
            return
        self._store.update_job(job_id, status="done", finished_at=time.time())
        self._store.update_source(source["id"], status="done")

    def _index_video(self, job_id: int, source: dict) -> None:
        params = IngestParams(**source["params"])
        path = source["path"]
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

        for n, plan in enumerate(plans, start=1):
            frames, times = [], []
            for t in plan.sample_ms:
                try:
                    frames.append(media.extract_frame(path, t, self._settings.image_max_side))
                    times.append(t)
                except media.MediaError:
                    log.warning("フレームを取り出せないため飛ばします: %s @%sms", path, t)
            if not frames:
                raise media.MediaError(f"{plan.start_ms}ms からの窓でフレームを1枚も取り出せません")
            slices = None
            if use_audio:
                pcm = media.extract_audio(path, plan.start_ms, plan.end_ms - plan.start_ms)
                slices = split_window_pcm(pcm, plan.start_ms, plan.end_ms, len(frames))
            parts = build_window_parts(frames, times, slices)
            vec = self._embedder.embed_parts(parts, high=False)
            kind = "tav" if any(isinstance(p, AudioPart) for p in parts) else "frames"
            self._store.add_window(
                source_id=source["id"],
                start_ms=plan.start_ms,
                end_ms=plan.end_ms,
                kind=kind,
                group_id=source["group_id"],
                abs_ts=source["start_ts"] + plan.start_ms / 1000,
                vec=vec,
            )
            self._store.update_job(job_id, progress=n)

    def _index_audio(self, job_id: int, source: dict) -> None:
        params = IngestParams(**source["params"])
        info = media.probe(source["path"])
        if not info.has_audio or info.duration_ms <= 0:
            raise media.MediaError("音声トラックが見つからないか、長さが 0 です")
        self._store.update_source(source["id"], duration_ms=info.duration_ms)
        self._store.delete_windows(source["id"])
        chunk_ms = params.chunk_sec * 1000
        starts = list(range(0, info.duration_ms, chunk_ms))
        self._store.update_job(job_id, total=len(starts), progress=0)
        for n, start in enumerate(starts, start=1):
            end = min(start + chunk_ms, info.duration_ms)
            if n > 1 and end - start < 500:
                self._store.update_job(job_id, progress=n)
                continue
            pcm = media.extract_audio(source["path"], start, end - start)
            if pcm.size == 0:
                continue
            vec = self._embedder.embed_audio(pcm, high=False)
            self._store.add_window(
                source_id=source["id"],
                start_ms=start,
                end_ms=end,
                kind="audio",
                group_id=source["group_id"],
                abs_ts=source["start_ts"] + start / 1000,
                vec=vec,
            )
            self._store.update_job(job_id, progress=n)

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
        from PIL import Image

        thumb_dir.mkdir(parents=True, exist_ok=True)
        thumb = Image.fromarray(frames[0])
        thumb.thumbnail((320, 320))
        thumb.save(thumb_dir / f"w{window_id}.jpg", format="JPEG", quality=80)
        return {"source_id": source_id, "window_id": window_id, "kind": kind}
