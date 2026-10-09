"""ネットワークカメラなどの RTSP ストリームを、受信しながらベクトルにします。

1 つのストリームにつき 1 つのスレッドと 1 つの ffmpeg のプロセスを使い、接続を保ったまま連続で復号します。

- 窓は、ストリームの時刻(受信の開始からの経過)で区切ります。窓の時刻(abs_ts)は、受信を始めた時刻を起点にした実時刻です。
- 1 つの登録が 1 つの取り込み元(kind=stream)になり、窓は追加した時点で検索の対象になります
  (リアルタイム検索で通知できる)。
- 推論が追いつかない場合は、古い窓を捨てて最新に追いつきます(捨てた数は状態の dropped_windows で確認できる)。
  遅れたまま溜め続けると、メモリが増え続け、検索結果も実時刻から遅れ続けるためです。
- 接続が切れたら、間隔を広げながら(最大 60 秒)接続し直します。登録はサーバーを再起動しても残り、起動時に再開します。
- URL には認証情報(rtsp://ユーザー:パスワード@...)が含まれることがあるため、API の応答とログでは伏せます。
  ffmpeg に読ませるプロトコルは RTSP とその下位のもの(TCP / UDP / TLS / RTP)に限り、
  ローカルのファイルなどは読ませません。
"""

from __future__ import annotations

import json
import logging
import math
import os
import queue
import subprocess
import threading
import time
from urllib.parse import urlsplit, urlunsplit

import numpy as np

from . import media
from .config import Settings
from .embedders import SAMPLE_RATE, AudioPart, Embedder
from .pipeline import IngestParams, build_window_parts, plan_windows, save_thumbnail, split_window_pcm
from .store import Store

log = logging.getLogger(__name__)

ALLOWED_SCHEMES = ("rtsp", "rtsps")
# ffmpeg が使ってよいプロトコル(RTSP が内部で使うもの)。file などは含めない
PROTOCOL_WHITELIST = "rtsp,rtsps,tcp,udp,tls,rtp,srtp,crypto"
CONNECT_TIMEOUT_SEC = 15
STALL_SEC = 30  # この秒数フレームが届かなければ、切断とみなして接続し直す
MAX_BACKOFF_SEC = 60


class StreamError(ValueError):
    """ストリームの登録内容の誤り(API では 400)。"""


def mask_url(url: str) -> str:
    """URL の認証情報(パスワード)を伏せます。"""
    try:
        parts = urlsplit(url)
    except ValueError:
        return "(不正な URL)"
    if parts.password is None and parts.username is None:
        return url
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    user = parts.username or ""
    return urlunsplit((parts.scheme, f"{user}:***@{host}" if user else host, parts.path, parts.query, ""))


def validate_url(url: str) -> str:
    url = (url or "").strip()
    if not url or len(url) > 2048 or any(c.isspace() for c in url):
        raise StreamError("URL が空か、空白・改行を含んでいます")
    parts = urlsplit(url)
    if parts.scheme.lower() not in ALLOWED_SCHEMES:
        raise StreamError(f"URL は {' / '.join(s + '://' for s in ALLOWED_SCHEMES)} で始まるものにしてください")
    if not parts.hostname:
        raise StreamError("URL にホスト名がありません")
    return url


def decode_step_for(params: IngestParams) -> tuple[int, list[int], int]:
    """(復号の間隔 ms, 窓の中でフレームを取る位置 ms, 窓の開始の間隔 ms)。"""
    window_ms = params.window_sec * 1000
    stride = max(window_ms - params.overlap_sec * 1000, 1000)
    offsets = plan_windows(window_ms, params.window_sec, params.frames_per_window, 0)[0].sample_ms
    step = stride
    for t in offsets:
        step = math.gcd(step, t)
    return max(step, 100), offsets, stride


def _ffprobe_stream(url: str) -> tuple[int, int, bool]:
    """ストリームの映像の大きさと、音声の有無を調べます。"""
    cmd = ["ffprobe", "-v", "error", "-protocol_whitelist", PROTOCOL_WHITELIST]
    if urlsplit(url).scheme in ALLOWED_SCHEMES:
        cmd += ["-rtsp_transport", "tcp", "-timeout", str(CONNECT_TIMEOUT_SEC * 1_000_000)]
    cmd += ["-show_streams", "-print_format", "json", url]
    try:
        out = media._run(cmd, timeout=CONNECT_TIMEOUT_SEC + 10)
    except media.MediaError as exc:
        raise media.MediaError(str(exc).replace(url, mask_url(url))) from None
    streams = json.loads(out).get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if not video or not video.get("width"):
        raise media.MediaError("ストリームに映像がありません")
    return int(video["width"]), int(video["height"]), any(s.get("codec_type") == "audio" for s in streams)


class StreamWorker:
    """1 つのストリームを受信し続けるスレッド。"""

    def __init__(self, settings: Settings, store: Store, embedder: Embedder, stream: dict) -> None:
        self._s = settings
        self._store = store
        self._embedder = embedder
        self.stream = stream
        self.params = IngestParams(**stream["params"])
        self._stop = threading.Event()
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._hw_failed = False
        self.state: dict = {
            "status": "stopped",  # connecting / running / retrying / stopped
            "error": None,
            "connected_at": None,
            "last_frame_at": None,
            "frames": 0,
            "windows": 0,
            "dropped_windows": 0,
            "reconnects": 0,
            "lag_ms": 0,
            "decoder": None,
            "has_audio": None,
        }

    # ------------------------------------------------------------------ 開始・停止
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=f"stream-{self.stream['id']}", daemon=True)
        self._thread.start()

    def stop(self, wait: float = 10) -> None:
        self._stop.set()
        proc = self._proc
        if proc and proc.poll() is None:
            proc.kill()
        if self._thread:
            self._thread.join(wait)
        self.state["status"] = "stopped"

    @property
    def alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    # ------------------------------------------------------------------ 受信
    def _run(self) -> None:
        backoff = 1.0
        url = self.stream["url"]
        while not self._stop.is_set():
            self.state.update(status="connecting")
            started = time.monotonic()
            try:
                self._session()
                if not self._stop.is_set():
                    self.state["error"] = "ストリームが終了しました"
            except Exception as exc:  # 接続の失敗・推論の失敗などは、間隔をあけて再試行する
                message = f"{type(exc).__name__}: {exc}".replace(url, mask_url(url))
                self.state["error"] = message
                log.warning("ストリーム %s の受信に失敗しました: %s", self.stream["name"], message)
            if self._stop.is_set():
                break
            if time.monotonic() - started > 60:  # しばらく受信できていたら、間隔を戻す
                backoff = 1.0
            self.state.update(status="retrying")
            self.state["reconnects"] += 1
            self._stop.wait(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF_SEC)
        self.state["status"] = "stopped"

    def _command(self, url: str, width: int, height: int, step: int, audio_fd: int | None, hw: str | None) -> list[str]:
        cmd = ["ffmpeg", "-v", "error", "-nostdin", "-protocol_whitelist", PROTOCOL_WHITELIST]
        scheme = urlsplit(url).scheme
        if scheme in ALLOWED_SCHEMES:
            cmd += ["-rtsp_transport", "tcp", "-timeout", str(CONNECT_TIMEOUT_SEC * 1_000_000)]
        else:
            cmd += ["-re"]  # テスト用のファイル入力は、実時間の速さで読む
        if hw:
            cmd += ["-hwaccel", hw]
            if self._s.ffmpeg_hwaccel_device:
                cmd += ["-hwaccel_device", self._s.ffmpeg_hwaccel_device]
        cmd += [
            "-i", url,
            "-map", "0:v:0", "-an", "-sn", "-dn",
            "-vf", f"fps=fps={1000.0 / step:.6f}:round=near,scale={width}:{height}:flags=area",
            "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1",
        ]  # fmt: skip
        if audio_fd is not None:
            cmd += ["-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", f"pipe:{audio_fd}"]
        return cmd

    def _session(self) -> None:
        url = self.stream["url"]
        p = self.params
        src_w, src_h, has_audio = _ffprobe_stream(url)
        width, height = media.output_size(src_w, src_h, self._s.image_max_side)
        step, offsets, stride = decode_step_for(p)
        window_ms = p.window_sec * 1000
        use_audio = p.include_audio and has_audio
        hw, _ = (None, None) if self._hw_failed else media.select_hwaccel(self._s.ffmpeg_hwaccel, "")
        self.state.update(has_audio=has_audio, decoder=hw or "cpu")

        audio_r = audio_w = None
        if use_audio:
            audio_r, audio_w = os.pipe()
        cmd = self._command(url, width, height, step, audio_w, hw)
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            pass_fds=(audio_w,) if audio_w is not None else (),
        )
        self._proc = proc
        if audio_w is not None:
            os.close(audio_w)  # 書き込み側は ffmpeg だけが持つ(終了時に EOF が届くように)

        frames: queue.Queue = queue.Queue()
        audio_buf = bytearray()
        audio_lock = threading.Lock()
        size = width * height * 3

        def read_video() -> None:
            k = 0
            while True:
                buf = proc.stdout.read(size)
                if len(buf) < size:
                    frames.put(None)
                    return
                frames.put((k, np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3)))
                k += 1

        def read_audio() -> None:
            with os.fdopen(audio_r, "rb", buffering=0) as f:
                while chunk := f.read(65536):
                    with audio_lock:
                        audio_buf.extend(chunk)

        readers = [threading.Thread(target=read_video, daemon=True)]
        if use_audio:
            readers.append(threading.Thread(target=read_audio, daemon=True))
        for th in readers:
            th.start()

        source = self._store.get_source(self.stream["source_id"])
        if source is None:
            raise RuntimeError("取り込み元が削除されています")
        info = self._embedder.info()
        self._store.ensure_compat(info.model_id, info.dims)

        wall0: float | None = None  # 最初のフレームを受け取った実時刻(ストリームの時刻 0)
        audio_base = 0  # audio_buf の先頭が、ストリームの何サンプル目か
        held: dict[int, np.ndarray] = {}
        n = 0  # 次に作る窓
        latest = -1
        try:
            while not self._stop.is_set():
                try:
                    item = frames.get(timeout=STALL_SEC)
                except queue.Empty:
                    raise media.MediaError(f"{STALL_SEC} 秒間フレームが届きません") from None
                if item is None:
                    err = proc.stderr.read().decode("utf-8", "replace").strip() if proc.stderr else ""
                    if latest < 0 and hw:
                        self._hw_failed = True  # GPU で復号できなかった。次の接続から CPU で復号する
                    raise media.MediaError(" / ".join(err.splitlines()[-2:]) or "ストリームが終了しました")
                k, frame = item
                if wall0 is None:
                    wall0 = time.time()
                    self.state.update(status="running", connected_at=wall0, error=None)
                latest = k
                held[k] = frame
                self.state["frames"] += 1
                self.state["last_frame_at"] = time.time()

                # 推論が遅れている場合は、古い窓を捨てて追いつく
                max_lag = max(3 * window_ms, 10_000)
                while (latest * step) - (n * stride + window_ms) > max_lag:
                    n += 1
                    self.state["dropped_windows"] += 1
                self.state["lag_ms"] = max(0, latest * step - (n * stride + window_ms))

                while True:
                    start = n * stride
                    need = [(start + off) // step for off in offsets]
                    if latest < max(need):
                        break
                    picked = [(held[i], start + off) for i, off in zip(need, offsets, strict=True) if i in held]
                    slices = None
                    if use_audio:
                        with audio_lock:
                            lo = start * SAMPLE_RATE // 1000 - audio_base
                            hi = (start + window_ms) * SAMPLE_RATE // 1000 - audio_base
                            pcm = np.frombuffer(bytes(audio_buf[2 * max(lo, 0) : 2 * max(hi, 0)]), dtype="<i2")
                        pcm = pcm.astype(np.float32) / 32768.0
                        slices = split_window_pcm(pcm, start, start + window_ms, len(picked)) if pcm.size else None
                    if picked:
                        self._store_window(source, wall0, start, window_ms, picked, slices)
                    n += 1
                    # 次の窓で使わないフレームと音声を捨てる
                    keep = (n * stride) // step
                    for i in [i for i in held if i < keep]:
                        del held[i]
                    if use_audio:
                        with audio_lock:
                            drop = (n * stride) * SAMPLE_RATE // 1000 - audio_base
                            if drop > 0:
                                del audio_buf[: 2 * drop]
                                audio_base += drop
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait()
            self._proc = None

    def _store_window(
        self,
        source: dict,
        wall0: float,
        start: int,
        window_ms: int,
        picked: list[tuple[np.ndarray, int]],
        slices: list | None,
    ) -> None:
        images = [f for f, _ in picked]
        times = [t for _, t in picked]
        parts = build_window_parts(images, times, slices)
        vec = self._embedder.embed_parts(parts, high=False)
        kind = "tav" if any(isinstance(p, AudioPart) for p in parts) else "frames"
        abs_ts = wall0 + start / 1000
        rel_ms = int(round((abs_ts - source["start_ts"]) * 1000))  # 取り込み元(登録時刻)からの経過
        window_id = self._store.add_window(
            source_id=source["id"],
            start_ms=rel_ms,
            end_ms=rel_ms + window_ms,
            kind=kind,
            group_id=source["group_id"],
            abs_ts=abs_ts,
            vec=vec,
        )
        if self.params.store_media:
            save_thumbnail(images[len(images) // 2], self._s.data_dir / "thumbs" / f"w{window_id}.jpg")
        self.state["windows"] += 1
        self._store.update_source(source["id"], duration_ms=rel_ms + window_ms)


class StreamManager:
    """登録されたストリームの受信を管理します(起動時に再開し、停止時に止める)。"""

    def __init__(self, settings: Settings, store: Store, embedder: Embedder) -> None:
        self._s = settings
        self._store = store
        self._embedder = embedder
        self._workers: dict[int, StreamWorker] = {}
        self._lock = threading.Lock()

    def start(self) -> None:
        for stream in self._store.list_streams():
            if stream["enabled"]:
                self._start_worker(stream)

    def stop(self) -> None:
        with self._lock:
            workers = list(self._workers.values())
            self._workers.clear()
        for w in workers:
            w.stop(wait=5)

    def _start_worker(self, stream: dict) -> None:
        with self._lock:
            old = self._workers.get(stream["id"])
            if old and old.alive:
                return
            worker = StreamWorker(self._s, self._store, self._embedder, stream)
            self._workers[stream["id"]] = worker
        worker.start()

    # ------------------------------------------------------------------ 操作(API から使う)
    def create(
        self,
        *,
        url: str,
        name: str | None,
        group_id: str | None,
        location: str | None,
        params: IngestParams,
        enabled: bool = True,
    ) -> dict:
        url = validate_url(url)
        active = sum(1 for s in self._store.list_streams() if s["enabled"])
        if enabled and active >= self._s.max_streams:
            raise StreamError(f"同時に受信できるストリームは {self._s.max_streams} 本までです(MAX_STREAMS)")
        name = name or (urlsplit(url).hostname or "stream") + urlsplit(url).path
        source_id = self._store.add_source(
            kind="stream",
            path=None,
            name=name,
            group_id=group_id,
            location=location,
            start_ts=time.time(),
            params=params.to_dict(),
            duration_ms=0,
            status="running",
        )
        stream_id = self._store.add_stream(
            name=name, url=url, group_id=group_id, location=location, params=params.to_dict(), source_id=source_id,
            enabled=enabled,
        )  # fmt: skip
        stream = self._store.get_stream(stream_id)
        if enabled:
            self._start_worker(stream)
        log.info("ストリームを登録しました: %s(%s)", name, mask_url(url))
        return self.describe(stream_id)

    def set_enabled(self, stream_id: int, enabled: bool) -> dict:
        stream = self._get(stream_id)
        if enabled and not stream["enabled"]:
            active = sum(1 for s in self._store.list_streams() if s["enabled"])
            if active >= self._s.max_streams:
                raise StreamError(f"同時に受信できるストリームは {self._s.max_streams} 本までです(MAX_STREAMS)")
        self._store.update_stream(stream_id, enabled=enabled)
        if enabled:
            self._start_worker(self._store.get_stream(stream_id))
        else:
            with self._lock:
                worker = self._workers.pop(stream_id, None)
            if worker:
                worker.stop()
        if stream["source_id"]:
            self._store.update_source(stream["source_id"], status="running" if enabled else "done")
        return self.describe(stream_id)

    def delete(self, stream_id: int, delete_data: bool = False) -> None:
        stream = self._get(stream_id)
        with self._lock:
            worker = self._workers.pop(stream_id, None)
        if worker:
            worker.stop()
        self._store.delete_stream(stream_id)
        if stream["source_id"]:
            if delete_data:
                self._store.delete_source(stream["source_id"])
            else:
                self._store.update_source(stream["source_id"], status="done")

    def _get(self, stream_id: int) -> dict:
        stream = self._store.get_stream(stream_id)
        if stream is None:
            raise LookupError("ストリームが見つかりません")
        return stream

    def describe(self, stream_id: int) -> dict:
        stream = self._get(stream_id)
        return self._describe(stream)

    def _describe(self, stream: dict) -> dict:
        worker = self._workers.get(stream["id"])
        state = dict(worker.state) if worker else {"status": "stopped"}
        if worker and not worker.alive:
            state["status"] = "stopped"
        out = {k: v for k, v in stream.items() if k != "url"}
        out["url"] = mask_url(stream["url"])
        out["enabled"] = bool(stream["enabled"])
        out["state"] = state
        return out

    def list(self) -> list[dict]:
        return [self._describe(s) for s in self._store.list_streams()]
