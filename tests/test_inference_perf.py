"""推論優先度キュー・Producer-Consumer パイプライン・キャッシュ永続化/ウォームアップ・前処理最適化のテスト。"""

from __future__ import annotations

import io
import threading
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from conftest import make_video, needs_ffmpeg
from fastapi.testclient import TestClient
from PIL import Image
from test_e2e_dummy import wait_done

from mediasearch import media
from mediasearch.config import Settings
from mediasearch.embedders.batch import MicroBatcher
from mediasearch.embedders.dummy import DummyEmbedder
from mediasearch.embedders.gate import PriorityGate
from mediasearch.embedders.local import configure_torch_cache, move_features, pin_features
from mediasearch.embedders.remote import RemoteEmbedder
from mediasearch.embedders.wire import image_to_jpeg, jpeg_to_image
from mediasearch.main import create_app
from mediasearch.pipeline import decode_select_grid, decode_step_ms, plan_windows
from mediasearch.service import Service


# ---------------------------------------------------------------------- Task 1: PriorityQueue 制御
def test_priority_gate_high_preempts_low_and_preserves_fifo():
    """高優先(検索)が待機中の低優先(取り込み)より先に実行され、同一優先度内は FIFO 順を守ること。"""
    gate = PriorityGate()
    order: list[str] = []
    holder_entered = threading.Event()
    release_holder = threading.Event()

    def hold_low() -> None:
        with gate.acquire(high=False):
            holder_entered.set()
            release_holder.wait(timeout=5)

    def waiter(name: str, high: bool) -> None:
        with gate.acquire(high=high):
            order.append(name)

    holder = threading.Thread(target=hold_low, daemon=True)
    holder.start()
    assert holder_entered.wait(timeout=5)

    def enqueue(name: str, high: bool, expected_waiting: int) -> threading.Thread:
        th = threading.Thread(target=waiter, args=(name, high), daemon=True)
        th.start()
        deadline = time.monotonic() + 5
        while gate.snapshot()["waiting"] < expected_waiting and time.monotonic() < deadline:
            time.sleep(0.002)
        assert gate.snapshot()["waiting"] >= expected_waiting
        return th

    threads = [
        enqueue("low-1", False, 1),
        enqueue("low-2", False, 2),
        enqueue("high-1", True, 3),
        enqueue("high-2", True, 4),
        enqueue("high-3", True, 5),
    ]
    assert gate.high_waiting == 3 and gate.has_high_waiting()

    release_holder.set()
    holder.join(timeout=5)
    for th in threads:
        th.join(timeout=5)

    assert order == ["high-1", "high-2", "high-3", "low-1", "low-2"]
    snap = gate.snapshot()
    assert snap["calls"] == {"high": 3, "low": 3}
    assert snap["preemptions"] == 3
    assert snap["waiting"] == 0 and snap["high_waiting"] == 0


def test_priority_gate_search_preempts_between_image_chunks():
    """画像バッチ取り込み(batch_size=1)の各チャンクの合間に検索が最優先で割り込めること。"""
    events: list[str] = []

    class _SlowImageDummy(DummyEmbedder):
        def embed_images(self, images, high=False):
            step = max(self._batch, 1)
            chunks = []
            for i in range(0, len(images), step):
                with self._gate.acquire(high=high):
                    events.append(f"img-{i}")
                    time.sleep(0.05)
                    chunks.append(np.ones((len(images[i : i + step]), self._dims), dtype=np.float32))
            return np.concatenate(chunks, axis=0)

        def embed_texts(self, texts, kind="query"):
            with self._gate.acquire(high=(kind == "query")):
                events.append("search")
                return super()._parts([])[None, :]

    emb = _SlowImageDummy(dims=128, batch_size=1)
    imgs = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(6)]
    t_ing = threading.Thread(target=lambda: emb.embed_images(imgs, high=False), daemon=True)
    t_ing.start()
    time.sleep(0.02)  # img-0 実行中に検索を投げる
    t0 = time.perf_counter()
    emb.embed_texts(["query"], kind="query")
    search_sec = time.perf_counter() - t0
    t_ing.join(timeout=5)

    # 全 6 枚(300ms)の完了を待たず、img-0 の直後(約 50ms)に検索が割り込む
    assert events.index("search") <= 2, events
    assert search_sec < 0.15


def test_micro_batcher_yields_wait_when_high_waiting():
    """高優先が待機中なら MicroBatcher が wait_sec を待たずに即座にバッチを解放すること。"""
    batcher = MicroBatcher(lambda items: [x * 2 for x in items], max_batch=8, wait_sec=0.5, should_yield=lambda: True)
    try:
        t0 = time.perf_counter()
        assert batcher.submit(21).result(timeout=2) == 42
        assert time.perf_counter() - t0 < 0.15
    finally:
        batcher.close()


# ---------------------------------------------------------------------- Task 2: Producer-Consumer
@needs_ffmpeg
def test_video_ingest_overlaps_prepare_and_embed_with_bounded_backpressure(settings, tmp_path):
    """動画取り込みで前処理(decode)と推論(ingest-embed)が重なり、有界キューのバックプレッシャーが効くこと。"""
    video = tmp_path / "20260101_090000.mp4"
    make_video(video, "blue", seconds=16)  # 2 秒窓 = 8 窓

    class _OverlapTracker(DummyEmbedder):
        def __init__(self) -> None:
            super().__init__(dims=256, batch_size=1)
            self._lock = threading.Lock()
            self.in_flight = 0
            self.max_in_flight = 0
            self.prep_threads: set[str] = set()
            self.embed_threads: set[str] = set()
            self.prep_times: list[tuple[float, float]] = []
            self.embed_times: list[tuple[float, float]] = []

        def prepare_parts_batch(self, items):
            t0 = time.perf_counter()
            time.sleep(0.04)  # CPU 前処理(テンソル化)を模擬
            t1 = time.perf_counter()
            with self._lock:
                self.prep_threads.add(threading.current_thread().name)
                self.prep_times.append((t0, t1))
                self.in_flight += 1
                self.max_in_flight = max(self.max_in_flight, self.in_flight)
            return list(items)

        def embed_prepared_parts(self, prepared, high=False):
            with self._gate.acquire(high=high):
                t0 = time.perf_counter()
                time.sleep(0.10)  # 推論を模擬
                out = [self._parts(parts) for parts in prepared]
                t1 = time.perf_counter()
            with self._lock:
                self.embed_threads.add(threading.current_thread().name)
                self.embed_times.append((t0, t1))
                self.in_flight -= 1
            return out

    tracker = _OverlapTracker()
    s = replace(settings, ingest_prefetch=2, dims=256)
    with TestClient(create_app(s, embedder=tracker)) as c:
        job = c.post("/api/ingest/path", json={"path": str(video)}).json()
        done = wait_done(c, job["job_id"])
        assert done["status"] == "done" and done["total"] == 8
        assert done["timings"]["prefetch"] == 2

    # 前処理と推論が別スレッドで動いていること
    assert tracker.prep_threads == {"decode"}
    assert tracker.embed_threads and tracker.embed_threads.isdisjoint(tracker.prep_threads)
    assert all(name.startswith("ingest-embed") for name in tracker.embed_threads)
    assert len(tracker.prep_times) == 8 and len(tracker.embed_times) == 8
    # 窓 1 の前処理が、窓 0 の推論が終わる前に開始されている(オーバーラップ)
    assert tracker.prep_times[1][0] < tracker.embed_times[0][1]
    # バックプレッシャー: maxsize=2 + 推論プール 2 + put 待ち 1 の範囲内に収まり、8 窓全部は先行生成されない
    assert tracker.max_in_flight <= 5 < 8


def test_pin_and_move_features_helpers():
    """pin_features と move_features が CPU での冗長な .to() を省き、別デバイス時のみ non_blocking 転送すること。"""

    class _Dev:
        def __init__(self, typ: str) -> None:
            self.type = typ

        def __eq__(self, other: object) -> bool:
            return isinstance(other, _Dev) and self.type == other.type

    class _FakeTensor:
        def __init__(self, dev: str, pinned: bool = False) -> None:
            self.device = _Dev(dev)
            self._pinned = pinned
            self.pin_calls = 0
            self.to_calls: list[tuple[str, bool]] = []

        def is_pinned(self) -> bool:
            return self._pinned

        def pin_memory(self):
            self.pin_calls += 1
            return _FakeTensor(self.device.type, pinned=True)

        def to(self, target, non_blocking: bool = False):
            self.to_calls.append((target.type, non_blocking))
            t = _FakeTensor(target.type, pinned=self._pinned)
            t.to_calls = self.to_calls
            return t

    cpu_t = _FakeTensor("cpu")
    pinned = pin_features({"input_ids": cpu_t, "modality": "text"})
    assert cpu_t.pin_calls == 1 and pinned["input_ids"].is_pinned() and pinned["modality"] == "text"

    # CPU -> CPU では .to() が一切呼ばれない
    same = move_features({"input_ids": cpu_t, "modality": "text"}, _Dev("cpu"), non_blocking=False)
    assert same["input_ids"] is cpu_t and cpu_t.to_calls == []

    # CPU -> CUDA では non_blocking=True で転送される
    moved = move_features({"input_ids": pinned["input_ids"]}, _Dev("cuda"), non_blocking=True)
    assert moved["input_ids"].device == _Dev("cuda")
    assert moved["input_ids"].to_calls == [("cuda", True)]


# ---------------------------------------------------------------------- Task 3: キャッシュ永続化 & ウォームアップ
def test_configure_torch_cache_sets_persistent_dirs_and_respects_overrides(tmp_path, monkeypatch):
    """未指定時は DATA_DIR 配下にキャッシュを設定・作成し、環境変数での上書きを尊重すること。"""
    for key in ("TORCHINDUCTOR_CACHE_DIR", "TORCHINDUCTOR_FX_GRAPH_CACHE", "TRITON_CACHE_DIR"):
        monkeypatch.delenv(key, raising=False)
    cache_root = tmp_path / "torch_cache"
    active = configure_torch_cache(cache_root)
    assert Path(active["TORCHINDUCTOR_CACHE_DIR"]) == cache_root / "inductor"
    assert Path(active["TRITON_CACHE_DIR"]) == cache_root / "triton"
    assert active["TORCHINDUCTOR_FX_GRAPH_CACHE"] == "1"
    assert (cache_root / "inductor").is_dir() and (cache_root / "triton").is_dir()

    # 明示的に環境変数で上書きした場合は上書きしない
    custom = tmp_path / "custom_inductor"
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(custom))
    active2 = configure_torch_cache(tmp_path / "other_cache")
    assert Path(active2["TORCHINDUCTOR_CACHE_DIR"]) == custom and custom.is_dir()


def test_startup_warmup_runs_before_readyz_and_gates_readiness(settings, tmp_path):
    """起動時ウォームアップが text / image / audio の 3 モダリティを実行し、完了まで /readyz が 503 になること。"""
    seen: list[str] = []

    class _SpyEmbedder(DummyEmbedder):
        def embed_texts(self, texts, kind="query"):
            seen.append("text")
            return super().embed_texts(texts, kind=kind)

        def embed_images(self, images, high=False):
            seen.append("image")
            return super().embed_images(images, high=high)

        def embed_audio(self, pcm, high=False):
            seen.append("audio")
            return super().embed_audio(pcm, high=high)

    # 1) lifespan 開始前(warmup 未完了)は /readyz が 503 になる
    app = create_app(settings, embedder=_SpyEmbedder(768))
    ok_before, body_before = Service(app.state.ctx).readiness()
    assert not ok_before and body_before["status"] == "not_ready"
    assert body_before["checks"]["embedder"]["ok"] is False
    assert body_before["checks"]["embedder"]["status"] == "warming_up"

    # 2) lifespan 実行後は text / image / audio が各 1 回実行され、/readyz が 200 ready になる
    with TestClient(app) as c:
        assert seen == ["text", "image", "audio"]
        r = c.get("/readyz")
        assert r.status_code == 200 and r.json()["status"] == "ready"
        assert r.json()["checks"]["embedder"]["warmed_up"] is True

    # 3) WARMUP_ON_START=false ならスキップしてそのまま ready
    seen.clear()
    s_no_warm = replace(settings, data_dir=tmp_path / "nowarm", warmup_on_start=False)
    with TestClient(create_app(s_no_warm, embedder=_SpyEmbedder(768))) as c:
        assert seen == []
        r = c.get("/readyz")
        assert r.status_code == 200 and r.json()["checks"]["embedder"]["warmed_up"] is False

    # 4) ROLE=compute でもウォームアップが走り /readyz が 200 になる
    seen.clear()
    s_compute = replace(settings, role="compute", data_dir=tmp_path / "comp")
    with TestClient(create_app(s_compute, embedder=_SpyEmbedder(768))) as c:
        assert seen == ["text", "image", "audio"]
        r = c.get("/readyz")
        assert r.status_code == 200 and r.json()["checks"]["embedder"]["warmed_up"] is True

    # 5) RemoteEmbedder はネットワークを呼ばずに warmup をスキップする
    remote = RemoteEmbedder("http://127.0.0.1:1", dims=768)
    try:
        assert remote.warmup()["skipped"] is True
    finally:
        remote.close()


# ---------------------------------------------------------------------- Task 4: 前処理・FFmpeg 最適化(ビット一致)
def test_decode_select_grid_patterns():
    """2 秒窓・2 フレーム等の規則的な格子で select 間引きが有効になり、不規則な端数がある場合は無効になること。"""
    plans_60s = plan_windows(60_000, window_sec=2, frames_per_window=2, overlap_sec=0)
    step = decode_step_ms(plans_60s)
    need = [[round(t / step) for t in p.sample_ms] for p in plans_60s]
    assert decode_select_grid(need) == (1, 2)

    plans_9s = plan_windows(9_000, window_sec=2, frames_per_window=2, overlap_sec=0)
    step9 = decode_step_ms(plans_9s)
    need9 = [[round(t / step9) for t in p.sample_ms] for p in plans_9s]
    assert decode_select_grid(need9) == (0, 1)


@needs_ffmpeg
def test_select_before_scale_is_bit_exact_and_writable(tmp_path):
    """select フィルタによる間引き前後で画素がビット単位で完全一致し、配列が書き込み可能であること。"""
    video = tmp_path / "v.mp4"
    make_video(video, "green", seconds=6)
    full = list(media.VideoFrameReader(video, step_ms=500, max_side=160, skip="none"))
    reader_sel = media.VideoFrameReader(video, step_ms=500, max_side=160, skip="none", select_offset=1, select_stride=2)
    sel = list(reader_sel)
    full_odd = full[1::2]
    assert len(sel) == len(full_odd) and len(sel) == 6 and reader_sel.frames == 6
    for (t_a, img_a), (t_b, img_b) in zip(full_odd, sel, strict=True):
        assert t_a == t_b
        assert np.array_equal(img_a, img_b)
        assert img_b.flags.writeable


@needs_ffmpeg
def test_probe_provides_dimensions_avoiding_second_ffprobe(tmp_path, monkeypatch):
    """media.probe が解像度を返し、VideoFrameReader に渡すと 2 回目の ffprobe(video_size) を呼ばないこと。"""
    video = tmp_path / "v.mp4"
    make_video(video, "red", seconds=2)
    info = media.probe(video)
    assert (info.width, info.height) == (320, 240)

    def _boom(_path):
        raise AssertionError("video_size(2 回目の ffprobe)は呼ばれないはず")

    monkeypatch.setattr(media, "video_size", _boom)
    reader = media.VideoFrameReader(video, step_ms=1000, max_side=160, source_size=(info.width, info.height))
    assert (reader.width, reader.height) == (160, 120)
    assert len(list(reader)) >= 1


def test_audio_scaling_and_rgb_loading_bit_exact(tmp_path):
    """PCM のインプレース正規化が全 int16 値でビット一致し、RGB 画像の読み込みも一致すること。"""
    raw = np.arange(-32768, 32768, dtype="<i2")
    ref = np.asarray(raw, dtype=np.float32) / 32768.0
    fast = np.asarray(raw, dtype=np.float32)
    fast *= media._INV_INT16
    assert np.array_equal(ref, fast)

    arr = np.full((64, 80, 3), [12, 134, 210], dtype=np.uint8)
    jpg = image_to_jpeg(arr)
    decoded = jpeg_to_image(jpg)
    assert decoded.shape == (64, 80, 3) and decoded.dtype == np.uint8

    p = tmp_path / "rgb.png"
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    p.write_bytes(buf.getvalue())
    loaded = media.load_image(p, max_side=448)
    assert np.array_equal(loaded, arr)

    # Settings.from_env の既定値確認
    s = Settings.from_env()
    assert s.ingest_prefetch == 2 and s.warmup_on_start is True and s.torch_compile == "auto"
    assert s.torch_cache_dir == s.data_dir / "cache" / "torch"
