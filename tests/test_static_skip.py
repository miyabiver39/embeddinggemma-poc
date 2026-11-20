"""変化の無い区間の省略(SKIP_STATIC)のテスト。"""

from __future__ import annotations

import subprocess
from dataclasses import replace

import numpy as np
from conftest import make_video, needs_ffmpeg
from fastapi.testclient import TestClient
from test_e2e_dummy import wait_done

from mediasearch import pipeline
from mediasearch.main import create_app
from mediasearch.pipeline import block_change, frame_signature, is_static_window


def _img(value: int, box: tuple[int, int] | None = None) -> np.ndarray:
    img = np.full((252, 448, 3), value, dtype=np.uint8)
    if box is not None:  # 小さな物体(20×20 ピクセル。画面の 0.35%)
        y, x = box
        img[y : y + 20, x : x + 20] = 255
    return img


def test_small_object_is_detected_but_global_mean_would_miss_it():
    a, b = frame_signature(_img(60)), frame_signature(_img(60, box=(100, 200)))
    assert float(np.abs(a - b).mean()) < 1.0  # 画面全体の平均では、ほぼ差が無い
    assert block_change(a, b) > 20  # 区画ごとの最大では、はっきり差が出る


def test_sensor_noise_is_static():
    rng = np.random.default_rng(0)
    base = _img(80).astype(np.int16)
    noisy = np.clip(base + rng.integers(-6, 7, base.shape), 0, 255).astype(np.uint8)
    assert is_static_window([frame_signature(noisy)], [frame_signature(base.astype(np.uint8))], 4.0)


def test_window_with_different_frame_count_is_not_static():
    sig = frame_signature(_img(10))
    assert not is_static_window([sig], [sig, sig], 4.0)
    assert not is_static_window([sig], None, 4.0)


def _settings(settings, **kw):
    return replace(settings, skip_static=True, **kw)


def _windows(client, source_id):
    body = {"query": "x", "top_k": 1000, "merge": False, "kind": "all", "min_score": -1}
    res = client.post("/api/search/text", json=body).json()
    return sorted((h["start_ms"], h["end_ms"]) for h in res["results"] if h["source_id"] == source_id)


def _ingest(client, path):
    r = client.post("/api/ingest/path", json={"path": str(path)})
    assert r.status_code == 200, r.text
    job = wait_done(client, r.json()["job_id"])
    assert job["status"] == "done", job
    return r.json()["source_id"], job


@needs_ffmpeg
def test_static_video_becomes_one_window(settings, tmp_path):
    video = tmp_path / "still.mp4"
    make_video(video, "gray", seconds=12)
    with TestClient(create_app(_settings(settings))) as c:
        sid, job = _ingest(c, video)
        wins = _windows(c, sid)
        assert len(wins) == 1, wins  # 2 秒窓 6 件が 1 件にまとまる
        assert wins[0][0] == 0 and wins[0][1] >= 11_000
        assert job["timings"]["skipped_static"] == 5
        assert job["progress"] == job["total"] == 6  # 省略した窓も処理済みに数える


@needs_ffmpeg
def test_noisy_static_video_is_skipped(settings, tmp_path):
    video = tmp_path / "noisy.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=gray:s=320x240:r=10:d=10",
         "-vf", "noise=alls=8:allf=t", "-pix_fmt", "yuv420p", str(video)],
        check=True,
    )  # fmt: skip
    with TestClient(create_app(_settings(settings))) as c:
        sid, _ = _ingest(c, video)
        assert len(_windows(c, sid)) == 1


@needs_ffmpeg
def test_moving_small_object_is_not_skipped(settings, tmp_path):
    video = tmp_path / "moving.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=gray:s=320x240:r=10:d=12",
         "-f", "lavfi", "-i", "color=c=white:s=16x16:r=10:d=12",
         "-filter_complex", "[0][1]overlay=x='mod(t*25,300)':y=110", "-pix_fmt", "yuv420p", str(video)],
        check=True,
    )  # fmt: skip
    with TestClient(create_app(_settings(settings))) as c:
        sid, job = _ingest(c, video)
        assert len(_windows(c, sid)) == 6  # 小さな物体が動き続けるため、どの窓も省略しない
        assert job["timings"]["skipped_static"] == 0


@needs_ffmpeg
def test_extension_is_capped(settings, tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "MAX_STATIC_SPAN_MS", 6_000)
    video = tmp_path / "still.mp4"
    make_video(video, "gray", seconds=12)
    with TestClient(create_app(_settings(settings))) as c:
        sid, _ = _ingest(c, video)
        wins = _windows(c, sid)
        assert len(wins) == 2, wins  # 6 秒ごとに推論し直す
        assert all(end - start <= 6_000 for start, end in wins)


@needs_ffmpeg
def test_disabled_by_default(client, tmp_path):
    video = tmp_path / "still.mp4"
    make_video(video, "gray", seconds=8)
    sid, job = _ingest(client, video)
    assert len(_windows(client, sid)) == 4
    assert "skipped_static" not in job["timings"]
