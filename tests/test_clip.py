"""取り込み元の一部の切り出し(GET /api/sources/{id}/clip)のテスト。"""

from __future__ import annotations

import subprocess
from dataclasses import replace

from conftest import make_video, needs_ffmpeg
from fastapi.testclient import TestClient
from PIL import Image
from test_e2e_dummy import wait_done

from mediasearch.main import create_app


def _ingest(client, path, **extra) -> int:
    r = client.post("/api/ingest/path", json={"path": str(path), **extra})
    assert r.status_code == 200, r.text
    job = wait_done(client, r.json()["job_id"])
    assert job["status"] == "done", job
    return r.json()["source_id"]


def _duration(path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


@needs_ffmpeg
def test_clip_video_copy_and_accurate(client, settings, tmp_path):
    video = tmp_path / "20260101_090000.mp4"
    make_video(video, "red", seconds=8, audio="tone")
    sid = _ingest(client, video)

    r = client.get(f"/api/sources/{sid}/clip", params={"start_ms": 2000, "end_ms": 4000})
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "video/mp4"
    assert "attachment" in r.headers["content-disposition"] and ".mp4" in r.headers["content-disposition"]
    out = tmp_path / "copy.mp4"
    out.write_bytes(r.content)
    # 合成の動画はキーフレームが先頭だけ。編集リストが無いと 4 秒になる(音声の端数で少し長くなることはある)
    assert 1.5 <= _duration(out) < 3.0

    r = client.get(f"/api/sources/{sid}/clip", params={"start_ms": 2000, "end_ms": 4000, "accurate": True})
    assert r.status_code == 200, r.text
    exact = tmp_path / "exact.mp4"
    exact.write_bytes(r.content)
    assert abs(_duration(exact) - 2.0) < 0.3

    # 一時ファイルは返し終えたら消す
    clips = settings.data_dir / "tmp" / "clips"
    assert not clips.exists() or not list(clips.iterdir())


@needs_ffmpeg
def test_clip_audio_is_m4a(client, tmp_path):
    wav = tmp_path / "tone.wav"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=6", str(wav)], check=True
    )
    sid = _ingest(client, wav, kind="audio")
    r = client.get(f"/api/sources/{sid}/clip", params={"start_ms": 1000, "end_ms": 3000})
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "audio/mp4"


@needs_ffmpeg
def test_clip_rejects_bad_ranges_images_and_unstored(client, tmp_path):
    video = tmp_path / "v.mp4"
    make_video(video, "blue", seconds=4)
    sid = _ingest(client, video)
    assert client.get(f"/api/sources/{sid}/clip", params={"start_ms": 3000, "end_ms": 1000}).status_code == 400
    assert client.get("/api/sources/999999/clip", params={"start_ms": 0, "end_ms": 1000}).status_code == 404

    photo = tmp_path / "p.png"
    Image.new("RGB", (64, 64), (0, 255, 0)).save(photo)
    pid = _ingest(client, photo, kind="image")
    assert client.get(f"/api/sources/{pid}/clip", params={"start_ms": 0, "end_ms": 1000}).status_code == 400

    other = tmp_path / "unstored.mp4"
    make_video(other, "green", seconds=4)
    uid = _ingest(client, other, store_media=False)
    assert client.get(f"/api/sources/{uid}/clip", params={"start_ms": 0, "end_ms": 1000}).status_code == 404


@needs_ffmpeg
def test_clip_length_limit(settings, tmp_path):
    video = tmp_path / "v.mp4"
    make_video(video, "red", seconds=4)
    with TestClient(create_app(replace(settings, clip_max_sec=1))) as c:
        sid = _ingest(c, video)
        r = c.get(f"/api/sources/{sid}/clip", params={"start_ms": 0, "end_ms": 2000})
        assert r.status_code == 400 and "1 秒" in r.json()["detail"]
    with TestClient(create_app(replace(settings, clip_max_sec=0))) as c:
        assert c.get(f"/api/sources/{sid}/clip", params={"start_ms": 0, "end_ms": 500}).status_code == 404
