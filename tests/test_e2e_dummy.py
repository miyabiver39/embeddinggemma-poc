"""ダミー埋め込みでの取り込み〜検索の通しテスト(モデル不要・数秒で終わる)。"""

from __future__ import annotations

import io
import time

from conftest import make_video, needs_ffmpeg
from PIL import Image


def wait_done(client, job_id: int, timeout: float = 60) -> dict:
    end = time.time() + timeout
    while time.time() < end:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "failed"):
            return job
        time.sleep(0.2)
    raise AssertionError("ジョブが終わりません")


def test_healthz_and_index(client):
    assert client.get("/healthz").json()["status"] == "ok"
    assert "vmsembed" in client.get("/").text
    info = client.get("/api/info").json()
    assert info["embedder"]["backend"] == "dummy"


@needs_ffmpeg
def test_ingest_and_search(client, tmp_path):
    red, blue = tmp_path / "red.mp4", tmp_path / "blue.mp4"
    make_video(red, "red")
    make_video(blue, "blue")
    jobs = []
    for p, cam in ((red, "cam1"), (blue, "cam2")):
        with open(p, "rb") as f:
            r = client.post("/api/ingest/video", files={"file": (p.name, f, "video/mp4")},
                            data={"camera_id": cam, "location": "玄関", "start_ts": "2026-01-01T09:00:00+09:00"})
        assert r.status_code == 200, r.text
        jobs.append(r.json()["job_id"])
    for j in jobs:
        job = wait_done(client, j)
        assert job["status"] == "done", job

    res = client.post("/api/search/text", json={"query": "red", "top_k": 5}).json()
    assert res["results"], res
    assert res["results"][0]["camera_id"] == "cam1"

    # カメラIDで絞り込み
    res = client.post("/api/search/text", json={"query": "red", "camera_id": "cam2"}).json()
    assert all(h["camera_id"] == "cam2" for h in res["results"])

    # 画像検索(青い画像 → 青い動画)
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (0, 0, 255)).save(buf, "JPEG")
    res = client.post("/api/search/image", files={"file": ("q.jpg", buf.getvalue(), "image/jpeg")}).json()
    assert res["results"][0]["camera_id"] == "cam2"

    # 時刻範囲の外では何も返らない
    res = client.post("/api/search/text", json={"query": "red", "from_ts": "2030-01-01T00:00:00+09:00"}).json()
    assert res["results"] == []

    # サムネイルと動画配信
    hit = client.post("/api/search/text", json={"query": "red"}).json()["results"][0]
    assert client.get(hit["thumb_url"]).status_code == 200
    assert client.get(hit["media_url"], headers={"Range": "bytes=0-99"}).status_code in (200, 206)

    # 削除
    sid = hit["source_id"]
    assert client.delete(f"/api/sources/{sid}").status_code == 200


@needs_ffmpeg
def test_ingest_path_and_audio(client, tmp_path):
    v = tmp_path / "tone.mp4"
    make_video(v, "green", audio="tone")
    r = client.post("/api/ingest/path", json={"path": str(v), "include_audio": True})
    assert r.status_code == 200, r.text
    assert wait_done(client, r.json()["job_id"])["status"] == "done"
    kinds = client.get("/api/info").json()["index"]
    assert kinds["windows_by_kind"].get("tav", 0) > 0

    # 音声ファイルとして取り込むと audio の窓になり、音声クエリの auto はそれを探す
    r = client.post("/api/ingest/path", json={"path": str(v), "kind": "audio"})
    assert wait_done(client, r.json()["job_id"])["status"] == "done"
    r = client.post("/api/search/audio", files={"file": ("tone.mp4", v.read_bytes())})
    assert r.status_code == 200, r.text
    hits = r.json()["results"]
    assert hits and {h["kind"] for h in hits} == {"audio"}


def test_ingest_path_missing(client):
    assert client.post("/api/ingest/path", json={"path": "/nonexistent.mp4"}).status_code == 400


def test_dims_mismatch_refused(settings):
    """同じDBを別の次元で開くと、検索・取り込みは 409 で拒否される(混在を防ぐ)。"""
    from dataclasses import replace

    from fastapi.testclient import TestClient

    from vmsembed.main import create_app

    with TestClient(create_app(settings)) as c:
        assert c.post("/api/search/text", json={"query": "x"}).status_code == 200
    with TestClient(create_app(replace(settings, dims=256))) as c:
        assert c.post("/api/search/text", json={"query": "x"}).status_code == 409
