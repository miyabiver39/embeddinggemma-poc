"""複数のファイルの同時取り込み(INGEST_WORKERS)のテスト。"""

from __future__ import annotations

import threading
import time
from dataclasses import replace

from conftest import make_video, needs_ffmpeg
from fastapi.testclient import TestClient
from test_e2e_dummy import wait_done

from mediasearch import pipeline
from mediasearch.config import Settings
from mediasearch.embedders.dummy import DummyEmbedder
from mediasearch.main import create_app


def test_workers_setting_is_clamped(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    assert Settings.from_env().ingest_workers == 1  # 既定は今までどおり 1 件ずつ
    monkeypatch.setenv("INGEST_WORKERS", "100")
    assert Settings.from_env().ingest_workers == 8
    monkeypatch.setenv("INGEST_WORKERS", "0")
    assert Settings.from_env().ingest_workers == 1


def _track_videos(monkeypatch):
    """_index_video の同時の実行数を、全体と取り込み元ごとに記録する。"""
    lock = threading.Lock()
    state = {"now": 0, "max": 0, "per_source": {}, "max_per_source": 0}
    original = pipeline.Ingestor._index_video

    def wrapped(self, job_id, source):
        with lock:
            state["now"] += 1
            state["max"] = max(state["max"], state["now"])
            n = state["per_source"].get(source["id"], 0) + 1
            state["per_source"][source["id"]] = n
            state["max_per_source"] = max(state["max_per_source"], n)
        try:
            time.sleep(0.5)  # 重なりを確かめられるよう、少し長く処理させる
            return original(self, job_id, source)
        finally:
            with lock:
                state["now"] -= 1
                state["per_source"][source["id"]] -= 1

    monkeypatch.setattr(pipeline.Ingestor, "_index_video", wrapped)
    return state


def _submit(client, path) -> dict:
    r = client.post("/api/ingest/path", json={"path": str(path)})
    assert r.status_code == 200, r.text
    return r.json()


@needs_ffmpeg
def test_two_workers_ingest_files_at_the_same_time(settings, tmp_path, monkeypatch):
    state = _track_videos(monkeypatch)
    videos = []
    for i, color in enumerate(("red", "blue", "green")):
        videos.append(tmp_path / f"v{i}.mp4")
        make_video(videos[-1], color, seconds=4)
    with TestClient(create_app(replace(settings, ingest_workers=2))) as c:
        assert c.get("/api/info").json()["index"]["ingest_workers"] == 2
        jobs = [_submit(c, v)["job_id"] for v in videos]
        for j in jobs:
            assert wait_done(c, j)["status"] == "done"
        assert state["max"] == 2  # 2 件まで同時に動き、3 件目は待つ
        # どれも正しく取り込まれ、検索で見分けられる
        res = c.post("/api/search/text", json={"query": "blue", "top_k": 3}).json()["results"]
        assert res[0]["source_name"] == "v1.mp4"
        assert c.get("/api/info").json()["index"]["running_jobs"] == 0


@needs_ffmpeg
def test_default_ingests_one_at_a_time(client, tmp_path, monkeypatch):
    state = _track_videos(monkeypatch)
    jobs = []
    for i, color in enumerate(("red", "blue")):
        v = tmp_path / f"v{i}.mp4"
        make_video(v, color, seconds=4)
        jobs.append(_submit(client, v)["job_id"])
    for j in jobs:
        assert wait_done(client, j)["status"] == "done"
    assert state["max"] == 1


@needs_ffmpeg
def test_same_source_is_never_processed_concurrently(settings, tmp_path, monkeypatch):
    state = _track_videos(monkeypatch)
    video = tmp_path / "v.mp4"
    make_video(video, "red", seconds=4)
    with TestClient(create_app(replace(settings, ingest_workers=4))) as c:
        first = _submit(c, video)
        sid = first["source_id"]
        jobs = [first["job_id"]]
        for _ in range(3):  # 取り込み直しを続けて頼む(同じ取り込み元のジョブが並ぶ)
            r = c.post(f"/api/sources/{sid}/reindex")
            assert r.status_code == 200, r.text
            jobs.append(r.json()["job_id"])
        for j in jobs:
            assert wait_done(c, j)["status"] == "done"
        assert state["max_per_source"] == 1
        # 窓が重複して残らない(4 秒・2 秒窓で 2 件)
        body = {"query": "red", "top_k": 100, "merge": False, "min_score": -1}
        hits = [h for h in c.post("/api/search/text", json=body).json()["results"] if h["source_id"] == sid]
        assert len(hits) == 2


@needs_ffmpeg
def test_stop_with_workers_requeues_running_jobs(settings, tmp_path):
    class Slow(DummyEmbedder):
        def embed_parts(self, parts, high=False):
            time.sleep(0.3)
            return super().embed_parts(parts, high=high)

    s = replace(settings, ingest_workers=2)
    videos = []
    for i in range(2):
        videos.append(tmp_path / f"long{i}.mp4")
        make_video(videos[-1], "red", seconds=20)
    with TestClient(create_app(s, embedder=Slow(768))) as c:
        jobs = [_submit(c, v)["job_id"] for v in videos]
        time.sleep(1.0)
        assert c.get("/api/info").json()["index"]["running_jobs"] == 2
    with TestClient(create_app(s)) as c:  # 次の起動で最後まで取り込む
        for j in jobs:
            done = wait_done(c, j)
            assert done["status"] == "done" and done["total"] == 10
