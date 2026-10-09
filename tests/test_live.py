"""取り込み中の検索(after_window_id と、SSE の /api/search/live)のテスト。"""

from __future__ import annotations

import json
import threading
import time

import httpx
import pytest
import uvicorn
from conftest import make_video, needs_ffmpeg
from test_e2e_dummy import wait_done

from mediasearch.main import create_app


@pytest.fixture()
def live_server(settings):
    """実際の HTTP サーバー(TestClient は応答を最後まで読んでから返すため、SSE の確認には使えない)。"""
    server = uvicorn.Server(uvicorn.Config(create_app(settings), host="127.0.0.1", port=0, log_level="warning"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.1)
    port = server.servers[0].sockets[0].getsockname()[1]
    with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=30) as client:
        yield client
    server.should_exit = True
    th.join(10)


def _read_events(client, url: str, ready: threading.Event, out: list) -> None:
    with client.stream("GET", url) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        name = None
        for line in r.iter_lines():
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                out.append((name, json.loads(line[6:])))
                if name == "ready":
                    ready.set()
                if name == "end":
                    return


@needs_ffmpeg
def test_live_search_notifies_new_windows(live_server, tmp_path):
    client = live_server
    video = tmp_path / "20260101_090000.mp4"
    make_video(video, "red", seconds=6)
    events: list = []
    ready = threading.Event()
    reader = threading.Thread(
        target=_read_events,
        args=(client, "/api/search/live?query=red&min_score=-1&duration_sec=6", ready, events),
        daemon=True,
    )
    reader.start()
    assert ready.wait(10)
    job = client.post("/api/ingest/path", json={"path": str(video)}).json()
    assert wait_done(client, job["job_id"])["status"] == "done"
    reader.join(15)
    hits = [d for n, d in events if n == "hit"]
    assert events[0][0] == "ready" and events[-1][0] == "end"
    assert len(hits) == 3 and {h["source_id"] for h in hits} == {job["source_id"]}  # 6 秒・2 秒窓
    assert len({h["window_id"] for h in hits}) == 3  # 同じ窓を重複して通知しない
    assert hits[0]["source_name"] == video.name and hits[0]["thumb_url"]


@needs_ffmpeg
def test_after_window_id_returns_only_new_windows(client, tmp_path):
    first = client.post("/api/search/text", json={"query": "x"}).json()
    assert first["last_window_id"] == 0
    video = tmp_path / "20260101_100000.mp4"
    make_video(video, "blue", seconds=4)
    job = client.post("/api/ingest/path", json={"path": str(video)}).json()
    assert wait_done(client, job["job_id"])["status"] == "done"
    body = {"query": "x", "min_score": -1}  # ダミーの埋め込みはスコアが負になることがある
    res = client.post("/api/search/text", json={**body, "after_window_id": first["last_window_id"]}).json()
    assert len(res["results"]) == 2 and res["last_window_id"] == max(h["window_id"] for h in res["results"])
    again = client.post("/api/search/text", json={**body, "after_window_id": res["last_window_id"]}).json()
    assert again["results"] == []
