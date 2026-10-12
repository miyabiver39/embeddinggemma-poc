"""取り込み中の検索(after_window_id と、SSE の /api/search/live)のテスト。"""

from __future__ import annotations

import json
import threading

from conftest import make_video, needs_ffmpeg
from test_e2e_dummy import wait_done


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


@needs_ffmpeg
def test_many_connections_share_matching_and_keep_their_filters(live_server, tmp_path):
    client = live_server
    va, vb = tmp_path / "20260101_090000.mp4", tmp_path / "20260101_100000.mp4"
    make_video(va, "red", seconds=4)
    make_video(vb, "blue", seconds=4)
    urls = {
        "all": "/api/search/live?query=x&min_score=-1&duration_sec=6",
        "group_a": "/api/search/live?query=x&min_score=-1&group_id=a&duration_sec=6",
        "none": "/api/search/live?query=x&min_score=1.5&duration_sec=6",  # スコアが届かない
    }
    events = {k: [] for k in urls}
    readies = {k: threading.Event() for k in urls}
    threads = [
        threading.Thread(target=_read_events, args=(client, url, readies[k], events[k]), daemon=True)
        for k, url in urls.items()
    ]
    for th in threads:
        th.start()
    assert all(r.wait(10) for r in readies.values())
    ja = client.post("/api/ingest/path", json={"path": str(va), "group_id": "a"}).json()
    jb = client.post("/api/ingest/path", json={"path": str(vb), "group_id": "b"}).json()
    assert wait_done(client, ja["job_id"])["status"] == "done" and wait_done(client, jb["job_id"])["status"] == "done"
    for th in threads:
        th.join(15)
    hits = {k: [d for n, d in ev if n == "hit"] for k, ev in events.items()}
    assert len(hits["all"]) == 4 and {h["group_id"] for h in hits["all"]} == {"a", "b"}
    assert len(hits["group_a"]) == 2 and {h["group_id"] for h in hits["group_a"]} == {"a"}
    assert hits["none"] == []
    assert hits["all"][0]["source_name"] and hits["all"][0]["thumb_url"]  # 表示に必要な情報も付いている

    # 再接続: 最後に受け取った ID を after_window_id に渡すと、取りこぼした窓から通知される
    first_id = hits["all"][0]["window_id"]
    again: list = []
    _read_events(client, f"/api/search/live?query=x&min_score=-1&after_window_id={first_id}&duration_sec=1",
                 threading.Event(), again)  # fmt: skip
    caught = [d["window_id"] for n, d in again if n == "hit"]
    assert caught == sorted(h["window_id"] for h in hits["all"] if h["window_id"] > first_id)
