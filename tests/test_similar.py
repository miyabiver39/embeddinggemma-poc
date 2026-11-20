"""似た場面の検索(POST /api/search/similar)と、結果の重複の除去(dedupe)のテスト。"""

from __future__ import annotations

from conftest import make_video, needs_ffmpeg
from test_e2e_dummy import wait_done


def _ingest(client, path, group):
    r = client.post("/api/ingest/path", json={"path": str(path), "group_id": group})
    assert r.status_code == 200, r.text
    assert wait_done(client, r.json()["job_id"])["status"] == "done"


@needs_ffmpeg
def test_similar_finds_same_scene_and_excludes_query(client, tmp_path):
    red1, red2, blue = tmp_path / "red1.mp4", tmp_path / "red2.mp4", tmp_path / "blue.mp4"
    make_video(red1, "red")
    make_video(red2, "red")
    make_video(blue, "blue")
    _ingest(client, red1, "cam1")
    _ingest(client, red2, "cam2")
    _ingest(client, blue, "cam3")
    hit = client.post("/api/search/text", json={"query": "red", "group_id": "cam1"}).json()["results"][0]

    res = client.post("/api/search/similar", json={"window_id": hit["window_id"], "top_k": 50}).json()
    ids = [h["window_id"] for h in res["results"]]
    assert hit["window_id"] not in ids  # 元の窓は含めない
    assert res["results"][0]["group_id"] in ("cam1", "cam2")  # 赤い映像が先に並ぶ
    assert res["query_window"]["id"] == hit["window_id"]
    assert all(h["kind"] in ("frames", "image") for h in res["results"])  # 同じ空間の種類だけ

    other = client.post(
        "/api/search/similar", json={"window_id": hit["window_id"], "exclude_same_source": True, "top_k": 50}
    ).json()
    assert other["results"]
    assert all(h["source_id"] != hit["source_id"] for h in other["results"])
    assert other["results"][0]["group_id"] == "cam2"  # 別のカメラの同じ場面


def test_similar_unknown_window_is_404(client):
    assert client.post("/api/search/similar", json={"window_id": 999999}).status_code == 404


@needs_ffmpeg
def test_dedupe_collapses_near_identical_windows(client, tmp_path):
    red = tmp_path / "red.mp4"
    make_video(red, "red", seconds=12)
    _ingest(client, red, "cam1")
    plain = client.post("/api/search/text", json={"query": "red", "top_k": 20, "merge": False}).json()["results"]
    assert len(plain) > 1
    assert all("duplicates" not in h or h["duplicates"] is None for h in plain)  # 指定しなければ数えない

    deduped = client.post(
        "/api/search/text", json={"query": "red", "top_k": 20, "merge": False, "dedupe": 0.5}
    ).json()["results"]
    assert len(deduped) < len(plain)
    assert sum(h["duplicates"] for h in deduped) >= len(plain) - len(deduped)
    assert deduped[0]["window_id"] == plain[0]["window_id"]  # いちばんスコアの高い窓を残す


def test_dedupe_rejects_out_of_range(client):
    assert client.post("/api/search/text", json={"query": "x", "dedupe": 1.5}).status_code == 422
    assert client.post("/api/search/text", json={"query": "x", "dedupe": 0}).status_code == 422
