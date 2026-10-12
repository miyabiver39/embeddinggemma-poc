"""本番運用のための API(ページ送り・取り込み元の変更・検索の過負荷・読み取り専用のトークン)のテスト。"""

from __future__ import annotations

import io
from dataclasses import replace

from fastapi.testclient import TestClient
from PIL import Image
from test_e2e_dummy import wait_done
from test_mcp import Mcp

from mediasearch.main import create_app

FULL = "full-token-0123456789abcdef"
READ = "read-token-0123456789abcdef"


def _png(color) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(buf, "PNG")
    return buf.getvalue()


def _ingest_images(c, n=3, headers=None):
    files = [("files", (f"img{i}.png", _png((60 * i, 20, 200)), "image/png")) for i in range(n)]
    body = c.post("/api/ingest/image", files=files, data={"group_id": "album"}, headers=headers or {}).json()
    for item in body["items"]:
        assert wait_done(c, item["job_id"])["status"] == "done"
    return body["items"]


def test_list_paging_and_filters(client):
    _ingest_images(client, 3)
    page = client.get("/api/sources?limit=2").json()
    assert page["total"] == 3 and len(page["sources"]) == 2 and page["next_offset"] == 2
    last = client.get("/api/sources?limit=2&offset=2").json()
    assert len(last["sources"]) == 1 and last["next_offset"] is None
    assert client.get("/api/sources?kind=video").json()["total"] == 0
    assert client.get("/api/sources?kind=image&status=done&group_id=album").json()["total"] == 3
    assert [s["name"] for s in client.get("/api/sources?name=img1").json()["sources"]] == ["img1.png"]
    assert client.get("/api/sources?name=%25").json()["total"] == 0  # % は文字として扱う(すべてに合わない)
    jobs = client.get("/api/jobs?limit=1&status=done").json()
    assert jobs["total"] == 3 and len(jobs["jobs"]) == 1 and jobs["next_offset"] == 1
    sid = page["sources"][0]["id"]
    assert client.get(f"/api/jobs?source_id={sid}").json()["total"] == 1


def test_patch_source_updates_search_filters(client):
    items = _ingest_images(client, 2)
    sid = items[0]["source_id"]
    r = client.patch(f"/api/sources/{sid}", json={"group_id": "renamed", "location": "gate", "name": "front.png"})
    assert r.status_code == 200 and r.json()["group_id"] == "renamed" and r.json()["name"] == "front.png"
    query = {"query": "x", "min_score": -1, "group_id": "renamed"}
    hits = client.post("/api/search/text", json=query).json()["results"]
    assert [h["source_id"] for h in hits] == [sid]
    assert hits[0]["location"] == "gate" and hits[0]["source_name"] == "front.png"
    cleared = client.patch(f"/api/sources/{sid}", json={"location": ""}).json()
    assert cleared["location"] is None and cleared["group_id"] == "renamed"
    assert client.patch("/api/sources/99999", json={"name": "x"}).status_code == 404


def test_search_overload_returns_503_with_retry_after(settings):
    with TestClient(create_app(replace(settings, max_pending_searches=2))) as c:
        limiter = c.app.state.ctx.search_limiter
        limiter.active = 2  # 処理中の検索が上限に達している状態
        r = c.post("/api/search/text", json={"query": "x"})
        assert r.status_code == 503 and r.headers["retry-after"] == "5" and limiter.rejected == 1
        limiter.active = 0
        assert c.post("/api/search/text", json={"query": "x"}).status_code == 200


def test_read_only_token(settings):
    s = replace(settings, api_token=FULL, api_read_token=READ)
    full, read = {"Authorization": f"Bearer {FULL}"}, {"Authorization": f"Bearer {READ}"}
    with TestClient(create_app(s), headers=full) as c:  # 既定は書き込みのトークン。read は要求ごとに上書きする
        items = _ingest_images(c, 1, headers=full)
        assert c.get("/api/info", headers=read).status_code == 200
        assert c.get("/api/sources", headers=read).json()["total"] == 1
        assert c.post("/api/search/text", json={"query": "x"}, headers=read).status_code == 200
        hit = c.post("/api/search/text", json={"query": "x", "min_score": -1}, headers=read).json()["results"][0]
        assert c.get(hit["thumb_url"], headers=read).status_code == 200  # 参照(GET)はできる
        refused = c.post("/api/ingest/path", json={"path": "/x.mp4"}, headers=read)
        assert refused.status_code == 403 and "読み取り専用" in refused.json()["detail"]
        assert c.delete(f"/api/sources/{items[0]['source_id']}", headers=read).status_code == 403
        assert c.patch(f"/api/sources/{items[0]['source_id']}", json={"name": "x"}, headers=read).status_code == 403
        assert c.get("/api/info", headers={"Authorization": "Bearer wrong"}).status_code == 401
        # MCP: 検索は使え、取り込みのツールは拒否する
        m = Mcp(c, read)
        hello = {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}}
        assert m.rpc("initialize", hello).status_code == 200
        assert m.rpc("tools/call", {"name": "search_text", "arguments": {"query": "x"}}).status_code == 200
        r = m.rpc("tools/call", {"name": "ingest_path", "arguments": {"path": "/x.mp4"}})
        assert r.status_code == 403
        # 書き込みのトークンは、全部できる
        assert c.delete(f"/api/sources/{items[0]['source_id']}", headers=full).status_code == 200
