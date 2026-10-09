"""MCP サーバー(/mcp)のテスト。JSON-RPC を HTTP で直接送ります(ダミー埋め込み)。"""

from __future__ import annotations

import base64
import io
import json
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from test_e2e_dummy import wait_done

from mediasearch.main import create_app

HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}


def _png(color) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(buf, "PNG")
    return buf.getvalue()


class Mcp:
    def __init__(self, client: TestClient, headers: dict | None = None) -> None:
        self.client = client
        self.headers = {**HEADERS, **(headers or {})}
        self.n = 0

    def rpc(self, method: str, params: dict | None = None):
        self.n += 1
        body = {"jsonrpc": "2.0", "id": self.n, "method": method, "params": params or {}}
        return self.client.post("/mcp", headers=self.headers, json=body)

    def call(self, name: str, **arguments) -> dict:
        r = self.rpc("tools/call", {"name": name, "arguments": arguments})
        assert r.status_code == 200, r.text
        return r.json()["result"]


@pytest.fixture()
def mcp(client):
    m = Mcp(client)
    hello = {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "test", "version": "1"}}
    init = m.rpc("initialize", hello)
    assert init.status_code == 200 and init.json()["result"]["serverInfo"]["name"] == "mediasearch"
    return m


def test_tools_are_listed_with_descriptions(mcp):
    tools = {t["name"]: t for t in mcp.rpc("tools/list").json()["result"]["tools"]}
    expected = {
        "search_text", "search_image", "search_audio", "get_thumbnail", "ingest_path", "ingest_dir",
        "get_job", "list_jobs", "list_sources", "get_status", "list_streams",
    }
    assert expected <= set(tools)
    assert tools["search_text"]["annotations"]["readOnlyHint"] is True
    assert tools["ingest_path"]["annotations"]["readOnlyHint"] is False
    assert "query" in tools["search_text"]["inputSchema"]["required"]
    assert all(t["description"] for t in tools.values())


def test_ingest_search_and_thumbnail(mcp, client, tmp_path):
    photo = tmp_path / "20260301_101500.png"
    photo.write_bytes(_png((30, 30, 220)))
    accepted = mcp.call("ingest_path", path=str(photo), group_id="album")
    data = accepted["structuredContent"]
    assert data["start_ts_from"] == "filename"
    assert wait_done(client, data["job_id"])["status"] == "done"
    assert mcp.call("get_job", job_id=data["job_id"])["structuredContent"]["status"] == "done"

    found = mcp.call("search_text", query="blue", top_k=3)["structuredContent"]
    top = found["results"][0]
    assert top["source_name"] == photo.name and top["kind"] == "image" and top["group_id"] == "album"
    assert top["abs_time"] == "2026-03-01T10:15:00"

    by_image = mcp.call("search_image", image_base64=base64.b64encode(_png((20, 20, 200))).decode())
    assert by_image["structuredContent"]["results"][0]["window_id"] == top["window_id"]

    thumb = mcp.call("get_thumbnail", window_id=top["window_id"])
    content = thumb["content"][0]
    assert content["type"] == "image" and content["mimeType"] == "image/jpeg"
    assert base64.b64decode(content["data"])[:2] == b"\xff\xd8"  # JPEG の先頭

    status = mcp.call("get_status")["structuredContent"]
    assert status["index"]["windows_by_kind"]["image"] == 1
    assert mcp.call("list_sources")["structuredContent"]["sources"][0]["name"] == photo.name


def test_errors_are_reported_as_tool_errors(mcp, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside") / "x.png"
    outside.write_bytes(_png((1, 2, 3)))
    r = mcp.call("ingest_path", path=str(outside))
    assert r["isError"] is True and "INGEST_ROOTS" in r["content"][0]["text"]
    r = mcp.call("search_text", query="x", kind="nope")
    assert r["isError"] is True and "kind" in r["content"][0]["text"]
    r = mcp.call("search_image")
    assert r["isError"] is True
    r = mcp.call("get_job", job_id=999)
    assert r["isError"] is True and "999" in r["content"][0]["text"]


def test_mcp_requires_token_when_configured(settings):
    s = replace(settings, api_token="mcp-token-0123456789abcdef")
    with TestClient(create_app(s)) as c:
        body = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
        assert c.post("/mcp", headers=HEADERS, json=body).status_code == 401
        ok = c.post("/mcp", headers={**HEADERS, "Authorization": "Bearer mcp-token-0123456789abcdef"}, json=body)
        assert ok.status_code == 200 and json.loads(ok.text)["result"]["tools"]
