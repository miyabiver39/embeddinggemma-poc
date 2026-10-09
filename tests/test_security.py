"""HTTP の入口のセキュリティ対策(security.py)と、取り込めるフォルダの制限のテスト。"""

from __future__ import annotations

from dataclasses import replace

import pytest
from conftest import make_video, needs_ffmpeg
from fastapi.testclient import TestClient

from mediasearch.embedders.dummy import DummyEmbedder
from mediasearch.embedders.remote import RemoteEmbedder
from mediasearch.main import create_app

TOKEN = "test-token-0123456789abcdef"


@pytest.fixture()
def secured(settings):
    s = replace(settings, api_token=TOKEN, embedding_token=TOKEN)
    with TestClient(create_app(s)) as c:
        yield c


def test_token_required_except_public_paths(secured):
    assert secured.get("/healthz").status_code == 200
    assert secured.get("/").status_code == 200
    r = secured.get("/api/info")
    assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"
    assert secured.get("/api/info", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert secured.get("/compute/info").status_code == 401
    assert secured.get("/openapi.json").status_code == 401


@pytest.mark.parametrize(
    "headers, cookies",
    [
        ({"Authorization": f"Bearer {TOKEN}"}, {}),
        ({"X-API-Key": TOKEN}, {}),
        ({}, {"mediasearch_token": TOKEN}),
    ],
)
def test_token_accepted_in_header_or_cookie(secured, headers, cookies):
    secured.cookies.clear()
    for k, v in cookies.items():
        secured.cookies.set(k, v)
    assert secured.get("/api/info", headers=headers).status_code == 200


def test_cross_site_requests_are_refused(client):
    body = {"query": "red"}
    evil = client.post("/api/search/text", json=body, headers={"Origin": "http://evil.example"})
    assert evil.status_code == 403
    assert client.post("/api/search/text", json=body, headers={"Sec-Fetch-Site": "cross-site"}).status_code == 403
    # 同じオリジン(WebUI)と、Origin を付けないクライアント(curl など)は通す
    assert client.post("/api/search/text", json=body, headers={"Origin": "http://testserver"}).status_code == 200
    assert client.post("/api/search/text", json=body).status_code == 200
    # 読み取り(GET)は対象外
    assert client.get("/api/info", headers={"Origin": "http://evil.example"}).status_code == 200


def test_allowed_origins(settings):
    s = replace(settings, allowed_origins=("http://app.example",))
    with TestClient(create_app(s)) as c:
        r = c.post("/api/search/text", json={"query": "x"}, headers={"Origin": "http://app.example"})
        assert r.status_code == 200


def test_allowed_hosts(settings):
    s = replace(settings, allowed_hosts=("search.local",))
    with TestClient(create_app(s), base_url="http://search.local:8000") as c:
        assert c.get("/healthz").status_code == 200
    with TestClient(create_app(s), base_url="http://attacker.example") as c:
        assert c.get("/healthz").status_code == 403


def test_body_limit(settings):
    s = replace(settings, max_upload_mb=1)
    big = b"\0" * (1024 * 1024 + 10)
    with TestClient(create_app(s)) as c:
        r = c.post("/api/ingest/video", files={"file": ("big.mp4", big, "video/mp4")})
        assert r.status_code == 413
        # Content-Length を付けない(chunked)送信でも、受け取った量で止める
        r = c.post(
            "/api/search/text",
            content=iter([b'{"query": "', b"x" * (1024 * 1024 + 10), b'"}']),
            headers={"Content-Type": "application/json"},
        )
        assert r.status_code == 413
    assert not (settings.data_dir / "media").exists() or not list((settings.data_dir / "media").iterdir())


def test_security_headers(client):
    r = client.get("/")
    assert "frame-ancestors 'none'" in r.headers["content-security-policy"]
    assert r.headers["x-content-type-options"] == "nosniff"
    assert client.get("/api/info").headers["x-frame-options"] == "DENY"
    assert "content-security-policy" not in client.get("/api/info").headers


def test_compute_rejects_bad_dims(settings, tmp_path):
    s = replace(settings, role="compute", data_dir=tmp_path / "c")
    with TestClient(create_app(s, embedder=DummyEmbedder(dims=768))) as c:
        assert c.post("/compute/embed/texts", json={"texts": ["a"], "dims": 300}).status_code == 400
        assert c.post("/compute/embed/images", data={"dims": "abc"}).status_code == 400


def test_remote_embedder_sends_token(settings, tmp_path):
    s = replace(settings, role="compute", data_dir=tmp_path / "c", api_token=TOKEN)
    with TestClient(create_app(s, embedder=DummyEmbedder(dims=768))) as http:
        ok = RemoteEmbedder("http://testserver", dims=768, client=http, token=TOKEN)
        assert ok.embed_texts(["a"]).shape == (1, 768)
    with TestClient(create_app(s, embedder=DummyEmbedder(dims=768))) as http:
        ng = RemoteEmbedder("http://testserver", dims=768, client=http, token="wrong", retries=1)
        with pytest.raises(ConnectionError, match="認証"):
            ng.info()


@needs_ffmpeg
def test_paths_outside_roots_are_refused(client, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside") / "20260101_090000.mp4"
    make_video(outside, "red", seconds=2)
    r = client.post("/api/ingest/path", json={"path": str(outside)})
    assert r.status_code == 403 and "INGEST_ROOTS" in r.json()["detail"]
    assert client.post("/api/ingest/dir", json={"dir": str(outside.parent)}).status_code == 403
    assert client.post("/api/ingest/path", json={"path": "/etc/passwd"}).status_code == 403


@needs_ffmpeg
def test_symlink_escape_is_refused(client, tmp_path, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside2") / "secret.mp4"
    make_video(outside, "red", seconds=2)
    link = tmp_path / "link.mp4"  # 許可したフォルダの中から、外のファイルを指すリンク
    link.symlink_to(outside)
    assert client.post("/api/ingest/path", json={"path": str(link)}).status_code == 403
