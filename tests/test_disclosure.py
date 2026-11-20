"""応答から内部の情報が漏れないことのテスト(CodeQL の指摘への対応)。"""

from __future__ import annotations

from fastapi.testclient import TestClient

from mediasearch.main import create_app


def test_ingest_path_outside_roots_does_not_reveal_existence(client, tmp_path):
    """許可したフォルダの外は、ファイルの有無にかかわらず同じ 403 を返す(有無を調べる手段にさせない)。"""
    outside = tmp_path.parent
    existing = outside / "exists-outside.mp4"
    existing.write_bytes(b"x")
    try:
        r1 = client.post("/api/ingest/path", json={"path": str(existing)})
        r2 = client.post("/api/ingest/path", json={"path": str(outside / "missing-outside.mp4")})
    finally:
        existing.unlink()
    assert r1.status_code == r2.status_code == 403


def test_ingest_path_inside_roots_reports_missing_file(client, tmp_path):
    r = client.post("/api/ingest/path", json={"path": str(tmp_path / "missing.mp4")})
    assert r.status_code == 400 and "見つかりません" in r.json()["detail"]


def test_readyz_hides_exception_details(settings, monkeypatch):
    """/readyz は認証なしで開けるため、例外の文言(パスなど)は返さず、種類だけを返す。"""
    with TestClient(create_app(settings)) as c:
        store = c.app.state.ctx.store

        def broken(_key):
            raise RuntimeError("/secret/path/db.sqlite を開けません")

        monkeypatch.setattr(store, "get_meta", broken)
        r = c.get("/readyz")
    assert r.status_code == 503
    db = r.json()["checks"]["database"]
    assert db["ok"] is False
    assert db["error"].startswith("RuntimeError")
    assert "/secret/path" not in r.text
