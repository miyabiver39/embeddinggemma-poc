"""取り込みの受付(検証・重複確認・フォルダ一括・監視フォルダ)のテスト。ダミー埋め込みを使います。"""

from __future__ import annotations

from datetime import datetime

import pytest
from conftest import make_video, needs_ffmpeg
from fastapi.testclient import TestClient
from test_e2e_dummy import wait_done

from vmsembed.config import Settings
from vmsembed.ingest_files import media_kind_of, ts_from_filename
from vmsembed.main import create_app


@pytest.mark.parametrize(
    "name, expected",
    [
        ("20260101_090000.mp4", datetime(2026, 1, 1, 9, 0, 0)),
        ("cam01-2026-01-01T09-30-15.mkv", datetime(2026, 1, 1, 9, 30, 15)),
        ("rec_20261231235959_ch1.ts", datetime(2026, 12, 31, 23, 59, 59)),
        ("20261301_090000.mp4", None),  # 13 月は日時として成り立たない
        ("clip.mp4", None),
        ("120260101090000.mp4", None),  # 数字の途中から切り出さない
    ],
)
def test_ts_from_filename(name, expected):
    got = ts_from_filename(name)
    assert got == (expected.timestamp() if expected else None)


def test_media_kind_of():
    assert media_kind_of("a.MP4") == "video"
    assert media_kind_of("a.wav") == "audio"
    assert media_kind_of("/etc/shadow") is None


def test_path_rejects_non_media(client, tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("password")
    r = client.post("/api/ingest/path", json={"path": str(secret)})
    assert r.status_code == 422, r.text
    assert "拡張子" in r.json()["detail"]
    assert client.get("/api/sources").json()["sources"] == []


@needs_ffmpeg
def test_path_rejects_broken_media(client, tmp_path):
    fake = tmp_path / "fake.mp4"
    fake.write_text("not a video")
    r = client.post("/api/ingest/path", json={"path": str(fake)})
    assert r.status_code == 422, r.text
    assert client.get("/api/sources").json()["sources"] == []


@needs_ffmpeg
def test_upload_rejects_broken_media_and_removes_file(client, settings):
    r = client.post("/api/ingest/video", files={"file": ("x.mp4", b"not a video", "video/mp4")})
    assert r.status_code == 422, r.text
    media_dir = settings.data_dir / "media"
    assert not media_dir.exists() or list(media_dir.iterdir()) == []


@needs_ffmpeg
def test_duplicate_and_filename_timestamp(client, tmp_path):
    v = tmp_path / "cam01_20260101_090000.mp4"
    make_video(v, "red", seconds=3)
    first = client.post("/api/ingest/path", json={"path": str(v)}).json()
    assert first["duplicate"] is False
    assert first["start_ts"] == "2026-01-01T09:00:00" and first["start_ts_from"] == "filename"
    assert wait_done(client, first["job_id"])["status"] == "done"

    again = client.post("/api/ingest/path", json={"path": str(v)}).json()
    assert again["duplicate"] is True and again["source_id"] == first["source_id"] and again["job_id"] is None

    forced = client.post("/api/ingest/path", json={"path": str(v), "force": True}).json()
    assert forced["duplicate"] is False and forced["source_id"] != first["source_id"]

    explicit = client.post(
        "/api/ingest/path", json={"path": str(v), "force": True, "start_ts": "2026-02-02T10:00:00"}
    ).json()
    assert explicit["start_ts_from"] == "request" and explicit["start_ts"] == "2026-02-02T10:00:00"


@needs_ffmpeg
def test_failed_source_is_not_served(client, tmp_path, settings):
    v = tmp_path / "ok.mp4"
    make_video(v, "blue", seconds=2)
    r = client.post("/api/ingest/path", json={"path": str(v)}).json()
    wait_done(client, r["job_id"])
    assert client.get(f"/api/media/{r['source_id']}").status_code == 200
    # 取り込みに失敗した扱いにすると、配信しない
    from vmsembed.store import Store

    store = Store(settings.data_dir / "vmsembed.db")
    store.update_source(r["source_id"], status="failed")
    store.close()
    assert client.get(f"/api/media/{r['source_id']}").status_code == 404


@needs_ffmpeg
def test_ingest_dir_with_camera_from_dir(client, tmp_path):
    root = tmp_path / "rec"
    (root / "cam01").mkdir(parents=True)
    (root / "cam02").mkdir()
    (root / ".tmp").mkdir()
    make_video(root / "cam01" / "20260101_090000.mp4", "red", seconds=2)
    make_video(root / "cam02" / "20260101_091000.mp4", "green", seconds=2)
    make_video(root / ".tmp" / "20260101_092000.mp4", "blue", seconds=2)  # 隠しフォルダは対象外
    (root / "cam01" / "notes.txt").write_text("memo")  # 拡張子が対象外

    r = client.post("/api/ingest/dir", json={"dir": str(root), "camera_from_dir": True})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["queued"] == 2 and body["duplicates"] == 0 and body["errors"] == []
    assert sorted(i["camera_id"] for i in body["items"]) == ["cam01", "cam02"]
    for item in body["items"]:
        assert wait_done(client, item["job_id"])["status"] == "done"

    again = client.post("/api/ingest/dir", json={"dir": str(root), "camera_from_dir": True}).json()
    assert again["queued"] == 0 and again["duplicates"] == 2

    assert client.post("/api/ingest/dir", json={"dir": str(tmp_path / "none")}).status_code == 422


@needs_ffmpeg
def test_watch_folder(tmp_path, monkeypatch):
    rec = tmp_path / "watch"
    (rec / "cam09").mkdir(parents=True)
    make_video(rec / "cam09" / "20260101_100000.mp4", "red", seconds=2)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("EMBEDDING_BACKEND", "dummy")
    monkeypatch.setenv("WATCH_DIRS", str(rec))
    monkeypatch.setenv("WATCH_SETTLE_SEC", "0")
    monkeypatch.setenv("WATCH_INTERVAL_SEC", "3600")  # 自動の走査はテスト中に走らせない(初回は起動直後)
    with TestClient(create_app(Settings.from_env())) as c:
        info = c.get("/api/info").json()["watch"]
        assert info["enabled"] and info["dirs"] == [str(rec)]
        # 起動直後の走査と、手動の走査のどちらで入っても、登録は 1 件だけ
        c.post("/api/watch/scan")
        sources = c.get("/api/sources").json()["sources"]
        assert len(sources) == 1 and sources[0]["camera_id"] == "cam09"
        assert c.post("/api/watch/scan").json()["queued"] == 0


def test_watch_scan_requires_setting(client):
    assert client.post("/api/watch/scan").status_code == 400


def test_info_has_version(client):
    v = client.get("/api/info").json()["version"]
    assert v["app"] and "revision" in v
