"""以前の版で作った DB を、そのまま使い続けられることのテスト。"""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest
from conftest import make_video, needs_ffmpeg
from fastapi.testclient import TestClient
from test_e2e_dummy import wait_done

from mediasearch.embedders.dummy import DummyEmbedder
from mediasearch.main import create_app
from mediasearch.store import SCHEMA_VERSION, SchemaError, Store

# 0.1.0 の初期の版の形式(グループ ID の列が別の名前だった)
OLD_SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE sources (
  id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, path TEXT, name TEXT NOT NULL,
  camera_id TEXT, location TEXT, start_ts REAL NOT NULL, duration_ms INTEGER, status TEXT NOT NULL,
  params TEXT NOT NULL, error TEXT, created_at REAL NOT NULL
);
CREATE TABLE windows (
  id INTEGER PRIMARY KEY AUTOINCREMENT, source_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
  start_ms INTEGER NOT NULL, end_ms INTEGER NOT NULL, kind TEXT NOT NULL, camera_id TEXT,
  abs_ts REAL NOT NULL, vec BLOB NOT NULL
);
CREATE TABLE jobs (
  id INTEGER PRIMARY KEY AUTOINCREMENT, source_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
  status TEXT NOT NULL, progress INTEGER NOT NULL DEFAULT 0, total INTEGER NOT NULL DEFAULT 0,
  error TEXT, created_at REAL NOT NULL, started_at REAL, finished_at REAL
);
"""


def _old_db(path, with_row: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.executescript(OLD_SCHEMA)
    if with_row:
        vec = DummyEmbedder(768).embed_texts(["green"])[0].astype(np.float32).tobytes()
        db.execute("INSERT INTO meta VALUES('model_id', 'dummy'), ('dims', '768')")
        db.execute(
            "INSERT INTO sources(kind, path, name, camera_id, location, start_ts, status, params, created_at)"
            " VALUES('frames', NULL, 'old.mp4', 'old-group', NULL, 0, 'done', '{}', 0)"
        )
        db.execute("INSERT INTO windows(source_id, start_ms, end_ms, kind, camera_id, abs_ts, vec)"
                   " VALUES(1, 0, 2000, 'frames', 'old-group', 0, ?)", (vec,))
    db.commit()
    db.close()


@needs_ffmpeg
def test_old_database_is_migrated_and_keeps_data(settings, tmp_path):
    _old_db(settings.data_dir / "mediasearch.db")
    video = tmp_path / "20260101_090000.mp4"
    make_video(video, "red", seconds=2)
    with TestClient(create_app(settings)) as c:
        # 以前のデータが、新しい列名で読める
        hit = c.post("/api/search/text", json={"query": "green", "top_k": 1}).json()["results"][0]
        assert hit["source_name"] == "old.mp4" and hit["group_id"] == "old-group"
        assert c.get("/api/sources/1").json()["group_id"] == "old-group"
        # 新しい取り込みも動く(以前はここで 500 になっていた)
        r = c.post("/api/ingest/path", json={"path": str(video), "group_id": "new-group"})
        assert r.status_code == 200, r.text
        job = wait_done(c, r.json()["job_id"])
        assert job["status"] == "done" and job["timings"]["windows"] >= 1  # 後の版で足した列も使える
        found = c.post("/api/search/text", json={"query": "x", "group_id": "new-group"}).json()["results"]
        assert found and {h["group_id"] for h in found} == {"new-group"}
    s = Store(settings.data_dir / "mediasearch.db")
    assert s.get_meta("schema_version") == str(SCHEMA_VERSION)
    s.close()


def test_unknown_database_is_refused_with_clear_message(tmp_path):
    path = tmp_path / "x.db"
    db = sqlite3.connect(path)
    db.executescript("CREATE TABLE sources (id INTEGER PRIMARY KEY, name TEXT);")
    db.close()
    with pytest.raises(SchemaError, match="必要な列"):
        Store(path)
