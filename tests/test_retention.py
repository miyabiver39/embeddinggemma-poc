"""保存期間(RETENTION_DAYS / ストリームの retention_hours)による削除のテスト。"""

from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
from conftest import make_video, needs_ffmpeg
from fastapi.testclient import TestClient
from test_e2e_dummy import wait_done

from mediasearch.main import create_app
from mediasearch.pipeline import IngestParams

DAY = 86400


def _vec(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=768).astype(np.float32)
    return v / np.linalg.norm(v)


@needs_ffmpeg
def test_old_windows_are_deleted_and_file_sources_marked_expired(settings, tmp_path):
    s = replace(settings, retention_days=7)
    old = tmp_path / "20200101_090000.mp4"  # 録画の時刻が保存期間より前
    new = tmp_path / "new.mp4"
    make_video(old, "red", seconds=4)
    make_video(new, "blue", seconds=4)
    with TestClient(create_app(s)) as c:
        ctx = c.app.state.ctx
        a = c.post("/api/ingest/path", json={"path": str(old)}).json()
        b = c.post("/api/ingest/path", json={"path": str(new)}).json()  # 日時が無いので受付時刻(今)
        assert wait_done(c, a["job_id"])["status"] == "done" and wait_done(c, b["job_id"])["status"] == "done"
        old_windows = [h["window_id"] for h in ctx.store.search(_vec(0), top_k=100, source_id=a["source_id"])]
        assert old_windows and all((s.data_dir / "thumbs" / f"w{w}.jpg").is_file() for w in old_windows)

        result = ctx.retention.run_once()
        assert result["deleted_windows"] == len(old_windows) and result["expired_sources"] == [a["source_id"]]
        src = c.get(f"/api/sources/{a['source_id']}").json()
        assert src["status"] == "expired"  # 行は残る(監視フォルダが同じファイルを取り込み直さないように)
        assert old.is_file()  # 利用者のファイルは削除しない
        assert not any((s.data_dir / "thumbs" / f"w{w}.jpg").exists() for w in old_windows)
        assert len(ctx.store.vectors) == c.get("/api/info").json()["index"]["windows"]  # ベクトルも削除済み
        hits = c.post("/api/search/text", json={"query": "x", "min_score": -1, "top_k": 100}).json()["results"]
        assert hits and {h["source_id"] for h in hits} == {b["source_id"]}
        assert c.get(f"/api/media/{a['source_id']}").status_code == 404
        assert c.get("/api/info").json()["retention"]["total_deleted_windows"] == len(old_windows)
        # 同じファイルを重複として扱い、自動では取り込み直さない
        assert c.post("/api/ingest/path", json={"path": str(old)}).json()["duplicate"] is True


def test_stream_retention_hours_overrides_global(settings):
    s = replace(settings, retention_days=30)
    with TestClient(create_app(s)) as c:
        store = c.app.state.ctx.store
        now = time.time()
        params = IngestParams(2, 2, 0, False).to_dict()
        sids = {}
        for name, hours in (("short", 1.0), ("forever", 0.0), ("global", None)):
            sid = store.add_source(kind="stream", path=None, name=name, group_id=None, location=None,
                                   start_ts=now - 60 * DAY, params=params, status="running")  # fmt: skip
            store.add_stream(name=name, url="rtsp://cam.invalid/x", group_id=None, location=None, params=params,
                             source_id=sid, enabled=False, retention_hours=hours)  # fmt: skip
            for i, age in enumerate((2 * 3600, 40 * DAY)):  # 2 時間前と 40 日前の窓
                store.add_window(source_id=sid, start_ms=i, end_ms=i + 1, kind="frames", group_id=None,
                                 abs_ts=now - age, vec=_vec(i))  # fmt: skip
            sids[name] = sid
        c.app.state.ctx.retention.run_once(now=now)
        left = {name: len(store.search(_vec(0), top_k=10, min_score=-1, source_id=sid)) for name, sid in sids.items()}
        assert left == {"short": 0, "forever": 2, "global": 1}
        assert store.get_source(sids["short"])["status"] == "running"  # ストリームの取り込み元は期限切れにしない


def test_retention_disabled_by_default(client):
    assert client.app.state.ctx.retention.run_once()["deleted_windows"] == 0
    assert client.get("/api/info").json()["retention"]["days"] is None
