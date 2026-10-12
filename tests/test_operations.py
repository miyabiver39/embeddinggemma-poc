"""本番運用のための機能(空き容量・準備完了の確認・再試行・取り消し・ジョブの履歴)のテスト。"""

from __future__ import annotations

import time
from dataclasses import replace

from conftest import make_video, needs_ffmpeg
from fastapi.testclient import TestClient
from test_e2e_dummy import wait_done

from mediasearch import pipeline
from mediasearch.embedders.dummy import DummyEmbedder
from mediasearch.main import create_app


def test_readyz_reports_checks_without_token(settings):
    s = replace(settings, api_token="t" * 32)
    with TestClient(create_app(s)) as c:
        r = c.get("/readyz")
        assert r.status_code == 200 and r.json()["status"] == "ready"
        assert set(r.json()["checks"]) == {"database", "vectors", "embedder", "disk"}
        assert c.get("/healthz").status_code == 200


@needs_ffmpeg
def test_low_disk_stops_new_ingest_but_not_search(settings, tmp_path):
    s = replace(settings, min_free_disk_mb=10**9)  # 空き容量が足りない状態を作る
    video = tmp_path / "20260101_090000.mp4"
    make_video(video, "red", seconds=2)
    with TestClient(create_app(s)) as c:
        r = c.post("/api/ingest/path", json={"path": str(video)})
        assert r.status_code == 507 and "空き容量" in r.json()["detail"]
        up = c.post("/api/ingest/video", files={"file": ("a.mp4", video.read_bytes(), "video/mp4")})
        assert up.status_code == 507
        assert not (s.data_dir / "media").exists() or not list((s.data_dir / "media").iterdir())  # 書き始めない
        assert c.post("/api/search/text", json={"query": "x"}).status_code == 200  # 検索は続けられる
        ready = c.get("/readyz")
        assert ready.status_code == 503 and ready.json()["checks"]["disk"]["ok"] is False
        assert c.get("/api/info").json()["disk"]["ok"] is False


class _Flaky(DummyEmbedder):
    """最初の数回は、推論側に接続できない(分離構成で compute が再起動中など)。"""

    def __init__(self, failures: int) -> None:
        super().__init__(768)
        self.failures = failures

    def embed_parts(self, parts, high=False):
        if self.failures > 0:
            self.failures -= 1
            raise ConnectionError("compute に接続できません")
        return super().embed_parts(parts, high=high)


@needs_ffmpeg
def test_transient_failure_is_retried(settings, tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "RETRY_BASE_SEC", 0.2)
    video = tmp_path / "20260101_090000.mp4"
    make_video(video, "red", seconds=2)
    with TestClient(create_app(replace(settings, job_retries=3), embedder=_Flaky(2))) as c:
        job = c.post("/api/ingest/path", json={"path": str(video)}).json()
        deadline = time.time() + 20
        while time.time() < deadline:
            j = c.get(f"/api/jobs/{job['job_id']}").json()
            if j["status"] == "done":
                break
            time.sleep(0.1)
        assert j["status"] == "done" and j["attempts"] == 2
    # 再試行の回数を超えたら失敗にする
    other = tmp_path / "20260101_100000.mp4"
    make_video(other, "blue", seconds=2)
    with TestClient(create_app(replace(settings, job_retries=1), embedder=_Flaky(5))) as c:
        job = c.post("/api/ingest/path", json={"path": str(other)}).json()
        deadline = time.time() + 20
        while time.time() < deadline:
            j = c.get(f"/api/jobs/{job['job_id']}").json()
            if j["status"] == "failed":
                break
            time.sleep(0.1)
        assert j["status"] == "failed" and "ConnectionError" in j["error"]


class _Slow(DummyEmbedder):
    def embed_parts(self, parts, high=False):
        time.sleep(0.3)
        return super().embed_parts(parts, high=high)


@needs_ffmpeg
def test_cancel_queued_and_running_jobs(settings, tmp_path):
    a, b = tmp_path / "20260101_090000.mp4", tmp_path / "20260101_100000.mp4"
    make_video(a, "red", seconds=20)
    make_video(b, "blue", seconds=4)
    with TestClient(create_app(settings, embedder=_Slow(768))) as c:
        ja = c.post("/api/ingest/path", json={"path": str(a)}).json()
        jb = c.post("/api/ingest/path", json={"path": str(b)}).json()
        time.sleep(1.0)  # a の取り込みが始まり、b は待っている
        assert c.post(f"/api/jobs/{jb['job_id']}/cancel").json()["status"] == "cancelled"
        assert c.post(f"/api/jobs/{ja['job_id']}/cancel").json()["status"] == "running"  # 窓の区切りで止まる
        done = wait_done_or_cancelled(c, ja["job_id"])
        assert done["status"] == "cancelled" and 0 < done["progress"] < 10
        assert c.get(f"/api/sources/{ja['source_id']}").json()["status"] == "cancelled"
        assert c.get(f"/api/sources/{jb['source_id']}").json()["status"] == "cancelled"
        assert c.post("/api/jobs/999999/cancel").status_code == 404
        # 取り消した取り込み元は、取り込み直せる
        again = c.post(f"/api/sources/{jb['source_id']}/reindex").json()
        assert wait_done(c, again["job_id"])["status"] == "done"


def wait_done_or_cancelled(c, job_id, timeout=20):
    end = time.time() + timeout
    while time.time() < end:
        job = c.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "failed", "cancelled"):
            return job
        time.sleep(0.1)
    raise AssertionError("ジョブが終わりません")


def test_old_job_history_is_pruned_but_latest_per_source_kept(settings):
    with TestClient(create_app(settings)) as c:
        ctx = c.app.state.ctx
        store = ctx.store
        sid = store.add_source(kind="frames", path=None, name="x", group_id=None, location=None, start_ts=0,
                               params={}, status="done")  # fmt: skip
        old = [store.create_job(sid) for _ in range(3)]
        for j in old:
            store.update_job(j, status="done", finished_at=time.time() - 90 * 86400)
        ctx.retention.run_once()
        left = [j["id"] for j in store.list_jobs(100)]
        assert left == [old[-1]]  # 取り込み元ごとの最新のジョブは残す
        assert ctx.retention.status["total_pruned_jobs"] == 2
