"""ベクトル DB(USearch)への保存・検索・復旧のテスト。"""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest
from conftest import make_video, needs_ffmpeg
from fastapi.testclient import TestClient
from test_e2e_dummy import wait_done

from mediasearch import vectors
from mediasearch.main import create_app
from mediasearch.vectors import IndexMismatch, VectorIndex


def _unit(rng, n, d=32):
    v = rng.normal(size=(n, d)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_exact_and_filtered_search(tmp_path):
    rng = np.random.default_rng(0)
    vecs = _unit(rng, 200)
    index = VectorIndex(tmp_path / "v.usearch")
    try:
        index.add(np.arange(1, 201), vecs)
        top = index.search(vecs[9], 3)
        assert top[0][0] == 10 and top[0][1] == pytest.approx(1.0, abs=1e-5)
        # 絞り込み: 偶数のキーだけ
        evens = np.arange(2, 201, 2)
        hits = index.search(vecs[9], 5, candidates=evens)
        assert hits and all(k % 2 == 0 for k, _ in hits)
        scores = [s for _, s in hits]
        assert scores == sorted(scores, reverse=True)
        index.remove([10])
        assert all(k != 10 for k, _ in index.search(vecs[9], 5))
        with pytest.raises(IndexMismatch):
            index.search(np.ones(8, dtype=np.float32), 1)
    finally:
        index.close()
    # 書き出したファイルから読み直せる
    again = VectorIndex(tmp_path / "v.usearch")
    try:
        assert len(again) == 199 and again.dims == 32
    finally:
        again.close()


def test_approximate_search_with_filter(tmp_path, monkeypatch):
    monkeypatch.setattr(vectors, "EXACT_LIMIT", 10)  # 近似検索の経路を通す
    rng = np.random.default_rng(1)
    vecs = _unit(rng, 500)
    index = VectorIndex(tmp_path / "v.usearch")
    try:
        index.add(np.arange(500), vecs)
        assert index.search(vecs[123], 1)[0][0] == 123
        few = np.array([3, 77, 123, 400])  # 対象が少ない絞り込みでも、件数が揃うまで取り直す
        monkeypatch.setattr(vectors, "EXACT_LIMIT", 2)
        hits = index.search(vecs[123], 3, candidates=few)
        assert len(hits) == 3 and hits[0][0] == 123 and {k for k, _ in hits} <= set(few.tolist())
    finally:
        index.close()


@needs_ffmpeg
def test_vectors_persist_and_lost_vectors_are_reindexed(settings, tmp_path):
    video = tmp_path / "20260101_090000.mp4"
    make_video(video, "red", seconds=4)
    with TestClient(create_app(settings)) as c:
        job = c.post("/api/ingest/path", json={"path": str(video)}).json()
        assert wait_done(c, job["job_id"])["status"] == "done"
        windows = c.get("/api/info").json()["index"]["windows"]
        assert windows >= 1
    vec_file = settings.data_dir / "vectors.usearch"
    assert vec_file.is_file()  # 終了時に書き出す
    # ベクトル DB を使わず SQLite にだけ窓が残った状態(書き出し前の異常終了)を作る
    vec_file.unlink()
    with TestClient(create_app(settings)) as c:
        jobs = c.get("/api/jobs").json()["jobs"]
        assert len(jobs) == 2  # 取り込み直しのジョブが作られる
        assert wait_done(c, jobs[0]["id"])["status"] == "done"
        assert c.get("/api/info").json()["index"]["windows"] == windows
        hits = c.post("/api/search/text", json={"query": "red"}).json()["results"]
        assert hits and hits[0]["source_id"] == job["source_id"]


@needs_ffmpeg
def test_new_windows_are_searchable_during_ingest(settings, tmp_path):
    """取り込み中に追加された窓も、すぐに検索の対象になる(ファイルへの書き出しを待たない)。"""
    from mediasearch.store import Store

    store = Store(tmp_path / "db" / "m.db", save_interval_sec=3600)
    try:
        sid = store.add_source(kind="frames", path=None, name="x", group_id="g", location=None, start_ts=0, params={})
        q = np.ones(16, dtype=np.float32) / 4
        assert store.search(q) == []
        store.add_window(source_id=sid, start_ms=0, end_ms=1000, kind="frames", group_id="g", abs_ts=0, vec=q)
        assert [h["window_id"] for h in store.search(q)] == [1]
        w2 = store.add_window(source_id=sid, start_ms=1000, end_ms=2000, kind="frames", group_id="g", abs_ts=1, vec=q)
        assert [h["window_id"] for h in store.search(q, after_id=1)] == [w2]
        assert not (tmp_path / "db" / "vectors.usearch").exists()  # まだ書き出していない
    finally:
        store.close()
    db = sqlite3.connect(tmp_path / "db" / "m.db")
    assert "vec" not in {r[1] for r in db.execute("PRAGMA table_info(windows)")}
    db.close()
