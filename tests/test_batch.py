"""推論のまとめ処理(MicroBatcher・EMBED_BATCH)のテスト。"""

from __future__ import annotations

import threading
import time
from dataclasses import replace

import pytest
from conftest import make_video, needs_ffmpeg
from fastapi.testclient import TestClient
from test_e2e_dummy import wait_done

from mediasearch.config import resolve_embed_batch
from mediasearch.embedders.batch import MicroBatcher
from mediasearch.embedders.dummy import DummyEmbedder
from mediasearch.main import create_app


def test_concurrent_requests_are_merged_and_results_keep_order():
    seen: list[int] = []

    def fn(items):
        seen.append(len(items))
        time.sleep(0.2)  # 推論の実行中に、他のスレッドの要求がたまる
        return [x * 10 for x in items]

    b = MicroBatcher(fn, max_batch=4)
    try:
        results: dict[int, int] = {}
        threads = [threading.Thread(target=lambda i=i: results.update({i: b.submit(i).result()})) for i in range(8)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        assert results == {i: i * 10 for i in range(8)}
        assert max(seen) > 1 and max(seen) <= 4 and sum(seen) == 8
        assert b.run_many([1, 2, 3]) == [10, 20, 30]
    finally:
        b.close()


def test_failed_batch_is_retried_one_by_one():
    def fn(items):
        if "bad" in items:
            raise ValueError("bad input")
        return [x.upper() for x in items]

    b = MicroBatcher(fn, max_batch=8)
    try:
        futures = [b.submit(x) for x in ("a", "bad", "c")]
        assert futures[0].result() == "A" and futures[2].result() == "C"
        with pytest.raises(ValueError):
            futures[1].result()
    finally:
        b.close()


def test_resolve_embed_batch():
    assert resolve_embed_batch("auto", "cpu") == 1  # CPU では速くならないため、まとめない(実測)
    assert resolve_embed_batch("auto", "cuda") == 4
    assert resolve_embed_batch("8", "cpu") == 8
    for bad in ("0", "abc", "100"):
        with pytest.raises(ValueError):
            resolve_embed_batch(bad, "cpu")


@needs_ffmpeg
def test_video_ingest_with_batching_gives_same_windows(settings, tmp_path):
    video = tmp_path / "20260101_090000.mp4"
    make_video(video, "red", seconds=9)
    results = []
    for batch in (1, 4):
        s = replace(settings, data_dir=tmp_path / f"d{batch}")
        with TestClient(create_app(s, embedder=DummyEmbedder(768, batch_size=batch))) as c:
            job = c.post("/api/ingest/path", json={"path": str(video)}).json()
            done = wait_done(c, job["job_id"])
            assert done["status"] == "done" and done["timings"]["embed_batch"] == batch
            hits = c.post("/api/search/text", json={"query": "red", "min_score": -1, "top_k": 50}).json()["results"]
            results.append(sorted((h["start_ms"], round(h["score"], 5)) for h in hits))
    assert results[0] == results[1] and len(results[0]) == 5  # 9 秒・2 秒窓(端数 1 秒の窓を含む)


def test_prepare_runs_in_its_own_thread_and_overlaps_with_inference():
    """prepare を渡すと、推論中に次の回の前処理が進む(前処理と推論の合計より早く終わる)。"""
    calls: list[tuple[str, list]] = []

    def prepare(items):
        calls.append(("prepare", list(items)))
        time.sleep(0.1)
        return [x * 2 for x in items]

    def fn(prepared):
        calls.append(("run", list(prepared)))
        time.sleep(0.1)
        return [x + 1 for x in prepared]

    b = MicroBatcher(fn, max_batch=1, prepare=prepare)
    try:
        t0 = time.perf_counter()
        assert b.run_many([1, 2, 3, 4]) == [3, 5, 7, 9]
        elapsed = time.perf_counter() - t0
        # 重ならなければ 0.8 秒。重なれば約 0.5 秒(前処理 4 回の後に推論 1 回分)
        assert elapsed < 0.7, elapsed
    finally:
        b.close()


def test_failed_prepare_is_retried_one_by_one():
    def prepare(items):
        if "bad" in items:
            raise ValueError("bad input")
        return items

    b = MicroBatcher(lambda prepared: [x.upper() for x in prepared], max_batch=8, prepare=prepare)
    try:
        futures = [b.submit(x) for x in ("a", "bad", "c")]
        assert futures[0].result() == "A" and futures[2].result() == "C"
        with pytest.raises(ValueError):
            futures[1].result()
    finally:
        b.close()


def test_zero_wait_does_not_delay_a_single_request():
    b = MicroBatcher(lambda items: items, max_batch=8, wait_sec=0)
    try:
        t0 = time.perf_counter()
        assert b.submit("x").result() == "x"
        assert time.perf_counter() - t0 < 0.05
    finally:
        b.close()


def test_overlapping_stage_is_not_double_counted():
    """推論を 2 つ並行させても、embed は「どれかが動いていた時間」として数える。"""
    from mediasearch.pipeline import StageTimer

    timer = StageTimer()

    def work():
        with timer.overlapping("embed"):
            time.sleep(0.2)

    threads = [threading.Thread(target=work) for _ in range(2)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert 0.18 < timer.stages["embed"] < 0.35
