"""運用の監視(リクエスト ID・JSON 形式のログ・GET /metrics)のテスト。"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import replace

from fastapi.testclient import TestClient

from mediasearch.main import create_app
from mediasearch.observability import JsonFormatter, RequestStats, _Writer, request_id_var

TOKEN = "full-token-0123456789abcdef"
READ = "read-token-0123456789abcdef"


def test_request_id_is_echoed_or_generated(client):
    res = client.get("/healthz", headers={"X-Request-ID": "abc-123"})
    assert res.headers["x-request-id"] == "abc-123"
    made = client.get("/healthz", headers={"X-Request-ID": "bad value;" + "x" * 70}).headers["x-request-id"]
    assert re.fullmatch(r"[0-9a-f]{16}", made)  # 不正な値は使わず、新しく作る


def test_json_log_has_request_id():
    fmt = JsonFormatter()
    token = request_id_var.set("rid-1")
    try:
        rec = logging.LogRecord("mediasearch", logging.INFO, __file__, 1, "取り込み %s 件", (3,), None)
        rec.request_id = request_id_var.get()
        out = json.loads(fmt.format(rec))
    finally:
        request_id_var.reset(token)
    assert out["message"] == "取り込み 3 件" and out["request_id"] == "rid-1" and out["level"] == "INFO"


def test_prometheus_groups_families():
    w = _Writer()
    w.metric("a", "gauge", "A", 1, {"x": 1})
    w.metric("b", "gauge", "B", 2)
    w.metric("a", "gauge", "A", 3, {"x": 2})
    w.metric("c", "gauge", "C", None)  # 値が無いものは出さない
    lines = w.text().splitlines()
    assert lines == ["# HELP a A", "# TYPE a gauge", 'a{x="1"} 1', 'a{x="2"} 3', "# HELP b B", "# TYPE b gauge", "b 2"]


def test_metrics_endpoint(settings):
    s = replace(settings, api_token=TOKEN, api_read_token=READ)
    with TestClient(create_app(s), headers={"Authorization": f"Bearer {TOKEN}"}) as c:
        c.get("/api/sources")
        c.get("/api/sources/999")
        assert c.get("/metrics", headers={"Authorization": ""}).status_code == 401
        res = c.get("/metrics", headers={"Authorization": f"Bearer {READ}"})  # 読み取り専用のトークンで読める
        assert res.status_code == 200 and res.headers["content-type"].startswith("text/plain; version=0.0.4")
    text = res.text
    for name in ("mediasearch_process_cpu_seconds_total", "mediasearch_vectors", "mediasearch_ingest_queue",
                 "mediasearch_disk_free_bytes", "mediasearch_inference_busy_seconds_total"):  # fmt: skip
        assert f"\n{name}" in text or f"\n{name}{{" in text, name
    assert 'mediasearch_http_requests_total{route="/api/sources",method="GET",status="200"} 1' in text
    assert 'route="/api/sources/{source_id}",method="GET",status="404"' in text  # 経路のひな形でまとめる
    assert 'route="/metrics",method="GET",status="401"' in text
    assert 'mediasearch_http_request_duration_seconds_bucket{route="/api/sources",method="GET",le="+Inf"} 1' in text
    # 同じ名前の値は 1 か所にまとまっている(HELP は 1 回だけ)
    helps = [line.split()[2] for line in text.splitlines() if line.startswith("# HELP")]
    assert len(helps) == len(set(helps))


def test_request_stats_buckets_are_cumulative():
    st = RequestStats()
    st.observe("/x", "GET", 200, 0.3)
    h = st.hist[("/x", "GET")]
    assert h[0] == 0 and h[3] == 1 and h[-1] == 1  # 0.05 以下は 0 件、0.5 以下は 1 件
