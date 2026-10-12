"""資源の使用状況の API(/api/metrics)と、ベンチマーク(bench/)のテスト。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from conftest import make_video, needs_ffmpeg
from test_e2e_dummy import wait_done

from mediasearch import metrics

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import analyze  # noqa: E402

import bench  # noqa: E402


@needs_ffmpeg
def test_metrics_endpoint_reports_resources_and_inference(client, tmp_path):
    before = client.get("/api/metrics").json()
    video = tmp_path / "20260101_090000.mp4"
    make_video(video, "red", seconds=4)
    job = client.post("/api/ingest/path", json={"path": str(video)}).json()
    assert wait_done(client, job["job_id"])["status"] == "done"
    after = client.get("/api/metrics").json()
    assert after["cpus"] >= 1 and after["process"]["rss_bytes"] > 0
    assert after["process"]["cpu_sec"] >= before["process"]["cpu_sec"]
    assert "ingestor" in after["threads_cpu_sec"]  # Python のスレッド名でまとめる
    gate = after["inference"]["gate"]
    assert gate["calls"]["low"] - before["inference"]["gate"]["calls"]["low"] == 2  # 4 秒・2 秒窓
    assert gate["busy_sec"] >= before["inference"]["gate"]["busy_sec"]
    assert after["index"]["windows"] == 2 and after["ingest"]["queue"] == 0
    assert client.get("/compute/metrics").json()["inference"]["gate"]["calls"]["low"] >= 2


def test_cgroup_v2_and_v1_are_read(tmp_path, monkeypatch):
    v2 = tmp_path / "v2"
    v2.mkdir()
    (v2 / "cgroup.controllers").write_text("cpu memory io\n")
    (v2 / "cpu.stat").write_text("usage_usec 2500000\nthrottled_usec 500000\n")
    (v2 / "cpu.max").write_text("200000 100000\n")
    (v2 / "memory.current").write_text("1048576\n")
    (v2 / "memory.max").write_text("max\n")
    (v2 / "memory.stat").write_text("anon 524288\n")
    (v2 / "io.stat").write_text("8:0 rbytes=100 wbytes=200 rios=1 wios=2\n")
    monkeypatch.setattr(metrics, "_CGROUP", v2)
    cg = metrics._cgroup()
    assert cg["version"] == 2 and cg["cpu_sec"] == 2.5 and cg["cpu_limit"] == 2.0 and cg["throttled_sec"] == 0.5
    assert cg["memory_limit_bytes"] is None and cg["memory_anon_bytes"] == 524288 and cg["io_write_bytes"] == 200

    v1 = tmp_path / "v1"
    for d, files in {
        "cpuacct": {"cpuacct.usage": "3000000000"},
        "cpu": {"cpu.cfs_quota_us": "300000", "cpu.cfs_period_us": "100000", "cpu.stat": "throttled_time 1000000000"},
        "memory": {"memory.usage_in_bytes": "2048", "memory.limit_in_bytes": "9223372036854771712",
                   "memory.stat": "total_rss 1024"},
        "blkio": {"blkio.throttle.io_service_bytes": "8:0 Read 10\n8:0 Write 20\nTotal 30\n"},
    }.items():  # fmt: skip
        (v1 / d).mkdir(parents=True)
        for name, text in files.items():
            (v1 / d / name).write_text(text + "\n")
    monkeypatch.setattr(metrics, "_CGROUP", v1)
    cg = metrics._cgroup()
    assert cg["version"] == 1 and cg["cpu_sec"] == 3.0 and cg["cpu_limit"] == 3.0 and cg["throttled_sec"] == 1.0
    assert cg["memory_limit_bytes"] is None and cg["memory_anon_bytes"] == 1024 and cg["io_read_bytes"] == 10


def _sample(t, cpu, busy, cg_cpu=None):
    app = {
        "monotonic": t, "cpus": 4, "process": {"cpu_sec": cpu, "rss_bytes": 2**20},
        "threads_cpu_sec": {"native": cpu}, "children_cpu_sec": {"ffmpeg": cpu / 2},
        "inference": {"gate": {"busy_sec": busy, "wait_sec": {"high": 0, "low": 0}, "calls": {"high": 0, "low": 1}}},
        "index": {"windows": int(t)}, "ingest": {"queue": 0},
    }  # fmt: skip
    if cg_cpu is not None:
        app["cgroup"] = {"cpu_sec": cg_cpu, "throttled_sec": 0, "cpu_limit": 2.0}
    return {"t": t, "app": app}


def test_derive_rates_and_diagnose_inference_bound():
    samples = [_sample(t, cpu=t * 2, busy=t * 0.95, cg_cpu=t * 1.9) for t in range(0, 6)]
    rows = analyze.derive(samples)
    assert rows[0]["cpu_app_cores"] == 2.0 and rows[0]["cpu_ffmpeg_cores"] == 1.0
    assert rows[0]["cpu_pct"] == 95.0  # cgroup の上限(2 コア)に対する割合
    assert rows[0]["inference_busy_pct"] == 95.0 and rows[0]["windows_per_sec"] == 1.0
    summary = analyze.summarize_rows(rows)
    texts = [d["text"] for d in analyze.diagnose(summary, {"kind": "ingest"}, {"embedder": {"accelerator": "cpu"}})]
    assert any("推論が律速" in t for t in texts) and any("CPU を使い切って" in t for t in texts)


@needs_ffmpeg
def test_bench_ingest_and_search_write_reports(live_server, tmp_path):
    video = tmp_path / "20260101_090000.mp4"
    make_video(video, "red", seconds=4)
    url = str(live_server.base_url)
    out = tmp_path / "results"
    bench.main(["--url", url, "--out", str(out), "--interval", "0.3", "--label", "t1",
                "ingest", "--path", str(video), "--repeat", "2"])  # fmt: skip
    bench.main(["--url", url, "--out", str(out), "--interval", "0.3", "search", "--duration", "1.5",
                "--concurrency", "2"])  # fmt: skip
    dirs = sorted(out.iterdir())
    assert len(dirs) == 2 and dirs[0].name.endswith("-ingest-t1")
    for d in dirs:
        assert {"result.json", "samples.jsonl", "timeseries.csv", "report.html"} <= {p.name for p in d.iterdir()}
        res = json.loads((d / "result.json").read_text(encoding="utf-8"))
        assert res["diagnosis"] and res["summary"].get("cpu_app_cores")
        assert "<script" in (d / "report.html").read_text(encoding="utf-8")
    ingest = json.loads((dirs[0] / "result.json").read_text(encoding="utf-8"))["result"]
    assert len(ingest["runs"]) == 2 and ingest["median_total_ms"] > 0
    search = json.loads((dirs[1] / "result.json").read_text(encoding="utf-8"))["result"]
    assert search["requests"] > 0 and search["latency_ms"]["p50"] > 0
