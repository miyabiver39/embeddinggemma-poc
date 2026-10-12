"""mediasearch のベンチマーク(速度と資源の使用率)。

起動中のサーバーに負荷をかけながら、1 秒ごとに GET /api/metrics を読み、
速度・CPU(スレッド・ffmpeg ごと)・メモリ・ディスク・GPU・推論の稼働率を記録して、ボトルネックを判断します。
Python の標準ライブラリだけで動きます。コンテナイメージには含めません(リポジトリの bench/ から実行する)。

使い方の例(詳細は bench/README.md):
    python bench/bench.py ingest --path /recordings/sample.mp4 --repeat 3
    python bench/bench.py search --concurrency 4 --duration 30
    python bench/bench.py search --during-ingest /recordings/sample.mp4     # 取り込み中の検索の速さ
    python bench/bench.py stream --rtsp rtsp://192.168.1.10/stream1 --duration 120
    python bench/bench.py monitor --duration 60                              # 計測だけ(負荷は別の手段でかける)

結果は bench/results/<日時>-<種類>/ に保存します(result.json、samples.jsonl、timeseries.csv、report.html)。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze import METRICS, derive, diagnose, summarize_rows  # noqa: E402
from report import write_report  # noqa: E402
from sampler import ApiError, Client, Sampler  # noqa: E402

DEFAULT_QUERIES = [
    "赤い車が通り過ぎる", "人が歩いている", "a dog running", "夜の駐車場", "サイレンの音",
    "a person opening a door", "雨が降っている", "white truck", "会議室で話している", "bicycle",
]  # fmt: skip


def _pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    return round(s[min(len(s) - 1, int(len(s) * p))], 1)


# ---------------------------------------------------------------------- シナリオ
def run_ingest(c: Client, s: Sampler, args) -> dict:
    """同じファイルを repeat 回取り込み(2 回目以降は取り込み直し)、ジョブの処理時間の内訳を集めます。"""
    options: dict = {"force": True}
    for key in ("preset", "window_sec", "frames_per_window", "overlap_sec"):
        if getattr(args, key, None) is not None:
            options[key] = getattr(args, key)
    if args.audio:
        options["include_audio"] = True
    runs, source_id = [], None
    for n in range(1, args.repeat + 1):
        s.mark(f"取り込み {n} 回目の開始")
        if source_id is None:
            if args.path:
                accepted = c.post("/api/ingest/path", {"path": args.path, **options})
            else:
                fields = {k: str(v).lower() if isinstance(v, bool) else str(v) for k, v in options.items() if k != "force"}
                kind = "audio" if Path(args.upload).suffix.lower() in (".wav", ".mp3", ".m4a", ".flac", ".ogg") else "video"
                accepted = c.post_file(f"/api/ingest/{kind}", Path(args.upload), fields)
            source_id, job_id = accepted["source_id"], accepted["job_id"]
        else:
            job_id = c.post(f"/api/sources/{source_id}/reindex")["job_id"]
        job = c.wait_job(job_id)
        s.mark(f"取り込み {n} 回目の終了")
        if job["status"] != "done":
            raise ApiError(f"取り込みに失敗しました: {job.get('error')}")
        t = job.get("timings") or {}
        runs.append(t)
        print(f"  [{n}/{args.repeat}] {t.get('total_ms', 0) / 1000:.2f} 秒(窓 {t.get('windows')}、"
              f"1 窓 {t.get('per_window_ms')} ms、実時間比 {t.get('realtime_factor')})")  # fmt: skip
    totals = [r.get("total_ms", 0) for r in runs]
    med = sorted(runs, key=lambda r: r.get("total_ms", 0))[len(runs) // 2]
    return {
        "source_id": source_id,
        "runs": runs,
        "median_total_ms": statistics.median(totals),
        "median_realtime_factor": statistics.median([r.get("realtime_factor") or 0 for r in runs]),
        "median_per_window_ms": statistics.median([r.get("per_window_ms") or 0 for r in runs]),
        "timings": med,
    }


def run_images(c: Client, s: Sampler, args) -> dict:
    """フォルダの画像を取り込み直し、1 秒あたりの枚数を測ります。"""
    s.mark("画像の取り込みの開始")
    t0 = time.monotonic()
    res = c.post("/api/ingest/dir", {"dir": args.dir, "kind": "image", "force": True})
    jobs = [item["job_id"] for item in res.get("items", [])]
    done = [c.wait_job(j) for j in jobs]
    elapsed = time.monotonic() - t0
    s.mark("画像の取り込みの終了")
    ok = [j for j in done if j["status"] == "done"]
    print(f"  {len(ok)} 枚 / {elapsed:.2f} 秒({len(ok) / elapsed:.2f} 枚/秒)")
    stages: dict[str, float] = {}
    for j in ok:
        for k, v in ((j.get("timings") or {}).get("stages_ms") or {}).items():
            stages[k] = stages.get(k, 0.0) + v / max(((j.get("timings") or {}).get("batch") or 1), 1)
    return {"images": len(ok), "failed": len(done) - len(ok), "elapsed_sec": round(elapsed, 3),
            "images_per_sec": round(len(ok) / elapsed, 3) if elapsed else None,
            "timings": {"stages_ms": stages, "total_ms": elapsed * 1000}}  # fmt: skip


def run_search(c: Client, s: Sampler, args) -> dict:
    """concurrency 本の並行で、duration 秒のあいだ文章の検索を繰り返し、応答時間を測ります。"""
    queries = args.query or DEFAULT_QUERIES
    lat: list[float] = []
    embed: list[float] = []
    took: list[float] = []
    errors: list[str] = []
    lock = threading.Lock()
    ingest_job = None
    if args.during_ingest:
        s.mark("取り込みの開始(検索と同時)")
        ingest_job = c.post("/api/ingest/path", {"path": args.during_ingest, "force": True})["job_id"]
        time.sleep(1.0)  # 取り込みの推論が始まってから検索する
    s.mark(f"検索の開始(並行 {args.concurrency})")
    deadline = time.monotonic() + args.duration

    def worker(i: int) -> None:
        n = i
        while time.monotonic() < deadline:
            q = queries[n % len(queries)]
            n += args.concurrency
            t = time.perf_counter()
            try:
                res = c.post("/api/search/text", {"query": q, "top_k": args.top_k})
            except ApiError as exc:
                with lock:
                    errors.append(str(exc))
                continue
            with lock:
                lat.append((time.perf_counter() - t) * 1000)
                embed.append(res.get("embed_ms", 0))
                took.append(res.get("took_ms", 0))

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(args.concurrency)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    s.mark("検索の終了")
    ingest = None
    if ingest_job is not None:
        job = c.wait_job(ingest_job)
        ingest = job.get("timings")
        s.mark("取り込みの終了")
    out = {
        "requests": len(lat), "errors": len(errors), "qps": round(len(lat) / args.duration, 2),
        "latency_ms": {"p50": _pct(lat, 0.5), "p95": _pct(lat, 0.95), "p99": _pct(lat, 0.99), "max": _pct(lat, 1.0)},
        "embed_ms": {"p50": _pct(embed, 0.5), "p95": _pct(embed, 0.95)},
        "db_ms": {"p50": _pct(took, 0.5), "p95": _pct(took, 0.95)},
        "during_ingest": bool(ingest_job),
        "timings": ingest or {},
    }  # fmt: skip
    if errors:
        out["first_error"] = errors[0]
    lm = out["latency_ms"]
    print(f"  {out['requests']} 件({out['qps']} 件/秒)応答 p50 {lm['p50']} ms / p95 {lm['p95']} ms / p99 {lm['p99']} ms"
          f"(うちベクトル化 p50 {out['embed_ms']['p50']} ms、DB p50 {out['db_ms']['p50']} ms)")  # fmt: skip
    return out


def run_stream(c: Client, s: Sampler, args) -> dict:
    """RTSP を duration 秒受信し、窓の作成の速さ・捨てた窓・遅れを測ります。"""
    body: dict = {"url": args.rtsp, "name": "bench", "group_id": "bench"}
    for key in ("preset", "window_sec", "frames_per_window", "overlap_sec"):
        if getattr(args, key, None) is not None:
            body[key] = getattr(args, key)
    if args.audio:
        body["include_audio"] = True
    st = c.post("/api/streams", body)
    s.mark("ストリームの受信の開始")
    sid = st["id"]
    try:
        time.sleep(args.duration)
        state = c.get(f"/api/streams/{sid}")["state"]
    finally:
        s.mark("ストリームの受信の終了")
        c.delete(f"/api/streams/{sid}?delete_data={'false' if args.keep else 'true'}")
    params = st["params"]
    stride = max(params["window_sec"] - params["overlap_sec"], 1)
    connected = state.get("connected_at")
    seconds = time.time() - connected if connected else 0
    expected = seconds / stride if seconds else 0
    made, dropped = state.get("windows") or 0, state.get("dropped_windows") or 0
    out = {"state": state, "params": params, "received_sec": round(seconds, 1),
           "windows": made, "dropped_windows": dropped, "expected_windows": round(expected, 1),
           "keep_up_ratio": round(made / expected, 3) if expected else None}  # fmt: skip
    print(f"  受信 {out['received_sec']} 秒: 窓 {made}(想定 {out['expected_windows']})、捨てた窓 {dropped}、"
          f"追いつけた割合 {out['keep_up_ratio']}、復号 {state.get('decoder')}")  # fmt: skip
    if state.get("error"):
        print(f"  ストリームのエラー: {state['error']}")
    return out


def run_monitor(c: Client, s: Sampler, args) -> dict:
    s.mark("計測の開始")
    time.sleep(args.duration)
    s.mark("計測の終了")
    return {}


SCENARIOS = {"ingest": run_ingest, "images": run_images, "search": run_search, "stream": run_stream,
             "monitor": run_monitor}  # fmt: skip


# ---------------------------------------------------------------------- 実行と保存
def _server_info(c: Client) -> dict:
    info = c.get("/api/info")
    return {k: info.get(k) for k in ("version", "revision", "embedder", "defaults", "index")}


def run(args) -> Path:
    c = Client(args.url, args.token)
    compute = Client(args.compute_url, args.compute_token or args.token) if args.compute_url else None
    try:
        info = _server_info(c)
        c.get("/api/metrics")
    except ApiError as exc:
        raise SystemExit(f"{exc}\n(/api/metrics が無い場合は、サーバーが古い版です)") from exc
    emb = info.get("embedder") or {}
    ver = info.get("version") or {}
    ver = ver.get("app") if isinstance(ver, dict) else ver
    print(f"サーバー: {args.url}(版 {ver}、推論 {emb.get('backend')} / {emb.get('accelerator')} / "
          f"{emb.get('dtype')}、まとめる件数 {emb.get('batch_size')})")  # fmt: skip
    print(f"シナリオ: {args.kind}")
    sampler = Sampler(c, compute, args.interval)
    sampler.start()
    started = time.time()
    error = None
    try:
        result = SCENARIOS[args.kind](c, sampler, args)
    except (ApiError, KeyboardInterrupt) as exc:
        error, result = str(exc) or "中断しました", {}
    finally:
        sampler.stop()
    rows = derive(sampler.samples)
    summary = summarize_rows(rows)
    scenario = {"kind": args.kind, **result}
    out = {
        "kind": args.kind,
        "label": args.label,
        "args": {k: v for k, v in vars(args).items() if k not in ("token", "compute_token", "func")},
        "server": info,
        "started_at": datetime.fromtimestamp(started).isoformat(timespec="seconds"),
        "duration_sec": round(time.time() - started, 1),
        "result": result,
        "summary": summary,
        "diagnosis": diagnose(summary, scenario, info),
        "error": error,
        "sampler_errors": sampler.errors[:20],
    }
    name = datetime.fromtimestamp(started).strftime("%Y%m%d-%H%M%S") + f"-{args.kind}" + (f"-{args.label}" if args.label else "")
    dest = Path(args.out) / name
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "result.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    with (dest / "samples.jsonl").open("w", encoding="utf-8") as f:
        for sample in sampler.samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    keys = ["t"] + [k for k, _, _ in METRICS if any(r.get(k) is not None for r in rows)]
    with (dest / "timeseries.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(keys + ["marks"])
        for r in rows:
            w.writerow([r.get(k, "") for k in keys] + [" / ".join(r.get("marks") or [])])
    write_report(dest / "report.html", out, rows)
    print_summary(out)
    print(f"\n結果: {dest}/report.html(ほかに result.json、timeseries.csv、samples.jsonl)")
    if error:
        print(f"エラー: {error}", file=sys.stderr)
    return dest


def print_summary(out: dict) -> None:
    summary = out["summary"]
    print("\n資源の使用状況(平均 / 95 パーセンタイル / 最大):")
    for key, _, _ in METRICS:
        m = summary.get(key)
        if m:
            pad = " " * max(44 - sum(2 if ord(ch) > 0x2E80 else 1 for ch in m["label"]), 0)
            print(f"  {m['label']}{pad} {m['avg']:>9} / {m['p95']:>9} / {m['max']:>9} {m['unit']}")
    if summary.get("cpu_by_thread"):
        print("CPU を使った処理(平均のコア数):")
        for name, v in list(summary["cpu_by_thread"].items())[:8]:
            print(f"  {name:<28} {v:.2f}")
    print("診断:")
    for d in out["diagnosis"]:
        print(f"  [{d['level']}] {d['text']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="mediasearch のベンチマーク(速度と資源の使用率)")
    ap.add_argument("--url", default=os.environ.get("MEDIASEARCH_URL", "http://localhost:8000"))
    ap.add_argument("--token", default=os.environ.get("MEDIASEARCH_TOKEN"), help="API_TOKEN(環境変数 MEDIASEARCH_TOKEN でも可)")
    ap.add_argument("--compute-url", help="分離構成のときの compute の URL(推論側の資源も計測する)")
    ap.add_argument("--compute-token")
    ap.add_argument("--interval", type=float, default=1.0, help="計測の間隔(秒)")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results"))
    ap.add_argument("--label", help="結果のフォルダ名に付ける名前(例 batch8)")
    sub = ap.add_subparsers(dest="kind", required=True)

    def window_opts(p):
        p.add_argument("--preset", choices=("object", "action", "speech"))
        p.add_argument("--window-sec", dest="window_sec", type=int)
        p.add_argument("--frames-per-window", dest="frames_per_window", type=int)
        p.add_argument("--overlap-sec", dest="overlap_sec", type=int)
        p.add_argument("--audio", action="store_true", help="音声も含める(include_audio)")

    p = sub.add_parser("ingest", help="動画・音声の取り込みの速さ")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--path", help="サーバー内のファイル(INGEST_ROOTS の下)")
    src.add_argument("--upload", help="手元のファイルをアップロードする")
    p.add_argument("--repeat", type=int, default=1)
    window_opts(p)
    p = sub.add_parser("images", help="画像の取り込みの速さ")
    p.add_argument("--dir", required=True, help="サーバー内の画像のフォルダ")
    p = sub.add_parser("search", help="検索の応答時間と処理量")
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--duration", type=float, default=30)
    p.add_argument("--top-k", dest="top_k", type=int, default=10)
    p.add_argument("--query", action="append", help="検索する文章(複数可。省略時は組み込みの 10 件)")
    p.add_argument("--during-ingest", dest="during_ingest", help="このサーバー内のファイルを取り込みながら検索する")
    p = sub.add_parser("stream", help="RTSP の受信と窓の作成の速さ")
    p.add_argument("--rtsp", required=True)
    p.add_argument("--duration", type=float, default=120)
    p.add_argument("--keep", action="store_true", help="作った窓を残す(既定は終了時に削除)")
    window_opts(p)
    p = sub.add_parser("monitor", help="負荷をかけずに計測だけ行う(取り込みや検索は別の手段で行う)")
    p.add_argument("--duration", type=float, default=60)
    args = ap.parse_args(argv)
    if getattr(args, "repeat", 1) < 1:
        ap.error("--repeat は 1 以上にしてください")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
