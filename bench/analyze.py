"""計測した値から、使用率の時系列・集計・ボトルネックの判断を作ります。

/api/metrics の値は起動からの積算なので、隣り合う 2 回の差を、その間の経過時間(サーバーの時計)で割って使用率にします。
"""

from __future__ import annotations

import statistics


def _get(d: dict | None, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def _children_total(m: dict) -> float:
    return sum((m.get("children_cpu_sec") or {}).values())


def _gpu_devices(m: dict) -> list[dict]:
    return _get(m, "gpu", "devices", default=[]) or []


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 2) if values else None


def derive(samples: list[dict]) -> list[dict]:
    """計測の間隔ごとの使用率の行を作ります。"""
    rows: list[dict] = []
    for a, b in zip(samples, samples[1:], strict=False):
        pa, pb = a["app"], b["app"]
        dt = pb["monotonic"] - pa["monotonic"]
        if dt <= 0:
            continue
        cores = _get(pb, "cgroup", "cpu_limit") or pb.get("cpus") or 1
        row: dict = {"t": b["t"], "cores": cores, "marks": b.get("marks") or []}
        app_cores = (pb["process"]["cpu_sec"] - pa["process"]["cpu_sec"]) / dt
        child_cores = max(_children_total(pb) - _children_total(pa), 0) / dt
        row["cpu_app_cores"] = round(app_cores, 3)
        row["cpu_ffmpeg_cores"] = round(child_cores, 3)  # 子プロセス(主に ffmpeg)
        cg_a, cg_b = pa.get("cgroup") or {}, pb.get("cgroup") or {}
        if cg_a.get("cpu_sec") is not None and cg_b.get("cpu_sec") is not None:
            row["cpu_pct"] = round((cg_b["cpu_sec"] - cg_a["cpu_sec"]) / dt / cores * 100, 1)
            row["throttled_pct"] = round((cg_b["throttled_sec"] - cg_a["throttled_sec"]) / dt * 100, 1)
        else:
            row["cpu_pct"] = round((app_cores + child_cores) / cores * 100, 1)
        ha, hb = pa.get("host") or {}, pb.get("host") or {}
        if ha.get("total_sec") and hb.get("total_sec", 0) > ha["total_sec"]:
            busy = 1 - (hb["idle_sec"] - ha["idle_sec"]) / (hb["total_sec"] - ha["total_sec"])
            row["host_cpu_pct"] = round(busy * 100, 1)
        threads = {}
        ta, tb = pa.get("threads_cpu_sec") or {}, pb.get("threads_cpu_sec") or {}
        for name, v in tb.items():
            threads[name] = round(max(v - ta.get(name, 0.0), 0) / dt, 3)
        row["threads_cores"] = threads

        # 推論の稼働率は、推論を行う側(分離構成なら compute)の値を使う
        side_a = a.get("compute") or pa
        side_b = b.get("compute") or pb
        ga, gb = _get(side_a, "inference", "gate"), _get(side_b, "inference", "gate")
        if ga and gb:
            dta = side_b["monotonic"] - side_a["monotonic"] if "monotonic" in side_b else dt
            row["inference_busy_pct"] = round(min((gb["busy_sec"] - ga["busy_sec"]) / max(dta, 1e-6), 1.0) * 100, 1)
            calls = gb["calls"]["high"] - ga["calls"]["high"]
            if calls > 0:
                row["search_wait_ms"] = round((gb["wait_sec"]["high"] - ga["wait_sec"]["high"]) / calls * 1000, 1)
            row["inference_waiting"] = gb.get("waiting")
        ba, bb = _get(side_a, "inference", "batches"), _get(side_b, "inference", "batches")
        if ba and bb and bb["calls"] > ba["calls"]:
            row["avg_batch"] = round((bb["items"] - ba["items"]) / (bb["calls"] - ba["calls"]), 2)
        devices = _gpu_devices(side_b)
        if devices:
            row["gpu_util_pct"] = _mean([d["util_pct"] for d in devices if d.get("util_pct") is not None])
            row["gpu_decoder_pct"] = _mean([d["decoder_pct"] for d in devices if d.get("decoder_pct") is not None])
            row["gpu_mem_mb"] = round(sum(d.get("memory_used_mb") or 0 for d in devices), 1)
            row["gpu_mem_total_mb"] = round(sum(d.get("memory_total_mb") or 0 for d in devices), 1)

        row["rss_mb"] = round(pb["process"]["rss_bytes"] / 2**20, 1)
        if cg_b.get("memory_bytes"):
            row["mem_mb"] = round(cg_b["memory_bytes"] / 2**20, 1)
            if cg_b.get("memory_anon_bytes") is not None:
                row["mem_anon_mb"] = round(cg_b["memory_anon_bytes"] / 2**20, 1)
            if cg_b.get("memory_limit_bytes"):
                row["mem_limit_mb"] = round(cg_b["memory_limit_bytes"] / 2**20, 1)
        src_a, src_b = (cg_a, cg_b) if cg_b.get("io_read_bytes") is not None else (pa["process"], pb["process"])
        if src_b.get("io_read_bytes") is not None and src_a.get("io_read_bytes") is not None:
            row["io_read_mbps"] = round((src_b["io_read_bytes"] - src_a["io_read_bytes"]) / dt / 2**20, 2)
            row["io_write_mbps"] = round((src_b["io_write_bytes"] - src_a["io_write_bytes"]) / dt / 2**20, 2)
        row["windows_per_sec"] = round((_get(pb, "index", "windows", default=0) - _get(pa, "index", "windows",
                                        default=0)) / dt, 3)  # fmt: skip
        row["ingest_queue"] = _get(pb, "ingest", "queue")
        streams_a = {s["id"]: s for s in pa.get("streams") or []}
        dropped, lag = 0, 0
        for s in pb.get("streams") or []:
            prev = streams_a.get(s["id"], {})
            dropped += max((s.get("dropped_windows") or 0) - (prev.get("dropped_windows") or 0), 0)
            lag = max(lag, s.get("lag_ms") or 0)
        if pb.get("streams"):
            row["stream_dropped"] = dropped
            row["stream_lag_ms"] = lag
        rows.append(row)
    return rows


METRICS = [  # (キー, 表示名, 単位)
    ("cpu_pct", "CPU 使用率(使える CPU に対する割合)", "%"),
    ("cpu_app_cores", "CPU: アプリ本体", "コア"),
    ("cpu_ffmpeg_cores", "CPU: ffmpeg など子プロセス", "コア"),
    ("throttled_pct", "CPU の上限で止められた時間", "%"),
    ("host_cpu_pct", "マシン全体の CPU 使用率", "%"),
    ("inference_busy_pct", "推論の稼働率", "%"),
    ("search_wait_ms", "検索が推論の順番を待った時間(平均)", "ms"),
    ("avg_batch", "まとめて推論した件数(平均)", "件"),
    ("gpu_util_pct", "GPU 使用率", "%"),
    ("gpu_decoder_pct", "GPU の動画復号回路の使用率", "%"),
    ("gpu_mem_mb", "GPU メモリ", "MB"),
    ("rss_mb", "メモリ: アプリ本体(RSS)", "MB"),
    ("mem_anon_mb", "メモリ: コンテナ全体(キャッシュを除く)", "MB"),
    ("io_read_mbps", "ディスクの読み込み", "MB/秒"),
    ("io_write_mbps", "ディスクの書き込み", "MB/秒"),
    ("windows_per_sec", "窓の作成", "件/秒"),
    ("stream_lag_ms", "ストリームの遅れ(最大)", "ms"),
]


def summarize_rows(rows: list[dict]) -> dict:
    out: dict = {}
    for key, label, unit in METRICS:
        vals = [r[key] for r in rows if r.get(key) is not None]
        if not vals:
            continue
        vals_sorted = sorted(vals)
        out[key] = {
            "label": label,
            "unit": unit,
            "avg": round(statistics.fmean(vals), 2),
            "p95": round(vals_sorted[min(len(vals) - 1, int(len(vals) * 0.95))], 2),
            "max": round(max(vals), 2),
        }
    # スレッドごとの CPU 時間の割合(計測の全期間)
    total: dict[str, float] = {}
    for r in rows:
        dt = 1.0  # 行ごとの値は「コア」なので、平均を出すために足して行数で割る
        for name, v in (r.get("threads_cores") or {}).items():
            total[name] = total.get(name, 0.0) + v * dt
        total["ffmpeg など子プロセス"] = total.get("ffmpeg など子プロセス", 0.0) + r.get("cpu_ffmpeg_cores", 0.0)
    n = max(len(rows), 1)
    out["cpu_by_thread"] = dict(sorted(((k, round(v / n, 3)) for k, v in total.items() if v / n >= 0.005),
                                       key=lambda kv: -kv[1]))  # fmt: skip
    out["cores"] = rows[-1]["cores"] if rows else None
    out["dropped_windows"] = sum(r.get("stream_dropped") or 0 for r in rows)
    out["mem_limit_mb"] = next((r["mem_limit_mb"] for r in reversed(rows) if r.get("mem_limit_mb")), None)
    out["gpu_mem_total_mb"] = next((r["gpu_mem_total_mb"] for r in reversed(rows) if r.get("gpu_mem_total_mb")), None)
    return out


THREAD_HINTS = {
    "native": "推論の演算(PyTorch)など、名前の無いスレッド",
    "decode": "ffmpeg からのフレームの受け取り(取り込み)",
    "ingestor": "ファイルの取り込み(窓の組み立て・保存と、このスレッドから呼ぶ推論の計算)",
    "embed-batch": "推論のまとめ処理",
    "stream": "RTSP の受信と窓の組み立て(このスレッドから呼ぶ推論の計算を含む)",
    "event-loop": "HTTP の受け付け",
    "worker": "API の処理(検索の推論の計算を含む)",
    "MainThread": "イベントループ(HTTP の受け付け)",
}


def diagnose(summary: dict, scenario: dict, info: dict) -> list[dict]:
    """ボトルネックと、資源を使い切れているかを判断します。{"level": 律速 / 注意 / 情報, "text": ...} のリスト。"""

    def avg(key):
        return (summary.get(key) or {}).get("avg")

    out: list[dict] = []
    device = (info.get("embedder") or {}).get("accelerator") or "?"
    busy, cpu = avg("inference_busy_pct"), avg("cpu_pct")
    gpu, dec = avg("gpu_util_pct"), avg("gpu_decoder_pct")
    timings = scenario.get("timings") or {}
    stages = timings.get("stages_ms") or {}
    total_ms = timings.get("total_ms") or 0
    decode_share = (stages.get("decode", 0) / total_ms * 100) if total_ms else None

    if busy is not None and busy >= 85:
        out.append({"level": "律速", "text": f"推論が律速です(稼働率 平均 {busy}%)。推論を速くすると全体が速くなります。"})
        if device == "cpu":
            out.append({"level": "情報", "text": "推論は CPU で行われています。GPU のイメージ(cuda / rocm / intel)を使うと大きく速くなる見込みです。"
                        "CPU のままなら、窓を長くする(WINDOW_SEC)か、フレーム数を減らす(FRAMES_PER_WINDOW)と負荷が下がります。"})  # fmt: skip
        elif gpu is not None and gpu < 60:
            out.append({"level": "注意", "text": f"推論は動き続けていますが、GPU 使用率が平均 {gpu}% です。GPU を使い切れていません。"
                        "EMBED_BATCH を上げる(例 8)と、1 回の推論でまとめて計算できます。CPU 側の前処理が詰まっていないかも確認してください。"})  # fmt: skip
        elif gpu is not None:
            out.append({"level": "情報", "text": f"GPU 使用率は平均 {gpu}% で、GPU をほぼ使い切っています。これ以上は GPU の性能か台数で決まります。"})
    elif busy is not None and busy < 50 and scenario.get("kind") in ("ingest", "stream"):
        if decode_share is not None and decode_share >= 30:
            out.append({"level": "律速", "text": f"動画の復号が律速です(処理時間の {decode_share:.0f}% が復号の待ち、推論の稼働率は {busy}%)。"
                        "FFMPEG_HWACCEL で GPU による復号を使う、FFMPEG_SKIP_FRAMES=keyframes を試す、などで改善できます。"})  # fmt: skip
        else:
            out.append({"level": "注意", "text": f"推論の稼働率が平均 {busy}% と低めです。推論以外(復号・保存・入出力・ネットワーク)で待っている可能性があります。"})
    if cpu is not None and cpu >= 90:
        out.append({"level": "律速", "text": f"CPU を使い切っています(平均 {cpu}%、使える CPU {summary.get('cores')} 個)。"})
    elif cpu is not None and busy is not None and busy >= 85 and device == "cpu" and cpu < 60:
        out.append({"level": "注意", "text": f"CPU で推論していますが、CPU 使用率は平均 {cpu}% です。PyTorch のスレッド数がコア数より少ない可能性があります"
                    "(OMP_NUM_THREADS / torch.set_num_threads)。"})  # fmt: skip
    if avg("throttled_pct") and avg("throttled_pct") > 5:
        out.append({"level": "注意", "text": f"コンテナの CPU の上限(--cpus)で、時間の {avg('throttled_pct')}% 止められています。上限を上げると速くなります。"})
    if dec is not None:
        out.append({"level": "情報", "text": f"GPU の動画復号回路の使用率は平均 {dec}% です(0% なら復号は CPU で行われています)。"})
    mem = summary.get("mem_anon_mb") or summary.get("rss_mb")
    limit = summary.get("mem_limit_mb")
    if mem and limit and mem["max"] >= limit * 0.9:
        out.append({"level": "注意", "text": f"メモリがコンテナの上限に近づいています(最大 {mem['max']:.0f} / {limit:.0f} MB)。"})
    gmem, gtotal = summary.get("gpu_mem_mb"), summary.get("gpu_mem_total_mb")
    if gmem and gtotal and gmem["max"] >= gtotal * 0.9:
        out.append({"level": "注意", "text": f"GPU メモリが上限に近づいています(最大 {gmem['max']:.0f} / {gtotal:.0f} MB)。EMBED_BATCH を下げてください。"})
    wait = summary.get("search_wait_ms") or {}
    lat = (scenario.get("latency_ms") or {}) if scenario.get("kind") == "search" else {}
    if (wait.get("max") or 0) > 1000 or (lat.get("p99") and lat.get("p50") and lat["p99"] > 5 * lat["p50"] and lat["p99"] > 1000):
        detail = f"応答 p50 {lat.get('p50')} ms に対し p99 {lat.get('p99')} ms。" if lat else ""
        out.append({"level": "注意", "text": f"{detail}検索が推論の順番を最大 {wait.get('max', '-')} ms 待っています。"
                    "検索は取り込みより優先されますが、実行中の取り込みの推論(1 窓分)は途中で止められないため、その終わりを待ちます。"
                    "1 窓の推論を短くする(GPU を使う、フレーム数を減らす)と、待ち時間の最大が短くなります。"})
    if summary.get("dropped_windows"):
        out.append({"level": "律速", "text": f"ストリームで、推論が追いつかずに {summary['dropped_windows']} 窓を捨てました。"
                    "窓を長くする、本数を減らす、GPU を使う、のいずれかが必要です。"})  # fmt: skip
    top = list((summary.get("cpu_by_thread") or {}).items())[:3]
    if top:
        parts = [f"{k}({THREAD_HINTS.get(k, '')}){v:.2f} コア" if THREAD_HINTS.get(k) else f"{k} {v:.2f} コア" for k, v in top]
        out.append({"level": "情報", "text": "CPU を多く使っている処理: " + "、".join(parts)})
    if not any(d["level"] == "律速" for d in out):
        out.append({"level": "情報", "text": "はっきりした律速の箇所は見つかりませんでした(負荷が軽いか、計測の時間が短い可能性があります)。"})
    return out
