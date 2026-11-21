"""共通ベンチマーク(bench/MATRIX.md)の結果を集計し、表とグラフを作ります。

matrix.py の出力のフォルダ(results.jsonl・machine.json・quality/)を読み、
summary.csv / summary.json、グラフ(SVG。GitHub の README に埋め込める)、report.html を書き出します。
フォルダを 2 つ以上渡すと、機種ごとの集計に加えて、機種を比べるグラフも作ります。
Python の標準ライブラリだけで動きます。

使い方:
    python bench/matrix_report.py bench/results/matrix-rtx3060 --out bench/matrix-results/rtx3060
    python bench/matrix_report.py bench/results/matrix-rtx3060 bench/results/matrix-rx9060xt --out bench/matrix-results/compare
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import shutil
import statistics
from pathlib import Path

TOTAL_SEC = 480
DEFAULT_TOKENS = 280  # 画像 1 枚あたりの上限トークン(モデルの既定)
BASELINE = ("float32", 448, 768, DEFAULT_TOKENS)
DEFAULT = ("bfloat16", 448, 768, DEFAULT_TOKENS)
COLORS = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed", "#0891b2", "#db2777", "#4b5563"]


# ---------------------------------------------------------------------- 読み込みと集計
def load(folder: Path) -> tuple[list[dict], dict, dict]:
    recs = [json.loads(x) for x in (folder / "results.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    mfile = folder / "machine.json"
    machine = json.loads(mfile.read_text(encoding="utf-8")) if mfile.exists() else {}
    vectors = {}
    for f in sorted((folder / "quality").glob("*.json")) if (folder / "quality").exists() else []:
        dtype, side, dims, *rest = f.stem.split("-")  # 上限トークンが既定でなければ末尾に -t<数>
        tokens = int(rest[0][1:]) if rest else DEFAULT_TOKENS
        vectors[(dtype, int(side), int(dims), tokens)] = json.loads(f.read_text(encoding="utf-8"))
    return recs, machine, vectors


def _median(vals: list) -> float | None:
    vals = [v for v in vals if v is not None]
    return round(statistics.median(vals), 4) if vals else None


def prec_of(r: dict) -> tuple:
    # 上限トークンの記録が無い行は、実験 3 を足す前の記録(既定の 280 で測った)
    return (r["dtype"], r["image_max_side"], r["dims"], r.get("image_max_tokens") or DEFAULT_TOKENS)


def _pkey(c: dict) -> tuple:
    return (c["dtype"], c["image_max_side"], c["dims"], c["image_max_tokens"])


def summarize(recs: list[dict], vectors: dict) -> dict:
    """セルごとの中央値と、精度の条件ごとの検索・品質をまとめます。"""
    groups: dict[tuple, list[dict]] = {}
    failures: dict[tuple, list[str]] = {}
    for r in recs:
        if r["experiment"] not in ("concurrency", "precision") or (r.get("rep") or 0) < 1:
            continue
        key = (r["experiment"], *prec_of(r), r["k"], r["w"], r["win"])
        if r.get("ok"):
            groups.setdefault(key, []).append(r)
        else:
            failures.setdefault(key, []).append(str(r.get("error"))[:300])
    cells = []
    for key in sorted(set(groups) | set(failures)):
        rs = groups.get(key, [])
        exp, dtype, side, dims, tokens, k, w, win = key
        cells.append({
            "experiment": exp, "dtype": dtype, "image_max_side": side, "dims": dims, "image_max_tokens": tokens,
            "k": k, "w": w, "win": win,
            "n": len(rs), "failed": len(failures.get(key, [])),
            "completion_sec": _median([r["completion_sec"] for r in rs]),
            "completion_min": min((r["completion_sec"] for r in rs), default=None),
            "completion_max": max((r["completion_sec"] for r in rs), default=None),
            "windows": _median([r["windows"] for r in rs]),
            "sec_per_window": _median([r["sec_per_window"] for r in rs]),
            "realtime_factor": _median([r["realtime_factor"] for r in rs]),
            "infer_util": _median([r.get("infer_util") for r in rs]),
            "gpu_util": _median([r.get("gpu_util") for r in rs]),
            "gpu_mem_max_mb": max((r["gpu_mem_max_mb"] for r in rs if r.get("gpu_mem_max_mb") is not None),
                                  default=None),
            "cpu_util": _median([r.get("cpu_util") for r in rs]),
            "mem_max_mb": max((r["mem_max_mb"] for r in rs if r.get("mem_max_mb") is not None), default=None),
            "mem_total_max_mb": max((r["mem_total_max_mb"] for r in rs if r.get("mem_total_max_mb") is not None),
                                    default=None),
            "error": (failures.get(key) or [None])[0],
        })  # fmt: skip
    # 同時数の効果: 同じ k・窓で、w=1 との差(%)。負なら短くなった
    by = {(c["experiment"], *_pkey(c), c["k"], c["w"], c["win"]): c for c in cells}
    for c in cells:
        base = by.get((c["experiment"], *_pkey(c), c["k"], 1, c["win"]))
        one = by.get((c["experiment"], *_pkey(c), 1, 1, c["win"]))
        c["vs_w1_pct"] = _pct(c["completion_sec"], base and base["completion_sec"])
        c["vs_k1_pct"] = _pct(c["completion_sec"], one and one["completion_sec"])
    precision = {}
    for r in recs:
        if r["experiment"] in ("search", "quality"):
            p = precision.setdefault(prec_of(r), _pdict(prec_of(r)))
            if r["experiment"] == "search" and r.get("ok"):
                p.update({"qps": r["qps"], "search_p50_ms": r["latency_ms"]["p50"],
                          "search_p95_ms": r["latency_ms"]["p95"]})  # fmt: skip
            if r["experiment"] == "quality":
                p.update({"top1_acc": r.get("top1_acc"), "margin": r.get("margin"), "nan": r.get("nan"),
                          "quality_error": r.get("error")})  # fmt: skip
    base_vecs = vectors.get(BASELINE)
    for prec, vecs in vectors.items():
        if base_vecs is None:
            break
        p = precision.setdefault(prec, _pdict(prec))
        sims = [_cos(_trunc(b, prec[2]), v)
                for kind in ("images", "texts") for b, v in zip(base_vecs[kind], vecs[kind], strict=True)]  # fmt: skip
        sims = [s for s in sims if not math.isnan(s)]
        p["cos_mean"] = round(statistics.fmean(sims), 5) if sims else None
        p["cos_min"] = round(min(sims), 5) if sims else None
    for c in cells:
        if c["experiment"] == "precision":
            p = precision.setdefault(_pkey(c), _pdict(_pkey(c)))
            p[f"sec_per_window_w{c['w']}"] = c["sec_per_window"]
            p[f"completion_w{c['w']}"] = c["completion_sec"]
    return {"cells": cells, "precision": sorted(precision.values(), key=_prec_sort)}


def _pdict(prec: tuple) -> dict:
    return dict(zip(("dtype", "image_max_side", "dims", "image_max_tokens"), prec, strict=True))


def _prec_sort(p: dict) -> tuple:
    return (["bfloat16", "float16", "float32"].index(p.get("dtype", "bfloat16")) if p.get("dtype") in
            ("bfloat16", "float16", "float32") else 9, -(p.get("image_max_tokens") or 0),
            -(p.get("image_max_side") or 0), -(p.get("dims") or 0))  # fmt: skip


def _pct(x, base) -> float | None:
    if x is None or not base:
        return None
    return round((x - base) / base * 100, 1)


def _trunc(v: list[float], dims: int) -> list[float]:
    v = v[:dims]
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _cos(a: list[float], b: list[float]) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if not na or not nb:
        return float("nan")
    return sum(x * y for x, y in zip(a, b, strict=True)) / na / nb


# ---------------------------------------------------------------------- グラフ(SVG)
def _esc(s) -> str:
    return html.escape(str(s))


def _nice_max(v: float) -> float:
    if v <= 0:
        return 1.0
    e = 10 ** math.floor(math.log10(v))
    for m in (1, 2, 2.5, 5, 10):
        if v <= m * e:
            return m * e
    return 10 * e


def _frame(title: str, w: int, h: int) -> list[str]:
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
            'font-family="sans-serif" font-size="12">',
            f'<rect width="{w}" height="{h}" fill="#ffffff"/>',
            f'<text x="{w / 2}" y="20" text-anchor="middle" font-size="14" font-weight="bold" fill="#111">'
            f"{_esc(title)}</text>"]  # fmt: skip


def line_chart(title: str, xlabel: str, ylabel: str, xs: list, series: dict[str, list], *, zero_line=False) -> str:
    """x は順序のある分類(等間隔)。series の値の None は、線を切ります。"""
    w, h, left, right, top, bottom = 720, 400, 70, 170, 36, 50
    pw, ph = w - left - right, h - top - bottom
    vals = [v for ys in series.values() for v in ys if v is not None]
    lo = min(min(vals, default=0), 0)
    hi = _nice_max(max(vals, default=1)) if max(vals, default=1) > 0 else 0
    if lo < 0:
        lo = -_nice_max(-lo)
    if hi == lo:
        hi = lo + 1

    def sx(i):
        return left + (pw * i / (len(xs) - 1) if len(xs) > 1 else pw / 2)

    def sy(v):
        return top + ph - (v - lo) / (hi - lo) * ph

    out = _frame(title, w, h)
    for t in range(6):
        v = lo + (hi - lo) * t / 5
        y = sy(v)
        out.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + pw}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        out.append(f'<text x="{left - 6}" y="{y + 4:.1f}" text-anchor="end" fill="#374151">{v:g}</text>')
    if zero_line and lo < 0 < hi:
        out.append(f'<line x1="{left}" y1="{sy(0):.1f}" x2="{left + pw}" y2="{sy(0):.1f}" stroke="#6b7280"/>')
    for i, x in enumerate(xs):
        out.append(f'<text x="{sx(i):.1f}" y="{top + ph + 18}" text-anchor="middle" fill="#374151">{_esc(x)}</text>')
    out.append(f'<text x="{left + pw / 2}" y="{h - 10}" text-anchor="middle" fill="#111">{_esc(xlabel)}</text>')
    out.append(f'<text x="16" y="{top + ph / 2}" text-anchor="middle" fill="#111" '
               f'transform="rotate(-90 16 {top + ph / 2})">{_esc(ylabel)}</text>')  # fmt: skip
    for n, (name, ys) in enumerate(series.items()):
        color = COLORS[n % len(COLORS)]
        seg: list[str] = []
        for i, v in enumerate(ys):
            if v is None:
                if len(seg) > 1:
                    out.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{" ".join(seg)}"/>')
                seg = []
                continue
            seg.append(f"{sx(i):.1f},{sy(v):.1f}")
            out.append(f'<circle cx="{sx(i):.1f}" cy="{sy(v):.1f}" r="3.5" fill="{color}">'
                       f"<title>{_esc(name)} / {_esc(xs[i])}: {v:g}</title></circle>")  # fmt: skip
        if len(seg) > 1:
            out.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{" ".join(seg)}"/>')
        ly = top + 10 + n * 18
        out.append(f'<rect x="{left + pw + 14}" y="{ly - 8}" width="12" height="12" fill="{color}"/>')
        out.append(f'<text x="{left + pw + 32}" y="{ly + 2}" fill="#111">{_esc(name)}</text>')
    out.append("</svg>")
    return "\n".join(out)


def bar_chart(title: str, xlabel: str, labels: list[str], series: dict[str, list]) -> str:
    """横向きの棒グラフ(条件が多いため)。"""
    rows = len(labels)
    nser = max(len(series), 1)
    bh = 10 if nser > 1 else 14
    gap = 8
    # 見出しの幅に合わせて左の余白を広げる(全角は 12px、半角は 7px として見積もる)
    left = max(200, 16 + max((sum(12 if ord(ch) > 0x2000 else 7 for ch in lab) for lab in labels), default=0))
    right, top, bottom = 150, 36, 40
    w = left + 410 + right
    h = top + bottom + rows * (bh * nser + gap)
    pw = w - left - right
    vals = [v for ys in series.values() for v in ys if v is not None]
    hi = _nice_max(max(vals, default=1))
    out = _frame(title, w, h)
    for t in range(6):
        v = hi * t / 5
        x = left + pw * t / 5
        out.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{h - bottom}" stroke="#e5e7eb"/>')
        out.append(f'<text x="{x:.1f}" y="{h - bottom + 16}" text-anchor="middle" fill="#374151">{v:g}</text>')
    out.append(f'<text x="{left + pw / 2}" y="{h - 8}" text-anchor="middle" fill="#111">{_esc(xlabel)}</text>')
    for i, lab in enumerate(labels):
        y0 = top + i * (bh * nser + gap)
        out.append(f'<text x="{left - 6}" y="{y0 + bh * nser / 2 + 4}" text-anchor="end" fill="#111">{_esc(lab)}</text>')
        for n, (name, ys) in enumerate(series.items()):
            v = ys[i]
            color = COLORS[n % len(COLORS)]
            if v is None:
                out.append(f'<text x="{left + 4}" y="{y0 + n * bh + bh - 2}" fill="#9ca3af">なし</text>')
                continue
            out.append(f'<rect x="{left}" y="{y0 + n * bh}" width="{pw * v / hi:.1f}" height="{bh - 1}" '
                       f'fill="{color}"><title>{_esc(lab)} / {_esc(name)}: {v:g}</title></rect>')  # fmt: skip
            out.append(f'<text x="{left + pw * v / hi + 4:.1f}" y="{y0 + n * bh + bh - 2}" font-size="10" '
                       f'fill="#374151">{v:g}</text>')  # fmt: skip
    if len(series) > 1:
        for n, name in enumerate(series):
            ly = top + 10 + n * 18
            out.append(f'<rect x="{left + pw + 40}" y="{ly - 8}" width="12" height="12" fill="{COLORS[n]}"/>')
            out.append(f'<text x="{left + pw + 58}" y="{ly + 2}" fill="#111">{_esc(name)}</text>')
    out.append("</svg>")
    return "\n".join(out)


def scatter(title: str, xlabel: str, ylabel: str, points: list[tuple]) -> str:
    """points: (x, y, 名前, 色の番号)。"""
    w, h, left, right, top, bottom = 720, 420, 70, 30, 36, 50
    pw, ph = w - left - right, h - top - bottom
    pts = [p for p in points if p[0] is not None and p[1] is not None]
    if not pts:
        return line_chart(title, xlabel, ylabel, ["-"], {})
    x0, x1 = 0, _nice_max(max(p[0] for p in pts))
    ylo = min(p[1] for p in pts)
    yhi = max(p[1] for p in pts)
    pad = (yhi - ylo) * 0.1 or 0.01
    ylo, yhi = ylo - pad, yhi + pad

    def sx(v):
        return left + (v - x0) / (x1 - x0) * pw

    def sy(v):
        return top + ph - (v - ylo) / (yhi - ylo) * ph

    out = _frame(title, w, h)
    for t in range(6):
        yv = ylo + (yhi - ylo) * t / 5
        xv = x0 + (x1 - x0) * t / 5
        out.append(f'<line x1="{left}" y1="{sy(yv):.1f}" x2="{left + pw}" y2="{sy(yv):.1f}" stroke="#e5e7eb"/>')
        out.append(f'<text x="{left - 6}" y="{sy(yv) + 4:.1f}" text-anchor="end" fill="#374151">{yv:.4g}</text>')
        out.append(f'<text x="{sx(xv):.1f}" y="{top + ph + 18}" text-anchor="middle" fill="#374151">{xv:g}</text>')
    out.append(f'<text x="{left + pw / 2}" y="{h - 10}" text-anchor="middle" fill="#111">{_esc(xlabel)}</text>')
    out.append(f'<text x="16" y="{top + ph / 2}" text-anchor="middle" fill="#111" '
               f'transform="rotate(-90 16 {top + ph / 2})">{_esc(ylabel)}</text>')  # fmt: skip
    # 重なる点と見出しは 1 つにまとめる(全部書くと読めなくなるため)。点の上の title には全部を残す
    placed: list[list] = []
    for x, y, name, ci in pts:
        out.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="5" fill="{COLORS[ci % len(COLORS)]}" '
                   f'fill-opacity="0.8"><title>{_esc(name)}: {x:g}, {y:g}</title></circle>')  # fmt: skip
    for x, y, name, _ in sorted(pts, key=lambda p: p[0]):  # 左から順に見出しを置く
        # 縦に近く、横は既にある見出しの幅(1 文字 7px と見積もる)に入るなら、重なるとみなす
        near = next((p for p in placed if abs(p[1] - sy(y)) < 12 and -12 < sx(x) - p[0] < 7 * len(p[2]) + 60), None)
        if near:
            near[3] += 1
        else:
            placed.append([sx(x), sy(y), name, 0])
    for px, py, name, more in placed:
        label = name + (f" ほか {more} 件" if more else "")
        out.append(f'<text x="{px + 7:.1f}" y="{py + 3:.1f}" font-size="9" fill="#374151">{_esc(label)}</text>')
    out.append("</svg>")
    return "\n".join(out)


# ---------------------------------------------------------------------- 機種ごとのグラフ
def _cell(cells, exp, prec, k, w, win, field="completion_sec"):
    for c in cells:
        if (c["experiment"], *_pkey(c), c["k"], c["w"], c["win"]) == (exp, *prec, k, w, win):
            return c.get(field)
    return None


def _plabel(p) -> str:
    short = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}
    tokens = p.get("image_max_tokens") or DEFAULT_TOKENS
    extra = "" if tokens == DEFAULT_TOKENS else f"・上限 {tokens} トークン"
    return f"{short.get(p['dtype'], p['dtype'])}・{p['image_max_side']}px・{p['dims']}次元{extra}"


def machine_charts(summary: dict, name: str) -> dict[str, tuple[str, str]]:
    """{ファイル名: (見出し, SVG)} を返します。"""
    cells, charts = summary["cells"], {}
    conc = [c for c in cells if c["experiment"] == "concurrency"]
    wins = sorted({c["win"] for c in conc})
    ks = sorted({c["k"] for c in conc})
    ws = sorted({c["w"] for c in conc})
    for win in wins:
        series = {f"{k} 本に分割": [_cell(cells, "concurrency", DEFAULT, k, w, win) if w <= k else None for w in ws]
                  for k in ks}  # fmt: skip
        charts[f"c1_completion_win{win}.svg"] = (
            f"分割数ごとの完了時間と同時数(窓 {win} 秒)",
            line_chart(f"{name}: 480 秒の取り込みの完了時間(窓 {win} 秒)", "同時数 INGEST_WORKERS", "完了時間(秒)",
                       ws, series))  # fmt: skip
    kmax = max(ks, default=8)
    if conc:
        charts["c2_k8_speedup.svg"] = (
            f"{kmax} 本の取り込み: 1 本ずつとの差",
            line_chart(f"{name}: {kmax} 本を同時に取り込んだときの、1 本ずつとの差", "同時数 INGEST_WORKERS",
                       "1 本ずつとの差(%)。負なら短い",
                       ws, {f"窓 {win} 秒": [_cell(cells, "concurrency", DEFAULT, kmax, w, win, "vs_w1_pct") for w in ws]
                            for win in wins}, zero_line=True))  # fmt: skip
        charts["c3_split_diag.svg"] = (
            "分けて全部同時に取り込んだとき(k = w)と、1 本のまま(k=1)の比較",
            line_chart(f"{name}: 分けて全部同時(k = w)の完了時間", "分割数 = 同時数", "完了時間(秒)",
                       ks, {f"窓 {win} 秒": [_cell(cells, "concurrency", DEFAULT, k, k, win) for k in ks]
                            for win in wins}))  # fmt: skip
        charts["c4_split_w1.svg"] = (
            "分けただけ(同時数 1)の影響",
            line_chart(f"{name}: 分けて 1 本ずつ取り込んだときの完了時間", "分割数(同時数 1)", "完了時間(秒)",
                       ks, {f"窓 {win} 秒": [_cell(cells, "concurrency", DEFAULT, k, 1, win) for k in ks]
                            for win in wins}))  # fmt: skip
        charts["c5_window.svg"] = (
            "窓の大きさと完了時間(1 本・同時数 1)",
            line_chart(f"{name}: 窓の大きさと、480 秒の取り込みの完了時間(1 本・同時数 1)", "窓(秒。1 秒 1 枚)",
                       "完了時間(秒)", wins,
                       {"完了時間": [_cell(cells, "concurrency", DEFAULT, 1, 1, win) for win in wins]}))  # fmt: skip
        charts["c6_infer_util.svg"] = (
            f"推論の稼働率と同時数({kmax} 本)",
            line_chart(f"{name}: 推論の稼働率({kmax} 本の取り込み)", "同時数 INGEST_WORKERS", "推論の稼働率(%)", ws,
                       {f"窓 {win} 秒": [_cell(cells, "concurrency", DEFAULT, kmax, w, win, "infer_util") for w in ws]
                        for win in wins}))  # fmt: skip
        charts["c7_gpu_mem.svg"] = (
            f"GPU メモリ(最大)と同時数({kmax} 本)",
            line_chart(f"{name}: GPU メモリの最大({kmax} 本の取り込み)", "同時数 INGEST_WORKERS", "GPU メモリ(MB)", ws,
                       {f"窓 {win} 秒": [_cell(cells, "concurrency", DEFAULT, kmax, w, win, "gpu_mem_max_mb")
                                         for w in ws] for win in wins}))  # fmt: skip
        charts["c8_host_mem.svg"] = (
            f"サーバーのメモリ(最大)と同時数({kmax} 本)",
            line_chart(f"{name}: サーバーのメモリの最大({kmax} 本の取り込み)", "同時数 INGEST_WORKERS", "メモリ(MB)", ws,
                       {f"窓 {win} 秒": [_cell(cells, "concurrency", DEFAULT, kmax, w, win, "mem_max_mb")
                                         for w in ws] for win in wins}))  # fmt: skip
    prec = summary["precision"]
    if any(p.get("sec_per_window_w1") is not None for p in prec):
        labels = [_plabel(p) for p in prec]
        charts["p1_speed.svg"] = (
            "精度の条件ごとの 1 窓あたりの時間(8 本・窓 2 秒)",
            bar_chart(f"{name}: 精度の条件ごとの 1 窓あたりの時間(8 本・窓 2 秒)", "1 窓あたりの時間(秒)", labels,
                      {"同時数 1": [p.get("sec_per_window_w1") for p in prec],
                       "同時数 4": [p.get("sec_per_window_w4") for p in prec]}))  # fmt: skip
        charts["p2_search.svg"] = (
            "精度の条件ごとの検索の処理量(並行 4)",
            bar_chart(f"{name}: 検索の処理量(並行 4・15 秒)", "件/秒", labels,
                      {"検索": [p.get("qps") for p in prec]}))  # fmt: skip
        dcolor = {"bfloat16": 0, "float16": 1, "float32": 2}
        charts["p3_speed_vs_cos.svg"] = (
            "速さと品質(基準との cos 類似度)",
            scatter(f"{name}: 速さと、基準(fp32・448px・768 次元)との cos 類似度(平均)", "1 窓あたりの時間(秒。同時数 1)",
                    "cos 類似度(平均)",
                    [(p.get("sec_per_window_w1"), p.get("cos_mean"), _plabel(p), dcolor.get(p["dtype"], 3))
                     for p in prec]))  # fmt: skip
        charts["p4_speed_vs_top1.svg"] = (
            "速さと品質(文章から画像を探した 1 位の正解率)",
            scatter(f"{name}: 速さと、文章から画像を探した 1 位の正解率", "1 窓あたりの時間(秒。同時数 1)", "1 位の正解率",
                    [(p.get("sec_per_window_w1"), p.get("top1_acc"), _plabel(p), dcolor.get(p["dtype"], 3))
                     for p in prec]))  # fmt: skip
    return charts


def compare_charts(named: list[tuple[str, dict]]) -> dict[str, tuple[str, str]]:
    charts = {}
    ws = sorted({c["w"] for _, s in named for c in s["cells"] if c["experiment"] == "concurrency"})
    ks = sorted({c["k"] for _, s in named for c in s["cells"] if c["experiment"] == "concurrency"})
    kmax = max(ks, default=8)
    wins = sorted({c["win"] for _, s in named for c in s["cells"] if c["experiment"] == "concurrency"})
    for win in wins:
        charts[f"x1_k8_win{win}.svg"] = (
            f"機種の比較: {kmax} 本の完了時間(窓 {win} 秒)",
            line_chart(f"機種の比較: {kmax} 本の取り込みの完了時間(窓 {win} 秒)", "同時数 INGEST_WORKERS", "完了時間(秒)",
                       ws, {n: [_cell(s["cells"], "concurrency", DEFAULT, kmax, w, win) for w in ws]
                            for n, s in named}))  # fmt: skip
        charts[f"x2_diag_win{win}.svg"] = (
            f"機種の比較: 分けて全部同時(窓 {win} 秒)",
            line_chart(f"機種の比較: 分けて全部同時(k = w)の完了時間(窓 {win} 秒)", "分割数 = 同時数", "完了時間(秒)",
                       ks, {n: [_cell(s["cells"], "concurrency", DEFAULT, k, k, win) for k in ks]
                            for n, s in named}))  # fmt: skip
    labels = None
    series = {}
    for n, s in named:
        prec = s["precision"]
        if labels is None:
            labels = [_plabel(p) for p in prec]
        by = {_plabel(p): p.get("sec_per_window_w1") for p in prec}
        series[n] = [by.get(lab) for lab in labels or []]
    if labels:
        charts["x3_precision_speed.svg"] = (
            "機種の比較: 精度の条件ごとの 1 窓あたりの時間",
            bar_chart("機種の比較: 精度の条件ごとの 1 窓あたりの時間(8 本・窓 2 秒・同時数 1)", "1 窓あたりの時間(秒)", labels,
                      series))  # fmt: skip
    return charts


# ---------------------------------------------------------------------- 書き出し
CELL_COLS = ["experiment", "dtype", "image_max_side", "dims", "image_max_tokens", "k", "w", "win", "n", "failed", "completion_sec",
             "completion_min", "completion_max", "vs_w1_pct", "vs_k1_pct", "windows", "sec_per_window",
             "realtime_factor", "infer_util", "gpu_util", "gpu_mem_max_mb", "cpu_util", "mem_max_mb", "mem_total_max_mb", "error"]  # fmt: skip
PREC_COLS = ["dtype", "image_max_side", "dims", "image_max_tokens", "completion_w1", "sec_per_window_w1", "completion_w4",
             "sec_per_window_w4", "qps", "search_p50_ms", "search_p95_ms", "cos_mean", "cos_min", "top1_acc", "margin",
             "nan", "quality_error"]  # fmt: skip


def write_csv(path: Path, rows: list[dict], cols: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in cols})


def html_table(rows: list[dict], cols: list[str]) -> str:
    head = "".join(f"<th>{_esc(c)}</th>" for c in cols)
    body = "".join("<tr>" + "".join(f"<td>{_esc('' if r.get(c) is None else r.get(c))}</td>" for c in cols) + "</tr>"
                   for r in rows)  # fmt: skip
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def write_html(path: Path, title: str, machines: list[dict], charts: dict, tables: list[tuple[str, str]]) -> None:
    parts = [f"<!doctype html><html lang='ja'><head><meta charset='utf-8'><title>{_esc(title)}</title>",
             "<meta name='viewport' content='width=device-width, initial-scale=1'>",
             "<style>body{font-family:sans-serif;margin:16px;background:#fff;color:#111}"
             "table{border-collapse:collapse;font-size:12px;margin:8px 0 24px}td,th{border:1px solid #d1d5db;"
             "padding:3px 6px;text-align:right}th{background:#f3f4f6}svg{max-width:100%;height:auto}"
             ".wrap{overflow-x:auto}</style></head><body>", f"<h1>{_esc(title)}</h1>"]  # fmt: skip
    for m in machines:
        parts.append("<h2>機器</h2>" + html_table([m], list(m.keys())))
    for fname, (head, svg) in charts.items():
        parts.append(f"<h2>{_esc(head)}</h2><div class='wrap'>{svg}</div><p><small>{_esc(fname)}</small></p>")
    for head, tbl in tables:
        parts.append(f"<h2>{_esc(head)}</h2><div class='wrap'>{tbl}</div>")
    parts.append("</body></html>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="共通ベンチマークの結果を集計し、表とグラフを作ります")
    ap.add_argument("folders", nargs="+", help="matrix.py の --out のフォルダ")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    named = []
    for folder in map(Path, args.folders):
        recs, machine, vectors = load(folder)
        name = machine.get("machine") or folder.name
        summary = summarize(recs, vectors)
        named.append((name, summary, machine, recs))
    if len(named) == 1:
        name, summary, machine, recs = named[0]
        charts = machine_charts(summary, name)
        for fname, (_, svg) in charts.items():
            (out / fname).write_text(svg, encoding="utf-8")
        write_csv(out / "summary.csv", summary["cells"], CELL_COLS)
        write_csv(out / "precision.csv", summary["precision"], PREC_COLS)
        (out / "summary.json").write_text(json.dumps({"machine": machine, **summary}, ensure_ascii=False, indent=2),
                                          encoding="utf-8")  # fmt: skip
        # 1 回ごとの記録(資源の使用状況の時系列は、もともと含めていない)
        shutil.copyfile(Path(args.folders[0]) / "results.jsonl", out / "results.jsonl")
        (out / "machine.json").write_text(json.dumps(machine, ensure_ascii=False, indent=2), encoding="utf-8")
        write_html(out / "report.html", f"共通ベンチマークの結果({name})", [machine], charts,
                   [("セルごとの中央値", html_table(summary["cells"], CELL_COLS)),
                    ("精度の条件ごと", html_table(summary["precision"], PREC_COLS))])  # fmt: skip
    else:
        charts = compare_charts([(n, s) for n, s, _, _ in named])
        for fname, (_, svg) in charts.items():
            (out / fname).write_text(svg, encoding="utf-8")
        rows = [{"machine": n, **c} for n, s, _, _ in named for c in s["cells"]]
        write_csv(out / "summary.csv", rows, ["machine", *CELL_COLS])
        prow = [{"machine": n, **p} for n, s, _, _ in named for p in s["precision"]]
        write_csv(out / "precision.csv", prow, ["machine", *PREC_COLS])
        write_html(out / "report.html", "共通ベンチマークの機種の比較", [m for _, _, m, _ in named], charts,
                   [("セルごとの中央値", html_table(rows, ["machine", *CELL_COLS])),
                    ("精度の条件ごと", html_table(prow, ["machine", *PREC_COLS]))])  # fmt: skip
    print(f"書き出しました: {out}(report.html、summary.csv、グラフ {len(charts)} 枚)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
