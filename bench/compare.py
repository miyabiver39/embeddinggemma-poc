"""2 つ以上のベンチマークの結果を並べて比べます(設定を変えた前後の比較用)。

使い方:
    python bench/compare.py bench/results/20261012-100000-ingest-before bench/results/20261012-101500-ingest-after
"""

from __future__ import annotations

import json
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze import METRICS  # noqa: E402

RESULT_KEYS = [  # (シナリオ, 取り出し方, 表示名, 小さいほど良いか)
    ("ingest", lambda r: r.get("median_total_ms"), "取り込みの時間(中央値, ms)", True),
    ("ingest", lambda r: r.get("median_realtime_factor"), "実時間比(中央値)", False),
    ("ingest", lambda r: r.get("median_per_window_ms"), "1 窓あたり(中央値, ms)", True),
    ("images", lambda r: r.get("images_per_sec"), "画像の取り込み(枚/秒)", False),
    ("search", lambda r: r.get("qps"), "検索の処理量(件/秒)", False),
    ("search", lambda r: (r.get("latency_ms") or {}).get("p50"), "検索の応答 p50(ms)", True),
    ("search", lambda r: (r.get("latency_ms") or {}).get("p95"), "検索の応答 p95(ms)", True),
    ("stream", lambda r: r.get("keep_up_ratio"), "ストリームの追いつけた割合", False),
    ("stream", lambda r: r.get("dropped_windows"), "ストリームの捨てた窓", True),
]


def _w(text: str) -> int:
    """表示の幅(全角の文字は 2)。日本語の項目名で列がずれないようにする。"""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


def ljust(text: str, width: int) -> str:
    return text + " " * max(width - _w(text), 0)


def rjust(text: str, width: int) -> str:
    return " " * max(width - _w(text), 0) + text


def load(path: str) -> dict:
    p = Path(path)
    return json.loads((p / "result.json" if p.is_dir() else p).read_text(encoding="utf-8"))


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    runs = [load(a) for a in argv]
    names = [r.get("label") or Path(a).name for r, a in zip(runs, argv, strict=True)]
    width = max(12, *(_w(n) for n in names))
    label_w = 58
    print(ljust("項目", label_w) + "".join(rjust(n, width + 2) for n in names) + "   変化(最初 → 最後)")
    rows = []
    for kind, get, label, lower_better in RESULT_KEYS:
        vals = [get(r.get("result") or {}) if r["kind"] == kind else None for r in runs]
        if any(v is not None for v in vals):
            rows.append((label, vals, lower_better))
    for key, label, unit in METRICS:
        vals = [((r.get("summary") or {}).get(key) or {}).get("avg") for r in runs]
        if any(v is not None for v in vals):
            rows.append((f"{label}(平均, {unit})", vals, None))
    for label, vals, lower_better in rows:
        cells = "".join(rjust("-" if v is None else str(v), width + 2) for v in vals)
        change = ""
        a, b = vals[0], vals[-1]
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and a:
            ratio = b / a
            change = f"   x{ratio:.2f}"
            if lower_better is not None:
                better = ratio < 1 if lower_better else ratio > 1
                change += "(改善)" if better and abs(ratio - 1) > 0.02 else ("(悪化)" if abs(ratio - 1) > 0.02 else "")
        print(ljust(label, label_w) + cells + change)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
