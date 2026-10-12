"""ベンチマークの結果を、1 つの HTML ファイル(外部の読み込み無し)にまとめます。"""

from __future__ import annotations

import html
import json
from pathlib import Path

from analyze import METRICS

_PAGE = """<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ベンチマーク結果</title>
<style>
:root { color-scheme: light; --surface:#fcfcfb; --card:#ffffff; --line:#e4e3df; --grid:#ecebe7;
  --text-primary:#0b0b0b; --text-secondary:#52514e; --text-muted:#76756f; --series-1:#2a78d6;
  --hit:#ea4335; --warn:#b26b00; --info:#52514e; }
@media (prefers-color-scheme: dark) { :root:where(:not([data-theme="light"])) { color-scheme: dark;
  --surface:#1a1a19; --card:#222220; --line:#3a3a37; --grid:#2e2e2b; --text-primary:#f0efec;
  --text-secondary:#c3c2b7; --text-muted:#9a998f; --series-1:#3987e5; --hit:#ff8a80; --warn:#f0b54a; --info:#c3c2b7; } }
:root[data-theme="dark"] { color-scheme: dark; --surface:#1a1a19; --card:#222220; --line:#3a3a37; --grid:#2e2e2b;
  --text-primary:#f0efec; --text-secondary:#c3c2b7; --text-muted:#9a998f; --series-1:#3987e5;
  --hit:#ff8a80; --warn:#f0b54a; --info:#c3c2b7; }
* { box-sizing: border-box; }
body { margin:0; background:var(--surface); color:var(--text-primary);
  font: 14px/1.6 system-ui, -apple-system, "Segoe UI", "Hiragino Sans", "Noto Sans JP", sans-serif; }
main { max-width: 1180px; margin: 0 auto; padding: 24px 16px 48px; }
h1 { font-size: 20px; margin: 0 0 4px; } h2 { font-size: 16px; margin: 28px 0 10px; }
.sub { color: var(--text-secondary); margin: 0 0 16px; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 14px 16px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 10px; }
.tile .v { font-size: 22px; font-weight: 600; } .tile .k { color: var(--text-secondary); font-size: 12px; }
.diag li { margin: 4px 0; list-style: none; } .diag ul { padding: 0; margin: 0; }
.tag { display:inline-block; min-width: 3.2em; font-size: 12px; font-weight: 600; margin-right: 8px; }
.tag.律速 { color: var(--hit); } .tag.注意 { color: var(--warn); } .tag.情報 { color: var(--info); }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 12px; }
.chart h3 { font-size: 13px; margin: 0 0 6px; font-weight: 600; } .chart .stat { color: var(--text-secondary); font-size: 12px; }
.chart svg { width: 100%; height: 150px; display: block; touch-action: none; }
.axis text { fill: var(--text-muted); font-size: 10px; } .gridline { stroke: var(--grid); stroke-width: 1; }
.mark { stroke: var(--text-muted); stroke-width: 1; stroke-dasharray: 2 3; opacity: .6; }
.series { fill: none; stroke: var(--series-1); stroke-width: 2; stroke-linejoin: round; }
.cross { stroke: var(--text-secondary); stroke-width: 1; } .dot { fill: var(--series-1); stroke: var(--card); stroke-width: 2; }
.tip { position: fixed; pointer-events: none; background: var(--card); border: 1px solid var(--line); border-radius: 6px;
  padding: 6px 8px; font-size: 12px; box-shadow: 0 2px 8px rgba(0,0,0,.12); display: none; z-index: 10; }
.tip b { font-size: 13px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { text-align: left; padding: 5px 8px; border-bottom: 1px solid var(--line); vertical-align: top; }
td.n, th.n { text-align: right; font-variant-numeric: tabular-nums; }
.bars .row { display: grid; grid-template-columns: minmax(120px, 34%) 1fr 64px; gap: 8px; align-items: center; margin: 4px 0; }
.bars .bar { height: 12px; background: var(--series-1); border-radius: 0 4px 4px 0; min-width: 2px; }
.bars .name { color: var(--text-primary); font-size: 13px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.bars .val { text-align: right; color: var(--text-secondary); font-variant-numeric: tabular-nums; font-size: 12px; }
.toolbar { display: flex; gap: 8px; justify-content: flex-end; margin-top: -8px; }
button { font: inherit; font-size: 12px; background: var(--card); color: var(--text-primary); border: 1px solid var(--line);
  border-radius: 6px; padding: 3px 10px; cursor: pointer; }
details summary { cursor: pointer; color: var(--text-secondary); }
pre { white-space: pre-wrap; font-size: 12px; color: var(--text-secondary); }
@media (max-width: 520px) { .grid { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<main>
<div class="toolbar"><button id="theme" type="button">表示: 自動</button></div>
<h1>ベンチマーク結果: __KIND__</h1>
<p class="sub">__SUB__</p>
<div class="tiles" id="tiles"></div>
<h2>診断</h2>
<div class="card diag"><ul id="diag"></ul></div>
<h2>時間の推移</h2>
<p class="sub">縦の点線は出来事(取り込みの開始など)の時刻です。グラフに指を置くか、マウスを重ねると値が出ます。</p>
<div class="grid" id="charts"></div>
<h2>CPU を使った処理(計測の期間の平均コア数)</h2>
<div class="card bars" id="bars"></div>
<h2>集計(平均 / 95 パーセンタイル / 最大)</h2>
<div class="card"><table id="summary"><thead><tr><th>項目</th><th class="n">平均</th><th class="n">95%</th><th class="n">最大</th><th>単位</th></tr></thead><tbody></tbody></table></div>
<h2>出来事</h2>
<div class="card"><table id="marks"><thead><tr><th class="n">経過(秒)</th><th>内容</th></tr></thead><tbody></tbody></table></div>
<h2>詳細</h2>
<div class="card"><details><summary>シナリオの結果とサーバーの設定(JSON)</summary><pre id="raw"></pre></details>
<details><summary>時系列の表(全項目)</summary><div style="overflow-x:auto"><table id="table"></table></div></details></div>
</main>
<div class="tip" id="tip"></div>
<script id="data" type="application/json">__DATA__</script>
<script>
(() => {
  const D = JSON.parse(document.getElementById("data").textContent);
  const rows = D.rows, M = D.metrics, el = (t, a = {}, p) => { const e = document.createElementNS("http://www.w3.org/2000/svg", t);
    for (const k in a) e.setAttribute(k, a[k]); if (p) p.appendChild(e); return e; };
  const txt = (tag, s, cls) => { const e = document.createElement(tag); e.textContent = s; if (cls) e.className = cls; return e; };
  const fmt = (v) => v == null ? "-" : (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2));

  // テーマの切り替え(自動 / 明るい / 暗い)
  const modes = ["自動", "明るい", "暗い"], vals = [null, "light", "dark"]; let mi = 0;
  document.getElementById("theme").onclick = (ev) => { mi = (mi + 1) % 3;
    if (vals[mi]) document.documentElement.dataset.theme = vals[mi]; else delete document.documentElement.dataset.theme;
    ev.target.textContent = "表示: " + modes[mi]; };

  for (const [k, v] of D.tiles) { const d = document.createElement("div"); d.className = "card tile";
    d.append(txt("div", v, "v"), txt("div", k, "k")); document.getElementById("tiles").append(d); }
  for (const d of D.diagnosis) { const li = document.createElement("li"); li.append(txt("span", d.level, "tag " + d.level), document.createTextNode(d.text));
    document.getElementById("diag").append(li); }

  const marks = []; rows.forEach((r) => (r.marks || []).forEach((m) => marks.push([r.t, m])));
  const tip = document.getElementById("tip");
  const tmax = rows.length ? rows[rows.length - 1].t : 1;
  for (const [key, label, unit] of M) {
    const pts = rows.filter((r) => r[key] != null).map((r) => [r.t, r[key]]);
    if (!pts.length) continue;
    const card = document.createElement("div"); card.className = "card chart";
    const s = D.summary[key] || {};
    card.append(txt("h3", label + "(" + unit + ")"), txt("div", "平均 " + fmt(s.avg) + " / 最大 " + fmt(s.max), "stat"));
    const W = 340, H = 150, L = 34, R = 8, T = 8, B = 20;
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": label }, card);
    let ymax = Math.max(...pts.map((p) => p[1]));
    if (unit === "%") ymax = Math.max(100, ymax); else ymax = ymax > 0 ? ymax * 1.1 : 1;
    const x = (t) => L + (t / tmax) * (W - L - R), y = (v) => T + (1 - v / ymax) * (H - T - B);
    const ax = el("g", { class: "axis" }, svg);
    for (const f of [0, 0.5, 1]) { const v = ymax * f;
      el("line", { x1: L, x2: W - R, y1: y(v), y2: y(v), class: "gridline" }, ax);
      el("text", { x: L - 4, y: y(v) + 3, "text-anchor": "end" }, ax).textContent = fmt(v); }
    el("text", { x: L, y: H - 4 }, ax).textContent = "0 秒";
    el("text", { x: W - R, y: H - 4, "text-anchor": "end" }, ax).textContent = fmt(tmax) + " 秒";
    for (const [t] of marks) el("line", { x1: x(t), x2: x(t), y1: T, y2: H - B, class: "mark" }, svg);
    el("path", { d: pts.map((p, i) => (i ? "L" : "M") + x(p[0]).toFixed(1) + " " + y(p[1]).toFixed(1)).join(" "), class: "series" }, svg);
    const cross = el("line", { y1: T, y2: H - B, class: "cross", visibility: "hidden" }, svg);
    const dot = el("circle", { r: 4, class: "dot", visibility: "hidden" }, svg);
    const hit = el("rect", { x: L, y: 0, width: W - L - R, height: H, fill: "transparent" }, svg);
    const move = (ev) => { const b = svg.getBoundingClientRect(), t = ((ev.clientX - b.left) / b.width * W - L) / (W - L - R) * tmax;
      let best = pts[0]; for (const p of pts) if (Math.abs(p[0] - t) < Math.abs(best[0] - t)) best = p;
      cross.setAttribute("x1", x(best[0])); cross.setAttribute("x2", x(best[0])); dot.setAttribute("cx", x(best[0])); dot.setAttribute("cy", y(best[1]));
      cross.setAttribute("visibility", "visible"); dot.setAttribute("visibility", "visible");
      tip.replaceChildren(txt("b", fmt(best[1]) + " " + unit), document.createElement("br"), document.createTextNode(fmt(best[0]) + " 秒"));
      tip.style.display = "block"; tip.style.left = Math.min(ev.clientX + 12, innerWidth - 140) + "px"; tip.style.top = ev.clientY + 12 + "px"; };
    const leave = () => { tip.style.display = "none"; cross.setAttribute("visibility", "hidden"); dot.setAttribute("visibility", "hidden"); };
    hit.addEventListener("pointermove", move); hit.addEventListener("pointerdown", move); hit.addEventListener("pointerleave", leave);
    document.getElementById("charts").append(card);
  }

  const bars = Object.entries(D.summary.cpu_by_thread || {}); const bmax = Math.max(0.01, ...bars.map((b) => b[1]));
  for (const [name, v] of bars) { const r = document.createElement("div"); r.className = "row";
    const bw = document.createElement("div"); const bar = document.createElement("div"); bar.className = "bar"; bar.style.width = (v / bmax * 100) + "%";
    bar.title = name + ": " + v.toFixed(2) + " コア"; bw.append(bar);
    const n = txt("div", name + (D.hints[name] ? "(" + D.hints[name] + ")" : ""), "name"); n.title = n.textContent;
    r.append(n, bw, txt("div", v.toFixed(2), "val")); document.getElementById("bars").append(r); }
  if (!bars.length) document.getElementById("bars").append(txt("div", "データがありません"));

  const sb = document.querySelector("#summary tbody");
  for (const [key, label, unit] of M) { const s = D.summary[key]; if (!s) continue; const tr = document.createElement("tr");
    tr.append(txt("td", label), txt("td", fmt(s.avg), "n"), txt("td", fmt(s.p95), "n"), txt("td", fmt(s.max), "n"), txt("td", unit)); sb.append(tr); }
  const mb = document.querySelector("#marks tbody");
  for (const [t, m] of marks) { const tr = document.createElement("tr"); tr.append(txt("td", fmt(t), "n"), txt("td", m)); mb.append(tr); }
  document.getElementById("raw").textContent = JSON.stringify({ result: D.result, server: D.server, args: D.args }, null, 2);
  const keys = ["t", ...M.map((m) => m[0]).filter((k) => rows.some((r) => r[k] != null))];
  const tb = document.getElementById("table"); const hr = document.createElement("tr"); keys.forEach((k) => hr.append(txt("th", k, "n"))); tb.append(hr);
  for (const r of rows) { const tr = document.createElement("tr"); keys.forEach((k) => tr.append(txt("td", fmt(r[k]), "n"))); tb.append(tr); }
})();
</script>
</body>
</html>
"""


def _tiles(out: dict) -> list[tuple[str, str]]:
    r, kind = out.get("result") or {}, out["kind"]
    tiles: list[tuple[str, str]] = []
    if kind == "ingest" and r:
        tiles += [("取り込みの時間(中央値)", f"{r['median_total_ms'] / 1000:.2f} 秒"),
                  ("実時間比(中央値。1 より大きければ実時間より速い)", f"{r['median_realtime_factor']}"),
                  ("1 窓あたり(中央値)", f"{r['median_per_window_ms']:.0f} ms")]  # fmt: skip
    elif kind == "images" and r:
        tiles += [("画像の取り込み", f"{r['images_per_sec']} 枚/秒"), ("枚数", str(r["images"]))]
    elif kind == "search" and r:
        lat = r["latency_ms"]
        tiles += [("処理量", f"{r['qps']} 件/秒"), ("応答 p50", f"{lat['p50']} ms"), ("応答 p95", f"{lat['p95']} ms"),
                  ("応答 p99", f"{lat['p99']} ms")]  # fmt: skip
    elif kind == "stream" and r:
        tiles += [("窓の作成", f"{r['windows']} / 想定 {r['expected_windows']}"), ("捨てた窓", str(r["dropped_windows"])),
                  ("追いつけた割合", str(r["keep_up_ratio"]))]  # fmt: skip
    s = out["summary"]
    for key, name in (("inference_busy_pct", "推論の稼働率(平均)"), ("cpu_pct", "CPU 使用率(平均)"), ("gpu_util_pct", "GPU 使用率(平均)")):
        if s.get(key):
            tiles.append((name, f"{s[key]['avg']} %"))
    return tiles


def _version(out: dict) -> str:
    v = (out.get("server") or {}).get("version") or {}
    return f"{v.get('app')}({v.get('variant')}、{(v.get('revision') or '')[:7]})" if isinstance(v, dict) else str(v)


def write_report(path: Path, out: dict, rows: list[dict]) -> None:
    from analyze import THREAD_HINTS

    emb = (out.get("server") or {}).get("embedder") or {}
    sub = (f"{out['started_at']} 開始、{out['duration_sec']} 秒。推論: {emb.get('backend')} / {emb.get('accelerator')} / "
           f"{emb.get('dtype')}(まとめる件数 {emb.get('batch_size')})。サーバーの版 {_version(out)}"
           + (f"。名前: {out['label']}" if out.get("label") else ""))  # fmt: skip
    data = {
        "rows": [{k: v for k, v in r.items() if k != "threads_cores"} for r in rows],
        "metrics": METRICS,
        "summary": out["summary"],
        "diagnosis": out["diagnosis"],
        "tiles": _tiles(out),
        "hints": THREAD_HINTS,
        "result": out.get("result"),
        "server": out.get("server"),
        "args": out.get("args"),
    }
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")  # script 要素を閉じさせない
    page = (_PAGE.replace("__KIND__", html.escape(out["kind"])).replace("__SUB__", html.escape(sub))
            .replace("__DATA__", payload))  # fmt: skip
    path.write_text(page, encoding="utf-8")
