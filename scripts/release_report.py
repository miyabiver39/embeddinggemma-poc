"""脆弱性検査(grype)の結果を集計し、リリースノート用の Markdown を作ります。

GitHub Actions(.github/workflows/security.yml)から呼び出します。手元でも動きます。

使い方:
    python scripts/release_report.py --title "v0.2.0" reports/ > notes.md

reports/ の下には、バリアントごとに次のファイルがある前提です(security.yml が作ります)。
    <variant>/vulnerabilities.json   grype の JSON 出力
    <variant>/image.txt              検査したイメージの参照(例: ghcr.io/...:0.2.0-cpu@sha256:...)

外部ライブラリを使わず、Python の標準ライブラリだけで動くようにしています
(ランナーに追加のインストールをしなくて済むようにするため)。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# grype の深刻度。表示はこの順に並べる
SEVERITIES = ["Critical", "High", "Medium", "Low", "Negligible", "Unknown"]
# 一覧に個別に載せる深刻度(件数が多くなりすぎないよう、上位 2 段階だけ)
LISTED = ("Critical", "High")
# バリアントの表示順(slim が先頭、GPU 系が後ろ)
VARIANT_ORDER = ["slim", "cpu", "cuda", "rocm", "intel"]


def load_matches(path: Path) -> tuple[list[dict], dict]:
    """grype の JSON を読み、検出結果(matches)と実行情報(descriptor)を返します。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("matches") or [], data.get("descriptor") or {}


def summarize(matches: list[dict]) -> dict:
    """深刻度ごとの件数と、修正版の有無を数えます。

    同じ脆弱性 ID・同じパッケージ・同じ版の組み合わせは 1 件として数えます
    (grype は検出の根拠ごとに同じ組み合わせを複数回出すことがあるため)。
    """
    seen: set[tuple[str, str, str]] = set()
    counts: Counter[str] = Counter()
    fixable: Counter[str] = Counter()
    listed: list[dict] = []
    for m in matches:
        vuln = m.get("vulnerability") or {}
        art = m.get("artifact") or {}
        key = (vuln.get("id", ""), art.get("name", ""), art.get("version", ""))
        if key in seen:
            continue
        seen.add(key)
        severity = vuln.get("severity") or "Unknown"
        if severity not in SEVERITIES:
            severity = "Unknown"
        counts[severity] += 1
        fix = vuln.get("fix") or {}
        fixed_in = [v for v in fix.get("versions") or [] if v]
        if fix.get("state") == "fixed" and fixed_in:
            fixable[severity] += 1
        if severity in LISTED:
            listed.append(
                {
                    "severity": severity,
                    "id": key[0],
                    "package": key[1],
                    "version": key[2],
                    "type": art.get("type", ""),
                    "fixed_in": ", ".join(fixed_in) or "なし",
                }
            )
    listed.sort(key=lambda r: (SEVERITIES.index(r["severity"]), r["package"], r["id"]))
    # ffmpeg のように 1 つのソースから多数のパッケージ(libavcodec など)に分かれるものは、
    # 同じ脆弱性がパッケージの数だけ数えられるため、ID の種類数も併記する
    unique_ids = len({k[0] for k in seen})
    return {"counts": counts, "fixable": fixable, "listed": listed, "total": len(seen), "unique_ids": unique_ids}


def db_built(descriptor: dict) -> str:
    """脆弱性データベースの作成日時を取り出します(grype の版によって場所が違う)。"""
    db = descriptor.get("db") or {}
    status = db.get("status") if isinstance(db.get("status"), dict) else {}
    return str(status.get("built") or db.get("built") or "不明")


def variant_dirs(root: Path) -> list[Path]:
    dirs = [p for p in root.iterdir() if (p / "vulnerabilities.json").is_file()]
    order = {v: i for i, v in enumerate(VARIANT_ORDER)}
    return sorted(dirs, key=lambda p: (order.get(p.name, len(order)), p.name))


def render(root: Path, title: str, max_rows: int = 30) -> str:
    """リリースノートの「脆弱性検査」節を Markdown で作ります。"""
    rows: list[str] = []
    details: list[str] = []
    tool_line = ""
    for vdir in variant_dirs(root):
        matches, descriptor = load_matches(vdir / "vulnerabilities.json")
        s = summarize(matches)
        image = (vdir / "image.txt").read_text(encoding="utf-8").strip() if (vdir / "image.txt").is_file() else ""
        cells = [f"{s['counts'][sev]}({s['fixable'][sev]})" for sev in SEVERITIES[:4]]
        rows.append(f"| `{vdir.name}` | " + " | ".join(cells) + f" | {s['total']} | {s['unique_ids']} |")
        if not tool_line:
            tool_line = (
                f"検査ツール: grype {descriptor.get('version', '不明')}"
                f"(脆弱性データベースの作成日時: {db_built(descriptor)})"
            )
        if s["listed"]:
            lines = [
                f"#### `{vdir.name}` の Critical / High",
                "",
                f"対象イメージ: `{image}`" if image else "",
                "",
                "| 深刻度 | ID | パッケージ | 版 | 種別 | 修正版 |",
                "|---|---|---|---|---|---|",
            ]
            for r in s["listed"][:max_rows]:
                lines.append(
                    f"| {r['severity']} | {r['id']} | {r['package']} | {r['version']} | {r['type']} | {r['fixed_in']} |"
                )
            if len(s["listed"]) > max_rows:
                lines.append("")
                rest = len(s["listed"]) - max_rows
                lines.append(f"ほか {rest} 件。全件は添付の `vulnerabilities-{vdir.name}.txt` を参照してください。")
            details.append("\n".join(lines))

    out = [
        f"## 脆弱性検査の結果({title})",
        "",
        "公開したコンテナイメージを、添付の SBOM をもとに検査した結果です。",
        "各欄は「検出件数(修正版が公開されている件数)」です。",
        "同じ脆弱性・同じパッケージの重複は 1 件として数えています。",
        "1 つの脆弱性が複数のパッケージ(ffmpeg の libavcodec・libavformat など)に該当する場合は、",
        "パッケージごとに数えます。",
        "このため、脆弱性 ID の種類数を最後の列に併記しています。",
        "",
        "| イメージ | Critical | High | Medium | Low | 合計 | 脆弱性 ID の種類数 |",
        "|---|---|---|---|---|---|---|",
        *rows,
        "",
        tool_line,
        "",
        "検出された脆弱性が、本アプリの使い方で実際に悪用できるかどうかは個別に判断が必要です。",
        "本ソフトウェアは開発・検証用途であり、認証などのセキュリティ機能を持たない点にご注意ください(README を参照)。",
        "",
        "### 添付ファイル",
        "",
        "| ファイル | 内容 |",
        "|---|---|",
        "| `sbom-<イメージ>.spdx.json` | SBOM(SPDX 形式) |",
        "| `sbom-<イメージ>.cdx.json` | SBOM(CycloneDX 形式) |",
        "| `vulnerabilities-<イメージ>.json` | 脆弱性検査の結果(grype の JSON 形式・全件) |",
        "| `vulnerabilities-<イメージ>.txt` | 脆弱性検査の結果(一覧表・全件) |",
    ]
    if details:
        out += ["", "### 深刻度の高い脆弱性", "", *("\n\n".join(details).splitlines())]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reports", type=Path, help="バリアントごとの結果を置いたディレクトリ")
    parser.add_argument("--title", default="", help="見出しに入れる版(例: v0.2.0)")
    parser.add_argument("--max-rows", type=int, default=30, help="イメージごとに一覧へ載せる最大件数")
    args = parser.parse_args(argv)
    if not args.reports.is_dir() or not variant_dirs(args.reports):
        print(f"検査結果が見つかりません: {args.reports}", file=sys.stderr)
        return 1
    sys.stdout.write(render(args.reports, args.title or "最新", args.max_rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
