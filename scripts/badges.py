"""README のバッジ用のデータ(shields.io の endpoint 形式の JSON)を作り、badges ブランチに公開します。

GitHub Actions から呼び出します。バッジの値(カバレッジ・テスト件数・脆弱性の件数)は CI の実行結果から作るため、
README を書き換えずに最新の状態を表示できます。表示は shields.io の endpoint バッジが、
badges ブランチの JSON を読み込んで行います(https://shields.io/badges/endpoint-badge)。

使い方:
    python scripts/badges.py coverage coverage.json out/coverage.json   # pytest-cov の JSON から
    python scripts/badges.py tests junit.xml out/tests.json             # pytest の JUnit XML から
    python scripts/badges.py vulns reports/ out/vulnerabilities.json    # security.yml の検査結果から
    python scripts/badges.py publish out/                               # badges ブランチへ push

Python の標準ライブラリだけで動きます(ランナーに追加のインストールをしなくて済むようにするため)。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

BRANCH = "badges"


def badge(label: str, message: str, color: str) -> dict:
    return {"schemaVersion": 1, "label": label, "message": message, "color": color}


def coverage_badge(path: Path) -> dict:
    """pytest-cov(coverage.py)の JSON レポートから、行カバレッジのバッジを作ります。"""
    pct = float(json.loads(path.read_text(encoding="utf-8"))["totals"]["percent_covered"])
    color = "brightgreen" if pct >= 80 else "green" if pct >= 70 else "yellow" if pct >= 60 else "orange"
    return badge("coverage", f"{pct:.0f}%", color)


def tests_badge(path: Path) -> dict:
    """JUnit XML から、成功・失敗・スキップの件数のバッジを作ります。"""
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    total = sum(int(s.get("tests", 0)) for s in suites)
    failed = sum(int(s.get("failures", 0)) + int(s.get("errors", 0)) for s in suites)
    skipped = sum(int(s.get("skipped", 0)) for s in suites)
    passed = total - failed - skipped
    parts = [f"{passed} passed"]
    if failed:
        parts.append(f"{failed} failed")
    if skipped:
        parts.append(f"{skipped} skipped")
    message = ", ".join(parts)
    return badge("tests", message, "red" if failed else "brightgreen")


def vulns_badge(reports: Path) -> dict:
    """全イメージの検査結果から、Critical / High の最大件数のバッジを作ります。

    イメージごとに件数が違うため、最も多いイメージの値を表示します(一番悪い状態を見せる)。
    """
    spec = importlib.util.spec_from_file_location("release_report", Path(__file__).with_name("release_report.py"))
    rr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rr)
    critical = high = 0
    for vdir in rr.variant_dirs(reports):
        matches, _ = rr.load_matches(vdir / "vulnerabilities.json")
        counts = rr.summarize(matches)["counts"]
        critical, high = max(critical, counts["Critical"]), max(high, counts["High"])
    color = "red" if critical else "orange" if high else "brightgreen"
    return badge("vulnerabilities", f"critical {critical} | high {high}", color)


def publish(src: Path, message: str) -> None:
    """src の JSON を badges ブランチに push します(他のファイルは残します)。

    build.yml と security.yml が同時に push することがあるため、競合したら取り直して再試行します。
    """
    repo = os.environ["GITHUB_REPOSITORY"]
    token = os.environ["GITHUB_TOKEN"]
    url = f"https://x-access-token:{token}@github.com/{repo}.git"

    def git(*args: str, cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(["git", *args], cwd=cwd, check=check, capture_output=True, text=True)

    for attempt in range(1, 6):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            git("init", "-q", cwd=work)
            git("config", "user.name", "github-actions[bot]", cwd=work)
            git("config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com", cwd=work)
            git("remote", "add", "origin", url, cwd=work)
            if git("fetch", "-q", "--depth", "1", "origin", BRANCH, cwd=work, check=False).returncode == 0:
                git("checkout", "-q", "-B", BRANCH, "FETCH_HEAD", cwd=work)
            else:  # 初回: 履歴を持たないブランチを作る
                git("checkout", "-q", "--orphan", BRANCH, cwd=work)
                (work / "README.md").write_text(
                    "README のバッジ用のデータです。GitHub Actions が自動で更新します(scripts/badges.py)。\n",
                    encoding="utf-8",
                )
            for f in src.glob("*.json"):
                shutil.copy(f, work / f.name)
            git("add", "-A", cwd=work)
            if git("diff", "--cached", "--quiet", cwd=work, check=False).returncode == 0:
                print("バッジに変更はありません")
                return
            git("commit", "-q", "-m", message, cwd=work)
            if git("push", "-q", "origin", f"HEAD:{BRANCH}", cwd=work, check=False).returncode == 0:
                print(f"badges ブランチを更新しました: {', '.join(f.name for f in src.glob('*.json'))}")
                return
        print(f"push が競合しました。取り直して再試行します({attempt}/5)", file=sys.stderr)
    raise SystemExit("badges ブランチを更新できませんでした")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("coverage", "tests", "vulns"):
        p = sub.add_parser(name)
        p.add_argument("source", type=Path)
        p.add_argument("out", type=Path)
    p = sub.add_parser("publish")
    p.add_argument("dir", type=Path)
    p.add_argument("--message", default="バッジを更新")
    args = parser.parse_args(argv)

    if args.cmd == "publish":
        publish(args.dir, args.message)
        return 0
    make = {"coverage": coverage_badge, "tests": tests_badge, "vulns": vulns_badge}[args.cmd]
    data = make(args.source)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(data, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
