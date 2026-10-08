"""リリースノート用の脆弱性集計(scripts/release_report.py)のテスト。"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "release_report", Path(__file__).resolve().parents[1] / "scripts" / "release_report.py"
)
release_report = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(release_report)


def _match(vid: str, sev: str, name: str, version: str = "1.0", fixed: list[str] | None = None) -> dict:
    return {
        "vulnerability": {
            "id": vid,
            "severity": sev,
            "fix": {"state": "fixed" if fixed else "not-fixed", "versions": fixed or []},
        },
        "artifact": {"name": name, "version": version, "type": "deb"},
    }


def _write(root: Path, variant: str, matches: list[dict]) -> None:
    d = root / variant
    d.mkdir(parents=True)
    descriptor = {"name": "grype", "version": "0.99.0", "db": {"status": {"built": "2026-10-01T00:00:00Z"}}}
    (d / "vulnerabilities.json").write_text(json.dumps({"matches": matches, "descriptor": descriptor}))
    (d / "image.txt").write_text(f"ghcr.io/example/app:0.2.0-{variant}@sha256:abc\n")


def test_summarize_dedupes_and_counts_fixable():
    s = release_report.summarize(
        [
            _match("CVE-1", "High", "openssl", fixed=["3.0.14"]),
            _match("CVE-1", "High", "openssl", fixed=["3.0.14"]),  # 重複
            _match("CVE-2", "Critical", "zlib"),
            _match("CVE-3", "Low", "bash"),
            _match("CVE-4", "Strange", "x"),  # 未知の深刻度は Unknown
        ]
    )
    assert s["total"] == 4
    assert s["counts"]["High"] == 1 and s["fixable"]["High"] == 1
    assert s["counts"]["Critical"] == 1 and s["fixable"]["Critical"] == 0
    assert s["counts"]["Unknown"] == 1
    assert [r["id"] for r in s["listed"]] == ["CVE-2", "CVE-1"]  # Critical が先


def test_render_orders_variants_and_lists_high(tmp_path):
    _write(tmp_path, "cpu", [_match("CVE-9", "High", "pip", fixed=["26.0"])])
    _write(tmp_path, "slim", [])
    md = release_report.render(tmp_path, "v0.2.0")
    assert md.index("| `slim` |") < md.index("| `cpu` |")
    assert "| `cpu` | 0(0) | 1(1) | 0(0) | 0(0) | 1 |" in md
    assert "| High | CVE-9 | pip | 1.0 | deb | 26.0 |" in md
    assert "grype 0.99.0" in md and "2026-10-01" in md


def test_main_fails_without_reports(tmp_path, capsys):
    assert release_report.main([str(tmp_path)]) == 1
