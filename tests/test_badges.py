"""README のバッジ用データ(scripts/badges.py)のテスト。"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location("badges", Path(__file__).resolve().parents[1] / "scripts" / "badges.py")
badges = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(badges)


def test_coverage_badge(tmp_path):
    p = tmp_path / "coverage.json"
    p.write_text(json.dumps({"totals": {"percent_covered": 83.4}}))
    expected = {"schemaVersion": 1, "label": "coverage", "message": "83%", "color": "brightgreen"}
    assert badges.coverage_badge(p) == expected
    p.write_text(json.dumps({"totals": {"percent_covered": 55}}))
    assert badges.coverage_badge(p)["color"] == "orange"


def test_tests_badge(tmp_path):
    p = tmp_path / "junit.xml"
    p.write_text('<testsuites><testsuite tests="10" failures="0" errors="0" skipped="1"/></testsuites>')
    assert badges.tests_badge(p)["message"] == "9 passed, 1 skipped"
    p.write_text('<testsuite tests="3" failures="1" errors="1" skipped="0"/>')
    b = badges.tests_badge(p)
    assert b["message"] == "1 passed, 2 failed" and b["color"] == "red"


def _report(root: Path, variant: str, severities: list[str]) -> None:
    d = root / variant
    d.mkdir(parents=True)
    matches = [
        {"vulnerability": {"id": f"CVE-{i}", "severity": s, "fix": {}}, "artifact": {"name": "p", "version": "1"}}
        for i, s in enumerate(severities)
    ]
    (d / "vulnerabilities.json").write_text(json.dumps({"matches": matches}))


def test_vulns_badge_takes_worst_image(tmp_path):
    _report(tmp_path, "slim", ["Medium"])
    _report(tmp_path, "cpu", ["High", "High", "Low"])
    assert badges.vulns_badge(tmp_path) == {
        "schemaVersion": 1,
        "label": "vulnerabilities",
        "message": "critical 0 | high 2",
        "color": "orange",
    }
    _report(tmp_path, "cuda", ["Critical"])
    assert badges.vulns_badge(tmp_path)["color"] == "red"
