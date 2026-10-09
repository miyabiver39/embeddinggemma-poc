"""OpenAPI の仕様書のテスト(開発者向けの資料として、最新で、説明が揃っていること)。"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("export_openapi", ROOT / "scripts" / "export_openapi.py")
export_openapi = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(export_openapi)


def test_committed_openapi_is_up_to_date():
    committed = (ROOT / "docs" / "openapi.json").read_text(encoding="utf-8")
    assert committed == export_openapi.render(export_openapi.build_schema()), (
        "docs/openapi.json が最新ではありません。python scripts/export_openapi.py を実行してください"
    )


def test_every_operation_is_documented():
    schema = json.loads((ROOT / "docs" / "openapi.json").read_text(encoding="utf-8"))
    assert {"bearer", "apiKey"} <= set(schema["components"]["securitySchemes"])
    for path, ops in schema["paths"].items():
        for method, op in ops.items():
            where = f"{method.upper()} {path}"
            assert op.get("summary"), f"{where} に summary がありません"
            assert op.get("tags"), f"{where} に tags がありません"
            ok = op["responses"].get("200", {})
            assert ok.get("content"), f"{where} の成功時の応答の型がありません"
            if path.startswith("/api/") or path.startswith("/compute/"):
                assert "401" in op["responses"], f"{where} に 401(認証)の応答がありません"
                assert {} in op["security"], f"{where} は認証なしでも呼べる旨(API_TOKEN 未設定時)がありません"
