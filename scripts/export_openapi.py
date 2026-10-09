"""OpenAPI の仕様書(docs/openapi.json)を書き出します。

サーバを起動しなくても API の仕様を参照・共有できるよう、リポジトリに置いておくためのものです
(コード生成ツールや、API クライアントへの取り込みにそのまま使えます)。
API を変更したら実行し直してください。仕様書がコードと食い違っていると、テスト(tests/test_openapi.py)が失敗します。

使い方:
    python scripts/export_openapi.py            # docs/openapi.json を更新
    python scripts/export_openapi.py --check    # 最新かどうかだけ確かめる(差分があれば終了コード 1)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "openapi.json"


def build_schema() -> dict:
    """モデルを読み込まない設定(ダミーの推論・全機能)でアプリを作り、仕様書を取り出します。"""
    sys.path.insert(0, str(ROOT / "src"))
    with tempfile.TemporaryDirectory() as tmp:
        os.environ.update({"DATA_DIR": tmp, "EMBEDDING_BACKEND": "dummy", "ROLE": "all", "LOG_LEVEL": "ERROR"})
        from mediasearch.config import Settings
        from mediasearch.main import create_app

        return create_app(Settings.from_env()).openapi()


def render(schema: dict) -> str:
    return json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="書き出さずに、最新かどうかだけ確かめる")
    args = parser.parse_args(argv)
    text = render(build_schema())
    if args.check:
        if not OUT.is_file() or OUT.read_text(encoding="utf-8") != text:
            print("docs/openapi.json が最新ではありません。python scripts/export_openapi.py を実行してください")
            return 1
        print("docs/openapi.json は最新です")
        return 0
    OUT.write_text(text, encoding="utf-8")
    print(f"{OUT.relative_to(ROOT)} を書き出しました({len(json.loads(text)['paths'])} 個のパス)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
