"""DATA_DIR の空き容量の確認。

ディスクが満杯になると、SQLite・ベクトル DB のジャーナル・サムネイルの書き込みが途中で失敗し、
取り込みが中途半端な状態で止まります。空き容量が MIN_FREE_DISK_MB を下回ったら、新しい取り込み
(アップロード・パス指定・監視フォルダ・ストリームの窓)を止め、検索と削除(容量を空ける操作)は続けます。
"""

from __future__ import annotations

import shutil
from pathlib import Path


class InsufficientStorage(RuntimeError):
    """空き容量が足りない(API では 507)。"""


def status(data_dir: Path, min_free_mb: int) -> dict:
    try:
        usage = shutil.disk_usage(data_dir)
    except OSError as exc:
        return {"ok": False, "error": str(exc)}
    free_mb = usage.free // 2**20
    return {
        "ok": min_free_mb <= 0 or free_mb >= min_free_mb,
        "free_mb": free_mb,
        "total_mb": usage.total // 2**20,
        "min_free_mb": min_free_mb,
    }


def check(data_dir: Path, min_free_mb: int) -> None:
    """空き容量が足りなければ InsufficientStorage を出します。"""
    s = status(data_dir, min_free_mb)
    if not s["ok"]:
        if "error" in s:
            raise InsufficientStorage(f"DATA_DIR の空き容量を確認できません: {s['error']}")
        raise InsufficientStorage(
            f"DATA_DIR の空き容量が足りません(残り {s['free_mb']} MB、下限 {min_free_mb} MB)。"
            "保存期間(RETENTION_DAYS)を設定するか、不要な取り込み元を削除してください"
            "(MIN_FREE_DISK_MB で下限を変更できます)"
        )
