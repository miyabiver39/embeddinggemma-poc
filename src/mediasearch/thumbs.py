"""サムネイルの置き場所(DATA_DIR/thumbs)。

1 つのフォルダに数百万のファイルを置くと、ファイルシステムによっては作成・削除・一覧が遅くなり、バックアップの
ツールも扱いにくくなるため、窓の ID で 1 万件ごとのフォルダに分けます(thumbs/<ID ÷ 10000>/w<ID>.jpg)。
以前の版は thumbs/w<ID>.jpg に置いていたため、読むときと消すときは、その場所も確かめます(移動はしない)。
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

log = logging.getLogger(__name__)

PER_DIR = 10_000  # 1 つのフォルダに置くサムネイルの数


def path_for(root: Path, window_id: int) -> Path:
    """新しく作るサムネイルのパス。"""
    return root / str(window_id // PER_DIR) / f"w{window_id}.jpg"


def legacy_path(root: Path, window_id: int) -> Path:
    """以前の版のサムネイルのパス(分けずに置いていた)。"""
    return root / f"w{window_id}.jpg"


def find(root: Path, window_id: int) -> Path | None:
    """あるサムネイルのパス。無ければ None。"""
    for p in (path_for(root, window_id), legacy_path(root, window_id)):
        if p.is_file():
            return p
    return None


def remove(root: Path, window_ids: Iterable[int]) -> int:
    """窓のサムネイルを削除し、削除した数を返します(窓を削除したときに呼び、ファイルが残り続けないようにする)。"""
    removed = 0
    for wid in window_ids:
        for p in (path_for(root, wid), legacy_path(root, wid)):
            try:
                p.unlink()
                removed += 1
            except FileNotFoundError:
                pass
            except OSError as exc:  # 権限など。窓の削除は続ける
                log.warning("サムネイルを削除できません: %s (%s)", p, exc)
    return removed
