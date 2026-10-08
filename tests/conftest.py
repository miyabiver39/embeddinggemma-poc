"""テスト共通の準備。モデル不要のダミー埋め込みと、ffmpeg で作る合成動画を使います。"""

from __future__ import annotations

import shutil
import subprocess

import pytest
from fastapi.testclient import TestClient

from vmsembed.config import Settings
from vmsembed.main import create_app

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg が必要です")


def make_video(path, color: str, seconds: int = 6, audio: str | None = None) -> None:
    """単色の合成動画を作る。audio は 'tone' なら正弦波を付ける。"""
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c={color}:s=320x240:r=10:d={seconds}"]
    if audio == "tone":
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}"]
    cmd += ["-pix_fmt", "yuv420p", str(path)]
    subprocess.run(cmd, check=True)


@pytest.fixture()
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("EMBEDDING_BACKEND", "dummy")
    monkeypatch.setenv("ROLE", "all")
    monkeypatch.setenv("INGEST_ROOTS", str(tmp_path))  # テストの合成動画は tmp_path の下に作る
    return Settings.from_env()


@pytest.fixture()
def client(settings):
    with TestClient(create_app(settings)) as c:
        yield c


def pytest_collection_modifyitems(config, items):
    """実モデルのテストは、RUN_MODEL_TESTS=1 を付けたときだけ実行する(重いため)。"""
    import os

    if os.environ.get("RUN_MODEL_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="RUN_MODEL_TESTS=1 を付けると実行します")
    for item in items:
        if "model" in item.keywords:
            item.add_marker(skip)
