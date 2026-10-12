"""テスト共通の準備。モデル不要のダミー埋め込みと、ffmpeg で作る合成動画を使います。"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time

import httpx
import pytest
import uvicorn
from fastapi.testclient import TestClient

from mediasearch.config import Settings
from mediasearch.main import create_app

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


@pytest.fixture()
def live_server(settings):
    """実際の HTTP サーバー(TestClient は応答を最後まで読んでから返すため、SSE の確認には使えない)。"""
    server = uvicorn.Server(uvicorn.Config(create_app(settings), host="127.0.0.1", port=0, log_level="warning"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.1)
    port = server.servers[0].sockets[0].getsockname()[1]
    with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=30) as client:
        yield client
    server.should_exit = True
    th.join(10)
