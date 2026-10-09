"""実モデルでの通しテスト(`pytest -m model`)。モデルのダウンロードと CPU 推論で数分かかります。"""

from __future__ import annotations

import os

import pytest
from conftest import make_video, needs_ffmpeg
from fastapi.testclient import TestClient
from test_e2e_dummy import wait_done

from mediasearch.config import Settings
from mediasearch.main import create_app

pytestmark = [pytest.mark.model, needs_ffmpeg]


def test_real_model_text_to_video(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("EMBEDDING_BACKEND", "local")
    monkeypatch.setenv("ROLE", "all")
    monkeypatch.setenv("DIMS", os.environ.get("DIMS", "256"))
    with TestClient(create_app(Settings.from_env())) as c:
        for color in ("red", "blue", "green"):
            p = tmp_path / f"{color}.mp4"
            make_video(p, color, seconds=4)
            r = c.post("/api/ingest/path", json={"path": str(p), "camera_id": color})
            assert wait_done(c, r.json()["job_id"], timeout=600)["status"] == "done"
        res = c.post("/api/search/text", json={"query": "a solid blue screen", "top_k": 3}).json()
        print([(h["camera_id"], round(h["score"], 3)) for h in res["results"]])
        assert res["results"][0]["camera_id"] == "blue"
