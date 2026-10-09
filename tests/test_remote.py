"""app と compute を分けた構成(外部ベクトル化)の通しテスト。TestClient を HTTP の代わりに使う。"""

from __future__ import annotations

import io
from dataclasses import replace

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from mediasearch.embedders.dummy import DummyEmbedder
from mediasearch.embedders.remote import RemoteEmbedder
from mediasearch.main import create_app


def test_remote_embedder_matches_local(settings, tmp_path):
    compute_settings = replace(settings, role="compute", data_dir=tmp_path / "c")
    local = DummyEmbedder(dims=settings.dims)
    with TestClient(create_app(compute_settings, embedder=local)) as http:
        remote = RemoteEmbedder("http://testserver", dims=settings.dims, timeout_sec=10, client=http)
        assert remote.info().model_id == local.info().model_id

        a = remote.embed_texts(["red car"])
        b = local.embed_texts(["red car"])
        assert np.allclose(a, b, atol=1e-5)

        buf = io.BytesIO()
        Image.new("RGB", (32, 32), (255, 0, 0)).save(buf, "JPEG")
        img = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))
        r = remote.embed_images([img])
        assert r.shape == (1, settings.dims)

        pcm = np.sin(np.linspace(0, 440 * 6.28, 16000)).astype(np.float32)
        assert remote.embed_audio(pcm).shape[-1] == settings.dims


def test_compute_role_has_no_app_routes(settings, tmp_path):
    compute_settings = replace(settings, role="compute", data_dir=tmp_path / "c")
    with TestClient(create_app(compute_settings, embedder=DummyEmbedder(dims=settings.dims))) as c:
        assert c.get("/compute/info").status_code == 200
        assert c.get("/api/info").status_code == 404
