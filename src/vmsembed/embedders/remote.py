"""外部の compute コンテナに、ベクトル化を依頼する実装。

計算資源が乏しい環境(app 側)から、GPU のあるサーバ(compute 側)へ HTTP で依頼します。
映像の加工(ffmpeg)は app 側で済ませ、画像(JPEG)と音声(PCM)だけを送ります。
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Sequence

import httpx
import numpy as np

from .base import (
    AudioPart,
    Embedder,
    EmbedderInfo,
    ImagePart,
    Part,
    TextPart,
    finalize,
)
from .wire import image_to_jpeg, pcm_to_bytes

log = logging.getLogger(__name__)


class RemoteEmbedder(Embedder):
    def __init__(
        self,
        base_url: str,
        dims: int = 768,
        timeout_sec: float = 300.0,
        client: httpx.Client | None = None,
        retries: int = 3,
    ) -> None:
        self._dims = dims
        self._retries = retries
        # テストでは、compute アプリを直接つないだクライアントを渡せます。
        self._client = client or httpx.Client(base_url=base_url, timeout=timeout_sec)
        self._info: EmbedderInfo | None = None

    # ------------------------------------------------------------------ 内部処理
    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        """接続エラーのときだけ、間隔をあけて再試行します(compute の起動待ちを想定)。"""
        last: Exception | None = None
        for attempt in range(1, self._retries + 1):
            try:
                response = self._client.request(method, path, **kwargs)
                response.raise_for_status()
                return response
            except (httpx.ConnectError, httpx.ReadTimeout) as exc:
                last = exc
                log.warning("compute に接続できません(%d/%d): %s", attempt, self._retries, exc)
                time.sleep(min(2**attempt, 10))
        raise ConnectionError(
            f"compute({self._client.base_url})に接続できませんでした。"
            "EMBEDDING_URL と compute コンテナの起動状態を確認してください。"
        ) from last

    @staticmethod
    def _priority(high: bool) -> dict[str, str]:
        return {"high": "1" if high else "0"}

    # ------------------------------------------------------------------ 公開 API
    def info(self) -> EmbedderInfo:
        if self._info is None:
            data = self._request("GET", "/compute/info").json()
            self._info = EmbedderInfo(
                backend="remote",
                model_id=data["model_id"],
                dims=self._dims,  # 次元は app 側の設定で切り詰める
                dtype=data["dtype"],
                device=data["device"],
                accelerator=data["accelerator"],
                modalities=data["modalities"],
            )
        return self._info

    def embed_texts(self, texts: Sequence[str], kind: str = "query") -> np.ndarray:
        body = {"texts": list(texts), "kind": kind, "dims": self._dims}
        data = self._request("POST", "/compute/embed/texts", json=body).json()
        return finalize(np.asarray(data["vectors"], dtype=np.float32), self._dims)

    def embed_images(self, images: Sequence[np.ndarray], high: bool = False) -> np.ndarray:
        files = [("files", (f"f{i}.jpg", image_to_jpeg(img), "image/jpeg")) for i, img in enumerate(images)]
        data = self._request(
            "POST",
            "/compute/embed/images",
            files=files,
            data={"dims": str(self._dims), **self._priority(high)},
        ).json()
        return finalize(np.asarray(data["vectors"], dtype=np.float32), self._dims)

    def embed_audio(self, pcm: np.ndarray, high: bool = False) -> np.ndarray:
        data = self._request(
            "POST",
            "/compute/embed/audio",
            files=[("file", ("a.pcm", pcm_to_bytes(pcm), "application/octet-stream"))],
            data={"dims": str(self._dims), **self._priority(high)},
        ).json()
        return finalize(np.asarray(data["vectors"][0], dtype=np.float32), self._dims)

    def embed_parts(self, parts: Sequence[Part], high: bool = False) -> np.ndarray:
        manifest: list[dict] = []
        files: list[tuple] = []
        for i, part in enumerate(parts):
            if isinstance(part, TextPart):
                manifest.append({"type": "text", "text": part.text})
            elif isinstance(part, ImagePart):
                manifest.append({"type": "image", "file": f"p{i}"})
                files.append((f"p{i}", (f"p{i}.jpg", image_to_jpeg(part.image), "image/jpeg")))
            elif isinstance(part, AudioPart):
                manifest.append({"type": "audio", "file": f"p{i}"})
                files.append(
                    (f"p{i}", (f"p{i}.pcm", pcm_to_bytes(part.pcm), "application/octet-stream"))
                )
            else:
                raise TypeError(f"未対応の部品です: {type(part)!r}")
        data = self._request(
            "POST",
            "/compute/embed/parts",
            files=files or None,
            data={
                "manifest": json.dumps(manifest, ensure_ascii=False),
                "dims": str(self._dims),
                **self._priority(high),
            },
        ).json()
        return finalize(np.asarray(data["vectors"][0], dtype=np.float32), self._dims)

    def close(self) -> None:
        self._client.close()
