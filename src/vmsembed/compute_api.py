"""compute(ベクトル化だけを行う)の HTTP API。

ROLE=compute のコンテナの本体です。ROLE=all のコンテナにも同じ API が付くので、
「全部入り」のコンテナを、別の app から使う compute として使うこともできます。

認証(API_TOKEN)などは security.py のミドルウェアが行います。app から呼ぶ場合は、app に EMBEDDING_TOKEN を設定します。
"""

from __future__ import annotations

import json

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from .embedders import AudioPart, Embedder, ImagePart, Part, TextPart, finalize
from .embedders.wire import bytes_to_pcm, jpeg_to_image


class TextsRequest(BaseModel):
    texts: list[str]
    kind: str = Field(default="query", pattern="^(query|document)$")
    dims: int = 768


def build_compute_router(embedder: Embedder) -> APIRouter:
    router = APIRouter(prefix="/compute", tags=["compute"])
    own_dims = embedder.info().dims

    def _dims(requested) -> int:
        """app が求める次元。compute の次元以下なら、切り詰めて返せます(MRL)。"""
        try:
            requested = int(requested)
        except (TypeError, ValueError) as exc:
            raise HTTPException(400, f"dims は整数で指定してください: {requested!r}") from exc
        if requested not in (768, 512, 256, 128):
            raise HTTPException(400, "dims は 768 / 512 / 256 / 128 のいずれかにしてください")
        if requested > own_dims:
            raise HTTPException(
                400, f"dims={requested} は compute の次元({own_dims})より大きいため返せません"
            )
        return requested

    def _vectors(array: np.ndarray, dims: int) -> dict:
        array = np.atleast_2d(array)
        return {"vectors": finalize(array, dims).tolist()}

    @router.get("/info")
    def info() -> dict:
        i = embedder.info()
        return {
            "model_id": i.model_id,
            "dims": i.dims,
            "dtype": i.dtype,
            "device": i.device,
            "accelerator": i.accelerator,
            "modalities": i.modalities,
            "backend": i.backend,
        }

    @router.post("/embed/texts")
    async def embed_texts(body: TextsRequest) -> dict:
        dims = _dims(body.dims)
        vectors = await run_in_threadpool(embedder.embed_texts, body.texts, body.kind)
        return _vectors(vectors, dims)

    @router.post("/embed/images")
    async def embed_images(request: Request) -> dict:
        form = await request.form()
        dims = _dims(form.get("dims", own_dims))
        high = form.get("high") == "1"
        images = [jpeg_to_image(await f.read()) for f in form.getlist("files")]
        if not images:
            raise HTTPException(400, "files(画像)がありません")
        vectors = await run_in_threadpool(embedder.embed_images, images, high)
        return _vectors(vectors, dims)

    @router.post("/embed/audio")
    async def embed_audio(request: Request) -> dict:
        form = await request.form()
        dims = _dims(form.get("dims", own_dims))
        high = form.get("high") == "1"
        upload = form.get("file")
        if upload is None:
            raise HTTPException(400, "file(16bit PCM・16kHz・モノラル)がありません")
        pcm = bytes_to_pcm(await upload.read())
        vector = await run_in_threadpool(embedder.embed_audio, pcm, high)
        return _vectors(vector, dims)

    @router.post("/embed/parts")
    async def embed_parts(request: Request) -> dict:
        """文字・画像・音声を、並べた順序のまま1ベクトルにします。"""
        form = await request.form()
        dims = _dims(form.get("dims", own_dims))
        high = form.get("high") == "1"
        try:
            manifest = json.loads(str(form.get("manifest", "[]")))
        except json.JSONDecodeError as exc:
            raise HTTPException(400, f"manifest が JSON として読めません: {exc}") from exc
        parts: list[Part] = []
        for item in manifest:
            kind = item.get("type")
            if kind == "text":
                parts.append(TextPart(str(item.get("text", ""))))
                continue
            upload = form.get(str(item.get("file")))
            if upload is None:
                raise HTTPException(400, f"manifest が参照するファイルがありません: {item}")
            data = await upload.read()
            if kind == "image":
                parts.append(ImagePart(jpeg_to_image(data)))
            elif kind == "audio":
                parts.append(AudioPart(bytes_to_pcm(data)))
            else:
                raise HTTPException(400, f"未対応の type です: {kind!r}")
        if not parts:
            raise HTTPException(400, "manifest が空です")
        vector = await run_in_threadpool(embedder.embed_parts, parts, high)
        return _vectors(vector, dims)

    return router
