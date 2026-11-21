"""ベクトル化の実装。設定から適切な実装を作ります。"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from ..config import Settings
from .base import (
    SAMPLE_RATE,
    AudioPart,
    Embedder,
    EmbedderInfo,
    ImagePart,
    Part,
    TextPart,
    finalize,
)

__all__ = [
    "SAMPLE_RATE",
    "AudioPart",
    "Embedder",
    "EmbedderInfo",
    "ImagePart",
    "Part",
    "TextPart",
    "build_embedder",
    "finalize",
    "should_use_onnx",
]


def _gpu_available() -> bool:
    """利用可能な GPU(CUDA / ROCm / Intel XPU)があるか判定します。"""
    if importlib.util.find_spec("torch") is None:
        return False
    from .local import resolve_device

    return resolve_device("auto") != "cpu"


def should_use_onnx(settings: Settings) -> bool:
    """EMBEDDING_RUNTIME と環境から、ONNX Runtime(OnnxEmbedder)を使うべきか決めます。

    - EMBEDDING_RUNTIME=onnx: 常に ONNX Runtime を使う
    - EMBEDDING_RUNTIME=pytorch: 常に PyTorch(LocalEmbedder)を使う
    - EMBEDDING_RUNTIME=auto: CPU 実行かつ onnxruntime が導入済みで、同梱済み ONNX モデル
      (ONNX_MODEL_DIR)が存在するかオンライン取得可能な場合に ONNX Runtime を使う
    """
    if settings.embedding_runtime == "onnx":
        return True
    if settings.embedding_runtime == "pytorch":
        return False
    # auto: GPU が明示指定、または自動検出で使える場合は PyTorch を使う
    if settings.device not in ("auto", "", "cpu"):
        return False
    if importlib.util.find_spec("onnxruntime") is None:
        return False
    if settings.onnx_model_dir:
        if not Path(settings.onnx_model_dir).is_dir():
            return False
    elif os.environ.get("HF_HUB_OFFLINE", "0") == "1":
        return False
    if settings.device in ("auto", "") and _gpu_available():
        return False
    return True


def build_embedder(settings: Settings) -> Embedder:
    """EMBEDDING_BACKEND と EMBEDDING_RUNTIME に応じて、推論の実装を選びます。

    local のときだけ torch / onnxruntime などを読み込みます。
    """
    if settings.embedding_backend == "dummy":
        from .dummy import DummyEmbedder

        return DummyEmbedder(settings.dims)
    if settings.embedding_backend == "remote":
        from .remote import RemoteEmbedder

        return RemoteEmbedder(
            settings.embedding_url,
            dims=settings.dims,
            timeout_sec=settings.embedding_timeout_sec,
            token=settings.embedding_token,
        )
    if should_use_onnx(settings):
        from .onnx import OnnxEmbedder

        return OnnxEmbedder(settings)
    from .local import LocalEmbedder

    return LocalEmbedder(settings)
