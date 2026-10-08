"""ベクトル化の実装。設定から適切な実装を作ります。"""

from __future__ import annotations

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
]


def build_embedder(settings: Settings) -> Embedder:
    """EMBEDDING_BACKEND に応じて、推論の実装を選びます。

    local のときだけ torch / sentence-transformers が必要です(ここで初めて読み込みます)。
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
        )
    from .local import LocalEmbedder

    return LocalEmbedder(settings)
