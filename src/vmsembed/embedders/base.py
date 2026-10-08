"""ベクトル化(埋め込み)の共通インターフェース。

取り込み・検索のコードは、このインターフェースだけを知っていればよく、
推論が「自プロセス」「外部の compute コンテナ」「ダミー」のどれかは意識しません。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 16_000  # モデルが想定する音声のサンプリングレート(Hz)・モノラル


@dataclass(frozen=True)
class EmbedderInfo:
    """推論側の情報。DB に記録して、モデルの混在を検出するのに使います。"""

    backend: str  # local / remote / dummy
    model_id: str
    dims: int
    dtype: str
    device: str
    accelerator: str  # cpu / nvidia-cuda / amd-rocm / intel-xpu / dummy
    modalities: list[str]


@dataclass
class TextPart:
    """文字列。窓の中の時刻ラベル("00:03")などに使います。"""

    text: str


@dataclass
class ImagePart:
    """画像。RGB の uint8 配列(高さ, 幅, 3)。"""

    image: np.ndarray


@dataclass
class AudioPart:
    """音声。16kHz モノラルの float32 配列(-1.0〜1.0)。"""

    pcm: np.ndarray


Part = TextPart | ImagePart | AudioPart


def finalize(vectors: np.ndarray, dims: int) -> np.ndarray:
    """MRL(Matryoshka)で次元を切り詰め、L2 正規化し直します。

    切り詰めたあとに正規化し直さないと、コサイン類似度が正しく比較できません。
    """
    vectors = np.asarray(vectors, dtype=np.float32)
    vectors = vectors[..., :dims]
    norm = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norm, 1e-12)


class Embedder(ABC):
    """ベクトル化の窓口。

    `high=True` は検索(待たせたくない処理)です。取り込み(`high=False`)より優先して実行します。
    """

    @abstractmethod
    def info(self) -> EmbedderInfo: ...

    @abstractmethod
    def embed_texts(self, texts: Sequence[str], kind: str = "query") -> np.ndarray:
        """文字列をベクトル化します。kind は query(検索文) / document(登録する文章)。"""

    @abstractmethod
    def embed_images(self, images: Sequence[np.ndarray], high: bool = False) -> np.ndarray:
        """画像をベクトル化します。画像1枚につき1ベクトルを返します。"""

    @abstractmethod
    def embed_audio(self, pcm: np.ndarray, high: bool = False) -> np.ndarray:
        """音声1本を1ベクトルにします(戻り値は (dims,))。"""

    @abstractmethod
    def embed_parts(self, parts: Sequence[Part], high: bool = False) -> np.ndarray:
        """並べた順序のまま、複数の部品(文字・画像・音声)を1ベクトルにします(戻り値は (dims,))。"""

    def close(self) -> None:  # noqa: B027 (任意で上書きする)
        """後始末(必要な実装だけ上書きします)。"""
