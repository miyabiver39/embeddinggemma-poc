"""動作確認用のダミー実装(モデルを使いません)。

テストや、モデルなしでの画面確認に使います。意味のある検索ができるよう、次のようにしています。
  - 文字列: 単語をハッシュしてベクトルにし、足し合わせる(同じ単語を含む文は近くなる)
  - 画像: 平均色にいちばん近い色名("red" など)の単語ベクトル
  - 音声: 音が平坦なら "noise"、そうでなければ "tone" の単語ベクトル
本物のモデルとは関係がないので、検索精度の評価には使えません。
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

import numpy as np

from .base import AudioPart, Embedder, EmbedderInfo, ImagePart, Part, TextPart, finalize

_FULL_DIMS = 768
_COLORS = {
    "red": (220, 30, 30),
    "green": (30, 180, 30),
    "blue": (30, 30, 220),
    "yellow": (230, 220, 30),
    "black": (10, 10, 10),
    "white": (245, 245, 245),
}


def _word_vec(word: str) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(word.encode("utf-8")).digest()[:8], "little")
    return np.random.default_rng(seed).standard_normal(_FULL_DIMS).astype(np.float32)


def _text_vec(text: str) -> np.ndarray:
    words = re.findall(r"\w+", text.lower()) or [""]
    return np.sum([_word_vec(w) for w in words], axis=0)


def _image_vec(image: np.ndarray) -> np.ndarray:
    mean = image.reshape(-1, 3).mean(axis=0)
    name = min(_COLORS, key=lambda c: float(np.sum((mean - np.array(_COLORS[c])) ** 2)))
    return _word_vec(name) + 0.2 * _word_vec("image")


def _audio_vec(pcm: np.ndarray) -> np.ndarray:
    spectrum = np.abs(np.fft.rfft(pcm.astype(np.float64))) + 1e-9
    flatness = float(np.exp(np.mean(np.log(spectrum))) / np.mean(spectrum))
    label = "noise" if flatness > 0.2 else "tone"
    return _word_vec(label) + 0.2 * _word_vec("audio")


class DummyEmbedder(Embedder):
    def __init__(self, dims: int = 768) -> None:
        self._dims = dims

    def info(self) -> EmbedderInfo:
        return EmbedderInfo(
            backend="dummy",
            model_id="dummy",
            dims=self._dims,
            dtype="float32",
            device="cpu",
            accelerator="dummy",
            modalities=["text", "image", "audio", "video", "message"],
        )

    def embed_texts(self, texts: Sequence[str], kind: str = "query") -> np.ndarray:
        return finalize(np.stack([_text_vec(t) for t in texts]), self._dims)

    def embed_images(self, images: Sequence[np.ndarray], high: bool = False) -> np.ndarray:
        return finalize(np.stack([_image_vec(i) for i in images]), self._dims)

    def embed_audio(self, pcm: np.ndarray, high: bool = False) -> np.ndarray:
        return finalize(_audio_vec(pcm), self._dims)

    def embed_parts(self, parts: Sequence[Part], high: bool = False) -> np.ndarray:
        total = np.zeros(_FULL_DIMS, dtype=np.float32)
        for part in parts:
            if isinstance(part, ImagePart):
                total += _image_vec(part.image)
            elif isinstance(part, AudioPart):
                total += _audio_vec(part.pcm)
            elif isinstance(part, TextPart):
                total += 0.05 * _text_vec(part.text)
        return finalize(total, self._dims)
