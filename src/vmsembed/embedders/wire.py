"""app と compute の間でやり取りするデータの変換。

画像は JPEG、音声は 16bit PCM(16kHz・モノラル)にして送ります。
動画そのものは送りません。app が ffmpeg でフレームと音声に分けてから、compute に渡します。
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image


def image_to_jpeg(image: np.ndarray, quality: int = 90) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def jpeg_to_image(data: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


def pcm_to_bytes(pcm: np.ndarray) -> bytes:
    clipped = np.clip(pcm.astype(np.float32), -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def bytes_to_pcm(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32767.0
