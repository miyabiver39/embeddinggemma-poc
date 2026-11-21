"""ベクトル化(埋め込み)の共通インターフェース。

取り込み・検索のコードは、このインターフェースだけを知っていればよく、
推論が「自プロセス」「外部の compute コンテナ」「ダミー」のどれかは意識しません。
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

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
    batch_size: int = 1  # まとめて推論する最大の件数(EMBED_BATCH。1 ならまとめない)
    runtime: str = "pytorch"  # pytorch / onnx / dummy


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

    def embed_parts_many(self, items: Sequence[Sequence[Part]], high: bool = False) -> list[np.ndarray]:
        """複数の窓をベクトルにします(窓ごとに 1 ベクトル。順序は items と同じ)。

        既定の実装は、batch_size が 2 以上なら embed_parts を並行して呼びます。
        分離構成では、compute に同時に届いた要求を
        compute 側の LocalEmbedder がまとめて推論します。自プロセスで推論する実装は、これを上書きして直接まとめます。
        """
        batch = self.info().batch_size
        if batch <= 1 or len(items) <= 1:
            return [self.embed_parts(parts, high=high) for parts in items]
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(batch, len(items))) as pool:
            return list(pool.map(lambda parts: self.embed_parts(parts, high=high), items))

    def prepare_parts_batch(self, items: Sequence[Sequence[Part]]) -> Any:
        """複数の窓に対する CPU 前処理(テンソル化・pin_memory 等)を推論より先に実行します。

        動画取り込みの Producer-Consumer パイプラインで、窓 N の推論中に別スレッドで窓 N+1 の
        前処理をオーバーラップさせるためのフックです。既定の実装はそのまま返します。
        """
        return list(items)

    def embed_prepared_parts(self, prepared: Any, high: bool = False) -> list[np.ndarray]:
        """prepare_parts_batch() の結果を受け取り、推論のみを実行してベクトル列を返します。"""
        return self.embed_parts_many(prepared, high=high)

    def warmup(self) -> dict[str, Any]:
        """起動直後の初回推論遅延を抑えるため、対応モダリティごとに 1 回ダミー推論を実行します。"""
        t0 = time.perf_counter()
        modalities = set(self.info().modalities or [])
        ran: list[str] = []
        if not modalities or "text" in modalities:
            self.embed_texts(["warmup"], kind="query")
            ran.append("text")
        if not modalities or "image" in modalities:
            dummy_img = np.zeros((64, 64, 3), dtype=np.uint8)
            self.embed_images([dummy_img], high=False)
            ran.append("image")
        if not modalities or "audio" in modalities:
            dummy_pcm = np.zeros(SAMPLE_RATE // 2, dtype=np.float32)
            self.embed_audio(dummy_pcm, high=False)
            ran.append("audio")
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        return {"ok": True, "skipped": False, "modalities": ran, "elapsed_ms": elapsed_ms}

    def metrics(self) -> dict:
        """推論の使用状況(GET /api/metrics の inference)。順番待ちの制御を持つ実装は、その積算値を返します。"""
        gate = getattr(self, "_gate", None)
        out: dict = {"backend": self.info().backend, "batch_size": self.info().batch_size}
        if gate is not None:
            out["gate"] = gate.snapshot()
        batcher = getattr(self, "_parts_batcher", None)
        if batcher is not None:
            out["batches"] = {"calls": batcher.batches, "items": batcher.items}  # items ÷ calls = 平均のまとめた件数
        return out

    def close(self) -> None:  # noqa: B027 (任意で上書きする)
        """後始末(必要な実装だけ上書きします)。"""
