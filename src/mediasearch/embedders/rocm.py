"""RX 9060 XT で常駐させる推論モデル。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import version
from threading import Lock
from typing import Any, Self

import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import batch_to_device
from torch import Tensor, device, inference_mode


@dataclass(frozen=True)
class PreparedWindows:
    features: dict[str, Any]
    order: np.ndarray


class ResidentSentenceTransformer(SentenceTransformer):
    """固定デバイスの評価モデルを毎回走査する処理を省きます。"""

    # encode() 内の整列と前処理を分けるため、照合済みの版に限って先読みする。
    supports_prepared_windows = version("sentence-transformers") == "6.1.0"
    _resident_device: device | None = None

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._processor_lock = Lock()
        self.eval()
        self._resident_device = self.device

    def to(self, *args, **kwargs) -> Self:
        # encode() は配置済みのデバイスを位置引数で渡す。他の指定は PyTorch に委ねる。
        if (
            self._resident_device is not None
            and len(args) == 1
            and isinstance(args[0], device)
            and args[0] == self._resident_device
            and not kwargs
        ):
            return self
        return super().to(*args, **kwargs)

    def train(self, mode: bool = True) -> Self:
        if mode is False and self._resident_device is not None:
            return self
        self._resident_device = None
        return super().train(mode)

    def _apply(self, fn: Callable[[Tensor], Tensor], recurse: bool = True) -> Self:
        # cpu() や dtype の変換が行われた場合も、それ以降は通常の配置・評価処理に戻す。
        self._resident_device = None
        return super()._apply(fn, recurse=recurse)

    def preprocess(self, *args, **kwargs) -> dict[str, Any]:
        # 検索の前処理と先読みが同じ tokenizer を呼ぶため、CPU 側の処理だけを直列化する。
        with self._processor_lock:
            return super().preprocess(*args, **kwargs)

    @inference_mode()
    def prepare_windows(self, messages: list[list[dict]]) -> PreparedWindows:
        # SentenceTransformer.encode() と同じ整列・プロンプトで、1 バッチの入力を CPU 上で作る。
        order = np.argsort([-self._input_length(message) for message in messages])
        if self._can_flatten_inputs():
            order = self._interleave_sorted_indices(order)
        features = self.preprocess([messages[i] for i in order], prompt=self._resolve_prompt(None, None))
        return PreparedWindows(features, np.argsort(order))

    @inference_mode()
    def encode_prepared(self, batch: PreparedWindows) -> Tensor:
        features = batch_to_device(dict(batch.features), self.device)
        out = self(features)["sentence_embedding"]
        return out[..., : self.truncate_dim]
