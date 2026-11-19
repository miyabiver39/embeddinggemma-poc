"""自プロセスで推論する実装(PyTorch + sentence-transformers)。

CPU / NVIDIA(CUDA) / AMD(ROCm) / Intel(XPU) のどれで動くかは、インストールされた PyTorch の
種類と `DEVICE` 環境変数で決まります。
  - PyTorch の ROCm 版は、デバイス名が "cuda" になります(AMD GPU でも DEVICE=cuda)。
  - Intel GPU は PyTorch の XPU 版を使い、デバイス名は "xpu" です。
GPU の種類ごとの手順は docs/gpu.md を参照してください。
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache, partial
from typing import TYPE_CHECKING

import numpy as np

from ..config import Settings, resolve_embed_batch
from .base import (
    AudioPart,
    Embedder,
    EmbedderInfo,
    ImagePart,
    Part,
    TextPart,
    WindowBatchEncode,
    WindowBatchPreparer,
    finalize,
)
from .batch import MicroBatcher
from .gate import PriorityGate

if TYPE_CHECKING:
    from torch import Tensor

    from .rocm import PreparedWindows

log = logging.getLogger(__name__)


# Intel GPU の有無を調べる子プロセスのコード。結果(デバイス数)を最後の行に出す
_XPU_PROBE = "import torch; print(torch.xpu.device_count())"


@dataclass(frozen=True)
class _PreparedWindow:
    parts: Sequence[Part]
    batch: PreparedWindows
    index: int


def run_probe(code: str, timeout: float = 120) -> str | None:
    """Python のコードを子プロセスで実行し、標準出力の最後の行を返します。失敗したら None。

    子プロセスが異常終了(セグメンテーション違反など)しても、このプロセスは巻き込まれません。
    """
    try:
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=timeout
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.warning("デバイスの確認に失敗しました: %s", exc)
        return None
    if result.returncode != 0:
        # 負の終了コードはシグナルによる終了(-11 = SIGSEGV)
        code = result.returncode
        reason = signal.Signals(-code).name if code < 0 and -code in signal.valid_signals() else code
        log.warning("デバイスの確認中に子プロセスが異常終了しました(%s)", reason)
        return None
    lines = result.stdout.strip().splitlines()
    return lines[-1] if lines else None


def xpu_usable() -> bool:
    """Intel GPU(XPU)が使えるかを、子プロセスで確かめます。

    Ubuntu 24.04 の Intel GPU ドライバ(libze-intel-gpu1)と PyTorch XPU 版の組み合わせでは、
    GPU が無い環境で torch.xpu.is_available() がセグメンテーション違反を起こし、
    プロセスごと落ちることを確認しています(:intel イメージを --device=/dev/dri なしで起動した場合)。
    アプリ本体を巻き込まないよう、判定だけを別プロセスで行います。
    """
    import torch

    xpu = getattr(torch, "xpu", None)
    if xpu is None or not getattr(xpu, "_is_compiled", lambda: True)():
        return False  # XPU 版ではない PyTorch(cpu / cuda / rocm イメージ)
    out = run_probe(_XPU_PROBE)
    try:
        count = int(out) if out is not None else 0
    except ValueError:
        count = 0
    if count <= 0:
        log.warning(
            "Intel GPU(XPU)が見つからないため CPU で動きます。"
            "GPU を使うには --device=/dev/dri を付けて起動してください(docs/gpu.md)"
        )
    return count > 0


def hide_xpu() -> None:
    """XPU を使わないときに、このプロセスから Intel GPU を「0 台」に見せます。

    transformers は import 時に torch.xpu.is_available() を呼ぶため、GPU の無い環境では
    xpu_usable() と同じセグメンテーション違反で落ちる(実測)。XPU 版でない PyTorch では何もしない。
    """
    import torch

    xpu = getattr(torch, "xpu", None)
    if xpu is None or not getattr(xpu, "_is_compiled", lambda: True)():
        return
    xpu.is_available = lambda: False
    xpu.device_count = lambda: 0


def resolve_device(requested: str) -> str:
    """DEVICE=auto のとき、使える GPU を自動で選びます。"""
    import torch

    if requested not in ("auto", ""):
        return requested
    if torch.cuda.is_available():  # NVIDIA CUDA / AMD ROCm
        return "cuda"
    if xpu_usable():  # Intel GPU
        return "xpu"
    return "cpu"


def accelerator_name(device: str) -> str:
    import torch

    if device.startswith("cuda"):
        return "amd-rocm" if getattr(torch.version, "hip", None) else "nvidia-cuda"
    if device.startswith("xpu"):
        return "intel-xpu"
    return "cpu"


def resolve_dtype(requested: str, device: str):
    """精度の選択。

    GPU では bfloat16 を既定にします。Gemma 系は float16 だと値があふれて NaN になることがあるためです。
    CPU は、bfloat16 が速いとは限らないので float32 を既定にします。
    """
    import torch

    table = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    if requested != "auto":
        if requested not in table:
            raise ValueError(f"DTYPE が不正です: {requested!r} (float32 / bfloat16 / float16 / auto)")
        return table[requested]
    return torch.float32 if device == "cpu" else torch.bfloat16


class LocalEmbedder(Embedder):
    def __init__(self, settings: Settings) -> None:
        import torch

        self._settings = settings
        self._device = resolve_device(settings.device)
        if not self._device.startswith("xpu"):
            hide_xpu()  # sentence-transformers(transformers)を読み込む前に行う

        self._torch_dtype = resolve_dtype(settings.dtype, self._device)
        self._accelerator = accelerator_name(self._device)
        self._dims = settings.dims
        self._gate = PriorityGate()
        rx9060xt = False
        if (
            self._accelerator == "amd-rocm"
            and settings.model_id == "google/embeddinggemma-2"
            and torch.cuda.is_available()
        ):
            gpu = torch.cuda.get_device_properties(self._device)
            # WSL2 では予約領域を引いた値が返るため、16 GB 版の判定には 15 GiB を使う。
            rx9060xt = gpu.name == "AMD Radeon RX 9060 XT" and gpu.total_memory >= 15 * 2**30
        if rx9060xt:
            # GPU が演算する構成では、CPU 前処理と推論のスレッドが実行枠を奪い合わないようにする。
            if not (os.environ.get("OMP_NUM_THREADS") or os.environ.get("MKL_NUM_THREADS")):
                torch.set_num_threads(min(torch.get_num_threads(), 2))
            from .rocm import ResidentSentenceTransformer as SentenceTransformer
        else:
            from sentence_transformers import SentenceTransformer

        log.info(
            "モデルを読み込みます: %s (device=%s, accelerator=%s, dtype=%s)",
            settings.model_id,
            self._device,
            self._accelerator,
            str(self._torch_dtype).replace("torch.", ""),
        )
        self._model = SentenceTransformer(
            settings.model_id,
            revision=settings.model_revision or None,
            device=self._device,
            model_kwargs={"dtype": self._torch_dtype},
        )
        self._modalities = list(getattr(self._model, "modalities", []) or [])
        self._apply_image_token_limit(settings.image_max_tokens)
        self._query_cache = lru_cache(maxsize=256)(self._encode_texts) if rx9060xt else None
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rocm-inference") if rx9060xt else None
        torch.set_grad_enabled(False)
        self._batch = resolve_embed_batch(settings.embed_batch, self._device)
        # 窓(複数の画像・音声の並び)は、同時に来た要求をまとめて 1 回で推論する(batch_size が 2 以上のとき)
        self._parts_batcher = MicroBatcher(self._encode_windows, self._batch) if self._batch > 1 else None
        log.info("モデルの読み込みが完了しました。対応する入力: %s、まとめる件数: %d", self._modalities, self._batch)

    # ------------------------------------------------------------------ 内部処理
    def _apply_image_token_limit(self, limit: int) -> None:
        """画像1枚あたりの上限トークンを下げると速くなりますが、細部は粗くなります。"""
        if limit <= 0:
            return
        try:
            processor = self._model[0].processor
            image_processor = processor.image_processor
            image_processor.max_soft_tokens = limit
            log.info("画像1枚あたりの上限トークンを %d にしました", limit)
        except Exception:  # モデル側の構造が変わっても起動は続ける
            log.warning("画像の上限トークンを設定できませんでした(既定値のまま動きます)", exc_info=True)

    def _to_numpy(self, tensor) -> np.ndarray:
        arr = tensor.float().cpu().numpy()
        if not np.isfinite(arr).all():
            raise RuntimeError(
                "ベクトルに NaN / 無限大が含まれています。精度が合っていない可能性があります。"
                "環境変数 DTYPE=float32 で試してください。"
            )
        return arr

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        """長辺を縮小して、推論時間とメモリを抑えます。"""
        side = self._settings.image_max_side
        h, w = image.shape[:2]
        if side > 0 and max(h, w) > side:
            from PIL import Image

            pil = Image.fromarray(image)
            pil.thumbnail((side, side))
            return np.asarray(pil)
        return image

    # ------------------------------------------------------------------ 公開 API
    def info(self) -> EmbedderInfo:
        return EmbedderInfo(
            backend="local",
            model_id=self._settings.model_id,
            dims=self._dims,
            dtype=str(self._torch_dtype).replace("torch.", ""),
            device=self._device,
            accelerator=self._accelerator,
            modalities=self._modalities,
            batch_size=self._batch,
        )

    def embed_texts(self, texts: Sequence[str], kind: str = "query") -> np.ndarray:
        prompt = "query" if kind == "query" else "document"
        if self._query_cache is not None and prompt == "query" and len(texts) == 1 and len(texts[0]) <= 1024:
            # 1 件ずつ推論したベクトルだけを再利用する。検索結果は毎回その時点の索引から求める。
            return self._query_cache((texts[0],), prompt).copy()
        return self._encode_texts(texts, prompt)

    def _encode_texts(self, texts: Sequence[str], prompt: str) -> np.ndarray:
        arr = self._encode_array(list(texts), high=True, prompt_name=prompt)
        return finalize(arr, self._dims)

    def _encode_array(self, inputs: list | dict, *, high: bool, **kwargs) -> np.ndarray:
        encode = partial(self._model.encode, inputs, convert_to_tensor=True, **kwargs)
        return self._run_encode(encode, high=high)

    def _run_encode(self, encode: Callable[[], Tensor], *, high: bool) -> np.ndarray:
        with self._gate.acquire(high=high):
            if self._worker is not None:
                # ROCm のキューと演算ライブラリの状態を同じスレッドで使い、CPU への転送も終えてから返す。
                return self._worker.submit(lambda: self._to_numpy(encode())).result()
            out = encode()
        return self._to_numpy(out)

    def embed_images(self, images: Sequence[np.ndarray], high: bool = False) -> np.ndarray:
        """画像を batch_size 枚ずつ、1 回の推論でまとめてベクトルにします(1 なら 1 枚ずつ)。"""
        vectors = []
        for i in range(0, len(images), self._batch):
            chunk = [{"image": self._prepare_image(im)} for im in images[i : i + self._batch]]
            if len(chunk) == 1:
                arr = self._encode_array(chunk[0], high=high)
            else:
                arr = self._encode_array(chunk, high=high, batch_size=len(chunk))
            vectors.append(arr.reshape(-1, arr.shape[-1]))
        return finalize(np.concatenate(vectors), self._dims)

    def embed_audio(self, pcm: np.ndarray, high: bool = False) -> np.ndarray:
        arr = self._encode_array({"audio": pcm.astype(np.float32)}, high=high)
        return finalize(arr, self._dims)

    def embed_parts(self, parts: Sequence[Part], high: bool = False) -> np.ndarray:
        """並べた順序のまま、1回の推論で1ベクトルにします。

        Google AI Edge Gallery と同じ「時刻 → 音声 → 時刻 → 画像 …」の交互配置もここで扱えます。
        batch_size が 2 以上なら、他のスレッドから同時に来た窓とまとめて推論します(検索の要求は待たせずに直接行う)。
        """
        if self._parts_batcher is not None and not high:
            return self._parts_batcher.submit(parts).result()
        return self._encode_windows([parts], high=high)[0]

    def embed_parts_many(self, items: Sequence[Sequence[Part]], high: bool = False) -> list[np.ndarray]:
        if self._parts_batcher is not None and not high:
            return self._parts_batcher.run_many(list(items))
        return [self._encode_windows([parts], high=high)[0] for parts in items]

    def video_batch_preparer(self) -> WindowBatchPreparer | None:
        if self._worker is None or self._batch > 4 or not self._model.supports_prepared_windows:
            return None
        if not 0 < self._settings.image_max_side <= 448:
            return None
        processor = getattr(self._model[0], "processor", None)
        image_processor = getattr(processor, "image_processor", None)
        if getattr(image_processor, "max_soft_tokens", None) in (70, 140, 280):
            return self._prepare_video_batch
        return None

    def _prepare_video_batch(self, items: list[Sequence[Part]]) -> WindowBatchEncode:
        # サイズの大きい入力をいくつも保持しないよう、標準的な窓に先読みを限定する。
        if sum(len(parts) for parts in items) > 16:
            return partial(self.embed_parts_many, items)
        try:
            batch = self._model.prepare_windows([self._message(parts) for parts in items])
        except Exception:
            # 入力に問題がある場合は従来どおり推論時に確認し、バッチ失敗時の再試行も維持する。
            return partial(self.embed_parts_many, items)
        prepared = [_PreparedWindow(parts, batch, i) for i, parts in enumerate(items)]
        if self._parts_batcher is not None:
            return partial(self._parts_batcher.run_many, prepared)
        return partial(self._encode_windows, prepared)

    def _message(self, parts: Sequence[Part]) -> list[dict]:
        content: list[dict] = []
        for part in parts:
            if isinstance(part, TextPart):
                content.append({"type": "text", "text": part.text})
            elif isinstance(part, ImagePart):
                content.append({"type": "image", "image": self._prepare_image(part.image)})
            elif isinstance(part, AudioPart):
                content.append({"type": "audio", "audio": part.pcm.astype(np.float32)})
            else:
                raise TypeError(f"未対応の部品です: {type(part)!r}")
        return [{"role": "user", "content": content}]

    def _encode_windows(self, items: list[Sequence[Part] | _PreparedWindow], high: bool = False) -> list[np.ndarray]:
        """複数の窓を 1 回で推論し、入力順のベクトルを返します。"""
        first = items[0]
        # ストリームとの混在や 1 件ずつの再試行時は、実際のバッチをそのまま前処理する。
        if isinstance(first, _PreparedWindow) and len(items) == len(first.batch.order) and all(
            isinstance(window, _PreparedWindow) and window.batch is first.batch and window.index == i
            for i, window in enumerate(items)
        ):
            arr = self._run_encode(partial(self._model.encode_prepared, first.batch), high=high)
            return list(finalize(arr[first.batch.order], self._dims))
        messages = [self._message(item.parts if isinstance(item, _PreparedWindow) else item) for item in items]
        arr = self._encode_array(messages, high=high, batch_size=len(messages))
        vectors = finalize(arr, self._dims)
        return list(vectors.reshape(len(messages), -1))

    def close(self) -> None:
        if self._query_cache is not None:
            self._query_cache.cache_clear()
        if self._parts_batcher is not None:
            self._parts_batcher.close()
        if self._worker is not None:
            self._worker.shutdown()
