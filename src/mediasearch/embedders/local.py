"""自プロセスで推論する実装(PyTorch + sentence-transformers)。

CPU / NVIDIA(CUDA) / AMD(ROCm) / Intel(XPU) のどれで動くかは、インストールされた PyTorch の
種類と `DEVICE` 環境変数で決まります。
  - PyTorch の ROCm 版は、デバイス名が "cuda" になります(AMD GPU でも DEVICE=cuda)。
  - Intel GPU は PyTorch の XPU 版を使い、デバイス名は "xpu" です。
GPU の種類ごとの手順は docs/gpu.md を参照してください。
"""

from __future__ import annotations

import logging
import signal
import subprocess
import sys
from collections.abc import Sequence

import numpy as np

from ..config import Settings
from .base import (
    AudioPart,
    Embedder,
    EmbedderInfo,
    ImagePart,
    Part,
    TextPart,
    finalize,
)
from .gate import PriorityGate

log = logging.getLogger(__name__)


# Intel GPU の有無を調べる子プロセスのコード。結果(デバイス数)を最後の行に出す
_XPU_PROBE = "import torch; print(torch.xpu.device_count())"


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
        from sentence_transformers import SentenceTransformer

        self._torch_dtype = resolve_dtype(settings.dtype, self._device)
        self._accelerator = accelerator_name(self._device)
        self._dims = settings.dims
        self._gate = PriorityGate()

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
        torch.set_grad_enabled(False)
        log.info("モデルの読み込みが完了しました。対応する入力: %s", self._modalities)

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
        )

    def embed_texts(self, texts: Sequence[str], kind: str = "query") -> np.ndarray:
        prompt = "query" if kind == "query" else "document"
        with self._gate.acquire(high=True):
            out = self._model.encode(list(texts), prompt_name=prompt, convert_to_tensor=True)
        return finalize(self._to_numpy(out), self._dims)

    def embed_images(self, images: Sequence[np.ndarray], high: bool = False) -> np.ndarray:
        vectors = []
        for image in images:
            with self._gate.acquire(high=high):
                out = self._model.encode(
                    {"image": self._prepare_image(image)}, convert_to_tensor=True
                )
            vectors.append(self._to_numpy(out))
        return finalize(np.stack(vectors), self._dims)

    def embed_audio(self, pcm: np.ndarray, high: bool = False) -> np.ndarray:
        with self._gate.acquire(high=high):
            out = self._model.encode({"audio": pcm.astype(np.float32)}, convert_to_tensor=True)
        return finalize(self._to_numpy(out), self._dims)

    def embed_parts(self, parts: Sequence[Part], high: bool = False) -> np.ndarray:
        """並べた順序のまま、1回の推論で1ベクトルにします。

        Google AI Edge Gallery と同じ「時刻 → 音声 → 時刻 → 画像 …」の交互配置もここで扱えます。
        """
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
        with self._gate.acquire(high=high):
            out = self._model.encode(
                [{"role": "user", "content": content}], convert_to_tensor=True
            )
        vector = finalize(self._to_numpy(out), self._dims)
        return vector[0] if vector.ndim > 1 else vector
