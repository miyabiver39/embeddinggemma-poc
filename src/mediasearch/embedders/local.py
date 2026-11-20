"""自プロセスで推論する実装(PyTorch + sentence-transformers)。

CPU / NVIDIA(CUDA) / AMD(ROCm) / Intel(XPU) のどれで動くかは、インストールされた PyTorch の
種類と `DEVICE` 環境変数で決まります。
  - PyTorch の ROCm 版は、デバイス名が "cuda" になります(AMD GPU でも DEVICE=cuda)。
  - Intel GPU は PyTorch の XPU 版を使い、デバイス名は "xpu" です。
GPU の種類ごとの手順は docs/gpu.md を参照してください。

推論の最適化方針(出力ベクトルのビット一致を維持する範囲):
  - `model.to(device)` と `model.eval()` は起動時に 1 回だけ実行し、毎回の推論で全パラメータを走査しない。
  - CPU 前処理(トークナイズ・画像リサイズ/正規化・音声特徴抽出・テンソル化)とモデル本体(forward)を分離し、
    前処理を PriorityGate の外(および動画取り込みの Producer スレッド)で実行する。
  - CUDA / ROCm 環境では Producer 側で CPU テンソルを pin_memory し、forward 時に non_blocking で転送する。
  - Inductor / Triton のキャッシュ先を永続化ディレクトリ(既定 DATA_DIR/cache/torch)へ向け、再起動後の
    コールドスタート遅延を抑える。
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np

from ..config import Settings, resolve_embed_batch
from . import accel
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
from .batch import MicroBatcher
from .gate import PriorityGate

log = logging.getLogger(__name__)

TEXT_BATCH = 8  # 検索の文章をまとめる最大の件数(TEXT_GRAPH_BATCHES の最大と合わせる)
TEXT_WAIT_SEC = 0.003  # 同時に検索が来ているとき、次の文章を待つ時間
# 文章の推論を CUDA Graph に記録する段階(accel.TextGraphs)。件数は TEXT_BATCH まで、長さは検索の文章が収まる 128 まで
TEXT_GRAPH_BATCHES = (1, 2, 4, 8)
TEXT_GRAPH_LENGTHS = (16, 32, 64, 128)


# Intel GPU の有無を調べる子プロセスのコード。結果(デバイス数)を最後の行に出す
_XPU_PROBE = "import torch; print(torch.xpu.device_count())"


def configure_torch_cache(cache_dir: Path) -> dict[str, str]:
    """PyTorch(Inductor / FX Graph / Triton)のキャッシュ先を永続化ディレクトリへ設定します。

    torch を import する前に呼ぶことで、Inductor 初期化時の既定キャッシュパスに反映されます。
    環境変数が利用側で明示指定されている場合は、その指定を優先します(setdefault)。
    """
    accel.setup_compile_cache(cache_dir, cache_dir=cache_dir)
    return {
        "TORCHINDUCTOR_CACHE_DIR": os.environ.get("TORCHINDUCTOR_CACHE_DIR", str(cache_dir / "inductor")),
        "TORCHINDUCTOR_FX_GRAPH_CACHE": os.environ.get("TORCHINDUCTOR_FX_GRAPH_CACHE", "1"),
        "TRITON_CACHE_DIR": os.environ.get("TRITON_CACHE_DIR", str(cache_dir / "triton")),
    }


def pin_features(features: dict[str, Any]) -> dict[str, Any]:
    """辞書内の CPU テンソルをページロック(pin_memory)します。

    CUDA / ROCm への転送前に Producer スレッド側で呼ぶことで、non_blocking 転送を有効にします。
    pin_memory に対応しないテンソルや CPU 専用環境では元の値をそのまま残します。
    """
    pinned: dict[str, Any] = {}
    for key, value in features.items():
        if (
            hasattr(value, "pin_memory")
            and getattr(getattr(value, "device", None), "type", None) == "cpu"
            and not getattr(value, "is_pinned", lambda: False)()
        ):
            try:
                value = value.pin_memory()
            except RuntimeError:
                pass
        pinned[key] = value
    return pinned


def move_features(features: dict[str, Any], target_device: Any, non_blocking: bool = False) -> dict[str, Any]:
    """辞書内のテンソルを target_device へ移します。すでに同じデバイス上のテンソルは .to() を省きます。"""
    moved: dict[str, Any] = {}
    for key, value in features.items():
        if hasattr(value, "to") and hasattr(value, "device"):
            if value.device != target_device:
                value = value.to(target_device, non_blocking=non_blocking)
        moved[key] = value
    return moved


@dataclass
class _PreparedBatch:
    """CPU 前処理(トークナイズ・画像/音声テンソル化)が終わった推論入力。"""

    features: dict[str, Any]
    restore_idx: np.ndarray | None
    count: int


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
    _TEXT_BATCH_SIZE = 32  # sentence-transformers.encode の既定値と揃える

    def __init__(self, settings: Settings) -> None:
        # compile の結果の保存先は、torch(inductor)を読み込む前に決める必要がある(後から変えても効かない)
        self._cache_env = configure_torch_cache(settings.torch_cache_dir)
        import torch

        self._torch = torch
        self._settings = settings
        self._device = resolve_device(settings.device)
        if not self._device.startswith("xpu"):
            hide_xpu()  # sentence-transformers(transformers)を読み込む前に行う
        from sentence_transformers import SentenceTransformer

        self._torch_dtype = resolve_dtype(settings.dtype, self._device)
        self._accelerator = accelerator_name(self._device)
        self._dims = settings.dims
        self._gate = PriorityGate()
        rx9060xt = False
        if self._accelerator == "amd-rocm" and settings.model_id == "google/embeddinggemma-2":
            gpu = torch.cuda.get_device_properties(self._device)
            # WSL2 で予約領域を差し引いた値でも 16 GB 版を識別する。
            rx9060xt = gpu.name == "AMD Radeon RX 9060 XT" and gpu.total_memory >= 15 * 2**30
        if rx9060xt and not (os.environ.get("OMP_NUM_THREADS") or os.environ.get("MKL_NUM_THREADS")):
            # CPU 前処理と動画の復号に CPU の実行枠を残す。
            torch.set_num_threads(min(torch.get_num_threads(), 2))

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
        # encode() は呼び出しごとに self.to(device) と self.eval() を全モジュールに実行するため、
        # 起動時に 1 回だけ設定して毎回の走査オーバーヘッドを省く。
        self._target_device = torch.device(self._device)
        self._model.to(self._target_device)
        self._model.eval()
        self._processor_lock = Lock() if rx9060xt else nullcontext()
        self._query_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rocm-search") if rx9060xt else None
        self._query_cache = lru_cache(maxsize=256)(self._encode_query) if rx9060xt else None
        self._use_pin_memory = bool(self._target_device.type == "cuda" and torch.cuda.is_available())
        self._modalities = list(getattr(self._model, "modalities", []) or [])
        self._apply_image_token_limit(settings.image_max_tokens)
        self._resolved_prompts: dict[str | None, str | None] = {
            name: self._model._resolve_prompt(None, name) for name in ("query", "document", None)
        }
        torch.set_grad_enabled(False)
        self._batch = resolve_embed_batch(settings.embed_batch, self._device)
        self._compiled = False
        self._accelerated_warmed_up = False
        self._text_graphs: accel.TextGraphs | None = None
        # 窓(複数の画像・音声の並び)は、同時に来た要求をまとめて 1 回で推論する(batch_size が 2 以上のとき)。
        # 前処理(CPU)と推論(GPU)は別のスレッドで動かし、推論中に次の回の前処理を済ませる。
        # 高優先(検索)が待機中なら、追加の取り込みを待たずに現在のバッチをすぐ解放する。
        self._parts_batcher = (
            MicroBatcher(
                self._run_windows,
                self._batch,
                prepare=self._prepare_windows,
                should_yield=self._gate.has_high_waiting,
            )
            if self._batch > 1
            else None
        )
        # 検索の文章も、同時に来たものをまとめる。まとめても float32 の結果との類似度は 1 件ずつと同じ(最小 0.99997)。
        # 同時に検索が来ているとき(前の回に 2 件以上まとまったとき)だけ、次の文章を 3 ms 待つ
        # (検索結果を返してから次の要求が届くまでの間を待つため。HTTP 経由・同時 4 件で 1 ms: 205 件/秒、3 ms: 207、
        # 5 ms: 175。同時 8 件で 261 / 273 / 329。RTX 3060 で実測)。単独の検索は待たない(待つと、その分だけ遅くなる)。
        # 文章の前処理は 1 ms 未満のため、窓と違い前処理を別のスレッドに分けない
        # (分けると、前処理が届いた順に 1 件ずつ取り出してしまい、まとまらない)
        self._text_batcher = (
            MicroBatcher(
                lambda texts: self._run_texts(self._prepare_texts(texts)),
                TEXT_BATCH,
                wait_sec=TEXT_WAIT_SEC,
                wait_only_when_busy=True,
            )
            if self._batch > 1 and self._accelerator == "nvidia-cuda"
            else None
        )
        if self._accelerator == "nvidia-cuda":  # 起動時の準備で上のまとめ処理も使うため、作った後に行う
            self._setup_acceleration(settings)
        log.info("モデルの読み込みが完了しました。対応する入力: %s、まとめる件数: %d", self._modalities, self._batch)

    # ------------------------------------------------------------------ 高速化(embedders/accel.py)
    def _setup_acceleration(self, settings: Settings) -> None:
        """NVIDIA の GPU で、結果を変えない(float32 の結果に近づける)最適化を有効にします。"""
        model = self._model[0].model
        accel.trim_image_padding(model)
        mode = settings.torch_compile
        if mode == "off":
            return
        if not accel.compiler_available():
            log.warning(
                "C コンパイラが見つからないため、torch.compile を使わずに動きます(推論が約 1.4 倍遅くなります)。"
                "cuda イメージには含まれています"
            )
            return
        accel.compile_layers(model)
        t0 = time.perf_counter()
        log.info("推論の準備(torch.compile)を行います。初回は 1 分ほどかかります(結果は DATA_DIR/cache/torch に保存)")
        try:
            self._warmup(settings)
        except Exception:
            # compile に失敗しても、通常の計算で動き続ける(遅くなるだけで、結果は同じ)
            log.warning("torch.compile の準備に失敗したため、使わずに動きます", exc_info=True)
            accel.uncompile_layers(model)
            return
        self._compiled = True
        self._accelerated_warmed_up = True
        self._install_text_graphs()
        log.info("推論の準備が完了しました(%.1f 秒)", time.perf_counter() - t0)

    def _install_text_graphs(self) -> None:
        """文章の推論を CUDA Graph に記録します。失敗しても、記録を使わずに動き続ける(遅くなるだけ)。"""
        graphs = accel.TextGraphs(self._model[0].model.language_model, TEXT_GRAPH_BATCHES, TEXT_GRAPH_LENGTHS)
        try:
            graphs.install()
            # 記録を使った推論で、正しく計算できることを確かめる(NaN なども _to_numpy で検出する)
            self.embed_texts(["warmup"])
            self._run_texts(self._prepare_texts(["warmup", "a longer warmup sentence for padding"]))
        except Exception:
            log.warning("文章の推論の CUDA Graph を使わずに動きます(準備に失敗しました)", exc_info=True)
            graphs.uninstall()
            return
        self._text_graphs = graphs

    def _warmup(self, settings: Settings) -> None:
        """よく使う形の入力で一度ずつ推論し、compile を起動時に済ませます(最初の検索・取り込みを待たせないため)。"""
        from ..media import output_size

        w, h = output_size(1920, 1080, settings.image_max_side)  # 16:9 の映像を縮小した大きさ
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        parts = [ImagePart(frame) for _ in range(max(settings.frames_per_window, 1))]
        self.embed_texts(["warmup"])
        self.embed_images([frame], high=True)
        self.embed_audio(np.zeros(SAMPLE_RATE * max(settings.audio_chunk_sec, 1), dtype=np.float32))
        self._run_windows(self._prepare_windows([parts]))
        if self._batch > 1:
            self._run_windows(self._prepare_windows([parts] * self._batch))
            self.embed_images([frame] * self._batch)
            # 長さの違う文章をまとめた形(詰め物とマスクがある)。検索の文章をまとめたときに使う
            self._run_texts(self._prepare_texts(["warmup", "a longer warmup sentence for padding"] * (TEXT_BATCH // 2)))

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
        if tensor.device.type != "cpu":
            tensor = tensor.cpu()
        if tensor.dtype != self._torch.float32:
            tensor = tensor.float()
        arr = tensor.numpy()
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

    def _preprocess_inputs(
        self,
        inputs: list[Any],
        prompt_name: str | None = None,
        pin: bool = False,
    ) -> _PreparedBatch:
        """CPU 前処理(トークナイズ・画像/音声のテンソル化・任意の pin_memory)を実行します。

        SentenceTransformer.encode() と同じ長さ順ソート・プロンプト付与を保つことで、
        出力ベクトルのビット一致を維持したまま PriorityGate の外へ前処理を分離します。
        """
        count = len(inputs)
        restore_idx: np.ndarray | None = None
        if count > 1:
            order = np.argsort([-self._model._input_length(sen) for sen in inputs])
            if self._model._can_flatten_inputs():
                order = self._model._interleave_sorted_indices(order)
            sorted_inputs = [inputs[idx] for idx in order]
            restore_idx = np.argsort(order)
        else:
            sorted_inputs = inputs
        prompt = self._resolved_prompts.get(prompt_name)
        with self._processor_lock:
            features = self._model.preprocess(sorted_inputs, prompt=prompt)
        if pin:
            features = pin_features(features)
        return _PreparedBatch(features=features, restore_idx=restore_idx, count=count)

    def _forward_prepared(self, prepared: _PreparedBatch, high: bool) -> np.ndarray:
        """前処理済みのテンソルに対してモデル本体の forward のみを PriorityGate 内で実行します。"""
        with self._gate.acquire(high=high):
            if high and self._query_worker is not None:
                # 呼び出し元にかかわらず、検索の GPU 実行と結果の転送に同じスレッドを使う。
                arr = self._query_worker.submit(self._forward_on_device, prepared.features).result()
            else:
                arr = self._forward_on_device(prepared.features)
        arr = arr.reshape(prepared.count, -1)
        if prepared.restore_idx is not None:
            arr = arr[prepared.restore_idx]
        return finalize(arr, self._dims)

    def _forward_on_device(self, features: dict[str, Any]) -> np.ndarray:
        with self._torch.inference_mode():
            features = move_features(features, self._target_device, non_blocking=self._use_pin_memory)
            return self._to_numpy(self._model(features)["sentence_embedding"])

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

    def warmup(self) -> dict[str, Any]:
        """起動時のウォームアップ。NVIDIA の高速化初期化で既に済ませていれば重複実行を省きます。"""
        if self._accelerated_warmed_up:
            return {"ok": True, "skipped": False, "modalities": ["text", "image", "audio"], "elapsed_ms": 0.0}
        return super().warmup()

    def embed_texts(self, texts: Sequence[str], kind: str = "query") -> np.ndarray:
        items = list(texts)
        if not items:
            return np.zeros((0, self._dims), dtype=np.float32)
        if kind == "query" and self._text_batcher is not None:
            return np.stack(self._text_batcher.run_many(items))
        if kind == "query" and self._query_cache is not None and len(items) == 1 and len(items[0]) <= 1024:
            return self._query_cache(items[0]).copy()
        prompt = "query" if kind == "query" else "document"
        high = kind == "query"
        step = max(self._batch, self._TEXT_BATCH_SIZE)
        if len(items) <= step:
            prepared = self._preprocess_inputs(items, prompt_name=prompt, pin=self._use_pin_memory)
            return self._forward_prepared(prepared, high=high)
        # 複数チャンクに分かれる場合も、encode() と同じく全体で長さ順に揃えてから分割する
        order = np.argsort([-self._model._input_length(sen) for sen in items])
        sorted_items = [items[idx] for idx in order]
        chunks = []
        for i in range(0, len(sorted_items), step):
            prepared = self._preprocess_inputs(sorted_items[i : i + step], prompt_name=prompt, pin=self._use_pin_memory)
            chunks.append(self._forward_prepared(prepared, high=high))
        merged = np.concatenate(chunks, axis=0)
        return merged[np.argsort(order)]

    def _encode_query(self, query: str) -> np.ndarray:
        return self._forward_prepared(self._prepare_texts([query]), high=True)

    def _prepare_texts(self, texts: list[str]) -> _PreparedBatch:
        return self._preprocess_inputs(texts, prompt_name="query", pin=self._use_pin_memory)

    def _run_texts(self, prepared: _PreparedBatch) -> list[np.ndarray]:
        return list(self._forward_prepared(prepared, high=True).reshape(prepared.count, -1))

    def embed_images(self, images: Sequence[np.ndarray], high: bool = False) -> np.ndarray:
        """画像を batch_size 枚ずつ、1 回の推論でまとめてベクトルにします(1 なら 1 枚ずつ)。"""
        if not images:
            return np.zeros((0, self._dims), dtype=np.float32)
        vectors = []
        for i in range(0, len(images), self._batch):
            chunk = [{"image": self._prepare_image(im)} for im in images[i : i + self._batch]]
            prepared = self._preprocess_inputs(chunk, prompt_name=None, pin=self._use_pin_memory)
            vectors.append(self._forward_prepared(prepared, high=high))
        return vectors[0] if len(vectors) == 1 else np.concatenate(vectors, axis=0)

    def embed_audio(self, pcm: np.ndarray, high: bool = False) -> np.ndarray:
        audio_f32 = np.asarray(pcm, dtype=np.float32)
        prepared = self._preprocess_inputs([{"audio": audio_f32}], prompt_name=None, pin=self._use_pin_memory)
        return self._forward_prepared(prepared, high=high)[0]

    def embed_parts(self, parts: Sequence[Part], high: bool = False) -> np.ndarray:
        """並べた順序のまま、1回の推論で1ベクトルにします。

        Google AI Edge Gallery と同じ「時刻 → 音声 → 時刻 → 画像 …」の交互配置もここで扱えます。
        batch_size が 2 以上なら、他のスレッドから同時に来た窓とまとめて推論します(検索の要求は待たせずに直接行う)。
        """
        if self._parts_batcher is not None and not high:
            return self._parts_batcher.submit(parts).result()
        return self._run_windows(self._prepare_windows([parts]), high=high)[0]

    def embed_parts_many(self, items: Sequence[Sequence[Part]], high: bool = False) -> list[np.ndarray]:
        if self._parts_batcher is not None and not high:
            return self._parts_batcher.run_many(list(items))
        return [self._run_windows(self._prepare_windows([parts]), high=high)[0] for parts in items]

    def prepare_parts_batch(self, items: Sequence[Sequence[Part]]) -> _PreparedBatch:
        """複数窓のメッセージ構築とテンソル化(および GPU 向け pin_memory)を前段スレッドで行います。"""
        messages = [self._message(parts) for parts in items]
        return self._preprocess_inputs(messages, prompt_name=None, pin=self._use_pin_memory)

    def embed_prepared_parts(self, prepared: _PreparedBatch, high: bool = False) -> list[np.ndarray]:
        """prepare_parts_batch() でテンソル化済みの入力を推論します。"""
        vectors = self._forward_prepared(prepared, high=high)
        return list(vectors.reshape(prepared.count, -1))

    def _message(self, parts: Sequence[Part]) -> list[dict]:
        content: list[dict] = []
        for part in parts:
            if isinstance(part, TextPart):
                content.append({"type": "text", "text": part.text})
            elif isinstance(part, ImagePart):
                content.append({"type": "image", "image": self._prepare_image(part.image)})
            elif isinstance(part, AudioPart):
                content.append({"type": "audio", "audio": np.asarray(part.pcm, dtype=np.float32)})
            else:
                raise TypeError(f"未対応の部品です: {type(part)!r}")
        return [{"role": "user", "content": content}]

    def _prepare_windows(self, items: list[Sequence[Part]]) -> _PreparedBatch:
        """複数の窓の前処理(CPU)。まとめても結果が変わらないことを実測(コサイン類似度 1.00000)。"""
        return self.prepare_parts_batch(items)

    def _run_windows(self, prepared: _PreparedBatch, high: bool = False) -> list[np.ndarray]:
        """前処理済みの複数の窓を、1 回の推論でベクトルにします。"""
        return self.embed_prepared_parts(prepared, high=high)

    def close(self) -> None:
        if self._parts_batcher is not None:
            self._parts_batcher.close()
        if self._text_batcher is not None:
            self._text_batcher.close()
        if self._query_worker is not None:
            self._query_worker.shutdown()
        if self._query_cache is not None:
            self._query_cache.cache_clear()
