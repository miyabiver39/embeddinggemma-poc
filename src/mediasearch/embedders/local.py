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
import time
from collections.abc import Sequence

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
        if settings.torch_compile != "off":
            # compile の結果の保存先は、torch(inductor)を読み込む前に決める必要がある(後から変えても効かない)
            accel.setup_compile_cache(settings.data_dir)
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
        # encode() は呼ぶたびにモデル全体の .to(device) と .eval() を行い、
        # 1 回あたり十数〜30 ms かかる(RTX 3060 で実測)。
        # ここで 1 回だけ行い、推論では前処理 → 転送 → 計算を直接呼ぶ(encode と同じ手順で、結果は完全に一致する)
        self._model.eval()
        self._batch = resolve_embed_batch(settings.embed_batch, self._device)
        self._compiled = False
        self._text_graphs: accel.TextGraphs | None = None
        # 窓(複数の画像・音声の並び)は、同時に来た要求をまとめて 1 回で推論する(batch_size が 2 以上のとき)。
        # 前処理(CPU)と推論(GPU)は別のスレッドで動かし、推論中に次の回の前処理を済ませる
        self._parts_batcher = (
            MicroBatcher(self._run_windows, self._batch, prepare=self._prepare_windows) if self._batch > 1 else None
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

    def _preprocess(self, inputs: list, prompt_name: str | None = None):
        """前処理(文字の分割・画像の変換。CPU)。推論の順番待ちの外で行い、その間も GPU を使えるようにする。"""
        prompt = self._model._resolve_prompt(None, prompt_name)
        return self._model.preprocess(inputs, prompt=prompt)

    def _forward(self, features, high: bool) -> np.ndarray:
        """前処理済みの入力を GPU に送って計算し、(件数, 次元) の配列を返します。"""
        import torch
        from sentence_transformers.util import batch_to_device

        with self._gate.acquire(high=high), torch.inference_mode():
            out = self._model(batch_to_device(features, self._model.device))["sentence_embedding"]
            arr = self._to_numpy(out)
        return arr.reshape(-1, arr.shape[-1])

    def embed_texts(self, texts: Sequence[str], kind: str = "query") -> np.ndarray:
        if kind == "query" and self._text_batcher is not None:
            return np.stack(self._text_batcher.run_many(list(texts)))
        prompt = "query" if kind == "query" else "document"
        return finalize(self._forward(self._preprocess(list(texts), prompt), high=True), self._dims)

    def _prepare_texts(self, texts: list[str]):
        return len(texts), self._preprocess(texts, "query")

    def _run_texts(self, prepared) -> list[np.ndarray]:
        count, features = prepared
        return list(finalize(self._forward(features, high=True), self._dims).reshape(count, -1))

    def embed_images(self, images: Sequence[np.ndarray], high: bool = False) -> np.ndarray:
        """画像を batch_size 枚ずつ、1 回の推論でまとめてベクトルにします(1 なら 1 枚ずつ)。"""
        vectors = []
        for i in range(0, len(images), self._batch):
            chunk = [{"image": self._prepare_image(im)} for im in images[i : i + self._batch]]
            vectors.append(self._forward(self._preprocess(chunk), high=high))
        return finalize(np.concatenate(vectors), self._dims)

    def embed_audio(self, pcm: np.ndarray, high: bool = False) -> np.ndarray:
        features = self._preprocess([{"audio": pcm.astype(np.float32)}])
        return finalize(self._forward(features, high=high), self._dims)

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

    def _prepare_windows(self, items: list[Sequence[Part]]):
        """複数の窓の前処理(CPU)。まとめても結果が変わらないことを実測(コサイン類似度 1.00000)。"""
        return len(items), self._preprocess([self._message(parts) for parts in items])

    def _run_windows(self, prepared, high: bool = False) -> list[np.ndarray]:
        """前処理済みの複数の窓を、1 回の推論でベクトルにします。"""
        count, features = prepared
        vectors = finalize(self._forward(features, high=high), self._dims)
        return list(vectors.reshape(count, -1))

    def close(self) -> None:
        if self._parts_batcher is not None:
            self._parts_batcher.close()
        if self._text_batcher is not None:
            self._text_batcher.close()
