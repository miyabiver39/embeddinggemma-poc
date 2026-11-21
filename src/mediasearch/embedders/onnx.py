"""ONNX Runtime による CPU 推論の実装(`onnx-community/embeddinggemma-2-ONNX`)。

`google/embeddinggemma-2` と同じモデル・同じ前処理(`EmbeddingGemma2Processor`)を使い、
推論エンジンだけを PyTorch から ONNX Runtime へ置き換えます。

- 既定の精度(`ONNX_PRECISION=fp32`)では、PyTorch FP32 とのコサイン類似度が全モダリティ
  (テキスト・文書・画像・動画窓・音声・TAV)で 1.000000 一致し、GPU(BF16)で作った
  ベクトル DB ともそのまま互換性を保てます(再インデックス不要)。
- 同じ大きさの画像だけのときは、画像エンコーダに渡す前に末尾の詰め物パッチ(`(-1, -1)`)を
  除去します(数値誤差 0 のまま、`vision_encoder` の計算量を削減)。
- 推論の順番待ちは `LocalEmbedder` と同じく `PriorityGate` で制御し、取り込み中も検索が
  1 窓/1 チャンクの合間に最優先で割り込めます。
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..config import Settings, resolve_embed_batch
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

ONNX_PRECISIONS: dict[str, str] = {
    "fp32": "",
    "q8": "_quantized",
    "q4": "_q4",
    "fp16": "_fp16",
}

ONNX_DTYPE_NAMES: dict[str, str] = {
    "fp32": "float32",
    "q8": "int8",
    "q4": "q4",
    "fp16": "float16",
}

_DEFAULT_PROMPTS: dict[str | None, str] = {
    "query": "task: search result | query: ",
    "document": "title: none | text: ",
    None: "",
}

_PROCESSOR_PATTERNS = [
    "config.json",
    "config_sentence_transformers.json",
    "chat_template.jinja",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
]


def _cgroup_cpu_limit() -> int | None:
    """コンテナの CPU 上限(cgroup v2 / v1)を切り上げ整数で返します。無制限・取得不可なら None。"""
    try:
        v2 = Path("/sys/fs/cgroup/cpu.max")
        if v2.is_file():
            parts = v2.read_text(encoding="utf-8").strip().split()
            if len(parts) == 2 and parts[0] != "max":
                quota, period = int(parts[0]), int(parts[1])
                if quota > 0 and period > 0:
                    return max((quota + period - 1) // period, 1)
    except (OSError, ValueError):
        pass
    try:
        q = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text(encoding="utf-8").strip())
        p = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text(encoding="utf-8").strip())
        if q > 0 and p > 0:
            return max((q + p - 1) // p, 1)
    except (OSError, ValueError):
        pass
    return None


def resolve_onnx_threads(requested: int = 0) -> int:
    """ONNX Runtime のスレッド数(intra_op_num_threads)を決めます。

    明示指定(`ONNX_THREADS` > 0)と `OMP_NUM_THREADS` を優先します。
    自動(0)の場合、利用可能な論理 CPU が 4 以下なら全コア、5 以上なら 75% を割り当てます。
    Ryzen 7 5700X(8 コア / 16 スレッド)での実測では、8 スレッド(5.59 秒/窓)や 16 スレッド
    (5.06 秒/窓)より 12 スレッド(4.86 秒/窓・検索 28.1 ms)が最速でした。
    残り 25% の枠を ffmpeg 復号や HTTP/API の処理に残すことで競合も抑えます。
    """
    if requested > 0:
        return requested
    omp = os.environ.get("OMP_NUM_THREADS", "").strip()
    if omp:
        try:
            val = int(omp)
            if val > 0:
                return val
        except ValueError:
            pass
    cpus = os.cpu_count() or 1
    limit = _cgroup_cpu_limit()
    avail = min(cpus, limit) if limit is not None else cpus
    if avail <= 4:
        return max(avail, 1)
    return max(4, (avail * 3) // 4)


def trim_vision_padding(
    pixel_values: np.ndarray,
    position_ids: np.ndarray,
    pooling_kernel: int = 9,
) -> tuple[np.ndarray, np.ndarray]:
    """画像の詰め物パッチ(`(-1, -1)`)を、同じ大きさの画像だけのときに外します。

    `embedders/accel.py` の `trim_image_padding` と同じ条件で末尾の詰め物だけを外します。
    詰め物は注意機構で無視されるため数値誤差 0(コサイン類似度 1.000000)のまま、
    16:9 画像(2520 → 2376 パッチ)で `vision_encoder` の計算時間を短縮できます。
    """
    if position_ids.ndim != 3 or pixel_values.ndim != 3:
        return pixel_values, position_ids
    valid = (position_ids != -1).all(axis=-1)
    counts = valid.sum(axis=-1)
    if counts.size == 0:
        return pixel_values, position_ids
    n = int(counts[0])
    if (
        0 < n < valid.shape[1]
        and n % pooling_kernel == 0
        and bool((counts == n).all())
        and bool(valid[:, :n].all())
    ):
        return (
            np.ascontiguousarray(pixel_values[:, :n]),
            np.ascontiguousarray(position_ids[:, :n]),
        )
    return pixel_values, position_ids


def required_onnx_files(precision: str) -> tuple[str, str, str]:
    """精度に対応する 3 つの ONNX ファイル名(model / vision / audio)を返します。"""
    if precision not in ONNX_PRECISIONS:
        raise ValueError(f"ONNX_PRECISION は {tuple(ONNX_PRECISIONS)} のいずれかにしてください: {precision!r}")
    suffix = ONNX_PRECISIONS[precision]
    return (
        f"onnx/model{suffix}.onnx",
        f"onnx/vision_encoder{suffix}.onnx",
        f"onnx/audio_encoder{suffix}.onnx",
    )


def _bundled_onnx_dir() -> Path | None:
    marker = Path("/etc/mediasearch_onnx_dir")
    if marker.is_file():
        try:
            val = marker.read_text(encoding="utf-8").strip()
            if val:
                return Path(val)
        except OSError:
            pass
    return None


def resolve_onnx_dir(settings: Settings) -> Path:
    """ONNX モデルとプロセッサの配置先を解決します(同梱先を優先し、無ければ Hugging Face から取得)。"""
    model_rel, vision_rel, audio_rel = required_onnx_files(settings.onnx_precision)
    if settings.onnx_model_dir:
        root = Path(settings.onnx_model_dir)
        missing = [rel for rel in (model_rel, vision_rel, audio_rel) if not (root / rel).is_file()]
        if not missing:
            return root
        bundled = _bundled_onnx_dir()
        offline = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
        if bundled is not None and root == bundled and not offline:
            log.info(
                "同梱 ONNX フォルダ(%s)に ONNX_PRECISION=%s が無いため、Hugging Face(%s)から取得します",
                root,
                settings.onnx_precision,
                settings.onnx_model_id,
            )
        else:
            raise RuntimeError(
                f"ONNX_MODEL_DIR({root})に必要な ONNX ファイルが見つかりません: {', '.join(missing)}。"
                f"ONNX_PRECISION={settings.onnx_precision!r} に対応するファイルを配置するか、"
                "ONNX_PRECISION=fp32(同梱版)または HF_HUB_OFFLINE=0 でオンライン取得を指定してください。"
            )

    from huggingface_hub import snapshot_download

    patterns = [
        *_PROCESSOR_PATTERNS,
        f"{model_rel}*",
        f"{vision_rel}*",
        f"{audio_rel}*",
    ]
    try:
        path = snapshot_download(
            settings.onnx_model_id,
            revision=settings.onnx_model_revision or None,
            allow_patterns=patterns,
        )
    except Exception as exc:
        raise RuntimeError(
            f"ONNX モデル({settings.onnx_model_id})の取得に失敗しました: {exc}。"
            "オフライン環境では ONNX_MODEL_DIR に同梱・配置済みのフォルダを指定してください。"
        ) from exc
    root = Path(path)
    missing = [rel for rel in (model_rel, vision_rel, audio_rel) if not (root / rel).is_file()]
    if missing:
        raise RuntimeError(f"取得した ONNX モデル({root})に必要なファイルがありません: {', '.join(missing)}")
    return root


@dataclass
class _OnnxPrepared:
    """CPU 前処理(トークナイズ・画像パッチ化・音声特徴抽出)済みの ONNX 入力。"""

    input_ids: np.ndarray
    attention_mask: np.ndarray
    pixel_values: np.ndarray | None
    pixel_position_ids: np.ndarray | None
    input_features: np.ndarray | None
    input_features_mask: np.ndarray | None
    count: int


class OnnxEmbedder(Embedder):
    """ONNX Runtime(CPUExecutionProvider)で EmbeddingGemma 2 を推論する実装。"""

    _TEXT_CHUNK_SIZE = 4  # 文書の一括取り込み時に、検索の割り込み待ちを抑えるための最大件数

    def __init__(self, settings: Settings) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime がインストールされていません。`pip install -e '.[onnx]'` を実行するか、"
                "EMBEDDING_RUNTIME=pytorch を指定してください。"
            ) from exc

        from transformers import AutoProcessor

        self._settings = settings
        self._dims = settings.dims
        self._precision = settings.onnx_precision
        self._dtype_name = ONNX_DTYPE_NAMES[self._precision]
        self._batch = resolve_embed_batch(settings.embed_batch, "cpu")
        self._threads = resolve_onnx_threads(settings.onnx_threads)
        self._gate = PriorityGate()
        self._processor_lock = threading.Lock()
        self._modalities = ["text", "image", "audio", "video", "message"]

        onnx_dir = resolve_onnx_dir(settings)
        self._onnx_dir = onnx_dir
        model_rel, vision_rel, audio_rel = required_onnx_files(self._precision)

        if self._precision != "fp32":
            log.warning(
                "ONNX 量子化モデル(ONNX_PRECISION=%s)で起動します。FP32 / GPU(BF16)とのベクトルに"
                "差が生じるため、同じ DB への混在は避けてください。",
                self._precision,
            )

        log.info(
            "ONNX モデルを読み込みます: %s (dir=%s, precision=%s, threads=%d)",
            settings.onnx_model_id,
            onnx_dir,
            self._precision,
            self._threads,
        )

        # プロセッサは ONNX ディレクトリ同梱のファイルを優先し、無ければ MODEL_ID から読み込む
        offline = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
        if (onnx_dir / "config.json").is_file() and (onnx_dir / "tokenizer.json").is_file():
            self._processor = AutoProcessor.from_pretrained(str(onnx_dir), local_files_only=True)
        else:
            self._processor = AutoProcessor.from_pretrained(
                settings.model_id,
                revision=settings.model_revision or None,
                local_files_only=offline,
            )
        self._apply_image_token_limit(settings.image_max_tokens)
        self._pooling_kernel = self._read_pooling_kernel(onnx_dir)
        self._prompts = self._read_prompts(onnx_dir)

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = self._threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = ["CPUExecutionProvider"]
        self._sess_model = ort.InferenceSession(str(onnx_dir / model_rel), sess_options=opts, providers=providers)
        self._sess_vision = ort.InferenceSession(str(onnx_dir / vision_rel), sess_options=opts, providers=providers)
        self._sess_audio = ort.InferenceSession(str(onnx_dir / audio_rel), sess_options=opts, providers=providers)
        log.info(
            "ONNX モデルの読み込みが完了しました(precision=%s、threads=%d、まとめる件数: %d)",
            self._precision,
            self._threads,
            self._batch,
        )

    # ------------------------------------------------------------------ 内部処理
    @staticmethod
    def _read_pooling_kernel(onnx_dir: Path) -> int:
        cfg_path = onnx_dir / "config.json"
        if cfg_path.is_file():
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                k = int(cfg.get("vision_config", {}).get("pooling_kernel_size", 3))
                return max(k * k, 1)
            except (OSError, ValueError, TypeError):
                pass
        return 9

    @staticmethod
    def _read_prompts(onnx_dir: Path) -> dict[str | None, str]:
        prompts = dict(_DEFAULT_PROMPTS)
        st_cfg = onnx_dir / "config_sentence_transformers.json"
        if st_cfg.is_file():
            try:
                raw = json.loads(st_cfg.read_text(encoding="utf-8")).get("prompts") or {}
                for key in ("query", "document"):
                    if isinstance(raw.get(key), str):
                        prompts[key] = raw[key]
            except (OSError, ValueError, TypeError):
                pass
        return prompts

    def _apply_image_token_limit(self, limit: int) -> None:
        if limit <= 0:
            return
        try:
            self._processor.image_processor.max_soft_tokens = limit
            log.info("画像1枚あたりの上限トークンを %d にしました", limit)
        except Exception:
            log.warning("画像の上限トークンを設定できませんでした(既定値のまま動きます)", exc_info=True)

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        side = self._settings.image_max_side
        h, w = image.shape[:2]
        if side > 0 and max(h, w) > side:
            from PIL import Image

            pil = Image.fromarray(image)
            pil.thumbnail((side, side))
            return np.array(pil, dtype=np.uint8, copy=True)
        return image if image.flags.writeable else np.array(image, dtype=np.uint8, copy=True)

    def _to_prepared(self, feat: dict[str, Any], count: int) -> _OnnxPrepared:
        input_ids = np.ascontiguousarray(feat["input_ids"], dtype=np.int64)
        attention_mask = np.ascontiguousarray(feat["attention_mask"], dtype=np.int64)

        pv = feat.get("pixel_values")
        pos = feat.get("image_position_ids")
        if pv is not None and pos is not None:
            pv_arr = np.ascontiguousarray(pv, dtype=np.float32)
            pos_arr = np.ascontiguousarray(pos, dtype=np.int64)
            pv_arr, pos_arr = trim_vision_padding(pv_arr, pos_arr, self._pooling_kernel)
        else:
            pv_arr = pos_arr = None

        af = feat.get("input_features")
        am = feat.get("input_features_mask")
        if af is not None and am is not None:
            af_arr = np.ascontiguousarray(af, dtype=np.float32)
            am_arr = np.ascontiguousarray(am, dtype=bool)
        else:
            af_arr = am_arr = None

        return _OnnxPrepared(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pv_arr,
            pixel_position_ids=pos_arr,
            input_features=af_arr,
            input_features_mask=am_arr,
            count=count,
        )

    def _run_prepared(self, prepared: _OnnxPrepared, high: bool) -> np.ndarray:
        with self._gate.acquire(high=high):
            if prepared.pixel_values is not None and prepared.pixel_position_ids is not None:
                image_features = self._sess_vision.run(
                    ["image_features"],
                    {
                        "pixel_values": prepared.pixel_values,
                        "pixel_position_ids": prepared.pixel_position_ids,
                    },
                )[0]
            else:
                image_features = np.zeros((0, 512), dtype=np.float32)

            if prepared.input_features is not None and prepared.input_features_mask is not None:
                audio_features = self._sess_audio.run(
                    ["audio_features"],
                    {
                        "input_features": prepared.input_features,
                        "input_features_mask": prepared.input_features_mask,
                    },
                )[0]
            else:
                audio_features = np.zeros((0, 512), dtype=np.float32)

            video_features = np.zeros((0, 512), dtype=np.float32)
            out = self._sess_model.run(
                ["sentence_embedding"],
                {
                    "input_ids": prepared.input_ids,
                    "attention_mask": prepared.attention_mask,
                    "image_features": image_features,
                    "video_features": video_features,
                    "audio_features": audio_features,
                },
            )[0]

        arr = np.asarray(out, dtype=np.float32)
        if not np.isfinite(arr).all():
            raise RuntimeError(
                "ベクトルに NaN / 無限大が含まれています。精度が合っていない可能性があります。"
                "ONNX_PRECISION=fp32 または EMBEDDING_RUNTIME=pytorch で試してください。"
            )
        return finalize(arr.reshape(prepared.count, -1), self._dims)

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

    # ------------------------------------------------------------------ 公開 API
    def info(self) -> EmbedderInfo:
        return EmbedderInfo(
            backend="local",
            model_id=self._settings.model_id,
            dims=self._dims,
            dtype=self._dtype_name,
            device="cpu",
            accelerator="cpu",
            modalities=self._modalities,
            batch_size=self._batch,
            runtime="onnx",
        )

    def embed_texts(self, texts: Sequence[str], kind: str = "query") -> np.ndarray:
        items = list(texts)
        if not items:
            return np.zeros((0, self._dims), dtype=np.float32)
        prefix = self._prompts["query" if kind == "query" else "document"]
        prefixed = [prefix + t for t in items]
        high = kind == "query"
        step = max(self._batch, self._TEXT_CHUNK_SIZE)
        chunks: list[np.ndarray] = []
        for i in range(0, len(prefixed), step):
            batch_texts = prefixed[i : i + step]
            with self._processor_lock:
                feat = self._processor(text=batch_texts, return_tensors="np", padding=True)
            prepared = self._to_prepared(feat, count=len(batch_texts))
            chunks.append(self._run_prepared(prepared, high=high))
        return chunks[0] if len(chunks) == 1 else np.concatenate(chunks, axis=0)

    def embed_images(self, images: Sequence[np.ndarray], high: bool = False) -> np.ndarray:
        if not images:
            return np.zeros((0, self._dims), dtype=np.float32)
        vectors: list[np.ndarray] = []
        for i in range(0, len(images), self._batch):
            chunk = images[i : i + self._batch]
            msgs = [
                [{"role": "user", "content": [{"type": "image", "image": self._prepare_image(im)}]}]
                for im in chunk
            ]
            with self._processor_lock:
                feat = self._processor.apply_chat_template(
                    msgs, tokenize=True, return_dict=True, return_tensors="np", padding=True
                )
            prepared = self._to_prepared(feat, count=len(chunk))
            vectors.append(self._run_prepared(prepared, high=high))
        return vectors[0] if len(vectors) == 1 else np.concatenate(vectors, axis=0)

    def embed_audio(self, pcm: np.ndarray, high: bool = False) -> np.ndarray:
        audio_f32 = np.asarray(pcm, dtype=np.float32)
        msgs = [[{"role": "user", "content": [{"type": "audio", "audio": audio_f32}]}]]
        with self._processor_lock:
            feat = self._processor.apply_chat_template(msgs, tokenize=True, return_dict=True, return_tensors="np")
        prepared = self._to_prepared(feat, count=1)
        return self._run_prepared(prepared, high=high)[0]

    def prepare_parts_batch(self, items: Sequence[Sequence[Part]]) -> list[_OnnxPrepared]:
        """動画取り込みの Producer スレッドで複数窓の前処理(トークナイズ・パッチ化・詰め物除去)を行います。"""
        out: list[_OnnxPrepared] = []
        for parts in items:
            msgs = [self._message(parts)]
            with self._processor_lock:
                feat = self._processor.apply_chat_template(msgs, tokenize=True, return_dict=True, return_tensors="np")
            out.append(self._to_prepared(feat, count=1))
        return out

    def embed_prepared_parts(self, prepared: Any, high: bool = False) -> list[np.ndarray]:
        """prepare_parts_batch() で前処理済みの窓を推論します。"""
        return [self._run_prepared(item, high=high)[0] for item in prepared]

    def embed_parts(self, parts: Sequence[Part], high: bool = False) -> np.ndarray:
        prepared = self.prepare_parts_batch([parts])
        return self.embed_prepared_parts(prepared, high=high)[0]

    def embed_parts_many(self, items: Sequence[Sequence[Part]], high: bool = False) -> list[np.ndarray]:
        prepared = self.prepare_parts_batch(items)
        return self.embed_prepared_parts(prepared, high=high)

    def close(self) -> None:
        self._sess_model = None
        self._sess_vision = None
        self._sess_audio = None
