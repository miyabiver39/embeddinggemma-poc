"""ONNX Runtime CPU 推論(embedders/onnx.py)の単体テスト(onnxruntime 未導入環境でも動作)。"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from mediasearch.config import Settings
from mediasearch.embedders import AudioPart, ImagePart, TextPart, should_use_onnx
from mediasearch.embedders.onnx import (
    OnnxEmbedder,
    required_onnx_files,
    resolve_onnx_dir,
    resolve_onnx_threads,
    trim_vision_padding,
)


def test_onnx_settings_validation(monkeypatch):
    monkeypatch.delenv("EMBEDDING_RUNTIME", raising=False)
    monkeypatch.delenv("ONNX_PRECISION", raising=False)
    monkeypatch.delenv("ONNX_THREADS", raising=False)
    s = Settings.from_env()
    assert s.embedding_runtime == "auto"
    assert s.onnx_precision == "fp32"
    assert s.onnx_threads == 0

    monkeypatch.setenv("EMBEDDING_RUNTIME", "ONNX")
    monkeypatch.setenv("ONNX_PRECISION", "Q8")
    monkeypatch.setenv("ONNX_THREADS", "12")
    s2 = Settings.from_env()
    assert s2.embedding_runtime == "onnx" and s2.onnx_precision == "q8" and s2.onnx_threads == 12

    monkeypatch.setenv("EMBEDDING_RUNTIME", "invalid")
    with pytest.raises(ValueError, match="EMBEDDING_RUNTIME"):
        Settings.from_env()

    monkeypatch.setenv("EMBEDDING_RUNTIME", "auto")
    monkeypatch.setenv("ONNX_PRECISION", "int4")
    with pytest.raises(ValueError, match="ONNX_PRECISION"):
        Settings.from_env()


def test_should_use_onnx_matrix(tmp_path, monkeypatch):
    onnx_dir = tmp_path / "onnx_bundle"
    onnx_dir.mkdir()
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    def _s(**env):
        for k in ("EMBEDDING_RUNTIME", "DEVICE", "ONNX_MODEL_DIR"):
            monkeypatch.delenv(k, raising=False)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        return Settings.from_env()

    # 明示指定
    assert should_use_onnx(_s(EMBEDDING_RUNTIME="onnx")) is True
    assert should_use_onnx(_s(EMBEDDING_RUNTIME="pytorch")) is False

    # auto: onnxruntime が無い場合は False
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None if name == "onnxruntime" else object())
    assert should_use_onnx(_s(EMBEDDING_RUNTIME="auto", ONNX_MODEL_DIR=str(onnx_dir))) is False

    # auto: onnxruntime あり + CPU + 同梱フォルダあり -> True
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr("mediasearch.embedders._gpu_available", lambda: False)
    assert should_use_onnx(_s(EMBEDDING_RUNTIME="auto", ONNX_MODEL_DIR=str(onnx_dir))) is True

    # auto: オフラインで ONNX_MODEL_DIR 空 -> False
    assert should_use_onnx(_s(EMBEDDING_RUNTIME="auto", ONNX_MODEL_DIR="")) is False

    # auto: GPU 明示指定 -> False
    assert should_use_onnx(_s(EMBEDDING_RUNTIME="auto", DEVICE="cuda", ONNX_MODEL_DIR=str(onnx_dir))) is False

    # auto: GPU 自動検出時は PyTorch(GPU)を優先 -> False、DEVICE=cpu 明示なら ONNX -> True
    monkeypatch.setattr("mediasearch.embedders._gpu_available", lambda: True)
    assert should_use_onnx(_s(EMBEDDING_RUNTIME="auto", DEVICE="auto", ONNX_MODEL_DIR=str(onnx_dir))) is False
    assert should_use_onnx(_s(EMBEDDING_RUNTIME="auto", DEVICE="cpu", ONNX_MODEL_DIR=str(onnx_dir))) is True


def test_trim_vision_padding_cases():
    def _pos(valid_counts: list[int], total: int = 36) -> np.ndarray:
        pos = np.full((len(valid_counts), total, 2), -1, dtype=np.int64)
        for i, n in enumerate(valid_counts):
            pos[i, :n, 0] = np.arange(n)
            pos[i, :n, 1] = np.arange(n)
        return pos

    pv = np.zeros((2, 36, 4), dtype=np.float32)
    pv_t, pos_t = trim_vision_padding(pv, _pos([27, 27], 36), pooling_kernel=9)
    assert pv_t.shape == (2, 27, 4) and pos_t.shape == (2, 27, 2) and pv_t.flags["C_CONTIGUOUS"]

    # 大きさ違い・割り切れない・詰め物なしはそのまま残す
    for counts in ([27, 18], [26, 26], [36, 36]):
        pv_k, pos_k = trim_vision_padding(pv, _pos(counts, 36), pooling_kernel=9)
        assert pv_k.shape == (2, 36, 4) and pos_k.shape == (2, 36, 2)

    # 詰め物が末尾にまとまっていない場合も残す
    pos_gap = _pos([27, 27], 36)
    pos_gap[0, 5] = -1
    pos_gap[0, 27] = 27
    pv_g, _ = trim_vision_padding(pv, pos_gap, pooling_kernel=9)
    assert pv_g.shape == (2, 36, 4)


def test_resolve_onnx_threads(monkeypatch):
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.setattr("mediasearch.embedders.onnx._cgroup_cpu_limit", lambda: None)
    assert resolve_onnx_threads(6) == 6

    monkeypatch.setenv("OMP_NUM_THREADS", "5")
    assert resolve_onnx_threads(0) == 5
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)

    monkeypatch.setattr("os.cpu_count", lambda: 16)
    assert resolve_onnx_threads(0) == 12

    monkeypatch.setattr("os.cpu_count", lambda: 4)
    assert resolve_onnx_threads(0) == 4

    # cgroup の上限が CPU 数より小さい場合は cgroup に従う
    monkeypatch.setattr("os.cpu_count", lambda: 16)
    monkeypatch.setattr("mediasearch.embedders.onnx._cgroup_cpu_limit", lambda: 4)
    assert resolve_onnx_threads(0) == 4


def test_required_onnx_files_and_missing_dir(tmp_path, monkeypatch):
    assert required_onnx_files("fp32") == (
        "onnx/model.onnx",
        "onnx/vision_encoder.onnx",
        "onnx/audio_encoder.onnx",
    )
    assert required_onnx_files("q8") == (
        "onnx/model_quantized.onnx",
        "onnx/vision_encoder_quantized.onnx",
        "onnx/audio_encoder_quantized.onnx",
    )
    assert required_onnx_files("q4") == (
        "onnx/model_q4.onnx",
        "onnx/vision_encoder_q4.onnx",
        "onnx/audio_encoder_q4.onnx",
    )

    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("ONNX_MODEL_DIR", str(tmp_path / "empty_onnx"))
    (tmp_path / "empty_onnx").mkdir()
    with pytest.raises(RuntimeError, match="ONNX_MODEL_DIR"):
        resolve_onnx_dir(Settings.from_env())

    # 同梱フォルダに無い精度(q8)を要求した場合: オフラインなら拒否、オンラインなら HF 取得へフォールバック
    bundled = tmp_path / "bundled_onnx"
    (bundled / "onnx").mkdir(parents=True)
    for rel in required_onnx_files("fp32"):
        (bundled / rel).write_bytes(b"fp32")
    downloaded = tmp_path / "hf_q8"
    (downloaded / "onnx").mkdir(parents=True)
    for rel in required_onnx_files("q8"):
        (downloaded / rel).write_bytes(b"q8")

    monkeypatch.setattr("mediasearch.embedders.onnx._bundled_onnx_dir", lambda: bundled)
    monkeypatch.setenv("ONNX_MODEL_DIR", str(bundled))
    monkeypatch.setenv("ONNX_PRECISION", "q8")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with pytest.raises(RuntimeError, match="ONNX_MODEL_DIR"):
        resolve_onnx_dir(Settings.from_env())

    fake_hf = types.ModuleType("huggingface_hub")
    fake_hf.snapshot_download = lambda *_a, **_kw: str(downloaded)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    assert resolve_onnx_dir(Settings.from_env()) == downloaded


def test_onnx_embedder_full_flow_with_fake_sessions(tmp_path, monkeypatch):
    """擬似 ONNX セッションと擬似プロセッサで OnnxEmbedder の全 API とパディング除去・NaN 検知を確認。"""
    onnx_dir = tmp_path / "onnx_bundle"
    (onnx_dir / "onnx").mkdir(parents=True)
    for rel in required_onnx_files("fp32"):
        (onnx_dir / rel).write_bytes(b"fake")
    (onnx_dir / "config.json").write_text(
        json.dumps({"vision_config": {"pooling_kernel_size": 3}}), encoding="utf-8"
    )
    (onnx_dir / "config_sentence_transformers.json").write_text(
        json.dumps({"prompts": {"query": "Q: ", "document": "D: "}}), encoding="utf-8"
    )
    (onnx_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    vision_shapes: list[tuple[int, ...]] = []
    text_seen: list[list[str]] = []
    emit_nan = {"flag": False}

    class _FakeSession:
        def __init__(self, path: str, **_kw) -> None:
            self.name = Path(path).name

        def run(self, _outputs, feeds: dict[str, np.ndarray]):
            if "vision" in self.name:
                pv = feeds["pixel_values"]
                vision_shapes.append(tuple(pv.shape))
                return [np.ones((pv.shape[0] * (pv.shape[1] // 9), 512), dtype=np.float32)]
            if "audio" in self.name:
                af = feeds["input_features"]
                return [np.ones((af.shape[0] * 4, 512), dtype=np.float32)]
            bsz = feeds["input_ids"].shape[0]
            out = np.ones((bsz, 768), dtype=np.float32)
            if emit_nan["flag"]:
                out[0, 0] = np.nan
            return [out]

    class _FakeSessionOptions:
        intra_op_num_threads = 0
        inter_op_num_threads = 0
        graph_optimization_level = None

    fake_ort = types.ModuleType("onnxruntime")
    fake_ort.InferenceSession = _FakeSession
    fake_ort.SessionOptions = _FakeSessionOptions
    fake_ort.GraphOptimizationLevel = types.SimpleNamespace(ORT_ENABLE_ALL=99)

    class _FakeProcessor:
        def __init__(self) -> None:
            self.image_processor = types.SimpleNamespace(max_soft_tokens=280)

        def __call__(self, text, **_kw):
            text_seen.append(list(text))
            return {
                "input_ids": np.ones((len(text), 8), dtype=np.int64),
                "attention_mask": np.ones((len(text), 8), dtype=np.int64),
            }

        def apply_chat_template(self, msgs, **_kw):
            has_img = any(c["type"] == "image" for m in msgs for c in m[0]["content"])
            has_aud = any(c["type"] == "audio" for m in msgs for c in m[0]["content"])
            out: dict[str, np.ndarray] = {
                "input_ids": np.ones((len(msgs), 16), dtype=np.int64),
                "attention_mask": np.ones((len(msgs), 16), dtype=np.int64),
            }
            if has_img:
                n_img = sum(1 for m in msgs for c in m[0]["content"] if c["type"] == "image")
                out["pixel_values"] = np.zeros((n_img, 36, 768), dtype=np.float32)
                pos = np.full((n_img, 36, 2), -1, dtype=np.int64)
                pos[:, :27, :] = 1  # 末尾 9 パッチが詰め物 -> 27 パッチへ除去されるはず
                out["image_position_ids"] = pos
            if has_aud:
                n_aud = sum(1 for m in msgs for c in m[0]["content"] if c["type"] == "audio")
                out["input_features"] = np.zeros((n_aud, 20, 128), dtype=np.float32)
                out["input_features_mask"] = np.ones((n_aud, 20), dtype=bool)
            return out

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoProcessor = types.SimpleNamespace(from_pretrained=lambda *_a, **_kw: _FakeProcessor())

    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("ONNX_MODEL_DIR", str(onnx_dir))
    monkeypatch.setenv("DIMS", "256")
    monkeypatch.setenv("IMAGE_MAX_TOKENS", "140")

    emb = OnnxEmbedder(Settings.from_env())
    try:
        info = emb.info()
        assert info.runtime == "onnx" and info.model_id == "google/embeddinggemma-2"
        assert info.dims == 256 and info.dtype == "float32" and info.accelerator == "cpu"
        assert emb._processor.image_processor.max_soft_tokens == 140

        # テキスト(query / document / 空)
        assert emb.embed_texts([]).shape == (0, 256)
        vt = emb.embed_texts(["hello", "world"], kind="query")
        assert vt.shape == (2, 256) and np.allclose(np.linalg.norm(vt, axis=-1), 1.0, atol=1e-5)
        assert text_seen[-1] == ["Q: hello", "Q: world"]
        emb.embed_texts(["doc"], kind="document")
        assert text_seen[-1] == ["D: doc"]

        # 画像(詰め物 36 -> 27 に除去されていること)
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        vi = emb.embed_images([img], high=True)
        assert vi.shape == (1, 256) and vision_shapes[-1] == (1, 27, 768)

        # 音声(1 次元ベクトル)
        pcm = np.zeros(8000, dtype=np.float32)
        va = emb.embed_audio(pcm, high=False)
        assert va.shape == (256,)

        # 動画窓(TAV)の Producer-Consumer 前処理 + 推論
        parts = [TextPart("00:00"), AudioPart(pcm), TextPart("00:00"), ImagePart(img)]
        prep = emb.prepare_parts_batch([parts])
        vecs = emb.embed_prepared_parts(prep, high=False)
        assert len(vecs) == 1 and vecs[0].shape == (256,)

        # ウォームアップとメトリクス
        warm = emb.warmup()
        assert warm["ok"] is True and warm["modalities"] == ["text", "image", "audio"]
        assert emb.metrics()["gate"]["calls"]["high"] >= 2

        # NaN 検知
        emit_nan["flag"] = True
        with pytest.raises(RuntimeError, match="NaN"):
            emb.embed_texts(["boom"])
    finally:
        emb.close()
