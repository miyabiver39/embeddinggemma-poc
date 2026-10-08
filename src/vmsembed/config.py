"""設定。

すべて環境変数から読み込みます。既定値のままで一発で動く値にしてあるので、
必須の設定はありません。環境変数の一覧は docs/operations.md を参照してください。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# 窓(ウィンドウ)設計のプリセット。
# Google AI Edge Gallery の Video Moments Finder のプリセットを参考にしています。
#   物体: 短い窓で、写っているものを探す
#   動作: やや長い窓で、動きを探す
#   会話: 長い窓で、音声(会話)も使って探す
PRESETS: dict[str, dict] = {
    "object": {"window_sec": 2, "frames_per_window": 2, "overlap_sec": 0, "include_audio": False},
    "action": {"window_sec": 4, "frames_per_window": 4, "overlap_sec": 0, "include_audio": False},
    "speech": {"window_sec": 6, "frames_per_window": 6, "overlap_sec": 0, "include_audio": True},
}

ROLES = ("all", "app", "compute")
BACKENDS = ("local", "remote", "dummy")


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value.strip() == "" else value.strip()


def _env_int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, "1" if default else "0").lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    """アプリ全体の設定。"""

    # --- 役割 ---
    role: str  # all: 取り込み+検索+推論 / app: 取り込み+検索 / compute: 推論だけ
    embedding_backend: str  # local: 自プロセスで推論 / remote: compute へ HTTP / dummy: 動作確認用
    embedding_url: str  # remote のときの compute の URL
    embedding_timeout_sec: float

    # --- モデル ---
    model_id: str
    device: str  # auto / cpu / cuda(NVIDIA と AMD ROCm) / xpu(Intel)
    dtype: str  # auto / float32 / bfloat16 / float16
    dims: int  # 出力次元(MRL で 768 / 512 / 256 / 128 に切り詰め)
    image_max_tokens: int  # 画像1枚あたりの上限トークン。0 ならモデルの既定
    image_max_side: int  # 推論前に縮小する画像の長辺(px)

    # --- 保存先 ---
    data_dir: Path

    # --- 窓設計(取り込みの既定値) ---
    window_sec: int
    frames_per_window: int
    overlap_sec: int
    include_audio: bool
    audio_chunk_sec: int  # 音声だけを取り込むときの区切り(秒)

    # --- 検索 ---
    top_k_default: int

    # --- サーバ ---
    host: str
    port: int

    @classmethod
    def from_env(cls) -> Settings:
        role = _env("ROLE", "all")
        if role not in ROLES:
            raise ValueError(f"ROLE は {ROLES} のいずれかにしてください: {role!r}")

        # 役割ごとの既定の推論方式: app は外部の compute を呼び、それ以外は自プロセスで推論する
        default_backend = "remote" if role == "app" else "local"
        backend = _env("EMBEDDING_BACKEND", default_backend)
        if backend not in BACKENDS:
            raise ValueError(f"EMBEDDING_BACKEND は {BACKENDS} のいずれかにしてください: {backend!r}")

        dims = _env_int("DIMS", 768)
        if dims not in (768, 512, 256, 128):
            raise ValueError("DIMS は 768 / 512 / 256 / 128 のいずれかにしてください")

        return cls(
            role=role,
            embedding_backend=backend,
            embedding_url=_env("EMBEDDING_URL", "http://localhost:8001").rstrip("/"),
            embedding_timeout_sec=_env_float("EMBEDDING_TIMEOUT_SEC", 300.0),
            model_id=_env("MODEL_ID", "google/embeddinggemma-2"),
            device=_env("DEVICE", "auto"),
            dtype=_env("DTYPE", "auto"),
            dims=dims,
            image_max_tokens=_env_int("IMAGE_MAX_TOKENS", 0),
            image_max_side=_env_int("IMAGE_MAX_SIDE", 448),
            data_dir=Path(_env("DATA_DIR", "/data")),
            window_sec=_env_int("WINDOW_SEC", PRESETS["object"]["window_sec"]),
            frames_per_window=_env_int("FRAMES_PER_WINDOW", PRESETS["object"]["frames_per_window"]),
            overlap_sec=_env_int("OVERLAP_SEC", 0),
            include_audio=_env_bool("INCLUDE_AUDIO", False),
            audio_chunk_sec=_env_int("AUDIO_CHUNK_SEC", 10),
            top_k_default=_env_int("TOP_K", 10),
            host=_env("HOST", "0.0.0.0"),
            port=_env_int("PORT", 8000),
        )
