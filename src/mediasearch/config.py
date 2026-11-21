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


def resolve_embed_batch(setting: str, device: str) -> int:
    """EMBED_BATCH を件数にします。auto は、CPU では 1(まとめない)、GPU では 4。

    CPU では、まとめても速くならず、窓(画像 2 枚)では 1 件ずつより遅かった(4 vCPU で 39.8 秒 → 51.2 秒を実測)。
    GPU では、まとめて計算したほうが処理量が増えるのが一般的なため 4 にしている(実機では未計測)。
    """
    setting = (setting or "auto").strip().lower()
    if setting == "auto":
        return 1 if device.startswith("cpu") else 4
    try:
        value = int(setting)
    except ValueError as exc:
        raise ValueError(f"EMBED_BATCH は auto か 1 以上の整数にしてください: {setting!r}") from exc
    if not 1 <= value <= 64:
        raise ValueError("EMBED_BATCH は 1〜64 にしてください")
    return value


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value.strip() == "" else value.strip()


def _env_int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, "1" if default else "0").lower() in ("1", "true", "yes", "on")


def _log_format(value: str) -> str:
    value = value.lower()
    if value not in ("text", "json"):
        raise ValueError(f"LOG_FORMAT は text か json を指定してください(指定: {value})")
    return value


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
    model_revision: str  # モデルの版(Hugging Face のコミット)。空なら最新。イメージでは同梱した版に固定
    device: str  # auto / cpu / cuda(NVIDIA と AMD ROCm) / xpu(Intel)
    dtype: str  # auto / float32 / bfloat16 / float16
    dims: int  # 出力次元(MRL で 768 / 512 / 256 / 128 に切り詰め)
    image_max_tokens: int  # 画像1枚あたりの上限トークン。0 ならモデルの既定
    image_max_side: int  # 推論前に縮小する画像の長辺(px)
    embed_batch: str  # まとめて推論する最大の件数(auto / 整数)。auto は CPU で 1、GPU で 4
    torch_cache_dir: Path  # PyTorch(Inductor / Triton 等)のコンパイルキャッシュ保存先(既定 DATA_DIR/cache/torch)
    warmup_on_start: bool  # 起動時(/readyz が Ready になる直前)にダミー入力でウォームアップ推論を行う
    ingest_prefetch: int  # 動画取り込みの Producer-Consumer キューの上限(有界にしてメモリ肥大化を防ぐ)
    ingest_workers: int  # 同時に取り込むファイルの数(復号・前処理を並べる。推論は PriorityGate で 1 回ずつ)
    ffmpeg_hwaccel: str  # 動画の復号に使う GPU(auto / none / cuda / vaapi / qsv)
    ffmpeg_hwaccel_device: str  # GPU 復号のデバイス(vaapi なら /dev/dri/renderD128 など。空なら自動)
    ffmpeg_skip_frames: str  # 復号を省く範囲(none / noref / keyframes。media.DECODE_SKIP)

    # --- 保存先 ---
    data_dir: Path
    store_media: bool  # 偽なら、アップロードした動画・画像とサムネイルを保存しない(検索結果は情報だけ)

    # --- 窓設計(取り込みの既定値) ---
    window_sec: int
    frames_per_window: int
    overlap_sec: int
    include_audio: bool
    audio_chunk_sec: int  # 音声だけを取り込むときの区切り(秒)

    # --- 検索 ---
    top_k_default: int

    # --- 監視フォルダ(空なら無効) ---
    watch_dirs: tuple[Path, ...]
    watch_interval_sec: int  # 走査の間隔(秒)
    watch_settle_sec: int  # 最終更新からこの秒数たったファイルだけを取り込む(書き込み中を避ける)
    watch_group_from_dir: bool  # ファイルが入っているフォルダ名をグループ ID にする
    watch_preset: str  # 取り込みのプリセット(空なら WINDOW_SEC などの既定値)
    max_streams: int  # 同時に受信できる RTSP ストリームの数
    retention_days: float  # この日数より古い窓を削除する(録画の時刻が基準。0 なら削除しない)
    min_free_disk_mb: int  # DATA_DIR の空き容量がこれを下回ったら、新しい取り込みを止める(0 なら確認しない)
    job_retries: int  # 推論側に接続できないなど、一時的な失敗のときに取り込みを再試行する回数
    job_history_days: float  # 終わったジョブの履歴を残す日数(取り込み元ごとの最新のジョブは残す)
    backup_dir: Path  # バックアップの保存先(既定 DATA_DIR/backups。別のディスクを指定するのが望ましい)
    backup_interval_hours: float  # この間隔で自動でバックアップを取る(0 なら取らない)
    backup_keep: int  # 残すバックアップの数(古いものから削除する。0 なら削除しない)
    max_pending_searches: int  # 同時に処理する検索の上限。超えたら 503(0 なら制限しない)
    api_read_token: str  # 検索・参照だけを許可するトークン(取り込み・削除はできない)
    public_url: str  # 外から見たこのサーバーの URL(通知の本文のサムネイル・動画の URL を絶対 URL にする)

    # --- サーバ ---
    host: str
    port: int
    log_level: str  # ログの出力の下限(DEBUG / INFO / WARNING / ERROR)
    log_format: str  # text(1 行の文字列)か json(1 行 1 件の JSON。ログの収集基盤向け)

    # --- セキュリティ(docs/security.md) ---
    api_token: str  # 空なら認証なし。設定すると /api と /compute にトークンが必要
    embedding_token: str  # remote の compute に送るトークン(既定は api_token と同じ)
    ingest_roots: tuple[Path, ...]  # パス指定・フォルダ一括で取り込めるフォルダ
    max_upload_mb: int  # 1 リクエストの本文の上限(MB)。0 なら無制限
    allowed_origins: tuple[str, ...]  # 別オリジンからの更新系リクエストを許可するオリジン
    allowed_hosts: tuple[str, ...]  # 受け付ける Host ヘッダー("*" なら制限なし)

    # --- 推論の高速化(embedders/accel.py) ---
    torch_compile: str = "auto"  # auto / on / off。auto は NVIDIA の GPU で C コンパイラがあるときに使う

    # --- 切り出し(GET /api/sources/{id}/clip) ---
    clip_max_sec: int = 600  # 1 回に切り出せる長さの上限(秒)。0 なら切り出しを無効にする

    # --- 変化の無い区間の省略(pipeline.is_static_window) ---
    skip_static: bool = False  # 真なら、前の窓と映像が変わらない窓を推論せず、前の窓を延ばす(映像のみの窓だけ)
    static_threshold: float = 4.0  # 区画ごとの輝度の差(0〜255 の平均)がこれ未満なら「変化なし」とみなす

    # --- 文書(PDF・Word・テキスト。documents.py) ---
    doc_chunk_chars: int = 800  # 1 つのベクトルにする文字数の目安
    doc_overlap_chars: int = 100  # 前の塊の終わりを、次の塊の頭に重ねる文字数

    # --- CPU 向け ONNX Runtime 推論(embedders/onnx.py) ---
    embedding_runtime: str = "auto"  # auto / pytorch / onnx。auto は CPU で ONNX モデルがあるときに onnx を使う
    onnx_model_id: str = "onnx-community/embeddinggemma-2-ONNX"
    onnx_model_revision: str = ""  # ONNX モデルの版(Hugging Face のコミット)。イメージでは同梱した版に固定
    onnx_model_dir: str = ""  # 同梱済み ONNX モデルのディレクトリ(空なら Hugging Face キャッシュから読む)
    onnx_precision: str = "fp32"  # fp32 / q8 / q4 / fp16。既定は GPU と互換・動画最速の fp32
    onnx_threads: int = 0  # ONNX Runtime のスレッド数(0 は自動)

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

        data_dir = Path(_env("DATA_DIR", "/data"))
        # パスの一覧はカンマ区切り(パスに空白を含められるよう、空白では区切らない)
        watch_dirs = tuple(Path(d.strip()) for d in _env("WATCH_DIRS", "").split(",") if d.strip())
        # 取り込めるフォルダ: 既定は録画のマウント先(/recordings)と DATA_DIR。監視フォルダは自動で加える
        roots = [Path(d.strip()) for d in _env("INGEST_ROOTS", "/recordings").split(",") if d.strip()]
        ingest_roots = tuple(dict.fromkeys([*roots, data_dir, *watch_dirs]))
        api_token = _env("API_TOKEN", "")

        watch_preset = _env("WATCH_PRESET", "")
        skip_frames = _env("FFMPEG_SKIP_FRAMES", "noref").lower()
        if skip_frames not in ("none", "noref", "keyframes"):
            raise ValueError(
                f"FFMPEG_SKIP_FRAMES は none / noref / keyframes のいずれかにしてください: {skip_frames!r}"
            )
        if watch_preset and watch_preset not in PRESETS:
            raise ValueError(f"WATCH_PRESET は {list(PRESETS)} のいずれかにしてください: {watch_preset!r}")

        torch_compile = _env("TORCH_COMPILE", "auto").lower()
        if torch_compile not in ("auto", "on", "off"):
            raise ValueError(f"TORCH_COMPILE は auto / on / off のいずれかにしてください: {torch_compile!r}")

        embedding_runtime = _env("EMBEDDING_RUNTIME", "auto").lower()
        if embedding_runtime not in ("auto", "pytorch", "onnx"):
            raise ValueError(
                f"EMBEDDING_RUNTIME は auto / pytorch / onnx のいずれかにしてください: {embedding_runtime!r}"
            )

        onnx_precision = _env("ONNX_PRECISION", "fp32").lower()
        if onnx_precision not in ("fp32", "q8", "q4", "fp16"):
            raise ValueError(
                f"ONNX_PRECISION は fp32 / q8 / q4 / fp16 のいずれかにしてください: {onnx_precision!r}"
            )

        if "ONNX_MODEL_DIR" in os.environ:
            onnx_model_dir = os.environ["ONNX_MODEL_DIR"].strip()
        else:
            onnx_model_dir = ""
            marker = Path("/etc/mediasearch_onnx_dir")
            if marker.is_file():
                try:
                    onnx_model_dir = marker.read_text(encoding="utf-8").strip()
                except OSError:
                    onnx_model_dir = ""

        return cls(
            role=role,
            embedding_backend=backend,
            embedding_url=_env("EMBEDDING_URL", "http://localhost:8001").rstrip("/"),
            embedding_timeout_sec=_env_float("EMBEDDING_TIMEOUT_SEC", 300.0),
            model_id=_env("MODEL_ID", "google/embeddinggemma-2"),
            model_revision=_env("MODEL_REVISION", ""),
            device=_env("DEVICE", "auto"),
            dtype=_env("DTYPE", "auto"),
            dims=dims,
            image_max_tokens=_env_int("IMAGE_MAX_TOKENS", 0),
            image_max_side=_env_int("IMAGE_MAX_SIDE", 448),
            embed_batch=_env("EMBED_BATCH", "auto").lower(),
            torch_cache_dir=(
                Path(_env("TORCH_CACHE_DIR", "")) if _env("TORCH_CACHE_DIR", "") else data_dir / "cache" / "torch"
            ),
            torch_compile=torch_compile,
            warmup_on_start=_env_bool("WARMUP_ON_START", True),
            ingest_prefetch=min(max(_env_int("INGEST_PREFETCH", 2), 1), 16),
            ingest_workers=min(max(_env_int("INGEST_WORKERS", 1), 1), 8),
            ffmpeg_hwaccel=_env("FFMPEG_HWACCEL", "auto"),
            ffmpeg_hwaccel_device=_env("FFMPEG_HWACCEL_DEVICE", ""),
            ffmpeg_skip_frames=skip_frames,
            data_dir=data_dir,
            store_media=_env_bool("STORE_MEDIA", True),
            window_sec=_env_int("WINDOW_SEC", PRESETS["object"]["window_sec"]),
            frames_per_window=_env_int("FRAMES_PER_WINDOW", PRESETS["object"]["frames_per_window"]),
            overlap_sec=_env_int("OVERLAP_SEC", 0),
            include_audio=_env_bool("INCLUDE_AUDIO", False),
            audio_chunk_sec=_env_int("AUDIO_CHUNK_SEC", 10),
            top_k_default=_env_int("TOP_K", 10),
            watch_dirs=watch_dirs,
            watch_interval_sec=max(_env_int("WATCH_INTERVAL_SEC", 60), 5),
            watch_settle_sec=max(_env_int("WATCH_SETTLE_SEC", 30), 0),
            watch_group_from_dir=_env_bool("WATCH_GROUP_FROM_DIR", True),
            watch_preset=watch_preset,
            max_streams=_env_int("MAX_STREAMS", 8),
            retention_days=max(_env_float("RETENTION_DAYS", 0.0), 0.0),
            min_free_disk_mb=max(_env_int("MIN_FREE_DISK_MB", 1024), 0),
            job_retries=max(_env_int("JOB_RETRIES", 3), 0),
            job_history_days=max(_env_float("JOB_HISTORY_DAYS", 30.0), 0.0),
            backup_dir=Path(_env("BACKUP_DIR", "")) if _env("BACKUP_DIR", "") else data_dir / "backups",
            backup_interval_hours=max(_env_float("BACKUP_INTERVAL_HOURS", 0.0), 0.0),
            backup_keep=max(_env_int("BACKUP_KEEP", 7), 0),
            max_pending_searches=max(_env_int("MAX_PENDING_SEARCHES", 32), 0),
            api_read_token=_env("API_READ_TOKEN", ""),
            public_url=_env("PUBLIC_URL", "").rstrip("/"),
            host=_env("HOST", "0.0.0.0"),
            port=_env_int("PORT", 8000),
            log_level=_env("LOG_LEVEL", "INFO").upper(),
            log_format=_log_format(_env("LOG_FORMAT", "text")),
            api_token=api_token,
            embedding_token=_env("EMBEDDING_TOKEN", api_token),
            ingest_roots=ingest_roots,
            max_upload_mb=max(_env_int("MAX_UPLOAD_MB", 4096), 0),
            allowed_origins=tuple(o.strip().rstrip("/") for o in _env("ALLOWED_ORIGINS", "").split(",") if o.strip()),
            allowed_hosts=tuple(h.strip().lower() for h in _env("ALLOWED_HOSTS", "*").split(",") if h.strip()),
            clip_max_sec=max(_env_int("CLIP_MAX_SEC", 600), 0),
            skip_static=_env_bool("SKIP_STATIC", False),
            static_threshold=_env_float("STATIC_THRESHOLD", 4.0),
            doc_chunk_chars=_env_int("DOC_CHUNK_CHARS", 800),
            doc_overlap_chars=_env_int("DOC_OVERLAP_CHARS", 100),
            embedding_runtime=embedding_runtime,
            onnx_model_id=_env("ONNX_MODEL_ID", "onnx-community/embeddinggemma-2-ONNX"),
            onnx_model_revision=_env("ONNX_MODEL_REVISION", ""),
            onnx_model_dir=onnx_model_dir,
            onnx_precision=onnx_precision,
            onnx_threads=max(_env_int("ONNX_THREADS", 0), 0),
        )
