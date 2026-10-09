"""HTTP API の要求・応答の型(OpenAPI の仕様書と Swagger UI に表示されます)。

型を定義しておくと、/docs や docs/openapi.json に各項目の意味と例が載り、
利用側は OpenAPI のコード生成ツールでクライアントを作れます。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

WindowKind = Literal["frames", "tav", "audio", "image"]
SearchKind = Literal["auto", "all", "frames", "tav", "audio", "image"]
SourceKind = Literal["video", "audio", "image", "frames"]
IngestKind = Literal["auto", "video", "audio", "image"]
Status = Literal["queued", "running", "done", "failed"]


class _Open(BaseModel):
    """将来の項目の追加で利用側が壊れないよう、定義していない項目も応答に含める。"""

    model_config = ConfigDict(extra="allow")


# ---------------------------------------------------------------------- 共通
class ErrorResponse(BaseModel):
    detail: Any = Field(description="エラーの内容(日本語の説明。入力の検証エラーでは項目ごとの一覧)")

    model_config = ConfigDict(json_schema_extra={"examples": [{"detail": "取り込み元が見つかりません"}]})


ERROR_RESPONSES: dict[int | str, dict] = {
    400: {"model": ErrorResponse, "description": "入力が正しくない"},
    401: {"model": ErrorResponse, "description": "認証が必要(API_TOKEN を設定している場合)"},
    403: {"model": ErrorResponse, "description": "許可していないフォルダ・別サイトからのリクエスト・ホスト名"},
    404: {"model": ErrorResponse, "description": "見つからない"},
    409: {"model": ErrorResponse, "description": "DB のモデル・次元と、現在の設定が一致しない"},
    413: {"model": ErrorResponse, "description": "リクエストが大きすぎる(MAX_UPLOAD_MB)"},
    422: {"model": ErrorResponse, "description": "映像・音声・画像として扱えないファイル、または入力の検証エラー"},
    503: {"model": ErrorResponse, "description": "推論側(compute)に接続できない"},
}


def errors(*codes: int) -> dict[int | str, dict]:
    """エンドポイントごとに、起こり得るエラー応答だけを仕様書に載せる。"""
    return {c: ERROR_RESPONSES[c] for c in codes}


class Health(BaseModel):
    status: Literal["ok"]
    role: Literal["all", "app", "compute"]
    version: str


# ---------------------------------------------------------------------- 状態
class VersionInfo(_Open):
    app: str = Field(description="アプリの版")
    variant: str = Field(description="イメージの種類(slim / cpu / cuda / rocm / intel。ソースから動かすと source)")
    revision: str = Field(description="ビルド元のコミット")
    build_ref: str = Field(description="ビルド元のタグ・ブランチ")


class EmbedderStatus(_Open):
    backend: str | None = Field(None, description="local / remote / dummy")
    model_id: str | None = None
    dims: int | None = Field(None, description="保存するベクトルの次元")
    dtype: str | None = None
    device: str | None = None
    accelerator: str | None = Field(None, description="cpu / nvidia-cuda / amd-rocm / intel-xpu / dummy")
    modalities: list[str] | None = None
    error: str | None = Field(None, description="推論側に接続できない場合の理由")


class IndexStatus(_Open):
    sources: int
    windows: int
    windows_by_kind: dict[str, int]
    model_id: str | None = Field(None, description="DB を作ったときのモデル(未作成なら null)")
    dims: str | None = None
    queue_size: int = Field(description="取り込みを待っているジョブの数")
    vector_db: dict[str, Any] | None = Field(
        None, description="ベクトル DB の状態(engine: 方式、vectors: 件数、file_bytes: ファイルの大きさ)"
    )


class InfoResponse(_Open):
    role: str
    version: VersionInfo
    embedder: EmbedderStatus
    defaults: dict[str, Any] = Field(description="取り込みの既定値(窓の長さなど)")
    presets: dict[str, dict[str, Any]] = Field(description="取り込みのプリセット")
    index: IndexStatus
    watch: dict[str, Any] = Field(description="監視フォルダの状態")


# ---------------------------------------------------------------------- 取り込み
class Accepted(BaseModel):
    source_id: int = Field(description="取り込み元の ID")
    job_id: int | None = Field(description="取り込みジョブの ID(重複のときは null)")
    duplicate: bool = Field(description="取り込み済みのファイルだった(新しいジョブは作っていない)")
    start_ts: str = Field(description="録画開始(撮影)の日時。ISO 8601")
    start_ts_from: Literal["request", "filename", "exif", "now", "existing"] = Field(
        description="start_ts を何から決めたか(指定値 / ファイル名 / EXIF / 受付時刻 / 取り込み済みの値)"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "source_id": 12,
                    "job_id": 34,
                    "duplicate": False,
                    "start_ts": "2026-01-01T09:00:00",
                    "start_ts_from": "filename",
                }
            ]
        }
    )


class ImageBatchError(BaseModel):
    name: str | None = Field(description="ファイル名")
    error: str


class ImageBatchResponse(BaseModel):
    queued: int = Field(description="受け付けた枚数")
    items: list[Accepted]
    errors: list[ImageBatchError] = Field(description="取り込めなかった画像")


class DirItem(Accepted):
    path: str
    group_id: str | None = None


class DirError(BaseModel):
    path: str
    error: str


class DirIngestResponse(BaseModel):
    dir: str
    queued: int = Field(description="新しく受け付けた件数")
    duplicates: int = Field(description="取り込み済みのため対象外にした件数")
    pending: int = Field(description="書き込み中の可能性があるため後回しにした件数(監視フォルダのみ)")
    errors: list[DirError] = Field(description="取り込めなかったファイル")
    items: list[DirItem]


class FramesResponse(BaseModel):
    source_id: int
    window_id: int
    kind: WindowKind


class WatchScanResponse(BaseModel):
    queued: int
    watch: dict[str, Any]


# ---------------------------------------------------------------------- 検索
class SearchHit(_Open):
    window_id: int = Field(description="窓(ベクトル)の ID。サムネイルの取得に使う")
    source_id: int = Field(description="取り込み元の ID。元ファイルの取得に使う")
    start_ms: int = Field(description="取り込み元の先頭からの開始位置(ミリ秒)。静止画は 0")
    end_ms: int = Field(description="終了位置(ミリ秒)。静止画は 0")
    kind: WindowKind
    group_id: str | None = None
    location: str | None = None
    abs_ts: float = Field(description="窓の開始の日時(UNIX 秒)")
    abs_time: str = Field(description="窓の開始の日時(ISO 8601)")
    score: float = Field(description="コサイン類似度(-1〜1。大きいほど近い)")
    source_name: str | None = None
    media_url: str | None = Field(
        description="元ファイルの URL(Range 対応。動画は #t=秒 で頭出しできる)。保存しない設定で取り込んだものは null"
    )
    thumb_url: str | None = Field(
        description="サムネイル(JPEG)の URL。音声の窓と、保存しない設定(store_media=false)で取り込んだものは null"
    )


class SearchInterval(_Open):
    source_id: int
    start_ms: int
    end_ms: int
    score: float = Field(description="区間に含まれる窓の最大のスコア")
    window_ids: list[int]
    abs_ts: float
    group_id: str | None = None
    location: str | None = None
    source_name: str | None = None


class SearchResponse(BaseModel):
    results: list[SearchHit] = Field(description="スコアの高い順の窓")
    intervals: list[SearchInterval] | None = Field(
        None, description="同じ取り込み元で隣り合う窓をまとめた区間(merge=true のとき)"
    )
    took_ms: int = Field(description="DB の検索にかかった時間(ミリ秒)")
    embed_ms: int = Field(description="クエリのベクトル化にかかった時間(ミリ秒)")
    searched_kinds: str = Field(description="指定された kind")


class TextSearchRequest(BaseModel):
    query: str = Field(description="探したい内容を表す文章(日本語・英語など)", min_length=1)
    top_k: int | None = Field(None, ge=1, le=1000, description="返す件数(省略時は TOP_K)")
    min_score: float = Field(0.0, description="これより低いスコアは返さない(該当なしを判定するため)")
    group_id: str | None = Field(None, description="グループ ID で絞り込む")
    location: str | None = Field(None, description="場所で絞り込む")
    from_ts: str | None = Field(None, description="この日時以降(UNIX 秒か ISO 8601)")
    to_ts: str | None = Field(None, description="この日時以前(UNIX 秒か ISO 8601)")
    kind: SearchKind = Field("auto", description="探す窓の種類")
    merge: bool = Field(True, description="隣り合う窓を区間にまとめた intervals も返す")

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"query": "赤い車が通り過ぎる", "top_k": 10, "min_score": 0.3, "from_ts": "2026-01-01T00:00:00"}
            ]
        }
    )


# ---------------------------------------------------------------------- 取り込み元・ジョブ
class Source(_Open):
    id: int
    kind: SourceKind
    path: str | None = Field(description="ファイルのパス(加工済みフレームは null)")
    name: str
    group_id: str | None = None
    location: str | None = None
    start_ts: float = Field(description="録画開始(撮影)の日時(UNIX 秒)")
    duration_ms: int | None = None
    status: Status
    params: dict[str, Any] = Field(description="取り込みの設定(窓の長さなど)")
    error: str | None = None
    created_at: float
    window_count: int | None = Field(None, description="窓の数(一覧のときのみ)")


class SourceList(BaseModel):
    sources: list[Source]


class JobTimings(_Open):
    """取り込みの処理時間。高速化の効果を確かめるために使います。"""

    total_ms: float = Field(description="ジョブ全体の処理時間(ms)")
    stages_ms: dict[str, float] = Field(
        description=(
            "段階ごとの時間(ms)。probe: ffprobe / decode: 映像の復号の待ち / audio: 音声の変換の待ち / "
            "image: 画像の読み込み / embed: 推論 / store: DB への保存 / thumb: サムネイルの保存。"
            "復号は推論と並行するため、decode が小さいほど推論が律速です"
        )
    )
    windows: int = Field(description="作った窓の数")
    media_ms: int = Field(description="取り込んだ映像・音声の長さ(ms。画像は 0)")
    per_window_ms: float | None = Field(None, description="窓 1 つあたりの時間(ms)")
    realtime_factor: float | None = Field(
        None, description="実時間比(1 秒の処理で取り込めた映像・音声の秒数。1 より大きければ実時間より速い)"
    )
    decoder: str | None = Field(None, description="映像の復号に使った方式(cpu / cuda / vaapi / qsv)")
    accelerator: str | None = Field(None, description="推論に使ったデバイス")


class Job(_Open):
    id: int
    source_id: int
    status: Status
    progress: int = Field(description="処理済みの窓の数")
    total: int = Field(description="窓の総数")
    error: str | None = None
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    source_name: str | None = None
    timings: JobTimings | None = Field(None, description="処理時間の内訳(完了したジョブのみ。ベンチマーク用)")


class JobList(BaseModel):
    jobs: list[Job]


class KindStats(BaseModel):
    jobs: int = Field(description="集計したジョブの数")
    windows: int = Field(description="作った窓の合計")
    media_ms: int = Field(description="取り込んだ映像・音声の長さの合計(ms)")
    total_ms: float = Field(description="処理時間の合計(ms)")
    stages_ms: dict[str, float] = Field(description="段階ごとの時間の合計(ms。段階の意味は Job の timings と同じ)")
    per_window_ms: float | None = Field(None, description="窓 1 つあたりの平均(ms)")
    realtime_factor: float | None = Field(None, description="実時間比(全体)")
    decoders: dict[str, int] = Field(description="映像の復号に使った方式ごとのジョブ数")


class StatsResponse(BaseModel):
    jobs: int = Field(description="集計したジョブの数(処理時間を記録した完了済みのもの)")
    by_kind: dict[str, KindStats] = Field(description="取り込み元の種類(video / audio / image)ごとの集計")


class Deleted(BaseModel):
    deleted: int


class Reindexed(BaseModel):
    source_id: int
    job_id: int


# ---------------------------------------------------------------------- compute
class TextsRequest(BaseModel):
    texts: list[str] = Field(description="ベクトル化する文章", min_length=1)
    kind: Literal["query", "document"] = Field("query", description="検索文(query)か、検索される側の文書(document)か")
    dims: int = Field(768, description="返す次元(768 / 512 / 256 / 128)")


class Vectors(BaseModel):
    vectors: list[list[float]] = Field(description="L2 正規化済みのベクトル")


class ComputeInfo(_Open):
    model_id: str
    dims: int
    dtype: str
    device: str
    accelerator: str
    modalities: list[str]
    backend: str
