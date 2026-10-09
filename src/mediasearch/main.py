"""アプリの入口。

起動:  uvicorn --factory mediasearch.main:create_app --host 0.0.0.0 --port 8000
ROLE 環境変数で、app / compute / all のどれとして動くかが決まります。
"""

from __future__ import annotations

import logging
import os
import sqlite3
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from . import __version__
from .api import build_api_router
from .compute_api import build_compute_router
from .config import Settings
from .embedders import Embedder, build_embedder
from .ingest_files import FileIntake, PathNotAllowed
from .media import MediaError
from .pipeline import Ingestor
from .schemas import Health
from .security import SecurityMiddleware, startup_warnings
from .service import Context, InvalidInput, NotFound
from .store import IndexMismatch, Store
from .watcher import FolderWatcher

log = logging.getLogger("mediasearch")
WEB_DIR = Path(__file__).parent / "web"


def _setup_logging() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


API_DESCRIPTION = """
録画映像・音声・画像を取り込み、**文章・画像・音声**で該当する場面を探すための API です。

- 取り込み(`/api/ingest/*`)はバックグラウンドのジョブで行い、応答の `job_id` で進み具合を確認します
  (`GET /api/jobs/{job_id}`)。
- 検索(`/api/search/*`)は、スコア(コサイン類似度)の高い順に窓(時間の区切り、または静止画 1 枚)を返します。
- 元ファイルは `GET /api/media/{source_id}`(Range 対応)、サムネイルは `GET /api/thumb/{window_id}` で取得します。
- エラーは `{"detail": "..."}` の形で返します(説明は日本語)。

**認証**: サーバに `API_TOKEN` が設定されている場合、`Authorization: Bearer <トークン>` か
`X-API-Key: <トークン>` が必要です(設定されていない場合は不要)。
Scalar(`/scalar`)では Authentication 欄、Swagger UI(`/docs`)では右上の Authorize にトークンを入力します。

AI エージェントからは、同じ機能を MCP(`/mcp`)で利用できます。
開発者向けの説明は docs/api.md と docs/mcp.md を参照してください。
"""

OPENAPI_TAGS = [
    {"name": "状態", "description": "サーバの状態と版の確認"},
    {
        "name": "取り込み",
        "description": "映像・音声・画像の登録(アップロード、コンテナ内のパス、フォルダ一括、監視フォルダ)",
    },
    {
        "name": "検索",
        "description": "文章・画像・音声による検索。グループ ID・場所・期間・種類・最小スコアで絞り込めます",
    },
    {"name": "取り込み元・ジョブ", "description": "登録したファイルと取り込みジョブの確認・削除・取り込み直し"},
    {"name": "配信", "description": "元ファイルとサムネイルの取得"},
    {"name": "compute", "description": "ベクトル化だけを行う API(ROLE=compute / all)。app から内部的に使います"},
]


# Scalar(API リファレンスの画面)。版を固定し、SRI のハッシュで改ざんされていないことをブラウザに確かめさせる。
# 版を上げるときは、ファイルの sha384 を計算し直して SCALAR_SRI も更新すること(docs/api.md)
SCALAR_VERSION = "1.72.1"
SCALAR_SRI = "sha384-U11tb2XnKvmwt8RlTvnwUnYgrN+ur4Xyh9htLhjajWNR/Oyl5AX5DEz00qRmlrmK"
# 外部への送信を止める設定。既定のままでは、試し呼び出しを Scalar 社の中継サーバー(proxy.scalar.com)経由で
# 送ることがあり、トークンや検索結果が外部に渡り得る。
# 利用状況の送信(telemetry)、AI 機能(api.scalar.com)、外部のフォントも使わない
SCALAR_CONFIG = {
    "url": "/openapi.json",
    "proxyUrl": "",
    "telemetry": False,
    "agent": {"disabled": True},
    "withDefaultFonts": False,
    "hideClientButton": True,
    "showDeveloperTools": "never",
    "persistAuth": False,
    "defaultHttpClient": {"targetKey": "shell", "clientKey": "curl"},
    "authentication": {"preferredSecurityScheme": "bearer"},
}
SCALAR_HTML = """<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="referrer" content="no-referrer">
<title>mediasearch API リファレンス</title>
</head>
<body>
<div id="app"></div>
<script src="https://cdn.jsdelivr.net/npm/@scalar/api-reference@{version}/dist/browser/standalone.js"
  integrity="{sri}" crossorigin="anonymous"></script>
<script>
  // MCP の案内は、外部のサービスではなく、このサーバーの /mcp を指す(接続先はページを開いた URL から決める)
  const config = {config};
  config.mcp = { name: "mediasearch", url: location.origin + "/mcp" };
  Scalar.createApiReference("#app", config);
</script>
</body>
</html>
"""


def _scalar_page() -> str:
    import json

    return (
        SCALAR_HTML.replace("{version}", SCALAR_VERSION)
        .replace("{sri}", SCALAR_SRI)
        .replace("{config}", json.dumps(SCALAR_CONFIG, ensure_ascii=False))
    )


def _openapi_with_security(app: FastAPI) -> dict:
    """OpenAPI の仕様書に認証方式を載せます(認証自体は security.py のミドルウェアが行います)。

    依存関係(Depends)で認証していないため FastAPI は認証方式を自動では載せません。
    Swagger UI の Authorize とクライアントの自動生成で使えるよう、ここで追記します。
    """
    if app.openapi_schema:
        return app.openapi_schema
    from fastapi.openapi.utils import get_openapi

    schema = get_openapi(
        title=app.title,
        version=app.version,
        summary=app.summary,
        description=app.description,
        routes=app.routes,
        tags=app.openapi_tags,
        license_info=app.license_info,
    )
    schema.setdefault("components", {})["securitySchemes"] = {
        "bearer": {"type": "http", "scheme": "bearer", "description": "API_TOKEN の値"},
        "apiKey": {"type": "apiKey", "in": "header", "name": "X-API-Key", "description": "API_TOKEN の値"},
    }
    # API_TOKEN を設定していないサーバでは認証は不要(空の要件 {} は「認証なしでも可」を表す)
    security = [{"bearer": []}, {"apiKey": []}, {}]
    for path, ops in schema.get("paths", {}).items():
        if path == "/healthz":
            continue
        for op in ops.values():
            op["security"] = security
    app.openapi_schema = schema
    return schema


def _warn_if_incompatible(store: Store, emb: Embedder) -> None:
    """DB のモデル・次元と現在の推論の設定が違えば、起動時に警告します(起動は止めません)。

    取り込み・検索のたびにも同じ確認をして 409 を返しますが、設定の誤りに早く気づけるよう、
    起動時にもログへ出します。外部の compute に届かない場合は、ここでは確認を見送ります
    (compute より先に app が起動しても動けるようにするため)。
    """
    saved_model, saved_dims = store.get_meta("model_id"), store.get_meta("dims")
    if saved_model is None:
        return
    try:
        info = emb.info()
    except ConnectionError as exc:
        log.warning("推論側に接続できないため、DB との整合性の確認を見送ります: %s", exc)
        return
    if saved_model != info.model_id or saved_dims != str(info.dims):
        log.warning(
            "DB(model_id=%s, dims=%s)と現在の設定(model_id=%s, dims=%s)が一致しません。"
            "このままでは取り込み・検索が 409 で拒否されます。設定を戻すか DATA_DIR を変えてください。",
            saved_model,
            saved_dims,
            info.model_id,
            info.dims,
        )


def create_app(settings: Settings | None = None, embedder: Embedder | None = None) -> FastAPI:
    """アプリを作ります。テストでは settings と embedder を差し込めます。"""
    _setup_logging()
    s = settings or Settings.from_env()
    log.info("mediasearch %s を起動します(ROLE=%s, EMBEDDING_BACKEND=%s)", __version__, s.role, s.embedding_backend)

    # モデルの読み込みなど重い初期化は、アプリを作るこの時点で1回だけ行う(ルーターもここで登録する)
    s.data_dir.mkdir(parents=True, exist_ok=True)
    emb = embedder or build_embedder(s)
    store: Store | None = None
    ingestor: Ingestor | None = None
    intake: FileIntake | None = None
    watcher: FolderWatcher | None = None
    if s.role in ("all", "app"):
        store = Store(s.data_dir / "mediasearch.db")
        ingestor = Ingestor(s, store, emb)
        intake = FileIntake(store, ingestor, s.ingest_roots)
        watcher = FolderWatcher(s, store, intake)
        _warn_if_incompatible(store, emb)

    mcp_run = None  # MCP サーバーの起動・停止(ROLE=all / app のとき、下で設定する)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        stack = AsyncExitStack()
        if mcp_run:
            await stack.enter_async_context(mcp_run())
        if ingestor:
            ingestor.start()
        if watcher:
            watcher.start()
        log.info("起動が完了しました。 http://localhost:%d/", s.port)
        try:
            yield
        finally:
            await stack.aclose()
            if watcher:
                watcher.stop()
            if ingestor:
                ingestor.stop()
            if store:
                store.close()
            emb.close()

    app = FastAPI(
        title="mediasearch",
        version=__version__,
        summary="EmbeddingGemma 2 を使った録画映像・音声・画像のマルチモーダル検索 API(開発用)",
        description=API_DESCRIPTION,
        openapi_tags=OPENAPI_TAGS,
        license_info={"name": "Apache-2.0", "identifier": "Apache-2.0"},
        lifespan=lifespan,
    )
    app.openapi = lambda: _openapi_with_security(app)

    app.state.embedder = emb
    app.add_middleware(SecurityMiddleware, settings=s)
    startup_warnings(s)
    if s.role in ("all", "compute"):
        app.include_router(build_compute_router(emb))
    if store and ingestor and intake and watcher:
        ctx = Context(s, store, emb, ingestor, intake, watcher)
        app.include_router(build_api_router(ctx))
        # AI エージェント向けの MCP サーバー(/mcp)。REST と同じ処理・同じ認証で動く
        from .mcp_server import build_mcp_server, mount_mcp

        mcp_run = mount_mcp(app, build_mcp_server(ctx), s.max_upload_mb * 1024 * 1024)

    @app.get("/scalar", include_in_schema=False)
    def scalar() -> HTMLResponse:
        """API リファレンス(Scalar)。仕様の閲覧と、API の試し呼び出しができます。"""
        return HTMLResponse(_scalar_page())

    @app.get("/healthz", tags=["状態"], summary="死活監視(認証不要)", response_model=Health)
    def healthz() -> dict:
        return {"status": "ok", "role": s.role, "version": __version__}

    @app.exception_handler(InvalidInput)
    async def _invalid_input(_: Request, exc: InvalidInput) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(NotFound)
    async def _not_found(_: Request, exc: NotFound) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=404)

    @app.exception_handler(sqlite3.Error)
    async def _db_error(_: Request, exc: sqlite3.Error) -> JSONResponse:
        # 想定外の DB のエラーも、原因の手がかりを返す(素の 500 では利用側が切り分けられないため)
        log.exception("DB の処理に失敗しました")
        return JSONResponse({"detail": f"DB の処理に失敗しました: {exc}"}, status_code=500)

    @app.exception_handler(IndexMismatch)
    async def _index_mismatch(_: Request, exc: IndexMismatch) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.exception_handler(ConnectionError)
    async def _compute_down(_: Request, exc: ConnectionError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=503)

    @app.exception_handler(PathNotAllowed)
    async def _path_not_allowed(_: Request, exc: PathNotAllowed) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=403)

    @app.exception_handler(MediaError)
    async def _media_error(_: Request, exc: MediaError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=422)

    if s.role in ("all", "app"):

        @app.get("/", include_in_schema=False)
        def index() -> FileResponse:
            return FileResponse(WEB_DIR / "index.html")

    else:

        @app.get("/", include_in_schema=False)
        def index_compute() -> dict:
            return {"role": "compute", "docs": "/docs", "info": "/compute/info"}

    return app
