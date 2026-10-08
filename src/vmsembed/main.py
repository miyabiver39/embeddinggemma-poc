"""アプリの入口。

起動:  uvicorn --factory vmsembed.main:create_app --host 0.0.0.0 --port 8000
ROLE 環境変数で、app / compute / all のどれとして動くかが決まります。
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse

from . import __version__
from .api import Context, build_api_router
from .compute_api import build_compute_router
from .config import Settings
from .embedders import Embedder, build_embedder
from .ingest_files import FileIntake, PathNotAllowed
from .media import MediaError
from .pipeline import Ingestor
from .security import SecurityMiddleware, startup_warnings
from .store import IndexMismatch, Store
from .watcher import FolderWatcher

log = logging.getLogger("vmsembed")
WEB_DIR = Path(__file__).parent / "web"


def _setup_logging() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


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
    log.info("vmsembed %s を起動します(ROLE=%s, EMBEDDING_BACKEND=%s)", __version__, s.role, s.embedding_backend)

    # モデルの読み込みなど重い初期化は、アプリを作るこの時点で1回だけ行う(ルーターもここで登録する)
    s.data_dir.mkdir(parents=True, exist_ok=True)
    emb = embedder or build_embedder(s)
    store: Store | None = None
    ingestor: Ingestor | None = None
    intake: FileIntake | None = None
    watcher: FolderWatcher | None = None
    if s.role in ("all", "app"):
        store = Store(s.data_dir / "vmsembed.db")
        ingestor = Ingestor(s, store, emb)
        intake = FileIntake(store, ingestor, s.ingest_roots)
        watcher = FolderWatcher(s, store, intake)
        _warn_if_incompatible(store, emb)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if ingestor:
            ingestor.start()
        if watcher:
            watcher.start()
        log.info("起動が完了しました。 http://localhost:%d/", s.port)
        try:
            yield
        finally:
            if watcher:
                watcher.stop()
            if ingestor:
                ingestor.stop()
            if store:
                store.close()
            emb.close()

    app = FastAPI(
        title="vmsembed",
        version=__version__,
        description="EmbeddingGemma 2 を使った VMS 向けマルチモーダル検索基盤(開発用)",
        lifespan=lifespan,
    )

    app.state.embedder = emb
    app.add_middleware(SecurityMiddleware, settings=s)
    startup_warnings(s)
    if s.role in ("all", "compute"):
        app.include_router(build_compute_router(emb))
    if store and ingestor and intake and watcher:
        app.include_router(build_api_router(Context(s, store, emb, ingestor, intake, watcher)))

    @app.get("/healthz", tags=["system"])
    def healthz() -> dict:
        return {"status": "ok", "role": s.role, "version": __version__}

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
