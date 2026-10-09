"""取り込み・検索・管理の HTTP API(ROLE=all / app で有効)。

認証・CSRF の防止・本文の大きさの上限などは、security.py のミドルウェアがすべての経路に対して行います。
処理の本体は service.py(検索・状態)と ingest_files.py(取り込みの受付)にあり、ここでは HTTP の入出力だけを扱います。
要求・応答の型は schemas.py にあり、OpenAPI の仕様書(/docs、docs/openapi.json)に反映されます。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
import tempfile
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from . import media
from .ingest_files import MEDIA_KINDS, is_within, media_kind_of
from .media import MediaError
from .pipeline import IngestParams
from .schemas import (
    Accepted,
    Deleted,
    DirIngestResponse,
    FramesResponse,
    ImageBatchResponse,
    InfoResponse,
    IngestKind,
    Job,
    JobList,
    Reindexed,
    SearchKind,
    SearchResponse,
    Source,
    SourceList,
    StatsResponse,
    TextSearchRequest,
    WatchScanResponse,
    errors,
)
from .service import Context, Service, media_stored, parse_opt_ts, parse_ts

log = logging.getLogger(__name__)


def _opt_bool(value: str | None) -> bool | None:
    if value is None or value.strip() == "":
        return None
    return value.strip().lower() in ("1", "true", "yes", "on")


def _opt_int(value: str | None, name: str) -> int | None:
    if value is None or value.strip() == "":
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise HTTPException(400, f"{name} は整数で指定してください: {value!r}") from exc


def _safe_name(name: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", name, flags=re.UNICODE)[:80] or "file"


_STORE_MEDIA_FORM = (
    "false にすると、ファイルを取り込み後に削除し、サムネイルも作らない(検索結果は情報だけ)。省略時は STORE_MEDIA"
)

# ---------------------------------------------------------------------- リクエストの型
class _WindowOptions(BaseModel):
    """取り込みの窓の設定(省略した項目は既定値。GET /api/info の defaults と presets を参照)。"""

    preset: str | None = Field(None, description="取り込みのプリセット(object / action / speech)")
    window_sec: int | None = Field(None, description="1 つのベクトルにする時間窓(秒)")
    frames_per_window: int | None = Field(None, description="窓あたりのフレーム数")
    overlap_sec: int | None = Field(None, description="窓の重なり(秒)")
    include_audio: bool | None = Field(None, description="動画の音声も同じベクトルに含める")
    store_media: bool | None = Field(
        None,
        description=(
            "偽にすると、アップロードしたファイルを取り込み後に削除し、サムネイルも作らない。検索結果は時刻などの情報だけになる"
            "(省略時はサーバの STORE_MEDIA)"
        ),
    )


class IngestPathRequest(_WindowOptions):
    path: str = Field(description="コンテナ内のファイルのパス(INGEST_ROOTS で許可したフォルダの下)")
    group_id: str | None = Field(None, description="任意のグループ ID(検索の絞り込みに使う)")
    location: str | None = Field(None, description="任意の場所の名前(検索の絞り込みに使う)")
    start_ts: str | None = Field(
        None, description="録画開始(撮影)の日時。UNIX 秒か ISO 8601。省略時はファイル名・EXIF・受付時刻の順"
    )
    kind: IngestKind = Field("auto", description="取り込む種類(auto は拡張子で判定)")
    force: bool = Field(False, description="取り込み済みでも、もう一度取り込む")

    model_config = ConfigDict(json_schema_extra={"examples": [{"path": "/recordings/group-a/20260101_090000.mp4"}]})


class IngestDirRequest(_WindowOptions):
    dir: str = Field(description="コンテナ内のフォルダのパス(INGEST_ROOTS で許可したフォルダの下)")
    recursive: bool = Field(True, description="下位のフォルダも対象にする")
    kind: IngestKind = Field("auto", description="対象にする種類(auto は拡張子で判定し、映像・音声・画像のすべて)")
    group_id: str | None = Field(None, description="任意のグループ ID")
    group_from_dir: bool = Field(False, description="ファイルが入っているフォルダ名をグループ ID にする")
    location: str | None = Field(None, description="任意の場所の名前")
    force: bool = Field(False, description="取り込み済みでも、もう一度取り込む")

    model_config = ConfigDict(json_schema_extra={"examples": [{"dir": "/recordings", "group_from_dir": True}]})


def _filters_form(kind_description: str):
    """multipart の検索(画像・音声)で共通の絞り込み条件。"""

    def dependency(
        top_k: str | None = Form(None, description="返す件数(省略時は TOP_K)"),
        min_score: str | None = Form(None, description="これより低いスコアは返さない"),
        group_id: str | None = Form(None, description="グループ ID で絞り込む"),
        location: str | None = Form(None, description="場所で絞り込む"),
        from_ts: str | None = Form(None, description="この日時以降(UNIX 秒か ISO 8601)"),
        to_ts: str | None = Form(None, description="この日時以前(UNIX 秒か ISO 8601)"),
        kind: SearchKind = Form("auto", description=kind_description),
        merge: str | None = Form(None, description="隣り合う窓を区間にまとめた intervals も返す(既定 true)"),
        after_window_id: str | None = Form(None, description="この ID より後に追加された窓だけを探す"),
    ) -> dict:
        try:
            score = float(min_score) if min_score else 0.0
        except ValueError as exc:
            raise HTTPException(400, f"min_score は数値で指定してください: {min_score!r}") from exc
        return {
            "top_k": _opt_int(top_k, "top_k"),
            "min_score": score,
            "group_id": group_id,
            "location": location,
            "from_ts": from_ts,
            "to_ts": to_ts,
            "kind": kind,
            "merge": _opt_bool(merge) is not False,
            "after_window_id": _opt_int(after_window_id, "after_window_id"),
        }

    return dependency


_IMAGE_FILTERS = _filters_form("探す窓の種類(auto は映像の窓と静止画)")
_AUDIO_FILTERS = _filters_form("探す窓の種類(auto は音声のみの窓)")


def build_api_router(ctx: Context) -> APIRouter:

    router = APIRouter(prefix="/api")
    s = ctx.settings
    svc = Service(ctx)

    # ------------------------------------------------------------------ 共通処理
    def _params(
        preset: str | None,
        window_sec: int | None,
        frames_per_window: int | None,
        overlap_sec: int | None,
        include_audio: bool | None,
        chunk_sec: int | None = None,
        store_media: bool | None = None,
    ) -> IngestParams:
        try:
            return IngestParams.from_settings(
                s,
                preset=preset or None,
                window_sec=window_sec,
                frames_per_window=frames_per_window,
                overlap_sec=overlap_sec,
                include_audio=include_audio,
                chunk_sec=chunk_sec,
                store_media=store_media,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    def _accept_upload(upload: UploadFile, kind: str, start_ts: str | None, **meta) -> dict:
        """アップロードされたファイルを保存して受け付けます。取り込めないファイルなら消して 422 を返します。"""
        dest = _save_upload(upload)
        try:
            result = ctx.intake.accept(
                dest, kind=kind, name=upload.filename or dest.name, start_ts=parse_opt_ts(start_ts), **meta
            )
        except MediaError:
            dest.unlink(missing_ok=True)
            raise
        return result.to_dict()

    def _save_upload(upload: UploadFile) -> Path:
        ctx.media_dir.mkdir(parents=True, exist_ok=True)
        dest = ctx.media_dir / f"{uuid.uuid4().hex[:8]}_{_safe_name(upload.filename or 'upload')}"
        with dest.open("wb") as out:
            shutil.copyfileobj(upload.file, out)
        return dest

    # ------------------------------------------------------------------ 状態
    @router.get(
        "/info",
        tags=["状態"],
        summary="状態・版・既定値・索引の件数",
        response_model=InfoResponse,
        responses=errors(401),
    )
    def info() -> dict:
        """推論側(モデル・デバイス)、アプリの版、取り込みの既定値とプリセット、索引の件数、監視フォルダの状態を返します。"""
        return svc.info()

    # ------------------------------------------------------------------ 取り込み
    @router.post(
        "/ingest/video",
        tags=["取り込み"],
        summary="動画をアップロードして取り込む",
        response_model=Accepted,
        responses=errors(400, 401, 409, 413, 422),
    )
    def ingest_video(
        file: UploadFile = File(..., description="動画ファイル"),
        group_id: str | None = Form(None, description="任意のグループ ID"),
        location: str | None = Form(None, description="任意の場所の名前"),
        start_ts: str | None = Form(None, description="録画開始の日時(省略時はファイル名の日時、なければ受付時刻)"),
        preset: str | None = Form(None, description="取り込みのプリセット(object / action / speech)"),
        window_sec: str | None = Form(None, description="時間窓(秒)"),
        frames_per_window: str | None = Form(None, description="窓あたりのフレーム数"),
        overlap_sec: str | None = Form(None, description="窓の重なり(秒)"),
        include_audio: str | None = Form(None, description="音声も同じベクトルに含める(true / false)"),
        store_media: str | None = Form(None, description=_STORE_MEDIA_FORM),
    ) -> dict:
        """動画を受け付け、バックグラウンドの取り込みジョブを作ります。

        進み具合は `GET /api/jobs/{job_id}` で確認します。
        """
        params = _params(
            preset,
            _opt_int(window_sec, "window_sec"),
            _opt_int(frames_per_window, "frames_per_window"),
            _opt_int(overlap_sec, "overlap_sec"),
            _opt_bool(include_audio),
            store_media=_opt_bool(store_media),
        )
        return _accept_upload(file, "video", start_ts, group_id=group_id, location=location, params=params)

    @router.post(
        "/ingest/audio",
        tags=["取り込み"],
        summary="音声をアップロードして取り込む",
        response_model=Accepted,
        responses=errors(400, 401, 409, 413, 422),
    )
    def ingest_audio(
        file: UploadFile = File(..., description="音声ファイル(音声つきの動画も可。音声だけを使う)"),
        group_id: str | None = Form(None, description="任意のグループ ID"),
        location: str | None = Form(None, description="任意の場所の名前"),
        start_ts: str | None = Form(None, description="録音開始の日時"),
        chunk_sec: str | None = Form(None, description="区切りの長さ(秒。既定は AUDIO_CHUNK_SEC)"),
        store_media: str | None = Form(None, description=_STORE_MEDIA_FORM),
    ) -> dict:
        """音声を区切りごとにベクトルにします(窓の種類は audio)。"""
        params = _params(
            None, None, None, None, None, _opt_int(chunk_sec, "chunk_sec"), store_media=_opt_bool(store_media)
        )
        return _accept_upload(file, "audio", start_ts, group_id=group_id, location=location, params=params)

    @router.post(
        "/ingest/image",
        tags=["取り込み"],
        summary="画像をアップロードして取り込む(複数可)",
        response_model=ImageBatchResponse,
        responses=errors(401, 409, 413),
    )
    def ingest_image(
        files: list[UploadFile] = File(..., description="画像ファイル(複数可。JPEG / PNG / WebP / BMP / TIFF)"),
        group_id: str | None = Form(None, description="任意のグループ ID"),
        location: str | None = Form(None, description="任意の場所の名前"),
        start_ts: str | None = Form(None, description="撮影日時(省略時はファイル名・EXIF・受付時刻の順)"),
        store_media: str | None = Form(None, description=_STORE_MEDIA_FORM),
    ) -> dict:
        """静止画を取り込みます。画像 1 枚が 1 件の取り込み元になります(動画から切り出せない場合など)。

        1 枚ずつ受け付け、取り込めない画像があっても残りは処理します(結果は items と errors に分けて返します)。
        """
        params = _params(None, None, None, None, None, store_media=_opt_bool(store_media))
        items, errs = [], []
        for upload in files:
            try:
                items.append(
                    _accept_upload(upload, "image", start_ts, group_id=group_id, location=location, params=params)
                )
            except MediaError as exc:
                errs.append({"name": upload.filename, "error": str(exc)})
        return {"queued": len(items), "items": items, "errors": errs}

    @router.post(
        "/ingest/path",
        tags=["取り込み"],
        summary="コンテナ内のファイルを取り込む",
        response_model=Accepted,
        responses=errors(400, 401, 403, 409, 422),
    )
    def ingest_path(body: IngestPathRequest) -> dict:
        """録画フォルダなどをマウントし、コンテナ内のパスで指定して取り込みます(アップロードより速い)。

        取り込み済みのファイルは重複として既存の取り込み元を返します(`force=true` で取り込み直し)。
        """
        path = Path(body.path)
        if not path.is_file():
            raise HTTPException(400, f"ファイルが見つかりません: {body.path}(コンテナ内のパスで指定してください)")
        ctx.intake.check_allowed(path)  # 許可したフォルダの外なら、拡張子に関係なく 403 にする
        kind = media_kind_of(path) if body.kind == "auto" else body.kind
        if kind is None:
            raise MediaError(f"映像・音声・画像として扱えない拡張子です: {path.name}")
        params = _params(
            body.preset,
            body.window_sec,
            body.frames_per_window,
            body.overlap_sec,
            body.include_audio,
            store_media=body.store_media,
        )
        return ctx.intake.accept(
            path,
            kind=kind,
            group_id=body.group_id,
            location=body.location,
            start_ts=parse_opt_ts(body.start_ts),
            params=params,
            force=body.force,
        ).to_dict()

    @router.post(
        "/ingest/dir",
        tags=["取り込み"],
        summary="コンテナ内のフォルダを一括で取り込む",
        response_model=DirIngestResponse,
        responses=errors(400, 401, 403, 409, 422),
    )
    def ingest_dir(body: IngestDirRequest) -> dict:
        """フォルダの中の映像・音声・画像を、まとめて受け付けます。

        取り込み済みのファイルは飛ばすので、同じフォルダに何度実行しても重複しません。
        """
        params = _params(
            body.preset,
            body.window_sec,
            body.frames_per_window,
            body.overlap_sec,
            body.include_audio,
            store_media=body.store_media,
        )
        return ctx.intake.accept_dir(
            Path(body.dir),
            params=params,
            recursive=body.recursive,
            kind=body.kind,
            group_id=body.group_id,
            group_from_dir=body.group_from_dir,
            location=body.location,
            force=body.force,
        )

    @router.post(
        "/ingest/frames",
        tags=["取り込み"],
        summary="加工済みのフレーム(と音声)を 1 窓として登録する",
        response_model=FramesResponse,
        responses=errors(400, 401, 409, 413, 422),
    )
    def ingest_frames(
        files: list[UploadFile] = File(..., description="フレームの画像(時刻の順)"),
        times_ms: str = Form(..., description="各画像の時刻(ミリ秒)。カンマ区切り"),
        audio: UploadFile | None = File(None, description="同じ区間の音声(任意)"),
        name: str | None = Form(None, description="取り込み元の名前"),
        group_id: str | None = Form(None, description="任意のグループ ID"),
        location: str | None = Form(None, description="任意の場所の名前"),
        start_ts: str | None = Form(None, description="区間の開始の日時"),
        store_media: str | None = Form(None, description=_STORE_MEDIA_FORM),
    ) -> dict:
        """利用側で切り出したフレームを渡して、その場で 1 つの窓にします(キューを通さない)。"""
        try:
            times = [int(t) for t in times_ms.split(",") if t.strip()]
        except ValueError as exc:
            raise HTTPException(400, "times_ms は整数のカンマ区切りで指定してください") from exc
        if len(times) != len(files):
            raise HTTPException(400, f"画像({len(files)}枚)と times_ms({len(times)}個)の数が違います")
        from .embedders.wire import jpeg_to_image

        try:
            frames = [jpeg_to_image(f.file.read()) for f in files]
        except Exception as exc:
            raise HTTPException(400, f"画像として読めません: {exc}") from exc
        pcm = None
        if audio is not None:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp) / _safe_name(audio.filename or "audio")
                tmp_path.write_bytes(audio.file.read())
                info = media.probe(tmp_path)
                pcm = media.extract_audio(tmp_path, 0, max(info.duration_ms, 1))
        return ctx.ingestor.index_frames_now(
            frames=frames,
            times_ms=times,
            audio=pcm,
            name=name or "frames",
            group_id=group_id or None,
            location=location or None,
            start_ts=parse_ts(start_ts),
            thumb_dir=ctx.thumb_dir,
            store_media=_opt_bool(store_media),
        )

    @router.post(
        "/watch/scan",
        tags=["取り込み"],
        summary="監視フォルダを今すぐ走査する",
        response_model=WatchScanResponse,
        responses=errors(400, 401),
    )
    def watch_scan() -> dict:
        """次の定期走査を待たずに、監視フォルダ(WATCH_DIRS)の新しいファイルを受け付けます。"""
        if not s.watch_dirs:
            raise HTTPException(400, "監視フォルダが設定されていません(環境変数 WATCH_DIRS)")
        return {"queued": ctx.watcher.scan_once(), "watch": ctx.watcher.status}

    # ------------------------------------------------------------------ 検索
    @router.post(
        "/search/text",
        tags=["検索"],
        summary="文章で探す",
        response_model=SearchResponse,
        responses=errors(400, 401, 409, 503),
    )
    def search_text(body: TextSearchRequest) -> dict:
        """文章に近い場面(窓)を、スコアの高い順に返します。`kind=auto` は映像の窓と静止画を探します。"""
        f = body.model_dump()
        return svc.search_text(f.pop("query"), **f)

    @router.post(
        "/search/image",
        tags=["検索"],
        summary="画像で探す",
        response_model=SearchResponse,
        responses=errors(400, 401, 409, 413, 503),
    )
    def search_image(
        file: UploadFile = File(..., description="クエリの画像(JPEG / PNG など)"),
        filters: dict = Depends(_IMAGE_FILTERS),
    ) -> dict:
        """画像に似た場面を探します。`kind=auto` は映像の窓と静止画を探します。"""
        from .embedders.wire import jpeg_to_image

        try:
            image = jpeg_to_image(file.file.read())  # JPEG 以外(PNG など)も PIL が読める
        except Exception as exc:
            raise HTTPException(400, f"画像として読めません: {exc}") from exc
        return svc.search_image(image, **filters)

    @router.post(
        "/search/audio",
        tags=["検索"],
        summary="音声で探す",
        response_model=SearchResponse,
        responses=errors(400, 401, 409, 413, 422, 503),
    )
    def search_audio(
        file: UploadFile = File(..., description="クエリの音声(先頭 60 秒まで使う)"),
        filters: dict = Depends(_AUDIO_FILTERS),
    ) -> dict:
        """音声に似た区間を探します。`kind=auto` は音声のみで取り込んだ窓(audio)を探します。"""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp) / _safe_name(file.filename or "audio")
            tmp_path.write_bytes(file.file.read())
            info = media.probe(tmp_path)
            pcm = media.extract_audio(tmp_path, 0, min(max(info.duration_ms, 1), 60_000))
        if pcm.size == 0:
            raise HTTPException(400, "音声を読み取れませんでした")
        return svc.search_audio(pcm, **filters)

    @router.get(
        "/search/live",
        tags=["検索"],
        summary="取り込み中の新しい窓から、文章に合うものを通知する(リアルタイム検索・SSE)",
        response_class=StreamingResponse,
        responses={
            200: {
                "description": (
                    "Server-Sent Events。`event: ready`(接続時。data は {last_window_id})、"
                    "`event: hit`(新しく見つかった窓。data は検索結果の 1 件と同じ形)、"
                    "`event: end`(duration_sec の経過)。"
                    "15 秒ごとに、接続を保つためのコメント行を送ります"
                ),
                "content": {"text/event-stream": {"schema": {"type": "string"}}},
            },
            **errors(400, 401, 409, 503),
        },
    )
    async def search_live(
        request: Request,
        query: str = Query(..., min_length=1, description="探したい内容を表す文章"),
        min_score: float = Query(0.3, description="これ以上のスコアの窓だけを通知する"),
        group_id: str | None = Query(None, description="グループ ID で絞り込む"),
        location: str | None = Query(None, description="場所で絞り込む"),
        kind: SearchKind = Query("auto", description="探す窓の種類"),
        after_window_id: int | None = Query(
            None, description="この ID より後の窓から通知する(省略時は接続した後に追加された窓だけ。再接続のときに使う)"
        ),
        duration_sec: int = Query(3600, ge=1, le=86_400, description="この秒数が経つと接続を閉じる(再接続して続ける)"),
    ) -> StreamingResponse:
        """取り込み中(動画・音声・画像・ストリーム)に追加された窓のうち、文章に合うものを、追加されるたびに通知します。

        ブラウザでは `new EventSource("/api/search/live?query=...")` で受け取れます(API_TOKEN を設定している場合は、
        WebUI と同じく Cookie で認証します)。クエリのベクトル化は接続時の 1 回だけで、以降は新しい窓だけを照合します。
        """
        qvec = await run_in_threadpool(svc.text_vector, query)
        filters = {"min_score": min_score, "group_id": group_id, "location": location, "kind": kind, "merge": False}
        seen = ctx.store.last_window_id if after_window_id is None else after_window_id

        def event(name: str, data: dict) -> str:
            return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

        async def stream():
            nonlocal seen
            yield event("ready", {"last_window_id": seen})
            started = last_sent = time.monotonic()
            while time.monotonic() - started < duration_sec:
                if await request.is_disconnected():
                    return
                latest = ctx.store.last_window_id
                if latest > seen:
                    res = await run_in_threadpool(
                        svc.search, qvec, {**filters, "after_window_id": seen, "top_k": 1000}
                    )
                    # 検索中に増えた窓は、次の回に通知する(重複して送らない)
                    hits = sorted((h for h in res["results"] if h["window_id"] <= latest), key=lambda h: h["window_id"])
                    for h in hits:
                        yield event("hit", h)
                        last_sent = time.monotonic()
                    seen = latest
                elif time.monotonic() - last_sent > 15:
                    yield ": keep-alive\n\n"
                    last_sent = time.monotonic()
                await asyncio.sleep(0.5)
            yield event("end", {"last_window_id": seen})

        headers = {"Cache-Control": "no-store", "X-Accel-Buffering": "no"}  # 中継のサーバーにためさせない
        return StreamingResponse(stream(), media_type="text/event-stream", headers=headers)

    # ------------------------------------------------------------------ 取り込み元・ジョブ
    @router.get(
        "/sources",
        tags=["取り込み元・ジョブ"],
        summary="取り込み元の一覧",
        response_model=SourceList,
        responses=errors(401),
    )
    def list_sources(limit: int = Query(200, ge=1, le=10_000, description="返す件数(新しい順)")) -> dict:
        return {"sources": ctx.store.list_sources(limit)}

    @router.get(
        "/sources/{source_id}",
        tags=["取り込み元・ジョブ"],
        summary="取り込み元の詳細",
        response_model=Source,
        responses=errors(401, 404),
    )
    def get_source(source_id: int) -> dict:
        src = ctx.store.get_source(source_id)
        if src is None:
            raise HTTPException(404, "取り込み元が見つかりません")
        return src

    @router.delete(
        "/sources/{source_id}",
        tags=["取り込み元・ジョブ"],
        summary="取り込み元とベクトルを削除する",
        response_model=Deleted,
        responses=errors(401, 404),
    )
    def delete_source(source_id: int) -> dict:
        """取り込み元と、その窓(ベクトル)を削除します。アップロードしたファイルは消し、パス指定の元ファイルは消しません。"""
        src = ctx.store.get_source(source_id)
        if src is None:
            raise HTTPException(404, "取り込み元が見つかりません")
        ctx.store.delete_source(source_id)
        if src["path"] and Path(src["path"]).is_relative_to(ctx.media_dir):
            Path(src["path"]).unlink(missing_ok=True)
        return {"deleted": source_id}

    @router.post(
        "/sources/{source_id}/reindex",
        tags=["取り込み元・ジョブ"],
        summary="取り込み直す",
        response_model=Reindexed,
        responses=errors(401, 403, 404),
    )
    def reindex_source(source_id: int) -> dict:
        """同じ設定で取り込み直します(古い窓は削除されます)。"""
        src = ctx.store.get_source(source_id)
        if src is None or src["kind"] not in MEDIA_KINDS:
            raise HTTPException(404, "再取り込みできる取り込み元が見つかりません")
        if not src["path"]:
            raise HTTPException(404, "元のファイルを保存していないため、取り込み直せません(store_media=false)")
        if not Path(src["path"]).is_file():
            raise HTTPException(404, f"元のファイルが見つかりません: {src['path']}")
        ctx.intake.check_allowed(Path(src["path"]))  # 以前の版で登録された、許可外のパスは処理しない
        ctx.store.update_source(source_id, status="queued", error=None)
        return {"source_id": source_id, "job_id": ctx.ingestor.submit(source_id)}

    @router.get(
        "/jobs",
        tags=["取り込み元・ジョブ"],
        summary="ジョブの一覧",
        response_model=JobList,
        responses=errors(401),
    )
    def list_jobs(limit: int = Query(100, ge=1, le=10_000, description="返す件数(新しい順)")) -> dict:
        return {"jobs": ctx.store.list_jobs(limit)}

    @router.get(
        "/stats",
        tags=["状態"],
        summary="取り込みの処理時間の集計(ベンチマーク)",
        response_model=StatsResponse,
        responses=errors(401),
    )
    def stats(limit: int = Query(100, ge=1, le=10_000, description="集計する完了ジョブの数(新しい順)")) -> dict:
        """直近の完了ジョブの処理時間を、種類ごとに合計・平均します。設定を変えて取り込み直し、前後で比べるのに使います。"""
        return svc.stats(limit)

    @router.get(
        "/jobs/{job_id}",
        tags=["取り込み元・ジョブ"],
        summary="ジョブの状態",
        response_model=Job,
        responses=errors(401, 404),
    )
    def get_job(job_id: int) -> dict:
        """取り込みの進み具合(progress / total)と状態(queued / running / done / failed)を返します。"""
        job = ctx.store.get_job(job_id)
        if job is None:
            raise HTTPException(404, "ジョブが見つかりません")
        return job

    # ------------------------------------------------------------------ 配信
    @router.get(
        "/media/{source_id}",
        tags=["配信"],
        summary="元ファイル(Range 対応)",
        response_class=FileResponse,
        responses={
            200: {
                "description": "元のファイル(Content-Type はファイルの種類に合わせる。Range 指定時は 206)",
                "content": {"application/octet-stream": {"schema": {"type": "string", "format": "binary"}}},
            },
            **errors(401, 404),
        },
    )
    def get_media(source_id: int) -> FileResponse:
        """元の動画・音声・画像を返します。Range リクエストに対応するため、動画は `#t=秒` で頭出し再生できます。"""
        src = ctx.store.get_source(source_id)
        # 配信するのは、受付時の検証を通ったものだけ(取り込みに失敗したものは返さない)
        if (
            src is None
            or src["kind"] not in MEDIA_KINDS
            or src["status"] == "failed"
            or not media_stored(src)  # 保存しない設定で取り込んだものは返さない(情報だけを返す)
            or not src["path"]
            or not is_within(src["path"], s.ingest_roots)  # 以前の版で登録された、許可外のパスも返さない
            or not Path(src["path"]).is_file()
        ):
            raise HTTPException(404, "ファイルが見つかりません")
        return FileResponse(src["path"])

    @router.get(
        "/thumb/{window_id}",
        tags=["配信"],
        summary="窓のサムネイル(JPEG)",
        response_class=FileResponse,
        responses={
            200: {
                "description": "長辺 320px の JPEG",
                "content": {"image/jpeg": {"schema": {"type": "string", "format": "binary"}}},
            },
            **errors(401, 404),
        },
    )
    def get_thumb(window_id: int) -> FileResponse:
        """取り込み時に作ったサムネイルを返します。保存しない設定(store_media=false)で取り込んだ窓は 404 です。"""
        try:
            thumb = svc.thumbnail(window_id)
        except MediaError as exc:
            raise HTTPException(404, f"サムネイルを作れません: {exc}") from exc
        return FileResponse(thumb, media_type="image/jpeg")

    return router
