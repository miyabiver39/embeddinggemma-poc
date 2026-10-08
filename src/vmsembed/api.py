"""取り込み・検索・管理の HTTP API(ROLE=all / app で有効)。

ここにはセキュリティの仕組み(認証など)はありません。開発用途で、閉じたネットワークで使ってください。
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import __version__, media
from .config import PRESETS, Settings
from .embedders import Embedder
from .ingest_files import FileIntake
from .media import MediaError
from .pipeline import Ingestor, IngestParams
from .store import Store
from .watcher import FolderWatcher

log = logging.getLogger(__name__)

KIND_CHOICES = ("auto", "all", "frames", "tav", "audio")


@dataclass
class Context:
    settings: Settings
    store: Store
    embedder: Embedder
    ingestor: Ingestor
    intake: FileIntake
    watcher: FolderWatcher

    @property
    def media_dir(self) -> Path:
        return self.settings.data_dir / "media"

    @property
    def thumb_dir(self) -> Path:
        return self.settings.data_dir / "thumbs"


# ---------------------------------------------------------------------- 入力の変換
def parse_opt_ts(value: str | None) -> float | None:
    """時刻の指定を UNIX 秒にします。空なら None(ファイル名からの推定や受付時刻に任せる)。"""
    if value is None or value.strip() == "":
        return None
    return parse_ts(value)


def parse_ts(value: str | None) -> float:
    """時刻の指定を UNIX 秒にします。空なら現在時刻。

    数値(UNIX 秒)か ISO 8601("2026-10-08T14:03:00")を受け付けます。
    タイムゾーンのない ISO 8601 は、コンテナのタイムゾーン(既定は Asia/Tokyo)として扱います。
    """
    if value is None or value.strip() == "":
        return time.time()
    text = value.strip()
    try:
        return float(text)
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError as exc:
        raise HTTPException(
            400, f"時刻の形式が正しくありません: {value!r}(UNIX 秒か ISO 8601 で指定してください)"
        ) from exc


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


def _fmt_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts).isoformat(timespec="seconds")


def _merge_intervals(results: list[dict], gap_ms: int = 500) -> list[dict]:
    """同じ取り込み元で、隣り合う・重なる窓を1つの区間にまとめます(再生位置の目安にする)。"""
    by_source: dict[int, list[dict]] = {}
    for r in results:
        by_source.setdefault(r["source_id"], []).append(r)
    merged: list[dict] = []
    for source_id, items in by_source.items():
        items.sort(key=lambda r: r["start_ms"])
        cur = None
        for r in items:
            if cur and r["start_ms"] <= cur["end_ms"] + gap_ms:
                cur["end_ms"] = max(cur["end_ms"], r["end_ms"])
                cur["score"] = max(cur["score"], r["score"])
                cur["window_ids"].append(r["window_id"])
            else:
                if cur:
                    merged.append(cur)
                cur = {
                    "source_id": source_id,
                    "start_ms": r["start_ms"],
                    "end_ms": r["end_ms"],
                    "score": r["score"],
                    "window_ids": [r["window_id"]],
                    "abs_ts": r["abs_ts"],
                    "camera_id": r["camera_id"],
                    "location": r["location"],
                    "source_name": r["source_name"],
                }
        if cur:
            merged.append(cur)
    merged.sort(key=lambda r: -r["score"])
    return merged


# ---------------------------------------------------------------------- リクエストの型
class IngestPathRequest(BaseModel):
    path: str
    camera_id: str | None = None
    location: str | None = None
    start_ts: str | None = None
    preset: str | None = None
    window_sec: int | None = None
    frames_per_window: int | None = None
    overlap_sec: int | None = None
    include_audio: bool | None = None
    kind: str = "video"  # video / audio
    force: bool = False  # 取り込み済みでも、もう一度取り込む


class IngestDirRequest(BaseModel):
    dir: str
    recursive: bool = True
    kind: str = "auto"  # auto(拡張子で判定) / video / audio
    camera_id: str | None = None
    camera_from_dir: bool = False  # ファイルが入っているフォルダ名をカメラ ID にする
    location: str | None = None
    preset: str | None = None
    window_sec: int | None = None
    frames_per_window: int | None = None
    overlap_sec: int | None = None
    include_audio: bool | None = None
    force: bool = False


class TextSearchRequest(BaseModel):
    query: str
    top_k: int | None = None
    min_score: float = 0.0
    camera_id: str | None = None
    location: str | None = None
    from_ts: str | None = None
    to_ts: str | None = None
    kind: str = "auto"
    merge: bool = True


def build_api_router(ctx: Context) -> APIRouter:
    router = APIRouter(prefix="/api", tags=["api"])
    s = ctx.settings

    # ------------------------------------------------------------------ 共通処理
    def _params(
        preset: str | None,
        window_sec: int | None,
        frames_per_window: int | None,
        overlap_sec: int | None,
        include_audio: bool | None,
        chunk_sec: int | None = None,
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

    def _kinds(kind: str, auto_kinds: list[str] | None = None) -> list[str] | None:
        """検索対象の種別を決めます。auto_kinds は、クエリの種類ごとの auto の解釈です。"""
        if kind not in KIND_CHOICES:
            raise HTTPException(400, f"kind は {list(KIND_CHOICES)} のいずれかにしてください")
        if kind == "all":
            return None
        if kind == "auto":
            if auto_kinds is not None:
                return auto_kinds
            return ["tav"] if s.include_audio else ["frames"]
        return [kind]

    def _search(qvec: np.ndarray, f: dict, auto_kinds: list[str] | None = None, embed_ms: int = 0) -> dict:
        started = time.time()
        info = ctx.embedder.info()
        ctx.store.ensure_compat(info.model_id, info.dims)
        hits = ctx.store.search(
            qvec,
            top_k=f.get("top_k") or s.top_k_default,
            kinds=_kinds(f.get("kind", "auto"), auto_kinds),
            camera_id=f.get("camera_id") or None,
            location=f.get("location") or None,
            ts_from=parse_ts(f["from_ts"]) if f.get("from_ts") else None,
            ts_to=parse_ts(f["to_ts"]) if f.get("to_ts") else None,
            min_score=f.get("min_score") or 0.0,
        )
        sources: dict[int, dict] = {}
        for h in hits:
            src = sources.get(h["source_id"]) or ctx.store.get_source(h["source_id"]) or {}
            sources[h["source_id"]] = src
            h["source_name"] = src.get("name")
            h["abs_time"] = _fmt_ts(h["abs_ts"])
            h["media_url"] = f"/api/media/{h['source_id']}" if src.get("path") else None
            h["thumb_url"] = f"/api/thumb/{h['window_id']}"
        out = {
            "results": hits,
            "took_ms": int((time.time() - started) * 1000),  # DB の検索だけにかかった時間
            "embed_ms": embed_ms,  # クエリのベクトル化にかかった時間
            "searched_kinds": f.get("kind", "auto"),
        }
        if f.get("merge", True):
            out["intervals"] = _merge_intervals(hits)
        return out

    # ------------------------------------------------------------------ 状態
    @router.get("/info")
    def info() -> dict:
        try:
            e = ctx.embedder.info()
            embedder = {
                "backend": e.backend,
                "model_id": e.model_id,
                "dims": e.dims,
                "dtype": e.dtype,
                "device": e.device,
                "accelerator": e.accelerator,
                "modalities": e.modalities,
            }
        except ConnectionError as exc:
            embedder = {"error": str(exc)}
        return {
            "role": s.role,
            "version": {
                "app": __version__,
                "variant": os.environ.get("VARIANT", "source"),
                "revision": os.environ.get("VMSEMBED_REVISION", "unknown"),
                "build_ref": os.environ.get("VMSEMBED_BUILD_REF", "local"),
            },
            "embedder": embedder,
            "defaults": {
                "window_sec": s.window_sec,
                "frames_per_window": s.frames_per_window,
                "overlap_sec": s.overlap_sec,
                "include_audio": s.include_audio,
                "audio_chunk_sec": s.audio_chunk_sec,
                "top_k": s.top_k_default,
            },
            "presets": PRESETS,
            "index": {
                **ctx.store.counts(),
                "model_id": ctx.store.get_meta("model_id"),
                "dims": ctx.store.get_meta("dims"),
                "queue_size": ctx.ingestor.queue_size,
            },
            "watch": ctx.watcher.status,
        }

    # ------------------------------------------------------------------ 取り込み
    @router.post("/ingest/video")
    def ingest_video(
        file: UploadFile = File(...),
        camera_id: str | None = Form(None),
        location: str | None = Form(None),
        start_ts: str | None = Form(None),
        preset: str | None = Form(None),
        window_sec: str | None = Form(None),
        frames_per_window: str | None = Form(None),
        overlap_sec: str | None = Form(None),
        include_audio: str | None = Form(None),
    ) -> dict:
        """動画ファイルをアップロードして取り込みます(処理はバックグラウンド)。"""
        params = _params(
            preset,
            _opt_int(window_sec, "window_sec"),
            _opt_int(frames_per_window, "frames_per_window"),
            _opt_int(overlap_sec, "overlap_sec"),
            _opt_bool(include_audio),
        )
        return _accept_upload(file, "video", start_ts, camera_id=camera_id, location=location, params=params)

    @router.post("/ingest/audio")
    def ingest_audio(
        file: UploadFile = File(...),
        camera_id: str | None = Form(None),
        location: str | None = Form(None),
        start_ts: str | None = Form(None),
        chunk_sec: str | None = Form(None),
    ) -> dict:
        """音声ファイル(または音声つきの動画)を、音声だけで取り込みます。"""
        params = _params(None, None, None, None, None, _opt_int(chunk_sec, "chunk_sec"))
        return _accept_upload(file, "audio", start_ts, camera_id=camera_id, location=location, params=params)

    @router.post("/ingest/path")
    def ingest_path(body: IngestPathRequest) -> dict:
        """サーバ(コンテナ)内にあるファイルを取り込みます。VMS の録画ディレクトリをマウントして使う想定です。

        start_ts を省略すると、ファイル名の日時(例: 20260101_090000)を録画開始時刻にします。
        取り込み済みのファイルは重複として既存の取り込み元を返します(force=true で取り込み直し)。
        """
        path = Path(body.path)
        if not path.is_file():
            raise HTTPException(
                400, f"ファイルが見つかりません: {body.path}(コンテナ内のパスで指定してください)"
            )
        if body.kind not in ("video", "audio"):
            raise HTTPException(400, "kind は video か audio にしてください")
        params = _params(
            body.preset,
            body.window_sec,
            body.frames_per_window,
            body.overlap_sec,
            body.include_audio,
        )
        return ctx.intake.accept(
            path,
            kind=body.kind,
            camera_id=body.camera_id,
            location=body.location,
            start_ts=parse_opt_ts(body.start_ts),
            params=params,
            force=body.force,
        ).to_dict()

    @router.post("/ingest/dir")
    def ingest_dir(body: IngestDirRequest) -> dict:
        """フォルダの中の映像・音声ファイルを、まとめて取り込みます。

        録画開始時刻はファイル名の日時から読み取ります(読み取れなければ受付時刻)。
        取り込み済みのファイルは飛ばすので、同じフォルダに何度実行しても重複しません。
        """
        if body.kind not in ("auto", "video", "audio"):
            raise HTTPException(400, "kind は auto / video / audio のいずれかにしてください")
        params = _params(
            body.preset,
            body.window_sec,
            body.frames_per_window,
            body.overlap_sec,
            body.include_audio,
        )
        return ctx.intake.accept_dir(
            Path(body.dir),
            params=params,
            recursive=body.recursive,
            kind=body.kind,
            camera_id=body.camera_id,
            camera_from_dir=body.camera_from_dir,
            location=body.location,
            force=body.force,
        )

    @router.post("/watch/scan")
    def watch_scan() -> dict:
        """監視フォルダを、次の定期走査を待たずに今すぐ走査します(WATCH_DIRS の設定が必要)。"""
        if not s.watch_dirs:
            raise HTTPException(400, "監視フォルダが設定されていません(環境変数 WATCH_DIRS)")
        return {"queued": ctx.watcher.scan_once(), "watch": ctx.watcher.status}

    @router.post("/ingest/frames")
    def ingest_frames(
        files: list[UploadFile] = File(...),
        times_ms: str = Form(..., description="各画像の時刻(ミリ秒)。カンマ区切り"),
        audio: UploadFile | None = File(None),
        name: str | None = Form(None),
        camera_id: str | None = Form(None),
        location: str | None = Form(None),
        start_ts: str | None = Form(None),
    ) -> dict:
        """事前に加工したフレーム(と音声)を渡して、1窓として登録します(その場で処理)。"""
        try:
            times = [int(t) for t in times_ms.split(",") if t.strip()]
        except ValueError as exc:
            raise HTTPException(400, "times_ms は整数のカンマ区切りで指定してください") from exc
        if len(times) != len(files):
            raise HTTPException(400, f"画像({len(files)}枚)と times_ms({len(times)}個)の数が違います")
        from .embedders.wire import jpeg_to_image

        frames = [jpeg_to_image(f.file.read()) for f in files]
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
            camera_id=camera_id or None,
            location=location or None,
            start_ts=parse_ts(start_ts),
            thumb_dir=ctx.thumb_dir,
        )

    # ------------------------------------------------------------------ 検索
    @router.post("/search/text")
    def search_text(body: TextSearchRequest) -> dict:
        if not body.query.strip():
            raise HTTPException(400, "query が空です")
        t0 = time.time()
        qvec = ctx.embedder.embed_texts([body.query], kind="query")[0]
        return _search(qvec, body.model_dump(), embed_ms=int((time.time() - t0) * 1000))

    def _form_filters(
        top_k: str | None,
        min_score: str | None,
        camera_id: str | None,
        location: str | None,
        from_ts: str | None,
        to_ts: str | None,
        kind: str,
        merge: str | None,
    ) -> dict:
        return {
            "top_k": _opt_int(top_k, "top_k"),
            "min_score": float(min_score) if min_score else 0.0,
            "camera_id": camera_id,
            "location": location,
            "from_ts": from_ts,
            "to_ts": to_ts,
            "kind": kind,
            "merge": _opt_bool(merge) is not False,
        }

    @router.post("/search/image")
    def search_image(
        file: UploadFile = File(...),
        top_k: str | None = Form(None),
        min_score: str | None = Form(None),
        camera_id: str | None = Form(None),
        location: str | None = Form(None),
        from_ts: str | None = Form(None),
        to_ts: str | None = Form(None),
        kind: str = Form("auto"),
        merge: str | None = Form(None),
    ) -> dict:
        """画像で、似た場面を探します。"""
        from .embedders.wire import jpeg_to_image

        try:
            image = jpeg_to_image(file.file.read())  # JPEG 以外(PNG など)も PIL が読める
        except Exception as exc:
            raise HTTPException(400, f"画像として読めません: {exc}") from exc
        t0 = time.time()
        qvec = ctx.embedder.embed_images([image], high=True)[0]
        return _search(
            qvec,
            _form_filters(top_k, min_score, camera_id, location, from_ts, to_ts, kind, merge),
            embed_ms=int((time.time() - t0) * 1000),
        )

    @router.post("/search/audio")
    def search_audio(
        file: UploadFile = File(...),
        top_k: str | None = Form(None),
        min_score: str | None = Form(None),
        camera_id: str | None = Form(None),
        location: str | None = Form(None),
        from_ts: str | None = Form(None),
        to_ts: str | None = Form(None),
        kind: str = Form("auto"),
        merge: str | None = Form(None),
    ) -> dict:
        """音声で、似た場面を探します。kind=auto のときは音声のみの窓(audio)を探します。"""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp) / _safe_name(file.filename or "audio")
            tmp_path.write_bytes(file.file.read())
            info = media.probe(tmp_path)
            pcm = media.extract_audio(tmp_path, 0, min(max(info.duration_ms, 1), 60_000))
        if pcm.size == 0:
            raise HTTPException(400, "音声を読み取れませんでした")
        t0 = time.time()
        qvec = ctx.embedder.embed_audio(pcm, high=True)
        embed_ms = int((time.time() - t0) * 1000)
        # 音声だけのベクトルは映像のみ(frames)の窓とはほぼ無関係な順位になる(実測)ため、
        # 音声クエリの auto は、音声のみで取り込んだ窓(audio)を探す
        return _search(
            qvec,
            _form_filters(top_k, min_score, camera_id, location, from_ts, to_ts, kind, merge),
            auto_kinds=["audio"],
            embed_ms=embed_ms,
        )

    # ------------------------------------------------------------------ 取り込み元・ジョブ
    @router.get("/sources")
    def list_sources(limit: int = 200) -> dict:
        return {"sources": ctx.store.list_sources(limit)}

    @router.delete("/sources/{source_id}")
    def delete_source(source_id: int) -> dict:
        src = ctx.store.get_source(source_id)
        if src is None:
            raise HTTPException(404, "取り込み元が見つかりません")
        ctx.store.delete_source(source_id)
        # アップロードしたファイルだけ消す(path 指定で取り込んだ元ファイルは触らない)
        if src["path"] and Path(src["path"]).is_relative_to(ctx.media_dir):
            Path(src["path"]).unlink(missing_ok=True)
        return {"deleted": source_id}

    @router.post("/sources/{source_id}/reindex")
    def reindex_source(source_id: int) -> dict:
        src = ctx.store.get_source(source_id)
        if src is None or src["kind"] not in ("video", "audio"):
            raise HTTPException(404, "再取り込みできる取り込み元が見つかりません")
        if not src["path"] or not Path(src["path"]).is_file():
            raise HTTPException(404, f"元のファイルが見つかりません: {src['path']}")
        ctx.store.update_source(source_id, status="queued", error=None)
        return {"source_id": source_id, "job_id": ctx.ingestor.submit(source_id)}

    @router.get("/jobs")
    def list_jobs(limit: int = 100) -> dict:
        return {"jobs": ctx.store.list_jobs(limit)}

    @router.get("/jobs/{job_id}")
    def get_job(job_id: int) -> dict:
        job = ctx.store.get_job(job_id)
        if job is None:
            raise HTTPException(404, "ジョブが見つかりません")
        return job

    # ------------------------------------------------------------------ 映像・サムネイル
    @router.get("/media/{source_id}")
    def get_media(source_id: int) -> FileResponse:
        src = ctx.store.get_source(source_id)
        # 配信するのは、受付時の検証を通った映像・音声だけ(取り込みに失敗したものは返さない)
        if (
            src is None
            or src["kind"] not in ("video", "audio")
            or src["status"] == "failed"
            or not src["path"]
            or not Path(src["path"]).is_file()
        ):
            raise HTTPException(404, "ファイルが見つかりません")
        return FileResponse(src["path"])  # Range リクエスト(シーク再生)に対応

    @router.get("/thumb/{window_id}")
    def get_thumb(window_id: int) -> FileResponse:
        thumb = ctx.thumb_dir / f"w{window_id}.jpg"
        if thumb.is_file():
            return FileResponse(thumb, media_type="image/jpeg")
        window = ctx.store.get_window(window_id)
        src = ctx.store.get_source(window["source_id"]) if window else None
        if not window or not src or src["kind"] != "video" or not src["path"]:
            raise HTTPException(404, "サムネイルがありません")
        mid = (window["start_ms"] + window["end_ms"]) // 2
        try:
            media.make_thumbnail(src["path"], mid, thumb)
        except MediaError as exc:
            raise HTTPException(404, f"サムネイルを作れません: {exc}") from exc
        return FileResponse(thumb, media_type="image/jpeg")

    return router
