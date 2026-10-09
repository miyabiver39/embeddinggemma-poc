"""MCP(Model Context Protocol)サーバー。AI エージェントから検索・取り込みを使えるようにします。

アプリと同じプロセスの `/mcp` で、Streamable HTTP の MCP サーバーとして動きます(ROLE=all / app)。

  - 処理は REST API と同じもの(service.py、ingest_files.py)を使うため、結果も同じです。
  - 認証(API_TOKEN)・CSRF の防止・本文の大きさの上限は、REST API と同じミドルウェアがかかります。
    MCP の利用側は `Authorization: Bearer <トークン>` を送ります。
  - 状態を持たない方式(stateless)で動かします。ツールの呼び出しが 1 回ずつ完結するため、セッションの管理が要りません。
  - DNS リバインディング対策は security.py(ALLOWED_HOSTS)で行うため、SDK 側の同じ機能は無効にしています
    (SDK の既定は localhost 以外からの接続を拒否するため、コンテナの外から使えなくなる)。

接続方法は docs/mcp.md を参照してください。
"""

from __future__ import annotations

import base64
import binascii
import io
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from mcp.server.mcpserver import Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations

from . import __version__, media
from .ingest_files import MEDIA_KINDS, media_kind_of
from .media import MediaError
from .pipeline import IngestParams
from .service import Context, InvalidInput, NotFound, Service, parse_opt_ts
from .store import IndexMismatch

MCP_PATH = "/mcp"

INSTRUCTIONS = """\
録画映像・音声・画像を、文章・画像・音声で検索するためのサーバーです。

- 探すときは search_text を使います。結果の window_id を get_thumbnail に渡すと、その場面の画像を確認できます。
- 結果の start_ms / end_ms は、取り込み元(source_id)の先頭からの位置(ミリ秒)、abs_time はその場面の日時です。
- score はコサイン類似度で、絶対的な基準ではありません。上位の結果を見比べて判断してください。
- 取り込みはサーバー内のパス(ingest_path / ingest_dir)で指定します。取り込みは非同期で、get_job で完了を確認します。
"""

_READ_ONLY = ToolAnnotations(readOnlyHint=True, openWorldHint=False)
# 取り込みはデータを追加するが、同じファイルは重複して登録しない(同じ呼び出しを繰り返しても結果が増えない)
_INGEST = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False)


def _trim_hit(h: dict) -> dict:
    """エージェントに渡す検索結果。判断に使わない項目(UNIX 秒など)は省き、トークンを節約する。"""
    keys = ("window_id", "source_id", "source_name", "kind", "start_ms", "end_ms", "abs_time", "score",
            "group_id", "location", "media_url", "thumb_url")
    out = {k: h.get(k) for k in keys}
    out["score"] = round(float(out["score"]), 4)
    return out


def _trim_search(res: dict) -> dict:
    return {
        "results": [_trim_hit(h) for h in res["results"]],
        "intervals": [
            {k: iv.get(k) for k in ("source_id", "source_name", "start_ms", "end_ms", "score", "window_ids")}
            for iv in res.get("intervals") or []
        ],
        "searched_kinds": res["searched_kinds"],
        "embed_ms": res["embed_ms"],
        "last_window_id": res.get("last_window_id"),
    }


def build_mcp_server(ctx: Context) -> MCPServer:
    svc = Service(ctx)
    s = ctx.settings
    server = MCPServer(
        name="mediasearch",
        title="mediasearch(録画映像・音声・画像の検索)",
        version=__version__,
        instructions=INSTRUCTIONS,
    )

    def _guard(fn, *args, **kwargs):
        """アプリの例外を、エージェントが理解できる日本語のエラーにする。"""
        try:
            return fn(*args, **kwargs)
        except (InvalidInput, NotFound, MediaError, IndexMismatch, ConnectionError, ValueError) as exc:
            raise ToolError(str(exc)) from exc

    def _filters(top_k, min_score, group_id, location, from_time, to_time, kind) -> dict:
        return {
            "top_k": top_k,
            "min_score": min_score,
            "group_id": group_id,
            "location": location,
            "from_ts": from_time,
            "to_ts": to_time,
            "kind": kind,
            "merge": True,
        }

    def _load_bytes(data_base64: str | None, path: str | None, what: str) -> tuple[bytes | None, Path | None]:
        if bool(data_base64) == bool(path):
            raise ToolError(f"{what}_base64 か path(サーバー内のパス)の、どちらか一方を指定してください")
        if path:
            p = Path(path)
            _guard(ctx.intake.check_allowed, p)  # REST と同じく、許可したフォルダの中だけを読む
            if not p.is_file():
                raise ToolError(f"ファイルが見つかりません: {path}")
            return None, p
        try:
            raw = base64.b64decode(data_base64 or "", validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ToolError(f"{what}_base64 が Base64 として読めません") from exc
        return raw, None

    # ------------------------------------------------------------------ 検索
    @server.tool(annotations=_READ_ONLY)
    def search_text(
        query: str,
        top_k: int = 10,
        min_score: float = 0.0,
        group_id: str | None = None,
        location: str | None = None,
        from_time: str | None = None,
        to_time: str | None = None,
        kind: str = "auto",
        after_window_id: int | None = None,
    ) -> dict[str, Any]:
        """文章で、録画映像・音声・画像の該当する場面を探します。

        Args:
            query: 探したい内容を表す文章(日本語・英語など。例「赤い車が通り過ぎる」)
            top_k: 返す件数(1〜100)
            min_score: これより低いスコアは返さない
            group_id: グループ ID で絞り込む
            location: 場所で絞り込む
            from_time: この日時以降(ISO 8601。例 2026-01-01T09:00:00)
            to_time: この日時以前(ISO 8601)
            kind: 探す種類(auto / all / frames / tav / audio / image)。auto は映像の窓と静止画
            after_window_id: この ID より後に追加された窓だけを探す。取り込み中に、前回の結果の last_window_id を渡して
                新しく増えた場面だけを確認するのに使う
        """
        top_k = max(1, min(int(top_k), 100))
        f = _filters(top_k, min_score, group_id, location, from_time, to_time, kind)
        res = _guard(svc.search_text, query, **f, after_window_id=after_window_id)
        return _trim_search(res)

    @server.tool(annotations=_READ_ONLY)
    def search_image(
        image_base64: str | None = None,
        path: str | None = None,
        top_k: int = 10,
        min_score: float = 0.0,
        group_id: str | None = None,
        location: str | None = None,
        from_time: str | None = None,
        to_time: str | None = None,
        kind: str = "auto",
    ) -> dict[str, Any]:
        """画像に似た場面を探します。画像は Base64 の文字列か、サーバー内のパスで渡します。

        Args:
            image_base64: クエリの画像(JPEG / PNG など)を Base64 にした文字列
            path: クエリの画像のサーバー内のパス(取り込みを許可したフォルダの中)
            top_k: 返す件数(1〜100)
            min_score: これより低いスコアは返さない
            group_id: グループ ID で絞り込む
            location: 場所で絞り込む
            from_time: この日時以降(ISO 8601)
            to_time: この日時以前(ISO 8601)
            kind: 探す種類(auto は映像の窓と静止画)
        """
        raw, p = _load_bytes(image_base64, path, "image")
        if p is not None:
            image = _guard(media.load_image, p, s.image_max_side)
        else:
            from PIL import Image as PILImage
            from PIL import UnidentifiedImageError

            try:
                image = np.asarray(PILImage.open(io.BytesIO(raw)).convert("RGB"))
            except (UnidentifiedImageError, OSError) as exc:
                raise ToolError(f"画像として読めません: {exc}") from exc
        top_k = max(1, min(int(top_k), 100))
        filters = _filters(top_k, min_score, group_id, location, from_time, to_time, kind)
        res = _guard(svc.search_image, image, **filters)
        return _trim_search(res)

    @server.tool(annotations=_READ_ONLY)
    def search_audio(
        audio_base64: str | None = None,
        path: str | None = None,
        top_k: int = 10,
        min_score: float = 0.0,
        group_id: str | None = None,
        location: str | None = None,
        from_time: str | None = None,
        to_time: str | None = None,
        kind: str = "auto",
    ) -> dict[str, Any]:
        """音声に似た区間を探します(先頭 60 秒まで使います)。音声は Base64 の文字列か、サーバー内のパスで渡します。

        Args:
            audio_base64: クエリの音声ファイル(WAV / MP3 など)を Base64 にした文字列
            path: クエリの音声のサーバー内のパス(取り込みを許可したフォルダの中)
            top_k: 返す件数(1〜100)
            min_score: これより低いスコアは返さない
            group_id: グループ ID で絞り込む
            location: 場所で絞り込む
            from_time: この日時以降(ISO 8601)
            to_time: この日時以前(ISO 8601)
            kind: 探す種類(auto は音声のみの窓)
        """
        raw, p = _load_bytes(audio_base64, path, "audio")
        with tempfile.TemporaryDirectory() as tmp:
            if p is None:
                p = Path(tmp) / "query_audio"
                p.write_bytes(raw)
            info = _guard(media.probe, p)
            pcm = _guard(media.extract_audio, p, 0, min(max(info.duration_ms, 1), 60_000))
        if pcm.size == 0:
            raise ToolError("音声を読み取れませんでした")
        top_k = max(1, min(int(top_k), 100))
        res = _guard(svc.search_audio, pcm, **_filters(top_k, min_score, group_id, location, from_time, to_time, kind))
        return _trim_search(res)

    @server.tool(annotations=_READ_ONLY)
    def get_thumbnail(window_id: int) -> Image:
        """検索結果の場面(窓)のサムネイル画像を返します。結果が意図どおりかを、画像で確認するのに使います。

        Args:
            window_id: 検索結果の window_id
        """
        thumb = _guard(svc.thumbnail, window_id)
        return Image(data=thumb.read_bytes(), format="jpeg")

    # ------------------------------------------------------------------ 取り込み
    def _params(preset: str | None, store_media: bool | None = None) -> IngestParams:
        return _guard(IngestParams.from_settings, s, preset=preset or None, store_media=store_media)

    @server.tool(annotations=_INGEST)
    def ingest_path(
        path: str,
        kind: str = "auto",
        group_id: str | None = None,
        location: str | None = None,
        start_time: str | None = None,
        preset: str | None = None,
        force: bool = False,
        store_media: bool | None = None,
    ) -> dict[str, Any]:
        """サーバー内のファイル(動画・音声・画像)を取り込みます。

        取り込みは非同期で行います。返る job_id を get_job に渡して、完了を確認してください。

        Args:
            path: サーバー内のファイルのパス(取り込みを許可したフォルダの中)
            kind: auto(拡張子で判定)/ video / audio / image
            group_id: 任意のグループ ID
            location: 任意の場所の名前
            start_time: 録画開始(撮影)の日時(ISO 8601)。省略時はファイル名・EXIF・受付時刻の順
            preset: 取り込みのプリセット(object / action / speech)
            force: 取り込み済みでも、もう一度取り込む
            store_media: false にすると、サムネイルを作らず、検索結果に画像・再生の URL を付けない(省略時はサーバー設定)
        """
        p = Path(path)
        _guard(ctx.intake.check_allowed, p)
        if not p.is_file():
            raise ToolError(f"ファイルが見つかりません: {path}")
        if kind not in ("auto", *MEDIA_KINDS):
            raise ToolError("kind は auto / video / audio / image のいずれかにしてください")
        resolved = media_kind_of(p) if kind == "auto" else kind
        if resolved is None:
            raise ToolError(f"映像・音声・画像として扱えない拡張子です: {p.name}")
        accepted = _guard(
            ctx.intake.accept,
            p,
            kind=resolved,
            group_id=group_id,
            location=location,
            start_ts=_guard(parse_opt_ts, start_time),
            params=_params(preset, store_media),
            force=force,
        )
        return accepted.to_dict()

    @server.tool(annotations=_INGEST)
    def ingest_dir(
        dir: str,
        recursive: bool = True,
        kind: str = "auto",
        group_id: str | None = None,
        group_from_dir: bool = False,
        location: str | None = None,
        preset: str | None = None,
        force: bool = False,
        store_media: bool | None = None,
    ) -> dict[str, Any]:
        """サーバー内のフォルダの動画・音声・画像を、まとめて取り込みます。取り込み済みのファイルは対象外です。

        Args:
            dir: サーバー内のフォルダのパス(取り込みを許可したフォルダの中)
            recursive: 下位のフォルダも対象にする
            kind: auto(拡張子で判定)/ video / audio / image
            group_id: 任意のグループ ID
            group_from_dir: ファイルが入っているフォルダ名をグループ ID にする
            location: 任意の場所の名前
            preset: 取り込みのプリセット(object / action / speech)
            force: 取り込み済みでも、もう一度取り込む
            store_media: false にすると、サムネイルを作らず、検索結果に画像・再生の URL を付けない(省略時はサーバー設定)
        """
        if kind not in ("auto", *MEDIA_KINDS):
            raise ToolError("kind は auto / video / audio / image のいずれかにしてください")
        result = _guard(
            ctx.intake.accept_dir,
            Path(dir),
            params=_params(preset, store_media),
            recursive=recursive,
            kind=kind,
            group_id=group_id,
            group_from_dir=group_from_dir,
            location=location,
            force=force,
        )
        # 件数が多いと応答が大きくなるため、個別の結果は先頭の 50 件に絞る
        items = result["items"]
        return {**result, "items": items[:50], "items_truncated": len(items) > 50}

    # ------------------------------------------------------------------ 状態
    @server.tool(annotations=_READ_ONLY)
    def get_job(job_id: int) -> dict[str, Any]:
        """取り込みジョブの状態(queued / running / done / failed)と進み具合(progress / total)を返します。

        Args:
            job_id: 取り込みの応答の job_id
        """
        job = ctx.store.get_job(job_id)
        if job is None:
            raise ToolError(f"job_id={job_id} のジョブが見つかりません")
        return job

    @server.tool(annotations=_READ_ONLY)
    def list_jobs(limit: int = 20) -> dict[str, Any]:
        """取り込みジョブを新しい順に返します。

        Args:
            limit: 返す件数(1〜200)
        """
        return {"jobs": ctx.store.list_jobs(max(1, min(int(limit), 200)))}

    @server.tool(annotations=_READ_ONLY)
    def list_sources(limit: int = 20) -> dict[str, Any]:
        """取り込んだファイル(取り込み元)を新しい順に返します。

        Args:
            limit: 返す件数(1〜200)
        """
        sources = ctx.store.list_sources(max(1, min(int(limit), 200)))
        keys = ("id", "kind", "name", "group_id", "location", "start_ts", "duration_ms", "status", "error")
        keys += ("window_count",)
        return {"sources": [{k: src.get(k) for k in keys} for src in sources]}

    @server.tool(annotations=_READ_ONLY)
    def get_status() -> dict[str, Any]:
        """サーバーの状態(推論のデバイス、版、索引の件数、取り込みの待ち件数、監視フォルダ)を返します。"""
        return svc.info()

    return server


def mount_mcp(app, server: MCPServer, max_body_bytes: int):
    """FastAPI のアプリに MCP の経路(/mcp)を追加し、起動・停止の処理を返します。

    Starlette の Mount を使うと /mcp が /mcp/ へ転送(307)され、POST を転送し直さない利用側があるため、
    SDK が作る経路をそのままアプリの経路に加えます。
    """
    mcp_app = server.streamable_http_app(
        streamable_http_path=MCP_PATH,
        stateless_http=True,
        json_response=True,
        max_request_body_size=max_body_bytes or 4 * 1024 * 1024,
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )
    app.router.routes.extend(mcp_app.routes)
    return server.session_manager.run
