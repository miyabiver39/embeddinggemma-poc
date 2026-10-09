"""取り込み・検索の処理(HTTP API と MCP サーバーの両方から使う)。

HTTP の要求・応答の形に依存しない処理をここにまとめ、api.py(REST)と mcp_server.py(MCP)は
入力の変換と出力の整形だけを行います。同じ検索を、どちらの入口からでも同じ結果で使えるようにするためです。
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from . import __version__
from .config import PRESETS, Settings
from .embedders import Embedder
from .ingest_files import FileIntake
from .pipeline import Ingestor
from .store import Store
from .watcher import FolderWatcher

KIND_CHOICES = ("auto", "all", "frames", "tav", "audio", "image")


class InvalidInput(ValueError):
    """入力の誤り(API では 400)。"""


class NotFound(LookupError):
    """対象が見つからない(API では 404)。"""


def media_stored(source: dict) -> bool:
    """元のファイルとサムネイルを保存する取り込み元か(STORE_MEDIA / store_media が偽なら、情報だけを返す)。"""
    return bool((source.get("params") or {}).get("store_media", True))


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
        raise InvalidInput(f"時刻の形式が正しくありません: {value!r}(UNIX 秒か ISO 8601 で指定してください)") from exc


def fmt_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts).isoformat(timespec="seconds")


def merge_intervals(results: list[dict], gap_ms: int = 500) -> list[dict]:
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
                    "group_id": r["group_id"],
                    "location": r["location"],
                    "source_name": r["source_name"],
                }
        if cur:
            merged.append(cur)
    merged.sort(key=lambda r: -r["score"])
    return merged


class Service:
    """検索・状態の確認など、入口(REST / MCP)に共通の処理。"""

    def __init__(self, ctx: Context) -> None:
        self.ctx = ctx

    def resolve_kinds(self, kind: str, auto_kinds: list[str] | None = None) -> list[str] | None:
        """検索対象の種類を決めます。auto_kinds は、クエリの種類ごとの auto の解釈です。"""
        if kind not in KIND_CHOICES:
            raise InvalidInput(f"kind は {list(KIND_CHOICES)} のいずれかにしてください")
        if kind == "all":
            return None
        if kind == "auto":
            if auto_kinds is not None:
                return auto_kinds
            # 静止画(image)は映像のみの窓と同じく画像だけを入力にしたベクトルなので、文字・画像のクエリで一緒に探す
            return ["tav" if self.ctx.settings.include_audio else "frames", "image"]
        return [kind]

    def search(self, qvec: np.ndarray, f: dict, auto_kinds: list[str] | None = None, embed_ms: int = 0) -> dict:
        """クエリのベクトルで窓を探し、表示や再生に必要な情報を付けて返します。"""
        ctx, s = self.ctx, self.ctx.settings
        started = time.time()
        info = ctx.embedder.info()
        ctx.store.ensure_compat(info.model_id, info.dims)
        hits = ctx.store.search(
            qvec,
            top_k=f.get("top_k") or s.top_k_default,
            kinds=self.resolve_kinds(f.get("kind") or "auto", auto_kinds),
            group_id=f.get("group_id") or None,
            location=f.get("location") or None,
            ts_from=parse_ts(f["from_ts"]) if f.get("from_ts") else None,
            ts_to=parse_ts(f["to_ts"]) if f.get("to_ts") else None,
            min_score=f.get("min_score") or 0.0,
            after_id=f.get("after_window_id"),
        )
        sources: dict[int, dict] = {}
        for h in hits:
            src = sources.get(h["source_id"]) or ctx.store.get_source(h["source_id"]) or {}
            sources[h["source_id"]] = src
            h["source_name"] = src.get("name")
            h["abs_time"] = fmt_ts(h["abs_ts"])
            stored = media_stored(src)
            h["media_url"] = f"/api/media/{h['source_id']}" if stored and src.get("path") else None
            h["thumb_url"] = f"/api/thumb/{h['window_id']}" if stored and src.get("kind") != "audio" else None
        out = {
            "results": hits,
            "took_ms": int((time.time() - started) * 1000),  # DB の検索だけにかかった時間
            "embed_ms": embed_ms,  # クエリのベクトル化にかかった時間
            "searched_kinds": f.get("kind") or "auto",
            # 次に after_window_id へ渡す値(この検索の時点で、最後に追加されていた窓)
            "last_window_id": ctx.store.last_window_id,
        }
        if f.get("merge", True):
            out["intervals"] = merge_intervals(hits)
        return out

    def text_vector(self, query: str) -> np.ndarray:
        if not query.strip():
            raise InvalidInput("query が空です")
        return self.ctx.embedder.embed_texts([query], kind="query")[0]

    def search_text(self, query: str, **filters) -> dict:
        t0 = time.time()
        qvec = self.text_vector(query)
        return self.search(qvec, filters, embed_ms=int((time.time() - t0) * 1000))

    def search_image(self, image: np.ndarray, **filters) -> dict:
        t0 = time.time()
        qvec = self.ctx.embedder.embed_images([image], high=True)[0]
        return self.search(qvec, filters, embed_ms=int((time.time() - t0) * 1000))

    def search_audio(self, pcm: np.ndarray, **filters) -> dict:
        t0 = time.time()
        qvec = self.ctx.embedder.embed_audio(pcm, high=True)
        # 音声だけのベクトルは映像のみ(frames)の窓とはほぼ無関係な順位になる(実測)ため、
        # 音声クエリの auto は、音声のみで取り込んだ窓(audio)を探す
        return self.search(qvec, filters, auto_kinds=["audio"], embed_ms=int((time.time() - t0) * 1000))

    def thumbnail(self, window_id: int) -> Path:
        """窓のサムネイル(JPEG)のパス。取り込み時に作っていなければ(以前の版で取り込んだものなど)、ここで作ります。"""
        ctx = self.ctx
        thumb = ctx.thumb_dir / f"w{window_id}.jpg"
        window = ctx.store.get_window(window_id)
        src = ctx.store.get_source(window["source_id"]) if window else None
        if not window or not src:
            raise NotFound(f"window_id={window_id} の窓が見つかりません")
        if not media_stored(src):
            raise NotFound("この取り込み元は、画像を保存しない設定(store_media=false)で取り込まれています")
        if thumb.is_file():
            return thumb
        if src["kind"] not in ("video", "image") or not src["path"]:
            raise NotFound(f"window_id={window_id} のサムネイルがありません(音声、または元のファイルがありません)")
        from . import media
        from .pipeline import save_thumbnail

        if src["kind"] == "image":
            save_thumbnail(media.load_image(src["path"], 320), thumb)
        else:
            media.make_thumbnail(src["path"], (window["start_ms"] + window["end_ms"]) // 2, thumb)
        return thumb

    def stats(self, limit: int = 100) -> dict:
        """直近の完了ジョブの処理時間を、取り込み元の種類(video / audio / image)ごとに集計します。

        高速化の前後で比べられるよう、合計と平均だけを返します(個々の値は GET /api/jobs の timings)。
        """
        rows = self.ctx.store.finished_timings(limit)
        groups: dict[str, dict] = {}
        for r in rows:
            g = groups.setdefault(
                r["kind"], {"jobs": 0, "windows": 0, "media_ms": 0, "total_ms": 0.0, "stages_ms": {}, "decoders": {}}
            )
            g["jobs"] += 1
            g["windows"] += int(r.get("windows") or 0)
            g["media_ms"] += int(r.get("media_ms") or 0)
            g["total_ms"] += float(r.get("total_ms") or 0)
            for k, v in (r.get("stages_ms") or {}).items():
                g["stages_ms"][k] = g["stages_ms"].get(k, 0.0) + float(v)
            if r.get("decoder"):
                g["decoders"][r["decoder"]] = g["decoders"].get(r["decoder"], 0) + 1
        for g in groups.values():
            total = g["total_ms"]
            g["total_ms"] = round(total, 1)
            g["stages_ms"] = {k: round(v, 1) for k, v in g["stages_ms"].items()}
            g["per_window_ms"] = round(total / g["windows"], 1) if g["windows"] else None
            g["realtime_factor"] = round(g["media_ms"] / total, 2) if g["media_ms"] and total > 0 else None
        return {"jobs": len(rows), "by_kind": groups}

    def info(self) -> dict:
        ctx, s = self.ctx, self.ctx.settings
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
                "revision": os.environ.get("MEDIASEARCH_REVISION", "unknown"),
                "build_ref": os.environ.get("MEDIASEARCH_BUILD_REF", "local"),
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
