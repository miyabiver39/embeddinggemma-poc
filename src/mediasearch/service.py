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
        )
        sources: dict[int, dict] = {}
        for h in hits:
            src = sources.get(h["source_id"]) or ctx.store.get_source(h["source_id"]) or {}
            sources[h["source_id"]] = src
            h["source_name"] = src.get("name")
            h["abs_time"] = fmt_ts(h["abs_ts"])
            h["media_url"] = f"/api/media/{h['source_id']}" if src.get("path") else None
            h["thumb_url"] = f"/api/thumb/{h['window_id']}"
        out = {
            "results": hits,
            "took_ms": int((time.time() - started) * 1000),  # DB の検索だけにかかった時間
            "embed_ms": embed_ms,  # クエリのベクトル化にかかった時間
            "searched_kinds": f.get("kind") or "auto",
        }
        if f.get("merge", True):
            out["intervals"] = merge_intervals(hits)
        return out

    def search_text(self, query: str, **filters) -> dict:
        if not query.strip():
            raise InvalidInput("query が空です")
        t0 = time.time()
        qvec = self.ctx.embedder.embed_texts([query], kind="query")[0]
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
