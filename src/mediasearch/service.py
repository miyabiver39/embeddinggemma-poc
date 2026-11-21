"""取り込み・検索の処理(HTTP API と MCP サーバーの両方から使う)。

HTTP の要求・応答の形に依存しない処理をここにまとめ、api.py(REST)と mcp_server.py(MCP)は
入力の変換と出力の整形だけを行います。同じ検索を、どちらの入口からでも同じ結果で使えるようにするためです。
"""

from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__, disk, media, thumbs
from .config import PRESETS, Settings
from .embedders import Embedder
from .ingest_files import INGEST_KINDS, FileIntake, is_within
from .pipeline import Ingestor
from .retention import RetentionWorker
from .store import Store
from .streams import StreamManager
from .watcher import FolderWatcher

log = logging.getLogger(__name__)

KIND_CHOICES = ("auto", "all", "frames", "tav", "audio", "image", "document")
# 同じベクトルの空間に入る窓の種類(似た場面の検索の kind=auto で使う)。画像だけを入力にした frames と image は同じ空間
SAME_SPACE_KINDS = {
    "frames": ["frames", "image"],
    "image": ["frames", "image"],
    "tav": ["tav"],
    "audio": ["audio"],
    "document": ["document"],
}
# 同時に行える切り出しの数(ffmpeg のエンコードは CPU を多く使い、取り込みの復号と取り合うため)
CLIP_SLOTS = threading.BoundedSemaphore(2)
# 重複の除去・取り込み元の除外をするときに、多めに探す倍率と上限(除いた後も top_k 件を返せるように)
DEDUPE_FETCH_FACTOR = 5
MAX_FETCH = 1000


def _public_error(what: str, exc: BaseException) -> str:
    """認証なしで返す応答(/readyz)用の、エラーの要約。

    例外の文言には、ファイルのパス・接続先・内部の値が含まれることがあるため、応答には例外の種類だけを入れ、
    詳細はサーバーのログに残します(ロードバランサーなど、誰からでも見える場所に内部の情報を出さないため)。
    """
    log.warning("/readyz: %sを確認できません: %s: %s", what, type(exc).__name__, exc)
    return f"{type(exc).__name__}(詳細はサーバーのログ)"


class InvalidInput(ValueError):
    """入力の誤り(API では 400)。"""


class NotFound(LookupError):
    """対象が見つからない(API では 404)。"""


class Overloaded(RuntimeError):
    """同時に処理している検索が多すぎる(API では 503 と Retry-After)。"""


class SearchLimiter:
    """同時に処理する検索の数を制限します(MAX_PENDING_SEARCHES)。

    推論は 1 件ずつのため、検索が殺到すると推論の順番待ちが際限なく積み上がり、API のスレッドを使い切って
    他の要求(状態の確認・取り込みの受付)まで応答しなくなります。上限を超えた分は待たせずに 503 で断り、
    利用側に時間をおいて再送してもらいます。
    """

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.active = 0
        self.rejected = 0
        self._lock = threading.Lock()

    @contextmanager
    def slot(self):
        with self._lock:
            if self.limit > 0 and self.active >= self.limit:
                self.rejected += 1
                raise Overloaded(
                    f"検索が混み合っています(同時に {self.limit} 件まで)。少し時間をおいて再送してください"
                )
            self.active += 1
        try:
            yield
        finally:
            with self._lock:
                self.active -= 1


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
    streams: StreamManager | None = None
    retention: RetentionWorker | None = None
    search_limiter: SearchLimiter | None = None
    alerts: Any = None  # AlertEngine(循環の読み込みを避けるため、型は Any)
    backups: Any = None  # BackupManager
    warmup_state: dict[str, Any] | None = None  # 起動時ウォームアップの状態(/readyz で確認)

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
        if r.get("kind") == "document":  # 文書の塊には時刻の幅が無い(ページで示す)
            continue
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

    def search(
        self,
        qvec: np.ndarray,
        f: dict,
        auto_kinds: list[str] | None = None,
        embed_ms: int = 0,
        exclude_window_id: int | None = None,
        exclude_source_id: int | None = None,
    ) -> dict:
        """クエリのベクトルで窓を探し、表示や再生に必要な情報を付けて返します。

        exclude_window_id / exclude_source_id は、似た場面の検索で、元の窓(と同じ取り込み元)を結果から外すために使う。
        """
        ctx, s = self.ctx, self.ctx.settings
        started = time.time()
        info = ctx.embedder.info()
        ctx.store.ensure_compat(info.model_id, info.dims)
        top_k = f.get("top_k") or s.top_k_default
        dedupe = f.get("dedupe")
        if dedupe is not None and not 0.0 < float(dedupe) <= 1.0:
            raise InvalidInput("dedupe は 0 より大きく 1 以下の値で指定してください(例 0.95)")
        # 結果から外す窓がある場合は、外した後も top_k 件を返せるよう、多めに探す
        fetch_k = top_k
        if dedupe is not None or exclude_source_id is not None:
            fetch_k = min(max(top_k * DEDUPE_FETCH_FACTOR, top_k + 1), MAX_FETCH)
        elif exclude_window_id is not None:
            fetch_k = top_k + 1
        hits = ctx.store.search(
            qvec,
            top_k=fetch_k,
            kinds=self.resolve_kinds(f.get("kind") or "auto", auto_kinds),
            group_id=f.get("group_id") or None,
            location=f.get("location") or None,
            ts_from=parse_ts(f["from_ts"]) if f.get("from_ts") else None,
            ts_to=parse_ts(f["to_ts"]) if f.get("to_ts") else None,
            min_score=f.get("min_score") or 0.0,
            after_id=f.get("after_window_id"),
        )
        if exclude_window_id is not None or exclude_source_id is not None:
            hits = [
                h
                for h in hits
                if h["window_id"] != exclude_window_id
                and (exclude_source_id is None or h["source_id"] != exclude_source_id)
            ]
        if dedupe is not None:
            hits = self._dedupe(hits, float(dedupe), top_k)
        hits = hits[:top_k]
        self.decorate_hits(hits)
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

    def _dedupe(self, hits: list[dict], threshold: float, top_k: int) -> list[dict]:
        """ほぼ同じ場面の窓を、スコアの高いもの 1 件にまとめます。

        結果どうしのコサイン類似度が threshold 以上なら同じ場面とみなし、先に残した(スコアの高い)窓に、
        まとめた件数(duplicates)を数えます。隣り合う窓は intervals でまとめるため、ここでは時刻に関係なく、
        別の時刻・別のカメラに映った同じ場面(動きの無い場面が長く続く場合など)をまとめるのが目的です。
        ベクトルは保存済みのものを読むため、推論はしません。
        """
        if not hits:
            return hits
        vecs = self.ctx.store.vectors.get(np.array([h["window_id"] for h in hits], dtype=np.uint64))
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / np.where(norms == 0, 1.0, norms)
        kept: list[int] = []
        for i, h in enumerate(hits):
            if kept:
                sims = vecs[kept] @ vecs[i]
                j = int(np.argmax(sims))
                if sims[j] >= threshold:
                    rep = hits[kept[j]]
                    rep["duplicates"] = rep.get("duplicates", 0) + 1
                    continue
            h.setdefault("duplicates", 0)
            kept.append(i)
            if len(kept) >= top_k:
                break
        return [hits[i] for i in kept]

    def search_similar(self, window_id: int, *, exclude_same_source: bool = False, **filters) -> dict:
        """保存済みの窓のベクトルをそのままクエリにして、似た場面を探します(推論はしません)。

        ベクトルの空間は窓の種類で異なるため(画像だけの窓と静止画は同じ空間、音声つきの窓 tav と音声 audio は
        それぞれ別の空間。設計書の 3.)、kind=auto では元の窓と同じ空間の種類だけを探します。元の窓は結果から外します。
        """
        ctx = self.ctx
        w = ctx.store.get_window(window_id)
        if w is None:
            raise NotFound(f"窓が見つかりません: {window_id}")
        qvec = ctx.store.vectors.get(np.array([window_id], dtype=np.uint64))[0]
        if not np.any(qvec):
            raise NotFound(f"窓のベクトルが見つかりません: {window_id}")
        auto = SAME_SPACE_KINDS.get(w["kind"], [w["kind"]])
        with self._slot():
            out = self.search(
                qvec,
                filters,
                auto_kinds=auto,
                exclude_window_id=window_id,
                exclude_source_id=w["source_id"] if exclude_same_source else None,
            )
        out["query_window"] = {k: w[k] for k in ("id", "source_id", "start_ms", "end_ms", "kind", "group_id")}
        out["query_window"]["abs_time"] = fmt_ts(w["abs_ts"])
        return out

    def decorate_hits(self, hits: list[dict]) -> list[dict]:
        """検索結果に、表示や再生に必要な情報(取り込み元の名前・日時・URL)を付けます。"""
        sources = self.ctx.store.get_sources(h["source_id"] for h in hits)  # 1 回の問い合わせで読む
        for h in hits:
            src = sources.get(h["source_id"]) or {}
            h["source_name"] = src.get("name")
            h["abs_time"] = fmt_ts(h["abs_ts"])
            stored = media_stored(src)
            h["media_url"] = f"/api/media/{h['source_id']}" if stored and src.get("path") else None
            h["thumb_url"] = (
                f"/api/thumb/{h['window_id']}" if stored and src.get("kind") not in ("audio", "document") else None
            )
        docs = [h["window_id"] for h in hits if h.get("kind") == "document"]
        if docs:
            texts = self.ctx.store.get_window_texts(docs)
            for h in hits:
                if h["window_id"] in texts:
                    h["page"] = texts[h["window_id"]]["page"]
                    h["text"] = texts[h["window_id"]]["text"]
        return hits

    def text_vector(self, query: str) -> np.ndarray:
        if not query.strip():
            raise InvalidInput("query が空です")
        return self.ctx.embedder.embed_texts([query], kind="query")[0]

    def _slot(self):
        limiter = self.ctx.search_limiter
        return limiter.slot() if limiter is not None else nullcontext()

    def search_text(self, query: str, **filters) -> dict:
        with self._slot():
            t0 = time.time()
            qvec = self.text_vector(query)
            return self.search(qvec, filters, embed_ms=int((time.time() - t0) * 1000))

    def search_image(self, image: np.ndarray, **filters) -> dict:
        with self._slot():
            t0 = time.time()
            qvec = self.ctx.embedder.embed_images([image], high=True)[0]
            return self.search(qvec, filters, embed_ms=int((time.time() - t0) * 1000))

    def search_audio(self, pcm: np.ndarray, **filters) -> dict:
        with self._slot():
            return self._search_audio(pcm, **filters)

    def _search_audio(self, pcm: np.ndarray, **filters) -> dict:
        t0 = time.time()
        qvec = self.ctx.embedder.embed_audio(pcm, high=True)
        # 音声だけのベクトルは映像のみ(frames)の窓とはほぼ無関係な順位になる(実測)ため、
        # 音声クエリの auto は、音声のみで取り込んだ窓(audio)を探す
        return self.search(qvec, filters, auto_kinds=["audio"], embed_ms=int((time.time() - t0) * 1000))

    def _stream_summary(self) -> dict:
        items = self.ctx.streams.list() if self.ctx.streams else []
        return {
            "registered": len(items),
            "running": sum(1 for s in items if s["state"]["status"] == "running"),
            "max": self.ctx.settings.max_streams,
        }

    def readiness(self) -> tuple[bool, dict]:
        """要求を処理できる状態か(GET /readyz)。DB・ベクトル DB・推論側・空き容量を確かめます。

        /healthz(プロセスが動いているか)と分けるのは、推論側(compute)が落ちている・ディスクが満杯などで
        処理できない間は、ロードバランサーなどが要求を振り分けないようにするためです。
        """
        ctx, s = self.ctx, self.ctx.settings
        checks: dict[str, dict] = {}
        try:
            ctx.store.get_meta("schema_version")
            checks["database"] = {"ok": True}
        except Exception as exc:
            checks["database"] = {"ok": False, "error": _public_error("DB", exc)}
        try:
            checks["vectors"] = {"ok": True, "count": len(ctx.store.vectors)}
        except Exception as exc:
            checks["vectors"] = {"ok": False, "error": _public_error("ベクトル DB", exc)}
        try:
            info = ctx.embedder.info()
            warm = ctx.warmup_state
            warm_ok = True if warm is None or not warm.get("required") else bool(warm.get("done"))
            embed_check: dict[str, Any] = {
                "ok": warm_ok,
                "backend": info.backend,
                "accelerator": info.accelerator,
                "warmed_up": bool(warm.get("done") and not warm.get("skipped")) if warm else False,
            }
            if not warm_ok:
                embed_check["status"] = "warming_up"
                if warm and warm.get("error"):
                    embed_check["error"] = warm["error"]
            checks["embedder"] = embed_check
        except Exception as exc:  # 分離構成で compute に接続できない場合など
            checks["embedder"] = {"ok": False, "error": _public_error("推論側", exc)}
        d = disk.status(s.data_dir, s.min_free_disk_mb)
        if "error" in d:  # パスや OS のエラーの文言は、認証なしの応答に含めない(ログに残す)
            log.warning("/readyz: DATA_DIR の空き容量を確認できません: %s", d["error"])
            d = {**d, "error": "空き容量を確認できません(詳細はサーバーのログ)"}
        checks["disk"] = {"ok": d["ok"], **{k: v for k, v in d.items() if k != "ok"}}
        ok = all(c["ok"] for c in checks.values())
        return ok, {"status": "ready" if ok else "not_ready", "checks": checks}

    def servable_source(self, source_id: int) -> dict:
        """元のファイルを返してよい取り込み元(配信と切り出しで共通の条件)。だめなら NotFound。

        返すのは、受付時の検証を通った映像・音声・画像・文書で、失敗・期限切れでなく、保存する設定で取り込み、
        許可したフォルダ(INGEST_ROOTS)の中に実在するものだけ(設計書の「メディア配信の範囲」)。
        """
        s = self.ctx.settings
        src = self.ctx.store.get_source(source_id)
        if (
            src is None
            or src["kind"] not in INGEST_KINDS
            or src["status"] in ("failed", "expired")
            or not media_stored(src)  # 保存しない設定で取り込んだものは返さない(情報だけを返す)
            or not src["path"]
            or not is_within(src["path"], s.ingest_roots)  # 以前の版で登録された、許可外のパスも返さない
            or not Path(src["path"]).is_file()
        ):
            raise NotFound("ファイルが見つかりません")
        return src

    def clip(self, source_id: int, start_ms: int, end_ms: int, accurate: bool = False) -> tuple[Path, str, str]:
        """取り込み元の一部を切り出した一時ファイルを作り、(パス, ファイル名, Content-Type) を返します。

        一時ファイルは DATA_DIR/tmp/clips に作ります。呼び出し側が、返し終えたら削除します。
        """
        s = self.ctx.settings
        if s.clip_max_sec <= 0:
            raise NotFound("切り出しは無効です(CLIP_MAX_SEC=0)")
        if start_ms < 0 or end_ms <= start_ms:
            raise InvalidInput("start_ms と end_ms は 0 以上で、end_ms を start_ms より大きくしてください")
        if end_ms - start_ms > s.clip_max_sec * 1000:
            raise InvalidInput(f"切り出せる長さは {s.clip_max_sec} 秒までです(CLIP_MAX_SEC)")
        src = self.servable_source(source_id)
        if src["kind"] in ("image", "document"):
            raise InvalidInput("静止画と文書は切り出せません")
        if not CLIP_SLOTS.acquire(blocking=False):
            raise Overloaded("切り出しが混み合っています。しばらくしてから再度お試しください")
        try:
            out_dir = s.data_dir / "tmp" / "clips"
            out_dir.mkdir(parents=True, exist_ok=True)
            ext, media_type = (".m4a", "audio/mp4") if src["kind"] == "audio" else (".mp4", "video/mp4")
            out = out_dir / f"{os.getpid()}_{threading.get_ident()}_{time.monotonic_ns()}{ext}"
            try:
                media.cut_clip(src["path"], start_ms, end_ms, out, src["kind"], accurate=accurate)
            except Exception:
                out.unlink(missing_ok=True)
                raise
        finally:
            CLIP_SLOTS.release()
        stem = Path(src.get("name") or f"source{source_id}").stem
        name = f"{stem}_{media.format_time(start_ms)}-{media.format_time(end_ms)}{ext}".replace(":", "")
        return out, name, media_type

    def thumbnail(self, window_id: int) -> Path:
        """窓のサムネイル(JPEG)のパス。取り込み時に作っていなければ(以前の版で取り込んだものなど)、ここで作ります。"""
        ctx = self.ctx
        found = thumbs.find(ctx.thumb_dir, window_id)
        thumb = found or thumbs.path_for(ctx.thumb_dir, window_id)
        window = ctx.store.get_window(window_id)
        src = ctx.store.get_source(window["source_id"]) if window else None
        if not window or not src:
            raise NotFound(f"window_id={window_id} の窓が見つかりません")
        if not media_stored(src):
            raise NotFound("この取り込み元は、画像を保存しない設定(store_media=false)で取り込まれています")
        if found:
            return found
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
                "batch_size": e.batch_size,  # まとめて推論する件数(EMBED_BATCH)
                "runtime": e.runtime,
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
                "running_jobs": ctx.ingestor.running,
                "ingest_workers": ctx.ingestor.workers,
            },
            "watch": ctx.watcher.status,
            "streams": self._stream_summary(),
            "retention": dict(ctx.retention.status) if ctx.retention else None,
            "backup": dict(ctx.backups.status) if ctx.backups else None,
            "disk": disk.status(s.data_dir, s.min_free_disk_mb),
        }
