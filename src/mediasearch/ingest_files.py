"""ファイルの取り込み受付(検証・重複確認・キュー投入)。

アップロード、コンテナ内パスの指定、フォルダの一括取り込み、監視フォルダの自動取り込みの
どれも、ここを通してキューに入れます。受付の時点で次を確認し、だめなら理由をすぐ返します
(バックグラウンドのジョブが失敗してから気づく、という手戻りを減らすため)。

  - INGEST_ROOTS で許可したフォルダの下にあるか(コンテナ内の任意のファイルを登録させない)
  - 拡張子が映像・音声のものか(設定ファイルなど、メディア以外のファイルを登録しない)
  - ffprobe で読めて、必要なトラック(映像 / 音声)があり、長さが 0 でないか
  - 同じファイルを、すでに取り込んでいないか(重複した窓で検索結果が埋まるのを防ぐ)
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from . import media
from .media import MediaError
from .pipeline import Ingestor, IngestParams
from .store import Store

log = logging.getLogger(__name__)

# 取り込める拡張子。録画でよく使われる形式を挙げています
VIDEO_EXTS = frozenset(
    {".mp4", ".m4v", ".mov", ".mkv", ".avi", ".ts", ".mts", ".m2ts", ".webm", ".flv", ".wmv", ".3gp"}
)
AUDIO_EXTS = frozenset({".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".oga", ".opus", ".wma"})

# ファイル名に含まれる日時。例: 20260101_090000 / 2026-01-01T09-00-00 / cam01-20260101-090000.mp4
_TS_PATTERN = re.compile(
    r"(?<!\d)(20\d{2})[-_]?(\d{2})[-_]?(\d{2})[T_\- ]?(\d{2})[-_:]?(\d{2})[-_:]?(\d{2})(?!\d)"
)


class PathNotAllowed(MediaError):
    """INGEST_ROOTS の外のパス(API では 403)。"""


def is_within(path: str | Path, roots: tuple[Path, ...]) -> bool:
    """path が roots のどれかの下にあるか。シンボリックリンクと .. は解決してから比べます。"""
    resolved = Path(path).resolve()
    return any(resolved.is_relative_to(Path(r).resolve()) for r in roots)


def media_kind_of(path: str | Path) -> str | None:
    """拡張子から、映像(video)か音声(audio)かを判定します。どちらでもなければ None。"""
    ext = Path(path).suffix.lower()
    if ext in VIDEO_EXTS:
        return "video"
    if ext in AUDIO_EXTS:
        return "audio"
    return None


def ts_from_filename(name: str) -> float | None:
    """ファイル名から録画開始の日時を読み取り、UNIX 秒で返します。読み取れなければ None。

    多くの録画システムは録画ファイル名に開始日時を入れるため、start_ts を省略したときの既定に使います。
    タイムゾーンはコンテナの設定(TZ、既定は Asia/Tokyo)として解釈します。
    """
    m = _TS_PATTERN.search(Path(name).name)
    if not m:
        return None
    try:
        return datetime(*(int(g) for g in m.groups())).timestamp()
    except ValueError:  # 13月など、日時として成り立たない数字の並び
        return None


@dataclass(frozen=True)
class Accepted:
    """受付の結果。duplicate が真なら、新しいジョブは作っていません。"""

    source_id: int
    job_id: int | None
    duplicate: bool
    start_ts: float
    start_ts_from: str  # request(指定) / filename(ファイル名) / now(受付時刻)

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "job_id": self.job_id,
            "duplicate": self.duplicate,
            "start_ts": datetime.fromtimestamp(self.start_ts).isoformat(timespec="seconds"),
            "start_ts_from": self.start_ts_from,
        }


class FileIntake:
    """ファイルを検証し、取り込みキューに入れます。"""

    def __init__(self, store: Store, ingestor: Ingestor, roots: tuple[Path, ...]) -> None:
        self._store = store
        self._ingestor = ingestor
        self._roots = roots
        # 重複の確認から登録までを直列にする(監視フォルダの走査と API が同じファイルを同時に受け付けないように)
        self._lock = threading.Lock()

    def check_allowed(self, path: Path) -> None:
        """許可したフォルダの外なら PathNotAllowed を送出します。"""
        if not is_within(path, self._roots):
            allowed = ", ".join(str(r) for r in self._roots)
            raise PathNotAllowed(
                f"取り込めるフォルダの外です: {path}"
                f"(許可しているフォルダ: {allowed}。環境変数 INGEST_ROOTS で変更できます)"
            )

    def validate(self, path: Path, kind: str) -> media.MediaInfo:
        """取り込めるファイルか確かめます。だめなら MediaError(API では 422 か 403)を送出します。"""
        self.check_allowed(path)
        expected = media_kind_of(path)
        if expected is None:
            raise MediaError(
                f"映像・音声として扱えない拡張子です: {path.name}"
                f"(対応: {', '.join(sorted(VIDEO_EXTS | AUDIO_EXTS))})"
            )
        info = media.probe(path)
        if kind == "video" and not info.has_video:
            raise MediaError(f"映像トラックがありません: {path.name}(音声だけなら kind=audio で取り込んでください)")
        if kind == "audio" and not info.has_audio:
            raise MediaError(f"音声トラックがありません: {path.name}")
        if info.duration_ms <= 0:
            raise MediaError(f"長さが 0 です: {path.name}")
        return info

    def accept(
        self,
        path: Path,
        *,
        kind: str,
        name: str | None = None,
        camera_id: str | None = None,
        location: str | None = None,
        start_ts: float | None = None,
        params: IngestParams,
        force: bool = False,
        retry_failed: bool = True,
    ) -> Accepted:
        """1つのファイルを受け付けます。

        force が偽なら、同じパス・同じ種類で取り込み済み(または処理中)のものは重複として扱い、
        既存の取り込み元を返します。前回失敗したものは、retry_failed が真なら作り直します
        (監視フォルダでは偽にして、壊れたファイルを繰り返し処理しないようにします)。
        """
        path = path.resolve()
        name = name or path.name
        if start_ts is not None:
            ts, ts_from = start_ts, "request"
        elif (guessed := ts_from_filename(name)) is not None:
            ts, ts_from = guessed, "filename"
        else:
            ts, ts_from = time.time(), "now"

        with self._lock:
            if not force:
                for src in self._store.find_sources_by_path(str(path)):
                    if src["kind"] != kind:
                        continue
                    if src["status"] == "failed" and retry_failed:
                        self._store.delete_source(src["id"])
                        continue
                    return Accepted(src["id"], None, True, src["start_ts"], "existing")

            self.validate(path, kind)
            source_id = self._store.add_source(
                kind=kind,
                path=str(path),
                name=name,
                camera_id=camera_id or None,
                location=location or None,
                start_ts=ts,
                params=params.to_dict(),
            )
        return Accepted(source_id, self._ingestor.submit(source_id), False, ts, ts_from)

    def accept_dir(
        self,
        root: Path,
        *,
        params: IngestParams,
        recursive: bool = True,
        kind: str = "auto",
        camera_id: str | None = None,
        camera_from_dir: bool = False,
        location: str | None = None,
        force: bool = False,
        retry_failed: bool = True,
        settle_sec: float = 0,
        skip_paths: set[str] | None = None,
    ) -> dict:
        """フォルダの中の映像・音声ファイルを、まとめて受け付けます。

        camera_from_dir が真なら、ファイルが入っているフォルダの名前をカメラ ID にします
        (cam01/20260101_090000.mp4 → cam01)。root 直下のファイルは camera_id を使います。
        settle_sec を指定すると、最終更新から指定秒数たっていないファイル(録画中の可能性がある)を後回しにします。
        """
        root = root.resolve()
        self.check_allowed(root)
        if not root.is_dir():
            raise MediaError(f"フォルダが見つかりません: {root}(コンテナ内のパスで指定してください)")
        now = time.time()
        queued: list[dict] = []
        duplicates = pending = 0
        errors: list[dict] = []
        for path in sorted(_iter_files(root, recursive)):
            file_kind = media_kind_of(path)
            if file_kind is None or (kind != "auto" and kind != file_kind):
                continue
            if skip_paths is not None and str(path) in skip_paths:
                duplicates += 1
                continue
            try:
                if settle_sec and now - path.stat().st_mtime < settle_sec:
                    pending += 1
                    continue
                cam = camera_id
                if camera_from_dir and path.parent != root:
                    cam = path.parent.name
                result = self.accept(
                    path,
                    kind=file_kind,
                    camera_id=cam,
                    location=location,
                    params=params,
                    force=force,
                    retry_failed=retry_failed,
                )
            except (MediaError, OSError) as exc:
                errors.append({"path": str(path), "error": str(exc)})
                continue
            if result.duplicate:
                duplicates += 1
            else:
                queued.append({"path": str(path), "camera_id": cam, **result.to_dict()})
        return {
            "dir": str(root),
            "queued": len(queued),
            "duplicates": duplicates,
            "pending": pending,
            "errors": errors,
            "items": queued,
        }


def _iter_files(root: Path, recursive: bool):
    """隠しファイル・隠しフォルダ(録画ソフトの作業用など)は除いて、ファイルを列挙します。"""
    if not recursive:
        yield from (p for p in root.iterdir() if p.is_file() and not p.name.startswith("."))
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for f in filenames:
            if not f.startswith("."):
                yield Path(dirpath) / f
