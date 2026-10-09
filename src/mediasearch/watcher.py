"""監視フォルダの自動取り込み。

録画システムは録画を一定時間ごとのファイルに区切って書き出すことが多いため、録画フォルダを定期的に見て、
新しいファイルを自動でキューに入れます(WATCH_DIRS を設定したときだけ動きます)。

  - 書き込み中のファイルを取り込まないよう、最終更新から WATCH_SETTLE_SEC 秒たったものだけを対象にします。
  - 一度登録したファイル(失敗したものを含む)は、再び自動では取り込みません。
    壊れたファイルを毎回処理し直すのを避けるためです。やり直すときは WebUI か API の再取り込みを使います。
  - ファイルシステムの通知(inotify)ではなく定期的な走査にしています。
    Docker のバインドマウントや NAS(SMB / NFS)では通知が届かないことがあるためです。
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from .config import Settings
from .ingest_files import FileIntake
from .pipeline import IngestParams
from .store import Store

log = logging.getLogger(__name__)


class FolderWatcher:
    def __init__(self, settings: Settings, store: Store, intake: FileIntake) -> None:
        self._s = settings
        self._store = store
        self._intake = intake
        self._stop = threading.Event()
        self._scan_lock = threading.Lock()  # 定期の走査と、API からの手動の走査を同時に走らせない
        self._thread: threading.Thread | None = None
        self._status: dict = {
            "enabled": bool(settings.watch_dirs),
            "dirs": [str(d) for d in settings.watch_dirs],
            "interval_sec": settings.watch_interval_sec,
            "settle_sec": settings.watch_settle_sec,
            "camera_from_dir": settings.watch_camera_from_dir,
            "preset": settings.watch_preset or None,
            "last_scan_at": None,
            "last_queued": 0,
            "total_queued": 0,
            "pending": 0,
            "errors": [],
        }

    @property
    def status(self) -> dict:
        return dict(self._status)

    def start(self) -> None:
        if not self._s.watch_dirs:
            return
        # 設定の誤りは起動時に分かるようにする(窓の設定が不正なら、ここで例外になる)
        self._params()
        for d in self._s.watch_dirs:
            if not d.is_dir():
                log.warning("監視フォルダが見つかりません(作成されるまで待ちます): %s", d)
        log.info(
            "監視フォルダの自動取り込みを始めます: %s(%d 秒ごと)", self._status["dirs"], self._s.watch_interval_sec
        )
        self._thread = threading.Thread(target=self._loop, name="watcher", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _params(self) -> IngestParams:
        return IngestParams.from_settings(self._s, preset=self._s.watch_preset or None)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.scan_once()
            except Exception:  # 走査の失敗で監視自体が止まらないようにする
                log.exception("監視フォルダの走査に失敗しました")
            self._stop.wait(self._s.watch_interval_sec)

    def scan_once(self) -> int:
        """全ての監視フォルダを1回走査し、新しくキューに入れた件数を返します。"""
        with self._scan_lock:
            return self._scan()

    def _scan(self) -> int:
        params = self._params()
        known = self._store.known_paths()
        queued = pending = 0
        errors: list[dict] = []
        for d in self._s.watch_dirs:
            if not Path(d).is_dir():
                errors.append({"path": str(d), "error": "フォルダが見つかりません"})
                continue
            result = self._intake.accept_dir(
                Path(d),
                params=params,
                camera_from_dir=self._s.watch_camera_from_dir,
                retry_failed=False,
                settle_sec=self._s.watch_settle_sec,
                skip_paths=known,
            )
            queued += result["queued"]
            pending += result["pending"]
            errors += result["errors"]
            for item in result["items"]:
                log.info("監視フォルダの新しいファイルを取り込みます: %s", item["path"])
        for e in errors:
            log.warning("監視フォルダのファイルを取り込めません: %s: %s", e["path"], e["error"])
        self._status.update(
            last_scan_at=time.time(),
            last_queued=queued,
            total_queued=self._status["total_queued"] + queued,
            pending=pending,
            errors=errors[-20:],  # 状態表示が大きくなりすぎないよう、直近のものだけ残す
        )
        return queued
