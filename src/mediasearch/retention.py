"""保存期間(RETENTION_DAYS)を過ぎたデータの定期削除。

ストリームや監視フォルダは、録画が続く限り窓が増え続けます。ベクトルはメモリにも載るため(768 次元で 1 件約 3.2KB)、
2 秒窓のストリーム 1 本で 1 日約 140MB ずつ増えます。保存期間を決めて、古い窓を定期的に削除します。

- 基準は録画の時刻(窓の abs_ts)です。取り込んだ時刻ではありません。
  古い録画を取り込んでも、期間外ならすぐに削除されます。
- 削除するもの: 窓(SQLite)、ベクトル(ベクトル DB)、サムネイル、窓が無くなった取り込み元のアップロードしたファイル。
  パス指定・監視フォルダの元のファイルは利用者のものなので削除しません。
- ベクトル DB は削除した場所を次の追加で使い直すため、保存期間を設定すれば、メモリの使用量は一定の範囲に収まります。
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from .config import Settings
from .store import Store

log = logging.getLogger(__name__)

INTERVAL_SEC = 600  # 削除を確認する間隔


class RetentionWorker:
    def __init__(self, settings: Settings, store: Store) -> None:
        self._s = settings
        self._store = store
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.status: dict = {
            "days": settings.retention_days or None,
            "interval_sec": INTERVAL_SEC,
            "last_run_at": None,
            "last_deleted_windows": 0,
            "total_deleted_windows": 0,
            "total_expired_sources": 0,
            "job_history_days": settings.job_history_days or None,
            "total_pruned_jobs": 0,
            "error": None,
        }

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="retention", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.run_once()
            except Exception as exc:  # 次の回に再試行する
                self.status["error"] = f"{type(exc).__name__}: {exc}"
                log.exception("保存期間を過ぎたデータの削除に失敗しました")
            self._stop.wait(INTERVAL_SEC)

    def run_once(self, now: float | None = None) -> dict:
        """1 回分の削除を行い、結果を返します(テストや手動の実行にも使う)。"""
        with self._lock:
            now = time.time() if now is None else now
            days = self._s.retention_days
            cutoff = now - days * 86400 if days and days > 0 else None
            result = self._store.expire_windows(cutoff, now)
            if self._s.job_history_days > 0:  # 保存期間の設定とは別に、ジョブの履歴は既定で 30 日分だけ残す
                before = now - self._s.job_history_days * 86400
                self.status["total_pruned_jobs"] += self._store.prune_jobs(before)
                self._store.prune_alert_events(before)  # 送り終えた通知の履歴も、同じ期間で削除する
            thumbs = self._s.data_dir / "thumbs"
            for wid in result["window_ids"]:
                (thumbs / f"w{wid}.jpg").unlink(missing_ok=True)
            uploads = (self._s.data_dir / "media").resolve()
            for src in result["expired_sources"]:
                path = Path(src["path"]) if src.get("path") else None
                if path and path.resolve().is_relative_to(uploads):
                    path.unlink(missing_ok=True)
                    self._store.update_source(src["id"], path=None)
            n = len(result["window_ids"])
            self.status.update(
                last_run_at=now,
                last_deleted_windows=n,
                total_deleted_windows=self.status["total_deleted_windows"] + n,
                total_expired_sources=self.status["total_expired_sources"] + len(result["expired_sources"]),
                error=None,
            )
            if n:
                log.info("保存期間を過ぎた窓を %d 件削除しました(取り込み元 %d 件が期限切れ)", n,
                         len(result["expired_sources"]))  # fmt: skip
            return {"deleted_windows": n, "expired_sources": [s["id"] for s in result["expired_sources"]]}
