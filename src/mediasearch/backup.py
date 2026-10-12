"""バックアップ: 動かしたまま、DB とベクトルの整合した複製を取り、停止中に戻します。

DATA_DIR のファイルを動作中にそのまま複製すると、SQLite(WAL)とベクトル DB の書き込みの途中を写し、
戻したときに壊れている・窓とベクトルが食い違う、ということが起こり得ます。そのため、Store の排他の中で
SQLite のバックアップの機能とベクトルの書き出しを行い、両方がそろった時点の複製を作ります(`Store.backup_to`)。

- 保存先: BACKUP_DIR(既定 DATA_DIR/backups)/<日時>/ に mediasearch.db・vectors.usearch・manifest.json。
  同じディスクに置くと、ディスクの故障では一緒に失われるため、別のディスク・NAS をマウントして指定するのが望ましい。
- 作り方: POST /api/admin/backup、または BACKUP_INTERVAL_HOURS ごとに自動で。BACKUP_KEEP 個を超えた古いものは削除する。
- サムネイル(thumbs/)とアップロードした動画(media/)は、書いた後に変更しないファイルなので、動作中に rsync などで
  そのまま複製できます(バックアップには含めない。容量が大きく、サムネイルは元の動画から作り直せるため)。
- 戻し方: 停止してから `python -m mediasearch.backup restore <バックアップのフォルダ>`(docs/operations.md)。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from . import __version__, disk

log = logging.getLogger(__name__)

FILES = ("mediasearch.db", "vectors.usearch")
MANIFEST = "manifest.json"


class BackupError(RuntimeError):
    """バックアップを作れない・戻せないときのエラー(API では 409)。"""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def read_manifest(folder: Path) -> dict | None:
    try:
        return json.loads((folder / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


class BackupManager:
    """バックアップの作成・一覧・世代の管理と、定期的な作成。"""

    def __init__(self, settings, store) -> None:
        self._s = settings
        self._store = store
        self._lock = threading.Lock()  # 同時に 2 つ作らない(排他の中で DB 全体を写すため、重ねると取り込みが長く止まる)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.status: dict = {"last": None, "last_error": None}

    @property
    def root(self) -> Path:
        return self._s.backup_dir

    def create(self) -> dict:
        """バックアップを 1 つ作り、その情報(manifest)を返します。"""
        if not self._lock.acquire(blocking=False):
            raise BackupError("ほかのバックアップを作成中です。終わってから再度実行してください")
        try:
            return self._create()
        except BackupError:
            raise
        except Exception as exc:
            self.status["last_error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._lock.release()

    def _create(self) -> dict:
        db = self._store.path
        need_mb = (_size(db) + _size(db.with_name(db.name + "-wal")) + _size(self._store.vectors.path)) / 2**20
        self.root.mkdir(parents=True, exist_ok=True)
        free_mb = shutil.disk_usage(self.root).free / 2**20
        if free_mb - need_mb < self._s.min_free_disk_mb:
            raise disk.InsufficientStorage(
                f"バックアップの保存先の空き容量が足りません(空き {free_mb:.0f} MB、必要 約 {need_mb:.0f} MB、"
                f"下限 MIN_FREE_DISK_MB={self._s.min_free_disk_mb} MB): {self.root}"
            )
        name = datetime.now().strftime("%Y%m%d-%H%M%S")
        final = self.root / name
        n = 1
        while final.exists():  # 1 秒以内に続けて作った場合
            final = self.root / f"{name}-{n}"
            n += 1
        tmp = self.root / f".{final.name}.partial"  # 途中で止まっても、作りかけを完成品と見間違えないように
        shutil.rmtree(tmp, ignore_errors=True)
        started = time.monotonic()
        try:
            info = self._store.backup_to(tmp)
            files = {f: {"bytes": _size(tmp / f), "sha256": _sha256(tmp / f)} for f in FILES if (tmp / f).exists()}
            manifest = {
                "name": final.name,
                "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "created_ts": time.time(),  # 並べ替え用(同じ秒に作ると、フォルダ名の順と作った順が一致しないことがある)
                "app_version": __version__,
                "schema_version": self._store.get_meta("schema_version"),
                "model_id": self._store.get_meta("model_id"),
                "dims": self._store.get_meta("dims"),
                "windows": info["windows"],
                "files": files,
                "duration_sec": round(time.monotonic() - started, 2),
            }
            (tmp / MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, final)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        manifest["path"] = str(final)
        manifest["bytes"] = sum(f["bytes"] for f in manifest["files"].values())
        self.status.update(last=manifest, last_error=None)
        log.info("バックアップを作成しました: %s(窓 %d 件、%.1f 秒)", final, info["windows"], manifest["duration_sec"])
        self.prune()
        return manifest

    def list(self) -> list[dict]:
        """作成済みのバックアップ(新しい順)。"""
        if not self.root.is_dir():
            return []
        out = []
        for folder in self.root.iterdir():
            if not folder.is_dir() or folder.name.startswith("."):
                continue
            m = read_manifest(folder)
            if m is None:
                continue
            m["path"] = str(folder)
            m["bytes"] = sum(f.get("bytes", 0) for f in m.get("files", {}).values())
            m.setdefault("created_ts", folder.stat().st_mtime)
            out.append(m)
        out.sort(key=lambda m: m["created_ts"], reverse=True)
        return out

    def last(self) -> dict | None:
        """最後に作ったバックアップ(起動の後に作っていなければ、保存先から探す)。"""
        if self.status["last"] is None:
            found = self.list()[:1]
            self.status["last"] = found[0] if found else None
        return self.status["last"]

    def prune(self) -> int:
        """BACKUP_KEEP 個を超えた古いバックアップを削除します。"""
        keep = self._s.backup_keep
        if keep <= 0:
            return 0
        removed = 0
        for m in self.list()[keep:]:
            shutil.rmtree(m["path"], ignore_errors=True)
            removed += 1
        return removed

    # ------------------------------------------------------------------ 定期的な作成
    def start(self) -> None:
        if self._s.backup_interval_hours <= 0:
            return
        self._thread = threading.Thread(target=self._loop, name="backup", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=30)

    def _loop(self) -> None:
        interval = self._s.backup_interval_hours * 3600
        while True:
            latest = self.list()[:1]
            # 再起動しても間隔を保つよう、最後のバックアップの時刻から数える
            last = Path(latest[0]["path"]).stat().st_mtime if latest else 0.0
            wait = max(last + interval - time.time(), 60.0 if not latest else 0.0)
            if self._stop.wait(wait):
                return
            try:
                self.create()
            except Exception as exc:
                self.status["last_error"] = f"{type(exc).__name__}: {exc}"
                log.warning("自動のバックアップに失敗しました: %s", exc)
                if self._stop.wait(600):  # 失敗したら 10 分後に再試行する
                    return


# ---------------------------------------------------------------------- 戻す(停止中に実行する)
def verify(folder: Path) -> dict:
    """バックアップのファイルが、作成時から変わっていないか(SHA-256)を確かめます。"""
    m = read_manifest(folder)
    if m is None:
        raise BackupError(f"バックアップではありません({MANIFEST} がありません): {folder}")
    for name, meta in m.get("files", {}).items():
        path = folder / name
        if not path.is_file():
            raise BackupError(f"ファイルがありません: {path}")
        if _sha256(path) != meta["sha256"]:
            raise BackupError(f"ファイルが壊れています(SHA-256 が一致しません): {path}")
    if "mediasearch.db" not in m.get("files", {}):
        raise BackupError(f"DB のファイルがありません: {folder}")
    return m


def restore(folder: Path, data_dir: Path) -> Path:
    """バックアップを DATA_DIR に戻します。今のファイルは DATA_DIR/backups/pre-restore-<日時>/ に移します。

    サーバーを止めてから実行してください(動作中に置き換えると、動作中のプロセスが古いファイルに書き続ける)。
    ジャーナル(vectors.journal)と SQLite の WAL も移す(残すと、戻した DB に別の時点の変更が混ざるため)。
    """
    verify(folder)
    data_dir.mkdir(parents=True, exist_ok=True)
    saved = data_dir / "backups" / f"pre-restore-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    current = ["mediasearch.db", "mediasearch.db-wal", "mediasearch.db-shm", "vectors.usearch", "vectors.journal"]
    for name in current:
        path = data_dir / name
        if path.exists():
            saved.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), saved / name)
    for name in FILES:
        if (folder / name).is_file():
            shutil.copy2(folder / name, data_dir / name)
    _drop_newer_thumbs(data_dir)
    return saved


def _drop_newer_thumbs(data_dir: Path) -> int:
    """バックアップの後に作られた窓のサムネイルを削除します。

    戻した DB は、窓の ID をバックアップの時点の続きから振るため、残しておくと、新しい窓に別の場面の
    サムネイルが表示されてしまう。
    """
    import sqlite3

    con = sqlite3.connect(data_dir / "mediasearch.db")
    try:
        row = con.execute("SELECT seq FROM sqlite_sequence WHERE name='windows'").fetchone()
    except sqlite3.Error:
        row = None
    finally:
        con.close()
    last = int(row[0]) if row else 0
    removed = 0
    for path in (data_dir / "thumbs").rglob("w*.jpg"):
        try:
            wid = int(path.stem[1:])
        except ValueError:
            continue
        if wid > last:
            path.unlink(missing_ok=True)
            removed += 1
    return removed


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m mediasearch.backup", description="バックアップの確認と、DATA_DIR への復元"
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify", help="バックアップのファイルが壊れていないかを確かめる")
    v.add_argument("folder", type=Path)
    r = sub.add_parser("restore", help="バックアップを DATA_DIR に戻す(サーバーを止めてから実行する)")
    r.add_argument("folder", type=Path)
    r.add_argument("--data-dir", type=Path, default=Path(os.environ.get("DATA_DIR", "/data")))
    args = p.parse_args(argv)
    try:
        if args.cmd == "verify":
            m = verify(args.folder)
            print(f"問題ありません: {args.folder}(作成 {m.get('created_at')}、窓 {m.get('windows')} 件)")
        else:
            saved = restore(args.folder, args.data_dir)
            print(f"{args.folder} を {args.data_dir} に戻しました。")
            if saved.exists():
                print(f"戻す前のファイルは {saved} に移しました(不要になったら削除してください)。")
            print("サムネイル(thumbs/)・アップロードした動画(media/)を別に複製している場合は、それも戻してください。")
    except BackupError as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
