"""メタデータ(SQLite)とベクトル(USearch)の保存、およびベクトル検索。

- 取り込み元・窓・ジョブ・設定は SQLite(DATA_DIR/mediasearch.db)に保存します。
- ベクトルは、組み込み型のベクトル DB の USearch(DATA_DIR/vectors.usearch。vectors.py)に、
  窓の ID をキーにして保存します。
- グループ ID や時刻などのメタデータは、ベクトルに埋め込まず、SQLite の列として保存して絞り込みに使います。
- 以前の版(ベクトルを SQLite の BLOB に保存していた)の DB は、起動時にベクトルを USearch へ移し、
  SQLite から列を削除します。
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import thumbs
from .vectors import IndexMismatch, VectorIndex

__all__ = ["IndexMismatch", "SchemaError", "Store", "WINDOW_KINDS", "SCHEMA_VERSION"]
log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sources (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  kind TEXT NOT NULL,            -- video / audio / image / frames / stream
  path TEXT,                     -- 映像・音声ファイルのパス(frames は NULL)
  name TEXT NOT NULL,
  group_id TEXT,
  location TEXT,
  start_ts REAL NOT NULL,        -- 録画開始の時刻(UNIX 秒)
  duration_ms INTEGER,
  status TEXT NOT NULL,          -- queued / running / done / failed / cancelled / expired(保存期間を過ぎ、窓を削除した)
  params TEXT NOT NULL,          -- 取り込み設定(JSON)
  error TEXT,
  created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS windows (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
  start_ms INTEGER NOT NULL,
  end_ms INTEGER NOT NULL,
  kind TEXT NOT NULL,            -- frames(映像のみ) / tav(映像+音声) / audio(音声のみ) / image(静止画 1 枚)
  group_id TEXT,
  abs_ts REAL NOT NULL           -- 窓の開始時刻(UNIX 秒) = source.start_ts + start_ms / 1000
);

CREATE TABLE IF NOT EXISTS jobs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
  status TEXT NOT NULL,          -- queued / running / done / failed
  progress INTEGER NOT NULL DEFAULT 0,
  total INTEGER NOT NULL DEFAULT 0,
  error TEXT,
  created_at REAL NOT NULL,
  started_at REAL,
  finished_at REAL,
  timings TEXT,                  -- 処理時間の内訳(JSON。ベンチマーク用)
  attempts INTEGER NOT NULL DEFAULT 0  -- 一時的な失敗で再試行した回数
);
"""

# RTSP などのストリームの登録(streams.py)。url は認証情報を含むことがあるため、API では伏せて返す
STREAMS_SCHEMA = """
CREATE TABLE IF NOT EXISTS streams (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  url TEXT NOT NULL,
  group_id TEXT,
  location TEXT,
  params TEXT NOT NULL,          -- 取り込みの設定(JSON)
  source_id INTEGER REFERENCES sources(id) ON DELETE SET NULL,
  enabled INTEGER NOT NULL DEFAULT 1,
  created_at REAL NOT NULL,
  retention_hours REAL           -- 窓を残す時間。NULL なら RETENTION_DAYS に従い、0 なら削除しない
);
"""

# 通知ルール(alerts.py)。検索の条件と通知先(Webhook)を保存し、新しい窓が条件に合ったら通知する
ALERTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS alerts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  query TEXT NOT NULL,           -- 探す内容を表す文章
  kind TEXT NOT NULL DEFAULT 'auto',
  group_id TEXT,
  location TEXT,
  min_score REAL NOT NULL,
  webhook_url TEXT NOT NULL,
  secret TEXT NOT NULL,          -- 通知の署名(HMAC-SHA256)の鍵
  cooldown_sec REAL NOT NULL DEFAULT 60,  -- 同じ取り込み元からの通知を、この秒数は送らない
  enabled INTEGER NOT NULL DEFAULT 1,
  created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS alert_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  alert_id INTEGER NOT NULL REFERENCES alerts(id) ON DELETE CASCADE,
  window_id INTEGER NOT NULL,
  source_id INTEGER NOT NULL,
  score REAL NOT NULL,
  payload TEXT NOT NULL,         -- 送る内容(JSON)
  status TEXT NOT NULL,          -- pending / delivered / failed
  attempts INTEGER NOT NULL DEFAULT 0,
  error TEXT,
  created_at REAL NOT NULL,
  delivered_at REAL,
  UNIQUE(alert_id, window_id)    -- 同じ窓を二重に通知しない(再起動の前後で照合が重なっても)
);
"""

# 索引は、以前の版の DB を移行した後に作る(移行前の列名のままだと作れないため)
INDEXES = """
-- 同じファイルの重複取り込みを確認するため(監視フォルダでは定期的に全件を照合する)
CREATE INDEX IF NOT EXISTS idx_sources_path ON sources(path);
CREATE INDEX IF NOT EXISTS idx_windows_source ON windows(source_id);
-- 保存期間(RETENTION_DAYS)を過ぎた窓を探すため
CREATE INDEX IF NOT EXISTS idx_windows_abs_ts ON windows(abs_ts);
CREATE INDEX IF NOT EXISTS idx_alert_events_status ON alert_events(status);
"""

WINDOW_KINDS = ("frames", "tav", "audio", "image")

# DB の形式の版。列の追加・改名をしたら上げ、_migrate() に移行の処理を足す
SCHEMA_VERSION = 8
# 旧い版で使っていた列名 → 現在の列名(以前の版の DB をそのまま使い続けられるよう、起動時に改名する)
_RENAMED_COLUMNS = {
    "sources": {"camera_id": "group_id"},
    "windows": {"camera_id": "group_id"},
}
# 後の版で追加した列(以前の版の DB には、起動時に ALTER TABLE で足す)。版 3: jobs.timings
_ADDED_COLUMNS = {
    "jobs": {"timings": "TEXT"},  # 版 3
    "streams": {"retention_hours": "REAL"},  # 版 6
    "jobs#2": {"attempts": "INTEGER NOT NULL DEFAULT 0"},  # 版 7(同じ表への追加は「表名#番号」で書き分ける)
}
_REQUIRED_COLUMNS = {
    "sources": {"id", "kind", "path", "name", "group_id", "location", "start_ts", "status", "params"},
    "windows": {"id", "source_id", "start_ms", "end_ms", "kind", "group_id", "abs_ts"},
    "jobs": {"id", "source_id", "status", "progress", "total"},
}


class SchemaError(RuntimeError):
    """DB の形式が、この版で扱えないときのエラー。"""


class Store:
    def __init__(
        self,
        path: Path,
        vector_path: Path | None = None,
        snapshot_every: int | None = None,
        thumbs_dir: Path | None = None,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # 窓を削除したときに、そのサムネイルも消す(消さないと、取り込み直し・削除のたびにファイルが残り続ける)
        self.thumbs_dir = thumbs_dir or path.parent / "thumbs"
        self._lock = threading.RLock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")  # 書き込み中も検索(読み取り)できる
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.executescript(SCHEMA)
        self._db.executescript(STREAMS_SCHEMA)
        self._db.executescript(ALERTS_SCHEMA)
        self._db.commit()
        self._migrate()
        self._db.executescript(INDEXES)
        self._db.commit()
        self.path = path
        self._cache: dict[str, Any] | None = None
        self._pending: list[tuple] = []  # 検索用のメタデータに、まだ反映していない追加分
        self.vectors = VectorIndex(vector_path or path.with_name("vectors.usearch"), snapshot_every)
        self._move_legacy_vectors()
        # 異常終了で、ベクトルを書き出す前に止まった取り込み元(Ingestor が起動時に取り込み直す)
        self.lost_sources: list[int] = self._reconcile()
        row = self._db.execute("SELECT MAX(id) AS m FROM windows").fetchone()
        self.last_window_id: int = int(row["m"] or 0)  # 最後に追加した窓の ID(リアルタイム検索で新しい窓を見分ける)

    # ------------------------------------------------------------------ DB の形式の移行
    def _columns(self, table: str) -> set[str]:
        return {r["name"] for r in self._db.execute(f"PRAGMA table_info({table})")}

    def _migrate(self) -> None:
        """以前の版で作った DB を、現在の形式に合わせます(起動時に 1 回)。

        CREATE TABLE IF NOT EXISTS は既存の表を変更しないため、列の改名などはここで行います。
        移行できない形式だった場合は、原因が分かるメッセージで起動を止めます
        (そのまま動かすと、取り込みや検索のたびに SQL のエラーで 500 になるため)。
        """
        renamed = []
        with self._db:  # 失敗したら途中の変更を取り消す
            for table, renames in _RENAMED_COLUMNS.items():
                cols = self._columns(table)
                for old, new in renames.items():
                    if old in cols and new not in cols:
                        self._db.execute(f"ALTER TABLE {table} RENAME COLUMN {old} TO {new}")
                        renamed.append(f"{table}.{old} → {new}")
            for key, added in _ADDED_COLUMNS.items():
                table = key.split("#", 1)[0]
                cols = self._columns(table)
                for name, decl in added.items():
                    if cols and name not in cols:
                        self._db.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
                        renamed.append(f"{table}.{name} を追加")
            for table, required in _REQUIRED_COLUMNS.items():
                missing = required - self._columns(table)
                if missing:
                    raise SchemaError(
                        f"DB({table} 表)に必要な列がありません: {', '.join(sorted(missing))}。"
                        "別のアプリの DB か、壊れている可能性があります。DATA_DIR を確認してください"
                    )
            self._db.execute(
                "INSERT INTO meta(key, value) VALUES('schema_version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(SCHEMA_VERSION),),
            )
        if renamed:
            log.warning("以前の版の DB を現在の形式に移行しました: %s", ", ".join(renamed))

    def _move_legacy_vectors(self) -> None:
        """以前の版の DB(windows.vec に BLOB でベクトルを保存)から、ベクトルを USearch へ移します。

        移した後にファイルへ書き出してから SQLite の列を削除するため、途中で止まっても、次の起動でやり直せます。
        """
        if "vec" not in self._columns("windows"):
            return
        moved = 0
        with self._lock:
            rows = self._db.execute("SELECT id, vec FROM windows ORDER BY id")
            while batch := rows.fetchmany(10_000):
                existing = set(self.vectors.keys().tolist())
                keep = [(r["id"], r["vec"]) for r in batch if r["id"] not in existing]
                if keep:
                    vecs = np.stack([np.frombuffer(v, dtype=np.float32) for _, v in keep])
                    self.vectors.add([k for k, _ in keep], vecs, journal=False)  # 最後にまとめて書き出す
                    moved += len(keep)
                    log.warning("以前の版の DB からベクトルを移しています: %d 件(1 件あたり約 0.5 ミリ秒)", moved)
            self.vectors.save(force=True)
            self._db.execute("ALTER TABLE windows DROP COLUMN vec")
            self._db.commit()
            self._db.execute("VACUUM")  # ベクトルが占めていた領域を解放する
        log.warning("以前の版の DB から、%d 件のベクトルをベクトル DB(%s)へ移しました", moved, self.vectors.path.name)

    def _reconcile(self) -> list[int]:
        """SQLite の窓と、ベクトル DB の内容を突き合わせます。

        ベクトルはまとめて書き出すため、書き出す前に異常終了すると、SQLite にだけ窓が残ります。
        その窓は削除し、取り込み元の ID を返します(ファイルがある取り込み元は、Ingestor が取り込み直す)。
        逆に、ベクトル DB にだけ残ったもの(削除の直後に止まった場合)は、ベクトル DB から削除します。
        """
        with self._lock:
            rows = self._db.execute("SELECT id, source_id FROM windows").fetchall()
        in_db = {r["id"]: r["source_id"] for r in rows}
        in_index = set(self.vectors.keys().tolist())
        missing = [wid for wid in in_db if wid not in in_index]
        orphans = [k for k in in_index if k not in in_db]
        if orphans:
            self.vectors.remove(orphans)
            self.vectors.save()
        if not missing:
            return []
        lost = sorted({in_db[w] for w in missing})
        with self._lock:
            self._db.executemany("DELETE FROM windows WHERE id=?", [(w,) for w in missing])
            self._db.commit()
        log.warning(
            "ベクトル DB に無い窓が %d 件ありました(前回、ベクトルを書き出す前に停止した可能性があります)。"
            "該当の取り込み元(%d 件)は、取り込み直します",
            len(missing), len(lost),
        )  # fmt: skip
        return lost

    # ------------------------------------------------------------------ メタ情報
    def get_meta(self, key: str) -> str | None:
        with self._lock:
            row = self._db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO meta(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )
            self._db.commit()

    def ensure_compat(self, model_id: str, dims: int) -> None:
        """モデルと次元が、DB に記録されたものと同じか確認します。

        違うモデルや次元のベクトルを同じ DB に混ぜると、検索結果が意味をなさなくなります。
        初回は記録し、2回目以降は一致を確認します。
        """
        with self._lock:
            saved_model = self.get_meta("model_id")
            saved_dims = self.get_meta("dims")
            if saved_model is None:
                self.set_meta("model_id", model_id)
                self.set_meta("dims", str(dims))
                return
            if saved_model != model_id or saved_dims != str(dims):
                raise IndexMismatch(
                    f"DB には model_id={saved_model}, dims={saved_dims} で作ったベクトルがあります。"
                    f"現在の設定(model_id={model_id}, dims={dims})では、同じ DB を使えません。"
                    "設定を元に戻すか、DATA_DIR を変えて新しい DB を作ってください。"
                )

    # ------------------------------------------------------------------ 取り込み元
    def add_source(
        self,
        *,
        kind: str,
        path: str | None,
        name: str,
        group_id: str | None,
        location: str | None,
        start_ts: float,
        params: dict,
        duration_ms: int | None = None,
        status: str = "queued",
    ) -> int:
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO sources(kind, path, name, group_id, location, start_ts, duration_ms,"
                " status, params, created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    kind,
                    path,
                    name,
                    group_id,
                    location,
                    start_ts,
                    duration_ms,
                    status,
                    json.dumps(params, ensure_ascii=False),
                    time.time(),
                ),
            )
            self._db.commit()
            return int(cur.lastrowid)

    def get_source(self, source_id: int) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM sources WHERE id=?", (source_id,)).fetchone()
        return self._source_dict(row) if row else None

    def list_sources(self, limit: int = 200) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT s.*, (SELECT COUNT(*) FROM windows w WHERE w.source_id=s.id) AS window_count"
                " FROM sources s ORDER BY s.id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._source_dict(r) for r in rows]

    def query_sources(
        self,
        *,
        offset: int = 0,
        limit: int = 100,
        status: str | None = None,
        kind: str | None = None,
        group_id: str | None = None,
        name: str | None = None,
    ) -> tuple[list[dict], int]:
        """絞り込みとページ送りをした取り込み元の一覧(新しい順)と、条件に合う総数。"""
        where, params = [], []
        for col, val in (("status", status), ("kind", kind), ("group_id", group_id)):
            if val:
                where.append(f"s.{col} = ?")
                params.append(val)
        if name:
            where.append("s.name LIKE ? ESCAPE '\\'")
            params.append("%" + name.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%")
        cond = (" WHERE " + " AND ".join(where)) if where else ""
        with self._lock:
            total = self._db.execute(f"SELECT COUNT(*) AS n FROM sources s{cond}", params).fetchone()["n"]
            rows = self._db.execute(
                "SELECT s.*, (SELECT COUNT(*) FROM windows w WHERE w.source_id=s.id) AS window_count"
                f" FROM sources s{cond} ORDER BY s.id DESC LIMIT ? OFFSET ?",
                (*params, limit, offset),
            ).fetchall()
        return [self._source_dict(r) for r in rows], int(total)

    def update_source_meta(
        self, source_id: int, *, name: str | None = None, group_id: str | None = None, location: str | None = None,
        clear: tuple[str, ...] = (),
    ) -> None:
        """取り込み元の名前・グループ ID・場所を変えます。グループ ID は窓にも写す(検索の絞り込みに使うため)。

        clear に列の名前を入れると、その列を空(NULL)にします。
        """
        fields = {k: v for k, v in (("name", name), ("group_id", group_id), ("location", location)) if v is not None}
        for col in clear:
            if col in ("group_id", "location"):
                fields[col] = None
        if not fields:
            return
        with self._lock:
            sets = ", ".join(f"{k}=?" for k in fields)
            self._db.execute(f"UPDATE sources SET {sets} WHERE id=?", (*fields.values(), source_id))
            if "group_id" in fields:
                self._db.execute("UPDATE windows SET group_id=? WHERE source_id=?", (fields["group_id"], source_id))
            self._db.commit()
            self._cache = None  # 絞り込みに使う値が変わるため、次の検索で読み直す

    def find_sources_by_path(self, path: str) -> list[dict]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM sources WHERE path=? ORDER BY id", (path,)).fetchall()
        return [self._source_dict(r) for r in rows]

    def known_paths(self) -> set[str]:
        """登録済みのファイルのパス(状態を問わない)。監視フォルダで新しいファイルを見分けるのに使います。"""
        with self._lock:
            rows = self._db.execute("SELECT DISTINCT path FROM sources WHERE path IS NOT NULL").fetchall()
        return {r["path"] for r in rows}

    def update_source(self, source_id: int, **fields: Any) -> None:
        allowed = {"status", "error", "duration_ms", "params", "path"}
        assert set(fields) <= allowed, f"更新できない列です: {set(fields) - allowed}"
        if "params" in fields:
            fields["params"] = json.dumps(fields["params"], ensure_ascii=False)
        sets = ", ".join(f"{k}=?" for k in fields)
        with self._lock:
            self._db.execute(f"UPDATE sources SET {sets} WHERE id=?", (*fields.values(), source_id))
            self._db.commit()

    def delete_source(self, source_id: int) -> None:
        with self._lock:
            ids = self._window_ids(source_id)
            self._db.execute("DELETE FROM sources WHERE id=?", (source_id,))
            self._db.commit()
            self.vectors.remove(ids)
            self._drop_from_cache(ids)
        thumbs.remove(self.thumbs_dir, ids)

    @staticmethod
    def _source_dict(row: sqlite3.Row) -> dict:
        d = dict(row)
        d["params"] = json.loads(d["params"])
        return d

    # ------------------------------------------------------------------ 窓(ベクトル)
    def add_window(
        self,
        *,
        source_id: int,
        start_ms: int,
        end_ms: int,
        kind: str,
        group_id: str | None,
        abs_ts: float,
        vec: np.ndarray,
    ) -> int:
        assert kind in WINDOW_KINDS, kind
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO windows(source_id, start_ms, end_ms, kind, group_id, abs_ts) VALUES(?,?,?,?,?,?)",
                (source_id, start_ms, end_ms, kind, group_id, abs_ts),
            )
            self._db.commit()
            window_id = int(cur.lastrowid)
            self.vectors.add([window_id], vec)  # すぐに検索の対象になる(取り込み中でも探せる)
            if self._cache is not None:
                row = self._db.execute("SELECT location FROM sources WHERE id=?", (source_id,)).fetchone()
                location = row["location"] if row else None
                self._pending.append((window_id, source_id, start_ms, end_ms, kind, group_id, location, abs_ts))
            self.last_window_id = max(self.last_window_id, window_id)
            return window_id

    def _window_ids(self, source_id: int) -> list[int]:
        rows = self._db.execute("SELECT id FROM windows WHERE source_id=?", (source_id,)).fetchall()
        return [r["id"] for r in rows]

    def delete_windows(self, source_id: int) -> int:
        with self._lock:
            ids = self._window_ids(source_id)
            cur = self._db.execute("DELETE FROM windows WHERE source_id=?", (source_id,))
            self._db.commit()
            self.vectors.remove(ids)
            self._drop_from_cache(ids)
        thumbs.remove(self.thumbs_dir, ids)
        return cur.rowcount

    def get_window(self, window_id: int) -> dict | None:
        with self._lock:
            row = self._db.execute(
                "SELECT id, source_id, start_ms, end_ms, kind, group_id, abs_ts FROM windows"
                " WHERE id=?",
                (window_id,),
            ).fetchone()
        return dict(row) if row else None

    def counts(self) -> dict:
        with self._lock:
            by_kind = {
                r["kind"]: r["n"]
                for r in self._db.execute("SELECT kind, COUNT(*) AS n FROM windows GROUP BY kind")
            }
            sources = self._db.execute("SELECT COUNT(*) AS n FROM sources").fetchone()["n"]
        path = self.vectors.path
        vector_db = {
            "engine": "usearch",
            "vectors": len(self.vectors),
            "file_bytes": path.stat().st_size if path.exists() else 0,
        }
        return {
            "sources": sources,
            "windows": sum(by_kind.values()),
            "windows_by_kind": by_kind,
            "vector_db": vector_db,
        }

    # ------------------------------------------------------------------ ストリーム
    def add_stream(
        self,
        *,
        name: str,
        url: str,
        group_id: str | None,
        location: str | None,
        params: dict,
        source_id: int,
        enabled: bool = True,
        retention_hours: float | None = None,
    ) -> int:
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO streams(name, url, group_id, location, params, source_id, enabled, created_at,"
                " retention_hours) VALUES(?,?,?,?,?,?,?,?,?)",
                (name, url, group_id, location, json.dumps(params, ensure_ascii=False), source_id, int(enabled),
                 time.time(), retention_hours),
            )  # fmt: skip
            self._db.commit()
            return int(cur.lastrowid)

    def get_stream(self, stream_id: int) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM streams WHERE id=?", (stream_id,)).fetchone()
        return self._source_dict(row) if row else None

    def list_streams(self) -> list[dict]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM streams ORDER BY id").fetchall()
        return [self._source_dict(r) for r in rows]

    def update_stream(self, stream_id: int, *, enabled: bool) -> None:
        with self._lock:
            self._db.execute("UPDATE streams SET enabled=? WHERE id=?", (int(enabled), stream_id))
            self._db.commit()

    def delete_stream(self, stream_id: int) -> None:
        with self._lock:
            self._db.execute("DELETE FROM streams WHERE id=?", (stream_id,))
            self._db.commit()

    # ------------------------------------------------------------------ 通知ルール
    _ALERT_FIELDS = ("name", "query", "kind", "group_id", "location", "min_score", "webhook_url", "secret",
                     "cooldown_sec", "enabled")  # fmt: skip

    def add_alert(self, **fields) -> int:
        cols = [k for k in self._ALERT_FIELDS if k in fields]
        with self._lock:
            cur = self._db.execute(
                f"INSERT INTO alerts({', '.join(cols)}, created_at) VALUES({', '.join('?' * len(cols))}, ?)",
                (*[fields[k] for k in cols], time.time()),
            )
            self._db.commit()
            return int(cur.lastrowid)

    def get_alert(self, alert_id: int) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM alerts WHERE id=?", (alert_id,)).fetchone()
        return dict(row) if row else None

    def list_alerts(self) -> list[dict]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM alerts ORDER BY id").fetchall()
        return [dict(r) for r in rows]

    def update_alert(self, alert_id: int, **fields) -> None:
        fields = {k: v for k, v in fields.items() if k in self._ALERT_FIELDS}
        if not fields:
            return
        with self._lock:
            sets = ", ".join(f"{k}=?" for k in fields)
            self._db.execute(f"UPDATE alerts SET {sets} WHERE id=?", (*fields.values(), alert_id))
            self._db.commit()

    def delete_alert(self, alert_id: int) -> None:
        with self._lock:
            self._db.execute("DELETE FROM alerts WHERE id=?", (alert_id,))
            self._db.commit()

    def add_alert_event(
        self, *, alert_id: int, window_id: int, source_id: int, score: float, payload: dict
    ) -> int | None:
        """通知を記録します。同じルール・同じ窓が記録済みなら None。"""
        with self._lock:
            cur = self._db.execute(
                "INSERT OR IGNORE INTO alert_events(alert_id, window_id, source_id, score, payload, status, created_at)"
                " VALUES(?,?,?,?,?, 'pending', ?)",
                (alert_id, window_id, source_id, score, json.dumps(payload, ensure_ascii=False), time.time()),
            )
            self._db.commit()
            return int(cur.lastrowid) if cur.rowcount else None

    def get_alert_event(self, event_id: int) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM alert_events WHERE id=?", (event_id,)).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["payload"] = json.loads(d["payload"])
        return d

    def last_alert_event(self, alert_id: int, source_id: int) -> float | None:
        with self._lock:
            row = self._db.execute(
                "SELECT MAX(created_at) AS t FROM alert_events WHERE alert_id=? AND source_id=?", (alert_id, source_id)
            ).fetchone()
        return row["t"] if row and row["t"] is not None else None

    def update_alert_event(self, event_id: int, **fields) -> None:
        allowed = {"status", "attempts", "error", "delivered_at"}
        fields = {k: v for k, v in fields.items() if k in allowed}
        with self._lock:
            sets = ", ".join(f"{k}=?" for k in fields)
            self._db.execute(f"UPDATE alert_events SET {sets} WHERE id=?", (*fields.values(), event_id))
            self._db.commit()

    def list_alert_events(
        self, alert_id: int | None = None, status: str | None = None, limit: int = 100
    ) -> list[dict]:
        where, params = [], []
        if alert_id is not None:
            where.append("alert_id=?")
            params.append(alert_id)
        if status:
            where.append("status=?")
            params.append(status)
        cond = (" WHERE " + " AND ".join(where)) if where else ""
        with self._lock:
            rows = self._db.execute(
                f"SELECT * FROM alert_events{cond} ORDER BY id DESC LIMIT ?", (*params, limit)
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["payload"] = json.loads(d["payload"])
            out.append(d)
        return out

    def alert_counts(self) -> dict:
        with self._lock:
            rows = self._db.execute("SELECT status, COUNT(*) AS n FROM alert_events GROUP BY status").fetchall()
        return {r["status"]: r["n"] for r in rows}

    def prune_alert_events(self, before: float) -> int:
        with self._lock:
            cur = self._db.execute(
                "DELETE FROM alert_events WHERE status != 'pending' AND created_at < ?", (before,)
            )
            self._db.commit()
            return cur.rowcount

    # ------------------------------------------------------------------ ジョブ
    def create_job(self, source_id: int) -> int:
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO jobs(source_id, status, created_at) VALUES(?, 'queued', ?)",
                (source_id, time.time()),
            )
            self._db.commit()
            return int(cur.lastrowid)

    def update_job(self, job_id: int, **fields: Any) -> None:
        allowed = {"status", "progress", "total", "error", "started_at", "finished_at", "timings", "attempts"}
        assert set(fields) <= allowed, f"更新できない列です: {set(fields) - allowed}"
        if isinstance(fields.get("timings"), dict):
            fields["timings"] = json.dumps(fields["timings"], ensure_ascii=False)
        sets = ", ".join(f"{k}=?" for k in fields)
        with self._lock:
            self._db.execute(f"UPDATE jobs SET {sets} WHERE id=?", (*fields.values(), job_id))
            self._db.commit()

    def get_job(self, job_id: int) -> dict | None:
        with self._lock:
            row = self._db.execute(
                "SELECT j.*, s.name AS source_name FROM jobs j JOIN sources s ON s.id=j.source_id"
                " WHERE j.id=?",
                (job_id,),
            ).fetchone()
        return self._job_dict(row) if row else None

    def list_jobs(self, limit: int = 100) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT j.*, s.name AS source_name FROM jobs j JOIN sources s ON s.id=j.source_id"
                " ORDER BY j.id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._job_dict(r) for r in rows]

    def finished_timings(self, limit: int = 100) -> list[dict]:
        """処理時間を記録した完了済みのジョブ(新しい順)。取り込み元の種類を kind に入れて返します。"""
        with self._lock:
            rows = self._db.execute(
                "SELECT j.id, j.timings, s.kind FROM jobs j JOIN sources s ON s.id=j.source_id"
                " WHERE j.status='done' AND j.timings IS NOT NULL ORDER BY j.id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [{"job_id": r["id"], "kind": r["kind"], **json.loads(r["timings"])} for r in rows]

    @staticmethod
    def _job_dict(row: sqlite3.Row) -> dict:
        d = dict(row)
        d["timings"] = json.loads(d["timings"]) if d.get("timings") else None
        return d

    def query_jobs(
        self, *, offset: int = 0, limit: int = 100, status: str | None = None, source_id: int | None = None
    ) -> tuple[list[dict], int]:
        where, params = [], []
        if status:
            where.append("j.status = ?")
            params.append(status)
        if source_id is not None:
            where.append("j.source_id = ?")
            params.append(source_id)
        cond = (" WHERE " + " AND ".join(where)) if where else ""
        with self._lock:
            total = self._db.execute(f"SELECT COUNT(*) AS n FROM jobs j{cond}", params).fetchone()["n"]
            rows = self._db.execute(
                "SELECT j.*, s.name AS source_name FROM jobs j JOIN sources s ON s.id=j.source_id"
                f"{cond} ORDER BY j.id DESC LIMIT ? OFFSET ?",
                (*params, limit, offset),
            ).fetchall()
        return [self._job_dict(r) for r in rows], int(total)

    def prune_jobs(self, before: float) -> int:
        """before より前に終わったジョブの履歴を削除します。取り込み元ごとの最新のジョブ(最後の処理時間)は残します。"""
        with self._lock:
            cur = self._db.execute(
                "DELETE FROM jobs WHERE status IN ('done','failed','cancelled') AND finished_at < ?"
                " AND id NOT IN (SELECT MAX(id) FROM jobs GROUP BY source_id)",
                (before,),
            )
            self._db.commit()
            return cur.rowcount

    def unfinished_jobs(self) -> list[dict]:
        """起動時に、途中で止まった(queued / running)ジョブを取り出します。"""
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM jobs WHERE status IN ('queued','running') ORDER BY id"
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ 検索
    _COLUMNS = ("ids", "source_ids", "start_ms", "end_ms", "kinds", "groups", "locations", "abs_ts")
    _DTYPES = (np.int64, np.int64, np.int64, np.int64, object, object, object, np.float64)

    def _columnar(self, rows: list[tuple]) -> dict[str, np.ndarray]:
        cols = list(zip(*rows, strict=True)) if rows else [[] for _ in self._COLUMNS]
        out = {}
        for name, dtype, values in zip(self._COLUMNS, self._DTYPES, cols, strict=True):
            if dtype is object:
                values = [v or "" for v in values]
            out[name] = np.array(values, dtype=dtype)
        return out

    def _drop_from_cache(self, ids: list[int]) -> None:
        """削除した窓を、検索用のメタデータから外します(全件を読み直さない。保存期間の削除は定期的に起きるため)。"""
        if not ids:
            return
        with self._lock:
            if self._cache is None:
                return
            if self._pending:
                self._meta()  # 未反映の追加分を先に足す
            keep = ~np.isin(self._cache["ids"], np.asarray(ids, dtype=np.int64))
            self._cache = {k: v[keep] for k, v in self._cache.items()}

    # ------------------------------------------------------------------ 保存期間
    def expire_windows(self, cutoff: float | None, now: float, batch: int = 10_000) -> dict:
        """保存期間を過ぎた窓を削除します。削除した窓の ID と、窓が無くなった取り込み元を返します。

        - cutoff: この時刻(UNIX 秒)より前の窓を削除する(RETENTION_DAYS。None なら全体の設定は無し)。
        - ストリームに retention_hours があれば、そのストリームはそちらに従う(0 なら削除しない)。
        - 取り込み中(queued / running)のファイルは対象外にする(途中の窓を消して、取り込みと食い違わないように)。
        - 窓が無くなったファイルの取り込み元は、行を残して状態を expired にする。行ごと消すと、監視フォルダが
          同じファイルを新しいファイルとして取り込み直してしまうため。
        """
        busy = "SELECT id FROM sources WHERE kind != 'stream' AND status IN ('queued','running')"
        own = "SELECT source_id FROM streams WHERE retention_hours IS NOT NULL AND source_id IS NOT NULL"
        conds: list[tuple[str, tuple]] = []
        if cutoff is not None:
            conds.append((f"abs_ts < ? AND source_id NOT IN ({own}) AND source_id NOT IN ({busy})", (cutoff,)))
        with self._lock:
            for r in self._db.execute(
                "SELECT source_id, retention_hours FROM streams WHERE retention_hours > 0 AND source_id IS NOT NULL"
            ).fetchall():
                conds.append(("source_id = ? AND abs_ts < ?", (r["source_id"], now - r["retention_hours"] * 3600)))
        deleted: list[int] = []
        touched: set[int] = set()
        for where, params in conds:
            while True:
                with self._lock:
                    rows = self._db.execute(
                        f"SELECT id, source_id FROM windows WHERE {where} LIMIT ?", (*params, batch)
                    ).fetchall()
                    if not rows:
                        break
                    ids = [r["id"] for r in rows]
                    self._db.executemany("DELETE FROM windows WHERE id=?", [(i,) for i in ids])
                    self._db.commit()
                    self.vectors.remove(ids)
                    self._drop_from_cache(ids)
                deleted += ids
                touched |= {r["source_id"] for r in rows}
        expired: list[dict] = []
        with self._lock:
            for sid in sorted(touched):
                src = self.get_source(sid)
                if src is None or src["kind"] == "stream":
                    continue
                left = self._db.execute("SELECT COUNT(*) AS n FROM windows WHERE source_id=?", (sid,)).fetchone()["n"]
                if left == 0:
                    self._db.execute("UPDATE sources SET status='expired' WHERE id=?", (sid,))
                    expired.append(src)
            self._db.commit()
        thumbs.remove(self.thumbs_dir, deleted)
        return {"window_ids": deleted, "expired_sources": expired}

    def _meta(self) -> dict[str, np.ndarray]:
        """絞り込みに使うメタデータ(ベクトルは含まない)。追加分は、次の検索のときにまとめて反映します。"""
        with self._lock:
            if self._cache is None:
                rows = self._db.execute(
                    "SELECT w.id, w.source_id, w.start_ms, w.end_ms, w.kind, w.group_id, s.location, w.abs_ts"
                    " FROM windows w JOIN sources s ON s.id = w.source_id ORDER BY w.id"
                ).fetchall()
                self._cache = self._columnar([tuple(r) for r in rows])
                self._pending = []
            elif self._pending:
                add = self._columnar(self._pending)
                self._cache = {k: np.concatenate([self._cache[k], add[k]]) for k in self._COLUMNS}
                self._pending = []
            return self._cache

    def search(
        self,
        query: np.ndarray,
        *,
        top_k: int = 10,
        kinds: list[str] | None = None,
        group_id: str | None = None,
        location: str | None = None,
        ts_from: float | None = None,
        ts_to: float | None = None,
        source_id: int | None = None,
        min_score: float = 0.0,
        after_id: int | None = None,
    ) -> list[dict]:
        """メタデータで絞り込んだうえで、コサイン類似度の高い順に返します。

        after_id を渡すと、その ID より後に追加された窓だけを探します(取り込み中の新しい窓の通知に使う)。
        """
        c = self._meta()
        n = len(c["ids"])
        if n == 0:
            return []
        mask = np.ones(n, dtype=bool)
        filtered = False
        for cond, m in (
            (kinds, lambda: np.isin(c["kinds"], kinds)),
            (group_id, lambda: c["groups"] == group_id),
            (location, lambda: c["locations"] == location),
            (ts_from is not None, lambda: c["abs_ts"] >= ts_from),
            (ts_to is not None, lambda: c["abs_ts"] <= ts_to),
            (source_id is not None, lambda: c["source_ids"] == source_id),
            (after_id is not None, lambda: c["ids"] > after_id),
        ):
            if cond:
                mask &= m()
                filtered = True
        idx = np.nonzero(mask)[0]
        if len(idx) == 0:
            return []
        hits = self.vectors.search(query, top_k, candidates=c["ids"][idx] if filtered else None)
        pos = {int(k): i for i, k in enumerate(c["ids"][idx])} if filtered else None
        results = []
        for key, score in hits:
            if score < min_score:
                break
            if pos is not None:
                i = idx[pos[key]]
            else:
                i = int(np.searchsorted(c["ids"], key))  # ids は昇順
                if i >= n or c["ids"][i] != key:
                    continue  # メタデータの反映前に追加された窓(次の検索で見つかる)
            results.append(
                {
                    "window_id": int(c["ids"][i]),
                    "source_id": int(c["source_ids"][i]),
                    "start_ms": int(c["start_ms"][i]),
                    "end_ms": int(c["end_ms"][i]),
                    "kind": str(c["kinds"][i]),
                    "group_id": str(c["groups"][i]) or None,
                    "location": str(c["locations"][i]) or None,
                    "abs_ts": float(c["abs_ts"][i]),
                    "score": score,
                }
            )
        return results

    def windows_between(self, after_id: int, upto_id: int, limit: int = 10_000) -> list[dict]:
        """ID が after_id より大きく upto_id 以下の窓(リアルタイム検索で、新しく増えた窓を照合するため)。"""
        with self._lock:
            rows = self._db.execute(
                "SELECT w.id AS window_id, w.source_id, w.start_ms, w.end_ms, w.kind, w.group_id, w.abs_ts, s.location"
                " FROM windows w JOIN sources s ON s.id = w.source_id"
                " WHERE w.id > ? AND w.id <= ? ORDER BY w.id LIMIT ?",
                (after_id, upto_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def flush(self) -> None:
        """ベクトルをすぐにファイルへ書き出します(通常は一定の間隔で自動的に書き出す)。"""
        self.vectors.save()

    def close(self) -> None:
        self.vectors.close()
        with self._lock:
            self._db.close()
