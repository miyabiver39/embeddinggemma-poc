"""SQLite によるメタデータとベクトルの保存、およびベクトル検索。

- ベクトルは BLOB(float32)として SQLite に保存します。
- 検索は、全ベクトルをメモリに載せた総当たり(行列の内積)で行います。
  ベクトルは正規化済みなので、内積はコサイン類似度と同じです。
  数十万件程度までは十分速い想定です。それ以上は、専用のベクトル DB への移行を検討してください。
- グループ ID や時刻などのメタデータは、ベクトルに埋め込まず、別の列として保存して絞り込みに使います。
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sources (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  kind TEXT NOT NULL,            -- video / audio / image / frames
  path TEXT,                     -- 映像・音声ファイルのパス(frames は NULL)
  name TEXT NOT NULL,
  group_id TEXT,
  location TEXT,
  start_ts REAL NOT NULL,        -- 録画開始の時刻(UNIX 秒)
  duration_ms INTEGER,
  status TEXT NOT NULL,          -- queued / running / done / failed
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
  abs_ts REAL NOT NULL,          -- 窓の開始時刻(UNIX 秒) = source.start_ts + start_ms / 1000
  vec BLOB NOT NULL
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
  timings TEXT                   -- 処理時間の内訳(JSON。ベンチマーク用)
);
"""

# 索引は、以前の版の DB を移行した後に作る(移行前の列名のままだと作れないため)
INDEXES = """
-- 同じファイルの重複取り込みを確認するため(監視フォルダでは定期的に全件を照合する)
CREATE INDEX IF NOT EXISTS idx_sources_path ON sources(path);
CREATE INDEX IF NOT EXISTS idx_windows_source ON windows(source_id);
"""

WINDOW_KINDS = ("frames", "tav", "audio", "image")

# DB の形式の版。列の追加・改名をしたら上げ、_migrate() に移行の処理を足す
SCHEMA_VERSION = 3
# 旧い版で使っていた列名 → 現在の列名(以前の版の DB をそのまま使い続けられるよう、起動時に改名する)
_RENAMED_COLUMNS = {
    "sources": {"camera_id": "group_id"},
    "windows": {"camera_id": "group_id"},
}
# 後の版で追加した列(以前の版の DB には、起動時に ALTER TABLE で足す)。版 3: jobs.timings
_ADDED_COLUMNS = {
    "jobs": {"timings": "TEXT"},
}
_REQUIRED_COLUMNS = {
    "sources": {"id", "kind", "path", "name", "group_id", "location", "start_ts", "status", "params"},
    "windows": {"id", "source_id", "start_ms", "end_ms", "kind", "group_id", "abs_ts", "vec"},
    "jobs": {"id", "source_id", "status", "progress", "total"},
}


class SchemaError(RuntimeError):
    """DB の形式が、この版で扱えないときのエラー。"""


class IndexMismatch(RuntimeError):
    """DB に記録されたモデル・次元と、現在の設定が一致しないときのエラー。"""


class Store:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")  # 書き込み中も検索(読み取り)できる
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.executescript(SCHEMA)
        self._db.commit()
        self._migrate()
        self._db.executescript(INDEXES)
        self._db.commit()
        self.path = path
        self._cache: dict[str, Any] | None = None

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
            for table, added in _ADDED_COLUMNS.items():
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
            import logging

            logging.getLogger(__name__).warning("以前の版の DB を現在の形式に移行しました: %s", ", ".join(renamed))

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
        allowed = {"status", "error", "duration_ms", "params"}
        assert set(fields) <= allowed, f"更新できない列です: {set(fields) - allowed}"
        if "params" in fields:
            fields["params"] = json.dumps(fields["params"], ensure_ascii=False)
        sets = ", ".join(f"{k}=?" for k in fields)
        with self._lock:
            self._db.execute(f"UPDATE sources SET {sets} WHERE id=?", (*fields.values(), source_id))
            self._db.commit()

    def delete_source(self, source_id: int) -> None:
        with self._lock:
            self._db.execute("DELETE FROM sources WHERE id=?", (source_id,))
            self._db.commit()
            self._cache = None

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
        blob = np.asarray(vec, dtype=np.float32).tobytes()
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO windows(source_id, start_ms, end_ms, kind, group_id, abs_ts, vec)"
                " VALUES(?,?,?,?,?,?,?)",
                (source_id, start_ms, end_ms, kind, group_id, abs_ts, blob),
            )
            self._db.commit()
            self._cache = None  # 次の検索で作り直す
            return int(cur.lastrowid)

    def delete_windows(self, source_id: int) -> int:
        with self._lock:
            cur = self._db.execute("DELETE FROM windows WHERE source_id=?", (source_id,))
            self._db.commit()
            self._cache = None
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
        return {"sources": sources, "windows": sum(by_kind.values()), "windows_by_kind": by_kind}

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
        allowed = {"status", "progress", "total", "error", "started_at", "finished_at", "timings"}
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

    def unfinished_jobs(self) -> list[dict]:
        """起動時に、途中で止まった(queued / running)ジョブを取り出します。"""
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM jobs WHERE status IN ('queued','running') ORDER BY id"
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ 検索
    def _build_cache(self, dims: int) -> dict[str, Any]:
        with self._lock:
            rows = self._db.execute(
                "SELECT w.id, w.source_id, w.start_ms, w.end_ms, w.kind, w.group_id, w.abs_ts,"
                " w.vec, s.location FROM windows w JOIN sources s ON s.id = w.source_id"
            ).fetchall()
        n = len(rows)
        vecs = np.zeros((n, dims), dtype=np.float32)
        for i, r in enumerate(rows):
            v = np.frombuffer(r["vec"], dtype=np.float32)
            if v.shape[0] != dims:
                raise IndexMismatch(
                    f"窓 id={r['id']} のベクトルの次元({v.shape[0]})が、設定({dims})と違います"
                )
            vecs[i] = v
        return {
            "dims": dims,
            "ids": np.array([r["id"] for r in rows], dtype=np.int64),
            "source_ids": np.array([r["source_id"] for r in rows], dtype=np.int64),
            "start_ms": np.array([r["start_ms"] for r in rows], dtype=np.int64),
            "end_ms": np.array([r["end_ms"] for r in rows], dtype=np.int64),
            "kinds": np.array([r["kind"] for r in rows], dtype=object),
            "groups": np.array([r["group_id"] or "" for r in rows], dtype=object),
            "locations": np.array([r["location"] or "" for r in rows], dtype=object),
            "abs_ts": np.array([r["abs_ts"] for r in rows], dtype=np.float64),
            "vecs": vecs,
        }

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
    ) -> list[dict]:
        """メタデータで絞り込んだうえで、コサイン類似度の高い順に返します。"""
        query = np.asarray(query, dtype=np.float32).reshape(-1)
        with self._lock:
            if self._cache is None or self._cache["dims"] != query.shape[0]:
                self._cache = self._build_cache(query.shape[0])
            c = self._cache
        if len(c["ids"]) == 0:
            return []
        mask = np.ones(len(c["ids"]), dtype=bool)
        if kinds:
            mask &= np.isin(c["kinds"], kinds)
        if group_id:
            mask &= c["groups"] == group_id
        if location:
            mask &= c["locations"] == location
        if ts_from is not None:
            mask &= c["abs_ts"] >= ts_from
        if ts_to is not None:
            mask &= c["abs_ts"] <= ts_to
        if source_id is not None:
            mask &= c["source_ids"] == source_id
        idx = np.nonzero(mask)[0]
        if len(idx) == 0:
            return []
        scores = c["vecs"][idx] @ query
        order = np.argsort(-scores)[:top_k]
        results = []
        for j in order:
            score = float(scores[j])
            if score < min_score:
                break
            i = idx[j]
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

    def close(self) -> None:
        with self._lock:
            self._db.close()
