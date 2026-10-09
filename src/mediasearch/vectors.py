"""ベクトルの保存と近傍検索(USearch)。

ベクトルは、SQLite とは別のファイル(DATA_DIR/vectors.usearch)に、組み込み型のベクトル DB の USearch
(Apache-2.0。https://github.com/unum-cloud/usearch)で保存します。サーバーを別に立てる必要はなく、
SQLite と同じく 1 つのファイルで完結します。窓のメタデータ(時刻・グループ ID など)は SQLite に置き、
窓の ID をキーに対応させます。

- 検索はコサイン類似度。件数が少ないうちは総当たり(正確)、多くなったら HNSW の近似検索に切り替えます。
- 絞り込み(グループ ID・時刻など)は SQLite 側のメタデータで対象を決め、対象が少なければその分だけを総当たりし、
  多ければ近似検索で多めに取ってから絞り込みます(USearch の Python API には、検索中の絞り込みがないため)。
- 追加・削除はメモリ上の索引にすぐ反映し(取り込み中でも検索できる)、ファイルへの書き出しは一定の間隔でまとめて行います。
  書き出し前に異常終了した場合は、起動時に SQLite と突き合わせて、失われた窓を取り込み直します(Store を参照)。
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# この件数以下は総当たりで探す(USearch の SIMD 実装で、768 次元・5 万件が数十ミリ秒)
EXACT_LIMIT = 50_000
# 近似検索の探索幅。既定(64)より広げて、取りこぼしを減らす
EXPANSION_SEARCH = 128


class IndexMismatch(RuntimeError):
    """DB に記録されたモデル・次元と、現在の設定が一致しないときのエラー。"""


class VectorIndex:
    """窓の ID をキーにしたベクトルの索引。スレッドから安全に使えます。"""

    def __init__(self, path: Path, save_interval_sec: float = 10.0) -> None:
        from usearch.index import Index

        self._Index = Index
        self.path = path
        self._lock = threading.RLock()
        self._index = None
        self._dirty = False
        self._last_save = time.monotonic()
        self._save_cost = 0.0
        self._interval = save_interval_sec
        if path.exists() and path.stat().st_size > 0:
            index = Index.restore(str(path))
            if index is None:
                raise IndexMismatch(f"ベクトルのファイルを読めません: {path}(壊れている可能性があります)")
            index.expansion_search = EXPANSION_SEARCH
            self._index = index
        self._stop = threading.Event()
        self._saver = threading.Thread(target=self._autosave, name="vector-save", daemon=True)
        self._saver.start()

    # ------------------------------------------------------------------ 基本
    @property
    def dims(self) -> int | None:
        return int(self._index.ndim) if self._index is not None else None

    def __len__(self) -> int:
        with self._lock:
            return len(self._index) if self._index is not None else 0

    def keys(self) -> np.ndarray:
        with self._lock:
            if self._index is None or len(self._index) == 0:
                return np.zeros(0, dtype=np.int64)
            return np.asarray(self._index.keys, dtype=np.int64)

    def _ensure(self, dims: int):
        if self._index is None:
            self._index = self._Index(ndim=dims, metric="cos", dtype="f32", expansion_search=EXPANSION_SEARCH)
        elif self._index.ndim != dims:
            raise IndexMismatch(
                f"保存済みのベクトルの次元({self._index.ndim})と、追加・検索するベクトルの次元({dims})が違います"
            )
        return self._index

    def add(self, keys: np.ndarray | list[int], vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors[None, :]
        keys = np.asarray(keys, dtype=np.uint64).reshape(-1)
        with self._lock:
            self._ensure(vectors.shape[1]).add(keys, vectors)
            self._dirty = True

    def remove(self, keys: np.ndarray | list[int]) -> None:
        keys = np.asarray(keys, dtype=np.uint64).reshape(-1)
        if keys.size == 0:
            return
        with self._lock:
            if self._index is not None and len(self._index):
                self._index.remove(keys)
                self._dirty = True

    def get(self, keys: np.ndarray) -> np.ndarray:
        """キーの順にベクトルを返します(無いキーは 0 ベクトル)。"""
        with self._lock:
            dims = self.dims or 0
            if self._index is None or len(keys) == 0:
                return np.zeros((len(keys), dims), dtype=np.float32)
            found = self._index.get(np.asarray(keys, dtype=np.uint64), dtype=np.float32)
        if isinstance(found, np.ndarray) and found.ndim == 2:
            return found
        out = np.zeros((len(keys), dims), dtype=np.float32)
        for i, v in enumerate(found):
            if v is not None:
                out[i] = v
        return out

    # ------------------------------------------------------------------ 検索
    def search(self, query: np.ndarray, top_k: int, candidates: np.ndarray | None = None) -> list[tuple[int, float]]:
        """コサイン類似度の高い順に (キー, スコア) を返します。

        candidates を渡すと、そのキーの中だけを探します(SQLite 側で絞り込んだ窓)。
        """
        query = np.asarray(query, dtype=np.float32).reshape(-1)
        with self._lock:
            index = self._index
            if index is None or len(index) == 0:
                return []
            self._ensure(query.shape[0])
            total = len(index)
            if candidates is None and total <= EXACT_LIMIT:
                m = index.search(query, min(top_k, total), exact=True)
                return [(int(k), 1.0 - float(d)) for k, d in zip(m.keys, m.distances, strict=True)]
        if candidates is not None and len(candidates) <= EXACT_LIMIT:
            return self._exact_subset(query, candidates, top_k)
        return self._approx(query, top_k, candidates, total)

    def _exact_subset(self, query: np.ndarray, keys: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        if len(keys) == 0:
            return []
        vecs = self.get(keys)
        norms = np.linalg.norm(vecs, axis=1)
        scores = (vecs @ query) / np.maximum(norms * float(np.linalg.norm(query)), 1e-12)
        order = np.argsort(-scores)[:top_k]
        return [(int(keys[i]), float(scores[i])) for i in order if norms[i] > 0]

    def _approx(
        self, query: np.ndarray, top_k: int, candidates: np.ndarray | None, total: int
    ) -> list[tuple[int, float]]:
        """近似検索で多めに取り、絞り込みの対象だけを残します。足りなければ取る数を増やします。"""
        allowed = set(candidates.tolist()) if candidates is not None else None
        ratio = len(allowed) / total if allowed is not None else 1.0
        count = min(total, max(top_k * 4, int(top_k / max(ratio, 1e-6) * 2)))
        while True:
            with self._lock:
                m = self._index.search(query, count)
            hits = [
                (int(k), 1.0 - float(d))
                for k, d in zip(m.keys, m.distances, strict=True)
                if allowed is None or int(k) in allowed
            ]
            if len(hits) >= top_k or count >= total:
                return hits[:top_k]
            count = min(total, count * 4)

    # ------------------------------------------------------------------ 書き出し
    def save(self, force: bool = False) -> None:
        """ファイルに書き出します。途中で止まっても前のファイルが壊れないよう、一時ファイルに書いてから置き換えます。"""
        with self._lock:
            if self._index is None or not (self._dirty or force):
                return
            t = time.monotonic()
            tmp = self.path.with_suffix(".tmp")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._index.save(str(tmp))
            os.replace(tmp, self.path)
            self._dirty = False
            self._last_save = time.monotonic()
            self._save_cost = self._last_save - t

    def _autosave(self) -> None:
        while not self._stop.wait(1.0):
            # 索引が大きく書き出しに時間がかかる場合は、間隔を広げる(書き出しが処理の大半を占めないように)
            interval = max(self._interval, self._save_cost * 10)
            if self._dirty and time.monotonic() - self._last_save >= interval:
                try:
                    self.save()
                except Exception:  # 次の機会に再試行する(ディスクの空き不足など)
                    log.exception("ベクトルのファイルを書き出せませんでした: %s", self.path)

    def close(self) -> None:
        self._stop.set()
        self._saver.join(timeout=5)
        self.save()
