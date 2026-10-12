"""ベクトルの保存と近傍検索(USearch)。

ベクトルは、SQLite とは別のファイル(DATA_DIR/vectors.usearch)に、組み込み型のベクトル DB の USearch
(Apache-2.0。https://github.com/unum-cloud/usearch)で保存します。サーバーを別に立てる必要はなく、
SQLite と同じく 1 つのファイルで完結します。窓のメタデータ(時刻・グループ ID など)は SQLite に置き、
窓の ID をキーに対応させます。

- 検索はコサイン類似度。件数が少ないうちは総当たり(正確)、多くなったら HNSW の近似検索に切り替えます。
- 絞り込み(グループ ID・時刻など)は SQLite 側のメタデータで対象を決め、対象が少なければその分だけを総当たりし、
  多ければ近似検索で多めに取ってから絞り込みます(USearch の Python API には、検索中の絞り込みがないため)。
- 追加・削除はメモリ上の索引にすぐ反映し(取り込み中でも検索できる)、同時に追記型のジャーナル
  (DATA_DIR/vectors.journal)へ書き足します。索引全体の書き出し(スナップショット)は、変更が SNAPSHOT_EVERY 件
  たまったときと停止時だけ行い、書き出したらジャーナルを空にします。起動時は、スナップショットを読んでから
  ジャーナルを再生します。
  USearch の書き出しは、その間 Python 全体を止める(GIL を持ったまま書く。実測で 6 万件・768 次元が 0.25 秒)ため、
  頻繁に書き出すと、索引が大きいときに API 全体が止まる。ジャーナルにすれば 1 件の追加は数 KB の追記で済む。
- それでも整合しない場合(ジャーナルの書き込み前に異常終了したなど)は、起動時に SQLite と突き合わせて、
  失われた窓を取り込み直します(Store を参照)。
"""

from __future__ import annotations

import logging
import os
import struct
import threading
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# この件数以下は総当たりで探す(USearch の SIMD 実装で、768 次元・5 万件が数十ミリ秒)
EXACT_LIMIT = 50_000
# 近似検索の探索幅。既定(64)より広げて、取りこぼしを減らす
EXPANSION_SEARCH = 128
# ジャーナルにこの件数の変更がたまったら、スナップショットを書き出してジャーナルを空にする
# (多いほど書き出しによる停止が減り、異常終了後の起動時の再生に時間がかかる。再生は 1 件約 0.5 ミリ秒)
SNAPSHOT_EVERY = 20_000

_MAGIC = b"MSVJ1"  # ジャーナルの形式: 先頭に _MAGIC と次元(uint32)、以降は操作ごとの記録
_ADD, _REMOVE = b"A", b"R"  # 記録: 操作(1 バイト)+ キー(uint64)+ 追加ならベクトル(float32 × 次元)


class IndexMismatch(RuntimeError):
    """DB に記録されたモデル・次元と、現在の設定が一致しないときのエラー。"""


class VectorIndex:
    """窓の ID をキーにしたベクトルの索引。スレッドから安全に使えます。"""

    def __init__(self, path: Path, snapshot_every: int | None = None) -> None:
        from usearch.index import Index

        self._Index = Index
        self.path = path
        self.journal_path = path.with_suffix(".journal")
        self._lock = threading.RLock()
        self._index = None
        self._snapshot_every = snapshot_every or SNAPSHOT_EVERY
        self._pending = 0  # 前回のスナップショットからの変更の数(= ジャーナルの記録の数)
        self._journal = None
        self._unsynced = False
        if path.exists() and path.stat().st_size > 0:
            index = Index.restore(str(path))
            if index is None:
                raise IndexMismatch(f"ベクトルのファイルを読めません: {path}(壊れている可能性があります)")
            index.expansion_search = EXPANSION_SEARCH
            self._index = index
        self._replay()
        self._stop = threading.Event()
        self._saver = threading.Thread(target=self._background, name="vector-save", daemon=True)
        self._saver.start()

    # ------------------------------------------------------------------ ジャーナル
    def _replay(self) -> None:
        """前回のスナップショットの後の変更を、ジャーナルから戻します(異常終了したときだけ中身がある)。"""
        if not self.journal_path.exists() or self.journal_path.stat().st_size == 0:
            return
        data = self.journal_path.read_bytes()
        if not data.startswith(_MAGIC) or len(data) < len(_MAGIC) + 4:
            raise IndexMismatch(f"ベクトルのジャーナルを読めません: {self.journal_path}(壊れている可能性があります)")
        (dims,) = struct.unpack_from("<I", data, len(_MAGIC))
        pos, count = len(_MAGIC) + 4, 0
        rec_add = 9 + 4 * dims
        while pos < len(data):
            op = data[pos : pos + 1]
            if op == _ADD and pos + rec_add <= len(data):
                (key,) = struct.unpack_from("<Q", data, pos + 1)
                vec = np.frombuffer(data, dtype="<f4", count=dims, offset=pos + 9).astype(np.float32)
                index = self._ensure(dims)
                if key in index:  # スナップショットの書き出し後、ジャーナルを空にする前に止まった場合
                    index.remove(key)
                index.add(key, vec)
                pos += rec_add
            elif op == _REMOVE and pos + 9 <= len(data):
                (key,) = struct.unpack_from("<Q", data, pos + 1)
                if self._index is not None and key in self._index:
                    self._index.remove(key)
                pos += 9
            else:
                break  # 書き込みの途中で止まった最後の記録は捨てる
            count += 1
        self._pending = count
        log.warning("ベクトルのジャーナルから %d 件の変更を戻しました(前回は正常に停止していません)", count)
        self.save(force=True)

    def _write(self, records: list[bytes]) -> None:
        if self._journal is None:
            fresh = not self.journal_path.exists() or self.journal_path.stat().st_size == 0
            # 停止まで開いたままにする(追加のたびに開き直さない)
            self._journal = open(self.journal_path, "ab")  # noqa: SIM115
            if fresh:
                self._journal.write(_MAGIC + struct.pack("<I", self.dims or 0))
        self._journal.write(b"".join(records))
        self._journal.flush()  # OS には即時に渡す。ディスクへの同期(fsync)は 1 秒ごとにまとめる
        self._unsynced = True
        self._pending += len(records)

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

    def add(self, keys: np.ndarray | list[int], vectors: np.ndarray, journal: bool = True) -> None:
        """追加します。journal=False は、直後にスナップショットを書き出す一括の取り込み(移行など)用です。"""
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors[None, :]
        keys = np.asarray(keys, dtype=np.uint64).reshape(-1)
        with self._lock:
            self._ensure(vectors.shape[1]).add(keys, vectors)
            if not journal:
                self._pending += len(keys)
                return
            pairs = zip(keys, vectors, strict=True)
            self._write([_ADD + struct.pack("<Q", int(k)) + v.astype("<f4").tobytes() for k, v in pairs])

    def remove(self, keys: np.ndarray | list[int]) -> None:
        keys = np.asarray(keys, dtype=np.uint64).reshape(-1)
        if keys.size == 0:
            return
        with self._lock:
            if self._index is not None and len(self._index):
                self._index.remove(keys)
                self._write([_REMOVE + struct.pack("<Q", int(k)) for k in keys])

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
    @property
    def pending(self) -> int:
        """前回のスナップショットからの変更の数。"""
        return self._pending

    def save(self, force: bool = False) -> None:
        """スナップショットを書き出し、ジャーナルを空にします。

        途中で止まっても前のファイルが壊れないよう、一時ファイルに書いてから置き換えます。
        置き換えた後、ジャーナルを空にする前に止まった場合は、次の起動で同じ変更を再生する(結果は同じになる)。
        """
        with self._lock:
            if self._index is None or not (self._pending or force):
                return
            tmp = self.path.with_suffix(".tmp")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._index.save(str(tmp))
            os.replace(tmp, self.path)
            if self._journal is not None:
                self._journal.close()
                self._journal = None
            self.journal_path.unlink(missing_ok=True)
            self._pending = 0
            self._unsynced = False

    def save_copy(self, dest: Path) -> bool:
        """今の索引(ジャーナルの分を含む)を dest に書き出します(バックアップ用)。空なら書かずに False。

        元のファイル・ジャーナルには触れないため、スナップショットの周期は変わらない。
        """
        with self._lock:
            if self._index is None or len(self._index) == 0:
                return False
            dest.parent.mkdir(parents=True, exist_ok=True)
            self._index.save(str(dest))
            return True

    def _background(self) -> None:
        while not self._stop.wait(1.0):
            try:
                with self._lock:
                    if self._unsynced and self._journal is not None:
                        os.fsync(self._journal.fileno())
                        self._unsynced = False
                if self._pending >= self._snapshot_every:
                    self.save()
            except Exception:  # 次の機会に再試行する(ディスクの空き不足など)
                log.exception("ベクトルのファイルを書き出せませんでした: %s", self.path)

    def close(self) -> None:
        self._stop.set()
        self._saver.join(timeout=5)
        self.save()
