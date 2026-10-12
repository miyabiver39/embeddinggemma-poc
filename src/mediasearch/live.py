"""リアルタイム検索の通知(GET /api/search/live)の照合を、接続をまたいでまとめて行います。

接続ごとに検索すると、接続の数だけ DB とベクトル DB を読み直すことになります。ここでは 1 つの処理(イベントループ上の
タスク)が 0.5 秒ごとに新しい窓を確認し、新しい窓のベクトルを 1 回だけ読み出して、すべての接続のクエリとの類似度を
1 回の行列の積で計算し、接続ごとの絞り込み(種類・グループ ID・場所・最小スコア)を当てて、それぞれへ配ります。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

import numpy as np
from starlette.concurrency import run_in_threadpool

log = logging.getLogger(__name__)

QUEUE_SIZE = 1000  # 接続ごとにためておける通知の数。受け取りが遅く、あふれた分は捨てる(dropped で数える)


@dataclass(eq=False)
class Subscriber:
    qvec: np.ndarray
    kinds: list[str] | None
    group_id: str | None
    location: str | None
    min_score: float
    queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=QUEUE_SIZE))
    dropped: int = 0


class LiveHub:
    def __init__(self, service, interval_sec: float = 0.5) -> None:
        self._svc = service
        self._interval = interval_sec
        self._subs: set[Subscriber] = set()
        self._task: asyncio.Task | None = None
        self.seen = 0  # 照合を終えた最後の窓の ID
        self.rounds = 0  # 照合した回数(テストと確認用)

    @property
    def subscribers(self) -> int:
        return len(self._subs)

    def subscribe(self, sub: Subscriber) -> int:
        """接続を登録し、この時点で照合を終えている最後の窓の ID を返します(これより後の窓を通知する)。"""
        if self._task is None or self._task.done():
            self.seen = self._svc.ctx.store.last_window_id
            self._task = asyncio.get_running_loop().create_task(self._run())
        self._subs.add(sub)
        return self.seen

    def unsubscribe(self, sub: Subscriber) -> None:
        self._subs.discard(sub)

    async def _run(self) -> None:
        while self._subs:
            await asyncio.sleep(self._interval)
            latest = self._svc.ctx.store.last_window_id
            if latest <= self.seen:
                continue
            subs = list(self._subs)
            try:
                matched = await run_in_threadpool(self._match, self.seen, latest, subs)
            except Exception:
                log.exception("リアルタイム検索の照合に失敗しました")
                continue
            self.seen = latest
            self.rounds += 1
            for sub, hits in matched:
                for h in hits:
                    try:
                        sub.queue.put_nowait(h)
                    except asyncio.QueueFull:
                        sub.dropped += 1

    def _match(self, after: int, upto: int, subs: list[Subscriber]) -> list[tuple[Subscriber, list[dict]]]:
        store = self._svc.ctx.store
        out: dict[int, tuple[Subscriber, list[dict]]] = {id(s): (s, []) for s in subs}
        while after < upto:
            rows = store.windows_between(after, upto)
            if not rows:
                break
            after = rows[-1]["window_id"]
            ids = np.array([r["window_id"] for r in rows], dtype=np.int64)
            vecs = store.vectors.get(ids)  # 新しい窓のベクトルを 1 回だけ読む
            queries = np.stack([s.qvec for s in subs]).astype(np.float32)
            if vecs.shape[1] != queries.shape[1]:
                break  # 次元が違う(DB とクエリのモデルが違う)場合は通知しない。検索の API が 409 を返す
            scores = vecs @ queries.T  # ベクトルは正規化済みなので、内積 = コサイン類似度
            kinds = np.array([r["kind"] for r in rows], dtype=object)
            groups = np.array([r["group_id"] or "" for r in rows], dtype=object)
            locs = np.array([r["location"] or "" for r in rows], dtype=object)
            for j, sub in enumerate(subs):
                mask = scores[:, j] >= sub.min_score
                if sub.kinds:
                    mask &= np.isin(kinds, sub.kinds)
                if sub.group_id:
                    mask &= groups == sub.group_id
                if sub.location:
                    mask &= locs == sub.location
                hits = [{**rows[i], "score": float(scores[i, j])} for i in np.nonzero(mask)[0]]
                out[id(sub)][1].extend(self._svc.decorate_hits(hits))
        return list(out.values())
