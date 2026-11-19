"""推論の優先度付きキュー(Priority Queue)による排他制御。

取り込み(低優先)と検索(高優先)を同じプロセスで動かすと、取り込みの推論が続いている間、
検索が待たされます。ここでは「同時に動かす推論は 1 つ」としたうえで、待機中の要求を
優先度付きキュー(min-heap)で管理します。

- 優先度: 検索(PRIORITY_HIGH=0)が取り込み(PRIORITY_LOW=1)より常に先に取り出されます。
  実行中の取り込み 1 窓(1 チャンク)が終わった直後の合間に、検索が最優先で割り込みます。
- 同一優先度内の公平性: 到着順の連番(seq)をタイブレーカーに使い、FIFO 順で実行権を渡します。
  複数スレッドの検索が同時に走っても、後着の検索による追い越し(飢餓)を防ぎ、p95 / p99 を安定させます。
- 直接引き渡し(direct handoff): 解放時に待機キュー先頭のスレッドへ実行権を直接渡し、その 1 スレッド
  だけの Event を起床します(notify_all による全スレッド一斉起床や新規スレッドの割り込みを防ぎます)。

ボトルネックの確認のため、推論が動いていた時間と順番を待った時間を積算します(GET /api/metrics の inference)。
"""

from __future__ import annotations

import heapq
import itertools
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

PRIORITY_HIGH = 0  # 検索(待たせたくない処理)
PRIORITY_LOW = 1  # 取り込み(バックグラウンド処理)


class _Waiter:
    """待機キューのエントリ。個別の Event を持ち、自分の番が来たときだけ起床する。"""

    __slots__ = ("priority", "seq", "key", "is_high", "enqueued_at", "event", "granted", "cancelled")

    def __init__(self, priority: int, seq: int, key: str, is_high: bool, enqueued_at: float) -> None:
        self.priority = priority
        self.seq = seq
        self.key = key
        self.is_high = is_high
        self.enqueued_at = enqueued_at
        self.event = threading.Event()
        self.granted = False
        self.cancelled = False


class PriorityGate:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._heap: list[tuple[int, int, _Waiter]] = []
        self._seq = itertools.count()
        self._busy = False
        self._busy_since = 0.0
        self._waiting = 0
        self._high_waiting = 0
        self._low_waiting = 0
        # 積算の値(秒・回数)。利用側が 2 回読んだ差から、その間の使用率を計算する
        self.stats = {
            "busy_sec": 0.0,  # 推論が動いていた時間の合計
            "wait_sec": {"high": 0.0, "low": 0.0},  # 順番を待った時間の合計(high: 検索、low: 取り込み)
            "calls": {"high": 0, "low": 0},
            "preemptions": 0,  # 待機中の低優先より先に高優先が割り込んだ回数
        }

    @property
    def high_waiting(self) -> int:
        """現在順番を待っている高優先(検索)の件数。"""
        return self._high_waiting

    def has_high_waiting(self) -> bool:
        """高優先(検索)が待機中かどうかを返します。"""
        return self._high_waiting > 0

    def snapshot(self) -> dict:
        """現在の積算値(実行中の推論の経過分も含める)と、今待っている数。"""
        with self._lock:
            busy = self.stats["busy_sec"] + (time.monotonic() - self._busy_since if self._busy else 0.0)
            return {
                "busy_sec": round(busy, 3),
                "wait_sec": {k: round(v, 3) for k, v in self.stats["wait_sec"].items()},
                "calls": dict(self.stats["calls"]),
                "running": self._busy,
                "waiting": self._waiting,
                "high_waiting": self._high_waiting,
                "preemptions": self.stats["preemptions"],
            }

    def _handoff_locked(self, now: float) -> None:
        """待機キューから最優先・最古のエントリを 1 件取り出し、実行権を直接引き渡します。"""
        while self._heap:
            _, _, nxt = heapq.heappop(self._heap)
            if nxt.cancelled:
                continue
            self._waiting -= 1
            if nxt.is_high:
                self._high_waiting -= 1
                if self._low_waiting > 0:
                    self.stats["preemptions"] += 1
            else:
                self._low_waiting -= 1
            nxt.granted = True
            self._busy = True
            self._busy_since = now
            self.stats["wait_sec"][nxt.key] += now - nxt.enqueued_at
            self.stats["calls"][nxt.key] += 1
            nxt.event.set()
            return
        self._busy = False

    @contextmanager
    def acquire(self, high: bool = False, priority: int | None = None) -> Iterator[None]:
        prio = (PRIORITY_HIGH if high else PRIORITY_LOW) if priority is None else int(priority)
        is_high = prio <= PRIORITY_HIGH
        key = "high" if is_high else "low"
        waiter: _Waiter | None = None
        with self._lock:
            now = time.monotonic()
            if not self._busy and not self._heap:
                # 競合が無いときは Event を作らず即座に取得する
                self._busy = True
                self._busy_since = now
                self.stats["calls"][key] += 1
            else:
                waiter = _Waiter(prio, next(self._seq), key, is_high, now)
                heapq.heappush(self._heap, (waiter.priority, waiter.seq, waiter))
                self._waiting += 1
                if is_high:
                    self._high_waiting += 1
                else:
                    self._low_waiting += 1
        if waiter is not None:
            try:
                waiter.event.wait()
            except BaseException:
                with self._lock:
                    if waiter.granted:
                        # すでに実行権を引き継いでいた場合は、次の待機者へ回してデッドロックを防ぐ
                        self._handoff_locked(time.monotonic())
                    elif not waiter.cancelled:
                        waiter.cancelled = True
                        self._waiting -= 1
                        if waiter.is_high:
                            self._high_waiting -= 1
                        else:
                            self._low_waiting -= 1
                raise
        try:
            yield
        finally:
            with self._lock:
                now = time.monotonic()
                self.stats["busy_sec"] += now - self._busy_since
                self._handoff_locked(now)
