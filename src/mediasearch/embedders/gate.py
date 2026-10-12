"""推論の優先制御。

取り込み(低優先)と検索(高優先)を同じプロセスで動かすと、取り込みの推論が続いている間、
検索が待たされます。ここでは「同時に動かす推論は1つ」としたうえで、
検索が待っているときは、取り込みが次の推論を始めないようにします。
(実行中の推論を途中で止めることはできないため、待ち時間は最大で推論1回分です。)

ボトルネックの確認のため、推論が動いていた時間と、順番を待った時間を積算します(GET /api/metrics の inference)。
「推論が動いていた時間 ÷ 経過時間」が 100% に近ければ、推論が律速です。
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager


class PriorityGate:
    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._busy = False
        self._high_waiting = 0
        self._waiting = 0
        self._busy_since = 0.0
        # 積算の値(秒・回数)。利用側が 2 回読んだ差から、その間の使用率を計算する
        self.stats = {
            "busy_sec": 0.0,  # 推論が動いていた時間の合計
            "wait_sec": {"high": 0.0, "low": 0.0},  # 順番を待った時間の合計(high: 検索、low: 取り込み)
            "calls": {"high": 0, "low": 0},
        }

    def snapshot(self) -> dict:
        """現在の積算値(実行中の推論の経過分も含める)と、今待っている数。"""
        with self._cond:
            busy = self.stats["busy_sec"] + (time.monotonic() - self._busy_since if self._busy else 0.0)
            return {
                "busy_sec": round(busy, 3),
                "wait_sec": {k: round(v, 3) for k, v in self.stats["wait_sec"].items()},
                "calls": dict(self.stats["calls"]),
                "running": self._busy,
                "waiting": self._waiting,
            }

    @contextmanager
    def acquire(self, high: bool) -> Iterator[None]:
        key = "high" if high else "low"
        t0 = time.monotonic()
        with self._cond:
            self._waiting += 1
            if high:
                self._high_waiting += 1
            try:
                # 実行中の推論がある間は待つ。低優先は、高優先が待っている間も譲る。
                while self._busy or (not high and self._high_waiting > 0):
                    self._cond.wait()
                self._busy = True
                self._busy_since = time.monotonic()
                self.stats["wait_sec"][key] += self._busy_since - t0
                self.stats["calls"][key] += 1
            finally:
                self._waiting -= 1
                if high:
                    self._high_waiting -= 1
        try:
            yield
        finally:
            with self._cond:
                self._busy = False
                self.stats["busy_sec"] += time.monotonic() - self._busy_since
                self._cond.notify_all()
