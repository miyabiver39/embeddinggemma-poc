"""推論の優先制御。

取り込み(低優先)と検索(高優先)を同じプロセスで動かすと、取り込みの推論が続いている間、
検索が待たされます。ここでは「同時に動かす推論は1つ」としたうえで、
検索が待っているときは、取り込みが次の推論を始めないようにします。
(実行中の推論を途中で止めることはできないため、待ち時間は最大で推論1回分です。)
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager


class PriorityGate:
    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._busy = False
        self._high_waiting = 0

    @contextmanager
    def acquire(self, high: bool) -> Iterator[None]:
        with self._cond:
            if high:
                self._high_waiting += 1
            try:
                # 実行中の推論がある間は待つ。低優先は、高優先が待っている間も譲る。
                while self._busy or (not high and self._high_waiting > 0):
                    self._cond.wait()
                self._busy = True
            finally:
                if high:
                    self._high_waiting -= 1
        try:
            yield
        finally:
            with self._cond:
                self._busy = False
                self._cond.notify_all()
