"""推論のまとめ処理(マイクロバッチ)。

複数のスレッド(ストリームごとの受信、ファイルの取り込み、分離構成で compute に届く複数の要求)から同時に来た推論を、
1 回のモデル呼び出しにまとめます。GPU は 1 件ずつより、まとめて計算したほうが処理量が増えるためです
(CPU では速くならないことを実測したため、既定では CPU はまとめない。config.resolve_embed_batch を参照)。

- 推論の実行中に届いた要求は待ち行列にたまり、次の回でまとめて処理されます(待ち時間を足さずにまとまる)。
- まとめた推論が失敗した場合は、原因の要求を特定するため、1 件ずつやり直します。
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from typing import Any

log = logging.getLogger(__name__)

_STOP = object()


class MicroBatcher:
    """fn(要求のリスト) -> 結果のリスト を、専用のスレッドでまとめて呼び出します。"""

    def __init__(self, fn: Callable[[list[Any]], Sequence[Any]], max_batch: int, wait_sec: float = 0.005) -> None:
        self._fn = fn
        self._max = max(int(max_batch), 1)
        self._wait = wait_sec
        self._queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._loop, name="embed-batch", daemon=True)
        self._thread.start()
        self.batches = 0  # 実行した回数(テストと確認用)
        self.items = 0

    def submit(self, item: Any) -> Future:
        fut: Future = Future()
        self._queue.put((item, fut))
        return fut

    def run_many(self, items: Sequence[Any]) -> list[Any]:
        """複数の要求をまとめて投入し、同じ順序で結果を返します(どれかが失敗したら、その例外を出す)。"""
        futures = [self.submit(item) for item in items]
        return [f.result() for f in futures]

    def _loop(self) -> None:
        while True:
            first = self._queue.get()
            if first is _STOP:
                return
            batch = [first]
            stop = False
            while len(batch) < self._max:
                try:
                    nxt = self._queue.get(timeout=self._wait)
                except queue.Empty:
                    break
                if nxt is _STOP:
                    stop = True
                    break
                batch.append(nxt)
            self._run(batch)
            if stop:
                return

    def _run(self, batch: list[tuple[Any, Future]]) -> None:
        items = [item for item, _ in batch]
        try:
            results = list(self._fn(items))
            if len(results) != len(items):
                raise RuntimeError(f"まとめた推論の結果の数({len(results)})が、要求の数({len(items)})と違います")
        except Exception as exc:
            if len(batch) == 1:
                batch[0][1].set_exception(exc)
                return
            log.warning("まとめた推論に失敗したため、1 件ずつやり直します: %s", exc)
            for one in batch:
                self._run([one])
            return
        self.batches += 1
        self.items += len(items)
        for (_, fut), res in zip(batch, results, strict=True):
            fut.set_result(res)

    def close(self) -> None:
        self._queue.put(_STOP)
        self._thread.join(timeout=10)
