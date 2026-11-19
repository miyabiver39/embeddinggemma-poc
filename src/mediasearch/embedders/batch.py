"""推論のまとめ処理(マイクロバッチ)。

複数のスレッド(ストリームごとの受信、ファイルの取り込み、分離構成で compute に届く複数の要求)から同時に来た推論を、
1 回のモデル呼び出しにまとめます。GPU は 1 件ずつより、まとめて計算したほうが処理量が増えるためです
(CPU では速くならないことを実測したため、既定では CPU はまとめない。config.resolve_embed_batch を参照)。

- 推論の実行中に届いた要求は待ち行列にたまり、次の回でまとめて処理されます(待ち時間を足さずにまとまる)。
- まとめた推論が失敗した場合は、原因の要求を特定するため、1 件ずつやり直します。
- prepare を渡すと、前処理(CPU)と推論(GPU)を別のスレッドで動かします。GPU が 1 回分を計算している間に、
  次の回の前処理を済ませておくためです(RTX 3060 で 1 窓の前処理は約 17 ms、推論は約 150 ms)。
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

    def __init__(
        self,
        fn: Callable[[Any], Sequence[Any]],
        max_batch: int,
        wait_sec: float = 0.005,
        prepare: Callable[[list[Any]], Any] | None = None,
        wait_only_when_busy: bool = False,
        should_yield: Callable[[], bool] | None = None,
    ) -> None:
        """prepare を渡したときは fn(prepare(要求のリスト)) -> 結果のリスト、渡さないときは fn(要求のリスト)。

        wait_only_when_busy が真なら、前の回が 1 件だけだったとき(同時に要求が来ていないとき)は次の要求を待たない。
        単独の要求に待ち時間を足さず、同時に来ているときだけまとめるため(検索の文章に使う)。
        should_yield が真を返す間(高優先の検索が待機中など)は、追加の要求を待たずに現在のバッチをすぐ実行へ回す。
        """
        self._fn = fn
        self._prepare = prepare
        self._max = max(int(max_batch), 1)
        self._wait = wait_sec
        self._wait_only_when_busy = wait_only_when_busy
        self._should_yield = should_yield
        self._last_size = 1
        self._queue: queue.Queue = queue.Queue()
        # 前処理を済ませた回を、推論のスレッドへ渡す。1 回分だけ先に用意する(先読みしすぎてメモリを使わないように)
        self._handoff: queue.Queue = queue.Queue(maxsize=1)
        self._thread = threading.Thread(target=self._loop, name="embed-batch", daemon=True)
        self._runner = None
        if prepare is not None:
            self._runner = threading.Thread(target=self._run_loop, name="embed-run", daemon=True)
            self._runner.start()
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
                break
            batch = [first]
            stop = False
            wait = self._wait if (self._last_size > 1 or not self._wait_only_when_busy) else 0
            while len(batch) < self._max:
                if self._should_yield is not None and self._should_yield():
                    # 高優先(検索)が待機中なら、追加の取り込みを待たずに現在のバッチをすぐ解放する
                    break
                try:
                    nxt = self._queue.get(timeout=wait)
                except queue.Empty:
                    break
                if nxt is _STOP:
                    stop = True
                    break
                batch.append(nxt)
            self._last_size = len(batch)
            if self._prepare is None:
                self._run(batch)
            else:
                self._handoff.put((batch, self._prepare_batch(batch)))
            if stop:
                break
        if self._prepare is not None:
            self._handoff.put(_STOP)

    def _prepare_batch(self, batch: list[tuple[Any, Future]]) -> Any:
        """前処理に失敗したら、その例外を推論のスレッドへ渡し、1 件ずつやり直させます。"""
        try:
            return self._prepare([item for item, _ in batch])
        except Exception as exc:
            if len(batch) > 1:
                log.warning("まとめた前処理に失敗したため、1 件ずつやり直します: %s", exc)
            return _Failed(exc)

    def _run_loop(self) -> None:
        while True:
            got = self._handoff.get()
            if got is _STOP:
                return
            batch, prepared = got
            self._run(batch, prepared)

    def _run(self, batch: list[tuple[Any, Future]], prepared: Any = None) -> None:
        items = [item for item, _ in batch]
        try:
            if self._prepare is None:
                results = list(self._fn(items))
            else:
                if prepared is None:
                    prepared = self._prepare(items)
                elif isinstance(prepared, _Failed):
                    if len(batch) == 1:
                        raise prepared.exc
                    raise RuntimeError(f"前処理に失敗しました: {prepared.exc}")
                results = list(self._fn(prepared))
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
        if self._runner is not None:
            self._runner.join(timeout=10)


class _Failed:
    """前処理の失敗(例外)を、推論のスレッドへ渡すための入れ物。"""

    def __init__(self, exc: Exception) -> None:
        self.exc = exc
