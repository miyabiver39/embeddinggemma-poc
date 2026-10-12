"""運用の監視: リクエスト ID・アクセスログ・JSON 形式のログ・Prometheus 形式の値(GET /metrics)。

- リクエスト ID: 要求の `X-Request-ID`(英数字・-・_ の 64 文字まで)をそのまま使い、無ければ作ります。
  応答のヘッダーに返し、その要求の間に出たログに `request_id` として付けます
  (問い合わせのときに、利用側のログと突き合わせられる)。
- LOG_FORMAT=json: ログを 1 行 1 件の JSON にします(ログの収集基盤で扱いやすい)。
  この場合、uvicorn のアクセスログの代わりに、リクエスト ID と処理時間を含むアクセスログを出します。
- GET /metrics: Prometheus の形式(text/plain; version=0.0.4)。認証は /api と同じ(API_TOKEN を設定していれば
  Bearer が必要。読み取り専用のトークンでも読める)。
"""

from __future__ import annotations

import contextvars
import json
import logging
import re
import threading
import time
import uuid

request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)
_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")
access_log = logging.getLogger("mediasearch.access")

# 処理時間の区切り(秒)。検索は数百ミリ秒〜数秒(推論の待ちを含む)
BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60)


class RequestStats:
    """HTTP の要求の数と処理時間(経路のひな形・メソッド・状態コードごと)。"""

    def __init__(self) -> None:
        self.counts: dict[tuple[str, str, int], int] = {}
        self.hist: dict[tuple[str, str], list[float]] = {}  # (route, method) -> [各区切りの件数..., 合計, 件数]
        # 記録はイベントループ、読み出しは /metrics のスレッドで行うため、読み出し中の変更を防ぐ
        self._lock = threading.Lock()

    def snapshot(self) -> tuple[dict, dict]:
        with self._lock:
            return dict(self.counts), {k: list(v) for k, v in self.hist.items()}

    def observe(self, route: str, method: str, status: int, seconds: float) -> None:
        with self._lock:
            self._observe(route, method, status, seconds)

    def _observe(self, route: str, method: str, status: int, seconds: float) -> None:
        key = (route, method, status)
        self.counts[key] = self.counts.get(key, 0) + 1
        h = self.hist.setdefault((route, method), [0.0] * (len(BUCKETS) + 2))
        for i, b in enumerate(BUCKETS):
            if seconds <= b:
                h[i] += 1
        h[-2] += seconds
        h[-1] += 1


class RequestContextMiddleware:
    """全ての要求に ID を付け、処理時間を数えます(いちばん外側に置き、拒否した要求も数える)。"""

    def __init__(self, app, stats: RequestStats, log_access: bool) -> None:
        self.app = app
        self.stats = stats
        self.log_access = log_access

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        given = next((v.decode("latin-1") for k, v in scope.get("headers", []) if k == b"x-request-id"), "")
        rid = given if _ID_RE.match(given) else uuid.uuid4().hex[:16]
        token = request_id_var.set(rid)
        started = time.perf_counter()
        status = 500

        async def send_with_id(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message = {**message, "headers": [*message.get("headers", []), (b"x-request-id", rid.encode())]}
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            elapsed = time.perf_counter() - started
            route = getattr(scope.get("route"), "path", None) or _route_of(scope.get("path", ""))
            method = scope.get("method", "GET")
            method = method if method in _METHODS else "OTHER"  # 任意のメソッド名で値の種類が増えないように
            self.stats.observe(route, method, status, elapsed)
            if self.log_access and route not in ("/healthz", "/readyz", "/metrics"):
                access_log.info(
                    "%s %s %s %.1fms", method, scope.get("path", ""), status, elapsed * 1000,
                    extra={"http": {"method": method, "path": scope.get("path", ""), "route": route,
                                    "status": status, "duration_ms": round(elapsed * 1000, 1),
                                    "client": (scope.get("client") or ("", 0))[0]}},
                )  # fmt: skip
            request_id_var.reset(token)


def _route_of(path: str) -> str:
    """経路が決まらなかった要求(認証で断った・存在しない経路など)。

    パスをそのまま使うと、任意のパスを送るだけで値の種類(メモリ)がいくらでも増えるため、先頭の部分にまとめる。
    """
    for prefix in ("/api/", "/compute/"):
        if path.startswith(prefix):
            return prefix + "*"
    if path in _FIXED_PATHS:
        return path
    return "other"


_FIXED_PATHS = frozenset(
    {"/", "/mcp", "/healthz", "/readyz", "/metrics", "/docs", "/redoc", "/scalar", "/openapi.json"}
)
_METHODS = frozenset({"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"})


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created))
            + f".{int(record.msecs):03d}"
            + time.strftime("%z", time.localtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if getattr(record, "request_id", None):
            out["request_id"] = record.request_id
        if getattr(record, "http", None):
            out["http"] = record.http
        if record.exc_info:
            out["exception"] = self.formatException(record.exc_info)
        return json.dumps(out, ensure_ascii=False)


def setup_logging(level: str, fmt: str) -> None:
    """ログの設定。fmt が json なら 1 行 1 件の JSON、それ以外は従来の 1 行の文字列(リクエスト ID つき)。"""
    root = logging.getLogger()
    root.setLevel(level.upper())
    handler = logging.StreamHandler()
    handler._mediasearch = True  # 作り直すとき、自分の出力先だけを置き換える(テストのログの取得などは残す)
    handler.addFilter(_RequestIdFilter())
    if fmt == "json":
        handler.setFormatter(JsonFormatter())
        # uvicorn のアクセスログは、リクエスト ID と処理時間を含む mediasearch.access に置き換える
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
        for name in ("uvicorn", "uvicorn.error"):
            lg = logging.getLogger(name)
            lg.handlers = []
            lg.propagate = True
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.handlers = [h for h in root.handlers if not getattr(h, "_mediasearch", False)] + [handler]


# ---------------------------------------------------------------------- Prometheus
def _esc(v) -> str:
    return str(v).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


class _Writer:
    """Prometheus のテキスト形式。同じ名前の値は 1 か所にまとめて出す(形式の決まり。分かれると読み込めない)。"""

    def __init__(self) -> None:
        self._families: dict[str, tuple[str, str, list[str]]] = {}

    def _family(self, name: str, kind: str, help_text: str) -> list[str]:
        return self._families.setdefault(name, (kind, help_text, []))[2]

    def metric(self, name: str, kind: str, help_text: str, value, labels: dict | None = None) -> None:
        if value is None:
            return
        lab = "{" + ",".join(f'{k}="{_esc(v)}"' for k, v in labels.items()) + "}" if labels else ""
        self._family(name, kind, help_text).append(f"{name}{lab} {float(value):g}")

    def histogram(self, name: str, help_text: str, labels: dict, counts: list[float]) -> None:
        """counts は、各区切り以下の件数(累積)・合計・件数の順。"""
        lines = self._family(name, "histogram", help_text)
        base = ",".join(f'{k}="{_esc(v)}"' for k, v in labels.items())
        for i, b in enumerate(BUCKETS):
            lines.append(f'{name}_bucket{{{base},le="{b:g}"}} {counts[i]:g}')
        lines.append(f'{name}_bucket{{{base},le="+Inf"}} {counts[-1]:g}')
        lines.append(f"{name}_sum{{{base}}} {counts[-2]:g}")
        lines.append(f"{name}_count{{{base}}} {counts[-1]:g}")

    def text(self) -> str:
        out: list[str] = []
        for name, (kind, help_text, lines) in self._families.items():
            out += [f"# HELP {name} {help_text}", f"# TYPE {name} {kind}", *lines]
        return "\n".join(out) + "\n"


def prometheus(stats: RequestStats, ctx=None, embedder=None) -> str:
    """GET /metrics の本文。ctx が無い(ROLE=compute)ときは、資源と推論の値だけを出します。"""
    from . import disk
    from .metrics import app_metrics, process_metrics

    if ctx is not None:
        m = app_metrics(ctx)
    else:
        m = {**process_metrics(), "inference": embedder.metrics() if embedder is not None else {}}
    w = _Writer()
    p = "mediasearch_"
    w.metric(p + "process_cpu_seconds_total", "counter", "アプリ本体の CPU 時間", m["process"]["cpu_sec"])
    w.metric(p + "process_resident_memory_bytes", "gauge", "アプリ本体のメモリ(RSS)", m["process"]["rss_bytes"])
    for name, v in m["threads_cpu_sec"].items():
        w.metric(p + "thread_cpu_seconds_total", "counter", "スレッドごとの CPU 時間", v, {"thread": name})
    w.metric(p + "children_cpu_seconds_total", "counter", "子プロセス(ffmpeg など)の CPU 時間",
             sum(m["children_cpu_sec"].values()))  # fmt: skip
    cg = m.get("cgroup") or {}
    w.metric(p + "container_cpu_seconds_total", "counter", "コンテナ全体の CPU 時間", cg.get("cpu_sec"))
    w.metric(
        p + "container_cpu_throttled_seconds_total", "counter", "CPU の上限で止められた時間", cg.get("throttled_sec")
    )
    w.metric(p + "container_cpu_limit", "gauge", "コンテナが使える CPU の数", cg.get("cpu_limit") or m["cpus"])
    w.metric(p + "container_memory_bytes", "gauge", "コンテナ全体のメモリ", cg.get("memory_bytes"))
    w.metric(p + "container_memory_limit_bytes", "gauge", "コンテナのメモリの上限", cg.get("memory_limit_bytes"))
    gate = (m.get("inference") or {}).get("gate") or {}
    if gate:
        w.metric(p + "inference_busy_seconds_total", "counter", "推論が動いていた時間", gate["busy_sec"])
        for prio in ("high", "low"):
            w.metric(p + "inference_wait_seconds_total", "counter", "推論の順番を待った時間(high: 検索、low: 取り込み)",
                     gate["wait_sec"][prio], {"priority": prio})  # fmt: skip
            w.metric(p + "inference_calls_total", "counter", "推論の回数", gate["calls"][prio], {"priority": prio})
        w.metric(p + "inference_waiting", "gauge", "推論の順番を待っている数", gate["waiting"])
    for g in (m.get("gpu") or {}).get("devices", []):
        lab = {"gpu": g.get("index"), "name": g.get("name")}
        w.metric(p + "gpu_utilization_percent", "gauge", "GPU の使用率", g.get("util_pct"), lab)
        w.metric(
            p + "gpu_decoder_utilization_percent", "gauge", "GPU の動画復号回路の使用率", g.get("decoder_pct"), lab
        )
        used = g.get("memory_used_mb")
        w.metric(
            p + "gpu_memory_used_bytes", "gauge", "GPU メモリの使用量", None if used is None else used * 2**20, lab
        )
    if ctx is not None:
        w.metric(p + "windows_total", "counter", "作った窓の数(最後の窓の ID)", m["index"]["windows"])
        w.metric(p + "vectors", "gauge", "ベクトル DB の件数", m["index"]["vectors"])
        w.metric(
            p + "vector_journal_pending",
            "gauge",
            "スナップショットに書き出していない変更の数",
            m["index"]["journal_pending"],
        )
        w.metric(p + "ingest_queue", "gauge", "取り込みを待っているジョブの数", m["ingest"]["queue"])
        for s in m.get("streams") or []:
            lab = {"stream": s["id"], "name": s["name"]}
            w.metric(
                p + "stream_up", "gauge", "ストリームを受信中か(1 = 受信中)", 1 if s["status"] == "running" else 0, lab
            )
            w.metric(p + "stream_windows_total", "counter", "ストリームで作った窓の数", s.get("windows") or 0, lab)
            w.metric(
                p + "stream_dropped_windows_total",
                "counter",
                "推論が追いつかず捨てた窓の数",
                s.get("dropped_windows") or 0,
                lab,
            )
            w.metric(p + "stream_lag_seconds", "gauge", "ストリームの遅れ", (s.get("lag_ms") or 0) / 1000, lab)
        d = disk.status(ctx.settings.data_dir, ctx.settings.min_free_disk_mb)
        if "free_mb" in d:
            w.metric(p + "disk_free_bytes", "gauge", "DATA_DIR の空き容量", d["free_mb"] * 2**20)
            w.metric(p + "disk_ok", "gauge", "空き容量が下限(MIN_FREE_DISK_MB)以上か", 1 if d["ok"] else 0)
        if ctx.search_limiter is not None:
            w.metric(p + "search_active", "gauge", "処理中の検索の数", ctx.search_limiter.active)
            w.metric(p + "search_rejected_total", "counter", "混雑で断った検索の数", ctx.search_limiter.rejected)
        last = ctx.backups.last() if ctx.backups is not None else None
        if last:
            help_text = "最後にバックアップを作った時刻(UNIX 秒)"
            w.metric(p + "backup_last_timestamp_seconds", "gauge", help_text, last.get("created_ts"))
        for status, n in ctx.store.alert_counts().items():
            w.metric(p + "alert_events", "gauge", "通知の数(状態ごと。履歴の保存期間内)", n, {"status": status})
    counts, hist = stats.snapshot()
    for (route, method, status), n in sorted(counts.items()):
        w.metric(
            p + "http_requests_total",
            "counter",
            "HTTP の要求の数",
            n,
            {"route": route, "method": method, "status": status},
        )
    for (route, method), h in sorted(hist.items()):
        w.histogram(p + "http_request_duration_seconds", "HTTP の要求の処理時間", {"route": route, "method": method}, h)
    return w.text()
