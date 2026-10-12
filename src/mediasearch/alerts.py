"""通知ルール: 検索の条件を保存しておき、新しく作られた窓が条件に合ったら Webhook で知らせます。

リアルタイム検索の通知(GET /api/search/live)は、利用側が接続している間しか届きません。監視の用途
(「人が倒れている」「車が進入した」などを、見ていない間も知らせる)では、条件をサーバーに保存し、
サーバーが自分で照合して通知する必要があります。

- 照合: 1 つのスレッドが 1 秒ごとに新しい窓を確かめ、全ルールのクエリのベクトルとの類似度を 1 回の行列の積で計算する
  (LiveHub と同じ方法)。照合を終えた窓の ID は DB(meta)に残し、再起動の後は続きから照合する。
- 抑制: 同じルールで同じ取り込み元(ストリーム・ファイル)からの通知は、cooldown_sec の間は送らない
  (同じ場面が続くと、2 秒ごとに通知が届いてしまうため)。
- 送信: 別のスレッドが JSON を POST する。本文と時刻に HMAC-SHA256 の署名を付け、
  受け取り側が偽の通知を見分けられるようにする。失敗したら 5 秒・30 秒・2 分・10 分・30 分の間隔で再送し、
  それでも届かなければ failed にする。送る内容は DB に残すため、再起動しても未送信の通知は送り直す。
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import queue
import secrets
import threading
import time
from datetime import datetime
from urllib.parse import urlsplit

import httpx
import numpy as np

log = logging.getLogger(__name__)

RETRY_DELAYS = (5, 30, 120, 600, 1800)  # 送信に失敗したときの再送の間隔(秒)
SEEN_KEY = "alerts_last_window_id"


class AlertError(ValueError):
    """通知ルールの内容の誤り(API では 400)。"""


def validate_webhook(url: str) -> str:
    url = (url or "").strip()
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname or any(c.isspace() for c in url):
        raise AlertError("webhook_url は http:// か https:// で始まる URL にしてください")
    return url


def sign(secret: str, timestamp: str, body: bytes) -> str:
    """署名: HMAC-SHA256(鍵, "<時刻>.<本文>") の 16 進表記。

    時刻を含めて、同じ通知の使い回し(リプレイ)を見分けられるようにする。
    """
    return hmac.new(secret.encode(), timestamp.encode() + b"." + body, hashlib.sha256).hexdigest()


def public_alert(alert: dict) -> dict:
    """API で返す形(鍵は伏せる)。"""
    out = {k: v for k, v in alert.items() if k != "secret"}
    out["enabled"] = bool(alert.get("enabled"))
    return out


class AlertEngine:
    def __init__(self, ctx, service) -> None:
        self._ctx = ctx
        self._svc = service
        self._store = ctx.store
        self._settings = ctx.settings
        self._rules: list[dict] = []  # 有効なルールと、そのクエリのベクトル
        self._rules_lock = threading.Lock()
        self._last_fire: dict[tuple[int, int], float] = {}
        self._outbox: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._client = httpx.Client(timeout=10, follow_redirects=False)
        self.seen = 0
        self.status = {"rules": 0, "matched": 0, "delivered": 0, "failed": 0, "last_error": None}

    # ------------------------------------------------------------------ 開始・停止
    def start(self) -> None:
        saved = self._store.get_meta(SEEN_KEY)
        self.seen = int(saved) if saved else self._store.last_window_id  # 初回は、作った時点より後の窓だけを照合する
        self.reload()
        for ev in self._store.list_alert_events(status="pending", limit=10_000):
            self._outbox.put((ev["id"], 0.0))  # 前回送れなかった通知を送り直す
        for target, name in ((self._match_loop, "alert-match"), (self._deliver_loop, "alert-send")):
            th = threading.Thread(target=target, name=name, daemon=True)
            th.start()
            self._threads.append(th)

    def stop(self) -> None:
        self._stop.set()
        self._outbox.put(None)
        for th in self._threads:
            th.join(timeout=15)
        self._client.close()

    def reload(self) -> None:
        """ルールを DB から読み直し、クエリをベクトルにします(作成・変更・削除のたびに呼ぶ)。"""
        rules = []
        for alert in self._store.list_alerts():
            if not alert["enabled"]:
                continue
            try:
                vec = np.asarray(self._svc.text_vector(alert["query"]), dtype=np.float32)
                kinds = self._svc.resolve_kinds(alert["kind"] or "auto")
            except Exception as exc:  # 推論側に接続できない場合などは、次の読み直しまで使わない
                log.warning("通知ルール %s のクエリをベクトルにできません: %s", alert["id"], exc)
                self.status["last_error"] = f"ルール {alert['id']}: {exc}"
                continue
            rules.append({**alert, "vec": vec, "kinds": kinds})
        with self._rules_lock:
            self._rules = rules
        self.status["rules"] = len(rules)

    # ------------------------------------------------------------------ 照合
    def _match_loop(self) -> None:
        while not self._stop.wait(1.0):
            try:
                self.match_once()
            except Exception as exc:
                self.status["last_error"] = f"{type(exc).__name__}: {exc}"
                log.exception("通知ルールの照合に失敗しました")

    def match_once(self) -> int:
        """新しい窓を全ルールと照合し、通知を作った数を返します。"""
        latest = self._store.last_window_id
        if latest <= self.seen:
            return 0
        with self._rules_lock:
            rules = list(self._rules)
        made = 0
        after = self.seen
        while rules and after < latest:
            rows = self._store.windows_between(after, latest)
            if not rows:
                break
            after = rows[-1]["window_id"]
            ids = np.array([r["window_id"] for r in rows], dtype=np.int64)
            vecs = self._store.vectors.get(ids)
            queries = np.stack([r["vec"] for r in rules])
            if vecs.shape[1] != queries.shape[1]:
                break
            scores = vecs @ queries.T
            for j, rule in enumerate(rules):
                for i in np.nonzero(scores[:, j] >= rule["min_score"])[0]:
                    row = rows[i]
                    if rule["kinds"] and row["kind"] not in rule["kinds"]:
                        continue
                    if rule["group_id"] and row["group_id"] != rule["group_id"]:
                        continue
                    if rule["location"] and row["location"] != rule["location"]:
                        continue
                    if self._cooling(rule, row["source_id"]):
                        continue
                    made += self._record(rule, {**row, "score": float(scores[i, j])})
        self.seen = latest
        self._store.set_meta(SEEN_KEY, str(latest))
        return made

    def _cooling(self, rule: dict, source_id: int) -> bool:
        key = (rule["id"], source_id)
        last = self._last_fire.get(key)
        if last is None:
            last = self._store.last_alert_event(rule["id"], source_id) or 0.0
            self._last_fire[key] = last
        return time.time() - last < float(rule["cooldown_sec"] or 0)

    def _record(self, rule: dict, hit: dict) -> int:
        hit = self._svc.decorate_hits([hit])[0]
        base = (self._settings.public_url or "").rstrip("/")
        for key in ("thumb_url", "media_url"):  # 受け取り側から開けるよう、PUBLIC_URL があれば絶対 URL にする
            if hit.get(key) and base:
                hit[key] = base + hit[key]
        payload = {
            "event": "alert.match",
            "alert": {"id": rule["id"], "name": rule["name"], "query": rule["query"]},
            "match": {k: hit.get(k) for k in ("window_id", "source_id", "source_name", "kind", "group_id", "location",
                                               "start_ms", "end_ms", "abs_time", "abs_ts", "score", "thumb_url",
                                               "media_url")},
        }  # fmt: skip
        event_id = self._store.add_alert_event(
            alert_id=rule["id"], window_id=hit["window_id"], source_id=hit["source_id"], score=hit["score"],
            payload=payload,
        )  # fmt: skip
        if event_id is None:
            return 0
        self._last_fire[(rule["id"], hit["source_id"])] = time.time()
        self.status["matched"] += 1
        self._outbox.put((event_id, 0.0))
        return 1

    # ------------------------------------------------------------------ 送信
    def _deliver_loop(self) -> None:
        delayed: list[tuple[float, int]] = []  # (送る時刻, 通知の ID)
        while not self._stop.is_set():
            now = time.monotonic()
            due = [d for d in delayed if d[0] <= now]
            delayed = [d for d in delayed if d[0] > now]
            for _, event_id in due:
                self._outbox.put((event_id, 0.0))
            try:
                item = self._outbox.get(timeout=1.0)
            except queue.Empty:
                continue
            if item is None:
                return
            event_id, _ = item
            retry_in = self._deliver(event_id)
            if retry_in is not None:
                delayed.append((time.monotonic() + retry_in, event_id))

    def _deliver(self, event_id: int) -> float | None:
        """1 件を送ります。再送する場合は、次に送るまでの秒数を返します。"""
        ev = self._store.get_alert_event(event_id)
        if ev is None or ev["status"] != "pending":
            return None
        alert = self._store.get_alert(ev["alert_id"])
        if alert is None:
            return None
        ok, error = self.send(alert, {**ev["payload"], "event_id": event_id})
        attempts = ev["attempts"] + 1
        if ok:
            self._store.update_alert_event(event_id, status="delivered", attempts=attempts, delivered_at=time.time(),
                                           error=None)  # fmt: skip
            self.status["delivered"] += 1
            return None
        if attempts > len(RETRY_DELAYS):
            self._store.update_alert_event(event_id, status="failed", attempts=attempts, error=error)
            self.status["failed"] += 1
            log.warning("通知を送れませんでした(再送の上限): alert=%s event=%s %s", alert["id"], event_id, error)
            return None
        self._store.update_alert_event(event_id, attempts=attempts, error=error)
        return RETRY_DELAYS[attempts - 1]

    def send(self, alert: dict, payload: dict) -> tuple[bool, str | None]:
        body = json.dumps({**payload, "sent_at": datetime.now().astimezone().isoformat(timespec="seconds")},
                          ensure_ascii=False).encode()  # fmt: skip
        ts = str(int(time.time()))
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "mediasearch-alert",
            "X-Mediasearch-Event": payload.get("event", "alert.match"),
            "X-Mediasearch-Timestamp": ts,
            "X-Mediasearch-Signature": "sha256=" + sign(alert["secret"], ts, body),
        }
        try:
            res = self._client.post(alert["webhook_url"], content=body, headers=headers)
        except httpx.HTTPError as exc:
            return False, f"{type(exc).__name__}: {exc}"
        if 200 <= res.status_code < 300:
            return True, None
        return False, f"HTTP {res.status_code}: {res.text[:200]}"

    def test(self, alert: dict) -> dict:
        """試しの通知を 1 件送ります(受け取り側の設定の確認用)。"""
        payload = {"event": "alert.test", "alert": {"id": alert["id"], "name": alert["name"], "query": alert["query"]}}
        ok, error = self.send(alert, payload)
        return {"delivered": ok, "error": error}


def new_secret() -> str:
    return secrets.token_hex(32)
