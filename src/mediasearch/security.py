"""HTTP の入口で行うセキュリティ対策(ASGI ミドルウェア)。

既定値のままでも起動できること(追加の設定なしで動くこと)を保ちながら、次を行います。

  1. トークン認証(API_TOKEN を設定したときだけ)
       Authorization: Bearer <トークン> / X-API-Key ヘッダー / Cookie(mediasearch_token)のどれかで受け付けます。
       Cookie は、<img> や <video> のようにヘッダーを付けられない WebUI のリクエストのためです。
  2. CSRF(他サイトのページからの、なりすましリクエスト)の防止
       更新系のメソッド(POST / PUT / PATCH / DELETE)は、ブラウザが付ける Origin / Sec-Fetch-Site を確認し、
       別のサイトからのものを 403 で拒否します。curl などブラウザ以外は、これらのヘッダーを付けないので影響しません。
  3. Host ヘッダーの確認(ALLOWED_HOSTS を設定したときだけ)
       DNS リバインディング(攻撃者のドメインをこのサーバの IP に向ける手口)への対策です。
  4. リクエスト本文の大きさの上限(MAX_UPLOAD_MB)
       巨大なアップロードでディスクやメモリを使い切られないようにします。
  5. セキュリティ関連の応答ヘッダー
       MIME の推測の禁止、他サイトへの埋め込み(クリックジャッキング)の禁止、WebUI の CSP など。

FastAPI の依存関係ではなく ASGI ミドルウェアにしているのは、ルーターの追加漏れで認証が抜けることを防ぎ、
本文を読み込む前(multipart の展開前)に上限を確かめるためです。
"""

from __future__ import annotations

import hmac
import json
import logging
from http.cookies import SimpleCookie
from urllib.parse import urlsplit

from .config import Settings

log = logging.getLogger(__name__)

TOKEN_COOKIE = "mediasearch_token"
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
# 認証なしで開けるパス。/healthz は Docker の死活監視、/ と /docs は WebUI と API 仕様の画面
# (画面自体には情報がなく、データの取得には認証が要る)
PUBLIC_PATHS = frozenset({"/", "/healthz", "/docs", "/docs/oauth2-redirect", "/redoc"})

# WebUI(/)に付ける CSP。WebUI は 1 ファイルで、スクリプトとスタイルを埋め込んでいるため 'unsafe-inline' が必要
WEBUI_CSP = (
    "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data: blob:; media-src 'self' blob:; connect-src 'self'; "
    "object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
)
COMMON_HEADERS = [
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),
    (b"referrer-policy", b"no-referrer"),
    (b"cross-origin-resource-policy", b"same-origin"),
]


class BodyTooLarge(Exception):
    pass


def _headers(scope) -> dict[str, str]:
    return {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}


def _host_name(host: str) -> str:
    """Host ヘッダーからポートを除いたホスト名(IPv6 の [::1]:8000 にも対応)。"""
    host = host.strip().lower()
    if host.startswith("["):
        return host[1 : host.find("]")] if "]" in host else host
    return host.rsplit(":", 1)[0] if host.count(":") == 1 else host


def token_from_request(headers: dict[str, str]) -> str | None:
    auth = headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    if headers.get("x-api-key"):
        return headers["x-api-key"].strip()
    if "cookie" in headers:
        cookie = SimpleCookie()
        try:
            cookie.load(headers["cookie"])
        except Exception:  # 壊れた Cookie は、トークンなしとして扱う
            return None
        if TOKEN_COOKIE in cookie:
            return cookie[TOKEN_COOKIE].value
    return None


class SecurityMiddleware:
    def __init__(self, app, settings: Settings) -> None:
        self.app = app
        self.token = settings.api_token
        self.allowed_origins = {o.lower() for o in settings.allowed_origins}
        self.allowed_hosts = set(settings.allowed_hosts)
        self.max_body = settings.max_upload_mb * 1024 * 1024

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = _headers(scope)
        path = scope.get("path", "")
        method = scope.get("method", "GET").upper()

        refusal = self._check(method, path, headers)
        if refusal is not None:
            await _send_json(send, *refusal)
            return

        if self.max_body:
            length = headers.get("content-length")
            if length and length.isdigit() and int(length) > self.max_body:
                await _send_json(send, 413, self._too_large_message())
                return
            receive, exceeded = self._limited(receive)
        else:
            exceeded = None

        started = replaced = False

        async def send_with_headers(message):
            nonlocal started, replaced
            # 本文の途中で上限を超えた場合、FastAPI は本文の解析エラー(400)として応答するため、413 に差し替える
            if replaced:
                return
            if message["type"] == "http.response.start" and exceeded and exceeded():
                replaced = started = True
                await _send_json(send, 413, self._too_large_message())
                return
            if message["type"] == "http.response.start":
                started = True
                extra = list(COMMON_HEADERS)
                if path == "/":
                    extra.append((b"content-security-policy", WEBUI_CSP.encode()))
                message = {**message, "headers": [*message.get("headers", []), *extra]}
            await send(message)

        try:
            await self.app(scope, receive, send_with_headers)
        except BodyTooLarge:
            if not started:
                await _send_json(send, 413, self._too_large_message())

    def _check(self, method: str, path: str, headers: dict[str, str]) -> tuple[int, str, dict] | None:
        """拒否する場合は (状態コード, メッセージ, 追加ヘッダー) を返します。"""
        host = _host_name(headers.get("host", ""))
        if "*" not in self.allowed_hosts and host not in self.allowed_hosts:
            return 403, f"許可されていないホスト名です: {host}(ALLOWED_HOSTS を確認してください)", {}

        if method not in SAFE_METHODS:
            origin = headers.get("origin")
            if origin:
                o = origin.lower().rstrip("/")
                same = urlsplit(o).netloc == headers.get("host", "").lower()
                if not same and o not in self.allowed_origins:
                    return 403, "別のサイトからのリクエストは受け付けません(ALLOWED_ORIGINS で許可できます)", {}
            elif headers.get("sec-fetch-site") == "cross-site":
                return 403, "別のサイトからのリクエストは受け付けません", {}

        if self.token and path not in PUBLIC_PATHS:
            given = token_from_request(headers)
            if given is None or not hmac.compare_digest(given.encode(), self.token.encode()):
                return 401, "認証が必要です(API_TOKEN で設定したトークンを指定してください)", {
                    "www-authenticate": "Bearer"
                }
        return None

    def _too_large_message(self) -> str:
        return f"リクエストが大きすぎます(上限 {self.max_body // (1024 * 1024)} MB。MAX_UPLOAD_MB で変更できます)"

    def _limited(self, receive):
        """Content-Length の無い(chunked)リクエストも、受け取った量で上限を確かめます。

        受け取る関数と、上限を超えたかどうかを返す関数の組を返します。
        """
        received = 0
        over = False

        async def wrapped():
            nonlocal received, over
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body:
                    over = True
                    raise BodyTooLarge
            return message

        return wrapped, lambda: over


async def _send_json(send, status: int, detail: str, headers: dict | None = None) -> None:
    body = json.dumps({"detail": detail}, ensure_ascii=False).encode()
    raw = [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode()), *COMMON_HEADERS]
    raw += [(k.encode(), v.encode()) for k, v in (headers or {}).items()]
    await send({"type": "http.response.start", "status": status, "headers": raw})
    await send({"type": "http.response.body", "body": body})


def startup_warnings(settings: Settings) -> None:
    """安全でない設定のまま起動したときに、ログで知らせます(起動は止めません)。"""
    if not settings.api_token:
        log.warning(
            "認証が無効です(API_TOKEN が未設定)。同じネットワークの誰でも映像の取得・削除ができます。"
            "信頼できるネットワーク以外で使う場合は API_TOKEN を設定してください(docs/security.md)"
        )
    elif len(settings.api_token) < 16:
        log.warning("API_TOKEN が短すぎます(16 文字以上を推奨)。推測されにくい値にしてください")
    if settings.max_upload_mb == 0:
        log.warning("MAX_UPLOAD_MB=0 のため、アップロードの大きさに上限がありません")
