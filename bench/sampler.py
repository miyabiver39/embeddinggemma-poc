"""サーバーへの接続と、資源の使用状況の定期的な取得(GET /api/metrics)。

Python の標準ライブラリだけで動きます(サーバーと別のマシンからでも実行できる)。
"""

from __future__ import annotations

import json
import mimetypes
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path


class ApiError(RuntimeError):
    pass


class Client:
    def __init__(self, url: str, token: str | None = None, timeout: float = 600) -> None:
        self.url = url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def request(self, method: str, path: str, body: bytes | None = None, content_type: str | None = None) -> dict:
        req = urllib.request.Request(self.url + path, data=body, method=method)
        if content_type:
            req.add_header("Content-Type", content_type)
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as res:
                return json.loads(res.read() or b"{}")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            raise ApiError(f"{method} {path} が失敗しました({exc.code}): {detail}") from exc
        except urllib.error.URLError as exc:
            raise ApiError(f"{self.url} に接続できません: {exc.reason}") from exc

    def get(self, path: str) -> dict:
        return self.request("GET", path)

    def post(self, path: str, payload: dict | None = None) -> dict:
        return self.request("POST", path, json.dumps(payload or {}).encode(), "application/json")

    def delete(self, path: str) -> dict:
        return self.request("DELETE", path)

    def post_file(self, path: str, file: Path, fields: dict[str, str], field: str = "file") -> dict:
        boundary = uuid.uuid4().hex
        chunks = [
            f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
            for k, v in fields.items()
        ]
        ctype = mimetypes.guess_type(file.name)[0] or "application/octet-stream"
        head = (
            f'--{boundary}\r\nContent-Disposition: form-data; name="{field}"; filename="{file.name}"\r\n'
            f"Content-Type: {ctype}\r\n\r\n"
        )
        body = b"".join(chunks) + head.encode() + file.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
        return self.request("POST", path, body, f"multipart/form-data; boundary={boundary}")

    def wait_job(self, job_id: int, poll: float = 0.5) -> dict:
        while True:
            job = self.get(f"/api/jobs/{job_id}")
            if job["status"] in ("done", "failed"):
                return job
            time.sleep(poll)


class Sampler:
    """別のスレッドで、一定の間隔で /api/metrics(と、分離構成なら /compute/metrics)を読みます。"""

    def __init__(self, app: Client, compute: Client | None = None, interval: float = 1.0) -> None:
        self.app = app
        self.compute = compute
        self.interval = interval
        self.samples: list[dict] = []  # {"t": 経過秒, "app": {...}, "compute": {...}, "marks": [...]}
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = 0.0
        self._marks: list[str] = []
        self._lock = threading.Lock()

    def mark(self, text: str) -> None:
        """次の計測に、出来事(「取り込み開始」など)を記録します。レポートの表に出ます。"""
        with self._lock:
            self._marks.append(text)

    def _take(self) -> None:
        sample: dict = {"t": round(time.monotonic() - self._t0, 3)}
        try:
            sample["app"] = self.app.get("/api/metrics")
        except ApiError as exc:
            self.errors.append(str(exc))
            return
        if self.compute is not None:
            try:
                sample["compute"] = self.compute.get("/compute/metrics")
            except ApiError as exc:
                self.errors.append(str(exc))
        with self._lock:
            sample["marks"], self._marks = self._marks, []
        self.samples.append(sample)

    def _loop(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            self._take()
            self._stop.wait(max(self.interval - (time.monotonic() - started), 0.05))

    def start(self) -> None:
        self._t0 = time.monotonic()
        self._take()  # 開始の時点の値(差を取るための起点)
        self._thread = threading.Thread(target=self._loop, name="sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(10)
        self._take()  # 終了の時点の値
