"""通知ルール(Webhook)のテスト。手元に受け取り用の HTTP サーバーを立てて確かめます。"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from conftest import make_video, needs_ffmpeg
from fastapi.testclient import TestClient
from test_e2e_dummy import wait_done

from mediasearch import alerts
from mediasearch.alerts import sign
from mediasearch.main import create_app


class Receiver:
    def __init__(self, fail_first: int = 0) -> None:
        self.received: list[dict] = []
        self.fail_first = fail_first
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                body = self.rfile.read(int(self.headers["Content-Length"]))
                outer.received.append({"headers": dict(self.headers), "body": body})
                code = 500 if outer.fail_first > 0 else 200
                outer.fail_first -= 1
                self.send_response(code)
                self.end_headers()

            def log_message(self, *args):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}/hook"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()

    def wait(self, n: int, timeout: float = 15) -> list[dict]:
        end = time.time() + timeout
        while time.time() < end and len(self.received) < n:
            time.sleep(0.1)
        return self.received


@pytest.fixture()
def receiver():
    r = Receiver()
    yield r
    r.close()


@needs_ffmpeg
def test_matching_window_is_delivered_with_signature_and_cooldown(client, receiver, tmp_path):
    rule = {"name": "赤", "query": "red", "min_score": -1, "webhook_url": receiver.url, "group_id": "cam",
            "cooldown_sec": 3600}  # fmt: skip
    created = client.post("/api/alerts", json=rule).json()
    secret = created["secret"]
    assert len(secret) == 64 and client.get(f"/api/alerts/{created['id']}").json().get("secret") is None  # 以降は伏せる
    other = {**rule, "name": "別のグループ", "group_id": "other"}
    client.post("/api/alerts", json=other)
    video = tmp_path / "20260101_090000.mp4"
    make_video(video, "red", seconds=6)  # 3 窓
    job = client.post("/api/ingest/path", json={"path": str(video), "group_id": "cam"}).json()
    assert wait_done(client, job["job_id"])["status"] == "done"
    got = receiver.wait(1)
    time.sleep(2)  # 抑制(cooldown)で、同じ取り込み元からの 2 件目以降は送らない
    assert len(got) == 1
    h, body = got[0]["headers"], got[0]["body"]
    assert h["X-Mediasearch-Signature"] == "sha256=" + sign(secret, h["X-Mediasearch-Timestamp"], body)
    payload = json.loads(body)
    assert payload["event"] == "alert.match" and payload["alert"]["name"] == "赤"
    assert payload["match"]["source_id"] == job["source_id"] and payload["match"]["group_id"] == "cam"
    events = client.get(f"/api/alerts/{created['id']}/events").json()["events"]
    assert len(events) == 1 and events[0]["status"] == "delivered"
    listing = client.get("/api/alerts").json()
    assert len(listing["alerts"]) == 2 and listing["status"]["events"] == {"delivered": 1}


@needs_ffmpeg
def test_failed_delivery_is_retried(client, tmp_path, monkeypatch):
    monkeypatch.setattr(alerts, "RETRY_DELAYS", (0.2, 0.2, 0.2))
    r = Receiver(fail_first=2)
    try:
        rule = {"name": "x", "query": "x", "min_score": -1, "webhook_url": r.url}
        aid = client.post("/api/alerts", json=rule).json()["id"]
        video = tmp_path / "20260101_090000.mp4"
        make_video(video, "blue", seconds=2)
        job = client.post("/api/ingest/path", json={"path": str(video)}).json()
        assert wait_done(client, job["job_id"])["status"] == "done"
        assert len(r.wait(3)) == 3  # 2 回失敗して、3 回目で届く
        deadline = time.time() + 5
        while time.time() < deadline:
            ev = client.get(f"/api/alerts/{aid}/events").json()["events"][0]
            if ev["status"] == "delivered":
                break
            time.sleep(0.1)
        assert ev["status"] == "delivered" and ev["attempts"] == 3
    finally:
        r.close()


def test_rule_validation_test_send_and_crud(client, receiver):
    assert client.post("/api/alerts", json={"name": "x", "query": "x", "webhook_url": "ftp://h/x"}).status_code == 400
    aid = client.post("/api/alerts", json={"name": "x", "query": "x", "webhook_url": receiver.url}).json()["id"]
    assert client.post(f"/api/alerts/{aid}/test").json() == {"delivered": True, "error": None}
    assert json.loads(receiver.wait(1)[0]["body"])["event"] == "alert.test"
    assert client.patch(f"/api/alerts/{aid}", json={"enabled": False, "min_score": 0.5}).json()["enabled"] is False
    assert client.get("/api/alerts").json()["status"]["rules"] == 0  # 無効なルールは照合しない
    assert client.delete(f"/api/alerts/{aid}").json() == {"deleted": aid}
    assert client.get(f"/api/alerts/{aid}").status_code == 404


@needs_ffmpeg
def test_pending_events_are_sent_after_restart(settings, tmp_path, monkeypatch):
    """送る前に停止しても、通知の内容は DB に残っており、再起動の後に送る。"""
    r = Receiver()
    try:
        with TestClient(create_app(settings)) as c:
            rule = {"name": "x", "query": "x", "min_score": -1, "webhook_url": "http://127.0.0.1:9/unreachable"}
            aid = c.post("/api/alerts", json=rule).json()["id"]
            video = tmp_path / "20260101_090000.mp4"
            make_video(video, "red", seconds=2)
            job = c.post("/api/ingest/path", json={"path": str(video)}).json()
            assert wait_done(c, job["job_id"])["status"] == "done"
            deadline = time.time() + 10
            while time.time() < deadline and not c.get(f"/api/alerts/{aid}/events").json()["events"]:
                time.sleep(0.1)
            c.patch(f"/api/alerts/{aid}", json={"webhook_url": r.url})  # 受け取り先を直してから停止する
        with TestClient(create_app(settings)):
            assert len(r.wait(1)) == 1
    finally:
        r.close()
