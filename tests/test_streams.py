"""RTSP ストリームの取り込みのテスト。

CI には RTSP のサーバーが無いため、受信の処理は、ローカルのファイルを実時間の速さで読ませて確認する
(ffmpeg に渡す引数と、窓を作る処理は RTSP と同じ)。実際の RTSP での確認手順は docs/operations.md を参照。
"""

from __future__ import annotations

import time
from dataclasses import replace

import pytest
from conftest import make_video, needs_ffmpeg
from fastapi.testclient import TestClient

from mediasearch import streams
from mediasearch.main import create_app
from mediasearch.pipeline import IngestParams
from mediasearch.streams import StreamError, decode_step_for, mask_url, validate_url


def test_url_validation_and_masking():
    assert validate_url(" rtsp://cam.local/stream1 ") == "rtsp://cam.local/stream1"
    for bad in ("http://cam.local/x", "file:///etc/passwd", "rtsp:///nohost", "rtsp://a b", ""):
        with pytest.raises(StreamError):
            validate_url(bad)
    assert mask_url("rtsp://admin:secret@10.0.0.1:554/s?x=1") == "rtsp://admin:***@10.0.0.1:554/s?x=1"
    assert mask_url("rtsp://10.0.0.1/s") == "rtsp://10.0.0.1/s"


def test_decode_step_for_stream():
    step, offsets, stride = decode_step_for(IngestParams(2, 2, 0, False))
    assert (step, offsets, stride) == (500, [500, 1500], 2000)
    step, offsets, stride = decode_step_for(IngestParams(4, 4, 2, False))
    assert stride == 2000 and all(o % step == 0 for o in offsets) and stride % step == 0


def test_stream_api_masks_credentials_and_limits(settings):
    s = replace(settings, max_streams=1)
    with TestClient(create_app(s)) as c:
        assert c.post("/api/streams", json={"url": "http://cam.local/x"}).status_code == 400
        body = {"url": "rtsp://admin:secret@cam.invalid/s1", "group_id": "gate", "enabled": False}
        r = c.post("/api/streams", json=body)
        assert r.status_code == 200, r.text
        st = r.json()
        assert st["url"] == "rtsp://admin:***@cam.invalid/s1" and "secret" not in r.text
        assert st["enabled"] is False and st["state"]["status"] == "stopped"
        assert c.get("/api/streams").json()["streams"][0]["id"] == st["id"]
        assert c.get(f"/api/sources/{st['source_id']}").json()["kind"] == "stream"
        # 同時に受信できる数(MAX_STREAMS)を超える登録は拒否する
        c.post("/api/streams", json={"url": "rtsp://cam.invalid/s2"})
        assert c.post("/api/streams", json={"url": "rtsp://cam.invalid/s3"}).status_code == 400
        assert c.delete(f"/api/streams/{st['id']}").json() == {"deleted": st["id"]}
        assert c.get(f"/api/streams/{st['id']}").status_code == 404


@needs_ffmpeg
def test_stream_worker_creates_windows_while_receiving(settings, tmp_path, monkeypatch):
    # テスト用に、ffmpeg がローカルのファイルを読めるようにする(本番は RTSP 関連のプロトコルだけ)
    monkeypatch.setattr(streams, "PROTOCOL_WHITELIST", streams.PROTOCOL_WHITELIST + ",file")
    video = tmp_path / "cam.mp4"
    make_video(video, "red", seconds=5, audio="tone")
    with TestClient(create_app(settings)) as c:
        manager = c.app.state.ctx.streams
        store = manager._store
        params = IngestParams(2, 2, 0, True)
        source_id = store.add_source(kind="stream", path=None, name="cam", group_id="gate", location=None,
                                     start_ts=time.time(), params=params.to_dict(), status="running")  # fmt: skip
        stream_id = store.add_stream(name="cam", url=f"file:{video}", group_id="gate", location=None,
                                     params=params.to_dict(), source_id=source_id)  # fmt: skip
        manager._start_worker(store.get_stream(stream_id))
        deadline = time.time() + 30
        state = {}
        while time.time() < deadline:
            state = c.get(f"/api/streams/{stream_id}").json()["state"]
            if state["windows"] >= 2:
                break
            time.sleep(0.3)
        assert state["windows"] >= 2 and state["has_audio"] is True, state
        query = {"query": "red", "min_score": -1, "group_id": "gate", "kind": "tav"}
        hits = c.post("/api/search/text", json=query).json()["results"]
        assert hits and {h["kind"] for h in hits} == {"tav"} and hits[0]["source_name"] == "cam"
        assert hits[0]["thumb_url"] and c.get(hits[0]["thumb_url"]).status_code == 200
        assert hits[0]["media_url"] is None  # ストリームには元のファイルが無い
        stopped = c.post(f"/api/streams/{stream_id}/stop").json()
        assert stopped["enabled"] is False and stopped["state"]["status"] == "stopped"

