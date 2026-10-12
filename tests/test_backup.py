"""バックアップ(動作中の作成・世代の管理・壊れの検出・復元)のテスト。"""

from __future__ import annotations

import io
import json
from dataclasses import replace

from fastapi.testclient import TestClient
from PIL import Image
from test_e2e_dummy import wait_done

from mediasearch import backup, thumbs
from mediasearch.main import create_app

FULL = "full-token-0123456789abcdef"
READ = "read-token-0123456789abcdef"


def _png(color) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(buf, "PNG")
    return buf.getvalue()


def _ingest(c, color, name):
    item = c.post("/api/ingest/image", files=[("files", (name, _png(color), "image/png"))]).json()["items"][0]
    assert wait_done(c, item["job_id"])["status"] == "done"
    return item


def test_backup_and_restore(settings):
    s = replace(settings, api_token=FULL, api_read_token=READ, backup_keep=2)
    with TestClient(create_app(s), headers={"Authorization": f"Bearer {FULL}"}) as c:
        _ingest(c, (200, 30, 30), "red.png")
        assert c.post("/api/admin/backup", headers={"Authorization": f"Bearer {READ}"}).status_code == 403
        res = c.post("/api/admin/backup")
        assert res.status_code == 200, res.text
        first = res.json()
        assert first["windows"] == 1 and set(first["files"]) == {"mediasearch.db", "vectors.usearch"}
        later = _ingest(c, (30, 30, 200), "blue.png")
        for _ in range(2):
            assert c.post("/api/admin/backup").status_code == 200
        listed = c.get("/api/admin/backups", headers={"Authorization": f"Bearer {READ}"}).json()
        assert len(listed["backups"]) == 2  # BACKUP_KEEP=2 を超えた古いものは削除する
        assert first["name"] not in [b["name"] for b in listed["backups"]]
        assert c.get("/api/info").json()["backup"]["last"]["windows"] == 2
        assert "\nmediasearch_backup_last_timestamp_seconds " in c.get("/metrics").text
        hits = c.post("/api/search/text", json={"query": "x", "min_score": -1, "top_k": 10}).json()["results"]
        newest = max(h["window_id"] for h in hits)

    # ファイルが壊れたバックアップは検出する
    third, second = listed["backups"]  # 新しい順
    target = s.backup_dir / third["name"]
    assert backup.verify(target)["windows"] == 2
    (s.backup_dir / second["name"] / "mediasearch.db").write_bytes(b"broken")
    assert backup.main(["verify", str(s.backup_dir / second["name"])]) == 1

    # 窓が 1 つの時点のバックアップを作り、その後に窓を足してから戻す
    with TestClient(create_app(s), headers={"Authorization": f"Bearer {FULL}"}) as c:
        c.delete(f"/api/sources/{later['source_id']}")
        snap = c.post("/api/admin/backup").json()
        _ingest(c, (30, 200, 30), "green.png")  # この窓のサムネイルは、戻した後に残してはいけない
        hits = c.post("/api/search/text", json={"query": "x", "min_score": -1, "top_k": 10}).json()["results"]
        green = max(h["window_id"] for h in hits)
    assert green > newest and thumbs.path_for(s.data_dir / "thumbs", green).is_file()
    assert backup.main(["restore", str(s.backup_dir / snap["name"]), "--data-dir", str(s.data_dir)]) == 0
    assert not thumbs.path_for(s.data_dir / "thumbs", green).exists()
    with TestClient(create_app(s), headers={"Authorization": f"Bearer {FULL}"}) as c:
        info = c.get("/api/info").json()
        assert info["index"]["windows"] == 1
        names = [x["name"] for x in c.get("/api/sources").json()["sources"]]
        assert names == ["red.png"]
        hits = c.post("/api/search/text", json={"query": "x", "min_score": -1}).json()["results"]
        assert len(hits) == 1 and c.get(f"/api/thumb/{hits[0]['window_id']}").status_code == 200
    manifest = json.loads((s.backup_dir / snap["name"] / "manifest.json").read_text())
    assert manifest["schema_version"] and manifest["model_id"]
