"""サムネイルの置き場所(フォルダの分割・以前の版の場所・削除のときの後始末)のテスト。"""

from __future__ import annotations

import io

from PIL import Image
from test_e2e_dummy import wait_done

from mediasearch import thumbs


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (200, 30, 30)).save(buf, "PNG")
    return buf.getvalue()


def test_paths(tmp_path):
    assert thumbs.path_for(tmp_path, 12) == tmp_path / "0" / "w12.jpg"
    assert thumbs.path_for(tmp_path, 1_234_567) == tmp_path / "123" / "w1234567.jpg"
    assert thumbs.find(tmp_path, 5) is None
    old = thumbs.legacy_path(tmp_path, 5)
    old.write_bytes(b"x")
    assert thumbs.find(tmp_path, 5) == old  # 以前の版の場所も探す
    assert thumbs.remove(tmp_path, [5, 6]) == 1 and not old.exists()


def test_thumbnails_removed_with_source(client, settings):
    root = settings.data_dir / "thumbs"
    item = client.post("/api/ingest/image", files=[("files", ("a.png", _png(), "image/png"))]).json()["items"][0]
    assert wait_done(client, item["job_id"])["status"] == "done"
    hit = client.post("/api/search/text", json={"query": "red", "min_score": -1}).json()["results"][0]
    wid = hit["window_id"]
    assert thumbs.path_for(root, wid).is_file()
    # 以前の版の場所にだけあるサムネイルも配信する
    legacy = thumbs.legacy_path(root, wid)
    thumbs.path_for(root, wid).rename(legacy)
    assert client.get(f"/api/thumb/{wid}").status_code == 200
    assert client.delete(f"/api/sources/{item['source_id']}").status_code in (200, 204)
    assert not legacy.exists() and not thumbs.path_for(root, wid).exists()  # 削除した窓のサムネイルは残さない
