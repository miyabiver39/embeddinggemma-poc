"""静止画の取り込みと検索のテスト(ダミー埋め込み)。"""

from __future__ import annotations

import io
from datetime import datetime

from conftest import make_video, needs_ffmpeg
from PIL import Image
from test_e2e_dummy import wait_done

from mediasearch import media


def _png(color: tuple[int, int, int], size: int = 64) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (size, size), color).save(buf, "PNG")
    return buf.getvalue()


def _jpeg_with_exif(color: tuple[int, int, int], taken: str) -> bytes:
    img = Image.new("RGB", (64, 48), color)
    exif = Image.Exif()
    exif.get_ifd(0x8769)[0x9003] = taken  # DateTimeOriginal
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=exif.tobytes())
    return buf.getvalue()


def test_upload_images_and_search(client):
    files = [
        ("files", ("blue.png", _png((30, 30, 220)), "image/png")),
        ("files", ("photo.jpg", _jpeg_with_exif((220, 30, 30), "2026:05:01 12:34:56"), "image/jpeg")),
        ("files", ("broken.png", b"not an image", "image/png")),
    ]
    r = client.post("/api/ingest/image", files=files, data={"group_id": "album"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["queued"] == 2 and len(body["errors"]) == 1 and body["errors"][0]["name"] == "broken.png"
    red = next(i for i in body["items"] if i["start_ts_from"] == "exif")
    assert red["start_ts"] == "2026-05-01T12:34:56"
    for item in body["items"]:
        assert wait_done(client, item["job_id"])["status"] == "done"

    assert client.get("/api/info").json()["index"]["windows_by_kind"]["image"] == 2
    # 文字のクエリの auto は、画像も探す
    hits = client.post("/api/search/text", json={"query": "blue"}).json()["results"]
    assert hits[0]["kind"] == "image" and hits[0]["source_name"] == "blue.png"
    # 画像のクエリでも見つかる。サムネイルと元画像を返せる
    q = client.post("/api/search/image", files={"file": ("q.png", _png((200, 20, 20)), "image/png")}).json()
    top = q["results"][0]
    assert top["source_name"] == "photo.jpg" and top["group_id"] == "album"
    assert client.get(top["thumb_url"]).headers["content-type"] == "image/jpeg"
    assert client.get(top["media_url"]).status_code == 200
    # kind=image に絞り込める
    only = client.post("/api/search/text", json={"query": "blue", "kind": "image"}).json()["results"]
    assert {h["kind"] for h in only} == {"image"}


def test_image_by_path_and_dir(client, tmp_path):
    root = tmp_path / "photos"
    (root / "site-a").mkdir(parents=True)
    (root / "site-a" / "20260102_080000.png").write_bytes(_png((30, 180, 30)))
    (root / "tiny.png").write_bytes(_png((30, 180, 30), size=2))  # 小さすぎる画像
    r = client.post("/api/ingest/path", json={"path": str(root / "site-a" / "20260102_080000.png")})
    assert r.status_code == 200, r.text
    assert r.json()["start_ts_from"] == "filename"
    # 画像を kind=video として指定すると拒否する
    bad = client.post("/api/ingest/path", json={"path": str(root / "tiny.png"), "kind": "video"})
    assert bad.status_code == 422
    d = client.post("/api/ingest/dir", json={"dir": str(root), "group_from_dir": True}).json()
    assert d["duplicates"] == 1 and d["queued"] == 0 and len(d["errors"]) == 1  # tiny.png は小さすぎる


@needs_ffmpeg
def test_mixed_folder_auto_kinds(client, tmp_path):
    root = tmp_path / "mixed"
    root.mkdir()
    make_video(root / "clip.mp4", "red", seconds=2)
    (root / "still.png").write_bytes(_png((220, 30, 30)))
    d = client.post("/api/ingest/dir", json={"dir": str(root)}).json()
    assert d["queued"] == 2, d
    kinds = sorted(client.get(f"/api/sources/{i['source_id']}").json()["kind"] for i in d["items"])
    assert kinds == ["image", "video"]
    assert client.get("/api/sources/99999").status_code == 404


def test_image_taken_at(tmp_path):
    p = tmp_path / "x.jpg"
    p.write_bytes(_jpeg_with_exif((1, 2, 3), "2025:12:31 23:59:00"))
    assert media.image_taken_at(p) == datetime(2025, 12, 31, 23, 59, 0).timestamp()
    q = tmp_path / "y.png"
    q.write_bytes(_png((1, 2, 3)))
    assert media.image_taken_at(q) is None


def test_many_images_are_embedded_in_batches(client, tmp_path):
    root = tmp_path / "batch"
    root.mkdir()
    for i in range(6):
        (root / f"img{i}.png").write_bytes(_png((40 * i, 30, 200)))
    d = client.post("/api/ingest/dir", json={"dir": str(root)}).json()
    assert d["queued"] == 6
    jobs = [wait_done(client, item["job_id"]) for item in d["items"]]
    assert {j["status"] for j in jobs} == {"done"}
    # 各ジョブに処理時間が残り、続けて登録された画像はまとめて推論している
    assert all(j["timings"]["windows"] >= 1 and "embed" in j["timings"]["stages_ms"] for j in jobs)
    assert max(j["timings"]["batch"] for j in jobs) > 1
