"""文書(PDF・Word・テキスト)の取り込みと検索のテスト。"""

from __future__ import annotations

import os
import sqlite3
import time
import zipfile
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from test_e2e_dummy import wait_done

from mediasearch import documents
from mediasearch.documents import Chunk, DocumentError, Page, chunk_pages, extract
from mediasearch.main import create_app
from mediasearch.store import SCHEMA_VERSION, Store


# ---------------------------------------------------------------------------- 文書を作る(テスト用)
def make_pdf(path, pages: list[str]) -> None:
    """文字だけの PDF を作ります(ASCII のみ。ページごとに 1 行ずつ書く)。"""
    objs: list[bytes] = []
    kids = " ".join(f"{3 + i * 2} 0 R" for i in range(len(pages)))
    objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objs.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode())
    font_id = 3 + len(pages) * 2
    for i, text in enumerate(pages):
        content_id = 4 + i * 2
        objs.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content_id} 0 R "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>".encode()
        )
        lines = "".join(f"({line}) Tj 0 -14 Td " for line in text.split("\n"))
        stream = f"BT /F1 12 Tf 72 720 Td {lines}ET".encode()
        objs.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for n, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{n} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(bytes(out))


def make_docx(path, pages: list[list[str]]) -> None:
    """段落の並びを、Word の改ページの印(lastRenderedPageBreak)で区切った docx を作ります。"""
    w = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    body = []
    for i, paras in enumerate(pages):
        for j, text in enumerate(paras):
            brk = "<w:lastRenderedPageBreak/>" if i > 0 and j == 0 else ""
            body.append(f"<w:p><w:r>{brk}<w:t>{text}</w:t></w:r></w:p>")
    xml = (
        f'<?xml version="1.0" encoding="UTF-8"?><w:document xmlns:w="{w}">'
        f'<w:body>{"".join(body)}</w:body></w:document>'
    )
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("[Content_Types].xml", "<Types/>")
        z.writestr("word/document.xml", xml)


# ---------------------------------------------------------------------------- 読み込みと分割
def test_pdf_pages_are_extracted(tmp_path):
    pdf = tmp_path / "a.pdf"
    make_pdf(pdf, ["apple orchard report", "banana harvest summary"])
    ex = extract(pdf)
    assert ex.total_pages == 2
    assert [p.number for p in ex.pages] == [1, 2]
    assert "apple" in ex.pages[0].text and "banana" in ex.pages[1].text


def test_pdf_without_text_is_rejected(tmp_path):
    pdf = tmp_path / "scan.pdf"
    make_pdf(pdf, [""])
    with pytest.raises(DocumentError, match="文字を取り出せません"):
        extract(pdf)


def test_docx_counts_pages_from_rendered_breaks(tmp_path):
    doc = tmp_path / "a.docx"
    make_docx(doc, [["第一章 りんご"], ["第二章 みかん", "続き"]])
    ex = extract(doc)
    assert [(p.number, p.text) for p in ex.pages] == [(1, "第一章 りんご"), (2, "第二章 みかん\n続き")]


def test_text_in_shift_jis_is_read(tmp_path):
    txt = tmp_path / "memo.txt"
    txt.write_bytes("会議のメモ\n来週は倉庫を点検する".encode("cp932"))
    ex = extract(txt)
    assert ex.pages == [Page(0, "会議のメモ\n来週は倉庫を点検する")]


@pytest.mark.parametrize(
    ("name", "data", "message"),
    [
        ("fake.pdf", b"hello", "PDF の形式ではありません"),
        ("fake.docx", b"not a zip", "docx"),
        ("bin.txt", b"\x00\x01\x02\x03", "バイナリ"),
    ],
)
def test_check_rejects_broken_files(tmp_path, name, data, message):
    p = tmp_path / name
    p.write_bytes(data)
    with pytest.raises(DocumentError, match=message):
        documents.check(p)


def test_chunks_respect_size_pages_and_overlap():
    para = "。".join(f"文{i:03d}の内容です" for i in range(60)) + "。"  # 約 660 字
    chunks = chunk_pages([Page(1, para), Page(2, "短いページ")], chunk_chars=200, overlap_chars=30)
    first = [c for c in chunks if c.page == 1]
    assert len(first) >= 3
    assert all(len(c.text) <= 200 + 30 + 1 for c in first)  # 重なりの分だけ長くなる
    assert first[1].text[:10] in first[0].text  # 前の塊の終わりが、次の塊の頭に重なる
    assert chunks[-1] == Chunk(2, "短いページ")  # ページをまたがない


def test_long_text_is_truncated(tmp_path):
    txt = tmp_path / "long.txt"
    txt.write_text("あ" * 5000, encoding="utf-8")
    ex = extract(txt, max_chars=1000)
    assert ex.truncated and len(ex.pages[0].text) == 1000


# ---------------------------------------------------------------------------- 取り込みと検索
def _ingest(client, path, **extra) -> dict:
    r = client.post("/api/ingest/path", json={"path": str(path), **extra})
    assert r.status_code == 200, r.text
    job = wait_done(client, r.json()["job_id"])
    assert job["status"] == "done", job
    return {**r.json(), "job": job}


def _search(client, query, **extra):
    r = client.post("/api/search/text", json={"query": query, **extra})
    assert r.status_code == 200, r.text
    return r.json()


def test_document_ingest_and_search(client, tmp_path):
    pdf = tmp_path / "report.pdf"
    make_pdf(pdf, ["apple orchard inspection", "banana warehouse inventory"])
    res = _ingest(client, pdf)
    assert res["job"]["total"] == 2 and res["job"]["timings"]["pages"] == 2
    assert res["start_ts_from"] == "mtime"

    out = _search(client, "banana warehouse", kind="document")
    top = out["results"][0]
    assert top["kind"] == "document" and top["page"] == 2 and "banana" in top["text"]
    assert top["thumb_url"] is None and top["media_url"] == f"/api/media/{res['source_id']}"
    assert out["intervals"] == []  # 文書には時刻の区間が無い

    # 既定(auto)の検索は映像・静止画が対象で、文書は混ざらない
    assert all(h["kind"] != "document" for h in _search(client, "banana warehouse")["results"])

    # 元の文書は、ダウンロードとして返す
    r = client.get(top["media_url"])
    assert r.status_code == 200 and r.headers["content-type"] == "application/pdf"
    assert r.headers["content-disposition"].startswith("attachment")

    # 切り出しはできない
    clip = client.get(f"/api/sources/{res['source_id']}/clip", params={"start_ms": 0, "end_ms": 1000})
    assert clip.status_code == 400


def test_document_upload_docx_and_similar(client, tmp_path):
    doc = tmp_path / "manual.docx"
    make_docx(doc, [["cherry blossom festival"], ["grape juice factory"]])
    with doc.open("rb") as f:
        r = client.post("/api/ingest/document", files={"file": ("manual.docx", f)}, data={"group_id": "docs"})
    assert r.status_code == 200, r.text
    assert wait_done(client, r.json()["job_id"])["status"] == "done"
    hit = _search(client, "grape juice", kind="document", group_id="docs")["results"][0]
    assert hit["page"] == 2

    sim = client.post("/api/search/similar", json={"window_id": hit["window_id"]}).json()
    assert sim["results"] and all(h["kind"] == "document" for h in sim["results"])


def test_upload_rejects_non_documents(client, tmp_path):
    r = client.post("/api/ingest/document", files={"file": ("x.pdf", b"not a pdf")})
    assert r.status_code == 422
    r = client.post("/api/ingest/document", files={"file": ("x.exe", b"MZ")})
    assert r.status_code == 422


def test_store_media_false_keeps_no_text(client, tmp_path):
    txt = tmp_path / "secret.txt"
    txt.write_text("melon contract terms", encoding="utf-8")
    res = _ingest(client, txt, store_media=False)
    hit = _search(client, "melon contract", kind="document")["results"][0]
    assert hit["source_id"] == res["source_id"]
    assert hit["text"] is None and hit["media_url"] is None and hit["page"] == 0


def test_dir_ingest_picks_documents_and_reindex(client, tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_text("# lemon\nlemon tart recipe", encoding="utf-8")
    make_docx(tmp_path / "docs" / "b.docx", [["peach pie recipe"]])
    r = client.post("/api/ingest/dir", json={"dir": str(tmp_path / "docs"), "kind": "document"})
    assert r.status_code == 200 and r.json()["queued"] == 2, r.text
    for item in r.json()["items"]:
        assert wait_done(client, item["job_id"])["status"] == "done"
    sid = r.json()["items"][0]["source_id"]
    before = client.get("/api/info").json()["index"]["windows_by_kind"]["document"]
    r = client.post(f"/api/sources/{sid}/reindex")
    assert r.status_code == 200, r.text
    assert wait_done(client, r.json()["job_id"])["status"] == "done"
    assert client.get("/api/info").json()["index"]["windows_by_kind"]["document"] == before  # 重複しない


def test_deleting_source_removes_texts(client, settings, tmp_path):
    txt = tmp_path / "plum.txt"
    txt.write_text("plum wine", encoding="utf-8")
    sid = _ingest(client, txt)["source_id"]
    assert client.delete(f"/api/sources/{sid}").status_code == 200
    db = sqlite3.connect(settings.data_dir / "mediasearch.db")
    assert db.execute("SELECT COUNT(*) FROM window_texts").fetchone()[0] == 0


def test_retention_keeps_documents(settings, tmp_path):
    txt = tmp_path / "old.txt"
    txt.write_text("kiwi history", encoding="utf-8")
    old = time.time() - 400 * 86400
    os.utime(txt, (old, old))  # 400 日前に更新した文書
    with TestClient(create_app(replace(settings, retention_days=30))) as c:
        sid = _ingest(c, txt)["source_id"]
        c.app.state.ctx.store.expire_windows(time.time() - 30 * 86400, time.time())
        hits = _search(c, "kiwi history", kind="document")["results"]
        assert hits and hits[0]["source_id"] == sid


def test_old_database_gets_texts_table(settings):
    path = settings.data_dir / "mediasearch.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    s = Store(path)
    s.close()
    db = sqlite3.connect(path)
    db.execute("DROP TABLE window_texts")  # 版 8 の DB(window_texts が無い)
    db.execute("UPDATE meta SET value='8' WHERE key='schema_version'")
    db.commit()
    db.close()
    s = Store(path)
    assert s.get_meta("schema_version") == str(SCHEMA_VERSION) == "9"
    assert s.get_window_texts([1, 2]) == {}
    s.close()
