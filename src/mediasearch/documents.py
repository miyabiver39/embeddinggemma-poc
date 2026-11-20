"""文書(PDF・Word・テキスト)から文字を取り出し、検索用の塊(チャンク)に分けます。

文書は、映像の窓と同じく「塊ごとに 1 ベクトル」にします。EmbeddingGemma 2 は文章を、検索文(query)と
登録する文章(document)で別のプロンプトを付けてベクトルにするため、文書の塊は document として埋め込みます。

- 塊はページをまたがない(検索結果で、どのページかを示すため)。ページの分からない形式(テキスト)は 0 ページとする。
- 長い段落は、文の区切り(。.!? など)で分け、それでも長ければ文字数で切る。
- 前の塊の終わりを、次の塊の頭に少し重ねる(文の途中で切れた内容も、どちらかの塊で探せるように)。

読み込みは信頼できないファイルを想定し、大きさに上限を設けます(zip 爆弾・巨大な PDF で、メモリや時間を使い切らない)。
"""

from __future__ import annotations

import re
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

from .media import MediaError

TEXT_EXTS = frozenset({".txt", ".md", ".markdown"})
PDF_EXTS = frozenset({".pdf"})
DOCX_EXTS = frozenset({".docx"})
DOC_EXTS = TEXT_EXTS | PDF_EXTS | DOCX_EXTS

# 取り出す文字数の上限(超えた分は捨て、取り込みの結果に記録する)。日本語の文庫本 1 冊で約 10〜20 万字
MAX_CHARS = 2_000_000
MAX_PAGES = 5_000
MAX_TEXT_BYTES = 32 * 1024 * 1024  # テキストファイルとして読む大きさの上限
MAX_DOCX_XML_BYTES = 64 * 1024 * 1024  # docx の本文(word/document.xml)の展開後の大きさの上限

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_SENTENCE_END = re.compile(r"(?<=[。．！？!?])|(?<=\.)(?=\s)")
_SPACES = re.compile(r"[ \t　]+")


class DocumentError(MediaError):
    """文書として読めないファイル(API では 422)。"""


@dataclass(frozen=True)
class Page:
    number: int  # 1 始まり。ページの分からない形式は 0
    text: str


@dataclass(frozen=True)
class Chunk:
    page: int
    text: str


@dataclass
class Extracted:
    pages: list[Page]
    total_pages: int  # 文書のページ数(テキストは 0)
    truncated: bool  # 上限を超えたため、後ろを捨てた


def is_document(path: str | Path) -> bool:
    return Path(path).suffix.lower() in DOC_EXTS


# ---------------------------------------------------------------------------- 読み込み
def check(path: Path) -> None:
    """文書として読めるかを、中身を全部は読まずに確かめます(受け付けの時点で、明らかに壊れたものを断る)。"""
    ext = path.suffix.lower()
    try:
        if ext in PDF_EXTS:
            with path.open("rb") as f:
                head = f.read(1024)
            if b"%PDF-" not in head:
                raise DocumentError(f"PDF の形式ではありません: {path.name}")
        elif ext in DOCX_EXTS:
            if not zipfile.is_zipfile(path):
                raise DocumentError(f"Word(docx)の形式ではありません: {path.name}")
            with zipfile.ZipFile(path) as z:
                if "word/document.xml" not in z.namelist():
                    raise DocumentError(f"Word(docx)の本文が見つかりません: {path.name}")
        elif ext in TEXT_EXTS:
            if path.stat().st_size > MAX_TEXT_BYTES:
                limit = MAX_TEXT_BYTES // 1024 // 1024
                raise DocumentError(f"テキストファイルが大きすぎます(上限 {limit}MB): {path.name}")
            with path.open("rb") as f:
                _decode(f.read(64 * 1024), path.name, partial=True)
        else:
            raise DocumentError(f"文書として扱えない拡張子です: {path.name}(対応: {', '.join(sorted(DOC_EXTS))})")
    except OSError as exc:
        raise DocumentError(f"文書を読めません: {path.name}({exc})") from exc


def extract(path: Path, max_chars: int = MAX_CHARS) -> Extracted:
    """文書から、ページごとの文字を取り出します。"""
    ext = path.suffix.lower()
    if ext in PDF_EXTS:
        pages, total = _extract_pdf(path)
    elif ext in DOCX_EXTS:
        pages = _extract_docx(path)
        total = max((p.number for p in pages), default=0)
    elif ext in TEXT_EXTS:
        pages, total = [Page(0, _read_text(path))], 0
    else:
        raise DocumentError(f"文書として扱えない拡張子です: {path.name}")
    out, used, truncated = [], 0, False
    for p in pages:
        text = _normalize(p.text)
        if not text:
            continue
        if used + len(text) > max_chars:
            text = text[: max(max_chars - used, 0)]
            truncated = True
        if text:
            out.append(Page(p.number, text))
            used += len(text)
        if truncated:
            break
    return Extracted(out, total, truncated)


def _decode(data: bytes, name: str, partial: bool = False) -> str:
    """文字コードを判定して読みます。日本語の環境で多い Shift_JIS(cp932)にも対応します。"""
    if b"\x00" in data and not data.startswith((b"\xff\xfe", b"\xfe\xff")):
        raise DocumentError(f"テキストファイルではありません(バイナリの可能性): {name}")
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16", errors="replace")
    for enc in ("utf-8-sig", "cp932"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError as exc:
            # 先頭だけを読んだ場合、多バイト文字の途中で切れていることがある(末尾 3 バイト以内の失敗は許す)
            if partial and exc.start >= len(data) - 3:
                return data[: exc.start].decode(enc, errors="replace")
    return data.decode("utf-8", errors="replace")


def _read_text(path: Path) -> str:
    if path.stat().st_size > MAX_TEXT_BYTES:
        raise DocumentError(f"テキストファイルが大きすぎます(上限 {MAX_TEXT_BYTES // 1024 // 1024}MB): {path.name}")
    return _decode(path.read_bytes(), path.name)


def _extract_pdf(path: Path) -> tuple[list[Page], int]:
    try:
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError
    except ImportError as exc:  # pragma: no cover - 依存関係に含めている
        raise DocumentError("PDF を読むためのライブラリ(pypdf)がありません") from exc
    try:
        reader = PdfReader(path)
        if reader.is_encrypted:
            # 開くためのパスワードが無い(閲覧の制限だけの)PDF は、空のパスワードで開ける
            try:
                ok = reader.decrypt("")
            except Exception as exc:  # 暗号の方式に対応していない(cryptography が無い)など
                raise DocumentError(f"暗号化された PDF は読めません: {path.name}") from exc
            if not ok:
                raise DocumentError(f"パスワードで保護された PDF は読めません: {path.name}")
        total = len(reader.pages)
        pages = []
        for i, page in enumerate(reader.pages):
            if i >= MAX_PAGES:
                break
            try:
                text = page.extract_text() or ""
            except Exception:  # 1 ページの壊れた描画命令で、文書全体を失敗にしない
                text = ""
            pages.append(Page(i + 1, text))
    except DocumentError:
        raise
    except (PdfReadError, OSError, ValueError, KeyError) as exc:
        raise DocumentError(f"PDF を読めません: {path.name}({type(exc).__name__}: {exc})") from exc
    if not any(p.text.strip() for p in pages):
        raise DocumentError(
            f"PDF から文字を取り出せません: {path.name}"
            "(画像だけの PDF の可能性があります。文字認識(OCR)には対応していません)"
        )
    return pages, total


def _extract_docx(path: Path) -> list[Page]:
    """docx の本文を段落ごとに読みます。ページは、Word が保存時に記録した改ページの位置で数えます(目安)。"""
    try:
        with zipfile.ZipFile(path) as z:
            info = z.getinfo("word/document.xml")
            if info.file_size > MAX_DOCX_XML_BYTES:
                raise DocumentError(f"Word の本文が大きすぎます: {path.name}")
            with z.open(info) as f:
                return list(_docx_pages(f))
    except DocumentError:
        raise
    except (zipfile.BadZipFile, KeyError, OSError, ElementTree.ParseError) as exc:
        raise DocumentError(f"Word(docx)を読めません: {path.name}({type(exc).__name__}: {exc})") from exc


def _docx_pages(f) -> Iterator[Page]:
    page, paragraphs, buf = 1, [], []
    for event, el in ElementTree.iterparse(f, events=("start", "end")):
        tag = el.tag
        if event == "start":
            if tag == _W + "lastRenderedPageBreak" or (tag == _W + "br" and el.get(_W + "type") == "page"):
                if buf:  # 段落の途中の改ページ。ここまでを前のページに入れる
                    paragraphs.append("".join(buf))
                    buf = []
                if paragraphs:
                    yield Page(page, "\n".join(paragraphs))
                    paragraphs = []
                page += 1
            continue
        if tag == _W + "t":
            buf.append(el.text or "")
        elif tag == _W + "tab":
            buf.append("\t")
        elif tag == _W + "br":
            buf.append("\n")
        elif tag == _W + "p":
            paragraphs.append("".join(buf))
            buf = []
            el.clear()  # 読み終えた段落を捨てて、メモリを抑える
    if buf:
        paragraphs.append("".join(buf))
    if paragraphs:
        yield Page(page, "\n".join(paragraphs))


def _normalize(text: str) -> str:
    """行ごとに空白を詰め、3 行以上の空行を 1 つの空行にします(段落の区切りは残す)。"""
    lines = [_SPACES.sub(" ", line).strip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    out: list[str] = []
    for line in lines:
        if line or (out and out[-1]):
            out.append(line)
    return "\n".join(out).strip()


# ---------------------------------------------------------------------------- 分割
def chunk_pages(pages: list[Page], chunk_chars: int, overlap_chars: int) -> list[Chunk]:
    """ページごとに、chunk_chars 字ほどの塊に分けます。段落・文の区切りを優先して切ります。"""
    chunks: list[Chunk] = []
    for page in pages:
        pieces = _pieces(page.text, chunk_chars)
        cur = ""
        for piece in pieces:
            if cur and len(cur) + 1 + len(piece) > chunk_chars:
                chunks.append(Chunk(page.number, cur))
                tail = _tail(cur, overlap_chars)
                cur = f"{tail} {piece}" if tail else piece
            else:
                cur = f"{cur}\n{piece}" if cur else piece
        if cur.strip():
            chunks.append(Chunk(page.number, cur))
    return chunks


def _pieces(text: str, limit: int) -> list[str]:
    """段落に分け、長すぎる段落は文に、長すぎる文は limit 字ごとに分けます。"""
    out: list[str] = []
    for para in re.split(r"\n\s*\n|\n", text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= limit:
            out.append(para)
            continue
        for sent in _SENTENCE_END.split(para):
            sent = sent.strip()
            while len(sent) > limit:
                out.append(sent[:limit])
                sent = sent[limit:]
            if sent:
                out.append(sent)
    return out


def _tail(text: str, n: int) -> str:
    """末尾の n 字ほど。単語の途中から始めないよう、空白か句読点の後ろから取ります。"""
    if n <= 0 or len(text) <= n:
        return "" if n <= 0 else text
    tail = text[-n:]
    m = re.search(r"[\s。、．，.,!?！？]", tail)
    if m and m.end() < len(tail):
        tail = tail[m.end() :]
    return tail.strip()
