from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pypdf import PdfWriter

from documents.chunks import (
    Chunk,
    DocumentError,
    Section,
    _chunk_end,
    chunk_sections,
    discover,
    extract,
)

# ---------------------------------------------------------------------------
# Chunk & Section models
# ---------------------------------------------------------------------------


def test_chunk_citation_formats() -> None:
    source = Path("/docs/guide.pdf")
    pdf_chunk = Chunk(
        id="c1",
        source=source,
        title="guide",
        text="content",
        page=4,
    )
    assert pdf_chunk.citation == f"{source}, page, 4"

    line_chunk = Chunk(
        id="c2",
        source=Path("/docs/pep8.rst"),
        title="pep8",
        text="content",
        line_start=10,
        line_end=22,
    )
    assert line_chunk.citation == "/docs/pep8.rst, line, 10-22"

    html_chunk = Chunk(
        id="c3",
        source=Path("/docs/index.html"),
        title="Index",
        text="content",
    )
    assert html_chunk.citation == "/docs/index.html"


# ---------------------------------------------------------------------------
# discover
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ext", [".pdf", ".txt", ".md", ".rst", ".html", ".htm", ".MD", ".PDF"])
def test_discover_single_supported_file(tmp_path: Path, ext: str) -> None:
    file_path = tmp_path / f"sample{ext}"
    file_path.write_text("hello", encoding="utf-8")

    result = discover(file_path)
    assert result == [file_path.resolve()]


def test_discover_single_unsupported_file_raises(tmp_path: Path) -> None:
    bad_file = tmp_path / "data.json"
    bad_file.write_text("{}", encoding="utf-8")

    with pytest.raises(DocumentError, match="Unsupported document type: .json"):
        discover(bad_file)


def test_discover_nonexistent_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        discover(tmp_path / "missing.txt")


def test_discover_directory_non_recursive_filters_and_sorts(tmp_path: Path) -> None:
    f_b = tmp_path / "b_doc.txt"
    f_a = tmp_path / "A_doc.md"
    hidden = tmp_path / ".hidden.txt"
    unsupported = tmp_path / "notes.json"
    subdir = tmp_path / "nested"
    subdir.mkdir()
    nested_file = subdir / "inside.rst"

    for f in (f_b, f_a, hidden, unsupported, nested_file):
        f.write_text("content", encoding="utf-8")

    symlink_file = tmp_path / "link.txt"
    symlink_file.symlink_to(f_b)

    result = discover(tmp_path, recursive=False)
    assert result == [f_a.resolve(), f_b.resolve()]


def test_discover_directory_recursive_skips_hidden_pycache_and_symlinks(
    tmp_path: Path,
) -> None:
    root_doc = tmp_path / "root.txt"
    sub = tmp_path / "sub"
    sub.mkdir()
    sub_doc = sub / "guide.rst"

    hidden_dir = tmp_path / ".git"
    hidden_dir.mkdir()
    (hidden_dir / "ignored.txt").write_text("ignored", encoding="utf-8")

    pycache_dir = tmp_path / "__pycache__"
    pycache_dir.mkdir()
    (pycache_dir / "ignored.md").write_text("ignored", encoding="utf-8")

    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    outside_doc = outside_dir / "external.md"

    for f in (root_doc, sub_doc, outside_doc):
        f.write_text("text", encoding="utf-8")

    # Symlinked directory inside sub should not be traversed
    symlink_dir = sub / "linked_dir"
    symlink_dir.symlink_to(outside_dir)

    result = discover(tmp_path / "sub", recursive=True)
    assert result == [sub_doc.resolve()]

    all_result = discover(tmp_path, recursive=True)
    assert all_result == sorted(
        [root_doc.resolve(), sub_doc.resolve(), outside_doc.resolve()],
        key=lambda p: str(p).casefold(),
    )


def test_discover_empty_directory_raises(tmp_path: Path) -> None:
    (tmp_path / ".hidden.md").write_text("hidden", encoding="utf-8")
    (tmp_path / "ignored.json").write_text("{}", encoding="utf-8")

    with pytest.raises(DocumentError, match="No supported documents found"):
        discover(tmp_path, recursive=False)


# ---------------------------------------------------------------------------
# extract - text, markdown, rst, html, pdf
# ---------------------------------------------------------------------------


def test_extract_txt_and_utf8_bom(tmp_path: Path) -> None:
    txt_file = tmp_path / "notes.txt"
    txt_file.write_bytes(b"\xef\xbb\xbfLine one\nLine two")

    sections = extract(txt_file)
    assert len(sections) == 1
    assert sections[0] == Section(
        source=txt_file,
        title="notes",
        text="Line one\nLine two",
        page=None,
        line_citations=True,
    )


def test_extract_blank_text_file_returns_empty_list(tmp_path: Path) -> None:
    empty_file = tmp_path / "blank.md"
    empty_file.write_text("   \n\t  \n", encoding="utf-8")

    assert extract(empty_file) == []


def test_extract_non_utf8_file_raises_document_error(tmp_path: Path) -> None:
    bad_file = tmp_path / "latin1.txt"
    bad_file.write_bytes(b"\xff\xfe\xfa\xfb\x80\x81")

    with pytest.raises(DocumentError, match="Document must use UTF-8"):
        extract(bad_file)


def test_extract_rst_extracts_title_header(tmp_path: Path) -> None:
    rst_file = tmp_path / "pep-0008.rst"
    rst_file.write_text(
        "PEP: 8\nTitle: Style Guide for Python Code\nAuthor: Guido\n\nBody text here.",
        encoding="utf-8",
    )

    sections = extract(rst_file)
    assert len(sections) == 1
    assert sections[0].title == "pep-0008: Style Guide for Python Code"
    assert sections[0].line_citations is True
    assert sections[0].page is None


def test_extract_rst_without_title_header_uses_stem(tmp_path: Path) -> None:
    rst_file = tmp_path / "readme.rst"
    rst_file.write_text("Heading\n=======\nSome content.", encoding="utf-8")

    sections = extract(rst_file)
    assert len(sections) == 1
    assert sections[0].title == "readme"


def test_extract_html_strips_boilerplate_and_extracts_article(tmp_path: Path) -> None:
    html_file = tmp_path / "page.html"
    html_file.write_text(
        """
        <html>
          <head>
            <title>  Clean Architecture  </title>
            <style>body { color: red; }</style>
            <script>console.log("ignore");</script>
          </head>
          <body>
            <header>Site Header</header>
            <nav>Navigation Links</nav>
            <article>
              <h1>Main Heading</h1>
              <p>First paragraph.</p>
              <form><input type="text" />Ignore form</form>
            </article>
            <aside>Sidebar info</aside>
            <footer>Footer text</footer>
          </body>
        </html>
        """,
        encoding="utf-8",
    )

    sections = extract(html_file)
    assert len(sections) == 1
    section = sections[0]
    assert section.title == "Clean Architecture"
    assert section.page is None
    assert section.line_citations is False
    assert section.text == "Main Heading\nFirst paragraph."
    assert "Site Header" not in section.text
    assert "Navigation" not in section.text
    assert "Ignore form" not in section.text
    assert "Footer" not in section.text


def test_extract_pdf_pages_and_skips_blank_pages(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pdf_file = tmp_path / "book.pdf"
    pdf_file.write_bytes(b"%PDF-dummy")

    page1 = MagicMock()
    page1.extract_text.return_value = "Page 1 content"
    page2 = MagicMock()
    page2.extract_text.return_value = "   \n  "
    page3 = MagicMock()
    page3.extract_text.return_value = "Page 3 content"

    fake_reader = MagicMock()
    fake_reader.is_encrypted = False
    fake_reader.pages = [page1, page2, page3]

    monkeypatch.setattr("documents.chunks.PdfReader", lambda _path: fake_reader)

    sections = extract(pdf_file)
    assert len(sections) == 2
    assert sections[0] == Section(
        source=pdf_file,
        title="book",
        text="Page 1 content",
        page=1,
        line_citations=False,
    )
    assert sections[1] == Section(
        source=pdf_file,
        title="book",
        text="Page 3 content",
        page=3,
        line_citations=False,
    )


def test_extract_encrypted_pdf_without_empty_password_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pdf_file = tmp_path / "locked.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.encrypt("secret")
    with pdf_file.open("wb") as f:
        writer.write(f)

    # Also verify via mock in case cryptography isn't needed for AES
    fake_reader = MagicMock()
    fake_reader.is_encrypted = True
    fake_reader.decrypt.return_value = 0
    monkeypatch.setattr("documents.chunks.PdfReader", lambda _path: fake_reader)

    with pytest.raises(DocumentError, match="PDF requires a password"):
        extract(pdf_file)


def test_extract_corrupt_pdf_raises_document_error(tmp_path: Path) -> None:
    bad_pdf = tmp_path / "corrupt.pdf"
    bad_pdf.write_bytes(b"not a valid pdf file")

    with pytest.raises(DocumentError, match="Cannot read PDF"):
        extract(bad_pdf)


# ---------------------------------------------------------------------------
# _chunk_end & chunk_sections
# ---------------------------------------------------------------------------


def test_chunk_end_boundary_preferences() -> None:
    # Fits completely within size
    assert _chunk_end("short text", 0, 100) == 10

    # Prefers paragraph break (\n\n) in second half [50, 100]
    text_para = "a" * 60 + "\n\n" + "b" * 60
    assert _chunk_end(text_para, 0, 100) == 62

    # Falls back to single newline (\n) in second half
    text_line = "a" * 65 + "\n" + "b" * 60
    assert _chunk_end(text_line, 0, 100) == 66

    # Falls back to sentence boundary (". ", "! ", "? ")
    text_sent = "a" * 70 + "! " + "b" * 60
    assert _chunk_end(text_sent, 0, 100) == 72

    # Falls back to word space (" ")
    text_space = "a" * 75 + " " + "b" * 60
    assert _chunk_end(text_space, 0, 100) == 76

    # Falls back to hard cut at start + size when no separator in [50, 100]
    text_nosplit = "a" * 150
    assert _chunk_end(text_nosplit, 0, 100) == 100


@pytest.mark.parametrize(
    ("size", "overlap"),
    [
        (99, 0),
        (200, -1),
        (200, 100),  # overlap == size // 2
        (200, 150),  # overlap > size // 2
    ],
)
def test_chunk_sections_invalid_size_or_overlap_raises(size: int, overlap: int) -> None:
    with pytest.raises(
        ValueError,
        match="Chunk size must be >= 100 and overlap must be below the half the size",
    ):
        chunk_sections([], size=size, overlap=overlap)


def test_chunk_sections_tracks_line_numbers_and_overlap() -> None:
    lines = [f"Line {i:02d}: " + ("x" * 20) for i in range(1, 15)]
    text = "\n".join(lines)
    source = Path("/abs/doc.txt")
    section = Section(
        source=source, title="doc", text=text, page=None, line_citations=True
    )

    chunks = chunk_sections([section], size=120, overlap=30)

    assert len(chunks) >= 3
    assert chunks[0].source == source
    assert chunks[0].title == "doc"
    assert chunks[0].page is None
    assert chunks[0].line_start == 1
    assert chunks[0].line_end is not None and chunks[0].line_end > 1
    assert len(chunks[0].id) == 24

    # Subsequent chunks advance line numbers and end at the last line
    assert chunks[-1].line_end == 14
    assert all(
        c.line_start is not None
        and c.line_end is not None
        and c.line_start <= c.line_end
        for c in chunks
    )


def test_chunk_sections_pdf_omits_line_numbers_and_keeps_page() -> None:
    source = Path("/abs/manual.pdf")
    section = Section(
        source=source,
        title="manual",
        text="Paragraph one.\n\nParagraph two.",
        page=7,
        line_citations=False,
    )

    chunks = chunk_sections([section], size=200, overlap=20)
    assert len(chunks) == 1
    assert chunks[0].page == 7
    assert chunks[0].line_start is None
    assert chunks[0].line_end is None


def test_chunk_sections_uses_config_defaults_when_omitted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("ORACLE_CONFIG", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    section = Section(
        source=Path("/abs/short.txt"),
        title="short",
        text="Short text content.",
        page=None,
        line_citations=True,
    )
    chunks = chunk_sections([section])
    assert len(chunks) == 1
    assert chunks[0].text == "Short text content."
    assert chunks[0].line_start == 1
    assert chunks[0].line_end == 1
