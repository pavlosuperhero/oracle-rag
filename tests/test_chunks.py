import asyncio
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pypdf import PdfWriter

from documents.chunks import (
    Chunk,
    DocumentError,
    Section,
    _chunk_end,
    chunk_files,
    chunk_files_async,
    chunk_sections,
    discover,
    extract,
    load_chunks,
    load_chunks_async,
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


@pytest.mark.parametrize(
    ("body", "expected", "forbidden"),
    [
        pytest.param(
            """<main id="content"><h1>CSS Basics</h1><p>Selectors match elements.</p>
            <p>Escaped &lt;footer&gt; stays as text.</p>
            <div class="article-footer__links"><a class="article-footer__contribute">
            Help improve MDN</a><p>Learn how to contribute</p>
            <p>View this page on GitHub • Report a problem with this content</p></div></main>""",
            "CSS Basics\nSelectors match elements.\nEscaped <footer> stays as text.",
            ["Help improve", "contribute", "Report a problem"],
            id="mdn-article-footer",
        ),
        pytest.param(
            """<body class="awsdocs"><h1>Bedrock</h1><p>Embed text.</p>
            <div id="quick-feedback-yes" style="display: none;">Did this page help you? - Yes
            Thanks for letting us know</div></body>""",
            "Bedrock\nEmbed text.",
            ["Did this page help you", "Thanks"],
            id="aws-hidden-feedback",
        ),
        pytest.param(
            """<body class="article toc2 toc-left"><h1>TDD</h1><p>Write a test.</p>
            <div id="footer"><div id="footer-text">Last updated: 2025</div></div>
            <div class="comments">Comments<noscript>Please enable JavaScript</noscript></div>
            </body>""",
            "TDD\nWrite a test.",
            ["Last updated", "Comments", "JavaScript"],
            id="asciidoctor-footer-and-disqus",
        ),
        pytest.param(
            """<main role="main"><h1>OpenTelemetry</h1><p>Configure an exporter.</p>
            <div class="d-print-none"><h2 class="feedback--title">Feedback</h2>
            <p>Was this page helpful?</p><button>Yes</button><button>No</button></div>
            <div class="td-page-meta__lastmod">Last modified September 28, 2026</div></main>""",
            "OpenTelemetry\nConfigure an exporter.",
            ["helpful", "Feedback", "Last modified"],
            id="docsy-feedback",
        ),
        pytest.param(
            """<main id="main"><h1>PowerShell</h1><p>Run commands.</p>
            <div data-bi-name="feedback-section"><p>Provide product feedback</p></div>
            <p>Open a documentation issue</p><p>Suggest a fix?</p></main>""",
            "PowerShell\nRun commands.",
            ["feedback", "documentation issue", "Suggest a fix"],
            id="microsoft-learn",
        ),
        pytest.param(
            """<div class="document"><h1>subprocess</h1><p>Spawn processes.</p>
            <p>Edit this page on GitHub</p><p>Found a bug ? Created using Sphinx 7.2.6.</p>
            <div class="sphinxsidebar">Navigation index</div></div>""",
            "subprocess\nSpawn processes.",
            ["Edit this page", "Found a bug", "Navigation index"],
            id="sphinx",
        ),
    ],
)
def test_extract_html_strips_documentation_boilerplate(
    tmp_path: Path, body: str, expected: str, forbidden: list[str]
) -> None:
    html_file = tmp_path / "doc.html"
    html_file.write_text(
        f"<!doctype html><html><head><title>Doc</title></head>{body}</html>", encoding="utf-8"
    )
    sections = extract(html_file)
    assert len(sections) == 1
    assert sections[0].text == expected
    for phrase in forbidden:
        assert phrase not in sections[0].text


def test_extract_html_noise_selectors_never_remove_the_content_root(tmp_path: Path) -> None:
    html_file = tmp_path / "doc.html"
    html_file.write_text(
        '<html><body class="site-footer-layout"><main class="feedback-page"><p>Kept.</p></main>'
        "</body></html>",
        encoding="utf-8",
    )
    assert extract(html_file)[0].text == "Kept."


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


def test_overlap_preserves_whole_words_and_source_offsets() -> None:
    words = [f"word{number:03d}" for number in range(100)]
    text = " ".join(words)
    section = Section(Path("/abs/words.txt"), "Words", text)
    chunks = chunk_sections([section], size=120, overlap=23)
    assert len(chunks) > 1
    covered = set()
    previous_start = -1
    previous_end = 0
    for chunk in chunks:
        start = text.index(chunk.text)
        assert start > previous_start
        assert start <= previous_end
        assert len(chunk.text) <= 120
        assert all(word in words for word in chunk.text.split())
        assert chunk.line_start == chunk.line_end == 1
        covered.update(chunk.text.split())
        previous_start, previous_end = start, start + len(chunk.text) + 1
    assert covered == set(words)


def test_overlap_always_advances_for_an_overlong_word() -> None:
    text = "x" * 500
    chunks = chunk_sections([Section(Path("/abs/word.txt"), "Word", text)], 100, 20)
    assert len(chunks) == 5
    assert "".join(chunk.text for chunk in chunks) == text


# ---------------------------------------------------------------------------
# Concurrent extraction
# ---------------------------------------------------------------------------


def _write_documents(root: Path, count: int) -> list[Path]:
    files = []
    for number in range(count):
        file = root / f"doc{number}.txt"
        file.write_text(f"Document {number}. " + "word " * 80, encoding="utf-8")
        files.append(file)
    return files


def test_extraction_overlaps_across_worker_threads(
    tmp_path: Path, settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = _write_documents(tmp_path, 2)
    rendezvous = threading.Barrier(2, timeout=5)
    real_extract = extract

    def meeting(path: Path) -> list[Section]:
        rendezvous.wait()  # breaks unless both files are being extracted at the same time
        return real_extract(path)

    monkeypatch.setattr("documents.chunks.extract", meeting)
    chunks = asyncio.run(chunk_files_async(files, settings.chunking, workers=2))
    assert {chunk.source for chunk in chunks} == set(files)


def test_chunk_order_is_independent_of_completion_order(
    tmp_path: Path, settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = _write_documents(tmp_path, 3)
    expected = chunk_sections(
        [section for file in files for section in extract(file)],
        settings.chunking.size,
        settings.chunking.overlap,
    )
    last_done = threading.Event()
    real_extract = extract

    def last_first(path: Path) -> list[Section]:
        if path == files[0]:
            assert last_done.wait(timeout=5)
        sections = real_extract(path)
        if path == files[-1]:
            last_done.set()
        return sections

    monkeypatch.setattr("documents.chunks.extract", last_first)
    chunks = asyncio.run(chunk_files_async(files, settings.chunking, workers=3))
    assert last_done.is_set()
    assert chunks == expected
    assert chunk_files(files, settings.chunking) == expected


def test_extract_progress_counts_completions_on_caller_thread(
    tmp_path: Path, settings
) -> None:
    files = _write_documents(tmp_path, 4)
    started, advanced, finished = [], [], []
    threads = set()

    class Recorder:
        def started(self, event) -> None:
            threads.add(threading.get_ident())
            started.append(event)

        def advanced(self, event) -> None:
            threads.add(threading.get_ident())
            advanced.append(event)

        def finished(self, event) -> None:
            threads.add(threading.get_ident())
            finished.append(event)

    chunk_files(files, settings.chunking, Recorder())
    assert threads == {threading.get_ident()}
    extract_events = [event.completed for event in advanced if event.stage == "extract"]
    assert extract_events == [1, 2, 3, 4]
    assert [event.stage for event in started] == ["extract", "chunk"]
    assert [event.stage for event in finished] == ["extract", "chunk"]
    assert started[0].total == 4


def test_extraction_failure_raises_document_error(tmp_path: Path, settings) -> None:
    files = _write_documents(tmp_path, 2)
    bad = tmp_path / "bad.txt"
    bad.write_bytes(b"\xff\xfe\x00bad")
    with pytest.raises(DocumentError, match="UTF-8"):
        chunk_files([*files, bad], settings.chunking)


def test_load_chunks_async_matches_sync_facade(tmp_path: Path, settings) -> None:
    _write_documents(tmp_path, 3)
    assert asyncio.run(load_chunks_async(tmp_path, settings=settings)) == load_chunks(
        tmp_path, settings=settings
    )
