import asyncio
import hashlib
import logging
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

from bs4 import BeautifulSoup, Comment
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from configs.configs import ChunkingConfig, Settings, load_config
from runtime.preparation import PreparationCache
from runtime.progress import IngestProgress, NullProgress, tracked_stage

logger = logging.getLogger(__name__)

SUPPORTED_PREFIXES = frozenset({".pdf", ".txt", ".md", ".rst", ".html", ".htm"})
DEFAULT_EXTRACT_WORKERS = 4

# Elements removed from the selected content root. Documentation generators embed
# feedback widgets, edit links, comment stubs and non-semantic footers inside
# <main>/<article>/<body>, so tag names alone are not enough.
HTML_NOISE_SELECTORS: tuple[str, ...] = (
    "script, style, noscript, template, svg, canvas, iframe",
    "nav, header, footer, aside, form, button, dialog, select, option",
    '[role="navigation"], [role="banner"], [role="contentinfo"]',
    '[role="complementary"], [role="search"], [role="dialog"]',
    '[hidden], [aria-hidden="true"], .d-print-none, .display-none-print',
    '[style*="display: none"], [style*="display:none"]',
    ".related, .sphinxsidebar",
    '[class*="footer"], [id*="footer"]',
    '[data-bi-name="page info"], [data-bi-name="feedback-section"]',
    '[class*="td-page-meta"]',
    '[class*="feedback"], [id*="feedback"]',
    '[class*="contribute"], [class*="edit-this-page"], [class*="edit-page"]',
    '.comments, #comments, [id*="disqus"], [class*="disqus"]',
    '#toc, .toc, [class*="breadcrumb"], [aria-label="breadcrumbs"]',
)

# Tried in priority order; the first selector that matches provides the content root.
HTML_CONTENT_SELECTORS: tuple[str, ...] = (
    "article",
    "main",
    '[role="main"]',
    ".post",
    ".document",
    ".markdown-body",
    "#content",
)

# Standalone UI lines that survive tag/class pruning (matched against a whole line).
BOILERPLATE_LINE = re.compile(
    r"(?:"
    r"was this page helpful\??|"
    r"did this page help you\??.*|"
    r"report a problem(?: with this (?:page|content))?|"
    r"view this page on github(?:\s*•\s*report a problem.*)?|"
    r"edit this page(?: on github)?|"
    r"improve this page|"
    r"help improve mdn|"
    r"learn how to contribute|"
    r"open a documentation issue|"
    r"provide product feedback|"
    r"suggest a fix\??|"
    r"found a bug\s*\??.*|"
    r"back to top|"
    r"copy (?:as )?markdown|"
    r"(?:please enable javascript to view the )?comments powered by disqus\.?"
    r")",
    re.IGNORECASE,
)


class DocumentError(ValueError):
    pass


@dataclass(frozen=True)
class Section:
    source: Path
    title: str
    text: str
    page: int | None = None
    line_citations: bool = True


@dataclass(frozen=True)
class Chunk:
    id: str
    source: Path
    title: str
    text: str
    page: int | None = None
    line_start: int | None = None
    line_end: int | None = None

    @property
    def citation(self) -> str:
        if self.page is not None:
            return f"{self.source}, page, {self.page}"
        if self.line_start is not None:
            return f"{self.source}, line, {self.line_start}-{self.line_end}"
        return str(self.source)


def discover(path: Path, recursive: bool = False) -> list[Path]:
    path = path.expanduser().resolve(strict=True)
    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_PREFIXES:
            raise DocumentError(f"Unsupported document type: {path.suffix}")
        return [path]
    files: list[Path] = []
    if recursive:
        for directory, subdirectories, names in os.walk(path, followlinks=False):
            subdirectories[:] = sorted(
                name
                for name in subdirectories
                if not name.startswith(".")
                and name != "__pycache__"
                and not (Path(directory) / name).is_symlink()
            )
            files.extend(Path(directory) / name for name in names if not name.startswith("."))
    else:
        files = list(path.iterdir())
    files = sorted(
        (
            file
            for file in files
            if file.is_file()
            and not file.is_symlink()
            and file.suffix.lower() in SUPPORTED_PREFIXES
            and not file.name.startswith(".")
        ),
        key=lambda file: str(file).casefold(),
    )
    if not files:
        raise DocumentError(f"No supported documents found in {path}; use -r for subfolders")
    return files


def extract_html(raw_html: str, default_title: str) -> tuple[str, str]:
    """Return ``(title, text)`` for an HTML document without page boilerplate.

    The content root is chosen first and noise is removed only from its descendants,
    so a broad selector such as ``[class*="footer"]`` can never discard the content
    itself. The title falls back to ``default_title`` when the page has none.
    """
    soup = BeautifulSoup(raw_html, "html.parser")
    title = default_title
    if soup.title and soup.title.get_text(" ", strip=True):
        title = soup.title.get_text(" ", strip=True)

    root = next(
        (match for selector in HTML_CONTENT_SELECTORS if (match := soup.select_one(selector))),
        soup.body or soup,
    )
    for comment in root.find_all(string=lambda node: isinstance(node, Comment)):
        comment.extract()
    for element in root.select(", ".join(HTML_NOISE_SELECTORS)):
        if not element.decomposed:
            element.decompose()

    lines = (line.strip() for line in root.get_text("\n").splitlines())
    text = "\n".join(line for line in lines if line and not BOILERPLATE_LINE.fullmatch(line))
    return title, text


def extract(path: Path) -> list[Section]:
    if path.suffix.lower() == ".pdf":
        try:
            reader = PdfReader(path)
            if reader.is_encrypted and not reader.decrypt(""):
                raise DocumentError(f"PDF requires a password: {path}")
            sections = [
                Section(path, path.stem, text, page=number, line_citations=False)
                for number, page in enumerate(reader.pages, 1)
                if (text := page.extract_text() or "").strip()
            ]
        except PdfReadError as e:
            raise DocumentError(f"Cannot read PDF {path}: {e}") from e
    else:
        try:
            text = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError as e:
            raise DocumentError(f"Document must use UTF-8: {path}") from e
        title = path.stem
        html = path.suffix.lower() in {".html", ".htm"}
        if html:
            title, text = extract_html(text, title)
        elif path.suffix.lower() == ".rst":
            match = re.search(r"^Title:\s*(.+)$", text, re.MULTILINE)
            if match:
                title = f"{path.stem}: {match.group(1).strip()}"
        sections = (
            [Section(path, title, text, page=None, line_citations=not html)] if text.strip() else []
        )
    return sections


def _chunk_end(text: str, start: int, size: int) -> int:
    end = min(start + size, len(text))
    if end == len(text):
        return end

    minimum = start + size // 2
    for separator in ("\n\n", "\n"):
        boundary = text.rfind(separator, minimum, end)
        if boundary != -1:
            return boundary + len(separator)

    boundary = max(
        text.rfind(". ", minimum, end),
        text.rfind("! ", minimum, end),
        text.rfind("? ", minimum, end),
    )
    if boundary != -1:
        return boundary + 2

    boundary = text.rfind(" ", minimum, end)
    return boundary + 1 if boundary != -1 else end


def chunk_sections(
    sections: list[Section], size: int | None = None, overlap: int | None = None
) -> list[Chunk]:
    if size is None or overlap is None:
        config = load_config().chunking
        size = config.size if size is None else size
        overlap = config.ovelap if overlap is None else overlap
    if size < 100 or not 0 <= overlap < size // 2:
        raise ValueError("Chunk size must be >= 100 and overlap must be below the half the size")
    chunks: list[Chunk] = []
    for section in sections:
        text = section.text
        start = 0
        while start < len(text):
            end = _chunk_end(text, start, size)
            raw = text[start:end]
            content = raw.strip()
            if content:
                content_start = start + len(raw) - len(raw.lstrip())
                content_end = end - (len(raw) - len(raw.rstrip()))
                line_start = (
                    text.count("\n", 0, content_start) + 1 if section.line_citations else None
                )
                line_end = text.count("\n", 0, content_end) + 1 if section.line_citations else None
                identity = f"{section.source}\0{section.page}\0{content_start}\0{content}"
                chunks.append(
                    Chunk(
                        hashlib.sha256(identity.encode()).hexdigest()[:24],
                        section.source,
                        section.title,
                        content,
                        section.page,
                        line_start,
                        line_end,
                    )
                )
            if end == len(text):
                break
            next_start = end - overlap
            # Expand overlap to include the whole word, while always advancing.
            while (
                next_start > start
                and not text[next_start - 1].isspace()
                and not text[next_start].isspace()
            ):
                next_start -= 1
            start = next_start if next_start > start else end
    return chunks


def file_fingerprint(path: Path) -> str:
    """Hash bytes, rather than trusting file size or modification timestamps."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def preparation_policy() -> str:
    """Invalidate preparation after parser code or extraction dependency changes."""
    policy = (
        file_fingerprint(Path(__file__)), version("pypdf"), version("beautifulsoup4"),
    )
    return hashlib.sha256(repr(policy).encode()).hexdigest()


async def chunk_files_async(
    files: Sequence[Path],
    config: ChunkingConfig,
    progress: IngestProgress | None = None,
    *,
    workers: int = DEFAULT_EXTRACT_WORKERS,
    preparation: PreparationCache | None = None,
) -> list[Chunk]:
    """Extract files in worker threads, then split them into passages.

    Sections are assembled in input order, so chunk IDs and order never depend on which
    file finishes first. Progress is reported from the event-loop thread as files complete.
    """
    progress = progress or NullProgress()
    if preparation is not None:
        files = [file.expanduser().resolve() for file in files]
    results: list[list[Section] | None] = [None] * len(files)
    fingerprints = [""] * len(files)
    policy = await asyncio.to_thread(preparation_policy) if preparation is not None else ""
    semaphore = asyncio.Semaphore(max(1, workers))

    with tracked_stage(progress, "extract", len(files)) as advance:
        completed = 0
        reused = 0

        async def read(index: int, file: Path) -> None:
            nonlocal completed, reused
            async with semaphore:
                if preparation is None:
                    results[index] = await asyncio.to_thread(extract, file)
                else:
                    fingerprint = await asyncio.to_thread(file_fingerprint, file)
                    fingerprints[index] = fingerprint
                    sections = await preparation.load_sections(file, fingerprint, policy)
                    if sections is None:
                        sections = await asyncio.to_thread(extract, file)
                        if await asyncio.to_thread(file_fingerprint, file) != fingerprint:
                            raise DocumentError(f"Document changed during extraction: {file}")
                        await preparation.save_sections(file, fingerprint, policy, sections)
                    else:
                        reused += 1
                    results[index] = sections
            completed += 1
            advance(completed, reused=reused)

        try:
            async with asyncio.TaskGroup() as group:
                for index, file in enumerate(files):
                    group.create_task(read(index, file))
        except BaseExceptionGroup as group_error:
            error = group_error.exceptions[0]
            raise error from error.__cause__
    if preparation is not None:
        logger.info("Reused extraction for %d/%d files", reused, len(files))
    sections = [section for result in results if result for section in result]
    with tracked_stage(progress, "chunk") as advance:
        if preparation is None:
            chunks = chunk_sections(sections, config.size, config.overlap)
            advance(len(chunks))
        else:
            chunks = []
            reused_chunks = 0
            chunk_policy = f"{policy}:{config.size}:{config.overlap}"
            for file, fingerprint, result in zip(files, fingerprints, results, strict=True):
                cached_chunks = await preparation.load_chunks(file, fingerprint, chunk_policy)
                if cached_chunks is None:
                    prepared: list[Chunk] = await asyncio.to_thread(
                        chunk_sections, result or [], config.size, config.overlap
                    )
                    await preparation.save_chunks(file, fingerprint, chunk_policy, prepared)
                else:
                    prepared = cached_chunks
                    reused_chunks += len(prepared)
                chunks.extend(prepared)
                advance(len(chunks), reused=reused_chunks)
            logger.info("Reused %d/%d prepared passages", reused_chunks, len(chunks))
    return chunks


def chunk_files(
    files: Sequence[Path], config: ChunkingConfig, progress: IngestProgress | None = None,
    *, preparation: PreparationCache | None = None,
) -> list[Chunk]:
    """Extract text from discovered files and split it into passages, reporting progress."""
    return asyncio.run(chunk_files_async(files, config, progress, preparation=preparation))


async def load_chunks_async(
    path: Path,
    recursive: bool = False,
    settings: Settings | None = None,
    progress: IngestProgress | None = None,
    *,
    workers: int = DEFAULT_EXTRACT_WORKERS,
    preparation: PreparationCache | None = None,
) -> list[Chunk]:
    config = (settings or load_config()).chunking
    with tracked_stage(progress or NullProgress(), "discover") as advance:
        files = discover(path, recursive)
        advance(len(files))
    chunks = await chunk_files_async(
        files, config, progress, workers=workers, preparation=preparation
    )
    if not chunks:
        raise DocumentError(f"No readable document text found in {path}")
    return chunks


def load_chunks(
    path: Path,
    recursive: bool = False,
    settings: Settings | None = None,
    progress: IngestProgress | None = None,
    *, preparation: PreparationCache | None = None,
) -> list[Chunk]:
    return asyncio.run(
        load_chunks_async(path, recursive, settings, progress, preparation=preparation)
    )
