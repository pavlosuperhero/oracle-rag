import hashlib
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from configs.configs import load_config

logger = logging.getLogger(__name__)

SUPPORTED_PREFIXES = frozenset({".pdf", ".txt", ".md", ".rst", ".html", ".htm"})


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
            soup = BeautifulSoup(text, "html.parser")
            if soup.title and soup.title.get_text(" ", strip=True):
                title = soup.title.get_text(" ", strip=True)
            for element in soup.select("script, style, nav, header, footer, aside, form"):
                element.decompose()
            content = soup.select_one("article, main, .post") or soup.body or soup
            text = "\n".join(
                line.strip() for line in content.get_text("\n").splitlines() if line.strip()
            )
        elif path.suffix.lower() == ".rst":
            match = re.search(r"^Title:\s*(.+)$", text, re.MULTILINE)
            if match:
                title = f"{path.stem}: {match.group(1).strip()}"
        sections = (
            [Section(path, title, text, page=None, line_citations=not html)]
            if text.strip()
            else []
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
            start = end - overlap
    return chunks
