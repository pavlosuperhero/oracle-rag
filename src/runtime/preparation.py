"""Port for complete per-file extraction and chunking artifacts."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from documents.chunks import Chunk, Section


class PreparationCache(Protocol):
    async def load_sections(
        self, source: Path, fingerprint: str, policy: str
    ) -> list[Section] | None: ...

    async def save_sections(
        self, source: Path, fingerprint: str, policy: str, sections: Sequence[Section]
    ) -> None: ...

    async def load_chunks(
        self, source: Path, fingerprint: str, policy: str
    ) -> list[Chunk] | None: ...

    async def save_chunks(
        self, source: Path, fingerprint: str, policy: str, chunks: Sequence[Chunk]
    ) -> None: ...
