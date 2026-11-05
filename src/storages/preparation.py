"""Atomic per-file preparation artifacts, independent of the usable vector index."""

import asyncio
import hashlib
import json
import logging
import os
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from threading import Lock
from types import TracebackType
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from documents.chunks import Chunk, Section
from storages.storage import StorageError, StoredChunk

logger = logging.getLogger(__name__)


def preparation_path(index: Path) -> Path:
    return index.with_name(index.name + ".prepared.sqlite3")


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    source: str
    title: str
    text: str
    page: int | None = Field(ge=1)
    line_citations: bool

    def section(self) -> Section:
        return Section(Path(self.source), self.title, self.text, self.page, self.line_citations)


class _Payload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    version: Literal[1] = 1
    count: int = Field(ge=0)
    items: list[dict[str, Any]]

    @model_validator(mode="after")
    def complete(self) -> Self:
        if self.count != len(self.items):
            raise ValueError("Incomplete preparation artifact")
        return self


class SQLitePreparationCache:
    """Commit complete file artifacts; a partial corpus never implies a complete file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = Lock()
        self._connection: sqlite3.Connection | None = None

    def __enter__(self) -> Self:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.is_symlink():
                raise StorageError(f"Refusing a symlink preparation cache: {self.path}")
            try:
                descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            except FileExistsError:
                pass
            else:
                os.close(descriptor)
            connection = sqlite3.connect(self.path, check_same_thread=False, autocommit=True)
            self._connection = connection
            if connection.execute("PRAGMA user_version").fetchone()[0] not in (0, 1):
                raise StorageError("Unsupported preparation cache version")
            connection.execute("PRAGMA journal_mode=DELETE")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS preparations "
                "(key TEXT PRIMARY KEY, payload TEXT NOT NULL, checksum TEXT NOT NULL)"
            )
            connection.execute("PRAGMA user_version=1")
        except (OSError, sqlite3.Error, StorageError) as exc:
            self.close()
            raise StorageError(f"Cannot open preparation cache {self.path}: {exc}") from exc
        return self

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _require_connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise StorageError("Preparation cache is closed")
        return self._connection

    @staticmethod
    def _key(kind: str, source: Path, fingerprint: str, policy: str) -> str:
        identity = json.dumps([kind, str(source.resolve()), fingerprint, policy])
        return hashlib.sha256(identity.encode()).hexdigest()

    def _read(self, key: str) -> _Payload | None:
        with self._lock:
            try:
                row = self._require_connection().execute(
                    "SELECT payload, checksum FROM preparations WHERE key=?", (key,)
                ).fetchone()
            except sqlite3.Error as exc:
                raise StorageError(f"Cannot read preparation cache {self.path}: {exc}") from exc
        if row is None:
            return None
        try:
            payload, checksum = row
            if hashlib.sha256(payload.encode()).hexdigest() != checksum:
                raise ValueError("Preparation checksum mismatch")
            return _Payload.model_validate_json(payload)
        except (AttributeError, TypeError, ValueError, ValidationError):
            logger.warning("Ignoring an invalid preparation artifact in %s", self.path)
            return None

    def _load_sections(self, source: Path, fingerprint: str, policy: str) -> list[Section] | None:
        payload = self._read(self._key("sections", source, fingerprint, policy))
        if payload is None:
            return None
        try:
            sections = [_Section.model_validate(item).section() for item in payload.items]
            if any(section.source != source for section in sections):
                raise ValueError("Prepared section source mismatch")
            return sections
        except (ValueError, ValidationError):
            logger.warning("Ignoring invalid prepared sections in %s", self.path)
            return None

    def _load_chunks(self, source: Path, fingerprint: str, policy: str) -> list[Chunk] | None:
        payload = self._read(self._key("chunks", source, fingerprint, policy))
        if payload is None:
            return None
        try:
            chunks = [StoredChunk.model_validate(item).chunk() for item in payload.items]
            if any(chunk.source != source for chunk in chunks):
                raise ValueError("Prepared chunk source mismatch")
            return chunks
        except (ValueError, ValidationError):
            logger.warning("Ignoring invalid prepared chunks in %s", self.path)
            return None

    def _save(
        self, kind: str, source: Path, fingerprint: str, policy: str,
        records: Sequence[Section] | Sequence[Chunk],
    ) -> None:
        items = [{**asdict(record), "source": str(source)} for record in records]
        payload = _Payload(count=len(items), items=items).model_dump_json()
        with self._lock:
            try:
                self._require_connection().execute(
                    "INSERT OR REPLACE INTO preparations VALUES (?, ?, ?)",
                    (self._key(kind, source, fingerprint, policy), payload,
                     hashlib.sha256(payload.encode()).hexdigest()),
                )
            except sqlite3.Error as exc:
                raise StorageError(f"Cannot write preparation cache {self.path}: {exc}") from exc

    async def _durable(self, operation: Callable[[], None]) -> None:
        task = asyncio.create_task(asyncio.to_thread(operation))
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            await task
            raise

    async def load_sections(
        self, source: Path, fingerprint: str, policy: str
    ) -> list[Section] | None:
        return await asyncio.to_thread(self._load_sections, source, fingerprint, policy)

    async def save_sections(
        self, source: Path, fingerprint: str, policy: str, sections: Sequence[Section]
    ) -> None:
        await self._durable(lambda: self._save("sections", source, fingerprint, policy, sections))

    async def load_chunks(
        self, source: Path, fingerprint: str, policy: str
    ) -> list[Chunk] | None:
        return await asyncio.to_thread(self._load_chunks, source, fingerprint, policy)

    async def save_chunks(
        self, source: Path, fingerprint: str, policy: str, chunks: Sequence[Chunk]
    ) -> None:
        await self._durable(lambda: self._save("chunks", source, fingerprint, policy, chunks))
