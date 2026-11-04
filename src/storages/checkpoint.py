"""Durable, content-addressed embedding checkpoints, separate from published indexes."""

import asyncio
import hashlib
import json
import logging
import os
import sqlite3
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from threading import Lock
from types import TracebackType
from typing import Self

import numpy as np

from embeddings.embeddings import EmbeddingError, normalize
from runtime.checkpoints import EmbeddingIdentity
from storages.storage import StorageError

logger = logging.getLogger(__name__)


def checkpoint_path(index: Path) -> Path:
    return index.with_name(index.name + ".embeddings.sqlite3")


class SQLiteCheckpoint:
    """Commit every vector before reporting progress; SQLite recovers interrupted writes.

    Only hashes and vectors are stored, never document text or credentials. One connection
    is shared by async worker threads under a lock; each write is its own transaction.
    """

    def __init__(self, path: Path, identity: EmbeddingIdentity) -> None:
        self.path = path
        self.dimensions = identity.dimensions
        identity_json = json.dumps({"version": 1, **asdict(identity)}, sort_keys=True)
        self._namespace = hashlib.sha256(identity_json.encode()).hexdigest()
        self._lock = Lock()
        self._connection: sqlite3.Connection | None = None

    def __enter__(self) -> Self:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.is_symlink():
                raise StorageError(f"Refusing a symlink checkpoint: {self.path}")
            try:
                descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            except FileExistsError:
                pass
            else:
                os.close(descriptor)
            connection = sqlite3.connect(self.path, check_same_thread=False, autocommit=True)
            self._connection = connection
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1):
                raise StorageError(f"Unsupported embedding checkpoint version: {version}")
            connection.execute("PRAGMA journal_mode=DELETE")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS embeddings ("
                "namespace TEXT NOT NULL, key TEXT NOT NULL, vector BLOB NOT NULL, "
                "digest TEXT NOT NULL, PRIMARY KEY (namespace, key))"
            )
            connection.execute("PRAGMA user_version=1")
        except (OSError, sqlite3.Error, StorageError) as exc:
            self.close()
            raise StorageError(f"Cannot open embedding checkpoint {self.path}: {exc}") from exc
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
            raise StorageError("Embedding checkpoint is closed.")
        return self._connection

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _load(self, texts: Sequence[str]) -> dict[str, list[float]]:
        found: dict[str, list[float]] = {}
        with self._lock:
            connection = self._require_connection()
            try:
                for text in texts:
                    row = connection.execute(
                        "SELECT vector, digest FROM embeddings WHERE namespace=? AND key=?",
                        (self._namespace, self._key(text)),
                    ).fetchone()
                    if row is None:
                        continue
                    try:
                        blob, digest = row
                        if (
                            not isinstance(blob, bytes)
                            or hashlib.sha256(blob).hexdigest() != digest
                        ):
                            raise ValueError("checksum mismatch")
                        vector = np.frombuffer(blob, dtype="<f4")
                        normalize(vector.reshape(1, -1), 1, self.dimensions)
                        found[text] = vector.tolist()
                    except (ValueError, EmbeddingError):
                        logger.warning("Ignoring an invalid cached embedding in %s", self.path)
                return found
            except sqlite3.Error as exc:
                raise StorageError(f"Cannot read embedding checkpoint {self.path}: {exc}") from exc

    async def load(self, texts: Sequence[str]) -> dict[str, list[float]]:
        return await asyncio.to_thread(self._load, texts)

    def _save(self, text: str, vector: list[float]) -> None:
        array = np.asarray([vector], dtype="<f4")
        normalize(array, 1, self.dimensions)
        blob = array.tobytes()
        with self._lock:
            connection = self._require_connection()
            try:
                connection.execute(
                    "INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?, ?)",
                    (self._namespace, self._key(text), blob, hashlib.sha256(blob).hexdigest()),
                )
            except sqlite3.Error as exc:
                raise StorageError(f"Cannot write embedding checkpoint {self.path}: {exc}") from exc

    async def save(self, text: str, vector: list[float]) -> None:
        # Cancellation must wait for the thread's transaction before the CLI closes SQLite.
        task = asyncio.create_task(asyncio.to_thread(self._save, text, vector))
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            await task
            raise
