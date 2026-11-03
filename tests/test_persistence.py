from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from configs.configs import Settings
from data.index import MemoryIndex
from documents.chunks import Chunk, DocumentError, load_chunks
from storages.storage import StorageError, StoredIndex, read_index, write_index


def test_write_index_round_trip(chunks: list[Chunk], tmp_path: Path) -> None:
    vectors = np.array([[3, 4], [0, 2]], dtype=np.float32)
    path = tmp_path / "nested" / "test.npz"
    assert write_index(path, StoredIndex("test", 2, tuple(chunks), vectors)) == path
    restored = read_index(path, "test", 2)
    assert restored.chunks == tuple(chunks)
    np.testing.assert_allclose(restored.vectors, [[0.6, 0.8], [0, 1]])


def test_failed_replace_preserves_database_and_cleans_temp(
    index: MemoryIndex, settings: Settings, monkeypatch
) -> None:
    index.save(settings.index_path)
    before = settings.index_path.read_bytes()

    def fail(*args):
        raise OSError("disk unavailable")

    monkeypatch.setattr("storages.storage.os.replace", fail)
    with pytest.raises(StorageError, match="disk unavailable"):
        index.save(settings.index_path)
    assert settings.index_path.read_bytes() == before
    assert list(settings.index_path.parent.iterdir()) == [settings.index_path]


def test_duplicate_ids_do_not_overwrite_database(
    index: MemoryIndex, settings: Settings, chunks: list[Chunk]
) -> None:
    index.save(settings.index_path)
    before = settings.index_path.read_bytes()
    duplicate = (chunks[0], replace(chunks[0], text="different"))
    with pytest.raises(StorageError, match="duplicate chunk IDs"):
        write_index(
            settings.index_path,
            StoredIndex("test", 2, duplicate, np.ones((2, 2), dtype=np.float32)),
        )
    assert settings.index_path.read_bytes() == before


def test_load_chunks_uses_supplied_settings(settings: Settings, tmp_path: Path) -> None:
    document = tmp_path / "book.md"
    document.write_text("apple " * 100)
    settings.chunking.size = 100
    settings.chunking.ovelap = 10
    chunks = load_chunks(document, settings=settings)
    assert len(chunks) > 1
    assert all(len(chunk.text) <= 100 for chunk in chunks)
    assert all(chunk.source == document for chunk in chunks)
    document.write_text("   ")
    with pytest.raises(DocumentError, match="No readable document text"):
        load_chunks(document, settings=settings)
