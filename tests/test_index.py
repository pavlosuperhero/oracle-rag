from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from conftest import FakeEmbedder

from configs.configs import Settings
from data.index import IndexNotReady, MemoryIndex
from documents.chunks import Chunk, DocumentError
from models.models import Query
from storages.storage import StorageError


def test_uninitialized_index(settings: Settings, embedder: FakeEmbedder) -> None:
    index = MemoryIndex(embedder, settings)
    assert index.stats.chunks == 0
    for operation in (
        index.sources,
        lambda: index.fetch("apple"),
        lambda: index.search(Query(query="apple")),
        lambda: index.save(settings.index_path),
    ):
        with pytest.raises(IndexNotReady):
            operation()
    with pytest.raises(DocumentError):
        index.build([])


def test_fetch_reuses_index_without_embedding_or_reading_source(
    index: MemoryIndex, embedder: FakeEmbedder,
) -> None:
    before = len(embedder.calls)
    chunk = index.fetch("apple")
    assert not chunk.source.exists()
    assert chunk.text == "Apple orchard growing guide."
    with pytest.raises(KeyError):
        index.fetch("missing")
    assert len(embedder.calls) == before


def test_fetch_does_not_return_removed_evidence(
    index: MemoryIndex, chunks: list[Chunk],
) -> None:
    index.build(chunks[1:])
    with pytest.raises(KeyError):
        index.fetch("apple")
    assert index.fetch("ocean") == chunks[1]


def test_search_ranks_and_cites(index: MemoryIndex) -> None:
    result = index.search(Query(query="apple", top_k=1))
    assert [passage.id for passage in result.results] == ["apple"]
    assert result.results[0].source_uri.startswith("file://")
    assert result.results[0].line_start == 1
    assert result.results[0].score > 0
    assert index.stats.documents == 2
    catalog = index.sources()
    assert catalog.documents == 2
    assert catalog.sources[1].pages == [2]


def test_missing_source_does_not_embed(index: MemoryIndex, embedder: FakeEmbedder) -> None:
    before = len(embedder.calls)
    result = index.search(Query(query="apple", source="__no_such_document__"))
    assert result.results == []
    assert result.message == "No documents match the source filter."
    assert len(embedder.calls) == before


def test_case_insensitive_source_filter(index: MemoryIndex) -> None:
    result = index.search(Query(query="ocean", source="OCEAN.MD"))
    assert [passage.id for passage in result.results] == ["ocean"]


def test_duplicate_passages_are_returned_once(index: MemoryIndex, chunks: list[Chunk]) -> None:
    index.build([*chunks, replace(chunks[0], id="copy", text="Apple  orchard growing guide.")])
    result = index.search(Query(query="apple", top_k=20))
    assert len({" ".join(p.text.split()) for p in result.results}) == len(result.results)


def test_failed_rebuild_keeps_previous_snapshot(
    index: MemoryIndex, embedder: FakeEmbedder, chunks: list[Chunk]
) -> None:
    embedder.fail = True
    with pytest.raises(RuntimeError, match="embedding failed"):
        index.build(chunks[:1])
    assert index.stats.chunks == 2
    embedder.fail = False
    assert index.search(Query(query="apple")).results[0].id == "apple"


def test_save_restore_does_not_reembed_documents(
    index: MemoryIndex, settings: Settings, embedder: FakeEmbedder
) -> None:
    index.save(settings.index_path)
    before = len(embedder.calls)
    restored = MemoryIndex(embedder, settings)
    assert restored.restore(settings.index_path).chunks == 2
    assert len(embedder.calls) == before
    assert restored.search(Query(query="apple")) == index.search(Query(query="apple"))


def test_restore_all_deduplicates_matching_chunks(
    index: MemoryIndex, settings: Settings, embedder: FakeEmbedder
) -> None:
    index.save(settings.index_path)
    index.save(settings.index_path.parent / "copy.npz")
    restored = MemoryIndex(embedder, settings)
    assert restored.restore_all(settings.index_path.parent).chunks == 2


def test_restore_all_rejects_conflicts_without_replacing_snapshot(
    index: MemoryIndex, settings: Settings, chunks: list[Chunk]
) -> None:
    index.save(settings.index_path)
    index.build([replace(chunks[0], text="Different apple passage.")])
    index.save(settings.index_path.parent / "conflict.npz")
    with pytest.raises(StorageError, match="Conflicting chunk identity"):
        index.restore_all(settings.index_path.parent)
    assert index.stats.chunks == 1


def test_restore_all_missing_and_symlink(
    index: MemoryIndex, settings: Settings, tmp_path: Path
) -> None:
    with pytest.raises(StorageError, match="No database files"):
        index.restore_all(tmp_path / "missing")
    index.save(settings.index_path)
    (settings.index_path.parent / "link.npz").symlink_to(settings.index_path)
    with pytest.raises(StorageError, match="symlinks"):
        index.restore_all(settings.index_path.parent)


def test_snapshot_vectors_are_immutable(index: MemoryIndex) -> None:
    assert index._snapshot is not None
    assert not index._snapshot.vectors.flags.writeable
    np.testing.assert_allclose(np.linalg.norm(index._snapshot.vectors, axis=1), 1)
