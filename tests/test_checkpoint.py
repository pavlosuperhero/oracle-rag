import asyncio
import hashlib
import os
import sqlite3
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest
from conftest import RecordingProgress
from test_embeddings import GatedEmbedding

import cli
from data.index import MemoryIndex
from documents.chunks import Chunk
from embeddings import EmbeddingError
from runtime.checkpoints import EmbeddingIdentity
from runtime.progress import StageAdvanced
from storages.checkpoint import SQLiteCheckpoint, checkpoint_path
from storages.storage import StorageError

IDENTITY = EmbeddingIdentity("test", "test-model", 256, "http://localhost:1234/v1/embeddings")
VECTOR = [1.0] + [0.0] * 255


def test_checkpoint_roundtrip_identity_and_exact_input_boundaries(tmp_path):
    path = tmp_path / "cache.sqlite3"
    with SQLiteCheckpoint(path, IDENTITY) as cache:
        asyncio.run(cache.save("Title\npassage", VECTOR))
    with SQLiteCheckpoint(path, IDENTITY) as cache:
        assert asyncio.run(cache.load(["Title\npassage", "Other title\npassage"])) == {
            "Title\npassage": VECTOR,
        }
    for identity in (
        replace(IDENTITY, provider="other"), replace(IDENTITY, model="other"),
        replace(IDENTITY, dimensions=512), replace(IDENTITY, endpoint="http://other/v1/embeddings"),
    ):
        with SQLiteCheckpoint(path, identity) as cache:
            assert asyncio.run(cache.load(["Title\npassage"])) == {}
    assert b"Title\npassage" not in path.read_bytes()
    assert b"localhost:1234" not in path.read_bytes()
    if os.name == "posix":
        assert path.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("kind", ["checksum", "shape", "nan", "zero"])
def test_invalid_cache_row_is_a_miss_not_a_published_vector(tmp_path, kind):
    path = tmp_path / "cache.sqlite3"
    with SQLiteCheckpoint(path, IDENTITY) as cache:
        asyncio.run(cache.save("a", VECTOR))
    array = np.asarray(VECTOR, dtype="<f4")
    if kind == "shape":
        array = array[:1]
    elif kind == "nan":
        array[0] = np.nan
    elif kind == "zero":
        array[:] = 0
    blob = array.tobytes()
    digest = "bad" if kind == "checksum" else hashlib.sha256(blob).hexdigest()
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE embeddings SET vector=?, digest=?", (blob, digest))
    with SQLiteCheckpoint(path, IDENTITY) as cache:
        assert asyncio.run(cache.load(["a"])) == {}
        asyncio.run(cache.save("a", VECTOR))
        assert asyncio.run(cache.load(["a"])) == {"a": VECTOR}


def test_unreadable_or_future_checkpoint_fails_without_replacing_it(tmp_path):
    path = tmp_path / "cache.sqlite3"
    path.write_bytes(b"not a database")
    with pytest.raises(StorageError, match="Cannot open embedding checkpoint"):
        with SQLiteCheckpoint(path, IDENTITY):
            pass
    assert path.read_bytes() == b"not a database"
    path.unlink()
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA user_version=99")
    with pytest.raises(StorageError, match="Unsupported embedding checkpoint version"):
        with SQLiteCheckpoint(path, IDENTITY):
            pass


def test_committed_vectors_survive_abrupt_process_exit(tmp_path):
    path = tmp_path / "cache.sqlite3"
    code = """
import os, sys
from runtime.checkpoints import EmbeddingIdentity
from storages.checkpoint import SQLiteCheckpoint
from pathlib import Path
cache = SQLiteCheckpoint(Path(sys.argv[1]), EmbeddingIdentity('test', 'test-model', 256,
    'http://localhost:1234/v1/embeddings'))
cache.__enter__()
cache._save('committed', [1.0] + [0.0] * 255)
os._exit(7)
"""
    result = subprocess.run([sys.executable, "-c", code, str(path)], check=False)
    assert result.returncode == 7
    with SQLiteCheckpoint(path, IDENTITY) as cache:
        assert asyncio.run(cache.load(["committed"])) == {"committed": VECTOR}


def test_sustained_ingestion_resumes_after_failure_beyond_1200_passages(tmp_path):
    texts = [f"a{number}" for number in range(1500)]
    first = GatedEmbedding(workers=2)
    first.failing = {"a1301"}
    path = tmp_path / "cache.sqlite3"
    with SQLiteCheckpoint(path, first.checkpoint_identity) as cache:
        first.checkpoint = cache
        with pytest.raises(EmbeddingError, match="boom a1301"):
            first.embed(texts)
    assert len(first.finished) >= 1200
    with SQLiteCheckpoint(path, first.checkpoint_identity) as cache:
        saved = asyncio.run(cache.load(texts))
    assert len(saved) >= 1200
    second = GatedEmbedding(workers=2)
    progress = RecordingProgress()
    with SQLiteCheckpoint(path, second.checkpoint_identity) as cache:
        second.checkpoint = cache
        vectors = second.embed([*texts, texts[0]], progress=progress)
    assert set(second.finished) == set(texts) - set(saved)
    assert set(second.finished).isdisjoint(saved)
    assert second.max_in_flight <= 2
    assert vectors.shape == (1501, 256)
    np.testing.assert_array_equal(vectors[0], vectors[-1])
    events = [event for event in progress.events if isinstance(event, StageAdvanced)]
    assert events[0].completed == events[0].reused == len(saved)
    assert events[-1].completed == 1500


def test_complete_checkpoint_needs_no_provider_session(tmp_path, monkeypatch):
    embedder = GatedEmbedding(workers=2)
    path = tmp_path / "cache.sqlite3"
    with SQLiteCheckpoint(path, embedder.checkpoint_identity) as cache:
        embedder.checkpoint = cache
        first = embedder.embed(["a", "b", "a"])
    other = GatedEmbedding(workers=2)
    monkeypatch.setattr(other, "_async_session", lambda: pytest.fail("must not open a provider"))
    with SQLiteCheckpoint(path, other.checkpoint_identity) as cache:
        other.checkpoint = cache
        np.testing.assert_array_equal(other.embed(["a", "b", "a"]), first)
    assert not other.finished


def test_cancellation_waits_for_a_pending_checkpoint_write(tmp_path, monkeypatch):
    import threading

    with SQLiteCheckpoint(tmp_path / "cache.sqlite3", IDENTITY) as cache:
        entered, release = threading.Event(), threading.Event()
        original = cache._save

        def save(text, vector):
            entered.set()
            assert release.wait(timeout=5)
            original(text, vector)

        monkeypatch.setattr(cache, "_save", save)

        async def run():
            task = asyncio.create_task(cache.save("a", VECTOR))
            assert await asyncio.to_thread(entered.wait, 5)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(run())
        assert asyncio.run(cache.load(["a"])) == {"a": VECTOR}


def test_cli_resume_preserves_published_index_and_no_resume_bypasses_cache(
    settings, index, monkeypatch, tmp_path,
):
    index.save(settings.index_path)
    previous = settings.index_path.read_bytes()
    chunks = [Chunk(str(n), tmp_path / "synthetic.md", "", text, line_start=n+1, line_end=n+1)
              for n, text in enumerate(["a", "b", "c"])]
    first = GatedEmbedding(workers=1)
    first.failing = {"\nb"}
    monkeypatch.setattr(cli, "create_embedder", lambda **kwargs: first)
    monkeypatch.setattr(cli, "managed_documents", lambda settings: [])
    monkeypatch.setattr(cli, "chunk_files", lambda *args: chunks)
    with pytest.raises(EmbeddingError):
        cli.ingest_index(None, False, settings.index_path, None, None, settings)
    assert settings.index_path.read_bytes() == previous
    assert first.checkpoint is None

    second = GatedEmbedding(workers=1)
    monkeypatch.setattr(cli, "create_embedder", lambda **kwargs: second)
    # GatedEmbedding uses the first character for a vector; leading newline is valid.
    loaded = cli.ingest_index(None, False, settings.index_path, None, None, settings)
    assert loaded.stats.chunks == 3
    assert second.finished == ["\nb", "\nc"]
    assert second.checkpoint is None
    restored = MemoryIndex(second, settings)
    assert restored.restore(settings.index_path).chunks == 3

    third = GatedEmbedding(workers=1)
    monkeypatch.setattr(cli, "create_embedder", lambda **kwargs: third)
    cli.ingest_index(None, False, settings.index_path, None, None, settings, resume=False)
    assert third.finished == ["\na", "\nb", "\nc"]
    assert checkpoint_path(settings.index_path) in cli.clean_targets(settings, False)

