import asyncio
import hashlib
import json
import os
import sqlite3
import subprocess
import sys

import pytest
from conftest import RecordingProgress
from test_embeddings import GatedEmbedding

import cli
import documents.chunks as documents
from documents.chunks import DocumentError, Section, chunk_files_async, load_chunks
from embeddings import EmbeddingError
from runtime.progress import StageAdvanced
from storages.preparation import SQLitePreparationCache, preparation_path
from storages.storage import StorageError


@pytest.mark.parametrize("extension", ["txt", "md", "rst", "html"])
def test_preparation_roundtrip_skips_extraction_and_chunking(
    settings, tmp_path, monkeypatch, extension,
):
    source = tmp_path / f"guide.{extension}"
    content = "Title: Document title\nFirst paragraph.\n\nSecond paragraph."
    if extension == "html":
        content = "<title>Document title</title><main><p>First paragraph.</p><p>Second.</p></main>"
    source.write_text(content, encoding="utf-8")
    expected = load_chunks(source, settings=settings)
    path = tmp_path / "prepared.sqlite3"
    with SQLitePreparationCache(path) as cache:
        assert load_chunks(source, settings=settings, preparation=cache) == expected

    def fail(*args, **kwargs):
        pytest.fail("unchanged documents must not be extracted or chunked again")

    monkeypatch.setattr(documents, "extract", fail)
    monkeypatch.setattr(documents, "chunk_sections", fail)
    progress = RecordingProgress()
    with SQLitePreparationCache(path) as cache:
        result = load_chunks(source, settings=settings, progress=progress, preparation=cache)
        assert result == expected
    events = [e for e in progress.events if isinstance(e, StageAdvanced)]
    assert next(e for e in events if e.stage == "extract").reused == 1
    assert [e for e in events if e.stage == "chunk"][-1].reused == len(expected)


def test_changed_bytes_invalidate_only_changed_file_even_with_same_size_and_mtime(
    settings, tmp_path, monkeypatch,
):
    first, second = tmp_path / "a.txt", tmp_path / "b.txt"
    first.write_text("Old text.", encoding="utf-8")
    second.write_text("Stable text.", encoding="utf-8")
    path = tmp_path / "cache.sqlite3"
    with SQLitePreparationCache(path) as cache:
        load_chunks(tmp_path, settings=settings, preparation=cache)
    previous = first.stat()
    first.write_text("New text.", encoding="utf-8")
    os.utime(first, ns=(previous.st_atime_ns, previous.st_mtime_ns))
    original = documents.extract
    calls = []

    def extract(file):
        calls.append(file)
        return original(file)

    monkeypatch.setattr(documents, "extract", extract)
    with SQLitePreparationCache(path) as cache:
        chunks = load_chunks(tmp_path, settings=settings, preparation=cache)
    assert calls == [first]
    assert [chunk.text for chunk in chunks] == ["New text.", "Stable text."]


def test_new_removed_files_and_order_follow_current_discovery(settings, tmp_path):
    first, second = tmp_path / "a.txt", tmp_path / "b.txt"
    first.write_text("First.", encoding="utf-8")
    second.write_text("Second.", encoding="utf-8")
    path = tmp_path / "cache.sqlite3"
    with SQLitePreparationCache(path) as cache:
        load_chunks(tmp_path, settings=settings, preparation=cache)
    first.unlink()
    third = tmp_path / "c.txt"
    third.write_text("Third.", encoding="utf-8")
    with SQLitePreparationCache(path) as cache:
        result = load_chunks(tmp_path, settings=settings, preparation=cache)
    assert [chunk.source for chunk in result] == [second, third]


def test_chunk_settings_change_reuses_extraction(settings, tmp_path, monkeypatch):
    source = tmp_path / "guide.txt"
    source.write_text("A long paragraph. " * 50, encoding="utf-8")
    path = tmp_path / "cache.sqlite3"
    with SQLitePreparationCache(path) as cache:
        old = load_chunks(source, settings=settings, preparation=cache)
    settings.chunking.size = 100
    settings.chunking.ovelap = 0
    expected = load_chunks(source, settings=settings)
    assert old != expected
    monkeypatch.setattr(documents, "extract", lambda *args: pytest.fail("must reuse extraction"))
    with SQLitePreparationCache(path) as cache:
        assert load_chunks(source, settings=settings, preparation=cache) == expected


def test_parser_policy_change_invalidates_preparation(settings, tmp_path, monkeypatch):
    source = tmp_path / "guide.txt"
    source.write_text("Original.", encoding="utf-8")
    path = tmp_path / "cache.sqlite3"
    with SQLitePreparationCache(path) as cache:
        load_chunks(source, settings=settings, preparation=cache)
    monkeypatch.setattr(documents, "preparation_policy", lambda: "new-parser-version")
    monkeypatch.setattr(
        documents, "extract", lambda file: [Section(file, "New title", "New parse.")]
    )
    with SQLitePreparationCache(path) as cache:
        result = load_chunks(source, settings=settings, preparation=cache)
    assert result[0].title == "New title"
    assert result[0].text == "New parse."


def test_partial_extraction_survives_failure_of_later_file(settings, tmp_path, monkeypatch):
    files = [tmp_path / "a.txt", tmp_path / "b.txt"]
    for file in files:
        file.write_text("A paragraph.", encoding="utf-8")
    original = documents.extract
    calls = []
    failing = True

    def extract(file):
        calls.append(file.name)
        if failing and file == files[1]:
            raise DocumentError("interrupted extraction")
        return original(file)

    monkeypatch.setattr(documents, "extract", extract)
    path = tmp_path / "cache.sqlite3"
    with SQLitePreparationCache(path) as cache:
        with pytest.raises(DocumentError, match="interrupted extraction"):
            asyncio.run(chunk_files_async(files, settings.chunking, workers=1, preparation=cache))
    failing = False
    with SQLitePreparationCache(path) as cache:
        result = asyncio.run(chunk_files_async(files, settings.chunking, preparation=cache))
    assert calls.count("a.txt") == 1
    assert calls.count("b.txt") == 2
    assert [chunk.source for chunk in result] == files


@pytest.mark.parametrize("kind", ["checksum", "incomplete", "schema"])
def test_invalid_extraction_artifact_is_recomputed(settings, tmp_path, monkeypatch, kind):
    source = tmp_path / "guide.txt"
    source.write_text("A paragraph.", encoding="utf-8")
    path = tmp_path / "cache.sqlite3"
    with SQLitePreparationCache(path) as cache:
        expected = load_chunks(source, settings=settings, preparation=cache)
    with sqlite3.connect(path) as connection:
        for key, raw in connection.execute("SELECT key, payload FROM preparations").fetchall():
            payload = json.loads(raw)
            if "line_citations" not in payload["items"][0]:
                continue
            if kind == "incomplete":
                payload["count"] += 1
            elif kind == "schema":
                payload["items"][0]["page"] = -1
            raw = json.dumps(payload)
            checksum = "invalid" if kind == "checksum" else hashlib.sha256(raw.encode()).hexdigest()
            connection.execute(
                "UPDATE preparations SET payload=?, checksum=? WHERE key=?", (raw, checksum, key)
            )
    original = documents.extract
    calls = []
    monkeypatch.setattr(documents, "extract", lambda file: (calls.append(file), original(file))[1])
    with SQLitePreparationCache(path) as cache:
        assert load_chunks(source, settings=settings, preparation=cache) == expected
    assert calls == [source]


def test_modified_during_extraction_is_not_checkpointed(settings, tmp_path, monkeypatch):
    source = tmp_path / "guide.txt"
    source.write_text("Before.", encoding="utf-8")

    def extract(file):
        file.write_text("Changed.", encoding="utf-8")
        return [Section(file, "title", "Mixed content.")]

    monkeypatch.setattr(documents, "extract", extract)
    path = tmp_path / "cache.sqlite3"
    with SQLitePreparationCache(path) as cache:
        with pytest.raises(DocumentError, match="changed during extraction"):
            load_chunks(source, settings=settings, preparation=cache)
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT count(*) FROM preparations").fetchone()[0] == 0


def test_empty_extraction_is_reused_but_does_not_become_a_complete_index(
    settings, tmp_path, monkeypatch,
):
    source = tmp_path / "blank.txt"
    source.write_text("   ", encoding="utf-8")
    path = tmp_path / "cache.sqlite3"
    with SQLitePreparationCache(path) as cache:
        with pytest.raises(DocumentError, match="No readable document text"):
            load_chunks(source, settings=settings, preparation=cache)
    monkeypatch.setattr(documents, "extract", lambda file: pytest.fail("empty artifact is valid"))
    with SQLitePreparationCache(path) as cache:
        with pytest.raises(DocumentError, match="No readable document text"):
            load_chunks(source, settings=settings, preparation=cache)


def test_prepared_citations_survive_abrupt_exit(tmp_path):
    source = tmp_path / "guide.pdf"
    path = tmp_path / "cache.sqlite3"
    code = """
import os, sys
from pathlib import Path
from documents.chunks import Section
from storages.preparation import SQLitePreparationCache
cache = SQLitePreparationCache(Path(sys.argv[1]))
cache.__enter__()
source = Path(sys.argv[2])
cache._save('sections', source, 'content-hash', 'parser',
    [Section(source, 'PDF title', 'Page text.', page=3, line_citations=False)])
os._exit(7)
"""
    result = subprocess.run([sys.executable, "-c", code, str(path), str(source)], check=False)
    assert result.returncode == 7
    with SQLitePreparationCache(path) as cache:
        assert asyncio.run(cache.load_sections(source, "content-hash", "parser")) == [
            Section(source, "PDF title", "Page text.", page=3, line_citations=False),
        ]


def test_invalid_database_fails_without_destroying_it(tmp_path):
    path = tmp_path / "cache.sqlite3"
    path.write_bytes(b"invalid SQLite data")
    with pytest.raises(StorageError, match="Cannot open preparation cache"):
        with SQLitePreparationCache(path):
            pass
    assert path.read_bytes() == b"invalid SQLite data"


def test_partial_chunking_preserves_complete_files(settings, tmp_path, monkeypatch):
    first, second = tmp_path / "a.txt", tmp_path / "b.txt"
    for file in (first, second):
        file.write_text("A paragraph.", encoding="utf-8")
    original = documents.chunk_sections
    calls = []
    failing = True

    def chunk(sections, *args):
        source = sections[0].source
        calls.append(source)
        if failing and source == second:
            raise DocumentError("interrupted chunking")
        return original(sections, *args)

    monkeypatch.setattr(documents, "chunk_sections", chunk)
    path = tmp_path / "cache.sqlite3"
    with SQLitePreparationCache(path) as cache:
        with pytest.raises(DocumentError, match="interrupted chunking"):
            load_chunks(tmp_path, settings=settings, preparation=cache)
    failing = False
    monkeypatch.setattr(documents, "extract", lambda file: pytest.fail("extraction was complete"))
    with SQLitePreparationCache(path) as cache:
        result = load_chunks(tmp_path, settings=settings, preparation=cache)
    assert calls.count(first) == 1
    assert calls.count(second) == 2
    assert [chunk.source for chunk in result] == [first, second]


def test_corrupt_chunks_reuse_extraction_and_recompute_only_chunks(settings, tmp_path, monkeypatch):
    source = tmp_path / "guide.txt"
    source.write_text("A paragraph.", encoding="utf-8")
    path = tmp_path / "cache.sqlite3"
    with SQLitePreparationCache(path) as cache:
        expected = load_chunks(source, settings=settings, preparation=cache)
    with sqlite3.connect(path) as connection:
        for key, raw in connection.execute("SELECT key, payload FROM preparations").fetchall():
            if '"line_citations"' not in raw:
                connection.execute("UPDATE preparations SET checksum='invalid' WHERE key=?", (key,))
    monkeypatch.setattr(documents, "extract", lambda file: pytest.fail("extraction remains valid"))
    with SQLitePreparationCache(path) as cache:
        assert load_chunks(source, settings=settings, preparation=cache) == expected


def test_cli_no_resume_bypasses_preparation(settings, embedder, tmp_path, monkeypatch):
    source = tmp_path / "guide.txt"
    source.write_text("A paragraph.", encoding="utf-8")
    monkeypatch.setattr(cli, "create_embedder", lambda **kwargs: embedder)
    cli.ingest_index(source, False, settings.index_path, None, None, settings)
    cache_path = preparation_path(settings.index_path)
    previous = cache_path.read_bytes()
    original = documents.extract
    calls = []
    monkeypatch.setattr(documents, "extract", lambda file: (calls.append(file), original(file))[1])
    cli.ingest_index(source, False, settings.index_path, None, None, settings, resume=False)
    assert calls == [source]
    assert cache_path.read_bytes() == previous


def test_cli_resume_skips_preparation_after_embedding_failure(
    settings, index, tmp_path, monkeypatch,
):
    index.save(settings.index_path)
    previous = settings.index_path.read_bytes()
    source = tmp_path / "guide.txt"
    source.write_text("A paragraph.", encoding="utf-8")
    first = GatedEmbedding(workers=1)
    first.failing = {"guide\nA paragraph."}
    monkeypatch.setattr(cli, "create_embedder", lambda **kwargs: first)
    with pytest.raises(EmbeddingError):
        cli.ingest_index(source, False, settings.index_path, None, None, settings)
    assert settings.index_path.read_bytes() == previous
    assert preparation_path(settings.index_path).exists()
    second = GatedEmbedding(workers=1)
    monkeypatch.setattr(cli, "create_embedder", lambda **kwargs: second)
    monkeypatch.setattr(documents, "extract", lambda *args: pytest.fail("must reuse extraction"))
    monkeypatch.setattr(documents, "chunk_sections", lambda *args: pytest.fail("must reuse chunks"))
    result = cli.ingest_index(source, False, settings.index_path, None, None, settings)
    assert result.stats.chunks == 1
    assert second.finished == ["guide\nA paragraph."]
    assert preparation_path(settings.index_path) in cli.clean_targets(settings, False)
