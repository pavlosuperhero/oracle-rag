"""Tests for provenance contracts, v1/v2 snapshot storage, and bounded retrieval traces."""

from __future__ import annotations

import ast
import json
import logging
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic import ValidationError

from configs.configs import Settings, load_config
from data.index import IndexNotReady, MemoryIndex
from documents.chunks import Chunk, chunk_sections, discover, extract, file_fingerprint
from models.models import Query
from models.provenance import (
    PROVENANCE_SCHEMA_VERSION,
    TRACE_SCHEMA_VERSION,
    EmbeddingProfile,
    IndexProvenance,
    ParserPolicy,
    PortableSnapshotManifest,
    PortableSourceEntry,
    RetrievalProfile,
    RetrievalTrace,
    SourceProvenance,
    canonical_json,
    compute_index_generation_id,
    fingerprint_corpus,
    fingerprint_embedding_profile,
    fingerprint_query,
    fingerprint_retrieval_profile,
    fingerprint_vectors,
    sanitize_endpoint,
)
from runtime.progress import IngestProgress
from storages.storage import (
    StorageError,
    StoredIndex,
    export_portable_manifest,
    read_index,
    verify_portable_manifest,
    write_index,
)

# ---------------------------------------------------------------------------
# Module-level fakes & helpers (public APIs only; zero private src imports)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SamplePayload:
    beta: int
    alpha: str


class DeterministicTopicEmbedder:
    """Deterministic multi-topic embedder with configurable provider metadata."""

    def __init__(
        self,
        *,
        provider_name: str = "OpenAI",
        model: str = "deterministic-embed-v1",
        dimensions: int = 8,
        checkpoint_endpoint: str = "http://127.0.0.1:1234/v1/embeddings",
        revision: str | None = None,
    ) -> None:
        self.provider_name = provider_name
        self.model = model
        self.dimensions = dimensions
        self.checkpoint_endpoint = checkpoint_endpoint
        self.revision = revision
        self.calls: list[list[str]] = []

    def embed(
        self, texts: Sequence[str], *, progress: IngestProgress | None = None
    ) -> NDArray[np.float32]:
        self.calls.append(list(texts))
        vectors = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        keywords = (
            "alpha",
            "beta",
            "gamma",
            "delta",
            "orchard",
            "marine",
            "provenance",
            "trace",
        )
        for row, text in enumerate(texts):
            lowered = text.casefold()
            for col in range(self.dimensions):
                token = keywords[col % len(keywords)]
                vectors[row, col] = float(lowered.count(token))
            if not np.any(vectors[row]):
                vectors[row, 0] = 0.25
                vectors[row, self.dimensions - 1] = 0.75
        return vectors


class FakeBedrockPublicRegionEmbedder:
    """Fake Bedrock embedder exposing only public ``region`` (no private ``_region``)."""

    provider_name = "Bedrock"
    model = "amazon.titan-embed-text-v2:0"
    dimensions = 8

    def __init__(self, region: str = "us-west-2") -> None:
        self.region = region

    def embed(
        self, texts: Sequence[str], *, progress: IngestProgress | None = None
    ) -> NDArray[np.float32]:
        vectors = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        vectors[:, 0] = 1.0
        return vectors


class RecordingTraceSink:
    """Injected TraceSink port fake recording bounded retrieval traces."""

    def __init__(self) -> None:
        self.traces: list[RetrievalTrace] = []

    def record(self, trace: RetrievalTrace) -> None:
        self.traces.append(trace)


def make_settings(
    tmp_path: Path,
    *,
    dimensions: int = 256,
    candidates: int = 20,
    candidate_multiplier: int = 4,
    rrf_constant: int = 60,
    top_k: int = 5,
    workers: int = 4,
    connection_timeout: int = 5,
    read_timeout: int = 30,
    server_port: int = 8000,
    aws_profile: str | None = None,
) -> Settings:
    profile_line = f"  profile: {aws_profile}\n" if aws_profile is not None else ""
    config_file = tmp_path / f"config-{server_port}-{workers}-{rrf_constant}-{top_k}.yaml"
    config_file.write_text(
        "provider:\n"
        f"  dimensions: {dimensions}\n"
        f"  worker: {workers}\n"
        f"  connection_timeout: {connection_timeout}\n"
        f"  read_timeout: {read_timeout}\n"
        f"{profile_line}"
        "retrieval:\n"
        f"  top_k: {top_k}\n"
        f"  candidates: {candidates}\n"
        f"  candidate_multiplier: {candidate_multiplier}\n"
        f"  rrf_constant: {rrf_constant}\n"
        "server:\n"
        f"  port: {server_port}\n"
        "storage:\n"
        "  index: db/index.npz\n",
        encoding="utf-8",
    )
    return load_config(config_file)


def write_legacy_v1_archive(
    path: Path,
    *,
    model: str,
    dimensions: int,
    chunks: Sequence[Chunk],
    vectors: NDArray[np.float32],
) -> Path:
    manifest = {
        "version": 1,
        "model": model,
        "dimensions": dimensions,
        "chunks": [
            {
                "id": chunk.id,
                "source": str(chunk.source.resolve()),
                "title": chunk.title,
                "text": chunk.text,
                "page": chunk.page,
                "line_start": chunk.line_start,
                "line_end": chunk.line_end,
            }
            for chunk in chunks
        ],
    }
    metadata = np.frombuffer(json.dumps(manifest).encode("utf-8"), dtype=np.uint8)
    np.savez(path, metadata=metadata, vectors=vectors)
    return path


# ---------------------------------------------------------------------------
# 1. Core architecture & canonical fingerprinting
# ---------------------------------------------------------------------------


def test_core_provenance_modules_do_not_import_presentation_libraries() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    for rel_path in (
        "src/models/provenance.py",
        "src/storages/storage.py",
        "src/data/index.py",
    ):
        tree = ast.parse((repo_root / rel_path).read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert not (imported & {"rich", "typer"}), f"{rel_path} imported presentation library"


def test_canonical_json_is_deterministic_across_orderings_and_types() -> None:
    payload_a = {
        "z": [1, 2],
        "a": {"d": 4, "c": 3},
        "item": SamplePayload(beta=7, alpha="ok"),
        "tags": {"b", "a"},
        "utf8": "café",
    }
    payload_b = {
        "utf8": "café",
        "tags": {"a", "b"},
        "item": {"alpha": "ok", "beta": 7},
        "a": {"c": 3, "d": 4},
        "z": (1, 2),
    }
    assert canonical_json(payload_a) == canonical_json(payload_b)
    with pytest.raises(ValueError):
        canonical_json({"bad": float("nan")})


def test_embedding_and_retrieval_fingerprints_react_only_to_meaningful_changes(
    tmp_path: Path,
) -> None:
    base_profile = EmbeddingProfile(
        provider="OpenAI",
        model="text-embedding-bge-large-en-v1.5",
        dimensions=1024,
        endpoint="http://127.0.0.1:1234/v1/embeddings",
        revision="rev-1",
    )
    base_id = fingerprint_embedding_profile(base_profile)
    assert base_id.startswith("sha256:")
    assert base_id == fingerprint_embedding_profile(base_profile.model_copy())

    for mutated in (
        base_profile.model_copy(update={"provider": "Bedrock"}),
        base_profile.model_copy(update={"model": "amazon.titan-embed-text-v2:0"}),
        base_profile.model_copy(update={"dimensions": 512}),
        base_profile.model_copy(update={"endpoint": "http://127.0.0.1:9999/v1/embeddings"}),
        base_profile.model_copy(update={"revision": "rev-2"}),
        base_profile.model_copy(update={"input_role": "text_only"}),
    ):
        assert fingerprint_embedding_profile(mutated) != base_id

    base_retrieval = RetrievalProfile(
        tokenizer="regex_word_v1",
        candidates=20,
        candidate_multiplier=4,
        rrf_constant=60,
        top_k=5,
    )
    base_ret_id = fingerprint_retrieval_profile(base_retrieval)
    for mutated_ret in (
        base_retrieval.model_copy(update={"tokenizer": "whitespace_v1"}),
        base_retrieval.model_copy(update={"candidates": 30}),
        base_retrieval.model_copy(update={"candidate_multiplier": 6}),
        base_retrieval.model_copy(update={"rrf_constant": 40}),
        base_retrieval.model_copy(update={"top_k": 10}),
        base_retrieval.model_copy(update={"tie_break": "score_desc_id_asc"}),
        base_retrieval.model_copy(update={"dedupe": "none"}),
    ):
        assert fingerprint_retrieval_profile(mutated_ret) != base_ret_id

    # Irrelevant runtime settings (workers, timeouts, server port, AWS profile) must not
    # alter embedding_profile_id, retrieval_profile_id, or index_generation_id.
    doc = tmp_path / "note.md"
    doc.write_text("Alpha provenance check document.", encoding="utf-8")
    chunks = chunk_sections(extract(doc), size=200, overlap=20)

    settings_a = make_settings(
        tmp_path, workers=2, connection_timeout=3, read_timeout=15, server_port=8101,
        aws_profile="dev-profile-a",
    )
    settings_b = make_settings(
        tmp_path, workers=8, connection_timeout=20, read_timeout=90, server_port=9202,
        aws_profile="prod-profile-b",
    )
    embedder_a = DeterministicTopicEmbedder(
        checkpoint_endpoint="https://user:secretA@embed.local/v1/embeddings?api_key=aaa"
    )
    embedder_b = DeterministicTopicEmbedder(
        checkpoint_endpoint="https://other:secretB@embed.local/v1/embeddings?api_key=bbb"
    )

    index_a = MemoryIndex(embedder_a, settings_a)
    index_b = MemoryIndex(embedder_b, settings_b)
    index_a.build(chunks)
    index_b.build(chunks)

    assert index_a.provenance is not None and index_b.provenance is not None
    assert index_a.provenance.embedding_profile.endpoint == "https://embed.local/v1/embeddings"
    assert index_a.provenance.embedding_profile_id == index_b.provenance.embedding_profile_id
    assert index_a.index_generation_id == index_b.index_generation_id
    assert index_a.retrieval_profile_id == index_b.retrieval_profile_id


def test_embedding_profile_rejects_secrets_and_sanitize_endpoint_strips_them() -> None:
    fake_api_key = "sk-" + "testSecretKey123456"
    fake_bearer = "Bearer " + fake_api_key
    fake_aws_key = "AKIA" + "0" * 16
    fake_aws_temp_key = "ASIA" + "0" * 16

    assert (
        sanitize_endpoint("https://alice:topsecret@api.example.com:8443/v1/embeddings/?token=abc#x")
        == "https://api.example.com:8443/v1/embeddings"
    )
    for unsafe in (
        "https://user:pass@api.example.com/v1",
        "https://api.example.com/v1?api_key=secret",
        fake_bearer,
        "Basic dXNlcjpwYXNz",
        fake_api_key,
        fake_aws_key,
        fake_aws_temp_key,
    ):
        with pytest.raises(ValidationError):
            EmbeddingProfile(
                provider="OpenAI",
                model="test-model",
                dimensions=8,
                endpoint=unsafe,
            )
    for unsafe_non_url in (
        "Bearer tok123",
        "Basic dXNlcjpwYXNz",
        fake_api_key,
        fake_aws_key,
        fake_aws_temp_key,
    ):
        with pytest.raises(ValueError, match="must not contain credentials"):
            sanitize_endpoint(unsafe_non_url)


def test_index_provenance_status_and_corpus_sha256_invariants() -> None:
    profile = EmbeddingProfile(provider="OpenAI", model="m", dimensions=4)
    profile_id = fingerprint_embedding_profile(profile)
    with pytest.raises(ValidationError, match="Complete provenance requires corpus_sha256"):
        IndexProvenance(
            provenance_status="complete",
            index_generation_id="sha256:" + "1" * 64,
            corpus_sha256=None,
            embedding_profile=profile,
            embedding_profile_id=profile_id,
        )
    with pytest.raises(ValidationError, match="legacy_unknown provenance must not set"):
        IndexProvenance(
            provenance_status="legacy_unknown",
            index_generation_id="sha256:" + "1" * 64,
            corpus_sha256="sha256:" + "2" * 64,
            embedding_profile=profile,
            embedding_profile_id=profile_id,
        )


def test_bedrock_embedder_reads_public_region_property(tmp_path: Path) -> None:
    doc = tmp_path / "region.md"
    doc.write_text("Bedrock region provenance check.", encoding="utf-8")
    chunks = chunk_sections(extract(doc), size=200, overlap=20)
    index = MemoryIndex(FakeBedrockPublicRegionEmbedder("us-west-2"), make_settings(tmp_path))
    index.build(chunks)
    assert index.provenance is not None
    assert index.provenance.embedding_profile.provider == "Bedrock"
    assert index.provenance.embedding_profile.endpoint == "bedrock://us-west-2"


def test_corpus_and_index_generation_fingerprints() -> None:
    s1 = SourceProvenance(
        uri="https://example.com/a.md",
        content_sha256="a" * 64,
        revision="v1",
        retrieved_at="2026-01-01T00:00:00Z",
        source_name="a.md",
    )
    s2 = SourceProvenance(
        uri="https://example.com/b.md",
        content_sha256="b" * 64,
        revision="v1",
        retrieved_at="2026-01-01T00:00:00Z",
        source_name="b.md",
    )
    policy = ParserPolicy(chunk_size=1000, chunk_overlap=100, preparation_policy_sha256="p" * 64)

    fp_forward = fingerprint_corpus([s1, s2], policy)
    fp_reversed = fingerprint_corpus([s2, s1], policy)
    fp_new_timestamp = fingerprint_corpus(
        [s1.model_copy(update={"retrieved_at": "2026-12-31T23:59:59Z"}), s2], policy
    )
    assert fp_forward == fp_reversed == fp_new_timestamp

    assert fingerprint_corpus([s1.model_copy(update={"content_sha256": "c" * 64}), s2], policy) != (
        fp_forward
    )
    assert fingerprint_corpus([s1, s2], policy.model_copy(update={"chunk_size": 500})) != fp_forward

    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    vec_fp = fingerprint_vectors(vectors)
    emb_fp = fingerprint_embedding_profile(
        EmbeddingProfile(provider="OpenAI", model="m", dimensions=2)
    )
    gen_a = compute_index_generation_id([("c1", "one"), ("c2", "two")], emb_fp, fp_forward, vec_fp)
    gen_reordered = compute_index_generation_id(
        [("c2", "two"), ("c1", "one")], emb_fp, fp_forward, vec_fp
    )
    gen_text_changed = compute_index_generation_id(
        [("c1", "ONE"), ("c2", "two")], emb_fp, fp_forward, vec_fp
    )
    assert gen_a != gen_reordered
    assert gen_a != gen_text_changed


# ---------------------------------------------------------------------------
# 2. Legacy v1 compatibility, v2 round-trip, rollback, and integrity checks
# ---------------------------------------------------------------------------


def test_legacy_v1_archive_loads_with_honest_legacy_unknown_provenance(tmp_path: Path) -> None:
    doc = tmp_path / "legacy.md"
    doc.write_text("Legacy passage text.", encoding="utf-8")
    chunks = [Chunk("legacy-1", doc, "Legacy", "Legacy passage text.", line_start=1, line_end=1)]
    vectors = np.array([[3.0, 4.0, 0.0, 0.0]], dtype=np.float32)
    archive_path = write_legacy_v1_archive(
        tmp_path / "legacy.npz",
        model="legacy-model",
        dimensions=4,
        chunks=chunks,
        vectors=vectors,
    )

    stored = read_index(archive_path, "legacy-model", 4)
    assert stored.provenance is not None
    prov = stored.provenance
    assert prov.schema_version == PROVENANCE_SCHEMA_VERSION
    assert prov.provenance_status == "legacy_unknown"
    assert prov.corpus_sha256 is None
    assert prov.parser_policy is None
    assert prov.embedding_profile.provider == "unknown"
    assert prov.embedding_profile.model == "legacy-model"
    assert prov.embedding_profile.dimensions == 4
    assert len(prov.sources) == 1
    assert prov.sources[0].content_sha256 is None
    assert prov.sources[0].revision is None
    assert prov.sources[0].retrieved_at is None
    assert prov.index_generation_id.startswith("sha256:")

    # Restoring legacy v1 into MemoryIndex preserves legacy_unknown status.
    embedder = DeterministicTopicEmbedder(model="legacy-model", dimensions=4)
    index = MemoryIndex(embedder, make_settings(tmp_path, dimensions=256))
    index.restore(archive_path)
    assert index.provenance == prov
    assert index.index_generation_id == prov.index_generation_id


def test_v2_round_trip_and_v1_format_rollback(tmp_path: Path) -> None:
    doc = tmp_path / "guide.md"
    doc.write_text("Alpha provenance verification guide.\nSecond line here.", encoding="utf-8")
    chunks = chunk_sections(extract(doc), size=200, overlap=20)
    embedder = DeterministicTopicEmbedder(dimensions=8, revision="2026-01")
    settings = make_settings(tmp_path)
    index = MemoryIndex(embedder, settings)
    index.build(
        chunks,
        source_provenance=[
            SourceProvenance(
                uri=doc.resolve().as_uri(),
                content_sha256=file_fingerprint(doc),
                revision="rev-42",
                retrieved_at="2026-05-01T12:00:00Z",
                source_name="guide.md",
            )
        ],
        parser_policy=ParserPolicy(chunk_size=200, chunk_overlap=20),
    )

    v2_path = tmp_path / "snapshot-v2.npz"
    index.save(v2_path)
    with np.load(v2_path, allow_pickle=False) as archive:
        raw_v2 = json.loads(archive["metadata"].tobytes())
    assert raw_v2["version"] == 2
    assert raw_v2["provenance"]["provenance_status"] == "complete"

    restored = MemoryIndex(embedder, settings)
    restored.restore(v2_path)
    assert restored.provenance == index.provenance
    assert restored.index_generation_id == index.index_generation_id

    # Reversible rollback to format_version=1 omits the provenance key entirely.
    v1_path = tmp_path / "snapshot-v1.npz"
    index.save(v1_path, format_version=1)
    with np.load(v1_path, allow_pickle=False) as archive:
        raw_v1 = json.loads(archive["metadata"].tobytes())
    assert raw_v1["version"] == 1
    assert "provenance" not in raw_v1

    rolled_back = read_index(v1_path, embedder.model, embedder.dimensions)
    assert rolled_back.provenance is not None
    assert rolled_back.provenance.provenance_status == "legacy_unknown"
    assert rolled_back.provenance.corpus_sha256 is None


def test_restore_all_merges_complete_and_degrades_honestly_with_legacy(tmp_path: Path) -> None:
    doc_a = tmp_path / "a.md"
    doc_b = tmp_path / "b.md"
    doc_a.write_text("Alpha document one.", encoding="utf-8")
    doc_b.write_text("Beta document two.", encoding="utf-8")
    embedder = DeterministicTopicEmbedder(dimensions=8)
    settings = make_settings(tmp_path)

    db_dir = tmp_path / "dbs"
    db_dir.mkdir()

    idx_a = MemoryIndex(embedder, settings)
    idx_a.build(chunk_sections(extract(doc_a), size=200, overlap=20))
    idx_a.save(db_dir / "a.npz")

    idx_b = MemoryIndex(embedder, settings)
    idx_b.build(chunk_sections(extract(doc_b), size=200, overlap=20))
    idx_b.save(db_dir / "b.npz")

    merged_complete = MemoryIndex(embedder, settings)
    merged_complete.restore_all(db_dir)
    assert merged_complete.provenance is not None
    assert merged_complete.provenance.provenance_status == "complete"
    assert len(merged_complete.provenance.sources) == 2

    # Replacing b.npz with a legacy v1 file must not fabricate hashes for the combined index.
    idx_b.save(db_dir / "b.npz", format_version=1)
    merged_mixed = MemoryIndex(embedder, settings)
    merged_mixed.restore_all(db_dir)
    assert merged_mixed.provenance is not None
    assert merged_mixed.provenance.provenance_status == "legacy_unknown"
    assert merged_mixed.provenance.corpus_sha256 is None
    assert all(source.content_sha256 is None for source in merged_mixed.provenance.sources)


def test_integrity_and_corruption_failures(tmp_path: Path) -> None:
    doc = tmp_path / "doc.md"
    doc.write_text("Alpha beta gamma delta passage.", encoding="utf-8")
    chunks = chunk_sections(extract(doc), size=200, overlap=20)
    embedder = DeterministicTopicEmbedder(dimensions=8)
    index = MemoryIndex(embedder, make_settings(tmp_path))
    index.build(chunks)
    v2_path = tmp_path / "valid.npz"
    index.save(v2_path)

    with np.load(v2_path, allow_pickle=False) as archive:
        orig_meta = json.loads(archive["metadata"].tobytes())
        orig_vecs = np.array(archive["vectors"], copy=True)

    # 1. Tampered vectors fail v2 index_generation_id integrity check.
    tampered_vecs = orig_vecs.copy()
    tampered_vecs[0] = np.roll(tampered_vecs[0], 1)
    bad_vec_path = tmp_path / "tampered-vecs.npz"
    np.savez(
        bad_vec_path,
        metadata=np.frombuffer(json.dumps(orig_meta).encode("utf-8"), dtype=np.uint8),
        vectors=tampered_vecs,
    )
    with pytest.raises(StorageError, match="index_generation_id mismatch"):
        read_index(bad_vec_path, embedder.model, embedder.dimensions)

    # 2. Tampered provenance hash fails integrity check.
    bad_prov_meta = json.loads(json.dumps(orig_meta))
    bad_prov_meta["provenance"]["index_generation_id"] = "sha256:" + "0" * 64
    bad_prov_path = tmp_path / "tampered-prov.npz"
    np.savez(
        bad_prov_path,
        metadata=np.frombuffer(json.dumps(bad_prov_meta).encode("utf-8"), dtype=np.uint8),
        vectors=orig_vecs,
    )
    with pytest.raises(StorageError, match="index_generation_id mismatch"):
        read_index(bad_prov_path, embedder.model, embedder.dimensions)

    # 3. Tampered chunk text fails integrity check.
    bad_text_meta = json.loads(json.dumps(orig_meta))
    bad_text_meta["chunks"][0]["text"] = "Tampered text payload."
    bad_text_path = tmp_path / "tampered-text.npz"
    np.savez(
        bad_text_path,
        metadata=np.frombuffer(json.dumps(bad_text_meta).encode("utf-8"), dtype=np.uint8),
        vectors=orig_vecs,
    )
    with pytest.raises(StorageError, match="index_generation_id mismatch"):
        read_index(bad_text_path, embedder.model, embedder.dimensions)

    # 4. Truncated archive fails with StorageError.
    truncated = tmp_path / "truncated.npz"
    truncated.write_bytes(v2_path.read_bytes()[:32])
    with pytest.raises(StorageError, match="Cannot load index"):
        read_index(truncated, embedder.model, embedder.dimensions)

    # 5. Invalid JSON metadata fails with StorageError.
    invalid_json = tmp_path / "invalid-json.npz"
    np.savez(
        invalid_json,
        metadata=np.frombuffer(b"{broken-json", dtype=np.uint8),
        vectors=orig_vecs,
    )
    with pytest.raises(StorageError, match="Cannot load index"):
        read_index(invalid_json, embedder.model, embedder.dimensions)

    # 6. Incompatible model or dimensions fails.
    with pytest.raises(StorageError, match="does not match query embedder"):
        read_index(v2_path, "other-model", embedder.dimensions)
    with pytest.raises(StorageError, match="does not match query embedder"):
        read_index(v2_path, embedder.model, embedder.dimensions + 4)

    # 7. Duplicate chunk IDs rejected on write and read.
    dup_chunks = (chunks[0], replace(chunks[0], text="Duplicate ID text"))
    with pytest.raises(StorageError, match="duplicate chunk IDs"):
        write_index(
            tmp_path / "dup.npz",
            StoredIndex(
                embedder.model,
                embedder.dimensions,
                dup_chunks,
                np.ones((2, 8), np.float32),
            ),
        )

    # 8. Unsupported version 3 rejected on write and read.
    with pytest.raises(StorageError, match="Unsupported index format version: 3"):
        index.save(tmp_path / "v3.npz", format_version=3)
    v3_meta = json.loads(json.dumps(orig_meta))
    v3_meta["version"] = 3
    v3_path = tmp_path / "bad-v3.npz"
    np.savez(
        v3_path,
        metadata=np.frombuffer(json.dumps(v3_meta).encode("utf-8"), dtype=np.uint8),
        vectors=orig_vecs,
    )
    with pytest.raises(StorageError, match="Unsupported index format version: 3"):
        read_index(v3_path, embedder.model, embedder.dimensions)

    # 9. Writing v2 without provenance raises StorageError.
    with pytest.raises(StorageError, match="without provenance"):
        write_index(
            tmp_path / "no-prov-v2.npz",
            StoredIndex(embedder.model, embedder.dimensions, tuple(chunks), orig_vecs),
            format_version=2,
        )

    # 10. Symlink rejection in read_index and restore_all.
    link_dir = tmp_path / "links"
    link_dir.mkdir()
    symlink_npz = link_dir / "linked.npz"
    symlink_npz.symlink_to(v2_path)
    with pytest.raises(StorageError, match="symlink"):
        read_index(symlink_npz, embedder.model, embedder.dimensions)
    with pytest.raises(StorageError, match="symlinks"):
        index.restore_all(link_dir)


def test_v2_integrity_detects_citation_title_and_source_tampering(tmp_path: Path) -> None:
    doc = tmp_path / "cited.md"
    doc.write_text("Alpha passage with citation metadata.", encoding="utf-8")
    chunks = [
        Chunk(
            id="c-line",
            source=doc.resolve(),
            title="Original Title",
            text="Alpha passage with citation metadata.",
            line_start=1,
            line_end=3,
        ),
        Chunk(
            id="c-page",
            source=(tmp_path / "book.pdf").resolve(),
            title="Book Title",
            text="Page two evidence text.",
            page=2,
        ),
    ]
    embedder = DeterministicTopicEmbedder(dimensions=8)
    index = MemoryIndex(embedder, make_settings(tmp_path))
    index.build(chunks)
    v2_path = tmp_path / "cited.npz"
    index.save(v2_path)

    with np.load(v2_path, allow_pickle=False) as archive:
        orig_meta = json.loads(archive["metadata"].tobytes())
        orig_vecs = np.array(archive["vectors"], copy=True)

    mutations: list[tuple[str, int, dict[str, object]]] = [
        ("title", 0, {"title": "Tampered Title"}),
        ("line_start", 0, {"line_start": 2}),
        ("line_end", 0, {"line_end": 99}),
        ("page", 1, {"page": 42}),
        ("source_name", 0, {"source": str((tmp_path / "renamed.md").resolve())}),
    ]
    for label, chunk_idx, patch in mutations:
        mutated = json.loads(json.dumps(orig_meta))
        mutated["chunks"][chunk_idx].update(patch)
        tampered_path = tmp_path / f"tampered-{label}.npz"
        np.savez(
            tampered_path,
            metadata=np.frombuffer(json.dumps(mutated).encode("utf-8"), dtype=np.uint8),
            vectors=orig_vecs,
        )
        with pytest.raises(StorageError, match="index_generation_id mismatch"):
            read_index(tampered_path, embedder.model, embedder.dimensions)


def test_restore_all_rejects_conflicting_embedding_profiles_and_degrades_parser_mismatch(
    tmp_path: Path,
) -> None:
    doc_a = tmp_path / "a.md"
    doc_b = tmp_path / "b.md"
    doc_a.write_text("Alpha archive one.", encoding="utf-8")
    doc_b.write_text("Beta archive two.", encoding="utf-8")
    settings = make_settings(tmp_path)

    # Conflicting embedding profiles (different revision, same model/dimensions) -> StorageError.
    conflict_dir = tmp_path / "conflict-emb"
    conflict_dir.mkdir()
    emb_rev1 = DeterministicTopicEmbedder(dimensions=8, revision="rev-1")
    emb_rev2 = DeterministicTopicEmbedder(dimensions=8, revision="rev-2")
    idx_1 = MemoryIndex(emb_rev1, settings)
    idx_1.build(chunk_sections(extract(doc_a), size=200, overlap=20))
    idx_1.save(conflict_dir / "a.npz")
    idx_2 = MemoryIndex(emb_rev2, settings)
    idx_2.build(chunk_sections(extract(doc_b), size=200, overlap=20))
    idx_2.save(conflict_dir / "b.npz")

    with pytest.raises(StorageError, match="conflicting embedding profiles"):
        MemoryIndex(emb_rev1, settings).restore_all(conflict_dir)

    # Same embedding profile but incompatible parser policies -> degrade honestly to legacy_unknown.
    policy_dir = tmp_path / "conflict-policy"
    policy_dir.mkdir()
    idx_p1 = MemoryIndex(emb_rev1, settings)
    idx_p1.build(
        chunk_sections(extract(doc_a), size=200, overlap=20),
        parser_policy=ParserPolicy(chunk_size=200, chunk_overlap=20),
    )
    idx_p1.save(policy_dir / "a.npz")
    idx_p2 = MemoryIndex(emb_rev1, settings)
    idx_p2.build(
        chunk_sections(extract(doc_b), size=200, overlap=20),
        parser_policy=ParserPolicy(chunk_size=500, chunk_overlap=50),
    )
    idx_p2.save(policy_dir / "b.npz")

    merged = MemoryIndex(emb_rev1, settings)
    merged.restore_all(policy_dir)
    assert merged.provenance is not None
    assert merged.provenance.provenance_status == "legacy_unknown"
    assert merged.provenance.corpus_sha256 is None
    assert merged.provenance.parser_policy is None


# ---------------------------------------------------------------------------
# 3. Portable fixture manifest export & verification
# ---------------------------------------------------------------------------


def test_portable_manifest_redacts_local_paths_and_verifies_round_trip(tmp_path: Path) -> None:
    secret_dev_dir = tmp_path / "secret_developer_home" / "private_workspace" / "corpus"
    (secret_dev_dir / "guides").mkdir(parents=True)
    doc_a = secret_dev_dir / "guides" / "alpha.md"
    doc_b = secret_dev_dir / "beta.md"
    doc_a.write_text("Alpha guide content for portable manifest.", encoding="utf-8")
    doc_b.write_text("Beta notes content for portable manifest.", encoding="utf-8")

    chunks = chunk_sections(extract(doc_a) + extract(doc_b), size=200, overlap=20)
    embedder = DeterministicTopicEmbedder(
        dimensions=8,
        checkpoint_endpoint="https://dev:supersecret@embed.example.com/v1/embeddings?api_key=xyz",
    )
    index = MemoryIndex(embedder, make_settings(tmp_path))
    index.build(
        chunks,
        source_provenance=[
            SourceProvenance(
                uri="https://docs.example.com/guides/alpha.md",
                content_sha256=file_fingerprint(doc_a),
                revision="sha-alpha",
                source_name="guides/alpha.md",
            ),
            SourceProvenance(
                uri=doc_b.resolve().as_uri(),
                content_sha256=file_fingerprint(doc_b),
                revision="sha-beta",
                source_name="beta.md",
            ),
        ],
    )
    snap_path = tmp_path / "portable.npz"
    index.save(snap_path)
    stored = read_index(snap_path, embedder.model, embedder.dimensions)

    manifest = export_portable_manifest(stored)
    serialized = manifest.model_dump_json()

    # Zero leakage of local developer paths, file:// URIs, or credentials.
    for forbidden in (
        "secret_developer_home",
        "private_workspace",
        str(tmp_path),
        "file://",
        "supersecret",
        "api_key=xyz",
    ):
        assert forbidden not in serialized

    assert manifest.chunk_count == len(stored.chunks)
    assert manifest.index_generation_id == stored.provenance.index_generation_id  # type: ignore[union-attr]
    assert [source.source_name for source in manifest.sources] == ["beta.md", "guides/alpha.md"]
    assert manifest.sources[1].uri == "https://docs.example.com/guides/alpha.md"
    assert manifest.sources[0].uri == "beta.md"

    # JSON round-trip and verification succeed.
    reloaded_manifest = PortableSnapshotManifest.model_validate_json(serialized)
    verify_portable_manifest(stored, reloaded_manifest)

    # Mismatches raise explicit StorageError.
    tampered = reloaded_manifest.model_copy(
        update={"index_generation_id": "sha256:" + "f" * 64}
    )
    with pytest.raises(StorageError, match="index_generation_id"):
        verify_portable_manifest(stored, tampered)

    # PortableSourceEntry rejects unsafe absolute local paths and credentials.
    for bad_name in ("/etc/passwd", "C:\\docs\\a.md", "../escape.md", "file:///tmp/a.md"):
        with pytest.raises(ValidationError):
            PortableSourceEntry(source_name=bad_name, chunk_count=1)
    for bad_uri in ("file:///Users/dev/a.md", "/abs/path/a.md", "https://u:p@example.com/a.md"):
        with pytest.raises(ValidationError):
            PortableSourceEntry(source_name="a.md", uri=bad_uri, chunk_count=1)


def test_portable_manifest_excludes_home_roots_from_shared_prefix() -> None:
    chunks = (
        Chunk(
            "u1",
            Path("/Users/alice/notes/guide.md"),
            "Guide",
            "Alice text.",
            line_start=1,
            line_end=1,
        ),
        Chunk(
            "u2",
            Path("/Users/bob/notes/guide.md"),
            "Guide",
            "Bob text.",
            line_start=1,
            line_end=1,
        ),
    )
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    stored = StoredIndex("test-model", 2, chunks, vectors)

    manifest = export_portable_manifest(stored)
    serialized = manifest.model_dump_json()
    assert "alice" not in serialized
    assert "bob" not in serialized
    assert "/Users" not in serialized
    assert [entry.source_name for entry in manifest.sources] == [
        "source-1/guide.md",
        "source-2/guide.md",
    ]

    linux_chunks = (
        Chunk("h1", Path("/home/carol/docs/a.md"), "A", "Carol text.", line_start=1, line_end=1),
        Chunk("h2", Path("/home/dave/docs/b.md"), "B", "Dave text.", line_start=1, line_end=1),
    )
    linux_manifest = export_portable_manifest(StoredIndex("test-model", 2, linux_chunks, vectors))
    linux_json = linux_manifest.model_dump_json()
    assert "carol" not in linux_json
    assert "dave" not in linux_json
    assert [entry.source_name for entry in linux_manifest.sources] == [
        "source-1/a.md",
        "source-2/b.md",
    ]


# ---------------------------------------------------------------------------
# 4. Fresh-process restart reproducibility
# ---------------------------------------------------------------------------


def test_fresh_process_restart_reproducibility(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "alpha.md").write_text(
        "Alpha orchard cultivation and pruning fundamentals.\n"
        "Provenance tracks deterministic retrieval snapshots.",
        encoding="utf-8",
    )
    (corpus_dir / "marine.md").write_text(
        "Marine ocean tides and coastal current dynamics.\n"
        "Beta delta trace diagnostics across stages.",
        encoding="utf-8",
    )
    files = discover(corpus_dir)
    sections = [section for file in files for section in extract(file)]
    chunks = chunk_sections(sections, size=200, overlap=20)

    settings = make_settings(
        tmp_path, candidates=10, candidate_multiplier=2, rrf_constant=60, top_k=3
    )
    embedder = DeterministicTopicEmbedder(dimensions=8)
    index = MemoryIndex(embedder, settings)
    index.build(chunks)
    snap_path = tmp_path / "repro.npz"
    index.save(snap_path)

    queries = ["alpha orchard provenance", "marine delta trace"]
    in_proc_results = []
    for query_text in queries:
        result, trace = index.search_with_trace(Query(query=query_text, top_k=3))
        fetched = [
            {"id": passage.id, "citation": index.fetch(passage.id).citation}
            for passage in result.results
        ]
        in_proc_results.append(
            {
                "query_fingerprint": trace.query_fingerprint,
                "index_generation_id": trace.index_generation_id,
                "retrieval_profile_id": trace.retrieval_profile_id,
                "returned_ids": trace.returned_ids,
                "passages": [
                    {
                        "id": p.id,
                        "score": p.score,
                        "semantic_score": p.semantic_score,
                        "keyword_score": p.keyword_score,
                        "citation": p.citation,
                    }
                    for p in result.results
                ],
                "fetched": fetched,
                "candidates": [c.model_dump(mode="json") for c in trace.candidates],
            }
        )
    expected_bytes = canonical_json(in_proc_results).encode("utf-8")

    worker_script = (
        "import sys\n"
        "from pathlib import Path\n"
        "import numpy as np\n"
        "from configs.configs import load_config\n"
        "from data.index import MemoryIndex\n"
        "from models.models import Query\n"
        "from models.provenance import canonical_json\n"
        "class SubprocessEmbedder:\n"
        "    provider_name = 'OpenAI'\n"
        "    model = 'deterministic-embed-v1'\n"
        "    dimensions = 8\n"
        "    checkpoint_endpoint = 'http://127.0.0.1:1234/v1/embeddings'\n"
        "    def embed(self, texts, *, progress=None):\n"
        "        vecs = np.zeros((len(texts), self.dimensions), dtype=np.float32)\n"
        "        kws = ('alpha','beta','gamma','delta','orchard','marine','provenance','trace')\n"
        "        for row, text in enumerate(texts):\n"
        "            low = text.casefold()\n"
        "            for col in range(self.dimensions):\n"
        "                vecs[row, col] = float(low.count(kws[col]))\n"
        "            if not np.any(vecs[row]):\n"
        "                vecs[row, 0] = 0.25\n"
        "                vecs[row, 7] = 0.75\n"
        "        return vecs\n"
        "cfg = load_config(Path(sys.argv[1]))\n"
        "idx = MemoryIndex(SubprocessEmbedder(), cfg)\n"
        "idx.restore(Path(sys.argv[2]))\n"
        "out = []\n"
        "for q in ['alpha orchard provenance', 'marine delta trace']:\n"
        "    res, tr = idx.search_with_trace(Query(query=q, top_k=3))\n"
        "    out.append({\n"
        "        'query_fingerprint': tr.query_fingerprint,\n"
        "        'index_generation_id': tr.index_generation_id,\n"
        "        'retrieval_profile_id': tr.retrieval_profile_id,\n"
        "        'returned_ids': tr.returned_ids,\n"
        "        'passages': [\n"
        "            {'id': p.id, 'score': p.score, 'semantic_score': p.semantic_score,\n"
        "             'keyword_score': p.keyword_score, 'citation': p.citation}\n"
        "            for p in res.results\n"
        "        ],\n"
        "        'fetched': [{'id': p.id, 'citation': idx.fetch(p.id).citation}\n"
        "                    for p in res.results],\n"
        "        'candidates': [c.model_dump(mode='json') for c in tr.candidates],\n"
        "    })\n"
        "sys.stdout.write(canonical_json(out))\n"
    )

    outputs: list[bytes] = []
    for _ in range(2):
        completed = subprocess.run(
            [sys.executable, "-c", worker_script, str(settings.path), str(snap_path)],
            check=True,
            capture_output=True,
        )
        outputs.append(completed.stdout)

    assert outputs[0] == outputs[1]
    assert outputs[0] == expected_bytes


# ---------------------------------------------------------------------------
# 5. End-to-end bounded retrieval traces & privacy-safe logging
# ---------------------------------------------------------------------------


def test_end_to_end_bounded_trace_stages_privacy_and_trace_sink(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    corpus_dir = tmp_path / "docs"
    corpus_dir.mkdir()
    secret_passage_sentinel = "CONFIDENTIAL_PASSAGE_SENTINEL_9942"
    secret_query_sentinel = "CONFIDENTIAL_QUERY_SENTINEL_7731"

    doc1 = corpus_dir / "primary.md"
    doc2 = corpus_dir / "duplicate.md"
    doc3 = corpus_dir / "tail.md"
    doc4 = corpus_dir / "lexical.md"
    doc1.write_text(f"Alpha orchard {secret_passage_sentinel} primary guide.", encoding="utf-8")
    # Normalized-whitespace duplicate of doc1 text -> should be dropped_duplicate.
    doc2.write_text(f"Alpha   orchard  {secret_passage_sentinel}  primary guide.", encoding="utf-8")
    doc3.write_text("Alpha beta third distinct passage for top_k cutoff.", encoding="utf-8")
    doc4.write_text(f"Unique {secret_query_sentinel} lexical match only.", encoding="utf-8")

    discovered = discover(corpus_dir)
    sections = [section for file in discovered for section in extract(file)]
    chunks = chunk_sections(sections, size=400, overlap=40)

    source_provenance = [
        SourceProvenance(
            uri=file.resolve().as_uri(),
            content_sha256=file_fingerprint(file),
            revision=f"rev-{file.stem}",
            source_name=file.name,
        )
        for file in discovered
    ]

    settings = make_settings(
        tmp_path, candidates=3, candidate_multiplier=1, rrf_constant=60, top_k=2
    )
    embedder = DeterministicTopicEmbedder(dimensions=8)
    sink = RecordingTraceSink()

    # Default allow_content_debug_logs=False must keep DEBUG logs privacy-safe.
    with caplog.at_level(logging.DEBUG, logger="data.index"):
        builder = MemoryIndex(embedder, settings, trace_sink=sink)
        builder.build(chunks, source_provenance=source_provenance)
        snap_path = tmp_path / "e2e.npz"
        builder.save(snap_path)

        restored = MemoryIndex(embedder, settings, trace_sink=sink)
        restored.restore(snap_path)

        query = Query(query=f"alpha orchard {secret_query_sentinel}", top_k=2)
        result_traced, trace = restored.search_with_trace(query, run_salt="run-1")
        result_plain = restored.search(query)

    # search() and search_with_trace() share exact ranking/scoring/dedupe logic.
    assert result_plain == result_traced
    assert len(sink.traces) == 2
    assert sink.traces[0] == trace

    # Stage & bound assertions.
    assert trace.schema_version == TRACE_SCHEMA_VERSION
    assert trace.query_fingerprint == fingerprint_query(query.query, run_salt="run-1")
    assert trace.index_generation_id == restored.index_generation_id
    assert trace.embedding_profile_id == restored.provenance.embedding_profile_id  # type: ignore[union-attr]
    assert trace.retrieval_profile_id == restored.retrieval_profile_id
    assert trace.eligible_candidates == len(chunks)
    assert trace.candidate_limit == 3
    assert trace.top_k == 2
    assert len(trace.dense_shortlist_ids) <= trace.candidate_limit
    assert len(trace.bm25l_shortlist_ids) <= trace.candidate_limit
    assert len(trace.candidates) == len(trace.fused_shortlist_ids)
    assert len(trace.candidates) <= 2 * trace.candidate_limit
    assert trace.returned_ids == [passage.id for passage in result_traced.results]
    assert len(trace.returned_ids) == 2

    actions = {candidate.dedupe_action for candidate in trace.candidates}
    assert "kept" in actions
    assert "dropped_duplicate" in actions
    assert "not_reached" in actions

    kept = [c for c in trace.candidates if c.dedupe_action == "kept"]
    assert [c.selected_rank for c in kept] == [1, 2]
    assert [c.passage_id for c in kept] == trace.returned_ids
    for candidate in trace.candidates:
        assert candidate.source_revision is not None
        assert candidate.source_revision.startswith("rev-")
        assert "/" not in candidate.source_name
        assert candidate.rrf_rank is not None and candidate.rrf_score is not None
        assert candidate.in_dense_shortlist or candidate.in_bm25l_shortlist
        if candidate.dedupe_action != "kept":
            assert candidate.selected_rank is None

    # Every returned passage resolves via fetch() with verified citation.
    for passage in result_traced.results:
        fetched = restored.fetch(passage.id)
        assert fetched.citation == passage.citation

    # Privacy guarantee: ZERO raw query or passage text in serialized trace or default DEBUG logs.
    trace_json = trace.model_dump_json()
    assert secret_query_sentinel not in trace_json
    assert secret_passage_sentinel not in trace_json
    assert str(tmp_path) not in trace_json

    all_log_text = "\n".join(record.getMessage() for record in caplog.records)
    assert secret_query_sentinel not in all_log_text
    assert secret_passage_sentinel not in all_log_text

    index_log_text = "\n".join(
        record.getMessage() for record in caplog.records if record.name == "data.index"
    )
    assert index_log_text
    assert str(tmp_path) not in index_log_text
    assert secret_query_sentinel not in index_log_text
    assert secret_passage_sentinel not in index_log_text

    # Opt-in allow_content_debug_logs=True enables raw content debug logging when requested.
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="data.index"):
        opt_in = MemoryIndex(embedder, settings, allow_content_debug_logs=True)
        opt_in.restore(snap_path)
        opt_in.search(query)
    opt_in_logs = "\n".join(record.getMessage() for record in caplog.records)
    assert secret_query_sentinel in opt_in_logs
    assert secret_passage_sentinel in opt_in_logs


def test_trace_zero_source_matches_custom_tokenizer_and_uninitialized_error(
    tmp_path: Path,
) -> None:
    doc = tmp_path / "custom.md"
    doc.write_text("Alpha-Beta compound token.", encoding="utf-8")
    chunks = chunk_sections(extract(doc), size=200, overlap=20)
    embedder = DeterministicTopicEmbedder(dimensions=8)
    settings = make_settings(tmp_path)
    sink = RecordingTraceSink()

    uninitialized = MemoryIndex(embedder, settings, trace_sink=sink)
    assert uninitialized.provenance is None
    assert uninitialized.index_generation_id is None
    with pytest.raises(IndexNotReady):
        uninitialized.search_with_trace(Query(query="alpha"))
    assert sink.traces == []

    # Custom injected tokenizer changes retrieval_profile_id.
    default_idx = MemoryIndex(embedder, settings)
    custom_idx = MemoryIndex(
        embedder,
        settings,
        tokenizer=lambda text: text.casefold().split(),
        tokenizer_name="whitespace_casefold_v1",
        trace_sink=sink,
    )
    assert custom_idx.retrieval_profile_id != default_idx.retrieval_profile_id

    custom_idx.build(chunks)
    before_embed_calls = len(embedder.calls)
    result, trace = custom_idx.search_with_trace(
        Query(query="alpha", source="nonexistent-doc.md")
    )
    assert result.results == []
    assert len(embedder.calls) == before_embed_calls
    assert trace.eligible_candidates == 0
    assert trace.source_filter == "nonexistent-doc.md"
    assert trace.returned_ids == []
    assert trace.candidates == []
    assert len(sink.traces) == 1
