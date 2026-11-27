import logging
import os
import tempfile
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from configs.configs import load_config
from documents.chunks import Chunk
from embeddings.embeddings import EmbeddingError, normalize
from models.provenance import (
    PORTABLE_MANIFEST_SCHEMA_VERSION,
    PROVENANCE_SCHEMA_VERSION,
    UNKNOWN_PROVIDER,
    EmbeddingProfile,
    IndexProvenance,
    PortableSnapshotManifest,
    PortableSourceEntry,
    SourceProvenance,
    canonical_chunk_binding,
    compute_index_generation_id,
    fingerprint_corpus,
    fingerprint_embedding_profile,
    fingerprint_vectors,
    sanitize_endpoint,
)

logger = logging.getLogger(__name__)

MANIFEST_VERSION_V1 = 1
MANIFEST_VERSION_V2 = 2
SUPPORTED_MANIFEST_VERSIONS = frozenset({MANIFEST_VERSION_V1, MANIFEST_VERSION_V2})
UNIT_NORM_ATOL = 1e-5
HOME_CONTAINER_NAMES = frozenset({"users", "home"})


class StorageError(ValueError):
    pass


def default_index_path() -> Path:
    return load_config().index_path


class StoredChunk(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    title: str
    text: str = Field(min_length=1)
    page: int | None = Field(ge=1)
    line_start: int | None = Field(ge=1)
    line_end: int | None = Field(ge=1)

    @model_validator(mode="after")
    def validate_citation(self) -> StoredChunk:
        if not Path(self.source).is_absolute():
            raise ValueError("Stored source path must be absolute")
        if not self.text.strip():
            raise ValueError("Stored passages cannot be blank")
        if (self.line_start is None) != (self.line_end is None):
            raise ValueError("Stored line citations must contain both start and end.")
        if self.line_start is not None and self.line_end is not None:
            if self.line_end < self.line_start or self.page is not None:
                raise ValueError("Stored citation range is invalid.")
        return self

    def chunk(self) -> Chunk:
        return Chunk(
            self.id,
            Path(self.source),
            self.title,
            self.text,
            self.page,
            self.line_start,
            self.line_end,
        )


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    version: int
    model: str = Field(min_length=1)
    dimensions: int = Field(gt=0)
    chunks: list[StoredChunk] = Field(min_length=1)
    provenance: IndexProvenance | None = None


@dataclass(frozen=True)
class StoredIndex:
    model: str
    dimensions: int
    chunks: tuple[Chunk, ...]
    vectors: NDArray[np.float32]
    provenance: IndexProvenance | None = None


def _chunk_identity_pairs(chunks: Sequence[Chunk]) -> list[tuple[str, str]]:
    return [
        (
            chunk.id,
            canonical_chunk_binding(
                chunk.source.name,
                chunk.title,
                chunk.text,
                chunk.page,
                chunk.line_start,
                chunk.line_end,
            ),
        )
        for chunk in chunks
    ]


def _ensure_unit_vectors(
    vectors: NDArray[np.float32], rows: int, dimensions: int
) -> NDArray[np.float32]:
    """Validate and unit-normalize vectors without introducing ULP drift on unit inputs."""
    if vectors.shape != (rows, dimensions) or not np.isfinite(vectors).all():
        raise EmbeddingError("Embedding response has an invalid shape or non-finite values.")
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(lengths == 0):
        raise EmbeddingError("Embedding response contains a zero-length vector.")
    if vectors.dtype == np.float32 and np.allclose(lengths, 1.0, rtol=0.0, atol=UNIT_NORM_ATOL):
        return np.array(vectors, dtype=np.float32, copy=True)
    return normalize(vectors, rows, dimensions)


def _build_legacy_provenance(
    model: str,
    dimensions: int,
    chunks: tuple[Chunk, ...],
    vectors: NDArray[np.float32],
) -> IndexProvenance:
    """Construct honest ``legacy_unknown`` provenance for v1 snapshots without fabricating data."""
    embedding_profile = EmbeddingProfile(
        provider=UNKNOWN_PROVIDER,
        model=model,
        dimensions=dimensions,
    )
    embedding_profile_id = fingerprint_embedding_profile(embedding_profile)
    unique_sources = sorted(
        {chunk.source for chunk in chunks}, key=lambda path: str(path).casefold()
    )
    sources = [
        SourceProvenance(
            uri=source.as_uri(),
            content_sha256=None,
            revision=None,
            retrieved_at=None,
            source_name=source.name,
        )
        for source in unique_sources
    ]
    generation_id = compute_index_generation_id(
        _chunk_identity_pairs(chunks),
        embedding_profile_id,
        None,
        fingerprint_vectors(vectors),
    )
    return IndexProvenance(
        schema_version=PROVENANCE_SCHEMA_VERSION,
        provenance_status="legacy_unknown",
        index_generation_id=generation_id,
        corpus_sha256=None,
        embedding_profile=embedding_profile,
        embedding_profile_id=embedding_profile_id,
        parser_policy=None,
        sources=sources,
    )


def _validate_v2_provenance(
    provenance: IndexProvenance,
    model: str,
    dimensions: int,
    chunks: tuple[Chunk, ...],
    vectors: NDArray[np.float32],
) -> IndexProvenance:
    if (
        provenance.embedding_profile.model != model
        or provenance.embedding_profile.dimensions != dimensions
    ):
        raise StorageError("Saved provenance embedding profile does not match index metadata")
    if provenance.provenance_status == "complete" and provenance.corpus_sha256 is not None:
        expected_corpus = fingerprint_corpus(provenance.sources, provenance.parser_policy)
        if provenance.corpus_sha256 != expected_corpus:
            raise StorageError(
                "Saved index provenance integrity check failed: corpus_sha256 mismatch"
            )
    expected_generation_id = compute_index_generation_id(
        _chunk_identity_pairs(chunks),
        provenance.embedding_profile_id,
        provenance.corpus_sha256,
        fingerprint_vectors(vectors),
    )
    if provenance.index_generation_id != expected_generation_id:
        raise StorageError(
            "Saved index provenance integrity check failed: index_generation_id mismatch"
        )
    return provenance


def read_index(path: Path, model: str, dimensions: int) -> StoredIndex:
    raw_path = path.expanduser()
    if raw_path.is_symlink():
        raise StorageError(f"Index archive cannot be a symlink: {raw_path.name}")
    path = raw_path.resolve()
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"metadata", "vectors"}:
                raise StorageError("Index archive must contain metadata and vectors.")
            metadata = archive["metadata"]
            if metadata.ndim != 1 or metadata.dtype != np.uint8:
                raise StorageError("Index metadata should be in utf8 byte array")
            manifest = Manifest.model_validate_json(metadata.tobytes())
            if manifest.version not in SUPPORTED_MANIFEST_VERSIONS:
                raise StorageError(f"Unsupported index format version: {manifest.version}")
            if manifest.version == MANIFEST_VERSION_V2 and manifest.provenance is None:
                raise StorageError(
                    f"Unsupported index format version: {manifest.version} "
                    "requires provenance metadata"
                )
            if manifest.version == MANIFEST_VERSION_V1 and manifest.provenance is not None:
                raise StorageError("Index format version 1 must not include provenance metadata")
            if manifest.model != model or manifest.dimensions != dimensions:
                raise StorageError("Saved model/dimension does not match query embedder")
            if len({chunk.id for chunk in manifest.chunks}) != len(manifest.chunks):
                raise StorageError("Saved index contains duplicate chunk IDs")
            raw_vectors = archive["vectors"]
            if raw_vectors.dtype != np.float32:
                raise StorageError("Saved embeddings must use float32")
            vectors = _ensure_unit_vectors(raw_vectors, len(manifest.chunks), dimensions)
            chunks = tuple(chunk.chunk() for chunk in manifest.chunks)
            if manifest.version == MANIFEST_VERSION_V1:
                provenance = _build_legacy_provenance(model, dimensions, chunks, vectors)
            else:
                if manifest.provenance is None:
                    raise StorageError("Saved v2 index is missing provenance metadata")
                provenance = _validate_v2_provenance(
                    manifest.provenance, model, dimensions, chunks, vectors
                )
    except (OSError, ValueError, BadZipFile, EOFError, ValidationError, EmbeddingError) as exc:
        raise StorageError(f"Cannot load index {path}: {exc}") from exc
    logger.info("Loaded index file %s (%d passages)", path, len(manifest.chunks))
    return StoredIndex(model, dimensions, chunks, vectors, provenance=provenance)


def _resolve_write_version(index: StoredIndex, format_version: int | None) -> int:
    if format_version is None:
        return MANIFEST_VERSION_V2 if index.provenance is not None else MANIFEST_VERSION_V1
    if format_version not in SUPPORTED_MANIFEST_VERSIONS:
        raise StorageError(f"Unsupported index format version: {format_version}")
    return format_version


def write_index(path: Path, index: StoredIndex, *, format_version: int | None = None) -> Path:
    path = path.expanduser().resolve()
    try:
        target_version = _resolve_write_version(index, format_version)
        vectors = _ensure_unit_vectors(index.vectors, len(index.chunks), index.dimensions)
        stored_chunks = [
            StoredChunk(
                id=chunk.id,
                source=str(chunk.source.resolve()),
                title=chunk.title,
                text=chunk.text,
                page=chunk.page,
                line_start=chunk.line_start,
                line_end=chunk.line_end,
            )
            for chunk in index.chunks
        ]
        if len({chunk.id for chunk in stored_chunks}) != len(stored_chunks):
            raise StorageError("Saved index contains duplicate chunk IDs")
        if target_version == MANIFEST_VERSION_V2:
            if index.provenance is None:
                raise StorageError("Cannot write index format version 2 without provenance")
            provenance = _validate_v2_provenance(
                index.provenance, index.model, index.dimensions, index.chunks, vectors
            )
            manifest = Manifest(
                version=MANIFEST_VERSION_V2,
                model=index.model,
                dimensions=index.dimensions,
                chunks=stored_chunks,
                provenance=provenance,
            )
            serialized = manifest.model_dump_json()
        else:
            manifest = Manifest(
                version=MANIFEST_VERSION_V1,
                model=index.model,
                dimensions=index.dimensions,
                chunks=stored_chunks,
                provenance=None,
            )
            serialized = manifest.model_dump_json(exclude={"provenance"})
        metadata = np.frombuffer(serialized.encode("utf-8"), dtype=np.uint8)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.stem}-", suffix=".npz", delete=False
        ) as temp_file:
            temp_path = Path(temp_file.name)
        try:
            np.savez(temp_path, metadata=metadata, vectors=vectors)
            os.replace(temp_path, path)
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise
    except (OSError, ValueError, ValidationError, EmbeddingError) as exc:
        raise StorageError(f"Cannot write index {path}: {exc}") from exc
    logger.info("Saved index file %s (%d passages)", path, len(index.chunks))
    return path


def _is_safe_relative_name(name: str | None) -> bool:
    if name is None:
        return False
    cleaned = name.strip()
    if not cleaned or "://" in cleaned or "\\" in cleaned:
        return False
    if cleaned.startswith("/"):
        return False
    if len(cleaned) >= 2 and cleaned[0].isalpha() and cleaned[1] == ":":
        return False
    return ".." not in PurePosixPath(cleaned).parts


def _is_disallowed_common_root(common: Path) -> bool:
    parts = common.parts
    if len(parts) <= 1:
        return True
    return len(parts) == 2 and parts[1].casefold() in HOME_CONTAINER_NAMES


def _relative_source_names(paths: Sequence[Path]) -> dict[Path, str]:
    if not paths:
        return {}
    if len(paths) == 1:
        return {paths[0]: paths[0].name}
    try:
        common = Path(os.path.commonpath([str(path) for path in paths]))
    except ValueError:
        common = None
    if common is not None and not _is_disallowed_common_root(common):
        return {path: path.relative_to(common).as_posix() for path in paths}
    return {path: f"source-{index}/{path.name}" for index, path in enumerate(paths, 1)}


def _match_source_provenance(
    source_path: Path, sources: Sequence[SourceProvenance]
) -> SourceProvenance | None:
    resolved_path = source_path.resolve()
    candidates = {
        source_path.as_uri(),
        resolved_path.as_uri(),
        str(source_path),
        str(resolved_path),
    }
    for entry in sources:
        if entry.uri in candidates:
            return entry
    posix_path = resolved_path.as_posix()
    suffix_matches = [
        entry
        for entry in sources
        if entry.source_name is not None
        and (entry.source_name == source_path.name or posix_path.endswith("/" + entry.source_name))
    ]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    return None


def export_portable_manifest(stored: StoredIndex) -> PortableSnapshotManifest:
    """Export a path-redacted, credential-free portable manifest for ``stored``."""
    if not stored.chunks:
        raise StorageError("Cannot export portable manifest for an empty index.")
    vectors = _ensure_unit_vectors(stored.vectors, len(stored.chunks), stored.dimensions)
    provenance = stored.provenance or _build_legacy_provenance(
        stored.model, stored.dimensions, stored.chunks, vectors
    )
    chunk_counts = Counter(chunk.source for chunk in stored.chunks)
    unique_paths = sorted(chunk_counts, key=lambda path: str(path).casefold())
    fallback_names = _relative_source_names(unique_paths)

    entries: list[PortableSourceEntry] = []
    used_names: set[str] = set()
    for source_path in unique_paths:
        matched = _match_source_provenance(source_path, provenance.sources)
        preferred = matched.source_name if matched is not None else None
        fallback = fallback_names[source_path]
        if (
            preferred is not None
            and preferred != source_path.name
            and _is_safe_relative_name(preferred)
            and preferred not in used_names
        ):
            source_name = preferred
        else:
            source_name = fallback
        used_names.add(source_name)

        portable_uri: str | None = source_name
        if matched is not None and matched.uri.casefold().startswith(("http://", "https://")):
            portable_uri = sanitize_endpoint(matched.uri)

        entries.append(
            PortableSourceEntry(
                source_name=source_name,
                uri=portable_uri,
                content_sha256=matched.content_sha256 if matched is not None else None,
                revision=matched.revision if matched is not None else None,
                retrieved_at=matched.retrieved_at if matched is not None else None,
                chunk_count=chunk_counts[source_path],
            )
        )
    entries.sort(key=lambda entry: (entry.source_name, entry.uri or ""))
    clean_profile = EmbeddingProfile(
        provider=provenance.embedding_profile.provider,
        model=provenance.embedding_profile.model,
        dimensions=provenance.embedding_profile.dimensions,
        endpoint=sanitize_endpoint(provenance.embedding_profile.endpoint),
        revision=provenance.embedding_profile.revision,
        input_role=provenance.embedding_profile.input_role,
    )
    return PortableSnapshotManifest(
        schema_version=PORTABLE_MANIFEST_SCHEMA_VERSION,
        provenance_status=provenance.provenance_status,
        index_generation_id=provenance.index_generation_id,
        corpus_sha256=provenance.corpus_sha256,
        embedding_profile=clean_profile,
        embedding_profile_id=fingerprint_embedding_profile(clean_profile),
        parser_policy=provenance.parser_policy,
        chunk_count=len(stored.chunks),
        sources=entries,
    )


def verify_portable_manifest(stored: StoredIndex, manifest: PortableSnapshotManifest) -> None:
    """Verify that ``stored`` matches ``manifest``, raising ``StorageError`` on mismatch."""
    expected = export_portable_manifest(stored)
    checks = (
        ("schema_version", manifest.schema_version, expected.schema_version),
        ("provenance_status", manifest.provenance_status, expected.provenance_status),
        ("index_generation_id", manifest.index_generation_id, expected.index_generation_id),
        ("corpus_sha256", manifest.corpus_sha256, expected.corpus_sha256),
        ("embedding_profile", manifest.embedding_profile, expected.embedding_profile),
        ("embedding_profile_id", manifest.embedding_profile_id, expected.embedding_profile_id),
        ("parser_policy", manifest.parser_policy, expected.parser_policy),
        ("chunk_count", manifest.chunk_count, expected.chunk_count),
        ("sources", manifest.sources, expected.sources),
    )
    mismatches = [name for name, actual, want in checks if actual != want]
    if mismatches:
        raise StorageError(
            "Portable manifest verification failed; mismatched fields: "
            + ", ".join(mismatches)
        )
