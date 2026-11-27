"""Versioned provenance, fingerprinting, and bounded retrieval-trace contracts.

All models in this module are strict, immutable Pydantic v2 value objects and pure
deterministic functions suitable for core domain modules without presentation imports.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Protocol
from urllib.parse import urlsplit, urlunsplit

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PROVENANCE_SCHEMA_VERSION = 1
TRACE_SCHEMA_VERSION = 1
PORTABLE_MANIFEST_SCHEMA_VERSION = 1

SHA256_PREFIX = "sha256:"
DEFAULT_EXTRACTOR = "oracle-builtin"
DEFAULT_INPUT_ROLE = "document_title_and_text"
DEFAULT_TOKENIZER_NAME = "regex_word_v1"
DEFAULT_TIE_BREAK = "score_desc_index_asc"
DEFAULT_DEDUPE_POLICY = "normalized_whitespace_exact"
DEFAULT_SUFFICIENCY_POLICY = "none"
UNKNOWN_PROVIDER = "unknown"

_FORBIDDEN_SECRET_MARKERS = (
    "bearer ",
    "basic ",
    "api_key=",
    "apikey=",
    "token=",
    "aws_secret",
)
_SECRET_TOKEN_RE = re.compile(
    r"(?:\bsk-[A-Za-z0-9_-]{6,}|\b(?:AKIA|ASIA|AIDA|AROA)[0-9A-Z]{16}\b|\bghp_[A-Za-z0-9]{10,}\b)"
)


def _validate_no_secret_leak(value: str, field_name: str) -> None:
    lowered = value.casefold()
    if any(marker in lowered for marker in _FORBIDDEN_SECRET_MARKERS) or _SECRET_TOKEN_RE.search(
        value
    ):
        raise ValueError(f"{field_name} must not contain credentials or bearer tokens.")


def _validate_clean_http_url(url: str, field_name: str) -> None:
    parsed = urlsplit(url)
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{field_name} must not embed credentials.")
    if parsed.query or parsed.fragment:
        raise ValueError(f"{field_name} must not include query parameters or fragments.")


def _is_local_filesystem_path(value: str) -> bool:
    lowered = value.casefold()
    if lowered.startswith("file:"):
        return True
    if value.startswith(("/", "\\")):
        return True
    if len(value) >= 2 and value[0].isalpha() and value[1] == ":":
        return True
    normalized = value.replace("\\", "/")
    return ".." in PurePosixPath(normalized).parts


def sanitize_endpoint(endpoint: str | None) -> str | None:
    """Return a credential-free, query-free endpoint URL or ``None``.

    HTTP(S) URLs retain only scheme, host, port, and path (stripping userinfo,
    query parameters, and fragments). Non-URL strings are rejected if they look
    like bearer tokens or embedded secrets.
    """
    if endpoint is None:
        return None
    cleaned = endpoint.strip()
    if not cleaned:
        return None
    lowered = cleaned.casefold()
    if lowered.startswith(("http://", "https://")):
        parsed = urlsplit(cleaned)
        if not parsed.hostname:
            raise ValueError("HTTP(S) endpoint must include a hostname.")
        host = parsed.hostname
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        netloc = f"{host}:{parsed.port}" if parsed.port is not None else host
        path = parsed.path.rstrip("/")
        sanitized = urlunsplit((parsed.scheme.lower(), netloc, path, "", ""))
        _validate_no_secret_leak(sanitized, "endpoint")
        return sanitized
    _validate_no_secret_leak(cleaned, "endpoint")
    if "@" in cleaned:
        raise ValueError("Endpoint must not embed userinfo or credentials.")
    return cleaned


def _to_canonical_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _to_canonical_value(value.model_dump(mode="json"))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _to_canonical_value(getattr(value, field.name))
            for field in sorted(dataclasses.fields(value), key=lambda f: f.name)
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _to_canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_to_canonical_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_to_canonical_value(item) for item in value]
        return sorted(items, key=canonical_json)
    return value


def canonical_json(payload: Any) -> str:
    """Serialize ``payload`` into deterministic UTF-8 JSON with sorted keys."""
    normalized = _to_canonical_value(payload)
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256_canonical(payload: Any) -> str:
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return f"{SHA256_PREFIX}{digest}"


class SourceProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    uri: str = Field(min_length=1)
    content_sha256: str | None = Field(default=None, min_length=1)
    revision: str | None = Field(default=None, min_length=1)
    retrieved_at: str | None = Field(default=None, min_length=1)
    source_name: str | None = Field(default=None, min_length=1)

    @field_validator("uri")
    @classmethod
    def validate_uri(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Source URI must not be blank.")
        _validate_no_secret_leak(value, "uri")
        if value.casefold().startswith(("http://", "https://")):
            _validate_clean_http_url(value, "uri")
        return value


class ParserPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    extractor: str = Field(default=DEFAULT_EXTRACTOR, min_length=1)
    chunk_size: int | None = Field(default=None, ge=1)
    chunk_overlap: int | None = Field(default=None, ge=0)
    preparation_policy_sha256: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_chunk_window(self) -> ParserPolicy:
        if (
            self.chunk_size is not None
            and self.chunk_overlap is not None
            and self.chunk_overlap >= self.chunk_size
        ):
            raise ValueError("chunk_overlap must be smaller than chunk_size.")
        return self


class EmbeddingProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    dimensions: int = Field(gt=0)
    endpoint: str | None = Field(default=None, min_length=1)
    revision: str | None = Field(default=None, min_length=1)
    input_role: str = Field(default=DEFAULT_INPUT_ROLE, min_length=1)

    @field_validator("endpoint")
    @classmethod
    def validate_endpoint(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value.strip():
            raise ValueError("Embedding endpoint must not be blank.")
        _validate_no_secret_leak(value, "endpoint")
        if value.casefold().startswith(("http://", "https://")):
            _validate_clean_http_url(value, "endpoint")
        elif "@" in value:
            raise ValueError("Embedding endpoint must not embed credentials.")
        return value


class RetrievalProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    tokenizer: str = Field(default=DEFAULT_TOKENIZER_NAME, min_length=1)
    candidates: int = Field(ge=1)
    candidate_multiplier: int = Field(ge=1)
    rrf_constant: int = Field(ge=1)
    top_k: int = Field(ge=1)
    tie_break: str = Field(default=DEFAULT_TIE_BREAK, min_length=1)
    dedupe: str = Field(default=DEFAULT_DEDUPE_POLICY, min_length=1)
    sufficiency_policy: str = Field(default=DEFAULT_SUFFICIENCY_POLICY, min_length=1)


def fingerprint_embedding_profile(profile: EmbeddingProfile) -> str:
    """Return a deterministic ``sha256:<hex>`` identity for an embedding profile."""
    return _sha256_canonical(profile)


def fingerprint_retrieval_profile(profile: RetrievalProfile) -> str:
    """Return a deterministic ``sha256:<hex>`` identity for a retrieval profile."""
    payload = profile.model_dump(mode="json")
    if profile.sufficiency_policy == DEFAULT_SUFFICIENCY_POLICY:
        payload.pop("sufficiency_policy", None)
    return _sha256_canonical(payload)


def fingerprint_corpus(
    sources: Sequence[SourceProvenance], parser_policy: ParserPolicy | None
) -> str:
    """Return a deterministic ``sha256:<hex>`` identity for corpus inputs and parser policy.

    Source order and fetch timestamps (``retrieved_at``) are excluded so the same
    content and parser policy produce an identical fingerprint across runs.
    """
    canonical_sources = sorted(
        (
            {
                "content_sha256": source.content_sha256,
                "revision": source.revision,
                "source_name": source.source_name,
                "uri": source.uri,
            }
            for source in sources
        ),
        key=lambda item: (
            item["source_name"] or "",
            item["uri"] or "",
            item["content_sha256"] or "",
            item["revision"] or "",
        ),
    )
    return _sha256_canonical(
        {
            "parser_policy": parser_policy,
            "schema_version": PROVENANCE_SCHEMA_VERSION,
            "sources": canonical_sources,
        }
    )


def fingerprint_vectors(vectors: NDArray[np.float32]) -> str:
    """Return a deterministic ``sha256:<hex>`` over little-endian float32 matrix bytes."""
    if vectors.ndim != 2:
        raise ValueError("Embedding vectors must be a 2D array.")
    if not np.isfinite(vectors).all():
        raise ValueError("Embedding vectors must be finite.")
    canonical = np.ascontiguousarray(vectors, dtype="<f4")
    digest = hashlib.sha256()
    digest.update(f"{canonical.shape[0]}x{canonical.shape[1]}:".encode("ascii"))
    digest.update(canonical.tobytes())
    return f"{SHA256_PREFIX}{digest.hexdigest()}"


def canonical_chunk_binding(
    source_name: str,
    title: str,
    text: str,
    page: int | None,
    line_start: int | None,
    line_end: int | None,
) -> str:
    """Return canonical JSON binding chunk text and citation metadata for tamper checks."""
    return canonical_json([source_name, title, text, page, line_start, line_end])


def compute_index_generation_id(
    chunks: Sequence[tuple[str, str]],
    embedding_profile_id: str,
    corpus_sha256: str | None,
    vectors_sha256: str,
) -> str:
    """Return a deterministic ``sha256:<hex>`` binding chunks, vectors, and profiles."""
    digest = hashlib.sha256()
    header = canonical_json(
        {
            "chunk_count": len(chunks),
            "corpus_sha256": corpus_sha256,
            "embedding_profile_id": embedding_profile_id,
            "schema_version": PROVENANCE_SCHEMA_VERSION,
            "vectors_sha256": vectors_sha256,
        }
    )
    digest.update(header.encode("utf-8"))
    for chunk_id, chunk_value in chunks:
        digest.update(b"\x00")
        digest.update(canonical_json([chunk_id, chunk_value]).encode("utf-8"))
    return f"{SHA256_PREFIX}{digest.hexdigest()}"


def fingerprint_query(query: str, *, run_salt: str = "") -> str:
    """Return a privacy-safe ``sha256:<hex>`` digest for a query without exposing raw text."""
    return _sha256_canonical({"query": query, "salt": run_salt})


class IndexProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: int = Field(default=PROVENANCE_SCHEMA_VERSION, ge=1)
    provenance_status: Literal["complete", "legacy_unknown"]
    index_generation_id: str = Field(min_length=1)
    corpus_sha256: str | None = Field(default=None, min_length=1)
    embedding_profile: EmbeddingProfile
    embedding_profile_id: str = Field(min_length=1)
    parser_policy: ParserPolicy | None = None
    sources: list[SourceProvenance] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_provenance_consistency(self) -> IndexProvenance:
        if self.schema_version != PROVENANCE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported provenance schema version: {self.schema_version}")
        expected_profile_id = fingerprint_embedding_profile(self.embedding_profile)
        if self.embedding_profile_id != expected_profile_id:
            raise ValueError("embedding_profile_id does not match embedding_profile.")
        if self.provenance_status == "complete" and self.corpus_sha256 is None:
            raise ValueError("Complete provenance requires corpus_sha256.")
        if self.provenance_status == "legacy_unknown" and self.corpus_sha256 is not None:
            raise ValueError("legacy_unknown provenance must not set corpus_sha256.")
        return self


class CandidateTrace(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    passage_id: str = Field(min_length=1)
    source_name: str = Field(min_length=1)
    source_revision: str | None = None
    dense_rank: int | None = Field(default=None, ge=1)
    dense_score: float | None = None
    bm25l_rank: int | None = Field(default=None, ge=1)
    bm25l_score: float | None = None
    in_dense_shortlist: bool
    in_bm25l_shortlist: bool
    rrf_rank: int | None = Field(default=None, ge=1)
    rrf_score: float | None = None
    dedupe_action: Literal["kept", "dropped_duplicate", "not_reached"]
    selected_rank: int | None = Field(default=None, ge=1)


class RetrievalTrace(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: int = Field(default=TRACE_SCHEMA_VERSION, ge=1)
    query_fingerprint: str = Field(min_length=1)
    index_generation_id: str | None = None
    embedding_profile_id: str | None = None
    retrieval_profile_id: str = Field(min_length=1)
    source_filter: str | None = None
    eligible_candidates: int = Field(ge=0)
    candidate_limit: int = Field(ge=1)
    top_k: int = Field(ge=1)
    dense_shortlist_ids: list[str] = Field(default_factory=list)
    bm25l_shortlist_ids: list[str] = Field(default_factory=list)
    fused_shortlist_ids: list[str] = Field(default_factory=list)
    candidates: list[CandidateTrace] = Field(default_factory=list)
    returned_ids: list[str] = Field(default_factory=list)
    elapsed_ms: float = Field(ge=0.0)


class TraceSink(Protocol):
    """Boundary port that receives bounded, privacy-safe per-query retrieval traces."""

    def record(self, trace: RetrievalTrace) -> None: ...


class PortableSourceEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    source_name: str = Field(min_length=1)
    uri: str | None = Field(default=None, min_length=1)
    content_sha256: str | None = Field(default=None, min_length=1)
    revision: str | None = Field(default=None, min_length=1)
    retrieved_at: str | None = Field(default=None, min_length=1)
    chunk_count: int = Field(ge=0)

    @field_validator("source_name")
    @classmethod
    def validate_relative_source_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Portable source_name must not be blank.")
        if "://" in value or "\\" in value or _is_local_filesystem_path(value):
            raise ValueError(
                "Portable source_name must be a safe relative POSIX path without local roots."
            )
        return value

    @field_validator("uri")
    @classmethod
    def validate_portable_uri(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value.strip():
            raise ValueError("Portable URI must not be blank.")
        _validate_no_secret_leak(value, "uri")
        if value.casefold().startswith(("http://", "https://")):
            _validate_clean_http_url(value, "uri")
            return value
        if _is_local_filesystem_path(value):
            raise ValueError("Portable URI must not contain absolute local filesystem paths.")
        return value


class PortableSnapshotManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: int = Field(default=PORTABLE_MANIFEST_SCHEMA_VERSION, ge=1)
    provenance_status: Literal["complete", "legacy_unknown"]
    index_generation_id: str = Field(min_length=1)
    corpus_sha256: str | None = Field(default=None, min_length=1)
    embedding_profile: EmbeddingProfile
    embedding_profile_id: str = Field(min_length=1)
    parser_policy: ParserPolicy | None = None
    chunk_count: int = Field(ge=1)
    sources: list[PortableSourceEntry] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_manifest_consistency(self) -> PortableSnapshotManifest:
        if self.schema_version != PORTABLE_MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported portable manifest schema version: {self.schema_version}"
            )
        expected_profile_id = fingerprint_embedding_profile(self.embedding_profile)
        if self.embedding_profile_id != expected_profile_id:
            raise ValueError("embedding_profile_id does not match embedding_profile.")
        if sum(source.chunk_count for source in self.sources) != self.chunk_count:
            raise ValueError("Sum of source chunk counts must match chunk_count.")
        return self
