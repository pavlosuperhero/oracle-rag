import logging
import re
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from threading import Lock
from time import perf_counter
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from rank_bm25 import BM25L

from configs.configs import Settings, load_config
from data.sufficiency import INSUFFICIENT_EVIDENCE_MESSAGE, EvidenceGate
from documents.chunks import (
    Chunk,
    DocumentError,
    file_fingerprint,
    load_chunks,
    preparation_policy,
)
from embeddings.embeddings import BaseEmbedding, Embedder, normalize
from models.models import IndexStats, Passage, Query, SearchResult, SourceCatalog, SourceInfo
from models.provenance import (
    DEFAULT_EXTRACTOR,
    DEFAULT_INPUT_ROLE,
    DEFAULT_SUFFICIENCY_POLICY,
    DEFAULT_TOKENIZER_NAME,
    PROVENANCE_SCHEMA_VERSION,
    TRACE_SCHEMA_VERSION,
    UNKNOWN_PROVIDER,
    CandidateTrace,
    EmbeddingProfile,
    IndexProvenance,
    ParserPolicy,
    RetrievalProfile,
    RetrievalTrace,
    SourceProvenance,
    TraceSink,
    canonical_chunk_binding,
    compute_index_generation_id,
    fingerprint_corpus,
    fingerprint_embedding_profile,
    fingerprint_query,
    fingerprint_retrieval_profile,
    fingerprint_vectors,
    sanitize_endpoint,
)
from runtime.preparation import PreparationCache
from runtime.progress import IngestProgress
from storages.storage import StorageError, StoredIndex, read_index, write_index

logger = logging.getLogger(__name__)


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())


@dataclass(frozen=True)
class Snapshot:
    chunks: tuple[Chunk, ...]
    vectors: NDArray[np.float32]
    keywords: BM25L
    provenance: IndexProvenance | None = None


class IndexNotReady(RuntimeError):
    pass


def _extract_embedder_profile(embedder: Embedder) -> EmbeddingProfile:
    raw_provider = getattr(embedder, "provider_name", None)
    provider = (
        raw_provider.strip()
        if isinstance(raw_provider, str) and raw_provider.strip()
        else UNKNOWN_PROVIDER
    )
    endpoint: str | None = None
    if provider == "Bedrock":
        region = getattr(embedder, "region", None)
        if isinstance(region, str) and region.strip():
            endpoint = f"bedrock://{region.strip()}"
    else:
        raw_endpoint = getattr(embedder, "checkpoint_endpoint", None) or getattr(
            embedder, "endpoint", None
        )
        if isinstance(raw_endpoint, str) and raw_endpoint.strip():
            endpoint = sanitize_endpoint(raw_endpoint)
    raw_revision = getattr(embedder, "revision", None)
    revision = (
        raw_revision.strip()
        if isinstance(raw_revision, str) and raw_revision.strip()
        else None
    )
    return EmbeddingProfile(
        provider=provider,
        model=embedder.model,
        dimensions=embedder.dimensions,
        endpoint=endpoint,
        revision=revision,
        input_role=DEFAULT_INPUT_ROLE,
    )


def _derive_source_provenance(chunks: Sequence[Chunk]) -> list[SourceProvenance]:
    unique_sources = sorted(
        {chunk.source for chunk in chunks}, key=lambda path: str(path).casefold()
    )
    entries: list[SourceProvenance] = []
    for source in unique_sources:
        resolved = source.resolve() if source.is_absolute() else source
        content_sha256 = file_fingerprint(resolved) if resolved.is_file() else None
        uri = resolved.as_uri() if resolved.is_absolute() else str(resolved)
        entries.append(
            SourceProvenance(
                uri=uri,
                content_sha256=content_sha256,
                revision=None,
                retrieved_at=None,
                source_name=source.name,
            )
        )
    return entries


def _is_safe_trace_source_name(name: str | None) -> bool:
    if name is None:
        return False
    cleaned = name.strip()
    if not cleaned or "://" in cleaned or "\\" in cleaned or cleaned.startswith("/"):
        return False
    if len(cleaned) >= 2 and cleaned[0].isalpha() and cleaned[1] == ":":
        return False
    return ".." not in PurePosixPath(cleaned).parts


def _source_metadata_for_chunk(
    chunk: Chunk, sources: Sequence[SourceProvenance]
) -> tuple[str, str | None]:
    resolved_path = chunk.source.resolve() if chunk.source.is_absolute() else chunk.source
    uri_candidates = {str(chunk.source), str(resolved_path)}
    if chunk.source.is_absolute():
        uri_candidates.add(chunk.source.as_uri())
    if resolved_path.is_absolute():
        uri_candidates.add(resolved_path.as_uri())
    matched: SourceProvenance | None = None
    for entry in sources:
        if entry.uri in uri_candidates:
            matched = entry
            break
    if matched is None:
        posix_path = resolved_path.as_posix()
        name_matches = [
            entry
            for entry in sources
            if entry.source_name is not None
            and (
                entry.source_name == chunk.source.name
                or posix_path.endswith("/" + entry.source_name)
            )
        ]
        if len(name_matches) == 1:
            matched = name_matches[0]
    if matched is not None:
        name = (
            matched.source_name
            if _is_safe_trace_source_name(matched.source_name)
            else chunk.source.name
        )
        return name or chunk.source.name, matched.revision
    return chunk.source.name, None


def _shortlist(
    candidates: Sequence[int], evidence: NDArray[Any], limit: int
) -> list[int]:
    return sorted(
        (number for number in candidates if evidence[number] > 0),
        key=lambda number: (-float(evidence[number]), number),
    )[:limit]


def _fuse_rrf(
    dense_shortlist: Sequence[int], bm25l_shortlist: Sequence[int], rrf_constant: int
) -> tuple[list[int], dict[int, float]]:
    scores: dict[int, float] = defaultdict(float)
    for shortlist in (dense_shortlist, bm25l_shortlist):
        for rank, number in enumerate(shortlist, 1):
            scores[number] += 1.0 / (rrf_constant + rank)
    ranking = sorted(scores, key=lambda number: (-scores[number], number))
    return ranking, dict(scores)


def _select_deduplicated(
    chunks: tuple[Chunk, ...], ranking: Sequence[int], top_k: int
) -> tuple[
    list[int],
    dict[int, Literal["kept", "dropped_duplicate", "not_reached"]],
    dict[int, int],
]:
    selected: list[int] = []
    seen: set[str] = set()
    actions: dict[int, Literal["kept", "dropped_duplicate", "not_reached"]] = {}
    selected_ranks: dict[int, int] = {}
    for number in ranking:
        if len(selected) == top_k:
            actions[number] = "not_reached"
            continue
        fingerprint = " ".join(chunks[number].text.split())
        if fingerprint in seen:
            actions[number] = "dropped_duplicate"
            continue
        seen.add(fingerprint)
        selected.append(number)
        actions[number] = "kept"
        selected_ranks[number] = len(selected)
    return selected, actions, selected_ranks


def _build_candidate_traces(
    chunks: tuple[Chunk, ...],
    sources: Sequence[SourceProvenance],
    dense_shortlist: Sequence[int],
    bm25l_shortlist: Sequence[int],
    fused_ranking: Sequence[int],
    semantic: NDArray[Any],
    keyword: NDArray[Any],
    rrf_scores: dict[int, float],
    dedupe_actions: dict[int, Literal["kept", "dropped_duplicate", "not_reached"]],
    selected_ranks: dict[int, int],
) -> list[CandidateTrace]:
    dense_ranks = {number: rank for rank, number in enumerate(dense_shortlist, 1)}
    bm25l_ranks = {number: rank for rank, number in enumerate(bm25l_shortlist, 1)}
    traces: list[CandidateTrace] = []
    for rrf_rank, number in enumerate(fused_ranking, 1):
        chunk = chunks[number]
        source_name, source_revision = _source_metadata_for_chunk(chunk, sources)
        traces.append(
            CandidateTrace(
                passage_id=chunk.id,
                source_name=source_name,
                source_revision=source_revision,
                dense_rank=dense_ranks.get(number),
                dense_score=float(semantic[number]),
                bm25l_rank=bm25l_ranks.get(number),
                bm25l_score=float(keyword[number]),
                in_dense_shortlist=number in dense_ranks,
                in_bm25l_shortlist=number in bm25l_ranks,
                rrf_rank=rrf_rank,
                rrf_score=rrf_scores[number],
                dedupe_action=dedupe_actions[number],
                selected_rank=selected_ranks.get(number),
            )
        )
    return traces


def _chunk_binding_pairs(chunks: Sequence[Chunk]) -> list[tuple[str, str]]:
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


def _merge_restored_provenance(
    loaded: Sequence[StoredIndex],
    merged_chunks: tuple[Chunk, ...],
    merged_vectors: NDArray[np.float32],
    fallback_profile: EmbeddingProfile,
) -> IndexProvenance:
    if len(loaded) == 1 and loaded[0].provenance is not None:
        return loaded[0].provenance
    provenances = [item.provenance for item in loaded]
    all_complete = all(
        prov is not None and prov.provenance_status == "complete" for prov in provenances
    )
    complete_provs = [prov for prov in provenances if prov is not None]
    unique_profiles = {prov.embedding_profile for prov in complete_provs}
    if all_complete and len(unique_profiles) > 1:
        raise StorageError("Cannot merge snapshots with conflicting embedding profiles")
    embedding_profile = (
        next(iter(unique_profiles)) if len(unique_profiles) == 1 else fallback_profile
    )
    embedding_profile_id = fingerprint_embedding_profile(embedding_profile)
    vectors_sha256 = fingerprint_vectors(merged_vectors)
    chunk_pairs = _chunk_binding_pairs(merged_chunks)

    if all_complete:
        unique_policies = {prov.parser_policy for prov in complete_provs}
        if len(unique_policies) == 1:
            unique_sources = sorted(
                {source for prov in complete_provs for source in prov.sources},
                key=lambda source: (source.source_name or "", source.uri),
            )
            merged_policy = next(iter(unique_policies))
            corpus_sha256 = fingerprint_corpus(unique_sources, merged_policy)
            generation_id = compute_index_generation_id(
                chunk_pairs, embedding_profile_id, corpus_sha256, vectors_sha256
            )
            return IndexProvenance(
                schema_version=PROVENANCE_SCHEMA_VERSION,
                provenance_status="complete",
                index_generation_id=generation_id,
                corpus_sha256=corpus_sha256,
                embedding_profile=embedding_profile,
                embedding_profile_id=embedding_profile_id,
                parser_policy=merged_policy,
                sources=unique_sources,
            )

    legacy_sources = [
        SourceProvenance(
            uri=source.as_uri(),
            content_sha256=None,
            revision=None,
            retrieved_at=None,
            source_name=source.name,
        )
        for source in sorted(
            {chunk.source for chunk in merged_chunks}, key=lambda path: str(path).casefold()
        )
    ]
    generation_id = compute_index_generation_id(
        chunk_pairs, embedding_profile_id, None, vectors_sha256
    )
    return IndexProvenance(
        schema_version=PROVENANCE_SCHEMA_VERSION,
        provenance_status="legacy_unknown",
        index_generation_id=generation_id,
        corpus_sha256=None,
        embedding_profile=embedding_profile,
        embedding_profile_id=embedding_profile_id,
        parser_policy=None,
        sources=legacy_sources,
    )


class MemoryIndex:
    def __init__(
        self,
        embedder: Embedder,
        settings: Settings | None = None,
        tokenizer: Callable[[str], list[str]] | None = None,
        tokenizer_name: str = DEFAULT_TOKENIZER_NAME,
        trace_sink: TraceSink | None = None,
        allow_content_debug_logs: bool = False,
        evidence_gate: EvidenceGate | None = None,
    ) -> None:
        if not tokenizer_name.strip():
            raise ValueError("tokenizer_name must not be blank.")
        if evidence_gate is not None and not evidence_gate.policy_name.strip():
            raise ValueError("evidence_gate.policy_name must not be blank.")
        self.settings = settings or load_config()
        self._embedder = embedder
        self._tokenizer: Callable[[str], list[str]] = tokenizer or tokenize
        self._tokenizer_name = tokenizer_name
        self._trace_sink = trace_sink
        self._allow_content_debug_logs = allow_content_debug_logs
        self._evidence_gate = evidence_gate
        self._snapshot: Snapshot | None = None
        self._write_lock = Lock()

    def close(self) -> None:
        """Release provider resources while keeping the loaded snapshot usable."""
        if isinstance(self._embedder, BaseEmbedding):
            self._embedder.close()

    @property
    def stats(self) -> IndexStats:
        snapshot = self._snapshot
        sources = sorted({str(chunk.source) for chunk in snapshot.chunks}) if snapshot else []
        return IndexStats(
            documents=len(sources),
            chunks=len(snapshot.chunks) if snapshot else 0,
            dimensions=self._embedder.dimensions,
            model=self._embedder.model,
            sources=sources,
        )

    @property
    def provenance(self) -> IndexProvenance | None:
        snapshot = self._snapshot
        return snapshot.provenance if snapshot is not None else None

    @property
    def index_generation_id(self) -> str | None:
        provenance = self.provenance
        return provenance.index_generation_id if provenance is not None else None

    def _retrieval_profile_for(self, top_k: int) -> RetrievalProfile:
        config = self.settings.retrieval
        sufficiency_policy = (
            self._evidence_gate.policy_name
            if self._evidence_gate is not None
            else DEFAULT_SUFFICIENCY_POLICY
        )
        return RetrievalProfile(
            tokenizer=self._tokenizer_name,
            candidates=config.candidates,
            candidate_multiplier=config.candidate_multiplier,
            rrf_constant=config.rrf_constant,
            top_k=top_k,
            sufficiency_policy=sufficiency_policy,
        )

    @property
    def retrieval_profile(self) -> RetrievalProfile:
        return self._retrieval_profile_for(self.settings.retrieval.top_k)

    @property
    def retrieval_profile_id(self) -> str:
        return fingerprint_retrieval_profile(self.retrieval_profile)

    def load(
        self, path: Path, recursive: bool = False, *, progress: IngestProgress | None = None,
        preparation: PreparationCache | None = None,
    ) -> IndexStats:
        chunks = load_chunks(path, recursive, self.settings, progress, preparation=preparation)
        policy = ParserPolicy(
            extractor=DEFAULT_EXTRACTOR,
            chunk_size=self.settings.chunking.size,
            chunk_overlap=self.settings.chunking.overlap,
            preparation_policy_sha256=preparation_policy(),
        )
        return self.build(
            chunks,
            progress=progress,
            parser_policy=policy,
        )

    def fetch(self, passage_id: str) -> Chunk:
        """Read an indexed passage by identity, without embedding or searching."""
        snapshot = self._snapshot
        if snapshot is None:
            raise IndexNotReady("The in-memory index has not been loaded.")
        for chunk in snapshot.chunks:
            if chunk.id == passage_id:
                return chunk
        raise KeyError(passage_id)

    def sources(self) -> SourceCatalog:
        snapshot = self._snapshot
        if snapshot is None:
            raise IndexNotReady("The in-memory index has not been loaded.")
        grouped: dict[Path, list[Chunk]] = defaultdict(list)
        for chunk in snapshot.chunks:
            grouped[chunk.source].append(chunk)
        sources = [
            SourceInfo(
                title=chunks[0].title,
                source=str(path),
                source_uri=path.as_uri(),
                chunks=len(chunks),
                pages=sorted({chunk.page for chunk in chunks if chunk.page is not None}),
            )
            for path, chunks in sorted(grouped.items(), key=lambda item: str(item[0]).casefold())
        ]
        return SourceCatalog(documents=len(sources), sources=sources)

    def restore(self, path: Path) -> IndexStats:
        stored = read_index(path, self._embedder.model, self._embedder.dimensions)
        with self._write_lock:
            self._publish(stored.chunks, stored.vectors, provenance=stored.provenance)
        return self.stats

    def restore_all(self, directory: Path) -> IndexStats:
        directory = directory.expanduser().resolve()
        paths = sorted(directory.glob("*.npz"))
        if not paths:
            raise StorageError(f"No database files found in {directory}; run oracle ingest first.")
        chunks: dict[str, Chunk] = {}
        vectors: list[NDArray[np.float32]] = []
        loaded: list[StoredIndex] = []
        for path in paths:
            if path.is_symlink():
                raise StorageError(f"Database files cannot be symlinks: {path}")
            stored = read_index(path, self._embedder.model, self._embedder.dimensions)
            loaded.append(stored)
            for chunk, vector in zip(stored.chunks, stored.vectors, strict=True):
                if chunk.id in chunks:
                    if chunks[chunk.id] != chunk:
                        raise StorageError(f"Conflicting chunk identity in {path}: {chunk.id}")
                    continue
                chunks[chunk.id] = chunk
                vectors.append(vector)
        merged_chunks = tuple(chunks.values())
        merged_vectors = np.asarray(vectors, dtype=np.float32)
        provenance = _merge_restored_provenance(
            loaded, merged_chunks, merged_vectors, _extract_embedder_profile(self._embedder)
        )
        with self._write_lock:
            self._publish(merged_chunks, merged_vectors, provenance=provenance)
        logger.info("Loaded %d database files into one index", len(paths))
        return self.stats

    def save(self, path: Path, *, format_version: int | None = None) -> None:
        snapshot = self._snapshot
        if snapshot is None:
            raise IndexNotReady("The in-memory index has not been loaded.")
        write_index(
            path,
            StoredIndex(
                self._embedder.model,
                self._embedder.dimensions,
                snapshot.chunks,
                snapshot.vectors,
                provenance=snapshot.provenance,
            ),
            format_version=format_version,
        )

    def _publish(
        self,
        chunks: tuple[Chunk, ...],
        vectors: NDArray[np.float32],
        *,
        provenance: IndexProvenance | None = None,
    ) -> None:
        keywords = BM25L([self._tokenizer(f"{chunk.title}\n{chunk.text}") for chunk in chunks])
        vectors.setflags(write=False)
        self._snapshot = Snapshot(chunks, vectors, keywords, provenance=provenance)
        for chunk in chunks:
            if self._allow_content_debug_logs:
                logger.debug(
                    "Indexed passage: id=%s source=%s content=%r",
                    chunk.id,
                    chunk.citation,
                    chunk.text,
                )
            else:
                logger.debug(
                    "Indexed passage: id=%s source=%s chars=%d",
                    chunk.id,
                    chunk.source.name,
                    len(chunk.text),
                )

    def build(
        self,
        chunks: list[Chunk],
        *,
        progress: IngestProgress | None = None,
        source_provenance: Sequence[SourceProvenance] | None = None,
        parser_policy: ParserPolicy | None = None,
    ) -> IndexStats:
        if not chunks:
            raise DocumentError("Cannot build an index without document text.")
        started = perf_counter()
        with self._write_lock:
            texts = [f"{chunk.title}\n{chunk.text}" for chunk in chunks]
            # Third-party embedders without the keyword keep working when no progress is requested.
            embedded = (
                self._embedder.embed(texts)
                if progress is None
                else self._embedder.embed(texts, progress=progress)
            )
            vectors = normalize(
                embedded,
                len(chunks),
                self._embedder.dimensions,
            )
            chunk_tuple = tuple(chunks)
            resolved_sources = (
                list(source_provenance)
                if source_provenance is not None
                else _derive_source_provenance(chunk_tuple)
            )
            resolved_policy = (
                parser_policy
                if parser_policy is not None
                else ParserPolicy(
                    extractor=DEFAULT_EXTRACTOR,
                    chunk_size=self.settings.chunking.size,
                    chunk_overlap=self.settings.chunking.overlap,
                )
            )
            embedding_profile = _extract_embedder_profile(self._embedder)
            embedding_profile_id = fingerprint_embedding_profile(embedding_profile)
            corpus_sha256 = fingerprint_corpus(resolved_sources, resolved_policy)
            generation_id = compute_index_generation_id(
                _chunk_binding_pairs(chunk_tuple),
                embedding_profile_id,
                corpus_sha256,
                fingerprint_vectors(vectors),
            )
            provenance = IndexProvenance(
                schema_version=PROVENANCE_SCHEMA_VERSION,
                provenance_status="complete",
                index_generation_id=generation_id,
                corpus_sha256=corpus_sha256,
                embedding_profile=embedding_profile,
                embedding_profile_id=embedding_profile_id,
                parser_policy=resolved_policy,
                sources=resolved_sources,
            )
            self._publish(chunk_tuple, vectors, provenance=provenance)
        logger.info(
            "Index ready: %d documents, %d chunks in %.2fs",
            self.stats.documents,
            len(chunks),
            perf_counter() - started,
        )
        return self.stats

    def _execute_search(
        self, request: Query, *, run_salt: str = ""
    ) -> tuple[SearchResult, RetrievalTrace]:
        started = perf_counter()
        top_k = request.top_k or self.settings.retrieval.top_k
        query_fp = fingerprint_query(request.query, run_salt=run_salt)
        if self._allow_content_debug_logs:
            logger.debug(
                "Retrieval request: query=%r query_fingerprint=%s source=%r top_k=%d",
                request.query,
                query_fp,
                request.source,
                top_k,
            )
        else:
            logger.debug(
                "Retrieval request: query_fingerprint=%s query_chars=%d source=%r top_k=%d",
                query_fp,
                len(request.query),
                request.source,
                top_k,
            )
        snapshot = self._snapshot
        if snapshot is None:
            raise IndexNotReady("The in-memory index has not been loaded.")
        config = self.settings.retrieval
        limit = max(config.candidates, top_k * config.candidate_multiplier)
        retrieval_profile_id = fingerprint_retrieval_profile(self._retrieval_profile_for(top_k))
        provenance = snapshot.provenance
        candidates = [
            number
            for number, chunk in enumerate(snapshot.chunks)
            if request.source is None or request.source.casefold() in str(chunk.source).casefold()
        ]
        if not candidates:
            elapsed_ms = (perf_counter() - started) * 1000.0
            logger.info("Retrieval completed: 0 source matches (no embedding request)")
            empty_result = SearchResult(
                query=request.query, results=[], message="No documents match the source filter."
            )
            empty_trace = RetrievalTrace(
                schema_version=TRACE_SCHEMA_VERSION,
                query_fingerprint=query_fp,
                index_generation_id=provenance.index_generation_id if provenance else None,
                embedding_profile_id=provenance.embedding_profile_id if provenance else None,
                retrieval_profile_id=retrieval_profile_id,
                source_filter=request.source,
                eligible_candidates=0,
                candidate_limit=limit,
                top_k=top_k,
                dense_shortlist_ids=[],
                bm25l_shortlist_ids=[],
                fused_shortlist_ids=[],
                candidates=[],
                returned_ids=[],
                elapsed_ms=elapsed_ms,
            )
            return empty_result, empty_trace

        query_vector = normalize(
            self._embedder.embed([request.query]), 1, self._embedder.dimensions
        )[0]
        semantic = snapshot.vectors @ query_vector
        query_tokens = self._tokenizer(request.query)
        keyword = np.asarray(snapshot.keywords.get_scores(query_tokens), dtype=np.float64)
        dense_shortlist = _shortlist(candidates, semantic, limit)
        bm25l_shortlist = _shortlist(candidates, keyword, limit)
        fused_ranking, rrf_scores = _fuse_rrf(
            dense_shortlist, bm25l_shortlist, config.rrf_constant
        )
        selected_indices, dedupe_actions, selected_ranks = _select_deduplicated(
            snapshot.chunks, fused_ranking, top_k
        )
        passages: list[Passage] = []
        for number in selected_indices:
            chunk = snapshot.chunks[number]
            score = rrf_scores[number]
            semantic_score = float(semantic[number])
            keyword_score = float(keyword[number])
            passages.append(
                Passage(
                    id=chunk.id,
                    text=chunk.text,
                    title=chunk.title,
                    source=str(chunk.source),
                    source_uri=chunk.source.as_uri(),
                    page=chunk.page,
                    line_start=chunk.line_start,
                    line_end=chunk.line_end,
                    citation=chunk.citation,
                    score=score,
                    semantic_score=semantic_score,
                    keyword_score=keyword_score,
                )
            )
        message: str | None = None if passages else "No passages have positive retrieval scores."
        if self._evidence_gate is not None and passages:
            chunk_tokens = [
                frozenset(
                    self._tokenizer(
                        f"{snapshot.chunks[number].title}\n{snapshot.chunks[number].text}"
                    )
                )
                for number in selected_indices
            ]
            if not self._evidence_gate.is_sufficient(query_tokens, passages, chunk_tokens):
                logger.debug(
                    "Evidence gate abstained: policy=%s candidates=%d",
                    self._evidence_gate.policy_name,
                    len(passages),
                )
                passages = []
                message = INSUFFICIENT_EVIDENCE_MESSAGE
        if passages:
            for number, passage in zip(selected_indices, passages, strict=True):
                chunk = snapshot.chunks[number]
                if self._allow_content_debug_logs:
                    logger.debug(
                        "Retrieved passage: id=%s source=%s score=%.6f content=%r",
                        chunk.id,
                        chunk.citation,
                        passage.score,
                        chunk.text,
                    )
                else:
                    logger.debug(
                        "Retrieved passage: id=%s source=%s score=%.6f semantic=%.6f keyword=%.6f",
                        chunk.id,
                        chunk.source.name,
                        passage.score,
                        passage.semantic_score,
                        passage.keyword_score,
                    )
        elapsed_seconds = perf_counter() - started
        logger.info(
            "Retrieval completed: %d candidates, %d results, %.2fs",
            len(candidates),
            len(passages),
            elapsed_seconds,
        )
        sources = provenance.sources if provenance is not None else []
        candidate_traces = _build_candidate_traces(
            snapshot.chunks,
            sources,
            dense_shortlist,
            bm25l_shortlist,
            fused_ranking,
            semantic,
            keyword,
            rrf_scores,
            dedupe_actions,
            selected_ranks,
        )
        trace = RetrievalTrace(
            schema_version=TRACE_SCHEMA_VERSION,
            query_fingerprint=query_fp,
            index_generation_id=provenance.index_generation_id if provenance else None,
            embedding_profile_id=provenance.embedding_profile_id if provenance else None,
            retrieval_profile_id=retrieval_profile_id,
            source_filter=request.source,
            eligible_candidates=len(candidates),
            candidate_limit=limit,
            top_k=top_k,
            dense_shortlist_ids=[snapshot.chunks[n].id for n in dense_shortlist],
            bm25l_shortlist_ids=[snapshot.chunks[n].id for n in bm25l_shortlist],
            fused_shortlist_ids=[snapshot.chunks[n].id for n in fused_ranking],
            candidates=candidate_traces,
            returned_ids=[passage.id for passage in passages],
            elapsed_ms=elapsed_seconds * 1000.0,
        )
        result = SearchResult(
            query=request.query,
            results=passages,
            message=message,
        )
        return result, trace

    def search_with_trace(
        self, request: Query, *, run_salt: str = ""
    ) -> tuple[SearchResult, RetrievalTrace]:
        """Execute hybrid retrieval and return both passages and a bounded retrieval trace."""
        result, trace = self._execute_search(request, run_salt=run_salt)
        if self._trace_sink is not None:
            self._trace_sink.record(trace)
        return result, trace

    def search(self, request: Query) -> SearchResult:
        result, trace = self._execute_search(request)
        if self._trace_sink is not None:
            self._trace_sink.record(trace)
        return result
