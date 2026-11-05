import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from time import perf_counter

import numpy as np
from numpy.typing import NDArray
from rank_bm25 import BM25L

from configs.configs import Settings, load_config
from documents.chunks import Chunk, DocumentError, load_chunks
from embeddings.embeddings import Embedder, normalize
from models.models import IndexStats, Passage, Query, SearchResult, SourceCatalog, SourceInfo
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


class IndexNotReady(RuntimeError):
    pass


class MemoryIndex:
    def __init__(self, embedder: Embedder, settings: Settings | None = None) -> None:
        self.settings = settings or load_config()
        self._embedder = embedder
        self._snapshot: Snapshot | None = None
        self._write_lock = Lock()

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

    def load(
        self, path: Path, recursive: bool = False, *, progress: IngestProgress | None = None,
        preparation: PreparationCache | None = None,
    ) -> IndexStats:
        return self.build(
            load_chunks(path, recursive, self.settings, progress, preparation=preparation),
            progress=progress,
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
            self._publish(stored.chunks, stored.vectors)
        return self.stats

    def restore_all(self, directory: Path) -> IndexStats:
        directory = directory.expanduser().resolve()
        paths = sorted(directory.glob("*.npz"))
        if not paths:
            raise StorageError(f"No database files found in {directory}; run oracle ingest first.")
        chunks: dict[str, Chunk] = {}
        vectors: list[NDArray[np.float32]] = []
        for path in paths:
            if path.is_symlink():
                raise StorageError(f"Database files cannot be symlinks: {path}")
            stored = read_index(path, self._embedder.model, self._embedder.dimensions)
            for chunk, vector in zip(stored.chunks, stored.vectors, strict=True):
                if chunk.id in chunks:
                    if chunks[chunk.id] != chunk:
                        raise StorageError(f"Conflicting chunk identity in {path}: {chunk.id}")
                    continue
                chunks[chunk.id] = chunk
                vectors.append(vector)
        with self._write_lock:
            self._publish(tuple(chunks.values()), np.asarray(vectors, dtype=np.float32))
        logger.info("Loaded %d database files into one index", len(paths))
        return self.stats

    def save(self, path: Path) -> None:
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
            ),
        )

    def _publish(self, chunks: tuple[Chunk, ...], vectors: NDArray[np.float32]) -> None:
        keywords = BM25L([tokenize(f"{chunk.title}\n{chunk.text}") for chunk in chunks])
        vectors.setflags(write=False)
        self._snapshot = Snapshot(chunks, vectors, keywords)
        for chunk in chunks:
            logger.debug("Indexed passage: source=%s content=%r", chunk.citation, chunk.text)

    def build(
        self, chunks: list[Chunk], *, progress: IngestProgress | None = None
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
            self._publish(tuple(chunks), vectors)
        logger.info(
            "Index ready: %d documents, %d chunks in %.2fs",
            self.stats.documents,
            len(chunks),
            perf_counter() - started,
        )
        return self.stats

    def search(self, request: Query) -> SearchResult:
        started = perf_counter()
        top_k = request.top_k or self.settings.retrieval.top_k
        logger.debug(
            "Retrieval request: query=%r source=%r top_k=%d",
            request.query,
            request.source,
            top_k,
        )
        snapshot = self._snapshot
        if snapshot is None:
            raise IndexNotReady("The in-memory index has not been loaded.")
        candidates = [
            number
            for number, chunk in enumerate(snapshot.chunks)
            if request.source is None or request.source.casefold() in str(chunk.source).casefold()
        ]
        if not candidates:
            logger.info("Retrieval completed: 0 source matches (no embedding request)")
            return SearchResult(
                query=request.query, results=[], message="No documents match the source filter."
            )
        query_vector = normalize(
            self._embedder.embed([request.query]), 1, self._embedder.dimensions
        )[0]
        semantic = snapshot.vectors @ query_vector
        keyword = np.asarray(
            snapshot.keywords.get_scores(tokenize(request.query)), dtype=np.float64
        )
        config = self.settings.retrieval
        limit = max(config.candidates, top_k * config.candidate_multiplier)
        scores: dict[int, float] = defaultdict(float)
        for evidence in (semantic, keyword):
            ranking = sorted(
                (number for number in candidates if evidence[number] > 0),
                key=lambda number: (-float(evidence[number]), number),
            )[:limit]
            for rank, number in enumerate(ranking, 1):
                scores[number] += 1.0 / (config.rrf_constant + rank)
        ranking = sorted(scores, key=lambda number: (-scores[number], number))
        passages: list[Passage] = []
        seen: set[str] = set()
        for number in ranking:
            chunk = snapshot.chunks[number]
            fingerprint = " ".join(chunk.text.split())
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
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
                    score=scores[number],
                    semantic_score=float(semantic[number]),
                    keyword_score=float(keyword[number]),
                )
            )
            logger.debug(
                "Retrieved passage: source=%s score=%.6f content=%r",
                chunk.citation,
                scores[number],
                chunk.text,
            )
            if len(passages) == top_k:
                break
        logger.info(
            "Retrieval completed: %d candidates, %d results, %.2fs",
            len(candidates),
            len(passages),
            perf_counter() - started,
        )
        return SearchResult(
            query=request.query,
            results=passages,
            message=None if passages else "No passages have positive retrieval scores.",
        )
