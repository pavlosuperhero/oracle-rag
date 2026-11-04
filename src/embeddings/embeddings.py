import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from time import perf_counter
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from configs.configs import AiProviderConfig
from runtime.progress import IngestProgress, NullProgress, tracked_stage

logger = logging.getLogger(__name__)

MAX_INPUT_CHARS = 50000


class EmbeddingError(ValueError):
    pass


class Embedder(Protocol):
    dimensions: int
    model: str

    def embed(
        self, texts: Sequence[str], *, progress: IngestProgress | None = None
    ) -> NDArray[np.float32]: ...


def validate_embedding_input(text: str) -> None:
    if not text.strip() or len(text) > MAX_INPUT_CHARS:
        raise EmbeddingError("Embedding input must contain 1-50,000 characters.")


def normalize(
    vectors: NDArray[np.float32], rows: int, dimensions: int
) -> NDArray[np.float32]:
    if vectors.shape != (rows, dimensions) or not np.isfinite(vectors).all():
        raise EmbeddingError("Embedding response has an invalid shape or non-finite values.")
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(lengths == 0):
        raise EmbeddingError("Embedding response contains a zero-length vector.")
    return np.asarray(vectors / lengths, dtype=np.float32)


class UsageMeter:
    """Thread-safe provider usage counters, updated from embedding worker threads."""

    def __init__(self) -> None:
        self._lock = Lock()
        self.tokens_in = 0
        self.retries = 0
        self.errors = 0

    def reset(self) -> None:
        with self._lock:
            self.tokens_in = self.retries = self.errors = 0

    def record(self, tokens_in: int = 0, retries: int = 0) -> None:
        with self._lock:
            self.tokens_in += tokens_in
            self.retries += retries

    def record_error(self) -> None:
        with self._lock:
            self.errors += 1

    def snapshot(self) -> tuple[int, int, int]:
        with self._lock:
            return self.tokens_in, self.retries, self.errors


class BaseEmbedding(ABC):
    provider_name: str = "Provider"
    dimensions: int
    model: str
    _config: AiProviderConfig

    @property
    def usage(self) -> UsageMeter:
        """Counters for the current ``embed()`` call; created lazily for subclasses."""
        meter = self.__dict__.get("_usage")
        if not isinstance(meter, UsageMeter):
            meter = UsageMeter()
            self.__dict__["_usage"] = meter
        return meter

    @abstractmethod
    def _embed_one(self, text: str) -> list[float]:
        raise NotImplementedError

    def embed(
        self, texts: Sequence[str], *, progress: IngestProgress | None = None
    ) -> NDArray[np.float32]:
        if not texts:
            return np.empty((0, self.dimensions), dtype=np.float32)
        unique = list(dict.fromkeys(texts))
        logger.info(
            "Requesting %d %s embeddings (%d unique)",
            len(texts),
            self.provider_name,
            len(unique),
        )
        workers = getattr(self._config, "workers", None) or self._config.worker
        usage = self.usage
        usage.reset()
        began = perf_counter()
        # Events are emitted from this consuming loop, never from worker threads.
        with (
            tracked_stage(progress or NullProgress(), "embed", len(unique)) as advance,
            ThreadPoolExecutor(max_workers=workers) as executor,
        ):
            embeddings: dict[str, list[float]] = {}
            for number, (text, vector) in enumerate(
                zip(unique, executor.map(self._embed_one, unique), strict=True), 1
            ):
                embeddings[text] = vector
                tokens_in, retries, errors = usage.snapshot()
                advance(number, tokens_in=tokens_in, retries=retries, errors=errors)
                if number % self._config.progress_interval == 0 or number == len(unique):
                    logger.info("Embedded %d/%d passages", number, len(unique))
        tokens_in, retries, errors = usage.snapshot()
        logger.info(
            "Embedding finished: passages=%d tokens_in=%d retries=%d errors=%d elapsed=%.2fs",
            len(unique),
            tokens_in,
            retries,
            errors,
            perf_counter() - began,
        )
        vectors = np.asarray([embeddings[text] for text in texts], dtype=np.float32)
        return normalize(vectors, len(texts), self.dimensions)
