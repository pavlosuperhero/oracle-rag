import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
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
        self.retries_unknown = 0

    def reset(self) -> None:
        with self._lock:
            self.tokens_in = self.retries = self.errors = 0
            self.retries_unknown = 0

    def record(self, tokens_in: int = 0, retries: int = 0) -> None:
        with self._lock:
            self.tokens_in += tokens_in
            self.retries += retries

    def record_error(self, *, retries_unknown: bool = False) -> None:
        with self._lock:
            self.errors += 1
            self.retries_unknown += int(retries_unknown)

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

    async def _embed_one_async(self, text: str) -> list[float]:
        """Embed one passage without blocking the event loop.

        Providers with a native async client override this; the default runs the
        blocking ``_embed_one`` in a worker thread.
        """
        return await asyncio.to_thread(self._embed_one, text)

    @asynccontextmanager
    async def _async_session(self) -> AsyncGenerator[None]:
        """Hold per-call async resources (for example an HTTP client bound to this loop)."""
        yield

    def embed(
        self, texts: Sequence[str], *, progress: IngestProgress | None = None
    ) -> NDArray[np.float32]:
        """Synchronous facade over :meth:`embed_async`."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.embed_async(texts, progress=progress))
        # Called from inside a running loop: drive the pipeline on a private loop instead.
        with ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(asyncio.run, self.embed_async(texts, progress=progress)).result()

    async def embed_async(
        self, texts: Sequence[str], *, progress: IngestProgress | None = None
    ) -> NDArray[np.float32]:
        """Embed ``texts`` concurrently; rows come back in input order.

        Progress is reported from the event-loop thread as each unique passage finishes,
        regardless of the order in which the provider answers.
        """
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
        pending = iter(unique)
        embeddings: dict[str, list[float]] = {}
        with tracked_stage(progress or NullProgress(), "embed", len(unique)) as advance:

            def publish() -> None:
                tokens_in, retries, errors = usage.snapshot()
                advance(
                    len(embeddings), tokens_in=tokens_in, retries=retries, errors=errors,
                    retries_unknown=usage.retries_unknown,
                )

            async def worker() -> None:
                for text in pending:
                    embeddings[text] = await self._embed_one_async(text)
                    publish()
                    number = len(embeddings)
                    if number % self._config.progress_interval == 0 or number == len(unique):
                        logger.info("Embedded %d/%d passages", number, len(unique))

            try:
                async with self._async_session(), asyncio.TaskGroup() as group:
                    for _ in range(min(workers, len(unique))):
                        group.create_task(worker())
            except BaseExceptionGroup as group_error:
                error = group_error.exceptions[0]
                raise error from error.__cause__
            finally:
                # TaskGroup has settled all workers before publishing terminal totals.
                if len(embeddings) != len(unique):
                    publish()
                    tokens_in, retries, errors = usage.snapshot()
                    logger.warning(
                        "Embedding stopped: passages=%d/%d tokens_in=%d retries=%d "
                        "errors=%d retries_unknown=%d elapsed=%.2fs",
                        len(embeddings), len(unique), tokens_in, retries, errors,
                        usage.retries_unknown, perf_counter() - began,
                    )
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
