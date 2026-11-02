import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from configs.configs import AiProviderConfig

logger = logging.getLogger(__name__)

MAX_INPUT_CHARS = 50000


class EmbeddingError(ValueError):
    pass


class Embedder(Protocol):
    dimensions: int
    model: str

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]: ...


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


class BaseEmbedding(ABC):
    provider_name: str = "Provider"
    dimensions: int
    model: str
    _config: AiProviderConfig

    @abstractmethod
    def _embed_one(self, text: str) -> list[float]:
        raise NotImplementedError

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
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
        with ThreadPoolExecutor(max_workers=workers) as executor:
            embeddings: dict[str, list[float]] = {}
            for number, (text, vector) in enumerate(
                zip(unique, executor.map(self._embed_one, unique), strict=True), 1
            ):
                embeddings[text] = vector
                if number % self._config.progress_interval == 0 or number == len(unique):
                    logger.info("Embedded %d/%d passages", number, len(unique))
        vectors = np.asarray([embeddings[text] for text in texts], dtype=np.float32)
        return normalize(vectors, len(texts), self.dimensions)
