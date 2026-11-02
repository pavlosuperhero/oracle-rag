import logging
from dataclasses import dataclass
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from configs.configs import load_config
from documents.chunks import Chunk
from embeddings.embeddings import EmbeddingError, normalize

logger = logging.getLogger(__name__)


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
    def validate_citation(self) -> "StoredChunk":
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


@dataclass(frozen=True)
class StoredIndex:
    model: str
    dimensions: int
    chunks: tuple[Chunk, ...]
    vectors: NDArray[np.float32]


def read_index(path: Path, model: str, dimensions: int) -> StoredIndex:
    path = path.expanduser().resolve()
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"metadata", "vectors"}:
                raise StorageError("Index archive must contain metadata and vectors.")
            metadata = archive["metadata"]
            if metadata.ndim != 1 or metadata.dtype != np.uint8:
                raise StorageError("Index metadata should be in utf8 byte array")
            manifest = Manifest.model_validate_json(metadata.tobytes())
            if manifest.version != 1:
                raise StorageError(f"Unsupported index format version: {manifest.version}")
            if manifest.model != model or manifest.dimensions != dimensions:
                raise StorageError("Saved model/dimension does not match query embedder")
            if len({chunk.id for chunk in manifest.chunks}) != len(manifest.chunks):
                raise StorageError("Saved index contains duplicate chunk IDs")
            raw_vectors = archive["vectors"]
            if raw_vectors.dtype != np.float32:
                raise StorageError("Saved embeddings must use float32")
            vectors = normalize(raw_vectors, len(manifest.chunks), dimensions)
    except (OSError, ValueError, BadZipFile, EOFError, ValidationError, EmbeddingError) as exc:
        raise StorageError(f"Cannot load index {path}: {exc}") from exc
    logger.info("Loaded index file %s (%d passages)", path, len(manifest.chunks))
    return StoredIndex(
        model, dimensions, tuple(chunk.chunk() for chunk in manifest.chunks), vectors
    )
