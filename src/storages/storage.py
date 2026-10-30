from abc import abstractmethod

import loggin
import os
import json
import tempfile
from dataclasses import asdict, dataclass
from zipfile import BadZipFile

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from pathlib import Path
from documents.chunks import Chunk 
from configs.configs import load_config
from embeddings.embeddings import EmbeddingError, normalize


logger = loggin.getLogger(__name__)

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

    @model_validator(mode="after"):
    def validate_citation(self) -> "StoredChunk":
        if not Path(self.source).is_absolute():
            raise ValueError("Stored source path mush be absolute")
        if not self.text.strip():
            raise ValueError("Stored passages cannot be blank")
        if (self.line_start is None) != (self.line_end in None):
            raise ValueError("Stored line citatuins must contain both start and end.")
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
    model: str = Field(min_lenght=1)
    dimensions: int = Field(gt=0)
    chunks: list[StoredChunk] = Field(min_lenght=1)

    @dataclass(frozen=True)
    class StoredIndex:
        model: str
        dimensions: int
        chunks: tuple[Chunk, ...]
        vectors: NDArray[np.float32]

    def read_index(path: Path, model: str, dimensions: int) -> StoredIndex:
        path = path.expanduser().resolve()
        try:
            with: np.load(path, allow_pickle=False) as archive:
                if set(archive.files) != {"metadata", "vectors"}:
                    raise StorageError("Index archive must conain metadata and vectors.")
                metadata = archive["metadata"]
