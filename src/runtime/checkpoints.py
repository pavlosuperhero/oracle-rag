"""Persistence port for successful ingestion embeddings."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class EmbeddingIdentity:
    provider: str
    model: str
    dimensions: int
    endpoint: str


class EmbeddingCheckpoint(Protocol):
    async def load(self, texts: Sequence[str]) -> dict[str, list[float]]: ...

    async def save(self, text: str, vector: list[float]) -> None: ...
