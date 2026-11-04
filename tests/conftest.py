from pathlib import Path

import numpy as np
import pytest

from configs.configs import Settings, load_config
from data.index import MemoryIndex
from documents.chunks import Chunk
from runtime.progress import IngestProgress, StageAdvanced, StageFinished, StageStarted


class RecordingProgress:
    def __init__(self) -> None:
        self.events: list[StageStarted | StageAdvanced | StageFinished] = []

    def started(self, event: StageStarted) -> None:
        self.events.append(event)

    def advanced(self, event: StageAdvanced) -> None:
        self.events.append(event)

    def finished(self, event: StageFinished) -> None:
        self.events.append(event)


class FakeEmbedder:
    model = "test-model"
    dimensions = 256

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.fail = False

    def embed(self, texts: list[str], *, progress: IngestProgress | None = None) -> np.ndarray:
        self.calls.append(list(texts))
        if self.fail:
            raise RuntimeError("embedding failed")
        vectors = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for number, text in enumerate(texts):
            vectors[number, 0 if "apple" in text.casefold() else 1] = 1
        return vectors


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    config = tmp_path / "config.yaml"
    config.write_text("provider:\n  dimensions: 256\nstorage:\n  index: db/index.npz\n")
    return load_config(config)


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def chunks(tmp_path: Path) -> list[Chunk]:
    return [
        Chunk(
            "apple",
            tmp_path / "fruit.md",
            "Fruit",
            "Apple orchard growing guide.",
            line_start=1,
            line_end=1,
        ),
        Chunk("ocean", tmp_path / "ocean.md", "Ocean", "Ocean waves and marine life.", page=2),
    ]


@pytest.fixture
def index(settings: Settings, embedder: FakeEmbedder, chunks: list[Chunk]) -> MemoryIndex:
    index = MemoryIndex(embedder, settings)
    index.build(chunks)
    return index
