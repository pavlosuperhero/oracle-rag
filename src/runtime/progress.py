"""Progress port for ingestion.

Core code (documents, embeddings, index) reports plain events through
``IngestProgress``; it never imports a terminal library. The CLI supplies an
adapter that renders the events.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter
from typing import Literal, Protocol

Stage = Literal["discover", "extract", "chunk", "embed"]
STAGES: tuple[Stage, ...] = ("discover", "extract", "chunk", "embed")


@dataclass(frozen=True)
class StageStarted:
    stage: Stage
    total: int | None  # None when unknown


@dataclass(frozen=True)
class StageAdvanced:
    stage: Stage
    completed: int
    tokens_in: int = 0  # cumulative, embed stage only
    retries: int = 0  # cumulative, embed stage only
    errors: int = 0  # cumulative, embed stage only
    retries_unknown: int = 0  # failed requests without SDK retry metadata


@dataclass(frozen=True)
class StageFinished:
    stage: Stage
    elapsed: float


class IngestProgress(Protocol):
    def started(self, event: StageStarted) -> None: ...

    def advanced(self, event: StageAdvanced) -> None: ...

    def finished(self, event: StageFinished) -> None: ...


class NullProgress:
    """Default so library callers and tests need no terminal."""

    def started(self, event: StageStarted) -> None:
        pass

    def advanced(self, event: StageAdvanced) -> None:
        pass

    def finished(self, event: StageFinished) -> None:
        pass


class TotalsRecorder(NullProgress):
    """Remember the latest cumulative counters for the final summary."""

    def __init__(self) -> None:
        self.tokens_in = 0
        self.retries = 0
        self.errors = 0
        self.retries_unknown = 0

    def advanced(self, event: StageAdvanced) -> None:
        self.tokens_in = event.tokens_in
        self.retries = event.retries
        self.errors = event.errors
        self.retries_unknown = event.retries_unknown


Advance = Callable[..., None]


@contextmanager
def tracked_stage(
    progress: IngestProgress, stage: Stage, total: int | None = None
) -> Iterator[Advance]:
    """Emit started/finished around a stage and yield an ``advance(completed, **totals)`` hook.

    ``finished`` is emitted only when the stage completes without raising.
    """
    began = perf_counter()
    progress.started(StageStarted(stage, total))

    def advance(
        completed: int, *, tokens_in: int = 0, retries: int = 0, errors: int = 0,
        retries_unknown: int = 0,
    ) -> None:
        progress.advanced(
            StageAdvanced(stage, completed, tokens_in, retries, errors, retries_unknown)
        )

    yield advance
    progress.finished(StageFinished(stage, perf_counter() - began))
