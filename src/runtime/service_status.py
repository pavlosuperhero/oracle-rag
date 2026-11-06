"""Immutable service inspection data shared with terminal presentation."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ServiceStatus:
    state: Literal["running", "starting", "stopping", "stopped"]
    pid: int | None = None
    uptime_seconds: float | None = None
    documents: int | None = None
    chunks: int | None = None
    dimensions: int | None = None
    model: str | None = None
    index_location: str | None = None
    log_path: str | None = None
