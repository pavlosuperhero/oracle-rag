from collections.abc import Mapping, Sequence
from pathlib import Path

import uvicorn

from configs.configs import Settings


class ServiceError(RuntimeError):
    pass


def run_with_control(
    server: uvicorn.Server,
    service_state: Path,
    service_token: str,
) -> None:
    raise NotImplementedError


def start_background(
    command: Sequence[str],
    environment: Mapping[str, str],
    settings: Settings,
    host: str,
    port: int,
    log_path: Path,
) -> None:
    raise NotImplementedError


def stop_background(settings: Settings, host: str, port: int) -> bool:
    raise NotImplementedError
