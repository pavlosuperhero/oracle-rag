"""Persistent launch state and the cross-process lifecycle lock for managed servers."""

import hashlib
import os
import secrets
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from configs.configs import Settings


class ServiceError(RuntimeError):
    pass


class ServiceState(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    token: str
    launcher_pid: int
    pid: int | None = None
    control_port: int | None = None
    log_path: str | None = None
    supervisor_pid: int | None = Field(default=None, gt=0)
    supervisor_port: int | None = Field(default=None, gt=0, le=65535)


def state_path(settings: Settings, host: str, port: int) -> Path:
    identity = f"{settings.path}:{host}:{port}"
    digest = hashlib.sha256(identity.encode()).hexdigest()[:20]
    return settings.index_path.parent / f".oracle-service-{digest}.json"


def read_state(path: Path) -> ServiceState | None:
    try:
        return ServiceState.model_validate_json(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValidationError) as exc:
        raise ServiceError(f"Cannot read service state {path}: {exc}") from exc


def remove_state(path: Path, token: str) -> None:
    state = read_state(path)
    if state is not None and secrets.compare_digest(state.token, token):
        path.unlink(missing_ok=True)


def write_state(path: Path, state: ServiceState) -> None:
    """Publish private launch state atomically, without replacing another token."""
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as file:
            temporary = Path(file.name)
            file.write(state.model_dump_json())
        current = read_state(path)
        if current is None or not secrets.compare_digest(current.token, state.token):
            raise ServiceError("Service state changed; refusing to replace another instance.")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def lifecycle_lock(directory: Path) -> Iterator[None]:
    """Serialize managed launch, stop and cleanup across CLI processes."""
    path = directory / ".oracle-lifecycle.lock"
    try:
        directory.mkdir(parents=True, exist_ok=True)
        if path.is_symlink():
            raise ServiceError(f"Refusing symlink lifecycle lock: {path}")
        descriptor = os.open(path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
    except OSError as exc:
        raise ServiceError(f"Cannot open Oracle lifecycle lock {path}: {exc}") from exc
    try:
        try:
            if sys.platform == "win32":
                import msvcrt

                if os.fstat(descriptor).st_size == 0:
                    os.write(descriptor, b"\0")
                    os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ServiceError("Another Oracle lifecycle operation is in progress; retry.") from exc
        yield
    finally:
        os.close(descriptor)  # Closing releases flock/the Windows byte-range lock.
