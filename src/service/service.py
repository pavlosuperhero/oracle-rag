import ctypes
import hashlib
import logging
import os
import secrets
import subprocess
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import IO, Any

import httpx
import uvicorn
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from configs.configs import Settings
from models.models import IndexStats
from runtime.service_status import ServiceStatus

logger = logging.getLogger(__name__)


class ServiceError(RuntimeError):
    pass


class ServiceState(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    token: str
    launcher_pid: int
    pid: int | None = None
    control_port: int | None = None
    log_path: str | None = None


class ControlStatus(BaseModel):
    model_config = ConfigDict(strict=True)

    pid: int = Field(gt=0)
    started: bool
    stopping: bool = False
    uptime_seconds: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    documents: int | None = Field(default=None, ge=0)
    chunks: int | None = Field(default=None, ge=0)
    dimensions: int | None = Field(default=None, gt=0)
    model: str | None = None
    index_location: str | None = None
    log_path: str | None = None


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


def process_alive(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    if os.name == "nt":
        return _windows_process_alive(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _windows_process_alive(pid: int) -> bool:
    """Poll a process handle without sending Windows console control events."""
    # ctypes exposes these dynamic APIs only on Windows.
    windows_ctypes: Any = ctypes
    kernel32 = windows_ctypes.WinDLL("kernel32", use_last_error=True)
    get_last_error = windows_ctypes.get_last_error
    kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    kernel32.WaitForSingleObject.restype = ctypes.c_ulong
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = ctypes.c_int
    handle = kernel32.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
    if not handle:
        error = get_last_error()
        if error == 87:  # ERROR_INVALID_PARAMETER: no such process
            return False
        if error == 5:  # ERROR_ACCESS_DENIED: conservatively treat as alive
            return True
        raise ServiceError(f"Cannot inspect managed Oracle process {pid}: Windows error {error}.")
    try:
        result = kernel32.WaitForSingleObject(handle, 0)
        if result == 258:  # WAIT_TIMEOUT: process has not exited
            return True
        if result == 0:  # WAIT_OBJECT_0: process has exited
            return False
        raise ServiceError(
            f"Cannot inspect managed Oracle process {pid}: Windows error {get_last_error()}."
        )
    finally:
        kernel32.CloseHandle(handle)


def control(state: ServiceState, *, stop: bool = False) -> bool:
    result = control_status(state, stop=stop)
    return result.started and not result.stopping


def control_status(state: ServiceState, *, stop: bool = False) -> ControlStatus:
    if state.control_port is None:
        raise ServiceError("Oracle is still starting; try again shortly.")
    try:
        with httpx.Client(trust_env=False, timeout=2) as client:
            response = client.request(
                "POST" if stop else "GET",
                f"http://127.0.0.1:{state.control_port}/{'stop' if stop else 'status'}",
                headers={"Authorization": f"Bearer {state.token}"},
            )
            response.raise_for_status()
            payload = response.json()
            try:
                result = ControlStatus.model_validate(payload)
            except ValidationError as exc:
                raise ServiceError("Invalid response from managed Oracle server.") from exc
            if state.pid is None or result.pid != state.pid:
                raise ServiceError("Service identity does not match; refusing control request.")
            return result
    except (httpx.HTTPError, ValueError) as exc:
        raise ServiceError(f"Cannot contact managed Oracle server: {exc}") from exc


def run_with_control(
    server: uvicorn.Server,
    service_state: Path,
    service_token: str,
    *,
    index_stats: IndexStats | None = None,
    index_location: Path | None = None,
) -> None:
    state = read_state(service_state)
    if state is None or not secrets.compare_digest(state.token, service_token):
        raise ServiceError("Missing or mismatched service launch state.")
    started_at = time.monotonic()
    log_path = state.log_path

    class Handler(BaseHTTPRequestHandler):
        def handle_control(self, stopping: bool) -> None:
            supplied = self.headers.get("Authorization", "")
            expected = f"Bearer {service_token}"
            if not secrets.compare_digest(supplied.encode(), expected.encode()):
                self.send_error(401)
                return
            if self.path != ("/stop" if stopping else "/status"):
                self.send_error(404)
                return
            body = ControlStatus(
                pid=os.getpid(),
                started=server.started,
                stopping=server.should_exit,
                uptime_seconds=max(0.0, time.monotonic() - started_at),
                documents=index_stats.documents if index_stats is not None else None,
                chunks=index_stats.chunks if index_stats is not None else None,
                dimensions=index_stats.dimensions if index_stats is not None else None,
                model=index_stats.model if index_stats is not None else None,
                index_location=str(index_location) if index_location is not None else None,
                log_path=log_path,
            ).model_dump_json().encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            if stopping:
                server.should_exit = True

        def do_GET(self) -> None:
            self.handle_control(False)

        def do_POST(self) -> None:
            self.handle_control(True)

        def log_message(self, format: str, *args: object) -> None:
            pass

    endpoint = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    state.pid = os.getpid()
    state.control_port = endpoint.server_port
    thread = threading.Thread(target=endpoint.serve_forever, daemon=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=service_state.parent, delete=False) as file:
            temporary = Path(file.name)
            file.write(state.model_dump_json())
        os.replace(temporary, service_state)
        thread.start()
        server.run()
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        if thread.is_alive():
            endpoint.shutdown()
            thread.join(timeout=5)
        endpoint.server_close()
        remove_state(service_state, service_token)


def discard_stale(path: Path, state: ServiceState) -> bool:
    if process_alive(state.pid if state.pid is not None else state.launcher_pid):
        return False
    remove_state(path, state.token)
    return True


def status_background(
    settings: Settings, host: str, port: int
) -> ServiceStatus:
    """Inspect the selected managed service without changing its state."""
    state = read_state(state_path(settings, host, port))
    if state is None:
        return ServiceStatus("stopped")
    pid = state.pid if state.pid is not None else state.launcher_pid
    if not process_alive(pid):
        return ServiceStatus("stopped", log_path=state.log_path)
    if state.control_port is None:
        return ServiceStatus("starting", pid=state.pid, log_path=state.log_path)
    result = control_status(state)
    return ServiceStatus(
        state="stopping" if result.stopping else "running" if result.started else "starting",
        pid=result.pid,
        uptime_seconds=result.uptime_seconds,
        documents=result.documents,
        chunks=result.chunks,
        dimensions=result.dimensions,
        model=result.model,
        index_location=result.index_location,
        log_path=result.log_path or state.log_path,
    )


def _windows_spawn_options() -> tuple[int, int, Any]:
    """Return (primary_flags, fallback_flags, startupinfo) for a hidden, detached process.

    CREATE_NO_WINDOW keeps console programs from allocating a visible window, and the
    hidden STARTUPINFO covers any window created later. CREATE_BREAKAWAY_FROM_JOB lets
    the server outlive a launching terminal whose job object kills its children; it is
    only in the primary flags because a job that forbids breakaway rejects it.
    """
    base_flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(
        subprocess, "CREATE_NO_WINDOW", 0
    )
    if not getattr(subprocess, "CREATE_NO_WINDOW", 0):
        base_flags |= getattr(subprocess, "DETACHED_PROCESS", 0)
    primary_flags = base_flags | getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0)
    startupinfo = None
    if hasattr(subprocess, "STARTUPINFO"):
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= getattr(subprocess, "STARTF_USESHOWWINDOW", 0)
        startupinfo.wShowWindow = getattr(subprocess, "SW_HIDE", 0)
    return primary_flags, base_flags, startupinfo


def _spawn_detached(
    command: Sequence[str], environment: Mapping[str, str], logs: IO[bytes]
) -> subprocess.Popen[bytes]:
    def popen(**platform_options: Any) -> subprocess.Popen[bytes]:
        return subprocess.Popen(
            command, env=environment, stdin=subprocess.DEVNULL, stdout=logs, stderr=logs,
            **platform_options,
        )

    if os.name != "nt":
        return popen(start_new_session=True)
    primary_flags, fallback_flags, startupinfo = _windows_spawn_options()
    try:
        return popen(creationflags=primary_flags, startupinfo=startupinfo)
    except PermissionError:
        # The parent job object forbids breakaway; stay hidden but remain in the job.
        return popen(creationflags=fallback_flags, startupinfo=startupinfo)


def start_background(
    command: Sequence[str],
    environment: Mapping[str, str],
    settings: Settings,
    host: str,
    port: int,
    log_path: Path,
) -> None:
    path = state_path(settings, host, port)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = read_state(path)
    if state is not None and not discard_stale(path, state):
        raise ServiceError(f"Oracle is already running or starting at {host}:{port}.")
    state = ServiceState(
        token=secrets.token_urlsafe(32), launcher_pid=os.getpid(),
        log_path=str(log_path.expanduser().resolve()),
    )
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise ServiceError("Another Oracle startup is in progress.") from exc
    process: subprocess.Popen[bytes] | None = None
    try:
        with os.fdopen(descriptor, "w") as file:
            file.write(state.model_dump_json())
        child_environment = dict(environment)
        child_environment.update(ORACLE_SERVICE_STATE=str(path), ORACLE_SERVICE_TOKEN=state.token)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("ab") as logs:
            process = _spawn_detached(command, child_environment, logs)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise ServiceError(f"Oracle exited with code {process.returncode}. See {log_path}")
            current = read_state(path)
            if current is not None and current.control_port is not None:
                try:
                    if control(current):
                        logger.info(
                            "Started managed Oracle pid=%d at %s:%d", process.pid, host, port
                        )
                        return
                except ServiceError:
                    pass
            time.sleep(0.1)
        raise ServiceError(f"Oracle startup timed out. See {log_path}")
    except BaseException:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=settings.development.shutdown_timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        remove_state(path, state.token)
        raise


def stop_background(settings: Settings, host: str, port: int) -> bool:
    path = state_path(settings, host, port)
    state = read_state(path)
    if state is None or discard_stale(path, state):
        return False
    control(state, stop=True)
    deadline = time.monotonic() + settings.development.shutdown_timeout + 5
    while time.monotonic() < deadline:
        current = read_state(path)
        if current is None or current.token != state.token:
            logger.info("Stopped managed Oracle at %s:%d", host, port)
            return True
        if not process_alive(state.pid):
            remove_state(path, state.token)
            return True
        time.sleep(0.1)
    raise ServiceError("Oracle shutdown timed out; the process was not forcibly killed.")
