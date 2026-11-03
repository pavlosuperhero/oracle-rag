import hashlib
import json
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

import httpx
import uvicorn
from pydantic import BaseModel, ConfigDict, ValidationError

from configs.configs import Settings

logger = logging.getLogger(__name__)


class ServiceError(RuntimeError):
    pass


class ServiceState(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    token: str
    launcher_pid: int
    pid: int | None = None
    control_port: int | None = None


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
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def control(state: ServiceState, *, stop: bool = False) -> bool:
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
            if payload.get("pid") != state.pid:
                raise ServiceError("Service identity does not match; refusing control request.")
            return bool(payload.get("started"))
    except (httpx.HTTPError, ValueError) as exc:
        raise ServiceError(f"Cannot contact managed Oracle server: {exc}") from exc


def run_with_control(
    server: uvicorn.Server,
    service_state: Path,
    service_token: str,
) -> None:
    state = read_state(service_state)
    if state is None or not secrets.compare_digest(state.token, service_token):
        raise ServiceError("Missing or mismatched service launch state.")

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
            body = json.dumps({"pid": os.getpid(), "started": server.started}).encode()
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
    state = ServiceState(token=secrets.token_urlsafe(32), launcher_pid=os.getpid())
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
        flags = 0
        if os.name == "nt":
            flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(
                subprocess, "DETACHED_PROCESS", 0
            )
        with log_path.open("ab") as logs:
            process = subprocess.Popen(
                command, env=child_environment, stdin=subprocess.DEVNULL, stdout=logs, stderr=logs,
                start_new_session=os.name != "nt", creationflags=flags,
            )
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
