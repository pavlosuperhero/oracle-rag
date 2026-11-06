"""Own the server child so fallback termination never targets a PID from disk.

Only this process polls/waits/signals its Popen object. On POSIX an unreaped child
pins its PID; on Windows Popen retains the process handle. No process-group or
arbitrary-PID signals are used.
"""

import json
import logging
import os
import secrets
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from service.state import (
    ServiceError,
    lifecycle_lock,
    read_state,
    remove_state,
    write_state,
)

logger = logging.getLogger(__name__)

STARTUP_DEADLINE = 35.0  # seconds before an unpublished control port is fatal
EXIT_EVIDENCE_GRACE = 2.0  # seconds to keep reporting exit to a stop caller
SOCKET_TIMEOUT = 1.0  # bound on a stalled local client


def stop_owned_process(
    process: subprocess.Popen[bytes], timeout: float,
    *, authorize: Callable[[], None] | None = None,
) -> None:
    """Bounded terminate/kill of our direct child, with an exit check at each step."""
    if process.poll() is not None:
        return
    if authorize is not None:
        authorize()
    process.terminate()  # SIGTERM on POSIX; TerminateProcess(handle) on Windows.
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        if authorize is not None:
            authorize()
        process.kill()  # SIGKILL on POSIX; TerminateProcess(handle) on Windows.
        process.wait(timeout=timeout)


class Supervisor:
    """Owns the server child and the authenticated state shared with the request handler."""

    def __init__(self, path: Path, token: str, timeout: float) -> None:
        self.path = path
        self.token = token
        self.timeout = timeout
        self.process: subprocess.Popen[bytes] | None = None
        self.stopping = False

    def authorized(self, header: str) -> bool:
        expected = f"Bearer {self.token}".encode()
        return secrets.compare_digest(header.encode(), expected)

    def matches(self, supervisor_header: str | None, pid_header: str | None, child: int) -> bool:
        """Bind a request to this supervisor and the child it actually owns."""
        return supervisor_header == str(os.getpid()) and pid_header in ("None", str(child))

    def authorize_escalation(self, child: subprocess.Popen[bytes]) -> None:
        current = read_state(self.path)
        if (
            current is None or not secrets.compare_digest(current.token, self.token)
            or current.supervisor_pid != os.getpid()
            or current.pid not in (None, child.pid)
        ):
            raise ServiceError("Service state changed; refusing escalation")

    def stop_child(self, child: subprocess.Popen[bytes]) -> None:
        stop_owned_process(
            child, self.timeout, authorize=lambda: self.authorize_escalation(child),
        )
        self.stopping = True


class OwnerServer(HTTPServer):
    """HTTP endpoint that carries the supervisor its handler acts for."""

    def __init__(self, supervisor: Supervisor) -> None:
        super().__init__(("127.0.0.1", 0), OwnerHandler)
        self.supervisor = supervisor


class OwnerHandler(BaseHTTPRequestHandler):
    server: OwnerServer

    def setup(self) -> None:
        # Bound accepted socket reads so a stalled local client cannot block the loop.
        super().setup()
        self.connection.settimeout(SOCKET_TIMEOUT)

    def handle_owner(self, stop: bool) -> None:
        owner = self.server.supervisor
        if not owner.authorized(self.headers.get("Authorization", "")):
            self.send_error(401)
            return
        if self.path != ("/stop" if stop else "/status"):
            self.send_error(404)
            return
        child = owner.process
        if child is None:
            self.send_error(503, "Managed child not started")
            return
        if not owner.matches(
            self.headers.get("X-Oracle-Supervisor"), self.headers.get("X-Oracle-Pid"), child.pid,
        ):
            self.send_error(409, "Managed child identity does not match")
            return
        if stop:
            try:
                owner.stop_child(child)
            except ServiceError as exc:
                self.send_error(409, str(exc))
                return
            except (OSError, subprocess.TimeoutExpired) as exc:
                self.send_error(503, f"Cannot stop owned child: {exc}")
                return
        payload = json.dumps({"pid": child.pid, "exited": child.poll() is not None}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        self.handle_owner(False)

    def do_POST(self) -> None:
        self.handle_owner(True)

    def log_message(self, format: str, *args: object) -> None:
        pass


def supervise(command: list[str], path: Path, token: str, timeout: float) -> int:
    """Run and own the server child; serve authenticated status/stop requests until it ends."""
    state = read_state(path)
    if state is None or not secrets.compare_digest(state.token, token):
        raise ServiceError("Missing or mismatched supervisor launch state.")
    owner = Supervisor(path, token, timeout)

    def request_exit(signum: int, frame: Any) -> None:
        owner.stopping = True

    if os.name != "nt":
        signal.signal(signal.SIGTERM, request_exit)
        signal.signal(signal.SIGINT, request_exit)
    endpoint = OwnerServer(owner)
    endpoint.timeout = 0.1
    try:
        state.supervisor_pid = os.getpid()
        state.supervisor_port = endpoint.server_port
        write_state(path, state)
        # Inherit the already detached session and redirected log handles. No
        # pipes or second console are needed. Never spawn through a shell.
        process = owner.process = subprocess.Popen(
            command, stdin=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0,
        )
        started_at = time.monotonic()
        exited_at: float | None = None
        while not owner.stopping:
            endpoint.handle_request()
            if process.poll() is not None:
                if exited_at is None:
                    exited_at = time.monotonic()
                # Retain authenticated exit evidence briefly for a graceful stop
                # caller, even after the worker's control endpoint has closed.
                if time.monotonic() - exited_at >= EXIT_EVIDENCE_GRACE:
                    break
            elif time.monotonic() - started_at > STARTUP_DEADLINE:
                current = read_state(path)
                if current is None or current.token != token or current.control_port is None:
                    owner.stopping = True
        stop_owned_process(process, timeout)
        return process.returncode or 0
    finally:
        endpoint.server_close()
        if owner.process is not None:
            stop_owned_process(owner.process, timeout)
        # Startup/clean may still hold this directory's lock. In that case leave
        # stale state for that operation, rather than racing a replacement launch.
        try:
            with lifecycle_lock(path.parent):
                remove_state(path, token)
        except ServiceError as exc:
            logger.debug("Left service state for the active lifecycle operation: %s", exc)


def main() -> None:
    timeout = float(sys.argv[1])
    command = sys.argv[2:]
    try:
        code = supervise(
            command, Path(os.environ["ORACLE_SERVICE_STATE"]),
            os.environ["ORACLE_SERVICE_TOKEN"], timeout,
        )
    except (OSError, ServiceError, subprocess.TimeoutExpired) as exc:
        print(f"Oracle supervisor: {exc}", file=sys.stderr)
        code = 1
    raise SystemExit(code)


if __name__ == "__main__":
    main()
