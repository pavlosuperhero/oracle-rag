import os
import sys

import pytest
from typer.testing import CliRunner

import cli
import service.service as lifecycle
from service.service import (
    owner_status,
    process_alive,
    start_background,
    stop_background,
)
from service.state import ServiceError, read_state, state_path, write_state

# Synthetic workers have no index/provider and never access user data. The test
# supervisor owns every child that can receive a termination signal.
WORKER = """
import os, signal, time
from pathlib import Path
from service.service import run_with_control
class Server:
    started = True
    should_exit = False
    def run(self):
        while not self.should_exit:
            time.sleep(0.01)
run_with_control(Server(), Path(os.environ['ORACLE_SERVICE_STATE']),
                 os.environ['ORACLE_SERVICE_TOKEN'])
"""


@pytest.fixture
def owned_supervisors(monkeypatch):
    spawned = []
    original = lifecycle._spawn_detached

    def spawn(*args, **kwargs):
        process = original(*args, **kwargs)
        spawned.append(process)
        return process

    monkeypatch.setattr(lifecycle, "_spawn_detached", spawn)
    yield spawned
    for process in spawned:
        process.wait(timeout=10)


@pytest.mark.parametrize("unresponsive", [False, True])
def test_real_supervised_child_stops_without_leaving_worker(
    settings, tmp_path, unresponsive, owned_supervisors,
):
    # Force escalation only on POSIX, where TERM and KILL differ. On Windows a
    # direct owned-process terminate is already TerminateProcess(handle).
    code = WORKER
    if unresponsive:
        code = code.replace("while not self.should_exit:", "while True:")
        if os.name != "nt":
            code = code.replace("class Server:", "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                                "class Server:")
    settings.development.shutdown_timout = 1
    start_background(
        [sys.executable, "-c", code], os.environ, settings,
        "127.0.0.1", 8742, tmp_path / "supervised.log",
    )
    path = state_path(settings, "127.0.0.1", 8742)
    state = read_state(path)
    assert state is not None and state.pid is not None
    try:
        assert state.supervisor_pid is not None
        assert state.supervisor_port is not None
        assert stop_background(settings, "127.0.0.1", 8742)
        assert not process_alive(state.pid)
        assert not path.exists()
        assert not stop_background(settings, "127.0.0.1", 8742)
    finally:
        # Request cleanup through the owner, never signal a PID from the file.
        if process_alive(state.pid):
            owner_status(state, stop=True, timeout=5)


def test_supervisor_rejects_forged_child_pid(settings, tmp_path, owned_supervisors):
    start_background(
        [sys.executable, "-c", WORKER], os.environ, settings,
        "127.0.0.1", 8742, tmp_path / "supervised.log",
    )
    state = read_state(state_path(settings, "127.0.0.1", 8742))
    assert state is not None
    try:
        forged = state.model_copy(update={"pid": os.getpid()})
        with pytest.raises(ServiceError, match="Cannot verify"):
            owner_status(forged, stop=True, timeout=2)
        assert not owner_status(state, stop=False, timeout=2)
    finally:
        stop_background(settings, "127.0.0.1", 8742)
    # Reap our direct child rather than treating a POSIX zombie as still running.
    owned_supervisors[0].wait(timeout=5)


def test_stop_with_reused_state_pid_preserves_real_child(settings, tmp_path, owned_supervisors):
    start_background(
        [sys.executable, "-c", WORKER], os.environ, settings,
        "127.0.0.1", 8742, tmp_path / "supervised.log",
    )
    path = state_path(settings, "127.0.0.1", 8742)
    state = read_state(path)
    try:
        write_state(path, state.model_copy(update={"pid": os.getpid()}))
        with pytest.raises(ServiceError, match="Cannot verify"):
            stop_background(settings, "127.0.0.1", 8742)
        assert not owner_status(state, stop=False, timeout=2)
        assert read_state(path).pid == os.getpid()
    finally:
        write_state(path, state)
        stop_background(settings, "127.0.0.1", 8742)


@pytest.mark.parametrize("all_data", [False, True])
def test_real_clean_stops_multiple_managed_readers(
    settings, index, tmp_path, owned_supervisors, all_data,
):
    index.save(settings.index_path)
    states = []
    try:
        for port in (8742, 8743):
            start_background(
                [sys.executable, "-c", WORKER], os.environ, settings,
                "127.0.0.1", port, tmp_path / f"supervised-{port}.log",
            )
            states.append(read_state(state_path(settings, "127.0.0.1", port)))
        args = ["--config", str(settings.path), "clean", *(["--all"] if all_data else [])]
        result = CliRunner().invoke(cli.app, args, input="n\n")
        assert result.exit_code == 0, result.output
        assert settings.index_path.exists()
        assert all(not owner_status(state, stop=False, timeout=2) for state in states)
        result = CliRunner().invoke(cli.app, args, input="y\n")
        assert result.exit_code == 0, result.output
        assert not settings.index_path.exists()
        assert all(not process_alive(state.pid) for state in states)
    finally:
        for port in (8742, 8743):
            stop_background(settings, "127.0.0.1", port)
