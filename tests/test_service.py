import json
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from models.models import IndexStats
from service.service import (
    _windows_process_alive,
    _windows_spawn_options,
    control,
    process_alive,
    run_with_control,
    start_background,
    status_background,
    stop_background,
    stopped_services,
)
from service.state import (
    ServiceError,
    ServiceState,
    lifecycle_lock,
    read_state,
    remove_state,
    state_path,
)


def save_state(settings, **overrides):
    path = state_path(settings, "127.0.0.1", 8742)
    path.parent.mkdir(parents=True, exist_ok=True)
    values = {"token": "test-secret", "launcher_pid": os.getpid()}
    values.update(overrides)
    path.write_text(ServiceState(**values).model_dump_json())
    return path


def test_stop_absent_service_is_noop(settings):
    assert not stop_background(settings, "127.0.0.1", 8742)


def test_stale_pid_state_is_removed_without_signalling(settings, monkeypatch):
    path = save_state(settings, pid=12345)
    monkeypatch.setattr("service.service.process_alive", lambda pid: False)
    assert not stop_background(settings, "127.0.0.1", 8742)
    assert not path.exists()


def test_stop_never_signals_unverified_live_pid(settings):
    path = save_state(settings, pid=os.getpid())
    with pytest.raises(ServiceError, match="still starting"):
        stop_background(settings, "127.0.0.1", 8742)
    assert path.is_file()


def test_timeout_asks_owned_supervisor_and_waits_for_exit(settings, monkeypatch):
    path = save_state(settings, pid=12345, control_port=1, supervisor_pid=123, supervisor_port=2)
    monkeypatch.setattr("service.service.process_alive", lambda pid: True)
    monkeypatch.setattr("service.service.control", Mock(side_effect=ServiceError("unreachable")))
    owner = Mock(side_effect=[False, True])
    monkeypatch.setattr("service.service.owner_status", owner)
    assert stop_background(settings, "127.0.0.1", 8742)
    assert [call.kwargs["stop"] for call in owner.call_args_list] == [False, True]
    assert not path.exists()


def test_graceful_timeout_escalates_through_supervisor(settings, monkeypatch):
    path = save_state(settings, pid=12345, control_port=1, supervisor_pid=123, supervisor_port=2)
    clock = iter([0, settings.development.shutdown_timeout])
    monkeypatch.setattr("service.service.time.monotonic", lambda: next(clock))
    monkeypatch.setattr("service.service.process_alive", lambda pid: True)
    monkeypatch.setattr("service.service.control", Mock(return_value=True))
    owner = Mock(side_effect=[False, True])
    monkeypatch.setattr("service.service.owner_status", owner)
    assert stop_background(settings, "127.0.0.1", 8742)
    assert [call.kwargs["stop"] for call in owner.call_args_list] == [False, True]
    assert not path.exists()


@pytest.mark.parametrize("replacement", [None, "replacement-token"])
def test_lost_state_does_not_prove_exit_or_allow_escalation(settings, monkeypatch, replacement):
    path = save_state(settings, pid=12345, control_port=1, supervisor_pid=123, supervisor_port=2)
    monkeypatch.setattr("service.service.process_alive", lambda pid: True)

    def control_failure(*args, **kwargs):
        if replacement is None:
            path.unlink()
        else:
            save_state(settings, pid=12346, token=replacement)
        raise ServiceError("unreachable")

    monkeypatch.setattr("service.service.control", control_failure)
    owner = Mock(return_value=False)
    monkeypatch.setattr("service.service.owner_status", owner)
    with pytest.raises(ServiceError, match="state changed"):
        stop_background(settings, "127.0.0.1", 8742)
    assert all(not call.kwargs["stop"] for call in owner.call_args_list)
    if replacement is not None:
        assert read_state(path).token == replacement


def test_old_process_exit_does_not_remove_replacement_state(settings, monkeypatch):
    path = save_state(settings, pid=12345, control_port=1)
    alive = iter([True, False])
    monkeypatch.setattr("service.service.process_alive", lambda pid: next(alive))

    def replace(*args, **kwargs):
        save_state(settings, pid=12346, token="new-instance")
        return True

    monkeypatch.setattr("service.service.control", replace)
    assert stop_background(settings, "127.0.0.1", 8742)
    assert read_state(path).token == "new-instance"


def test_cleanup_stops_all_registered_endpoints_under_one_lock(settings, monkeypatch):
    first = save_state(settings, pid=111)
    second = first.with_name(".oracle-service-another-endpoint.json")
    second.write_text(ServiceState(token="other", launcher_pid=1, pid=222).model_dump_json())
    stopped = []

    def stop(selected, path, state):
        stopped.append(state.pid)
        return True

    monkeypatch.setattr("service.service._stop_state", stop)
    with stopped_services(settings):
        assert sorted(stopped) == [111, 222]
        with pytest.raises(ServiceError, match="lifecycle operation"):
            with lifecycle_lock(settings.index_path.parent):
                pytest.fail("managed launch could race cleanup")


def test_cleanup_corrupt_state_prevents_deletion_body(settings):
    path = save_state(settings)
    path.write_text("invalid json")
    with pytest.raises(ServiceError, match="Cannot read service state"):
        with stopped_services(settings):
            pytest.fail("cleanup permitted with unreadable service state")


def test_shutdown_failure_keeps_state(settings, monkeypatch):
    path = save_state(settings, pid=12345, control_port=1, supervisor_pid=123, supervisor_port=2)
    monkeypatch.setattr("service.service.process_alive", lambda pid: True)
    monkeypatch.setattr("service.service.control", Mock(side_effect=ServiceError("unreachable")))
    monkeypatch.setattr("service.service.owner_status", Mock(return_value=False))
    with pytest.raises(ServiceError, match="has not exited"):
        stop_background(settings, "127.0.0.1", 8742)
    assert path.exists()


def test_control_checks_identity_before_requesting_stop(monkeypatch):
    methods = []

    def respond(request):
        methods.append(request.method)
        return httpx.Response(200, json={"pid": 99999, "started": True})

    client = httpx.Client(transport=httpx.MockTransport(respond))
    monkeypatch.setattr("service.service.httpx.Client", lambda **kwargs: client)
    with pytest.raises(ServiceError, match="identity does not match"):
        control(ServiceState(token="secret", launcher_pid=1, pid=123, control_port=2), stop=True)
    assert methods == ["GET"]


@pytest.mark.parametrize("denied", [False, True])
def test_windows_lifecycle_uses_nonblocking_byte_lock(settings, monkeypatch, denied):
    locking = Mock(side_effect=OSError("busy") if denied else None)
    monkeypatch.setitem(sys.modules, "msvcrt", SimpleNamespace(locking=locking, LK_NBLCK=2))
    monkeypatch.setattr(sys, "platform", "win32")
    if denied:
        with pytest.raises(ServiceError, match="lifecycle operation"):
            with lifecycle_lock(settings.index_path.parent):
                pytest.fail("entered a busy lifecycle lock")
    else:
        with lifecycle_lock(settings.index_path.parent):
            assert (settings.index_path.parent / ".oracle-lifecycle.lock").read_bytes() == b"\0"
    assert locking.call_args.args[1:] == (2, 1)


def test_lifecycle_lock_open_failure_has_actionable_error(settings, monkeypatch):
    monkeypatch.setattr("service.service.os.open", Mock(side_effect=PermissionError("denied")))
    with pytest.raises(ServiceError, match="Cannot open Oracle lifecycle lock"):
        with lifecycle_lock(settings.index_path.parent):
            pytest.fail("entered a lock without permission")


def test_other_launch_cannot_remove_state(settings):
    path = save_state(settings)
    remove_state(path, "different-secret")
    assert path.is_file()
    remove_state(path, "test-secret")
    assert not path.exists()


def test_duplicate_launch_does_not_replace_live_state(settings, tmp_path):
    path = save_state(settings, pid=os.getpid())
    before = path.read_bytes()
    with pytest.raises(ServiceError, match="already running"):
        start_background([sys.executable, "-c", "pass"], os.environ, settings,
                         "127.0.0.1", 8742, tmp_path / "server.log")
    assert path.read_bytes() == before


def test_failed_child_start_cleans_state_and_preserves_log(settings, tmp_path):
    log = tmp_path / "server.log"
    with pytest.raises(ServiceError, match="exited with code 7"):
        start_background([sys.executable, "-c", "print('startup failed'); raise SystemExit(7)"],
                         os.environ, settings, "127.0.0.1", 8742, log)
    assert "startup failed" in log.read_text()
    assert not state_path(settings, "127.0.0.1", 8742).exists()


def test_corrupted_state_returns_actionable_error(settings):
    path = save_state(settings)
    path.write_text("invalid json")
    with pytest.raises(ServiceError, match="Cannot read service state"):
        read_state(path)


def test_status_absent_service_does_not_create_state(settings):
    assert status_background(settings, "127.0.0.1", 8742).state == "stopped"
    assert not state_path(settings, "127.0.0.1", 8742).exists()


@pytest.mark.parametrize("pid", [None, 12345])
def test_status_dead_process_preserves_stale_state(settings, monkeypatch, pid):
    path = save_state(settings, pid=pid, control_port=1234)
    before = path.read_bytes()
    monkeypatch.setattr("service.service.process_alive", lambda pid: False)
    monkeypatch.setattr(
        "service.service.control_status", lambda state: pytest.fail("contacted dead PID")
    )
    assert status_background(settings, "127.0.0.1", 8742).state == "stopped"
    assert path.read_bytes() == before


def test_status_launch_without_control_endpoint_is_starting(settings):
    path = save_state(settings, log_path="/actual/startup.log")
    before = path.read_bytes()
    result = status_background(settings, "127.0.0.1", 8742)
    assert result.state == "starting"
    assert result.pid is None  # The launcher PID is not the server PID.
    assert result.log_path == "/actual/startup.log"
    assert path.read_bytes() == before


@pytest.mark.parametrize("started, expected", [(True, "running"), (False, "starting")])
def test_status_verifies_authenticated_readiness(settings, monkeypatch, started, expected):
    path = save_state(settings, pid=os.getpid(), control_port=1234)
    before = path.read_bytes()

    def respond(request):
        assert request.method == "GET"
        assert str(request.url) == "http://127.0.0.1:1234/status"
        assert request.headers["Authorization"] == "Bearer test-secret"
        return httpx.Response(200, json={"pid": os.getpid(), "started": started})

    client = httpx.Client(transport=httpx.MockTransport(respond))
    monkeypatch.setattr("service.service.httpx.Client", lambda **kwargs: client)
    result = status_background(settings, "127.0.0.1", 8742)
    assert result.state == expected
    assert result.pid == os.getpid()
    assert result.uptime_seconds is None  # Older servers do not report diagnostics.
    assert result.documents is None
    assert result.index_location is None
    assert path.read_bytes() == before


def test_status_reports_live_diagnostics_without_changing_state(settings, monkeypatch):
    path = save_state(settings, pid=os.getpid(), control_port=1234, log_path="launch.log")
    before = path.read_bytes()
    payload = {
        "pid": os.getpid(), "started": True, "uptime_seconds": 90061.5,
        "documents": 12, "chunks": 345, "dimensions": 256, "model": "loaded-model",
        "index_location": "/loaded/explicit.npz", "log_path": "/actual/override.log",
    }
    client = httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json=payload)
    ))
    monkeypatch.setattr("service.service.httpx.Client", lambda **kwargs: client)
    result = status_background(settings, "127.0.0.1", 8742)
    assert result.uptime_seconds == 90061.5
    assert (result.documents, result.chunks, result.dimensions) == (12, 345, 256)
    assert result.model == "loaded-model"
    assert result.index_location == "/loaded/explicit.npz"
    assert result.log_path == "/actual/override.log"
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "extra", [{"uptime_seconds": -1}, {"uptime_seconds": float("nan")},
              {"documents": -1}, {"chunks": "123"}, {"dimensions": 0}]
)
def test_status_rejects_invalid_diagnostics(settings, monkeypatch, extra):
    save_state(settings, pid=os.getpid(), control_port=1234)
    payload = {"pid": os.getpid(), "started": True, **extra}
    client = httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, content=json.dumps(payload))
    ))
    monkeypatch.setattr("service.service.httpx.Client", lambda **kwargs: client)
    with pytest.raises(ServiceError, match="Invalid response"):
        status_background(settings, "127.0.0.1", 8742)


@pytest.mark.parametrize("started, exiting", [(True, False), (False, False), (True, True)])
def test_real_control_endpoint_reports_diagnostics_and_requires_auth(
    settings, started, exiting
):
    path = save_state(settings, log_path="/actual/server.log")
    index_location = settings.index_path.parent / "explicit.npz"
    stats = IndexStats(documents=2, chunks=10, dimensions=256, model="test-model", sources=[])

    class Server:
        should_exit = exiting

        def __init__(self):
            self.started = started

        def run(self):
            before = path.read_bytes()
            result = status_background(settings, "127.0.0.1", 8742)
            expected = "stopping" if exiting else "running" if started else "starting"
            assert result.state == expected
            assert result.pid == os.getpid()
            assert result.uptime_seconds >= 0
            assert (result.documents, result.chunks, result.dimensions) == (2, 10, 256)
            assert result.model == "test-model"
            assert result.index_location == str(index_location)
            assert result.log_path == "/actual/server.log"
            current = read_state(path)
            with httpx.Client(trust_env=False) as client:
                response = client.get(f"http://127.0.0.1:{current.control_port}/status")
            assert response.status_code == 401
            assert "test-model" not in response.text
            assert path.read_bytes() == before

    run_with_control(
        Server(), path, "test-secret", index_stats=stats, index_location=index_location,
    )
    assert not path.exists()


def test_launcher_records_actual_log_override(settings, tmp_path, monkeypatch):
    log_path = tmp_path / "custom.log"

    def spawn(command, environment, logs):
        state = read_state(state_path(settings, "127.0.0.1", 8742))
        assert state.log_path == str(log_path.resolve())
        assert logs.name == str(log_path)
        return _ExitedProcess()

    monkeypatch.setattr("service.service._spawn_detached", spawn)
    with pytest.raises(ServiceError, match="exited with code 7"):
        start_background(
            ["python", "-m", "cli"], {}, settings, "127.0.0.1", 8742, log_path,
        )


@pytest.mark.parametrize(
    "payload, error",
    [
        ({"pid": os.getpid() + 1, "started": True}, "identity does not match"),
        ([], "Invalid response"),
        ({"pid": os.getpid(), "started": "false"}, "Invalid response"),
        ({"pid": os.getpid()}, "Invalid response"),
        ({"started": True}, "Invalid response"),
    ],
)
def test_status_rejects_unverified_control_responses(settings, monkeypatch, payload, error):
    path = save_state(settings, pid=os.getpid(), control_port=1234)
    before = path.read_bytes()
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    client = httpx.Client(transport=transport)
    monkeypatch.setattr("service.service.httpx.Client", lambda **kwargs: client)
    with pytest.raises(ServiceError, match=error):
        status_background(settings, "127.0.0.1", 8742)
    assert path.read_bytes() == before


@pytest.mark.parametrize("status_code", [401, 500])
def test_status_control_http_failure_is_not_readiness(settings, monkeypatch, status_code):
    save_state(settings, pid=os.getpid(), control_port=1234)
    transport = httpx.MockTransport(lambda request: httpx.Response(status_code))
    client = httpx.Client(transport=transport)
    monkeypatch.setattr("service.service.httpx.Client", lambda **kwargs: client)
    with pytest.raises(ServiceError, match="Cannot contact managed Oracle server"):
        status_background(settings, "127.0.0.1", 8742)


def test_status_control_connection_failure(settings, monkeypatch):
    save_state(settings, pid=os.getpid(), control_port=1234)

    def refuse(request):
        raise httpx.ConnectError("connection refused", request=request)

    client = httpx.Client(transport=httpx.MockTransport(refuse))
    monkeypatch.setattr("service.service.httpx.Client", lambda **kwargs: client)
    with pytest.raises(ServiceError, match="connection refused"):
        status_background(settings, "127.0.0.1", 8742)


def test_control_rejects_missing_service_pid(monkeypatch):
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"pid": os.getpid(), "started": True})
    )
    client = httpx.Client(transport=transport)
    monkeypatch.setattr("service.service.httpx.Client", lambda **kwargs: client)
    with pytest.raises(ServiceError, match="identity does not match"):
        control(ServiceState(token="secret", launcher_pid=os.getpid(), control_port=1234))


@pytest.mark.parametrize("wait_result, alive", [(258, True), (0, False)])
def test_windows_process_check_polls_handle_without_signalling(monkeypatch, wait_result, alive):
    kernel32 = Mock()
    kernel32.OpenProcess.return_value = 123
    kernel32.WaitForSingleObject.return_value = wait_result
    loader = Mock(return_value=kernel32)
    monkeypatch.setattr("service.service.ctypes.WinDLL", loader, raising=False)
    monkeypatch.setattr("service.service.ctypes.get_last_error", lambda: 0, raising=False)
    monkeypatch.setattr("service.service.os.name", "nt")
    monkeypatch.setattr("service.service.os.kill", Mock(side_effect=AssertionError("sent signal")))
    assert process_alive(456) is alive
    kernel32.OpenProcess.assert_called_once_with(0x00100000, False, 456)
    kernel32.WaitForSingleObject.assert_called_once_with(123, 0)
    kernel32.CloseHandle.assert_called_once_with(123)


@pytest.mark.parametrize(
    "error, alive", [(None, True), (ProcessLookupError(), False), (PermissionError(), True)]
)
def test_posix_process_check_uses_signal_zero(monkeypatch, error, alive):
    kill = Mock(side_effect=error)
    monkeypatch.setattr("service.service.os.name", "posix")
    monkeypatch.setattr("service.service.os.kill", kill)
    assert process_alive(456) is alive
    kill.assert_called_once_with(456, 0)


@pytest.mark.parametrize("pid", [None, 0, -1])
def test_process_check_ignores_invalid_pid(monkeypatch, pid):
    monkeypatch.setattr("service.service.os.kill", Mock(side_effect=AssertionError("sent signal")))
    assert not process_alive(pid)


@pytest.mark.parametrize("error, alive", [(87, False), (5, True), (6, None)])
def test_windows_process_check_open_failure(monkeypatch, error, alive):
    kernel32 = Mock()
    kernel32.OpenProcess.return_value = None
    monkeypatch.setattr("service.service.ctypes.WinDLL", Mock(return_value=kernel32), raising=False)
    monkeypatch.setattr("service.service.ctypes.get_last_error", lambda: error, raising=False)
    if alive is None:
        with pytest.raises(ServiceError, match="Windows error 6"):
            _windows_process_alive(456)
    else:
        assert _windows_process_alive(456) is alive
    kernel32.CloseHandle.assert_not_called()


def test_windows_process_check_closes_handle_after_wait_failure(monkeypatch):
    kernel32 = Mock()
    kernel32.OpenProcess.return_value = 123
    kernel32.WaitForSingleObject.return_value = 0xFFFFFFFF
    monkeypatch.setattr("service.service.ctypes.WinDLL", Mock(return_value=kernel32), raising=False)
    monkeypatch.setattr("service.service.ctypes.get_last_error", lambda: 6, raising=False)
    with pytest.raises(ServiceError, match="Windows error 6"):
        _windows_process_alive(456)
    kernel32.CloseHandle.assert_called_once_with(123)


NO_WINDOW = 0x08000000
NEW_PROCESS_GROUP = 0x00000200
BREAKAWAY_FROM_JOB = 0x01000000


@pytest.fixture
def windows_constants(monkeypatch):
    for name, value in (
        ("CREATE_NEW_PROCESS_GROUP", NEW_PROCESS_GROUP),
        ("CREATE_NO_WINDOW", NO_WINDOW),
        ("CREATE_BREAKAWAY_FROM_JOB", BREAKAWAY_FROM_JOB),
        ("STARTF_USESHOWWINDOW", 1),
        ("SW_HIDE", 0),
    ):
        monkeypatch.setattr(subprocess, name, value, raising=False)
    if not hasattr(subprocess, "STARTUPINFO"):
        class StartupInfo:
            dwFlags = 0
            wShowWindow = 1

        monkeypatch.setattr(subprocess, "STARTUPINFO", StartupInfo, raising=False)


def test_windows_spawn_options_are_hidden_and_breakaway_is_optional(windows_constants):
    primary, fallback, startupinfo = _windows_spawn_options()

    assert primary == NO_WINDOW | NEW_PROCESS_GROUP | BREAKAWAY_FROM_JOB
    assert fallback == NO_WINDOW | NEW_PROCESS_GROUP
    assert startupinfo.dwFlags & 1
    assert startupinfo.wShowWindow == 0


def test_windows_spawn_options_use_detached_process_without_no_window(
    windows_constants, monkeypatch
):
    monkeypatch.delattr(subprocess, "CREATE_NO_WINDOW")
    monkeypatch.setattr(subprocess, "DETACHED_PROCESS", 0x8, raising=False)

    _, fallback, _ = _windows_spawn_options()

    assert fallback == NEW_PROCESS_GROUP | 0x8


class _ExitedProcess:
    pid = 9999
    returncode = 7

    def poll(self):
        return 7


def _start(settings, tmp_path):
    return start_background(
        ["python", "-m", "cli", "serve", "--foreground"],
        os.environ, settings, "127.0.0.1", 8742, tmp_path / "server.log",
    )


def test_start_background_on_windows_requests_hidden_breakaway(
    settings, tmp_path, monkeypatch, windows_constants
):
    calls: list[dict] = []

    def fake_popen(*args, **kwargs):
        calls.append(kwargs)
        return _ExitedProcess()

    monkeypatch.setattr("service.service.os.name", "nt")
    monkeypatch.setattr("service.service.subprocess.Popen", fake_popen)

    with pytest.raises(ServiceError, match="exited with code 7"):
        _start(settings, tmp_path)

    assert len(calls) == 1
    assert calls[0]["creationflags"] == NO_WINDOW | NEW_PROCESS_GROUP | BREAKAWAY_FROM_JOB
    assert calls[0]["startupinfo"].wShowWindow == 0
    assert "start_new_session" not in calls[0]


def test_start_background_falls_back_when_job_breakaway_denied(
    settings, tmp_path, monkeypatch, windows_constants
):
    calls: list[dict] = []

    def fake_popen(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise PermissionError("Access is denied (job breakaway not permitted)")
        return _ExitedProcess()

    monkeypatch.setattr("service.service.os.name", "nt")
    monkeypatch.setattr("service.service.subprocess.Popen", fake_popen)

    with pytest.raises(ServiceError, match="exited with code 7"):
        _start(settings, tmp_path)

    assert [call["creationflags"] for call in calls] == [
        NO_WINDOW | NEW_PROCESS_GROUP | BREAKAWAY_FROM_JOB,
        NO_WINDOW | NEW_PROCESS_GROUP,
    ]
    assert calls[1]["startupinfo"].wShowWindow == 0
