import os
import subprocess
from unittest.mock import Mock

import pytest

from service.service import _escalate_stop, _wait_for_exit
from service.state import ServiceError, ServiceState, read_state, state_path
from service.supervisor import stop_owned_process


class FakeOwner:
    """In-memory stand-in for the supervisor port; records calls, never touches HTTP."""

    def __init__(self, *exited: bool):
        self.exited = list(exited)
        self.calls: list[bool] = []

    def __call__(self, state, *, stop, timeout):
        self.calls.append(stop)
        return self.exited.pop(0)


def supervised_state(settings):
    path = state_path(settings, "127.0.0.1", 8742)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = ServiceState(
        token="secret", launcher_pid=os.getpid(), pid=12345, control_port=1,
        supervisor_pid=123, supervisor_port=2,
    )
    path.write_text(state.model_dump_json())
    return path, state


def test_owned_process_escalates_and_waits():
    process = Mock()
    process.poll.return_value = None
    process.wait.side_effect = [subprocess.TimeoutExpired("server", 1), 0]
    stop_owned_process(process, 1)
    process.terminate.assert_called_once_with()
    process.kill.assert_called_once_with()
    assert process.wait.call_count == 2
    assert all(call.kwargs["timeout"] == 1 for call in process.wait.call_args_list)


def test_exited_owned_process_is_never_signalled():
    process = Mock()
    process.poll.return_value = 0
    stop_owned_process(process, 1)
    process.terminate.assert_not_called()
    process.kill.assert_not_called()


def test_owned_process_has_bounded_force_wait():
    process = Mock()
    process.poll.return_value = None
    process.wait.side_effect = subprocess.TimeoutExpired("server", 1)
    with pytest.raises(subprocess.TimeoutExpired):
        stop_owned_process(process, 1)


def test_owned_process_permission_failure_is_not_ignored():
    process = Mock()
    process.poll.return_value = None
    process.terminate.side_effect = PermissionError("denied")
    with pytest.raises(PermissionError):
        stop_owned_process(process, 1)
    process.kill.assert_not_called()


def test_replaced_identity_blocks_force_signal():
    process = Mock()
    process.poll.return_value = None
    process.wait.side_effect = subprocess.TimeoutExpired("server", 1)
    authorize = Mock(side_effect=[None, ServiceError("state replaced")])
    with pytest.raises(ServiceError, match="state replaced"):
        stop_owned_process(process, 1, authorize=authorize)
    process.terminate.assert_called_once_with()
    process.kill.assert_not_called()


def test_wait_reports_exit_confirmed_by_supervisor(settings, monkeypatch):
    path, state = supervised_state(settings)
    monkeypatch.setattr("service.service.process_alive", lambda pid: True)
    owner = FakeOwner(False, True)
    assert _wait_for_exit(settings, path, state, graceful_failed=False, owner=owner)
    assert owner.calls == [False, False]
    assert not path.exists()


def test_wait_gives_up_immediately_after_graceful_failure(settings, monkeypatch):
    path, state = supervised_state(settings)
    monkeypatch.setattr("service.service.process_alive", lambda pid: True)
    owner = FakeOwner(False)
    assert not _wait_for_exit(settings, path, state, graceful_failed=True, owner=owner)
    assert path.exists()


def test_escalation_stops_through_supervisor_and_removes_state(settings):
    path, state = supervised_state(settings)
    owner = FakeOwner(True)
    _escalate_stop(settings, path, state, None, owner)
    assert owner.calls == [True]
    assert not path.exists()


def test_escalation_aborts_when_child_has_not_exited(settings):
    path, state = supervised_state(settings)
    with pytest.raises(ServiceError, match="has not exited"):
        _escalate_stop(settings, path, state, None, FakeOwner(False))
    assert read_state(path) is not None


def test_escalation_refuses_without_supervisor(settings):
    path, state = supervised_state(settings)
    state.supervisor_port = None
    owner = FakeOwner()
    with pytest.raises(ServiceError, match="no verified supervisor"):
        _escalate_stop(settings, path, state, None, owner)
    assert owner.calls == []
