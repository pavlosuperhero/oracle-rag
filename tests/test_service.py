import os
import sys

import pytest

from service.service import (
    ServiceError,
    ServiceState,
    read_state,
    remove_state,
    start_background,
    state_path,
    stop_background,
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
