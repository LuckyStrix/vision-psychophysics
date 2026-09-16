"""Unit tests for vpsych.runner.status: atomic status file read/write."""

from __future__ import annotations

from pathlib import Path

from vpsych.runner.status import RunnerExitCode, RunnerStatus


def test_create_sets_timestamp() -> None:
    status = RunnerStatus.create(state="starting", n_tests=3)
    assert status.state == "starting"
    assert status.n_tests == 3
    assert status.updated_utc.tzinfo is not None


def test_write_atomic_then_read_roundtrip(tmp_path: Path) -> None:
    status = RunnerStatus.create(
        state="running", current_test_index=1, n_tests=3, task_id="acuity", trial_index=5
    )
    path = tmp_path / "status.json"
    status.write_atomic(path)
    assert path.exists()
    read_back = RunnerStatus.read(path)
    assert read_back == status


def test_write_atomic_creates_parent_dirs(tmp_path: Path) -> None:
    status = RunnerStatus.create(state="starting")
    path = tmp_path / "nested" / "dir" / "status.json"
    status.write_atomic(path)
    assert path.exists()


def test_write_atomic_leaves_no_tmp_file_on_success(tmp_path: Path) -> None:
    status = RunnerStatus.create(state="finished")
    path = tmp_path / "status.json"
    status.write_atomic(path)
    leftovers = list(tmp_path.glob(".*"))
    assert leftovers == []


def test_exit_code_values() -> None:
    assert RunnerExitCode.OK == 0
    assert RunnerExitCode.ERROR == 1
    assert RunnerExitCode.ABORTED_BY_USER == 2
    assert RunnerExitCode.REFRESH_MISMATCH == 3
    assert RunnerExitCode.REQUIREMENTS_UNMET == 4
