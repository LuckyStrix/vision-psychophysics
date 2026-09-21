"""Tests for `vpsych.app.viewmodels.run`: status polling and exit-code mapping."""

from __future__ import annotations

from pathlib import Path

import pytest

from vpsych.app.viewmodels.run import (
    explain_exit,
    progress_fraction,
    progress_text,
    read_status_safely,
)
from vpsych.runner.status import RunnerExitCode, RunnerStatus


def test_read_status_safely_missing_file_returns_none(tmp_path: Path) -> None:
    assert read_status_safely(tmp_path / "nope.json") is None


def test_read_status_safely_reads_written_status(tmp_path: Path) -> None:
    status = RunnerStatus.create(state="running", current_test_index=0, n_tests=2)
    path = tmp_path / "status.json"
    status.write_atomic(path)
    read_back = read_status_safely(path)
    assert read_back is not None
    assert read_back.state == "running"


def test_progress_fraction_none_when_n_tests_unknown() -> None:
    status = RunnerStatus.create(state="starting", current_test_index=0, n_tests=0)
    assert progress_fraction(status) is None


def test_progress_fraction_combines_test_and_trial_progress() -> None:
    status = RunnerStatus.create(
        state="running",
        current_test_index=1,
        n_tests=4,
        trial_index=5,
        n_trials_expected=10,
    )
    # test 1 of 4 => base 0.25; within-test 5/10 of one quarter => +0.125
    assert progress_fraction(status) == pytest.approx(0.375)


def test_progress_fraction_without_trial_detail_uses_test_index_only() -> None:
    status = RunnerStatus.create(state="running", current_test_index=2, n_tests=4)
    assert progress_fraction(status) == pytest.approx(0.5)


def test_progress_text_includes_task_and_trial() -> None:
    status = RunnerStatus.create(
        state="running",
        current_test_index=0,
        n_tests=3,
        task_id="visual_acuity",
        trial_index=4,
        n_trials_expected=20,
    )
    text = progress_text(status)
    assert "visual_acuity" in text
    assert "trial 5" in text
    assert "test 1 of 3" in text


def test_progress_text_falls_back_to_state_label() -> None:
    status = RunnerStatus.create(state="starting", current_test_index=0, n_tests=0)
    assert progress_text(status) == "Starting"


@pytest.mark.parametrize(
    "code",
    [
        RunnerExitCode.OK,
        RunnerExitCode.ERROR,
        RunnerExitCode.ABORTED_BY_USER,
        RunnerExitCode.REFRESH_MISMATCH,
        RunnerExitCode.REQUIREMENTS_UNMET,
    ],
)
def test_explain_exit_covers_every_known_exit_code_distinctly(code: RunnerExitCode) -> None:
    outcome = explain_exit(code, None)
    assert outcome.title
    assert outcome.explanation


def test_explain_exit_ok_is_success_and_retains_data() -> None:
    outcome = explain_exit(RunnerExitCode.OK, None)
    assert outcome.is_success is True
    assert outcome.data_retained is True


def test_explain_exit_refresh_mismatch_never_retains_data() -> None:
    outcome = explain_exit(RunnerExitCode.REFRESH_MISMATCH, None)
    assert outcome.is_success is False
    assert outcome.data_retained is False
    assert "refresh" in outcome.explanation.lower()


def test_explain_exit_requirements_unmet_explains_and_includes_detail() -> None:
    status = RunnerStatus.create(
        state="error", error="dummy_test: Needs a gamma calibration (currently uncalibrated)."
    )
    outcome = explain_exit(RunnerExitCode.REQUIREMENTS_UNMET, status)
    assert outcome.data_retained is False
    assert "gamma calibration" in outcome.explanation


def test_explain_exit_error_retains_data_and_includes_detail() -> None:
    status = RunnerStatus.create(state="error", error="boom")
    outcome = explain_exit(RunnerExitCode.ERROR, status)
    assert outcome.data_retained is True
    assert "boom" in outcome.explanation


def test_explain_exit_aborted_by_user() -> None:
    outcome = explain_exit(RunnerExitCode.ABORTED_BY_USER, None)
    assert outcome.data_retained is True
    assert outcome.is_success is False


def test_every_distinct_exit_code_gets_a_distinct_title() -> None:
    codes = [
        RunnerExitCode.OK,
        RunnerExitCode.ERROR,
        RunnerExitCode.ABORTED_BY_USER,
        RunnerExitCode.REFRESH_MISMATCH,
        RunnerExitCode.REQUIREMENTS_UNMET,
    ]
    titles = {explain_exit(c, None).title for c in codes}
    assert len(titles) == len(codes)
