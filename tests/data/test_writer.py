"""Tests for vpsych.data.writer.SessionWriter."""

from __future__ import annotations

import contextlib
import csv
import json
import os
import stat
from pathlib import Path

import pytest

from vpsych.core.trial import TrialRecord
from vpsych.data import paths
from vpsych.data.writer import (
    TRIAL_COLUMN_SIDECARS,
    SessionAlreadyFinalizedError,
    SessionWriter,
)

from .conftest import make_session_info, make_trial


@pytest.fixture
def cal_hash() -> str:
    return "deadbeef" * 8


def _chmod_writable_recursive(root: Path) -> None:
    for p in root.rglob("*"):
        with contextlib.suppress(OSError):
            os.chmod(p, 0o755 if p.is_dir() else 0o644)
    with contextlib.suppress(OSError):
        os.chmod(root, 0o755)


def test_trial_column_sidecars_cover_every_trialrecord_field() -> None:
    assert set(TRIAL_COLUMN_SIDECARS) == set(TrialRecord.model_fields)


def test_append_trial_writes_header_once_and_flushes(tmp_path: Path, cal_hash: str) -> None:
    info = make_session_info("sub-0001", "ses-20260916T103000", cal_hash)
    with SessionWriter("sub-0001", "ses-20260916T103000", info, tmp_path) as writer:
        writer.append_trial(make_trial(trial_index=0))
        writer.append_trial(make_trial(trial_index=1))

    tsv_path = paths.trials_tsv_path(
        "sub-0001", "ses-20260916T103000", "dummy_test", "OD", 1, tmp_path
    )
    lines = tsv_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3  # header + 2 rows
    assert lines[0].split("\t")[0] == "participant_id"

    _chmod_writable_recursive(tmp_path)


def test_append_trial_writes_sidecar_once(tmp_path: Path, cal_hash: str) -> None:
    info = make_session_info("sub-0001", "ses-20260916T103000", cal_hash)
    with SessionWriter("sub-0001", "ses-20260916T103000", info, tmp_path) as writer:
        writer.append_trial(make_trial(trial_index=0))
        writer.append_trial(make_trial(trial_index=1))

    sidecar_path = paths.trials_sidecar_path(
        "sub-0001", "ses-20260916T103000", "dummy_test", "OD", 1, tmp_path
    )
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert set(sidecar) == set(TrialRecord.model_fields)
    assert sidecar["intensity"]["description"]

    _chmod_writable_recursive(tmp_path)


def test_write_summary_and_frames(tmp_path: Path, cal_hash: str) -> None:
    from vpsych.core.procedures.base import ThresholdEstimate
    from vpsych.data.schemas import TestSummary

    info = make_session_info("sub-0001", "ses-20260916T103000", cal_hash)
    summary = TestSummary(
        task_id="dummy_test",
        task_version="1.0.0",
        eye="OD",
        run=1,
        estimate=ThresholdEstimate(
            value=0.1, ci_low=0.0, ci_high=0.2, ci_level=0.95, units="logMAR", method="mean"
        ),
        n_trials=5,
        n_catch=1,
        catch_lapse_rate=0.0,
        analysis_version="dummy-1.0.0",
    )
    with SessionWriter("sub-0001", "ses-20260916T103000", info, tmp_path) as writer:
        writer.append_trial(make_trial())
        writer.write_summary("dummy_test", "OD", 1, summary)
        writer.write_frames("dummy_test", "OD", 1, [0.0166, 0.0167])
        writer.write_frames("dummy_test", "OD", 1, [0.0168])

    summary_path = paths.summary_json_path(
        "sub-0001", "ses-20260916T103000", "dummy_test", "OD", 1, tmp_path
    )
    loaded = TestSummary.model_validate_json(summary_path.read_text(encoding="utf-8"))
    assert loaded == summary

    frames_path = paths.frames_tsv_path(
        "sub-0001", "ses-20260916T103000", "dummy_test", "OD", 1, tmp_path
    )
    with open(frames_path, encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f, delimiter="\t"))
    assert rows[0] == ["frame_interval_s"]
    assert len(rows) == 4  # header + 3 intervals across two calls

    _chmod_writable_recursive(tmp_path)


def test_exit_without_exception_finalizes_complete(tmp_path: Path, cal_hash: str) -> None:
    info = make_session_info("sub-0001", "ses-20260916T103000", cal_hash)
    with SessionWriter("sub-0001", "ses-20260916T103000", info, tmp_path) as writer:
        writer.append_trial(make_trial())
    assert writer.session_info.status == "complete"
    assert writer.session_info.ended_utc is not None

    _chmod_writable_recursive(tmp_path)


def test_exit_with_exception_finalizes_incomplete_and_keeps_data(
    tmp_path: Path, cal_hash: str
) -> None:
    info = make_session_info("sub-0001", "ses-20260916T103000", cal_hash)
    with (
        pytest.raises(RuntimeError),
        SessionWriter("sub-0001", "ses-20260916T103000", info, tmp_path) as writer,
    ):
        writer.append_trial(make_trial())
        raise RuntimeError("boom")

    assert writer.session_info.status == "incomplete"
    tsv_path = paths.trials_tsv_path(
        "sub-0001", "ses-20260916T103000", "dummy_test", "OD", 1, tmp_path
    )
    assert tsv_path.exists()
    lines = tsv_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2  # header + the one trial appended before the exception

    _chmod_writable_recursive(tmp_path)


def test_finalize_writes_manifest_covering_every_file(tmp_path: Path, cal_hash: str) -> None:
    info = make_session_info("sub-0001", "ses-20260916T103000", cal_hash)
    with SessionWriter("sub-0001", "ses-20260916T103000", info, tmp_path) as writer:
        writer.append_trial(make_trial())

    session_dir = paths.session_dir("sub-0001", "ses-20260916T103000", tmp_path)
    manifest_path = paths.session_manifest_path("sub-0001", "ses-20260916T103000", tmp_path)
    assert manifest_path.exists()
    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    listed_paths = {line.split("  ", 1)[1] for line in lines}
    on_disk = {
        p.relative_to(session_dir).as_posix()
        for p in session_dir.rglob("*")
        if p.is_file() and p != manifest_path
    }
    assert listed_paths == on_disk
    for line in lines:
        digest, _, _ = line.partition("  ")
        assert len(digest) == 64

    _chmod_writable_recursive(tmp_path)


def test_finalize_readonly_by_default(tmp_path: Path, cal_hash: str) -> None:
    info = make_session_info("sub-0001", "ses-20260916T103000", cal_hash)
    with SessionWriter("sub-0001", "ses-20260916T103000", info, tmp_path) as writer:
        writer.append_trial(make_trial())

    tsv_path = paths.trials_tsv_path(
        "sub-0001", "ses-20260916T103000", "dummy_test", "OD", 1, tmp_path
    )
    mode = stat.S_IMODE(tsv_path.stat().st_mode)
    assert mode == 0o444

    _chmod_writable_recursive(tmp_path)


def test_finalize_readonly_configurable_off(tmp_path: Path, cal_hash: str) -> None:
    info = make_session_info("sub-0001", "ses-20260916T103000", cal_hash)
    with SessionWriter(
        "sub-0001", "ses-20260916T103000", info, tmp_path, readonly_after_finalize=False
    ) as writer:
        writer.append_trial(make_trial())

    tsv_path = paths.trials_tsv_path(
        "sub-0001", "ses-20260916T103000", "dummy_test", "OD", 1, tmp_path
    )
    mode = stat.S_IMODE(tsv_path.stat().st_mode)
    assert mode != 0o444


def test_refuses_to_overwrite_finalized_session(tmp_path: Path, cal_hash: str) -> None:
    info = make_session_info("sub-0001", "ses-20260916T103000", cal_hash)
    with SessionWriter("sub-0001", "ses-20260916T103000", info, tmp_path) as writer:
        writer.append_trial(make_trial())

    info2 = make_session_info("sub-0001", "ses-20260916T103000", cal_hash)
    with pytest.raises(SessionAlreadyFinalizedError):
        SessionWriter("sub-0001", "ses-20260916T103000", info2, tmp_path).__enter__()

    _chmod_writable_recursive(tmp_path)


def test_finalize_running_status_raises(tmp_path: Path, cal_hash: str) -> None:
    info = make_session_info("sub-0001", "ses-20260916T103000", cal_hash)
    writer = SessionWriter(
        "sub-0001", "ses-20260916T103000", info, tmp_path, readonly_after_finalize=False
    )
    writer.__enter__()
    with pytest.raises(ValueError, match="running"):
        writer.finalize("running")
    writer.finalize("aborted")


def test_double_finalize_raises(tmp_path: Path, cal_hash: str) -> None:
    info = make_session_info("sub-0001", "ses-20260916T103000", cal_hash)
    writer = SessionWriter(
        "sub-0001", "ses-20260916T103000", info, tmp_path, readonly_after_finalize=False
    )
    writer.__enter__()
    writer.finalize("complete")
    with pytest.raises(SessionAlreadyFinalizedError):
        writer.finalize("complete")


def test_append_trial_after_finalize_raises(tmp_path: Path, cal_hash: str) -> None:
    info = make_session_info("sub-0001", "ses-20260916T103000", cal_hash)
    with SessionWriter("sub-0001", "ses-20260916T103000", info, tmp_path) as writer:
        writer.append_trial(make_trial())

    with pytest.raises(SessionAlreadyFinalizedError):
        writer.append_trial(make_trial(trial_index=1))

    _chmod_writable_recursive(tmp_path)
