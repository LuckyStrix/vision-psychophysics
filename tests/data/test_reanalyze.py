"""Tests for vpsych.data.reanalyze."""

from __future__ import annotations

import json
from pathlib import Path

from vpsych.data import paths, reanalyze

from .conftest import build_full_session


def test_reanalyze_reproduces_summary_exactly(
    tmp_path: Path, registered_dummy_test: object
) -> None:
    pid, sid, _ = build_full_session(tmp_path)
    session_dir = paths.session_dir(pid, sid, tmp_path)

    results = reanalyze.reanalyze_session(session_dir, write=False)
    assert len(results) == 1
    result = results[0]
    assert result.task_id == "dummy_test"
    assert result.old_summary is not None
    assert result.matches
    assert result.differing_fields == []
    assert result.new_summary == result.old_summary


def test_reanalyze_write_creates_new_file_without_touching_original(
    tmp_path: Path, registered_dummy_test: object
) -> None:
    pid, sid, _ = build_full_session(tmp_path)
    session_dir = paths.session_dir(pid, sid, tmp_path)
    original_summary_path = paths.summary_json_path(pid, sid, "dummy_test", "OD", 1, tmp_path)
    original_mtime = original_summary_path.stat().st_mtime
    original_bytes = original_summary_path.read_bytes()

    results = reanalyze.reanalyze_session(session_dir, write=True)
    result = results[0]

    assert result.written_path is not None
    assert result.written_path.exists()
    assert result.written_path != original_summary_path
    assert "reanalysis" in result.written_path.name

    # Original file is untouched.
    assert original_summary_path.read_bytes() == original_bytes
    assert original_summary_path.stat().st_mtime == original_mtime

    from vpsych.data.schemas import TestSummary

    written = TestSummary.model_validate_json(result.written_path.read_text(encoding="utf-8"))
    assert written == result.new_summary


def test_reanalyze_no_write_creates_no_file(tmp_path: Path, registered_dummy_test: object) -> None:
    pid, sid, _ = build_full_session(tmp_path)
    session_dir = paths.session_dir(pid, sid, tmp_path)
    beh_dir = session_dir / "beh"
    before = set(beh_dir.iterdir())

    reanalyze.reanalyze_session(session_dir, write=False)

    after = set(beh_dir.iterdir())
    assert before == after


def test_reanalyze_detects_mismatch_when_summary_was_stale(
    tmp_path: Path, registered_dummy_test: object
) -> None:
    pid, sid, _ = build_full_session(tmp_path)
    summary_path = paths.summary_json_path(pid, sid, "dummy_test", "OD", 1, tmp_path)
    summary_path.chmod(0o644)
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    data["estimate"]["value"] = 999.0
    summary_path.write_text(json.dumps(data), encoding="utf-8")

    session_dir = paths.session_dir(pid, sid, tmp_path)
    results = reanalyze.reanalyze_session(session_dir, write=False)
    result = results[0]
    assert not result.matches
    assert "estimate" in result.differing_fields


def test_reanalyze_no_stored_summary(tmp_path: Path, registered_dummy_test: object) -> None:
    pid, sid, _ = build_full_session(tmp_path, write_summary=False)
    session_dir = paths.session_dir(pid, sid, tmp_path)
    results = reanalyze.reanalyze_session(session_dir, write=False)
    result = results[0]
    assert result.old_summary is None
    assert not result.matches
    assert result.differing_fields == []


def test_reanalyze_uses_the_planned_per_test_viewing_distance_not_the_stored_display(
    tmp_path: Path, registered_dummy_test: object
) -> None:
    """A test's viewing_distance_cm may differ from the session-level display geometry
    (e.g. near vs. distance testing in one session); reanalysis must reconstruct each
    test at its own planned distance, the same way run_session did, or a
    display-geometry-dependent test silently diverges from its stored summary.
    """
    from vpsych.core.trial import TrialRecord
    from vpsych.data.schemas import PlannedTest, SessionPlan
    from vpsych.data.writer import SessionWriter

    from .conftest import make_calibration, make_display, make_session_info

    sid = "ses-20260916T103000"
    calibration = make_calibration()
    from vpsych.data import dataset

    dataset.init_dataset(tmp_path)
    pid = dataset.create_participant(tmp_path).participant_id
    dataset.save_calibration(calibration, tmp_path)

    plan = SessionPlan(
        participant_id=pid,
        tests=[PlannedTest(task_id="dummy_test", eye="OD", params={}, viewing_distance_cm=100.0)],
        ordering="fixed",
        seed=42,
    )
    session_info = make_session_info(
        pid, sid, calibration.content_hash(), plan=plan, display=make_display()
    )
    assert session_info.display.viewing_distance_cm == 57.0  # deliberately != the planned 100.0

    with SessionWriter(pid, sid, session_info, tmp_path) as writer:
        writer.append_trial(
            TrialRecord.model_validate(
                {
                    "participant_id": pid,
                    "session_id": sid,
                    "task_id": "dummy_test",
                    "task_version": "1.0.0",
                    "run": 1,
                    "eye": "OD",
                    "block": "main",
                    "trial_index": 0,
                    "is_catch": False,
                    "intensity": 0.5,
                    "intensity_units": "logMAR",
                    "stimulus_params": {},
                    "correct_response": None,
                    "response": None,
                    "correct": True,
                    "rt_s": None,
                    "stimulus_onset_s": 0.0,
                    "n_dropped_frames_trial": 0,
                    "procedure_state": {},
                    "timestamp_utc": "2026-09-16T10:30:00+00:00",
                    "rng_seed": 42,
                }
            )
        )

    from vpsych.data import paths, reanalyze

    session_dir = paths.session_dir(pid, sid, tmp_path)
    results = reanalyze.reanalyze_session(session_dir, write=False)
    assert len(results) == 1
    assert results[0].new_summary.fit_params["viewing_distance_cm"] == 100.0
