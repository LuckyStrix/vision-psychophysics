"""Unit tests for vpsych.core.trial: TrialRecord.to_tsv_row serialization."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from vpsych.core.trial import TrialRecord


def _record(**overrides: object) -> TrialRecord:
    kwargs: dict[str, object] = {
        "participant_id": "sub-0001",
        "session_id": "ses-20260916T103000",
        "task_id": "acuity",
        "task_version": "0.1.0",
        "run": 1,
        "eye": "OD",
        "block": "main",
        "trial_index": 0,
        "is_catch": False,
        "intensity": -0.2,
        "intensity_units": "logMAR",
        "stimulus_params": {"orientation_deg": 90, "gap_px": 3},
        "correct_response": "right",
        "response": "right",
        "correct": True,
        "rt_s": 0.42,
        "stimulus_onset_s": 12.345,
        "n_dropped_frames_trial": 0,
        "procedure_state": {"posterior_mean": -0.2},
        "timestamp_utc": datetime(2026, 9, 16, 10, 30, 5, tzinfo=timezone.utc),
        "rng_seed": 42,
    }
    kwargs.update(overrides)
    return TrialRecord(**kwargs)  # type: ignore[arg-type]


def test_to_tsv_row_encodes_dicts_as_json() -> None:
    rec = _record()
    row = rec.to_tsv_row()
    assert json.loads(row["stimulus_params"]) == {"orientation_deg": 90, "gap_px": 3}
    assert json.loads(row["procedure_state"]) == {"posterior_mean": -0.2}


def test_to_tsv_row_missing_values_are_n_a() -> None:
    rec = _record(response=None, correct=None, rt_s=None)
    row = rec.to_tsv_row()
    assert row["response"] == "n/a"
    assert row["correct"] == "n/a"
    assert row["rt_s"] == "n/a"


def test_to_tsv_row_all_values_are_strings() -> None:
    rec = _record()
    row = rec.to_tsv_row()
    assert all(isinstance(v, str) for v in row.values())


def test_to_tsv_row_datetime_is_iso8601() -> None:
    rec = _record()
    row = rec.to_tsv_row()
    assert row["timestamp_utc"] == "2026-09-16T10:30:05+00:00"


def test_to_tsv_row_bool_is_python_str() -> None:
    rec = _record(is_catch=True)
    row = rec.to_tsv_row()
    assert row["is_catch"] == "True"


def test_participant_id_pattern_enforced() -> None:
    with pytest.raises(ValidationError):
        _record(participant_id="patient_smith")


def test_timestamp_requires_timezone() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        _record(timestamp_utc=datetime(2026, 9, 16, 10, 30, 5))


def test_empty_stimulus_params_is_not_n_a() -> None:
    rec = _record(stimulus_params={})
    row = rec.to_tsv_row()
    assert row["stimulus_params"] == "{}"
