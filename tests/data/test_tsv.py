"""Round-trip tests for vpsych.data.tsv.read_trials_tsv against TrialRecord.to_tsv_row."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest

from vpsych.data.tsv import read_trials_tsv

from .conftest import make_trial


def _write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0].keys()), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_round_trip_basic_trial(tmp_path: Path) -> None:
    trial = make_trial()
    path = tmp_path / "trials.tsv"
    _write_tsv(path, [trial.to_tsv_row()])

    df = read_trials_tsv(path)
    assert len(df) == 1
    row = df.iloc[0]

    assert row["participant_id"] == trial.participant_id
    assert row["task_id"] == trial.task_id
    assert row["run"] == trial.run
    assert int(row["run"]) == trial.run
    assert row["trial_index"] == trial.trial_index
    assert row["is_catch"] == trial.is_catch
    assert bool(row["is_catch"]) is trial.is_catch
    assert row["intensity"] == trial.intensity
    assert row["stimulus_params"] == trial.stimulus_params
    assert row["correct_response"] == trial.correct_response
    assert row["response"] == trial.response
    assert row["correct"] == trial.correct
    assert row["rt_s"] == trial.rt_s
    assert row["procedure_state"] == trial.procedure_state
    assert row["rng_seed"] == trial.rng_seed


def test_round_trip_none_values_become_none(tmp_path: Path) -> None:
    trial = make_trial(response=None, correct=None, rt_s=None)
    path = tmp_path / "trials.tsv"
    _write_tsv(path, [trial.to_tsv_row()])

    df = read_trials_tsv(path)
    row = df.iloc[0]
    assert row["response"] is None
    # "correct" is pandas' nullable "boolean" dtype (not plain "object"), so
    # its missing-value sentinel is pd.NA, not None -- see read_trials_tsv's
    # _NULLABLE_BOOL_COLUMNS handling for why (`~` must negate logically, not
    # bitwise-invert a bare Python bool).
    assert pd.isna(row["correct"])
    assert row["rt_s"] is None


def test_round_trip_dict_and_list_response(tmp_path: Path) -> None:
    trial = make_trial(
        correct_response={"direction_deg": 90},
        response=[1, 2, 3],
    )
    path = tmp_path / "trials.tsv"
    _write_tsv(path, [trial.to_tsv_row()])

    df = read_trials_tsv(path)
    row = df.iloc[0]
    assert row["correct_response"] == {"direction_deg": 90}
    assert row["response"] == [1, 2, 3]


def test_round_trip_numeric_and_string_response(tmp_path: Path) -> None:
    trial = make_trial(correct_response=3, response="left")
    path = tmp_path / "trials.tsv"
    _write_tsv(path, [trial.to_tsv_row()])

    df = read_trials_tsv(path)
    row = df.iloc[0]
    assert row["correct_response"] == 3
    assert int(row["correct_response"]) == 3
    assert row["response"] == "left"


def test_round_trip_multiple_rows_preserves_order(tmp_path: Path) -> None:
    intensities = [0.0, 0.1, 0.2, 0.3, 0.4]
    trials = [make_trial(trial_index=i, intensity=v) for i, v in enumerate(intensities)]
    path = tmp_path / "trials.tsv"
    _write_tsv(path, [t.to_tsv_row() for t in trials])

    df = read_trials_tsv(path)
    assert list(df["trial_index"]) == [0, 1, 2, 3, 4]
    assert list(df["intensity"]) == pytest.approx(intensities)


def test_timestamp_parsed_as_datetime(tmp_path: Path) -> None:
    trial = make_trial()
    path = tmp_path / "trials.tsv"
    _write_tsv(path, [trial.to_tsv_row()])

    df = read_trials_tsv(path)
    assert str(df["timestamp_utc"].dtype).startswith("datetime64")
