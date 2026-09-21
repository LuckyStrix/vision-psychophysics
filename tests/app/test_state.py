"""Tests for `vpsych.app.state.AppState` (a plain `QObject`, no display needed)."""

from __future__ import annotations

from pathlib import Path

from tests.data.conftest import make_calibration
from vpsych.app.state import AppState
from vpsych.data import dataset


def test_app_state_initializes_dataset(tmp_path: Path) -> None:
    state = AppState(tmp_path)
    assert (tmp_path / "dataset_description.json").exists()
    assert state.data_root == tmp_path
    assert state.participant_id is None
    assert state.latest_calibration is None


def test_app_state_set_participant_emits_signal(tmp_path: Path) -> None:
    state = AppState(tmp_path)
    dataset.create_participant(tmp_path)
    received: list[str] = []
    state.participant_changed.connect(received.append)
    state.set_participant("sub-0001")
    assert state.participant_id == "sub-0001"
    assert received == ["sub-0001"]

    # Setting the same participant again should not re-emit.
    state.set_participant("sub-0001")
    assert received == ["sub-0001"]


def test_app_state_refresh_calibration_picks_up_latest(tmp_path: Path) -> None:
    state = AppState(tmp_path)
    assert state.latest_calibration is None
    cal = make_calibration()
    dataset.save_calibration(cal, tmp_path)

    events: list[None] = []
    state.calibration_changed.connect(lambda: events.append(None))
    state.refresh_calibration()
    assert state.latest_calibration is not None
    assert state.latest_calibration.content_hash() == cal.content_hash()
    assert len(events) == 1
