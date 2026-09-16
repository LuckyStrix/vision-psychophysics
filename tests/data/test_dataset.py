"""Tests for vpsych.data.dataset: dataset init, participants, calibration store."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from vpsych.data import dataset, paths
from vpsych.data.schemas import Participant

from .conftest import make_calibration


def test_init_dataset_creates_files(tmp_path: Path) -> None:
    desc = dataset.init_dataset(tmp_path, name="my dataset")
    assert desc.name == "my dataset"
    assert desc.schema_version == "1.0.0"
    assert paths.dataset_description_path(tmp_path).exists()
    assert paths.calibration_dir(tmp_path).exists()
    assert paths.participants_tsv_path(tmp_path).exists()
    assert paths.participants_json_path(tmp_path).exists()


def test_init_dataset_is_idempotent(tmp_path: Path) -> None:
    first = dataset.init_dataset(tmp_path, name="first")
    second = dataset.init_dataset(tmp_path, name="second")
    assert second == first
    assert second.name == "first"


def test_init_dataset_force_overwrites(tmp_path: Path) -> None:
    dataset.init_dataset(tmp_path, name="first")
    second = dataset.init_dataset(tmp_path, name="second", force=True)
    assert second.name == "second"


def test_load_dataset_description_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        dataset.load_dataset_description(tmp_path)


# ---------------------------------------------------------------------------
# Participants
# ---------------------------------------------------------------------------


def test_create_participant_allocates_sub_0001(tmp_path: Path) -> None:
    dataset.init_dataset(tmp_path)
    p = dataset.create_participant(tmp_path, year_of_birth=1990, sex="F")
    assert p.participant_id == "sub-0001"
    assert p.year_of_birth == 1990
    assert p.sex == "F"


def test_create_participant_allocates_sequential_ids(tmp_path: Path) -> None:
    dataset.init_dataset(tmp_path)
    p1 = dataset.create_participant(tmp_path)
    p2 = dataset.create_participant(tmp_path)
    p3 = dataset.create_participant(tmp_path)
    assert [p1.participant_id, p2.participant_id, p3.participant_id] == [
        "sub-0001",
        "sub-0002",
        "sub-0003",
    ]


def test_create_participant_fills_gaps(tmp_path: Path) -> None:
    dataset.init_dataset(tmp_path)
    p1 = dataset.create_participant(tmp_path)
    p2 = dataset.create_participant(tmp_path)
    p3 = dataset.create_participant(tmp_path)
    assert [p.participant_id for p in (p1, p2, p3)] == ["sub-0001", "sub-0002", "sub-0003"]

    dataset.update_participant(p2.participant_id, tmp_path, notes="keep for now")
    # Simulate removing sub-0002 by rewriting the participant list without it.
    remaining = [p for p in dataset.list_participants(tmp_path) if p.participant_id != "sub-0002"]
    dataset._write_participants(tmp_path, remaining)

    p4 = dataset.create_participant(tmp_path)
    assert p4.participant_id == "sub-0002"


def test_list_participants_empty_dataset(tmp_path: Path) -> None:
    assert dataset.list_participants(tmp_path) == []


def test_get_participant_found_and_not_found(tmp_path: Path) -> None:
    dataset.init_dataset(tmp_path)
    p = dataset.create_participant(tmp_path, sex="M")
    assert dataset.get_participant(p.participant_id, tmp_path) == p
    assert dataset.get_participant("sub-9999", tmp_path) is None


def test_update_participant(tmp_path: Path) -> None:
    dataset.init_dataset(tmp_path)
    p = dataset.create_participant(tmp_path)
    updated = dataset.update_participant(p.participant_id, tmp_path, sex="F", notes="updated")
    assert updated.sex == "F"
    assert updated.notes == "updated"
    assert updated.participant_id == p.participant_id


def test_update_participant_not_found_raises(tmp_path: Path) -> None:
    dataset.init_dataset(tmp_path)
    with pytest.raises(dataset.ParticipantNotFoundError):
        dataset.update_participant("sub-9999", tmp_path, sex="F")


def test_update_participant_cannot_change_id(tmp_path: Path) -> None:
    dataset.init_dataset(tmp_path)
    p = dataset.create_participant(tmp_path)
    with pytest.raises(ValueError, match="participant_id"):
        dataset.update_participant(p.participant_id, tmp_path, participant_id="sub-9999")


def test_participants_tsv_round_trip_preserves_none(tmp_path: Path) -> None:
    dataset.init_dataset(tmp_path)
    dataset.create_participant(tmp_path)  # all fields None
    participants = dataset.list_participants(tmp_path)
    assert participants[0].year_of_birth is None
    assert participants[0].sex is None
    assert participants[0].notes is None


def test_participant_rejects_extra_fields_privacy() -> None:
    with pytest.raises(ValidationError):
        Participant(participant_id="sub-0001", name="Alice Example")  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        Participant(participant_id="sub-0001", email="alice@example.com")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Calibration store
# ---------------------------------------------------------------------------


def test_save_and_load_calibration_round_trips(tmp_path: Path) -> None:
    dataset.init_dataset(tmp_path)
    calibration = make_calibration()
    path = dataset.save_calibration(calibration, tmp_path)
    assert path.exists()
    assert path.name == f"cal-{calibration.content_hash()}.json"

    loaded = dataset.load_calibration(calibration.content_hash(), tmp_path)
    assert loaded == calibration


def test_load_calibration_missing_raises(tmp_path: Path) -> None:
    dataset.init_dataset(tmp_path)
    with pytest.raises(FileNotFoundError):
        dataset.load_calibration("0" * 64, tmp_path)


def test_load_calibration_tampered_raises_integrity_error(tmp_path: Path) -> None:
    dataset.init_dataset(tmp_path)
    calibration = make_calibration()
    path = dataset.save_calibration(calibration, tmp_path)
    path.chmod(0o644)
    tampered = path.read_text(encoding="utf-8").replace(
        '"software_version": "0.1.0"', '"software_version": "9.9.9"'
    )
    path.write_text(tampered, encoding="utf-8")

    with pytest.raises(dataset.CalibrationIntegrityError):
        dataset.load_calibration(calibration.content_hash(), tmp_path)


def test_save_calibration_idempotent_same_content(tmp_path: Path) -> None:
    dataset.init_dataset(tmp_path)
    calibration = make_calibration()
    path1 = dataset.save_calibration(calibration, tmp_path)
    path2 = dataset.save_calibration(calibration, tmp_path)
    assert path1 == path2


def test_list_calibrations_sorted_by_created_utc(tmp_path: Path) -> None:
    from datetime import datetime, timedelta, timezone

    dataset.init_dataset(tmp_path)
    now = datetime.now(timezone.utc)
    older = make_calibration(now=now - timedelta(days=10))
    newer = make_calibration(now=now)
    dataset.save_calibration(older, tmp_path)
    dataset.save_calibration(newer, tmp_path)

    calibrations = dataset.list_calibrations(tmp_path)
    assert len(calibrations) == 2
    assert calibrations[0].created_utc < calibrations[1].created_utc


def test_latest_calibration(tmp_path: Path) -> None:
    from datetime import datetime, timedelta, timezone

    dataset.init_dataset(tmp_path)
    now = datetime.now(timezone.utc)
    older = make_calibration(now=now - timedelta(days=10))
    newer = make_calibration(now=now)
    dataset.save_calibration(older, tmp_path)
    dataset.save_calibration(newer, tmp_path)

    latest = dataset.latest_calibration(tmp_path)
    assert latest is not None
    assert latest.created_utc == newer.created_utc


def test_latest_calibration_empty_returns_none(tmp_path: Path) -> None:
    dataset.init_dataset(tmp_path)
    assert dataset.latest_calibration(tmp_path) is None
