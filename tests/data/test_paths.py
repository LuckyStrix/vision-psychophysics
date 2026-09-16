"""Unit tests for vpsych.data.paths: BIDS-like path builder."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from vpsych.data import paths


def test_data_root_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VPSYCH_DATA_ROOT", raising=False)
    root = paths.data_root()
    assert root == Path(os.path.expanduser("~/vpsych-data")).resolve()


def test_data_root_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VPSYCH_DATA_ROOT", str(tmp_path / "custom-data"))
    assert paths.data_root() == (tmp_path / "custom-data").resolve()


def test_data_root_expands_user(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VPSYCH_DATA_ROOT", "~/somewhere")
    root = paths.data_root()
    assert "~" not in str(root)


def test_dataset_level_paths(tmp_path: Path) -> None:
    assert paths.dataset_description_path(tmp_path) == tmp_path / "dataset_description.json"
    assert paths.participants_tsv_path(tmp_path) == tmp_path / "participants.tsv"
    assert paths.participants_json_path(tmp_path) == tmp_path / "participants.json"
    assert paths.catalog_sqlite_path(tmp_path) == tmp_path / "catalog.sqlite"
    assert paths.calibration_dir(tmp_path) == tmp_path / "calibration"


def test_calibration_path(tmp_path: Path) -> None:
    p = paths.calibration_path("abc123", root=tmp_path)
    assert p == tmp_path / "calibration" / "cal-abc123.json"


def test_participant_dir(tmp_path: Path) -> None:
    assert paths.participant_dir("sub-0001", tmp_path) == tmp_path / "sub-0001"


def test_participant_dir_invalid_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="participant_id"):
        paths.participant_dir("patient_x", tmp_path)
    with pytest.raises(ValueError, match="participant_id"):
        paths.participant_dir("sub-1", tmp_path)  # wrong digit count


def test_session_dir(tmp_path: Path) -> None:
    p = paths.session_dir("sub-0001", "ses-20260916T103000", tmp_path)
    assert p == tmp_path / "sub-0001" / "ses-20260916T103000"


def test_session_dir_invalid_session_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="session_id"):
        paths.session_dir("sub-0001", "session-1", tmp_path)


def test_session_json_and_manifest_paths(tmp_path: Path) -> None:
    session = paths.session_dir("sub-0001", "ses-20260916T103000", tmp_path)
    assert (
        paths.session_json_path("sub-0001", "ses-20260916T103000", tmp_path)
        == session / "session.json"
    )
    assert (
        paths.session_manifest_path("sub-0001", "ses-20260916T103000", tmp_path)
        == session / "MANIFEST.sha256"
    )


def test_beh_dir(tmp_path: Path) -> None:
    session = paths.session_dir("sub-0001", "ses-20260916T103000", tmp_path)
    assert paths.beh_dir("sub-0001", "ses-20260916T103000", tmp_path) == session / "beh"


@pytest.mark.parametrize(
    ("fn", "suffix"),
    [
        (paths.trials_tsv_path, "_trials.tsv"),
        (paths.trials_sidecar_path, "_trials.json"),
        (paths.summary_json_path, "_summary.json"),
        (paths.frames_tsv_path, "_frames.tsv"),
    ],
)
def test_beh_file_paths(tmp_path: Path, fn: object, suffix: str) -> None:
    p = fn("sub-0001", "ses-20260916T103000", "acuity", "OD", 1, root=tmp_path)  # type: ignore[operator]
    expected_stem = "sub-0001_ses-20260916T103000_task-acuity_eye-OD_run-1"
    assert p.name == f"{expected_stem}{suffix}"
    assert p.parent == paths.beh_dir("sub-0001", "ses-20260916T103000", tmp_path)


def test_beh_file_path_invalid_task_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="task_id"):
        paths.trials_tsv_path(
            "sub-0001", "ses-20260916T103000", "Acuity-Test", "OD", 1, root=tmp_path
        )


def test_beh_file_path_invalid_eye(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="eye"):
        paths.trials_tsv_path("sub-0001", "ses-20260916T103000", "acuity", "LEFT", 1, root=tmp_path)


def test_beh_file_path_invalid_run(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="run"):
        paths.trials_tsv_path("sub-0001", "ses-20260916T103000", "acuity", "OD", 0, root=tmp_path)


def test_ensure_dir_exists_creates_directory(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "c"
    assert not target.exists()
    result = paths.ensure_dir_exists(target)
    assert target.exists() and target.is_dir()
    assert result == target


def test_ensure_dir_exists_is_idempotent(tmp_path: Path) -> None:
    target = tmp_path / "a"
    paths.ensure_dir_exists(target)
    paths.ensure_dir_exists(target)  # should not raise
    assert target.exists()
