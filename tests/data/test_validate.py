"""Tests for vpsych.data.validate."""

from __future__ import annotations

import json
from pathlib import Path

from vpsych.data import paths, validate

from .conftest import build_full_session


def test_validate_dataset_clean_session_is_ok(tmp_path: Path) -> None:
    build_full_session(tmp_path)
    report = validate.validate_dataset(tmp_path)
    assert report.ok
    assert report.errors == []


def test_validate_session_missing_session_json(tmp_path: Path) -> None:
    report = validate.validate_session("sub-0001", "ses-20260916T103000", tmp_path)
    assert not report.ok
    assert any(i.code == "missing_session_json" for i in report.errors)


def test_validate_dataset_missing_dataset_description(tmp_path: Path) -> None:
    report = validate.validate_dataset(tmp_path)
    assert not report.ok
    assert any(i.code == "missing_dataset_description" for i in report.errors)


def test_manifest_checksum_mismatch_detected(tmp_path: Path) -> None:
    pid, sid, _ = build_full_session(tmp_path)
    trials_path = paths.trials_tsv_path(pid, sid, "dummy_test", "OD", 1, tmp_path)
    trials_path.chmod(0o644)
    with open(trials_path, "a", encoding="utf-8") as f:
        f.write("tampered\n")

    report = validate.validate_dataset(tmp_path)
    assert not report.ok
    assert any(i.code == "checksum_mismatch" for i in report.errors)


def test_manifest_missing_file_detected(tmp_path: Path) -> None:
    pid, sid, _ = build_full_session(tmp_path)
    frames_path = paths.frames_tsv_path(pid, sid, "dummy_test", "OD", 1, tmp_path)
    frames_path.chmod(0o644)
    session_dir = paths.session_dir(pid, sid, tmp_path)
    session_dir.chmod(0o755)
    (session_dir / "beh").chmod(0o755)
    frames_path.unlink()

    report = validate.validate_dataset(tmp_path)
    assert not report.ok
    assert any(i.code == "manifest_file_missing" for i in report.errors)


def test_untracked_file_warns(tmp_path: Path) -> None:
    pid, sid, _ = build_full_session(tmp_path)
    session_dir = paths.session_dir(pid, sid, tmp_path)
    session_dir.chmod(0o755)
    (session_dir / "extra_untracked_file.txt").write_text("hello", encoding="utf-8")

    report = validate.validate_dataset(tmp_path)
    assert any(i.code == "untracked_file" for i in report.warnings)


def test_sidecar_column_mismatch_detected(tmp_path: Path) -> None:
    pid, sid, _ = build_full_session(tmp_path)
    sidecar_path = paths.trials_sidecar_path(pid, sid, "dummy_test", "OD", 1, tmp_path)
    sidecar_path.chmod(0o644)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    del sidecar["intensity"]
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

    report = validate.validate_dataset(tmp_path)
    assert any(i.code == "sidecar_column_mismatch" for i in report.errors)


def test_unknown_calibration_reference_detected(tmp_path: Path) -> None:
    pid, sid, _ = build_full_session(tmp_path)
    session_json_path = paths.session_json_path(pid, sid, tmp_path)
    session_json_path.chmod(0o644)
    data = json.loads(session_json_path.read_text(encoding="utf-8"))
    data["calibration_hash"] = "0" * 64
    session_json_path.write_text(json.dumps(data), encoding="utf-8")

    report = validate.validate_dataset(tmp_path)
    assert any(i.code == "unknown_calibration" for i in report.errors)


def test_unknown_participant_reference_detected(tmp_path: Path) -> None:
    pid, sid, _cal = build_full_session(tmp_path)
    session_json_path = paths.session_json_path(pid, sid, tmp_path)
    session_json_path.chmod(0o644)
    data = json.loads(session_json_path.read_text(encoding="utf-8"))
    data["participant_id"] = "sub-9999"
    session_json_path.write_text(json.dumps(data), encoding="utf-8")

    report = validate.validate_session(pid, sid, tmp_path)
    assert any(i.code == "unknown_participant" for i in report.errors)


def test_stale_calibration_warns(tmp_path: Path) -> None:
    from datetime import datetime, timedelta, timezone

    from vpsych.data import dataset

    from .conftest import make_calibration

    dataset.init_dataset(tmp_path)
    old_cal = make_calibration(now=datetime.now(timezone.utc) - timedelta(days=90))
    dataset.save_calibration(old_cal, tmp_path)

    report = validate.validate_dataset(tmp_path)
    assert any(i.code == "stale_calibration" for i in report.warnings)
