"""Tests for vpsych.data.export."""

from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path

import pytest

from vpsych.data import dataset, export

from .conftest import build_full_session


def test_export_participant_creates_zip_with_expected_contents(tmp_path: Path) -> None:
    pid, sid, _ = build_full_session(tmp_path)
    out_zip = tmp_path.parent / "export.zip"

    result_path = export.export_participant(pid, out_zip, tmp_path)
    assert result_path == out_zip
    assert out_zip.exists()

    with zipfile.ZipFile(out_zip) as zf:
        names = set(zf.namelist())
        assert "summaries_tidy.csv" in names
        assert "data_dictionary.md" in names
        assert "validation_report.json" in names
        assert any(n.startswith("json_schemas/") for n in names)
        assert f"raw/{pid}/{sid}/session.json" in names
        assert any(
            n.startswith(f"raw/{pid}/{sid}/beh/") and n.endswith("_trials.tsv") for n in names
        )
        assert "raw/dataset_description.json" in names
        assert "raw/participants.tsv" in names
        assert any(n.startswith("raw/calibration/cal-") for n in names)

        tidy_bytes = zf.read("summaries_tidy.csv")
        rows = list(csv.DictReader(io.StringIO(tidy_bytes.decode("utf-8"))))
        assert len(rows) == 1
        assert rows[0]["task_id"] == "dummy_test"
        assert rows[0]["participant_id"] == pid

        report = json.loads(zf.read("validation_report.json"))
        assert report["issues"] == [] or all(
            issue["severity"] != "error" for issue in report["issues"]
        )


def test_export_participant_excludes_other_participants(tmp_path: Path) -> None:
    dataset.init_dataset(tmp_path)
    p1 = dataset.create_participant(tmp_path)
    p2 = dataset.create_participant(tmp_path)
    build_full_session(tmp_path, participant_id=p1.participant_id, session_id="ses-20260101T090000")
    build_full_session(tmp_path, participant_id=p2.participant_id, session_id="ses-20260102T090000")

    out_zip = tmp_path.parent / "export_p1.zip"
    export.export_participant(p1.participant_id, out_zip, tmp_path)

    with zipfile.ZipFile(out_zip) as zf:
        names = zf.namelist()
        assert any(p1.participant_id in n for n in names)
        assert not any(p2.participant_id in n for n in names)


def test_export_dataset_includes_all_participants(tmp_path: Path) -> None:
    dataset.init_dataset(tmp_path)
    p1 = dataset.create_participant(tmp_path)
    p2 = dataset.create_participant(tmp_path)
    build_full_session(tmp_path, participant_id=p1.participant_id, session_id="ses-20260101T090000")
    build_full_session(tmp_path, participant_id=p2.participant_id, session_id="ses-20260102T090000")

    out_zip = tmp_path.parent / "export_all.zip"
    export.export_dataset(out_zip, tmp_path)

    with zipfile.ZipFile(out_zip) as zf:
        names = zf.namelist()
        assert any(p1.participant_id in n for n in names)
        assert any(p2.participant_id in n for n in names)
        tidy_bytes = zf.read("summaries_tidy.csv")
        rows = list(csv.DictReader(io.StringIO(tidy_bytes.decode("utf-8"))))
        assert len(rows) == 2


def test_export_unknown_participant_raises(tmp_path: Path) -> None:
    dataset.init_dataset(tmp_path)
    with pytest.raises(FileNotFoundError):
        export.export_participant("sub-9999", tmp_path.parent / "x.zip", tmp_path)


def test_export_data_dictionary_documents_trial_columns(tmp_path: Path) -> None:
    pid, _sid, _ = build_full_session(tmp_path)
    out_zip = tmp_path.parent / "export.zip"
    export.export_participant(pid, out_zip, tmp_path)

    with zipfile.ZipFile(out_zip) as zf:
        dictionary = zf.read("data_dictionary.md").decode("utf-8")
        assert "intensity" in dictionary
        assert "participant_id" in dictionary
        assert "logMAR" in dictionary or "intensity_units" in dictionary


def test_export_does_not_include_the_zip_being_written(tmp_path: Path) -> None:
    pid, _, _ = build_full_session(tmp_path)
    out_zip = tmp_path / pid / "export.zip"

    export.export_participant(pid, out_zip, tmp_path)

    with zipfile.ZipFile(out_zip) as zf:
        assert not any(n.endswith("export.zip") for n in zf.namelist())
