"""Tests for vpsych.data.cli: the `vpsych-data` console script's argument handling."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vpsych.data import cli

from .conftest import build_full_session


def test_init_command(tmp_path: Path, capsys: object) -> None:
    rc = cli.main(["--root", str(tmp_path), "init", "--name", "cli test dataset"])
    assert rc == 0
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    out = json.loads(captured.out)
    assert out["name"] == "cli test dataset"
    assert (tmp_path / "dataset_description.json").exists()


def test_validate_command_clean_dataset(tmp_path: Path, capsys: object) -> None:
    build_full_session(tmp_path)
    rc = cli.main(["--root", str(tmp_path), "validate"])
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    report = json.loads(captured.out)
    assert report["issues"] == [] or all(issue["severity"] != "error" for issue in report["issues"])
    assert rc == 0


def test_validate_command_missing_dataset_returns_nonzero(tmp_path: Path) -> None:
    rc = cli.main(["--root", str(tmp_path), "validate"])
    assert rc == 1


def test_validate_command_specific_session(tmp_path: Path, capsys: object) -> None:
    pid, sid, _ = build_full_session(tmp_path)
    rc = cli.main(
        ["--root", str(tmp_path), "validate", "--participant-id", pid, "--session-id", sid]
    )
    assert rc == 0


def test_validate_command_session_without_participant_id_errors(tmp_path: Path) -> None:
    rc = cli.main(["--root", str(tmp_path), "validate", "--session-id", "ses-20260916T103000"])
    assert rc == 2


def test_rebuild_catalog_command(tmp_path: Path, capsys: object) -> None:
    build_full_session(tmp_path)
    rc = cli.main(["--root", str(tmp_path), "rebuild-catalog"])
    assert rc == 0
    assert (tmp_path / "catalog.sqlite").exists()


def test_export_command(tmp_path: Path, capsys: object) -> None:
    build_full_session(tmp_path)
    out_zip = tmp_path.parent / "cli_export.zip"
    rc = cli.main(["--root", str(tmp_path), "export", str(out_zip)])
    assert rc == 0
    assert out_zip.exists()


def test_reanalyze_command(tmp_path: Path, capsys: object, registered_dummy_test: object) -> None:
    pid, sid, _ = build_full_session(tmp_path)
    session_dir = tmp_path / pid / sid
    rc = cli.main(["--root", str(tmp_path), "reanalyze", str(session_dir)])
    assert rc == 0
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    results = json.loads(captured.out)
    assert len(results) == 1
    assert results[0]["matches"] is True


def _write_calsuite_record(tmp_path: Path, name: str, record: dict) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


_CALSUITE_NOMINAL = {
    "id": "display.nominal-20260912T180841Z-83393a",
    "kind": "display.nominal",
    "device": {"kind": "display", "model": "CSOT T3", "id": "csot-t3-unknown", "firmware": ""},
    "created": "20260912T180841Z",
    "provenance": "nominal",
    "status": "ok",
    "refusals": [],
    "result": {"physical_size_mm": [344.0, 215.0], "gamma": 2.2, "name": "CSOT T3"},
}

_CALSUITE_MEASUREMENT = {
    "id": "display.measurement-20260922T000248Z-f98b26",
    "kind": "display.measurement",
    "device": {"kind": "display", "model": "CSOT T3", "id": "csot-t3-unknown", "firmware": ""},
    "created": "20260922T000248Z",
    "provenance": "measured",
    "status": "ok",
    "refusals": [],
    "method": {"name": "calsuite.display.commands", "params": {"backend": "argyll", "steps": 5}},
    "result": {
        "trc": {"effective_gamma": {"r": 2.197, "g": 2.194, "b": 2.200}},
        "black_contrast": {"black_luminance_cdm2": 0.284342, "white_luminance_cdm2": 419.098114},
        "primaries": {
            "measured_chromaticity": {
                "r": [0.6397, 0.3333],
                "g": [0.3049, 0.5954],
                "b": [0.1399, 0.0570],
                "w": [0.3103, 0.3269],
            }
        },
        "primaries_measured": {
            "r": [173.048199, 90.322905, 7.734654],
            "g": [153.092823, 298.686085, 50.438894],
            "b": [71.0012, 29.10717, 406.564722],
            "w": [397.919419, 419.098114, 465.211748],
        },
    },
}


def test_import_calsuite_full_success(tmp_path: Path, capsys: object) -> None:
    nominal_path = _write_calsuite_record(tmp_path, "nominal.json", _CALSUITE_NOMINAL)
    measurement_path = _write_calsuite_record(tmp_path, "measurement.json", _CALSUITE_MEASUREMENT)
    root = tmp_path / "root"
    rc = cli.main(
        [
            "--root",
            str(root),
            "import-calsuite",
            "--nominal",
            str(nominal_path),
            "--measurement",
            str(measurement_path),
            "--width-px",
            "1920",
            "--height-px",
            "1200",
            "--viewing-distance-cm",
            "60",
            "--refresh-hz",
            "60",
            "--room-lighting-controlled",
            "--monitor-warmed-up",
            "--night-light-disabled",
            "--hdr-disabled",
        ]
    )
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert rc == 0, captured.err
    assert "Saved:" in captured.out
    assert "Luminance grade A" in captured.out
    assert "color grade A" in captured.out
    from vpsych.data.dataset import list_calibrations

    calibrations = list_calibrations(root)
    assert len(calibrations) == 1
    assert calibrations[0].geometry.width_cm == pytest.approx(34.4)


def test_import_calsuite_nominal_only_reports_still_needed(tmp_path: Path, capsys: object) -> None:
    nominal_path = _write_calsuite_record(tmp_path, "nominal.json", _CALSUITE_NOMINAL)
    root = tmp_path / "root"
    rc = cli.main(
        [
            "--root",
            str(root),
            "import-calsuite",
            "--nominal",
            str(nominal_path),
            "--width-px",
            "1920",
            "--height-px",
            "1200",
            "--viewing-distance-cm",
            "60",
            "--refresh-hz",
            "60",
        ]
    )
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert rc == 1
    assert "Still needed" in captured.err
    assert "gamma" in captured.err
    assert "color" in captured.err
    assert not (root / "calibration").exists()


def test_import_calsuite_refused_measurement_blocks_save(tmp_path: Path, capsys: object) -> None:
    refused = dict(_CALSUITE_MEASUREMENT)
    refused["status"] = "refused"
    refused["refusals"] = ["validation_failed: too few patches"]
    measurement_path = _write_calsuite_record(tmp_path, "measurement.json", refused)
    root = tmp_path / "root"
    rc = cli.main(
        [
            "--root",
            str(root),
            "import-calsuite",
            "--measurement",
            str(measurement_path),
            "--width-cm",
            "34.4",
            "--height-cm",
            "21.5",
            "--width-px",
            "1920",
            "--height-px",
            "1200",
            "--viewing-distance-cm",
            "60",
            "--refresh-hz",
            "60",
        ]
    )
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert rc == 1
    assert "REJECTED" in captured.err
    assert "validation_failed" in captured.err


def test_import_calsuite_requires_a_record(tmp_path: Path) -> None:
    rc = cli.main(
        [
            "--root",
            str(tmp_path / "root"),
            "import-calsuite",
            "--width-px",
            "1920",
            "--height-px",
            "1200",
            "--viewing-distance-cm",
            "60",
            "--refresh-hz",
            "60",
        ]
    )
    assert rc == 2
