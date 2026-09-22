"""Unit tests for vpsych.core.calibration.calsuite_import.

Records are built as plain dicts matching the real shape calsuite writes
(see shared/calibration_suite's committed `records/csot-t3-unknown/*.json`
and `display/commands.py`), not loaded from that sibling repo -- these
fixtures are the contract this module is written against, kept in one
place so a shape change is a deliberate, visible edit here.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from vpsych.core.calibration.calsuite_import import (
    CalsuiteImportError,
    import_calsuite_records,
)


def make_nominal(**overrides) -> dict:
    record = {
        "schema": 1,
        "id": "display.nominal-20260912T180841Z-83393a",
        "kind": "display.nominal",
        "device": {"kind": "display", "model": "CSOT T3", "id": "csot-t3-unknown", "firmware": ""},
        "created": "20260912T180841Z",
        "provenance": "nominal",
        "status": "ok",
        "refusals": [],
        "result": {
            "chromaticity": {
                "r": [0.6377, 0.334],
                "g": [0.2998, 0.5957],
                "b": [0.1406, 0.0576],
                "w": [0.3125, 0.3291],
            },
            "physical_size_mm": [344.0, 215.0],
            "gamma": 2.2,
            "name": "CSOT T3",
        },
    }
    record.update(overrides)
    return record


def make_measurement(**overrides) -> dict:
    record = {
        "schema": 1,
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
            "black_contrast": {
                "black_luminance_cdm2": 0.284342,
                "white_luminance_cdm2": 419.098114,
                "contrast_ratio": 1473.9,
            },
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
            "backend_accuracy": {"de00_estimate": 1.0, "cross_checked_against": None},
        },
    }
    record.update(overrides)
    return record


_NOW = datetime(2026, 9, 22, tzinfo=timezone.utc)


def test_requires_at_least_one_record() -> None:
    with pytest.raises(CalsuiteImportError):
        import_calsuite_records()


def test_nominal_only_imports_size_but_not_color(caplog=None) -> None:
    result = import_calsuite_records(nominal=make_nominal(), now=_NOW)
    assert result.width_cm == pytest.approx(34.4)
    assert result.height_cm == pytest.approx(21.5)
    assert result.gamma is None
    assert result.color is None
    assert any("EDID" in w for w in result.warnings)
    assert any("gamma calibration" in m for m in result.missing_required)
    assert any("color calibration" in m for m in result.missing_required)


def test_full_import_success() -> None:
    result = import_calsuite_records(nominal=make_nominal(), measurement=make_measurement(), now=_NOW)
    assert result.width_cm == pytest.approx(34.4)
    assert result.height_cm == pytest.approx(21.5)
    assert result.gamma is not None
    assert result.gamma.method == "photometer"
    assert result.gamma.gamma_r == pytest.approx(2.197)
    assert result.gamma.lum_min_cdm2 == pytest.approx(0.284342)
    assert result.gamma.lum_max_cdm2 == pytest.approx(419.098114)
    assert result.color is not None
    assert result.color.method == "measured"
    assert result.color.white.Y_cdm2 == pytest.approx(419.098114)
    assert result.color.red.x == pytest.approx(0.6397)
    assert not result.rejected
    assert result.source_created_utc == datetime(2026, 9, 22, 0, 2, 48, tzinfo=timezone.utc)
    assert "width_px" in " ".join(result.missing_required)
    assert not any("gamma calibration" in m for m in result.missing_required)


def test_measurement_only_still_needs_size() -> None:
    result = import_calsuite_records(measurement=make_measurement(), now=_NOW)
    assert result.width_cm is None
    assert result.gamma is not None
    assert result.color is not None
    assert any("width_cm" in m for m in result.missing_required)


def test_refused_measurement_is_rejected_not_used() -> None:
    bad = make_measurement(status="refused", refusals=["validation_failed: DeltaE00 too high"])
    result = import_calsuite_records(measurement=bad, now=_NOW)
    assert result.gamma is None
    assert result.color is None
    assert any("refused" in r and "validation_failed" in r for r in result.rejected)


def test_nominal_provenance_measurement_is_rejected() -> None:
    bad = make_measurement(provenance="nominal")
    result = import_calsuite_records(measurement=bad, now=_NOW)
    assert result.gamma is None
    assert result.color is None
    assert any("provenance" in r for r in result.rejected)


def test_synthetic_backend_is_rejected() -> None:
    bad = make_measurement()
    bad["method"]["params"]["backend"] = "synthetic"
    result = import_calsuite_records(measurement=bad, now=_NOW)
    assert result.gamma is None
    assert result.color is None
    assert any("synthetic" in r for r in result.rejected)


def test_unknown_backend_is_rejected() -> None:
    bad = make_measurement()
    bad["method"]["params"]["backend"] = "some-future-backend"
    result = import_calsuite_records(measurement=bad, now=_NOW)
    assert result.gamma is None
    assert any("unrecognized" in r for r in result.rejected)


def test_camera_backend_imports_with_accuracy_warning() -> None:
    record = make_measurement()
    record["method"]["params"]["backend"] = "camera"
    result = import_calsuite_records(measurement=record, now=_NOW)
    assert result.gamma is not None
    assert result.color is not None
    assert any("camera" in w and "cross-checked" in w for w in result.warnings)
    assert any("cross_checked_against" in w for w in result.warnings)


def test_camera_backend_cross_checked_suppresses_second_warning() -> None:
    record = make_measurement()
    record["method"]["params"]["backend"] = "camera"
    record["result"]["backend_accuracy"]["cross_checked_against"] = "argyll"
    result = import_calsuite_records(measurement=record, now=_NOW)
    assert any("camera" in w for w in result.warnings)
    assert not any("cross_checked_against" in w for w in result.warnings)


def test_wrong_kind_for_slot_is_rejected() -> None:
    result = import_calsuite_records(nominal=make_measurement(), now=_NOW)
    assert result.width_cm is None
    assert any("expected 'display.nominal'" in r for r in result.rejected)


def test_missing_gamma_fields_are_rejected_individually() -> None:
    record = make_measurement()
    del record["result"]["trc"]["effective_gamma"]["b"]
    result = import_calsuite_records(measurement=record, now=_NOW)
    assert result.gamma is None
    assert result.color is not None
    assert any("effective_gamma" in r for r in result.rejected)


def test_stale_measurement_warns() -> None:
    far_future = datetime(2027, 6, 1, tzinfo=timezone.utc)
    result = import_calsuite_records(measurement=make_measurement(), now=far_future)
    assert any("days old" in w for w in result.warnings)


def test_notes_mention_both_records() -> None:
    result = import_calsuite_records(nominal=make_nominal(), measurement=make_measurement(), now=_NOW)
    assert result.notes is not None
    assert "physical screen size" in result.notes.lower()
    assert "gamma/color" in result.notes.lower()
