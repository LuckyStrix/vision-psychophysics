"""Qt-level test of the calibration wizard's "Import from calsuite" button.

Runs against the offscreen Qt platform plugin (see
`tests.app.conftest.requires_offscreen_qt`). `QFileDialog.getOpenFileNames`
and `QMessageBox.information`/`.warning` are monkeypatched so the test never
blocks on a real modal dialog -- the same approach any headless Qt test uses
for a file/alert dialog it doesn't want to actually pop up.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.app.conftest import requires_offscreen_qt
from vpsych.app.state import AppState

_NOMINAL = {
    "id": "display.nominal-20260912T180841Z-83393a",
    "kind": "display.nominal",
    "device": {"kind": "display", "model": "CSOT T3", "id": "csot-t3-unknown", "firmware": ""},
    "created": "20260912T180841Z",
    "provenance": "nominal",
    "status": "ok",
    "refusals": [],
    "result": {"physical_size_mm": [344.0, 215.0], "gamma": 2.2, "name": "CSOT T3"},
}

_MEASUREMENT = {
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


def _write(tmp_path: Path, name: str, record: dict) -> str:
    path = tmp_path / name
    path.write_text(json.dumps(record), encoding="utf-8")
    return str(path)


@requires_offscreen_qt
def test_import_calsuite_button_prefills_geometry_gamma_color(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vpsych.app.screens import calibration_wizard as wizard_module
    from vpsych.app.screens.calibration_wizard import CalibrationWizardScreen

    nominal_path = _write(tmp_path, "nominal.json", _NOMINAL)
    measurement_path = _write(tmp_path, "measurement.json", _MEASUREMENT)

    monkeypatch.setattr(
        wizard_module.QFileDialog,
        "getOpenFileNames",
        staticmethod(lambda *a, **k: ([nominal_path, measurement_path], "")),
    )
    info_calls = []
    monkeypatch.setattr(
        wizard_module.QMessageBox, "information", staticmethod(lambda *a, **k: info_calls.append(a))
    )

    state = AppState(tmp_path / "data-root")
    screen = CalibrationWizardScreen(state)
    screen._on_import_calsuite()

    assert len(info_calls) == 1
    assert screen._wizard_state.geometry.width_cm == pytest.approx(34.4)
    assert screen._wizard_state.geometry.width_cm_locked is True
    assert screen._geometry_step.height_cm_spin.value() == pytest.approx(21.5)
    assert screen._gamma_step.external_radio.isChecked() is True
    assert screen._wizard_state.gamma.mode == "external"
    assert screen._color_step.measured_radio.isChecked() is True
    assert screen._wizard_state.color.measured is True

    # The rest of the wizard (resolution, refresh, viewing distance,
    # environment) is still required and unaffected by the import.
    screen._geometry_step.width_px_spin.setValue(1920)
    screen._geometry_step.height_px_spin.setValue(1200)
    screen._geometry_step.refresh_spin.setValue(60.0)
    screen._geometry_step.distance_spin.setValue(60.0)
    screen._environment_step.room_check.setChecked(True)
    screen._environment_step.warmup_check.setChecked(True)
    screen._environment_step.night_light_check.setChecked(True)
    screen._environment_step.hdr_check.setChecked(True)

    for step in (screen._geometry_step, screen._gamma_step, screen._color_step, screen._environment_step):
        step.commit()

    calibration = screen._wizard_state.build_calibration(software_version="0.1.0")
    assert calibration.luminance_grade == "A"
    assert calibration.color_grade == "A"
    assert calibration.geometry.width_cm == pytest.approx(34.4)


@requires_offscreen_qt
def test_import_calsuite_nominal_only_shows_still_needed_and_no_files_selected_is_noop(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vpsych.app.screens import calibration_wizard as wizard_module
    from vpsych.app.screens.calibration_wizard import CalibrationWizardScreen

    nominal_path = _write(tmp_path, "nominal.json", _NOMINAL)

    monkeypatch.setattr(
        wizard_module.QFileDialog,
        "getOpenFileNames",
        staticmethod(lambda *a, **k: ([nominal_path], "")),
    )
    info_calls = []
    monkeypatch.setattr(
        wizard_module.QMessageBox, "information", staticmethod(lambda *a, **k: info_calls.append(a))
    )

    state = AppState(tmp_path / "data-root")
    screen = CalibrationWizardScreen(state)
    screen._on_import_calsuite()

    assert len(info_calls) == 1
    message = info_calls[0][-1]
    assert "gamma calibration" in message
    assert "color calibration" in message
    assert screen._gamma_step.external_radio.isVisible() is False
    assert screen._wizard_state.gamma.mode == "none"

    # Cancelling the file dialog (empty selection) must not touch state.
    monkeypatch.setattr(
        wizard_module.QFileDialog, "getOpenFileNames", staticmethod(lambda *a, **k: ([], ""))
    )
    screen._on_import_calsuite()
    assert len(info_calls) == 1  # unchanged
