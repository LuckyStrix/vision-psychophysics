"""The wizard's photometer steps must never strand a ``spotread`` session.

Uses a fake session class in place of `ArgyllSpotreadSession` (no process, no
pty, no instrument); what is under test is the wizard closing it on every path.
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from tests.app.conftest import requires_offscreen_qt
from vpsych.core.calibration import argyll
from vpsych.core.calibration.argyll import (
    ArgyllCalibrationRequiredError,
    ArgyllNotAvailableError,
    ArgyllReadError,
    SpotReading,
)


class _FakeSession:
    instances: ClassVar[list[_FakeSession]] = []
    open_error: ClassVar[Exception | None] = None
    confirm_error: ClassVar[Exception | None] = None

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.closed = False
        self.reads = 0
        _FakeSession.instances.append(self)

    def open(self) -> None:
        if _FakeSession.open_error is not None:
            raise _FakeSession.open_error

    def confirm_calibration(self) -> None:
        if _FakeSession.confirm_error is not None:
            raise _FakeSession.confirm_error

    def read_patch(self) -> SpotReading:
        self.reads += 1
        return SpotReading(X=20.0, Y=21.0, Z=22.0)

    def close(self) -> None:
        self.closed = True


class _FakePatchWindow:
    def set_level(self, _level: float) -> None: ...
    def set_rgb(self, *_rgb: float) -> None: ...
    def show(self) -> None: ...
    def close(self) -> None: ...


@pytest.fixture(autouse=True)
def fake_photometer(monkeypatch: pytest.MonkeyPatch):
    from PySide6.QtWidgets import QMessageBox

    _FakeSession.instances = []
    _FakeSession.open_error = None
    _FakeSession.confirm_error = None
    monkeypatch.setattr(argyll, "ArgyllSpotreadSession", _FakeSession)
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)  # would block
    monkeypatch.setattr(
        "vpsych.app.screens.calibration_wizard.PatchWindow", lambda *a, **k: _FakePatchWindow()
    )
    return _FakeSession


def _gamma_step():
    from vpsych.app.screens.calibration_wizard import _GammaStep
    from vpsych.app.viewmodels.calibration_wizard import CalibrationWizardState

    return _GammaStep(CalibrationWizardState())


def _color_step():
    from vpsych.app.screens.calibration_wizard import _ColorStep
    from vpsych.app.viewmodels.calibration_wizard import CalibrationWizardState

    return _ColorStep(CalibrationWizardState())


@requires_offscreen_qt
@pytest.mark.parametrize(
    "error",
    [
        ArgyllNotAvailableError("no instrument"),
        ArgyllReadError("Timed out waiting for spotread output."),
    ],
)
def test_gamma_open_failure_closes_the_session_and_reports(qapp, error: Exception) -> None:
    _FakeSession.open_error = error
    step = _gamma_step()
    step._on_open_photometer()  # a read timeout used to escape this slot uncaught
    (session,) = _FakeSession.instances
    assert session.closed
    assert step._photometer_session is None
    assert str(error) in step.photometer_status_label.text()


@requires_offscreen_qt
def test_gamma_open_twice_closes_the_first_session(qapp) -> None:
    step = _gamma_step()
    step._on_open_photometer()
    step._on_open_photometer()
    first, second = _FakeSession.instances
    assert first.closed
    assert not second.closed
    assert step._photometer_session is second


@requires_offscreen_qt
def test_gamma_calibration_failure_closes_the_session(qapp) -> None:
    _FakeSession.open_error = ArgyllCalibrationRequiredError("Set the dial")
    _FakeSession.confirm_error = ArgyllNotAvailableError("instrument unplugged")
    step = _gamma_step()
    step._on_open_photometer()
    (session,) = _FakeSession.instances
    assert session.closed
    assert step._photometer_session is None
    assert "Calibration failed" in step.photometer_status_label.text()


@requires_offscreen_qt
def test_gamma_successful_calibration_keeps_the_session_open(qapp) -> None:
    _FakeSession.open_error = ArgyllCalibrationRequiredError("Set the dial")
    step = _gamma_step()
    step._on_open_photometer()
    (session,) = _FakeSession.instances
    assert not session.closed
    assert step._photometer is not None


@requires_offscreen_qt
def test_gamma_measurement_complete_releases_the_instrument(qapp) -> None:
    step = _gamma_step()
    step._on_open_photometer()
    (session,) = _FakeSession.instances
    for _ in range(len(step._photometer_levels)):
        step._on_take_gamma_reading()
    assert session.reads == len(step._photometer_levels)
    assert session.closed
    assert step._photometer_session is None


@requires_offscreen_qt
@pytest.mark.parametrize(
    "error",
    [ArgyllNotAvailableError("no instrument"), ArgyllReadError("Timed out")],
)
def test_color_open_failure_closes_the_session_and_reports(qapp, error: Exception) -> None:
    _FakeSession.open_error = error
    step = _color_step()
    step._on_start_primary_measurement()
    (session,) = _FakeSession.instances
    assert session.closed
    assert step._photometer_session is None
    assert str(error) in step.color_photometer_status_label.text()


@requires_offscreen_qt
def test_color_measuring_all_primaries_releases_the_instrument(qapp) -> None:
    step = _color_step()
    step._on_start_primary_measurement()
    (session,) = _FakeSession.instances
    assert session.reads == 4
    assert session.closed
    assert "All four primaries measured" in step.color_photometer_status_label.text()


@requires_offscreen_qt
def test_closing_the_wizard_releases_open_sessions(qapp, tmp_path) -> None:
    from vpsych.app.screens.calibration_wizard import CalibrationWizardScreen
    from vpsych.app.state import AppState

    screen = CalibrationWizardScreen(AppState(tmp_path))
    screen._gamma_step._on_open_photometer()
    (session,) = _FakeSession.instances
    screen.close()
    assert session.closed
