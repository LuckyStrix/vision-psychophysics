"""Widget-instantiation smoke tests, run against the offscreen Qt platform plugin.

These only assert that each screen constructs without error and exposes the
accessible name/labels a screen reader would announce -- not full behavior,
which is covered by the (non-Qt) viewmodel tests elsewhere in this package.
Skipped automatically if the `offscreen` Qt platform plugin isn't available
in this environment (see `tests.app.conftest._offscreen_available`).
"""

from __future__ import annotations

from pathlib import Path

from tests.app.conftest import requires_offscreen_qt
from vpsych.app.state import AppState


@requires_offscreen_qt
def test_home_screen_constructs(qapp, tmp_path: Path) -> None:
    from vpsych.app.screens.home import HomeScreen

    state = AppState(tmp_path)
    screen = HomeScreen(state)
    assert screen.accessibleName() == "Home screen"
    assert screen.participant_combo.count() == 0


@requires_offscreen_qt
def test_catalog_screen_constructs_with_no_calibration(qapp, tmp_path: Path) -> None:
    from vpsych.app.screens.catalog import TestCatalogScreen

    state = AppState(tmp_path)
    screen = TestCatalogScreen(state)
    assert screen.accessibleName() == "Test catalog screen"


@requires_offscreen_qt
def test_session_builder_screen_add_test(qapp, tmp_path, registered_dummy_test) -> None:
    from vpsych.app.screens.session_builder import SessionBuilderScreen

    state = AppState(tmp_path)
    screen = SessionBuilderScreen(state, tmp_path / "plans")
    screen.add_test("dummy_test")
    assert len(screen._builder_state.tests) == 1
    assert screen._builder_state.tests[0].task_id == "dummy_test"


@requires_offscreen_qt
def test_calibration_wizard_screen_constructs(qapp, tmp_path: Path) -> None:
    from vpsych.app.screens.calibration_wizard import CalibrationWizardScreen

    state = AppState(tmp_path)
    screen = CalibrationWizardScreen(state)
    assert screen.accessibleName() == "Calibration wizard"
    assert screen.stack.count() == 5


@requires_offscreen_qt
def test_results_screen_constructs(qapp, tmp_path: Path) -> None:
    from vpsych.app.screens.results import ResultsScreen

    state = AppState(tmp_path)
    screen = ResultsScreen(state)
    assert screen.accessibleName() == "Results screen"


@requires_offscreen_qt
def test_data_screen_constructs(qapp, tmp_path: Path) -> None:
    from vpsych.app.screens.data import DataScreen

    state = AppState(tmp_path)
    screen = DataScreen(state)
    assert screen.accessibleName() == "Data screen"


@requires_offscreen_qt
def test_run_screen_constructs(qapp, tmp_path: Path) -> None:
    from vpsych.app.screens.run import RunScreen

    state = AppState(tmp_path)
    screen = RunScreen(state, tmp_path / "runs")
    assert screen.accessibleName() == "Run screen"


@requires_offscreen_qt
def test_card_match_widget_keyboard_resizes(qapp) -> None:
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QKeyEvent

    from vpsych.app.widgets.card_match import CardMatchWidget

    widget = CardMatchWidget(initial_width_px=200.0)
    widget.resize(800, 300)
    before = widget.card_width_px
    event = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Right, Qt.KeyboardModifier.NoModifier)
    widget.keyPressEvent(event)
    assert widget.card_width_px == before + 1.0
