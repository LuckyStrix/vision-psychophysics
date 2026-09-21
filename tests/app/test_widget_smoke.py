"""Widget-instantiation smoke tests, run against the offscreen Qt platform plugin.

These only assert that each screen constructs without error and exposes the
accessible name/labels a screen reader would announce -- not full behavior,
which is covered by the (non-Qt) viewmodel tests elsewhere in this package.
Skipped automatically if the `offscreen` Qt platform plugin isn't available
in this environment (see `tests.app.conftest._offscreen_available`).
"""

from __future__ import annotations

from pathlib import Path

import pytest

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
def test_results_screen_renders_a_figure_canvas_from_a_real_summary(
    qapp, tmp_path: Path, registered_dummy_test
) -> None:
    """Regression guard for the Results screen's figure wiring.

    Builds a complete on-disk session with a summary written by
    `DummyTest.summarize` (a real, registered `PsychophysicalTest`, not a
    hand-built fixture dict) and checks the results screen actually embeds a
    `FigureCanvasQTAgg` for it, not just the old "figure generated (not
    wired up)" placeholder text.
    """
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

    from tests.data.conftest import build_full_session
    from vpsych.app.screens.results import ResultsScreen
    from vpsych.data import catalog

    participant_id, _session_id, _calibration = build_full_session(tmp_path)
    catalog.rebuild_catalog(tmp_path)
    state = AppState(tmp_path)
    state.set_participant(participant_id)

    screen = ResultsScreen(state)
    screen.session_list.setCurrentRow(0)
    screen.summary_combo.setCurrentIndex(0)

    canvases = screen._detail_widget.findChildren(FigureCanvasQTAgg)
    assert len(canvases) >= 1, "expected at least one rendered figure canvas"


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


@requires_offscreen_qt
def test_bisection_widget_keyboard_adjusts_gray_level(qapp) -> None:
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QKeyEvent

    from vpsych.app.widgets.bisection_widget import BisectionWidget

    widget = BisectionWidget()
    widget.set_gray_level(0.5)
    up_event = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Up, Qt.KeyboardModifier.NoModifier)
    widget.keyPressEvent(up_event)
    assert widget.gray_level == pytest.approx(0.505)
    down_event = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Down, Qt.KeyboardModifier.NoModifier)
    widget.keyPressEvent(down_event)
    widget.keyPressEvent(down_event)
    assert widget.gray_level == pytest.approx(0.495)


@requires_offscreen_qt
def test_gamma_step_psychophysical_flow_recovers_gamma_end_to_end(qapp) -> None:
    """Drive the whole grade-B UI flow: select psychophysical, confirm every bisection
    match with a level implied by a known gamma, commit, and build a `GammaCalibration`.
    """
    from vpsych.app.screens.calibration_wizard import _GammaStep
    from vpsych.app.viewmodels.calibration_wizard import CalibrationWizardState

    true_gamma = 2.2
    state = CalibrationWizardState()
    step = _GammaStep(state)
    step.psychophysical_radio.setChecked(True)

    while not step._bisection.is_complete:
        fraction = step._bisection.current_fraction
        assert fraction is not None
        step.bisection_widget.set_gray_level(fraction ** (1.0 / true_gamma))
        step._on_confirm_bisection_match()

    step.commit()
    assert state.gamma.mode == "psychophysical"
    gamma_cal = state.gamma.build()
    assert gamma_cal.method == "psychophysical"
    assert gamma_cal.grade == "B"
    assert gamma_cal.gamma_single == pytest.approx(true_gamma, rel=1e-3)


@requires_offscreen_qt
def test_color_step_manual_entry_commit_builds_grade_a(qapp) -> None:
    from vpsych.app.screens.calibration_wizard import _ColorStep
    from vpsych.app.viewmodels.calibration_wizard import CalibrationWizardState

    state = CalibrationWizardState()
    step = _ColorStep(state)
    step.measured_radio.setChecked(True)
    assert step.measured_panel.isHidden() is False

    step._spin_boxes["red"]["x"].setValue(0.68)
    step._spin_boxes["red"]["y"].setValue(0.32)
    step._spin_boxes["red"]["Y"].setValue(21.0)

    step.commit()
    assert state.color.measured is True
    color_cal = state.color.build()
    assert color_cal.method == "measured"
    assert color_cal.grade == "A"
    assert color_cal.red.x == pytest.approx(0.68)
    assert color_cal.red.Y_cdm2 == pytest.approx(21.0)
