"""Main window: a left navigation panel plus a stacked widget of screens.

Wires the screens' navigation signals together (Home -> Calibration/Catalog/
Session builder/Data, Catalog -> Session builder, Session builder -> Run,
Run -> Results) so the whole app is reachable both by mouse and by keyboard
(every nav button is a normal, tab-reachable `QPushButton`).
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QHBoxLayout,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from vpsych.app.screens.calibration_wizard import CalibrationWizardScreen
from vpsych.app.screens.catalog import TestCatalogScreen
from vpsych.app.screens.data import DataScreen
from vpsych.app.screens.home import HomeScreen
from vpsych.app.screens.results import ResultsScreen
from vpsych.app.screens.run import RunScreen
from vpsych.app.screens.session_builder import SessionBuilderScreen
from vpsych.app.state import AppState
from vpsych.data import catalog


class MainWindow(QMainWindow):
    """Top-level application window."""

    def __init__(
        self, state: AppState, run_scratch_dir: Path, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._state = state
        self.setWindowTitle("vpsych")
        self.setAccessibleName("vpsych main window")

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        nav_panel = QWidget()
        nav_panel.setFixedWidth(180)
        nav_layout = QVBoxLayout(nav_panel)
        nav_layout.setContentsMargins(12, 24, 12, 24)
        nav_layout.setSpacing(4)
        layout.addWidget(nav_panel)

        self.stack = QStackedWidget()
        layout.addWidget(self.stack, stretch=1)

        self.home_screen = HomeScreen(state)
        self.calibration_screen = CalibrationWizardScreen(state)
        self.catalog_screen = TestCatalogScreen(state)
        self.session_builder_screen = SessionBuilderScreen(state, run_scratch_dir / "plans")
        self.run_screen = RunScreen(state, run_scratch_dir / "status")
        self.results_screen = ResultsScreen(state)
        self.data_screen = DataScreen(state)

        self._pages: dict[str, QWidget] = {
            "Home": self.home_screen,
            "Calibration": self.calibration_screen,
            "Test catalog": self.catalog_screen,
            "Session builder": self.session_builder_screen,
            "Run": self.run_screen,
            "Results": self.results_screen,
            "Data": self.data_screen,
        }
        self._nav_buttons: dict[QWidget, QPushButton] = {}
        for label, page in self._pages.items():
            self.stack.addWidget(page)
            button = QPushButton(label)
            button.setAccessibleName(f"Go to {label}")
            button.setCheckable(True)
            button.clicked.connect(lambda _checked, w=page: self.show_page(w))
            nav_layout.addWidget(button)
            self._nav_buttons[page] = button
        nav_layout.addStretch(1)

        self.setStatusBar(QStatusBar())

        self._wire_navigation()
        self.show_page(self.home_screen)

    def show_page(self, page: QWidget) -> None:
        """Switch the stacked widget to `page` and reflect it in the nav panel."""
        self.stack.setCurrentWidget(page)
        for candidate, button in self._nav_buttons.items():
            button.setChecked(candidate is page)

    def _wire_navigation(self) -> None:
        self.home_screen.navigate_calibration.connect(
            lambda: self.show_page(self.calibration_screen)
        )
        self.home_screen.navigate_catalog.connect(lambda: self.show_page(self.catalog_screen))
        self.home_screen.navigate_session_builder.connect(
            lambda: self.show_page(self.session_builder_screen)
        )
        self.home_screen.navigate_data.connect(lambda: self.show_page(self.data_screen))

        self.calibration_screen.calibration_saved.connect(lambda: self.show_page(self.home_screen))

        self.catalog_screen.test_added.connect(self.session_builder_screen.add_test)
        self.catalog_screen.test_added.connect(
            lambda task_id: self.statusBar().showMessage(
                f"Added {task_id} to the session plan.", 4000
            )
        )

        self.session_builder_screen.session_started.connect(self._on_session_started)

        self.run_screen.run_finished.connect(self._on_run_finished)

    def _on_session_started(self, plan_path: Path, plan: object) -> None:
        self.show_page(self.run_screen)
        self.run_screen.start(plan_path, plan)  # type: ignore[arg-type]

    def _on_run_finished(self, exit_code: object) -> None:
        del exit_code
        # The runner subprocess writes session files but never touches the
        # catalog index the Results screen reads, so re-index before refreshing.
        try:
            catalog.rebuild_catalog(self._state.data_root)
        except Exception:
            pass  # raw data is intact; `rebuild-catalog` in the data CLI can recover
        self.results_screen.refresh()
