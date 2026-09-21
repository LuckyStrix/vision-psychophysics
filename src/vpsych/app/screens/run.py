"""Run screen: launches `vpsych-run` as a subprocess and shows live progress.

Never opens a stimulus window itself -- see `vpsych.app.runner_process` for
the `QProcess` launch and status-file polling this screen displays.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from vpsych.app.runner_process import RunnerProcessController
from vpsych.app.state import AppState
from vpsych.app.viewmodels.run import explain_exit, progress_fraction, progress_text
from vpsych.data.schemas import SessionPlan
from vpsych.runner.status import RunnerExitCode, RunnerStatus


class RunScreen(QWidget):
    """Run screen: progress display for one launched session.

    Signals:
        run_finished: Emitted with the `RunnerExitCode` once the subprocess
            exits (the main window may use this to unlock navigation back
            to Results).
    """

    run_finished = Signal(object)

    def __init__(self, state: AppState, run_dir: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._run_dir = run_dir
        self._controller: RunnerProcessController | None = None
        self.setAccessibleName("Run screen")

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(12)

        heading = QLabel("Session in progress")
        heading.setProperty("role", "heading")
        root.addWidget(heading)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setAccessibleName("Session progress")
        root.addWidget(self.progress_bar)

        self.status_label = QLabel("Not started.")
        self.status_label.setWordWrap(True)
        self.status_label.setAccessibleName("Current session status")
        root.addWidget(self.status_label)

        self.outcome_label = QLabel()
        self.outcome_label.setWordWrap(True)
        self.outcome_label.setStyleSheet("font-weight: 600;")
        root.addWidget(self.outcome_label)

        self.abort_button = QPushButton("Abort session")
        self.abort_button.setAccessibleName("Abort session")
        self.abort_button.clicked.connect(self._on_abort)
        root.addWidget(self.abort_button)

        root.addStretch(1)

    def start(
        self, session_plan_path: Path, plan: SessionPlan, simulate: str | None = None
    ) -> None:
        """Start a session for `plan` (already written to `session_plan_path`).

        Args:
            session_plan_path: Path to the written `SessionPlan` JSON.
            plan: The plan being run (used only for display, not re-sent to
                the subprocess -- the subprocess reads it from
                `session_plan_path` itself).
            simulate: Forwarded to the runner's `--simulate` flag; only ever
                set by tests/demos, never a real session.
        """
        self._run_dir.mkdir(parents=True, exist_ok=True)
        status_path = self._run_dir / f"status-{plan.participant_id}-{plan.seed}.json"
        self.status_label.setText("Starting runner subprocess...")
        self.outcome_label.clear()
        self.progress_bar.setValue(0)
        self.abort_button.setEnabled(True)

        self._controller = RunnerProcessController(
            session_plan_path,
            status_path,
            data_root=self._state.data_root,
            simulate=simulate,
        )
        self._controller.status_updated.connect(self._on_status_updated)
        self._controller.finished.connect(self._on_finished)
        self._controller.start()

    def _on_abort(self) -> None:
        if self._controller is not None:
            self._controller.abort()
        self.abort_button.setEnabled(False)

    def _on_status_updated(self, status: RunnerStatus) -> None:
        self.status_label.setText(progress_text(status))
        fraction = progress_fraction(status)
        if fraction is not None:
            self.progress_bar.setValue(round(fraction * 100))

    def _on_finished(self, exit_code: RunnerExitCode) -> None:
        self.abort_button.setEnabled(False)
        last_status = self._controller.last_status if self._controller else None
        outcome = explain_exit(exit_code, last_status)
        self.outcome_label.setText(f"{outcome.title}. {outcome.explanation}")
        self.outcome_label.setAccessibleName(f"{outcome.title}. {outcome.explanation}")
        if outcome.is_success:
            self.progress_bar.setValue(100)
        self._state.calibration_changed.emit()  # cheap way to nudge dependent screens to refresh
        self.run_finished.emit(exit_code)
