"""Session builder screen: choose tests, order, eye, viewing distance -> a `SessionPlan`."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from vpsych.app.state import AppState
from vpsych.app.viewmodels.session_plan import (
    SessionBuilderError,
    SessionBuilderState,
    build_session_plan,
    estimated_total_minutes,
    validate_builder_state,
    write_session_plan,
)
from vpsych.data.schemas import SessionPlan
from vpsych.tests_catalog.base import discover_tests, get_test


class _PlannedTestRow(QWidget):
    """One row of the plan list: eye picker and viewing-distance entry for one planned test."""

    changed = Signal()
    remove_requested = Signal(object)  # emits self

    def __init__(
        self, task_id: str, allowed_eyes: list[str], parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.task_id = task_id
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        label = QLabel(task_id)
        layout.addWidget(label, stretch=1)

        self.eye_combo = QComboBox()
        self.eye_combo.addItems(allowed_eyes)
        self.eye_combo.setAccessibleName(f"Eye for {task_id}")
        self.eye_combo.currentIndexChanged.connect(lambda _i: self.changed.emit())
        layout.addWidget(self.eye_combo)

        self.distance_spin = QDoubleSpinBox()
        self.distance_spin.setRange(1.0, 1000.0)
        self.distance_spin.setValue(60.0)
        self.distance_spin.setSuffix(" cm")
        self.distance_spin.setAccessibleName(f"Viewing distance for {task_id}")
        self.distance_spin.valueChanged.connect(lambda _v: self.changed.emit())
        layout.addWidget(self.distance_spin)

        remove_button = QPushButton("Remove")
        remove_button.setAccessibleName(f"Remove {task_id} from session")
        remove_button.clicked.connect(lambda: self.remove_requested.emit(self))
        layout.addWidget(remove_button)

    @property
    def eye(self) -> str:
        return self.eye_combo.currentText()

    @property
    def viewing_distance_cm(self) -> float:
        return self.distance_spin.value()


class SessionBuilderScreen(QWidget):
    """Session builder: choose tests, order, eye, and viewing distance for a session.

    Signals:
        session_started: Emitted with `(session_plan_path, plan)` once a
            valid plan has been written and the user asked to start the run
            (the main window switches to the Run screen on this signal).
    """

    session_started = Signal(object, object)

    def __init__(self, state: AppState, plan_dir: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._plan_dir = plan_dir
        self._builder_state = SessionBuilderState()
        self._rows: list[_PlannedTestRow] = []
        self.setAccessibleName("Session builder screen")
        discover_tests()

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(12)

        heading = QLabel("Build a session")
        heading.setProperty("role", "heading")
        root.addWidget(heading)

        self.participant_label = QLabel()
        root.addWidget(self.participant_label)

        self.plan_list_container = QVBoxLayout()
        root.addLayout(self.plan_list_container)

        self.empty_label = QLabel("No tests added yet. Add tests from the test catalog screen.")
        self.empty_label.setProperty("role", "caption")
        root.addWidget(self.empty_label)

        ordering_row = QHBoxLayout()
        ordering_label = QLabel("Order")
        ordering_row.addWidget(ordering_label)
        self.fixed_radio = QRadioButton("Fixed (as listed)")
        self.fixed_radio.setChecked(True)
        self.fixed_radio.toggled.connect(self._on_ordering_changed)
        ordering_row.addWidget(self.fixed_radio)
        self.randomized_radio = QRadioButton("Randomized")
        ordering_row.addWidget(self.randomized_radio)
        ordering_row.addStretch(1)
        root.addLayout(ordering_row)

        self.summary_label = QLabel()
        self.summary_label.setProperty("role", "caption")
        root.addWidget(self.summary_label)

        self.problems_label = QLabel()
        self.problems_label.setWordWrap(True)
        self.problems_label.setStyleSheet("font-weight: 600;")
        root.addWidget(self.problems_label)

        self.start_button = QPushButton("Start session")
        self.start_button.setProperty("role", "primary")
        self.start_button.setAccessibleName("Start session")
        self.start_button.clicked.connect(self._on_start)
        root.addWidget(self.start_button)

        root.addStretch(1)

        self._state.participant_changed.connect(lambda _pid: self._refresh_participant_label())
        self._refresh_participant_label()
        self._refresh_summary()

    def _refresh_participant_label(self) -> None:
        pid = self._state.participant_id
        self._builder_state.participant_id = pid
        self.participant_label.setText(
            f"Participant: {pid}" if pid else "No participant selected (choose one on Home)."
        )
        if hasattr(self, "start_button"):  # first call happens before the summary widgets exist
            self._refresh_summary()

    def add_test(self, task_id: str) -> None:
        """Add a test to the plan (called when a catalog card's "Add to session" fires)."""
        spec = get_test(task_id).spec
        self._builder_state.add_test(task_id, eye=spec.allowed_eyes[0])
        row = _PlannedTestRow(task_id, [str(e) for e in spec.allowed_eyes])
        row.changed.connect(self._sync_row_into_state)
        row.remove_requested.connect(self._remove_row)
        self._rows.append(row)
        self.plan_list_container.addWidget(row)
        self.empty_label.setVisible(False)
        self._sync_row_into_state()

    def _remove_row(self, row: _PlannedTestRow) -> None:
        index = self._rows.index(row)
        self._rows.pop(index)
        self._builder_state.remove_test(index)
        self.plan_list_container.removeWidget(row)
        row.deleteLater()
        self.empty_label.setVisible(not self._rows)
        self._refresh_summary()

    def _sync_row_into_state(self) -> None:
        for state_row, widget_row in zip(self._builder_state.tests, self._rows, strict=True):
            state_row.eye = widget_row.eye
            state_row.viewing_distance_cm = widget_row.viewing_distance_cm
        self._refresh_summary()

    def _on_ordering_changed(self) -> None:
        self._builder_state.ordering = "fixed" if self.fixed_radio.isChecked() else "randomized"
        self._refresh_summary()

    def _refresh_summary(self) -> None:
        durations = {}
        for row in self._builder_state.tests:
            try:
                durations[row.task_id] = get_test(row.task_id).spec.estimated_minutes
            except KeyError:
                continue
        total = estimated_total_minutes(self._builder_state, durations)
        self.summary_label.setText(
            f"{len(self._builder_state.tests)} test(s) planned, ~{total:g} min estimated."
        )
        problems = validate_builder_state(self._builder_state)
        self.problems_label.setText("\n".join(problems))
        self.start_button.setEnabled(not problems)

    def _on_start(self) -> None:
        try:
            plan: SessionPlan = build_session_plan(self._builder_state)
        except SessionBuilderError as exc:
            QMessageBox.warning(self, "Cannot start session", str(exc))
            return
        self._plan_dir.mkdir(parents=True, exist_ok=True)
        plan_path = self._plan_dir / f"session-plan-{plan.participant_id}-{plan.seed}.json"
        write_session_plan(plan, plan_path)
        self.session_started.emit(plan_path, plan)
