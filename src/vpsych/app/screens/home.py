"""Home screen: participant picker, recent sessions, calibration status badge."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from vpsych.app.state import AppState
from vpsych.app.viewmodels.calibration import calibration_badge
from vpsych.app.viewmodels.home import build_recent_sessions, format_participant_label
from vpsych.data import catalog, dataset


class NewParticipantDialog(QDialog):
    """Small dialog to create a new pseudonymous participant.

    Collects only the fields `vpsych.data.schemas.Participant` allows
    (`year_of_birth`, `sex`, `refractive_correction`, `notes`) -- no name or
    other identifying field is ever asked for.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("New participant")
        self.setAccessibleName("New participant dialog")

        layout = QVBoxLayout(self)
        note = QLabel(
            "Only a pseudonymous ID is created. Do not enter a name or other identifying "
            "information below."
        )
        note.setWordWrap(True)
        note.setProperty("role", "caption")
        layout.addWidget(note)

        form = QFormLayout()
        self.year_spin = QSpinBox()
        self.year_spin.setRange(0, 2100)
        self.year_spin.setSpecialValueText("Not recorded")
        self.year_spin.setValue(0)
        self.year_spin.setAccessibleName("Year of birth")
        form.addRow("Year of birth", self.year_spin)

        self.sex_edit = QLineEdit()
        self.sex_edit.setAccessibleName("Self-reported sex")
        form.addRow("Sex", self.sex_edit)

        self.refractive_edit = QLineEdit()
        self.refractive_edit.setPlaceholderText("e.g. glasses, contacts, none")
        self.refractive_edit.setAccessibleName("Refractive correction")
        form.addRow("Refractive correction", self.refractive_edit)

        self.notes_edit = QTextEdit()
        self.notes_edit.setAccessibleName("Pseudonymous notes")
        self.notes_edit.setFixedHeight(60)
        form.addRow("Notes", self.notes_edit)

        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def year_of_birth(self) -> int | None:
        """The entered birth year, or `None` if left at the special "not recorded" value."""
        return self.year_spin.value() or None

    def sex(self) -> str | None:
        """The entered self-reported sex, or `None` if left blank."""
        return self.sex_edit.text().strip() or None

    def refractive_correction(self) -> str | None:
        """The entered refractive correction, or `None` if left blank."""
        return self.refractive_edit.text().strip() or None

    def notes(self) -> str | None:
        """The entered pseudonymous notes, or `None` if left blank."""
        return self.notes_edit.toPlainText().strip() or None


class HomeScreen(QWidget):
    """Home screen: pick/create a participant, see recent sessions and calibration status.

    Signals:
        navigate_calibration: Emitted when "Run calibration" is activated.
        navigate_catalog: Emitted when "Browse tests" is activated.
        navigate_session_builder: Emitted when "Build a session" is activated.
        navigate_data: Emitted when "Browse data" is activated.
    """

    navigate_calibration = Signal()
    navigate_catalog = Signal()
    navigate_session_builder = Signal()
    navigate_data = Signal()

    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self.setAccessibleName("Home screen")

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(16)

        heading = QLabel("vpsych")
        heading.setProperty("role", "heading")
        heading.setAccessibleName("vpsych home")
        root.addWidget(heading)

        # --- Participant picker -------------------------------------------------
        picker_row = QHBoxLayout()
        picker_label = QLabel("Participant")
        picker_label.setBuddy(self._make_participant_combo())
        picker_row.addWidget(picker_label)
        picker_row.addWidget(self.participant_combo, stretch=1)
        self.new_participant_button = QPushButton("New participant")
        self.new_participant_button.setAccessibleName("Create new participant")
        self.new_participant_button.clicked.connect(self._on_new_participant)
        picker_row.addWidget(self.new_participant_button)
        root.addLayout(picker_row)

        # --- Calibration badge ----------------------------------------------------
        self.calibration_card = QFrame()
        self.calibration_card.setProperty("role", "card")
        cal_layout = QVBoxLayout(self.calibration_card)
        cal_title = QLabel("Calibration status")
        cal_title.setProperty("role", "subheading")
        cal_layout.addWidget(cal_title)
        self.calibration_summary_label = QLabel()
        self.calibration_summary_label.setWordWrap(True)
        cal_layout.addWidget(self.calibration_summary_label)
        self.calibration_stale_label = QLabel()
        self.calibration_stale_label.setWordWrap(True)
        self.calibration_stale_label.setStyleSheet("font-weight: 600;")
        cal_layout.addWidget(self.calibration_stale_label)
        self.calibrate_button = QPushButton("Run calibration")
        self.calibrate_button.setProperty("role", "primary")
        self.calibrate_button.setAccessibleName("Run calibration wizard")
        self.calibrate_button.clicked.connect(self.navigate_calibration)
        cal_layout.addWidget(self.calibrate_button)
        root.addWidget(self.calibration_card)

        # --- Navigation buttons ---------------------------------------------------
        nav_row = QHBoxLayout()
        self.catalog_button = QPushButton("Browse tests")
        self.catalog_button.setAccessibleName("Browse the test catalog")
        self.catalog_button.clicked.connect(self.navigate_catalog)
        nav_row.addWidget(self.catalog_button)

        self.session_button = QPushButton("Build a session")
        self.session_button.setAccessibleName("Build a session")
        self.session_button.clicked.connect(self.navigate_session_builder)
        nav_row.addWidget(self.session_button)

        self.data_button = QPushButton("Browse data")
        self.data_button.setAccessibleName("Browse recorded data")
        self.data_button.clicked.connect(self.navigate_data)
        nav_row.addWidget(self.data_button)
        root.addLayout(nav_row)

        # --- Recent sessions --------------------------------------------------------
        recent_title = QLabel("Recent sessions")
        recent_title.setProperty("role", "subheading")
        root.addWidget(recent_title)
        self.recent_sessions_list = QListWidget()
        self.recent_sessions_list.setAccessibleName("Recent sessions for this participant")
        root.addWidget(self.recent_sessions_list, stretch=1)

        self._state.participant_changed.connect(lambda _pid: self.refresh())
        self._state.calibration_changed.connect(self._refresh_calibration_badge)
        self.refresh()

    def _make_participant_combo(self) -> QComboBox:
        self.participant_combo = QComboBox()
        self.participant_combo.setAccessibleName("Select participant")
        self.participant_combo.currentIndexChanged.connect(self._on_participant_selected)
        return self.participant_combo

    def _on_participant_selected(self, index: int) -> None:
        participant_id = self.participant_combo.itemData(index)
        self._state.set_participant(participant_id)

    def _on_new_participant(self) -> None:
        dialog = NewParticipantDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            participant = dataset.create_participant(
                root=self._state.data_root,
                year_of_birth=dialog.year_of_birth(),
                sex=dialog.sex(),
                refractive_correction=dialog.refractive_correction(),
                notes=dialog.notes(),
            )
            self.refresh(select_participant_id=participant.participant_id)

    def refresh(self, select_participant_id: str | None = None) -> None:
        """Reload participants and this participant's recent sessions from the data layer."""
        participants = dataset.list_participants(self._state.data_root)
        self.participant_combo.blockSignals(True)
        self.participant_combo.clear()
        for p in participants:
            self.participant_combo.addItem(
                format_participant_label(p.participant_id, p.year_of_birth, p.sex),
                p.participant_id,
            )
        target = select_participant_id or self._state.participant_id
        if target:
            idx = self.participant_combo.findData(target)
            if idx >= 0:
                self.participant_combo.setCurrentIndex(idx)
        self.participant_combo.blockSignals(False)

        current_data = self.participant_combo.currentData()
        if current_data and current_data != self._state.participant_id:
            self._state.set_participant(current_data)
        elif not participants:
            self._state.set_participant(None)

        self._refresh_recent_sessions()
        self._refresh_calibration_badge()

    def _refresh_recent_sessions(self) -> None:
        self.recent_sessions_list.clear()
        pid = self._state.participant_id
        if not pid:
            item = QListWidgetItem("No participant selected.")
            self.recent_sessions_list.addItem(item)
            return
        sessions = catalog.sessions_for_participant(pid, self._state.data_root)
        rows = build_recent_sessions(sessions)
        if not rows:
            self.recent_sessions_list.addItem(QListWidgetItem("No sessions recorded yet."))
            return
        for row in rows:
            text = f"{row.session_id} — {row.status_text} — {row.started_text} — cal {row.calibration_grade_text}"
            item = QListWidgetItem(text)
            item.setData(0x0100, row.session_id)  # Qt.ItemDataRole.UserRole
            self.recent_sessions_list.addItem(item)

    def _refresh_calibration_badge(self) -> None:
        vm = calibration_badge(self._state.latest_calibration)
        self.calibration_summary_label.setText(vm.summary_text)
        self.calibration_summary_label.setAccessibleName(vm.summary_text)
        if vm.stale_warning:
            self.calibration_stale_label.setText(f"⚠ {vm.stale_warning}")
            self.calibration_stale_label.setVisible(True)
        else:
            self.calibration_stale_label.clear()
            self.calibration_stale_label.setVisible(False)
