"""Calibration wizard: geometry, gamma, color, environment -> an immutable `Calibration`.

Each step edits `vpsych.app.viewmodels.calibration_wizard.CalibrationWizardState`;
"Save calibration" on the summary step assembles and stores the result via
`vpsych.data.dataset.save_calibration`.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

import vpsych
from vpsych.app.state import AppState
from vpsych.app.viewmodels.calibration import calibration_badge, grade_c_limitations
from vpsych.app.viewmodels.calibration_wizard import (
    CalibrationWizardState,
    WizardStepError,
    environment_checklist_warnings,
)
from vpsych.app.widgets.card_match import CardMatchWidget
from vpsych.core.calibration.geometry import DisplayQueryError, query_os_resolution_refresh
from vpsych.core.calibration.models import (
    Calibration,
    ColorCalibration,
    EnvironmentChecklist,
    GammaCalibration,
)
from vpsych.core.display import DisplayGeometry
from vpsych.data.dataset import save_calibration


class _GeometryStep(QWidget):
    def __init__(self, state: CalibrationWizardState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        layout = QVBoxLayout(self)

        title = QLabel("1. Geometry")
        title.setProperty("role", "subheading")
        layout.addWidget(title)

        detect_row = QHBoxLayout()
        self.detect_button = QPushButton("Detect resolution and refresh rate")
        self.detect_button.setAccessibleName("Detect display resolution and refresh rate")
        self.detect_button.clicked.connect(self._on_detect)
        detect_row.addWidget(self.detect_button)
        self.detect_status_label = QLabel()
        detect_row.addWidget(self.detect_status_label, stretch=1)
        layout.addLayout(detect_row)

        form = QFormLayout()
        self.width_px_spin = _int_spin(1, 20000, 1920)
        form.addRow("Horizontal resolution (px)", self.width_px_spin)
        self.height_px_spin = _int_spin(1, 20000, 1080)
        form.addRow("Vertical resolution (px)", self.height_px_spin)
        self.refresh_spin = QDoubleSpinBox()
        self.refresh_spin.setRange(1.0, 1000.0)
        self.refresh_spin.setValue(60.0)
        self.refresh_spin.setSuffix(" Hz")
        form.addRow("Refresh rate", self.refresh_spin)
        self.distance_spin = QDoubleSpinBox()
        self.distance_spin.setRange(1.0, 1000.0)
        self.distance_spin.setValue(60.0)
        self.distance_spin.setSuffix(" cm")
        form.addRow("Viewing distance", self.distance_spin)
        self.height_cm_spin = QDoubleSpinBox()
        self.height_cm_spin.setRange(1.0, 300.0)
        self.height_cm_spin.setValue(30.0)
        self.height_cm_spin.setSuffix(" cm")
        form.addRow("Physical screen height", self.height_cm_spin)
        layout.addLayout(form)

        card_label = QLabel(
            "Hold a real credit/debit card (or ID-1 card) flat against the screen. Drag the "
            "rectangle below (or use the arrow keys) until its width matches the card."
        )
        card_label.setWordWrap(True)
        layout.addWidget(card_label)
        self.card_widget = CardMatchWidget()
        self.card_widget.width_changed.connect(self._on_card_changed)
        layout.addWidget(self.card_widget)
        self.width_estimate_label = QLabel()
        self.width_estimate_label.setProperty("role", "caption")
        layout.addWidget(self.width_estimate_label)

        for spin in (self.width_px_spin, self.height_px_spin):
            spin.valueChanged.connect(self._on_card_changed)

        self._on_card_changed()

    def _on_detect(self) -> None:
        try:
            width_px, height_px, refresh_hz = query_os_resolution_refresh()
        except DisplayQueryError as exc:
            self.detect_status_label.setText(f"Could not detect display: {exc}")
            return
        self.width_px_spin.setValue(width_px)
        self.height_px_spin.setValue(height_px)
        self.refresh_spin.setValue(refresh_hz)
        self.detect_status_label.setText("Detected from the OS.")

    def _on_card_changed(self) -> None:
        self._state.geometry.card_width_px = self.card_widget.card_width_px
        self._state.geometry.width_px = self.width_px_spin.value()
        estimated = self._state.geometry.resolve_width_cm()
        if estimated is not None:
            self.width_estimate_label.setText(f"Estimated screen width: {estimated:.1f} cm")

    def commit(self) -> None:
        """Write this step's widget values into the shared wizard state."""
        g = self._state.geometry
        g.width_px = self.width_px_spin.value()
        g.height_px = self.height_px_spin.value()
        g.refresh_hz = self.refresh_spin.value()
        g.card_width_px = self.card_widget.card_width_px
        g.height_cm = self.height_cm_spin.value()
        g.viewing_distance_cm = self.distance_spin.value()


class _GammaStep(QWidget):
    def __init__(self, state: CalibrationWizardState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        layout = QVBoxLayout(self)

        title = QLabel("2. Luminance / gamma")
        title.setProperty("role", "subheading")
        layout.addWidget(title)

        self.none_radio = QRadioButton("None (grade C, sRGB gamma assumed)")
        self.psychophysical_radio = QRadioButton("Psychophysical estimate, no photometer (grade B)")
        self.photometer_radio = QRadioButton("Photometer measurement (grade A)")
        self.none_radio.setChecked(True)
        for r in (self.none_radio, self.psychophysical_radio, self.photometer_radio):
            layout.addWidget(r)

        note = QLabel(
            "Grade A needs a supported photometer/colorimeter connected via PsychoPy hardware "
            "support. Grade B needs no hardware, only the observer's own brightness judgments. "
            "See docs/CALIBRATION.md for the full procedure; this wizard records whichever "
            "method you completed."
        )
        note.setWordWrap(True)
        note.setProperty("role", "caption")
        layout.addWidget(note)

        self.detect_photometer_button = QPushButton("Attempt photometer detection")
        self.detect_photometer_button.clicked.connect(self._on_detect_photometer)
        layout.addWidget(self.detect_photometer_button)
        self.photometer_status_label = QLabel()
        self.photometer_status_label.setWordWrap(True)
        layout.addWidget(self.photometer_status_label)

        layout.addStretch(1)

    def _on_detect_photometer(self) -> None:
        try:
            from vpsych.core.calibration.photometer import open_psychopy_photometer

            open_psychopy_photometer("PR655")
        except Exception as exc:  # pragma: no cover - hardware-dependent
            self.photometer_status_label.setText(
                f"No photometer detected ({exc}). Use the psychophysical method instead, or "
                "skip gamma calibration (grade C)."
            )
            return
        self.photometer_status_label.setText(  # pragma: no cover - hardware-dependent
            "Photometer detected. Full guided measurement is not yet available in this wizard "
            "build; run tools/gamma_measure.py to produce measured points."
        )

    def commit(self) -> None:
        """Write this step's widget values into the shared wizard state."""
        if self.photometer_radio.isChecked():
            self._state.gamma.mode = "photometer"
        elif self.psychophysical_radio.isChecked():
            self._state.gamma.mode = "psychophysical"
        else:
            self._state.gamma.mode = "none"


class _ColorStep(QWidget):
    def __init__(self, state: CalibrationWizardState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        layout = QVBoxLayout(self)

        title = QLabel("3. Color")
        title.setProperty("role", "subheading")
        layout.addWidget(title)

        self.assumed_radio = QRadioButton("sRGB assumed, uncalibrated (grade C)")
        self.measured_radio = QRadioButton(
            "Measured with a spectroradiometer/colorimeter (grade A)"
        )
        self.assumed_radio.setChecked(True)
        layout.addWidget(self.assumed_radio)
        layout.addWidget(self.measured_radio)

        note = QLabel(
            "Measured-primaries entry is not yet available in this wizard build; choose "
            "sRGB-assumed here, or record measured primaries via the calibration data model "
            "directly (vpsych.core.calibration.models.ColorCalibration)."
        )
        note.setWordWrap(True)
        note.setProperty("role", "caption")
        layout.addWidget(note)
        layout.addStretch(1)

    def commit(self) -> None:
        """Write this step's widget values into the shared wizard state."""
        self._state.color.measured = self.measured_radio.isChecked()


class _EnvironmentStep(QWidget):
    def __init__(self, state: CalibrationWizardState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        layout = QVBoxLayout(self)

        title = QLabel("4. Environment checklist")
        title.setProperty("role", "subheading")
        layout.addWidget(title)

        self.room_check = QCheckBox("Room lighting is dim and free of glare on the screen")
        self.warmup_check = QCheckBox("Monitor has been powered on for at least 15 minutes")
        self.night_light_check = QCheckBox("OS night-light / blue-light filter is disabled")
        self.hdr_check = QCheckBox("HDR / adaptive-brightness / auto-contrast is disabled")
        for c in (self.room_check, self.warmup_check, self.night_light_check, self.hdr_check):
            layout.addWidget(c)

        self.notes_edit = QTextEdit()
        self.notes_edit.setPlaceholderText(
            "Optional notes (e.g. ambient light reading). Keep pseudonymous."
        )
        self.notes_edit.setFixedHeight(60)
        layout.addWidget(self.notes_edit)
        layout.addStretch(1)

    def commit(self) -> None:
        """Write this step's widget values into the shared wizard state."""
        self._state.environment = EnvironmentChecklist(
            room_lighting_controlled=self.room_check.isChecked(),
            monitor_warmed_up=self.warmup_check.isChecked(),
            night_light_disabled=self.night_light_check.isChecked(),
            hdr_disabled=self.hdr_check.isChecked(),
            notes=self.notes_edit.toPlainText().strip() or None,
        )


class _SummaryStep(QWidget):
    save_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        title = QLabel("5. Summary")
        title.setProperty("role", "subheading")
        layout.addWidget(title)

        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        self.warnings_label = QLabel()
        self.warnings_label.setWordWrap(True)
        self.warnings_label.setStyleSheet("font-weight: 600;")
        layout.addWidget(self.warnings_label)

        self.limitations_label = QLabel()
        self.limitations_label.setWordWrap(True)
        layout.addWidget(self.limitations_label)

        self.save_button = QPushButton("Save calibration")
        self.save_button.setProperty("role", "primary")
        self.save_button.clicked.connect(self.save_requested)
        layout.addWidget(self.save_button)
        layout.addStretch(1)


class CalibrationWizardScreen(QWidget):
    """The full calibration wizard: five steps, Back/Next, and a final save.

    Signals:
        calibration_saved: Emitted after a calibration is successfully
            saved via `vpsych.data.dataset.save_calibration`.
    """

    calibration_saved = Signal()

    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._wizard_state = CalibrationWizardState()
        self.setAccessibleName("Calibration wizard")

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(12)

        heading = QLabel("Calibration wizard")
        heading.setProperty("role", "heading")
        root.addWidget(heading)

        self.stack = QStackedWidget()
        self._geometry_step = _GeometryStep(self._wizard_state)
        self._gamma_step = _GammaStep(self._wizard_state)
        self._color_step = _ColorStep(self._wizard_state)
        self._environment_step = _EnvironmentStep(self._wizard_state)
        self._summary_step = _SummaryStep()
        self._summary_step.save_requested.connect(self._on_save)
        for step in (
            self._geometry_step,
            self._gamma_step,
            self._color_step,
            self._environment_step,
            self._summary_step,
        ):
            self.stack.addWidget(step)
        root.addWidget(self.stack, stretch=1)

        nav_row = QHBoxLayout()
        self.back_button = QPushButton("Back")
        self.back_button.clicked.connect(self._on_back)
        nav_row.addWidget(self.back_button)
        nav_row.addStretch(1)
        self.next_button = QPushButton("Next")
        self.next_button.setProperty("role", "primary")
        self.next_button.clicked.connect(self._on_next)
        nav_row.addWidget(self.next_button)
        root.addLayout(nav_row)

        self._update_nav()

    def _on_back(self) -> None:
        idx = self.stack.currentIndex()
        if idx > 0:
            self.stack.setCurrentIndex(idx - 1)
        self._update_nav()

    def _on_next(self) -> None:
        idx = self.stack.currentIndex()
        committing_steps = (
            self._geometry_step,
            self._gamma_step,
            self._color_step,
            self._environment_step,
        )
        if idx < len(committing_steps):
            committing_steps[idx].commit()
        if idx == self.stack.count() - 2:  # about to enter the summary step
            self._refresh_summary()
        if idx < self.stack.count() - 1:
            self.stack.setCurrentIndex(idx + 1)
        self._update_nav()

    def _update_nav(self) -> None:
        idx = self.stack.currentIndex()
        self.back_button.setEnabled(idx > 0)
        is_last = idx == self.stack.count() - 1
        self.next_button.setVisible(not is_last)

    def _refresh_summary(self) -> None:
        try:
            geometry = self._wizard_state.geometry.build()
        except WizardStepError as exc:
            self._summary_step.summary_label.setText(f"Geometry incomplete: {exc}")
            self._summary_step.save_button.setEnabled(False)
            return
        try:
            gamma = self._wizard_state.gamma.build()
            color = self._wizard_state.color.build()
        except WizardStepError as exc:
            self._summary_step.summary_label.setText(str(exc))
            self._summary_step.save_button.setEnabled(False)
            return

        from vpsych.core.calibration.models import grade_color, grade_luminance

        self._summary_step.summary_label.setText(
            f"Luminance grade {grade_luminance(gamma.method)} ({gamma.method}); "
            f"color grade {grade_color(color.method)} ({color.method}); "
            f"{geometry.width_px}x{geometry.height_px} px, {geometry.refresh_hz:g} Hz, "
            f"viewing distance {geometry.viewing_distance_cm:g} cm."
        )
        warnings = environment_checklist_warnings(self._wizard_state.environment)
        self._summary_step.warnings_label.setText(
            "\n".join(f"⚠ {w}" for w in warnings) if warnings else ""
        )

        limitations = grade_c_limitations(
            geometry,
            self._preview_calibration(geometry, gamma, color),
        )
        if limitations:
            lines = [f"- {lim.name}: {'; '.join(lim.reasons)}" for lim in limitations]
            self._summary_step.limitations_label.setText(
                "This calibration would rule out:\n" + "\n".join(lines)
            )
        else:
            self._summary_step.limitations_label.setText(
                "Every currently visible test could run under this calibration."
            )
        self._summary_step.save_button.setEnabled(True)

    def _preview_calibration(
        self, geometry: DisplayGeometry, gamma: GammaCalibration, color: ColorCalibration
    ) -> Calibration:
        from datetime import datetime, timezone

        env = self._wizard_state.environment or EnvironmentChecklist(
            room_lighting_controlled=False,
            monitor_warmed_up=False,
            night_light_disabled=False,
            hdr_disabled=False,
        )
        return Calibration(
            created_utc=datetime.now(timezone.utc),
            geometry=geometry,
            gamma=gamma,
            color=color,
            environment=env,
            software_version=vpsych.__version__,
        )

    def _on_save(self) -> None:
        try:
            calibration = self._wizard_state.build_calibration(vpsych.__version__)
        except WizardStepError as exc:
            QMessageBox.warning(self, "Cannot save calibration", str(exc))
            return
        save_calibration(calibration, root=self._state.data_root)
        self._state.refresh_calibration()
        QMessageBox.information(
            self,
            "Calibration saved",
            calibration_badge(calibration).summary_text,
        )
        self.calibration_saved.emit()


def _int_spin(lo: int, hi: int, value: int) -> QSpinBox:
    spin = QSpinBox()
    spin.setRange(lo, hi)
    spin.setValue(value)
    return spin
