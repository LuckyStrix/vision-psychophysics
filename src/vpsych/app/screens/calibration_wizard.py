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
    QListWidget,
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
    DEFAULT_PHOTOMETER_GAMMA_LEVELS,
    BisectionSequenceState,
    CalibrationWizardState,
    WizardStepError,
    environment_checklist_warnings,
)
from vpsych.app.widgets.bisection_widget import BisectionWidget
from vpsych.app.widgets.card_match import CardMatchWidget
from vpsych.app.widgets.patch_window import PatchWindow
from vpsych.core.calibration.gamma import fit_gamma
from vpsych.core.calibration.geometry import DisplayQueryError, query_os_resolution_refresh
from vpsych.core.calibration.models import (
    Calibration,
    ColorCalibration,
    EnvironmentChecklist,
    GammaCalibration,
    GammaCalibrationPoint,
    PrimaryChromaticity,
)
from vpsych.core.calibration.photometer import measurement_levels
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
        self._bisection = BisectionSequenceState()
        self._photometer_session: object | None = None  # ArgyllSpotreadSession, once opened
        self._photometer: object | None = None  # ArgyllSpotreadPhotometer, once ready
        self._patch_window: PatchWindow | None = None
        self._photometer_levels: list[float] = []
        self._photometer_level_index = 0
        self._photometer_points: list[GammaCalibrationPoint] = []

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
            "Grade A needs a supported photometer/colorimeter (an X-Rite ColorMunki via "
            "ArgyllCMS's spotread, or a PsychoPy-supported device). Grade B needs no "
            "hardware, only the observer's own brightness judgments. See "
            "docs/CALIBRATION.md for the full procedure."
        )
        note.setWordWrap(True)
        note.setProperty("role", "caption")
        layout.addWidget(note)

        self.mode_stack = QStackedWidget()
        self.mode_stack.addWidget(QWidget())  # "none": nothing to configure
        self.mode_stack.addWidget(self._build_psychophysical_panel())
        self.mode_stack.addWidget(self._build_photometer_panel())
        layout.addWidget(self.mode_stack, stretch=1)

        self.none_radio.toggled.connect(self._on_none_toggled)
        self.psychophysical_radio.toggled.connect(self._on_psychophysical_toggled)
        self.photometer_radio.toggled.connect(self._on_photometer_toggled)

        layout.addStretch(1)

    def _on_none_toggled(self, checked: bool) -> None:
        if checked:
            self.mode_stack.setCurrentIndex(0)

    def _on_psychophysical_toggled(self, checked: bool) -> None:
        if checked:
            self.mode_stack.setCurrentIndex(1)

    def _on_photometer_toggled(self, checked: bool) -> None:
        if checked:
            self.mode_stack.setCurrentIndex(2)

    # -- Psychophysical (grade B) bisection sub-panel -----------------------

    def _build_psychophysical_panel(self) -> QWidget:
        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 8, 0, 0)

        instructions = QLabel(
            "For each level below, use the Up/Down arrow keys (or mouse wheel) to adjust "
            "the gray patch on the right until it looks equally bright as the checkerboard "
            "on the left, then confirm the match."
        )
        instructions.setWordWrap(True)
        panel_layout.addWidget(instructions)

        self.bisection_widget = BisectionWidget()
        self.bisection_widget.set_target_fraction(self._bisection.current_fraction or 0.5)
        panel_layout.addWidget(self.bisection_widget)

        self.bisection_progress_label = QLabel(self._bisection.progress_text)
        panel_layout.addWidget(self.bisection_progress_label)

        button_row = QHBoxLayout()
        self.confirm_match_button = QPushButton("Confirm match")
        self.confirm_match_button.clicked.connect(self._on_confirm_bisection_match)
        button_row.addWidget(self.confirm_match_button)
        self.reset_bisection_button = QPushButton("Reset sequence")
        self.reset_bisection_button.clicked.connect(self._on_reset_bisection)
        button_row.addWidget(self.reset_bisection_button)
        button_row.addStretch(1)
        panel_layout.addLayout(button_row)

        self.bisection_result_label = QLabel()
        self.bisection_result_label.setWordWrap(True)
        panel_layout.addWidget(self.bisection_result_label)
        return panel

    def _on_confirm_bisection_match(self) -> None:
        self._bisection.current_level = self.bisection_widget.gray_level
        try:
            self._bisection.confirm_match()
        except WizardStepError as exc:
            self.bisection_result_label.setText(str(exc))
            return
        self.bisection_progress_label.setText(self._bisection.progress_text)
        if self._bisection.is_complete:
            self.bisection_widget.setEnabled(False)
            self.confirm_match_button.setEnabled(False)
            try:
                estimate = self._bisection.result()
                self.bisection_result_label.setText(
                    f"Estimated gamma: {estimate.gamma:.3f} "
                    f"[{estimate.ci_low:.3f}, {estimate.ci_high:.3f}] "
                    f"({round(estimate.ci_level * 100)}% CI, {estimate.n_levels} levels). "
                    "Click Next to continue."
                )
            except WizardStepError as exc:
                self.bisection_result_label.setText(str(exc))
        else:
            self.bisection_widget.set_target_fraction(self._bisection.current_fraction or 0.5)
            self.bisection_widget.set_gray_level(0.5)
            self.bisection_result_label.setText("")

    def _on_reset_bisection(self) -> None:
        self._bisection = BisectionSequenceState()
        self.bisection_widget.setEnabled(True)
        self.confirm_match_button.setEnabled(True)
        self.bisection_widget.set_target_fraction(self._bisection.current_fraction or 0.5)
        self.bisection_widget.set_gray_level(0.5)
        self.bisection_progress_label.setText(self._bisection.progress_text)
        self.bisection_result_label.setText("")

    # -- Photometer (grade A) guided-measurement sub-panel -------------------

    def _build_photometer_panel(self) -> QWidget:
        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 8, 0, 0)

        self.spectrometer_radio = QRadioButton("ColorMunki Photo (spectrometer)")
        self.colorimeter_radio = QRadioButton("ColorMunki Display (colorimeter)")
        self.spectrometer_radio.setChecked(True)
        panel_layout.addWidget(self.spectrometer_radio)
        panel_layout.addWidget(self.colorimeter_radio)

        button_row = QHBoxLayout()
        self.detect_photometer_button = QPushButton("Detect instrument")
        self.detect_photometer_button.clicked.connect(self._on_detect_photometer)
        button_row.addWidget(self.detect_photometer_button)
        self.open_photometer_button = QPushButton("Open photometer session")
        self.open_photometer_button.clicked.connect(self._on_open_photometer)
        button_row.addWidget(self.open_photometer_button)
        self.take_reading_button = QPushButton("Take reading")
        self.take_reading_button.setEnabled(False)
        self.take_reading_button.clicked.connect(self._on_take_gamma_reading)
        button_row.addWidget(self.take_reading_button)
        button_row.addStretch(1)
        panel_layout.addLayout(button_row)

        self.photometer_status_label = QLabel(
            "Place a small window on the display for the instrument to read, click "
            "'Open photometer session', then 'Take reading' once for each patch level "
            "shown."
        )
        self.photometer_status_label.setWordWrap(True)
        panel_layout.addWidget(self.photometer_status_label)

        self.photometer_readings_list = QListWidget()
        self.photometer_readings_list.setAccessibleName("Photometer readings taken so far")
        panel_layout.addWidget(self.photometer_readings_list, stretch=1)
        return panel

    def _on_detect_photometer(self) -> None:
        from vpsych.core.calibration.argyll import detect_instrument

        result = detect_instrument()
        self.photometer_status_label.setText(result.message)

    def _on_open_photometer(self) -> None:
        from vpsych.core.calibration.argyll import (
            ArgyllCalibrationRequiredError,
            ArgyllNotAvailableError,
            ArgyllSpotreadSession,
            InstrumentKind,
        )

        instrument: InstrumentKind = (
            "colorimeter" if self.colorimeter_radio.isChecked() else "spectrometer"
        )
        session = ArgyllSpotreadSession(instrument=instrument)
        self._photometer_session = session
        try:
            session.open()
        except ArgyllCalibrationRequiredError as exc:
            self._handle_gamma_calibration_prompt(exc)
            return
        except ArgyllNotAvailableError as exc:
            self.photometer_status_label.setText(str(exc))
            return
        self._on_photometer_ready()

    def _handle_gamma_calibration_prompt(self, exc: Exception) -> None:
        from vpsych.core.calibration.argyll import (
            ArgyllCalibrationRequiredError,
            ArgyllNotAvailableError,
            ArgyllReadError,
        )

        prompt_text = getattr(exc, "prompt_text", str(exc))
        QMessageBox.information(
            self,
            "Instrument calibration needed",
            f"{prompt_text}\n\nPosition the instrument as instructed, then click OK.",
        )
        assert self._photometer_session is not None
        try:
            self._photometer_session.confirm_calibration()  # type: ignore[attr-defined]
        except ArgyllCalibrationRequiredError as exc2:
            self._handle_gamma_calibration_prompt(exc2)
            return
        except (ArgyllNotAvailableError, ArgyllReadError) as exc2:
            self.photometer_status_label.setText(f"Calibration failed: {exc2}")
            return
        self._on_photometer_ready()

    def _on_photometer_ready(self) -> None:
        from vpsych.core.calibration.argyll import ArgyllSpotreadPhotometer

        assert self._photometer_session is not None
        self._photometer = ArgyllSpotreadPhotometer(self._photometer_session)  # type: ignore[arg-type]
        self._photometer_levels = measurement_levels(DEFAULT_PHOTOMETER_GAMMA_LEVELS)
        self._photometer_level_index = 0
        self._photometer_points = []
        self.photometer_readings_list.clear()
        self.take_reading_button.setEnabled(True)
        if self._patch_window is None:
            self._patch_window = PatchWindow()
        self._patch_window.set_level(self._photometer_levels[0])
        self._patch_window.show()
        self.photometer_status_label.setText(
            f"Photometer ready. Place the instrument on the patch window, then click "
            f"'Take reading' for each of {len(self._photometer_levels)} levels."
        )

    def _on_take_gamma_reading(self) -> None:
        from vpsych.core.calibration.argyll import (
            ArgyllCalibrationRequiredError,
            ArgyllNotAvailableError,
            ArgyllReadError,
        )

        if self._photometer is None or self._photometer_level_index >= len(self._photometer_levels):
            return
        level = self._photometer_levels[self._photometer_level_index]
        try:
            luminance = self._photometer.measure_luminance_cdm2("gray", level)  # type: ignore[attr-defined]
        except ArgyllCalibrationRequiredError as exc:
            self._handle_gamma_calibration_prompt(exc)
            return
        except (ArgyllNotAvailableError, ArgyllReadError) as exc:
            self.photometer_status_label.setText(f"Reading failed: {exc}")
            return
        self._photometer_points.append(
            GammaCalibrationPoint(input_level=level, luminance_cdm2=luminance)
        )
        self.photometer_readings_list.addItem(f"v={level:.3f} -> {luminance:.3f} cd/m^2")
        self._photometer_level_index += 1
        if self._photometer_level_index >= len(self._photometer_levels):
            self._finish_photometer_gamma()
        else:
            next_level = self._photometer_levels[self._photometer_level_index]
            if self._patch_window is not None:
                self._patch_window.set_level(next_level)
            self.photometer_status_label.setText(
                f"Reading {self._photometer_level_index + 1} of {len(self._photometer_levels)}."
            )

    def _finish_photometer_gamma(self) -> None:
        self.take_reading_button.setEnabled(False)
        if self._patch_window is not None:
            self._patch_window.close()
        try:
            fit = fit_gamma(self._photometer_points)
            self.photometer_status_label.setText(
                f"Measurement complete: gamma={fit.gamma:.3f}, Lmin={fit.lum_min_cdm2:.3f}, "
                f"Lmax={fit.lum_max_cdm2:.3f} cd/m^2 ({len(self._photometer_points)} points). "
                "Click Next to continue."
            )
        except ValueError as exc:
            self.photometer_status_label.setText(f"Measurement complete, but fit failed: {exc}")

    def commit(self) -> None:
        """Write this step's widget values into the shared wizard state."""
        if self.photometer_radio.isChecked():
            self._state.gamma.mode = "photometer"
            self._state.gamma.photometer_points = list(self._photometer_points)
        elif self.psychophysical_radio.isChecked():
            self._state.gamma.mode = "psychophysical"
            self._state.gamma.psychophysical_matches = list(self._bisection.matches)
        else:
            self._state.gamma.mode = "none"


#: (label, key, sRGB default xyY) for the four rows of the color step's manual-entry form.
_PRIMARY_ROWS: tuple[tuple[str, str, tuple[float, float, float]], ...] = (
    ("Red", "red", (0.640, 0.330, 30.0)),
    ("Green", "green", (0.300, 0.600, 65.0)),
    ("Blue", "blue", (0.150, 0.060, 8.0)),
    ("White", "white", (0.3127, 0.3290, 100.0)),
)


class _ColorStep(QWidget):
    def __init__(self, state: CalibrationWizardState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._photometer_session: object | None = None
        self._photometer: object | None = None
        self._patch_window: PatchWindow | None = None
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

        self.measured_panel = QWidget()
        measured_layout = QVBoxLayout(self.measured_panel)
        measured_layout.setContentsMargins(0, 8, 0, 0)

        photometer_row = QHBoxLayout()
        self.detect_color_photometer_button = QPushButton("Detect instrument")
        self.detect_color_photometer_button.clicked.connect(self._on_detect_photometer)
        photometer_row.addWidget(self.detect_color_photometer_button)
        self.measure_primaries_button = QPushButton("Measure primaries with photometer")
        self.measure_primaries_button.clicked.connect(self._on_start_primary_measurement)
        photometer_row.addWidget(self.measure_primaries_button)
        photometer_row.addStretch(1)
        measured_layout.addLayout(photometer_row)

        self.color_photometer_status_label = QLabel(
            "Measuring with a photometer fills the fields below automatically, one primary "
            "at a time. Values can also be entered by hand if you already have them from "
            "another source."
        )
        self.color_photometer_status_label.setWordWrap(True)
        measured_layout.addWidget(self.color_photometer_status_label)

        form = QFormLayout()
        self._spin_boxes: dict[str, dict[str, QDoubleSpinBox]] = {}
        for label, key, (x_default, y_default, lum_default) in _PRIMARY_ROWS:
            row = QHBoxLayout()
            x_spin = _xy_spin(x_default)
            y_spin = _xy_spin(y_default)
            lum_spin = QDoubleSpinBox()
            lum_spin.setRange(0.0, 2000.0)
            lum_spin.setDecimals(3)
            lum_spin.setSuffix(" cd/m^2")
            lum_spin.setValue(lum_default)
            for prefix, spin in (("x", x_spin), ("y", y_spin), ("Y", lum_spin)):
                row.addWidget(QLabel(prefix))
                row.addWidget(spin)
            self._spin_boxes[key] = {"x": x_spin, "y": y_spin, "Y": lum_spin}
            row_widget = QWidget()
            row_widget.setLayout(row)
            form.addRow(label, row_widget)
        measured_layout.addLayout(form)

        layout.addWidget(self.measured_panel)
        self.measured_panel.setVisible(False)
        self.assumed_radio.toggled.connect(
            lambda checked: self.measured_panel.setVisible(not checked)
        )
        layout.addStretch(1)

    def _current_primary(self, key: str) -> PrimaryChromaticity:
        spins = self._spin_boxes[key]
        return PrimaryChromaticity(
            x=spins["x"].value(), y=spins["y"].value(), Y_cdm2=spins["Y"].value()
        )

    def _set_primary_fields(self, key: str, primary: PrimaryChromaticity) -> None:
        spins = self._spin_boxes[key]
        spins["x"].setValue(primary.x)
        spins["y"].setValue(primary.y)
        spins["Y"].setValue(primary.Y_cdm2)

    # -- Photometer-driven measurement ---------------------------------------

    def _on_detect_photometer(self) -> None:
        from vpsych.core.calibration.argyll import detect_instrument

        result = detect_instrument()
        self.color_photometer_status_label.setText(result.message)

    def _on_start_primary_measurement(self) -> None:
        from vpsych.core.calibration.argyll import (
            ArgyllCalibrationRequiredError,
            ArgyllNotAvailableError,
            ArgyllSpotreadSession,
        )

        session = ArgyllSpotreadSession(instrument="spectrometer")
        self._photometer_session = session
        try:
            session.open()
        except ArgyllCalibrationRequiredError as exc:
            self._handle_color_calibration_prompt(exc)
            return
        except ArgyllNotAvailableError as exc:
            self.color_photometer_status_label.setText(str(exc))
            return
        self._measure_all_primaries()

    def _handle_color_calibration_prompt(self, exc: Exception) -> None:
        from vpsych.core.calibration.argyll import (
            ArgyllCalibrationRequiredError,
            ArgyllNotAvailableError,
            ArgyllReadError,
        )

        prompt_text = getattr(exc, "prompt_text", str(exc))
        QMessageBox.information(
            self,
            "Instrument calibration needed",
            f"{prompt_text}\n\nPosition the instrument as instructed, then click OK.",
        )
        assert self._photometer_session is not None
        try:
            self._photometer_session.confirm_calibration()  # type: ignore[attr-defined]
        except ArgyllCalibrationRequiredError as exc2:
            self._handle_color_calibration_prompt(exc2)
            return
        except (ArgyllNotAvailableError, ArgyllReadError) as exc2:
            self.color_photometer_status_label.setText(f"Calibration failed: {exc2}")
            return
        self._measure_all_primaries()

    def _measure_all_primaries(self) -> None:
        from vpsych.core.calibration.argyll import (
            ArgyllCalibrationRequiredError,
            ArgyllNotAvailableError,
            ArgyllReadError,
            ArgyllSpotreadPhotometer,
            reading_to_primary_chromaticity,
        )

        assert self._photometer_session is not None
        photometer = ArgyllSpotreadPhotometer(self._photometer_session)  # type: ignore[arg-type]
        if self._patch_window is None:
            self._patch_window = PatchWindow()
        self._patch_window.show()

        rgb_by_key = {"red": (1.0, 0.0, 0.0), "green": (0.0, 1.0, 0.0), "blue": (0.0, 0.0, 1.0)}
        try:
            for label, key, _defaults in _PRIMARY_ROWS:
                if key == "white":
                    self._patch_window.set_rgb(1.0, 1.0, 1.0)
                else:
                    self._patch_window.set_rgb(*rgb_by_key[key])
                self.color_photometer_status_label.setText(f"Measuring {label}...")
                reading = photometer.measure_xyz(key, 1.0)
                self._set_primary_fields(key, reading_to_primary_chromaticity(reading))
        except ArgyllCalibrationRequiredError as exc:
            self._handle_color_calibration_prompt(exc)
            return
        except (ArgyllNotAvailableError, ArgyllReadError) as exc:
            self.color_photometer_status_label.setText(f"Measurement failed: {exc}")
            return
        finally:
            self._patch_window.close()
        self.color_photometer_status_label.setText(
            "All four primaries measured. Review the values below, then click Next."
        )

    def commit(self) -> None:
        """Write this step's widget values into the shared wizard state."""
        self._state.color.measured = self.measured_radio.isChecked()
        if self._state.color.measured:
            self._state.color.red = self._current_primary("red")
            self._state.color.green = self._current_primary("green")
            self._state.color.blue = self._current_primary("blue")
            self._state.color.white = self._current_primary("white")


def _xy_spin(value: float) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(0.0, 1.0)
    spin.setDecimals(4)
    spin.setSingleStep(0.001)
    spin.setValue(value)
    return spin


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
