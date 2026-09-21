"""Severity/grade badge widgets: always pair color with a text marker and label."""

from __future__ import annotations

from PySide6.QtWidgets import QHBoxLayout, QLabel, QWidget

from vpsych.app import theme
from vpsych.app.viewmodels.results import QualityFlagViewModel


class SeverityBadge(QWidget):
    """A small severity indicator: a glyph, a word, and a message -- never color alone."""

    def __init__(self, flag: QualityFlagViewModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        marker = QLabel(f"[{flag.marker}]")
        marker.setStyleSheet(f"color: {theme.severity_color(flag.severity)}; font-weight: 700;")
        marker.setAccessibleName(f"{flag.label} indicator")
        layout.addWidget(marker)

        text = QLabel(f"{flag.label}: {flag.message}")
        text.setWordWrap(True)
        text.setAccessibleName(f"{flag.label}: {flag.message}")
        layout.addWidget(text, stretch=1)


class CalibrationGradePill(QLabel):
    """A plain text pill showing a calibration grade, e.g. `"Luminance: A"`."""

    def __init__(self, label: str, value_text: str, parent: QWidget | None = None) -> None:
        super().__init__(f"{label}: {value_text}", parent)
        self.setProperty("role", "caption")
        self.setAccessibleName(f"{label}: {value_text}")
