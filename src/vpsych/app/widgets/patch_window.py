"""A borderless solid-color window for a photometer to physically read.

Used by the calibration wizard's guided photometer measurement (gamma and
color-primary steps) to show the patch level currently being measured. This
is a static calibration swatch -- one flat, unchanging RGB color, held on
screen for as long as it takes to place and read the instrument -- not a
frame-timed psychophysical stimulus, so drawing it in the Qt process does
not conflict with the project's rule against rendering *test* stimuli
there (see `vpsych.app.main`'s module docstring): there is no per-frame
timing to get right, only "this pixel value, held steady."

No gamma/dithering correction is applied here: the whole point of a
photometer measurement is to read the display's *actual* raw response to a
given 8-bit drive level, so this window paints the requested level exactly
as given, unmodified.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class PatchWindow(QWidget):
    """A top-level window filled with one solid RGB color, closable with Escape.

    Signals:
        None. The owning wizard step creates one instance, calls
        `set_level`/`set_rgb` before each reading, and closes it (or lets
        the observer close it with Escape) once the measurement sequence
        finishes.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle("Calibration patch -- place the instrument here")
        self.setAccessibleName(
            "Calibration patch. A solid color fills this window for the photometer to "
            "read. Press Escape to close it."
        )
        self._label = QLabel("", self)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setStyleSheet("color: rgba(128, 128, 128, 140); font-size: 11px;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._label)
        self.set_level(0.5)

    def set_level(self, level: float) -> None:
        """Fill the window with a uniform gray at normalized drive `level` in `[0, 1]`."""
        self.set_rgb(level, level, level)

    def set_rgb(self, r: float, g: float, b: float) -> None:
        """Fill the window with the given normalized (r, g, b) drive levels in `[0, 1]`."""
        r8, g8, b8 = (max(0, min(255, round(c * 255))) for c in (r, g, b))
        self.setStyleSheet(f"background-color: rgb({r8}, {g8}, {b8});")
        self._label.setText(f"R={r8} G={g8} B={b8}")

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 (Qt override)
        if event.key() == Qt.Key.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)
