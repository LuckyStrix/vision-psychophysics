"""A checkerboard vs. adjustable-gray-patch widget for the psychophysical gamma task.

Shows a fine, fixed pseudo-random black/white checkerboard (a known
fraction `target_fraction` of cells driven to white) beside a uniform gray
patch the observer adjusts -- with the mouse wheel, or the Up/Down arrow
keys for keyboard-only use -- until the two look equally bright. See
`vpsych.core.calibration.gamma.estimate_gamma_psychophysical` for why this
comparison recovers display gamma with no photometer, and
`vpsych.app.viewmodels.calibration_wizard.BisectionSequenceState` for the
(non-Qt) sequencing/fitting logic this widget is driven by.

This is a static calibration aid the observer studies at their own pace (no
frame-locked timing, no psychophysical trial), in the same spirit as
`vpsych.app.widgets.card_match.CardMatchWidget`'s hand-drawn on-screen
rectangle -- not the frame-timed stimulus rendering the project reserves
for the runner subprocess (see `vpsych.app.main`'s module docstring).
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QKeyEvent, QPainter, QPaintEvent, QWheelEvent
from PySide6.QtWidgets import QWidget

from vpsych.app import theme

_CHECKER_CELLS_PER_SIDE = 24
_MARGIN_PX = 16
_GAP_PX = 24


class BisectionWidget(QWidget):
    """Side-by-side checkerboard (fixed) and gray patch (adjustable) for gamma bisection.

    Signals: none -- the owning step reads/writes `target_fraction` and the
    caller's `BisectionSequenceState.current_level` directly (via
    `gray_level`/`set_gray_level`) rather than round-tripping through Qt
    signals, since this widget has no state of its own beyond the checker
    pattern's random seed.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._target_fraction = 0.5
        self._gray_level = 0.5
        self._checker_mask = self._make_checker_mask(0.5)
        self.setMinimumHeight(220)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._update_accessible_name()

    @staticmethod
    def _make_checker_mask(fraction: float) -> np.ndarray:
        """A fixed (seeded by `fraction` itself, so it's stable across repaints) boolean
        grid with exactly `round(fraction * n_cells)` cells set True (drawn white).
        """
        n_cells = _CHECKER_CELLS_PER_SIDE * _CHECKER_CELLS_PER_SIDE
        rng = np.random.default_rng(round(fraction * 1_000_003))
        mask = np.zeros(n_cells, dtype=bool)
        n_white = round(fraction * n_cells)
        mask[:n_white] = True
        rng.shuffle(mask)
        return mask.reshape(_CHECKER_CELLS_PER_SIDE, _CHECKER_CELLS_PER_SIDE)

    @property
    def target_fraction(self) -> float:
        """The checkerboard's white-cell fraction `p` (matches `BisectionSequenceState`)."""
        return self._target_fraction

    def set_target_fraction(self, fraction: float) -> None:
        """Set the checkerboard's target fraction and regenerate its (fixed) pattern."""
        self._target_fraction = fraction
        self._checker_mask = self._make_checker_mask(fraction)
        self._update_accessible_name()
        self.update()

    @property
    def gray_level(self) -> float:
        """The adjustable patch's current drive level, in `(0, 1)`."""
        return self._gray_level

    def set_gray_level(self, level: float) -> None:
        """Set the adjustable patch's drive level, clamped to `(0.005, 0.995)`."""
        self._gray_level = min(0.995, max(0.005, level))
        self.update()

    def _update_accessible_name(self) -> None:
        self.setAccessibleName(
            "Brightness match. Left: a fixed checkerboard pattern. Right: an adjustable "
            "gray patch -- use the Up and Down arrow keys, or the mouse wheel, to adjust "
            "it until the two look equally bright, then confirm the match."
        )

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 (Qt override)
        step = 0.02 if event.modifiers() & Qt.KeyboardModifier.ShiftModifier else 0.005
        if event.key() == Qt.Key.Key_Up:
            self.set_gray_level(self._gray_level + step)
        elif event.key() == Qt.Key.Key_Down:
            self.set_gray_level(self._gray_level - step)
        else:
            super().keyPressEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802 (Qt override)
        delta = 0.005 if event.angleDelta().y() > 0 else -0.005
        self.set_gray_level(self._gray_level + delta)

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 (Qt override)
        del event
        painter = QPainter(self)
        half_width = (self.width() - 2 * _MARGIN_PX - _GAP_PX) / 2.0
        height = self.height() - 2 * _MARGIN_PX
        if half_width <= 0 or height <= 0:
            painter.end()
            return

        # Left: fixed checkerboard.
        cell_w = half_width / _CHECKER_CELLS_PER_SIDE
        cell_h = height / _CHECKER_CELLS_PER_SIDE
        painter.setPen(Qt.PenStyle.NoPen)
        for row in range(_CHECKER_CELLS_PER_SIDE):
            for col in range(_CHECKER_CELLS_PER_SIDE):
                is_white = bool(self._checker_mask[row, col])
                painter.setBrush(QColor(255, 255, 255) if is_white else QColor(0, 0, 0))
                x = _MARGIN_PX + col * cell_w
                y = _MARGIN_PX + row * cell_h
                painter.drawRect(int(x), int(y), int(cell_w) + 1, int(cell_h) + 1)

        # Right: adjustable uniform gray patch.
        gray = round(self._gray_level * 255)
        painter.setBrush(QColor(gray, gray, gray))
        painter.setPen(QColor(theme.BORDER))
        right_x = _MARGIN_PX + half_width + _GAP_PX
        painter.drawRect(int(right_x), _MARGIN_PX, int(half_width), int(height))
        painter.end()
