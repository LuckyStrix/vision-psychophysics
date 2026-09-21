"""A draggable on-screen rectangle for the credit-card screen-width match.

The participant/experimenter holds a real ID-1 card (a standard credit or
debit card) against the screen and drags this widget's edge -- with the
mouse, or the Left/Right arrow keys for keyboard-only use -- until the
rectangle visually matches the card's width. See
`vpsych.core.calibration.geometry` for the math that turns the matched
pixel width into a physical screen-width estimate.
"""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QKeyEvent, QMouseEvent, QPainter, QPaintEvent
from PySide6.QtWidgets import QWidget

from vpsych.app import theme
from vpsych.core.calibration.geometry import ID1_CARD_HEIGHT_MM, ID1_CARD_WIDTH_MM

_MARGIN_PX = 16
_MIN_WIDTH_PX = 40.0


class CardMatchWidget(QWidget):
    """Draggable rectangle used to match a real ID-1 card's width on screen.

    Signals:
        width_changed: Emitted with the new width in pixels whenever the
            rectangle is resized (drag or keyboard).
    """

    width_changed = Signal(float)

    def __init__(self, initial_width_px: float = 300.0, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._card_width_px = initial_width_px
        self.setMinimumHeight(160)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setAccessibleName(
            "Card match rectangle. Drag with the mouse, or use the left and right arrow "
            "keys, to match its width to a real credit card held against the screen."
        )

    @property
    def card_width_px(self) -> float:
        """The rectangle's current width, in pixels."""
        return self._card_width_px

    def set_card_width_px(self, value: float) -> None:
        """Set the rectangle's width, clamped to the widget's usable area, emitting a signal."""
        max_width = max(_MIN_WIDTH_PX, self.width() - 2 * _MARGIN_PX)
        clamped = max(_MIN_WIDTH_PX, min(float(value), max_width))
        if clamped != self._card_width_px:
            self._card_width_px = clamped
            self.update()
            self.width_changed.emit(self._card_width_px)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 (Qt override)
        self.setFocus()
        self.set_card_width_px(event.position().x() - _MARGIN_PX)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 (Qt override)
        if event.buttons() & Qt.MouseButton.LeftButton:
            self.set_card_width_px(event.position().x() - _MARGIN_PX)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 (Qt override)
        step = 10.0 if event.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1.0
        if event.key() == Qt.Key.Key_Left:
            self.set_card_width_px(self._card_width_px - step)
        elif event.key() == Qt.Key.Key_Right:
            self.set_card_width_px(self._card_width_px + step)
        else:
            super().keyPressEvent(event)

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 (Qt override)
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        card_height_px = self._card_width_px * (ID1_CARD_HEIGHT_MM / ID1_CARD_WIDTH_MM)
        rect = QRectF(
            _MARGIN_PX,
            (self.height() - card_height_px) / 2.0,
            self._card_width_px,
            card_height_px,
        )
        painter.setBrush(QColor(theme.ACCENT))
        painter.setPen(QColor(theme.BORDER))
        painter.drawRoundedRect(rect, 8, 8)
        painter.end()
