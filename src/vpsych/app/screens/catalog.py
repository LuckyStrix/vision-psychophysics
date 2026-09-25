"""Test catalog screen: cards per visible test, grouped by domain."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from vpsych.app.state import AppState
from vpsych.app.viewmodels.catalog import TestCardViewModel, build_test_cards, group_by_domain
from vpsych.app.viewmodels.session_plan import DEFAULT_VIEWING_DISTANCE_CM
from vpsych.core.display import DisplayGeometry
from vpsych.tests_catalog.base import discover_tests


class TestCard(QFrame):
    """One test's catalog card: what it measures, duration, eyes, and requirement status.

    Signals:
        selected: Emitted with the card's `task_id` when "Add to session" is
            activated (only connected/enabled when the card is enabled).
    """

    selected = Signal(str)

    def __init__(self, card: TestCardViewModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.card = card
        self.setProperty("role", "card" if card.enabled else "card-disabled")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        name = QLabel(card.name)
        name.setProperty("role", "subheading")
        layout.addWidget(name)

        measures = QLabel(f"Measures: {card.measures}")
        measures.setWordWrap(True)
        layout.addWidget(measures)

        description = QLabel(card.description_participant)
        description.setWordWrap(True)
        layout.addWidget(description)

        meta = QLabel(
            f"Duration {card.duration_text} · Eyes: {', '.join(card.allowed_eyes)} · "
            f"Units: {card.output_units}"
        )
        meta.setProperty("role", "caption")
        layout.addWidget(meta)

        if card.enabled:
            add_button = QPushButton("Add to session")
            add_button.setAccessibleName(f"Add {card.name} to session")
            add_button.clicked.connect(lambda: self.selected.emit(card.task_id))
            layout.addWidget(add_button)
        else:
            reasons_label = QLabel("Cannot run: " + " ".join(card.unmet_reasons))
            reasons_label.setWordWrap(True)
            reasons_label.setStyleSheet("font-weight: 600;")
            reasons_label.setAccessibleName(
                f"{card.name} is disabled. " + " ".join(card.unmet_reasons)
            )
            layout.addWidget(reasons_label)

        self.setAccessibleName(
            f"{card.name}, {'available' if card.enabled else 'unavailable'} test card"
        )


class TestCatalogScreen(QWidget):
    """Test catalog screen: browse every visible test, grouped by domain.

    Signals:
        test_added: Emitted with a `task_id` when a card's "Add to session"
            is activated (the session builder screen listens for this).
    """

    test_added = Signal(str)

    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self.setAccessibleName("Test catalog screen")
        discover_tests()

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)

        heading = QLabel("Test catalog")
        heading.setProperty("role", "heading")
        root.addWidget(heading)

        subtitle = QLabel(
            "Tests you can currently run, grouped by what they measure. A disabled card "
            "states exactly why it cannot run yet."
        )
        subtitle.setWordWrap(True)
        subtitle.setProperty("role", "caption")
        root.addWidget(subtitle)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setSpacing(16)
        self.scroll_area.setWidget(self._content)
        root.addWidget(self.scroll_area, stretch=1)

        self._state.calibration_changed.connect(self.refresh)
        self.refresh()

    def _current_display(self) -> DisplayGeometry:
        calibration = self._state.latest_calibration
        if calibration is not None:
            return calibration.geometry.model_copy(
                update={"viewing_distance_cm": DEFAULT_VIEWING_DISTANCE_CM}
            )
        # No calibration yet: use a conservative placeholder geometry purely so
        # `check_requirements` has something to evaluate against -- never
        # invented as if it were a real measurement, and every gamma/color
        # requirement still correctly fails with calibration=None.
        return DisplayGeometry(
            width_px=1920,
            height_px=1080,
            width_cm=53.0,
            height_cm=30.0,
            viewing_distance_cm=DEFAULT_VIEWING_DISTANCE_CM,
            refresh_hz=60.0,
        )

    def refresh(self) -> None:
        """Rebuild every card from the current calibration/display state."""
        while self._content_layout.count():
            item = self._content_layout.takeAt(0)
            widget = item.widget() if item is not None else None
            if widget is not None:
                widget.deleteLater()

        display = self._current_display()
        cards = build_test_cards(display, self._state.latest_calibration)
        for domain_label, domain_cards in group_by_domain(cards):
            group_label = QLabel(domain_label)
            group_label.setProperty("role", "subheading")
            self._content_layout.addWidget(group_label)
            for card_vm in domain_cards:
                card = TestCard(card_vm)
                card.selected.connect(self.test_added)
                self._content_layout.addWidget(card)
        self._content_layout.addStretch(1)
