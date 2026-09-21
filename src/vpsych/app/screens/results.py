"""Results screen: per-test threshold/CI, quality flags, figure, and history.

The figure-rendering integration point (`vpsych.reports.figures`) is being
built in parallel and may not exist in this worktree; it is imported
dynamically (via `importlib`, not a top-level `import`) precisely so a
missing `vpsych.reports` submodule is neither a mypy failure here nor an
app crash at runtime -- this screen degrades to a clear placeholder instead.
"""

from __future__ import annotations

import importlib
from typing import Any

from PySide6.QtWidgets import (
    QComboBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from vpsych.app.state import AppState
from vpsych.app.viewmodels.data import list_summaries_for_session
from vpsych.app.viewmodels.results import build_result_view_model
from vpsych.app.widgets.badges import SeverityBadge
from vpsych.data import catalog
from vpsych.data.schemas import TestSummary


def _import_reports_figures() -> Any | None:
    """Dynamically import `vpsych.reports.figures`, or return `None` if unavailable.

    A plain top-level `from vpsych.reports.figures import ...` would make
    this whole screen module fail to import (and fail `mypy src/vpsych`)
    whenever `vpsych.reports.figures` doesn't exist yet, since another
    branch owns that module. `importlib.import_module` is resolved at
    runtime only, so this screen's own import always succeeds.
    """
    try:
        return importlib.import_module("vpsych.reports.figures")
    except Exception:
        return None


class ResultsScreen(QWidget):
    """Results screen: browse a participant's sessions, see per-test results and history."""

    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self.setAccessibleName("Results screen")

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)

        heading = QLabel("Results")
        heading.setProperty("role", "heading")
        root.addWidget(heading)

        splitter = QSplitter()
        root.addWidget(splitter, stretch=1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        session_label = QLabel("Sessions")
        session_label.setProperty("role", "subheading")
        left_layout.addWidget(session_label)
        self.session_list = QListWidget()
        self.session_list.setAccessibleName("Sessions for this participant")
        self.session_list.currentItemChanged.connect(self._on_session_selected)
        left_layout.addWidget(self.session_list)

        test_label = QLabel("Test runs in this session")
        test_label.setProperty("role", "subheading")
        left_layout.addWidget(test_label)
        self.summary_combo = QComboBox()
        self.summary_combo.setAccessibleName("Select a test run to view")
        self.summary_combo.currentIndexChanged.connect(self._on_summary_selected)
        left_layout.addWidget(self.summary_combo)
        splitter.addWidget(left)

        right = QScrollArea()
        right.setWidgetResizable(True)
        self._detail_widget = QWidget()
        self._detail_layout = QVBoxLayout(self._detail_widget)
        right.setWidget(self._detail_widget)
        splitter.addWidget(right)

        self._state.participant_changed.connect(lambda _pid: self.refresh())
        self._summaries: list[TestSummary] = []
        self.refresh()

    def refresh(self) -> None:
        """Reload this participant's sessions from the data layer."""
        self.session_list.clear()
        self.summary_combo.clear()
        self._clear_details()
        pid = self._state.participant_id
        if not pid:
            self._show_placeholder("No participant selected.")
            return
        sessions = catalog.sessions_for_participant(pid, self._state.data_root)
        if not sessions:
            self._show_placeholder("No sessions recorded for this participant yet.")
            return
        for s in sessions:
            item = QListWidgetItem(f"{s['session_id']} ({s['status']})")
            item.setData(0x0100, s["session_id"])
            self.session_list.addItem(item)
        self.session_list.setCurrentRow(0)

    def _on_session_selected(self, current: QListWidgetItem | None, _previous: object) -> None:
        self.summary_combo.clear()
        self._clear_details()
        if current is None or not self._state.participant_id:
            return
        session_id = current.data(0x0100)
        self._summaries = list_summaries_for_session(
            self._state.participant_id, session_id, self._state.data_root
        )
        if not self._summaries:
            self._show_placeholder("No test summaries recorded for this session yet.")
            return
        for s in self._summaries:
            self.summary_combo.addItem(f"{s.task_id} — {s.eye} — run {s.run}")

    def _on_summary_selected(self, index: int) -> None:
        self._clear_details()
        if index < 0 or index >= len(self._summaries):
            return
        summary = self._summaries[index]
        self._render_summary(summary)

    def _clear_details(self) -> None:
        while self._detail_layout.count():
            item = self._detail_layout.takeAt(0)
            widget = item.widget() if item is not None else None
            if widget is not None:
                widget.deleteLater()

    def _show_placeholder(self, text: str) -> None:
        self._clear_details()
        label = QLabel(text)
        label.setProperty("role", "caption")
        self._detail_layout.addWidget(label)

    def _render_summary(self, summary: TestSummary) -> None:
        vm = build_result_view_model(summary)

        title = QLabel(f"{vm.task_id} — {vm.eye} — run {vm.run}")
        title.setProperty("role", "subheading")
        self._detail_layout.addWidget(title)

        threshold_label = QLabel(f"Threshold: {vm.threshold_text}")
        self._detail_layout.addWidget(threshold_label)

        counts_label = QLabel(
            f"{vm.n_trials} main trials, {vm.n_catch} catch trials "
            f"(lapse rate {vm.catch_lapse_rate_text}). Analysis version {vm.analysis_version}."
        )
        counts_label.setProperty("role", "caption")
        self._detail_layout.addWidget(counts_label)

        if vm.flags:
            flags_title = QLabel("Quality flags")
            flags_title.setProperty("role", "subheading")
            self._detail_layout.addWidget(flags_title)
            for flag in vm.flags:
                self._detail_layout.addWidget(SeverityBadge(flag))
        else:
            ok_label = QLabel("No quality concerns flagged for this run.")
            self._detail_layout.addWidget(ok_label)

        self._render_figure(summary)
        self._render_history(summary)

    def _render_figure(self, summary: TestSummary) -> None:
        figures_module = _import_reports_figures()
        figure_title = QLabel("Figure")
        figure_title.setProperty("role", "subheading")
        self._detail_layout.addWidget(figure_title)
        if figures_module is None:
            placeholder = QLabel(
                "Figure rendering is not available in this build: the vpsych.reports.figures "
                "module has not been integrated yet."
            )
            placeholder.setWordWrap(True)
            placeholder.setProperty("role", "caption")
            self._detail_layout.addWidget(placeholder)
            return
        try:
            builder = None
            if hasattr(figures_module, "psychometric_figure"):
                builder = figures_module.psychometric_figure
            elif hasattr(figures_module, "csf_figure"):
                builder = figures_module.csf_figure
            if builder is None:
                raise AttributeError("no figure builder found in vpsych.reports.figures")
            builder(summary)
            placeholder = QLabel("Figure generated (rendering to the canvas is not yet wired up).")
        except Exception as exc:
            placeholder = QLabel(f"Could not render figure: {exc}")
        placeholder.setWordWrap(True)
        placeholder.setProperty("role", "caption")
        self._detail_layout.addWidget(placeholder)

    def _render_history(self, summary: TestSummary) -> None:
        history_title = QLabel(f"History: {summary.task_id} ({summary.eye})")
        history_title.setProperty("role", "subheading")
        self._detail_layout.addWidget(history_title)
        pid = self._state.participant_id
        if not pid:
            return
        history = catalog.task_history(pid, summary.task_id, summary.eye, self._state.data_root)
        if not history:
            self._detail_layout.addWidget(QLabel("No prior runs of this test on record."))
            return
        for row in history:
            text = (
                f"{row['started_utc']}: {row['value']:.3g} {row['units']} "
                f"[{row['ci_low']:.3g}, {row['ci_high']:.3g}]"
                if row.get("value") is not None
                else f"{row['started_utc']}: not available"
            )
            self._detail_layout.addWidget(QLabel(text))
