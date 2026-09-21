"""Data screen: browse sessions, validate, re-analyze, export, open the HTML report.

`vpsych.reports.session_report.render_session_report` is being built in
parallel and may not exist in this worktree; imported dynamically (see
`vpsych.app.screens.results._import_reports_figures` for why) so its
absence is neither a mypy failure nor an app crash here.
"""

from __future__ import annotations

import importlib
import webbrowser
from typing import Any

from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from vpsych.app.state import AppState
from vpsych.app.viewmodels.data import (
    SessionRow,
    format_validation_report,
    list_all_sessions,
    validation_summary_text,
)
from vpsych.data import paths
from vpsych.data.export import export_dataset, export_participant
from vpsych.data.reanalyze import reanalyze_session
from vpsych.data.validate import validate_dataset, validate_session


def _import_session_report() -> Any | None:
    try:
        return importlib.import_module("vpsych.reports.session_report")
    except Exception:
        return None


class DataScreen(QWidget):
    """Data screen: session browser plus validate/reanalyze/export/report actions."""

    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._rows: list[SessionRow] = []
        self.setAccessibleName("Data screen")

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)

        heading = QLabel("Data")
        heading.setProperty("role", "heading")
        root.addWidget(heading)

        self.session_list = QListWidget()
        self.session_list.setAccessibleName("All recorded sessions")
        root.addWidget(self.session_list, stretch=1)

        button_row = QHBoxLayout()
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh)
        button_row.addWidget(self.refresh_button)

        self.validate_session_button = QPushButton("Validate selected session")
        self.validate_session_button.clicked.connect(self._on_validate_session)
        button_row.addWidget(self.validate_session_button)

        self.validate_dataset_button = QPushButton("Validate whole dataset")
        self.validate_dataset_button.clicked.connect(self._on_validate_dataset)
        button_row.addWidget(self.validate_dataset_button)

        self.reanalyze_button = QPushButton("Re-analyze selected session")
        self.reanalyze_button.clicked.connect(self._on_reanalyze)
        button_row.addWidget(self.reanalyze_button)
        root.addLayout(button_row)

        export_row = QHBoxLayout()
        self.export_participant_button = QPushButton("Export selected participant")
        self.export_participant_button.clicked.connect(self._on_export_participant)
        export_row.addWidget(self.export_participant_button)

        self.export_dataset_button = QPushButton("Export whole dataset")
        self.export_dataset_button.clicked.connect(self._on_export_dataset)
        export_row.addWidget(self.export_dataset_button)

        self.open_report_button = QPushButton("Open HTML report")
        self.open_report_button.clicked.connect(self._on_open_report)
        export_row.addWidget(self.open_report_button)
        root.addLayout(export_row)

        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setAccessibleName("Validation and export output")
        root.addWidget(self.output_text, stretch=1)

        self.refresh()

    def refresh(self) -> None:
        """Reload the full session list from the data layer."""
        self.session_list.clear()
        self._rows = list_all_sessions(self._state.data_root)
        if not self._rows:
            self.session_list.addItem(QListWidgetItem("No sessions recorded yet."))
            return
        for row in self._rows:
            item = QListWidgetItem(f"{row.participant_id}/{row.session_id} ({row.status})")
            item.setData(0x0100, row)
            self.session_list.addItem(item)

    def _selected_row(self) -> SessionRow | None:
        item = self.session_list.currentItem()
        if item is None:
            return None
        data = item.data(0x0100)
        return data if isinstance(data, SessionRow) else None

    def _on_validate_session(self) -> None:
        row = self._selected_row()
        if row is None:
            QMessageBox.information(self, "Validate session", "Select a session first.")
            return
        report = validate_session(row.participant_id, row.session_id, self._state.data_root)
        self._show_validation(report)

    def _on_validate_dataset(self) -> None:
        report = validate_dataset(self._state.data_root)
        self._show_validation(report)

    def _show_validation(self, report: Any) -> None:
        lines = [validation_summary_text(report)]
        for issue in format_validation_report(report):
            location = f" ({issue.path})" if issue.path else ""
            lines.append(f"[{issue.severity}] {issue.code}: {issue.message}{location}")
        self.output_text.setPlainText("\n".join(lines))

    def _on_reanalyze(self) -> None:
        row = self._selected_row()
        if row is None:
            QMessageBox.information(self, "Re-analyze", "Select a session first.")
            return
        session_path = paths.session_dir(row.participant_id, row.session_id, self._state.data_root)
        try:
            results = reanalyze_session(session_path, write=True)
        except Exception as exc:
            self.output_text.setPlainText(f"Re-analysis failed: {exc}")
            return
        lines = [f"Re-analyzed {len(results)} test run(s):"]
        for r in results:
            status = "matches stored summary" if r.matches else "differs from stored summary"
            lines.append(f"- {r.task_id} ({r.eye}, run {r.run}): {status}")
            if r.written_path:
                lines.append(f"  written to {r.written_path}")
        self.output_text.setPlainText("\n".join(lines))

    def _on_export_participant(self) -> None:
        row = self._selected_row()
        if row is None:
            QMessageBox.information(self, "Export participant", "Select a session first.")
            return
        out_path, _ = QFileDialog.getSaveFileName(self, "Export participant", filter="Zip (*.zip)")
        if not out_path:
            return
        export_participant(row.participant_id, out_path, self._state.data_root)
        self.output_text.setPlainText(f"Exported {row.participant_id} to {out_path}")

    def _on_export_dataset(self) -> None:
        out_path, _ = QFileDialog.getSaveFileName(self, "Export dataset", filter="Zip (*.zip)")
        if not out_path:
            return
        export_dataset(out_path, self._state.data_root)
        self.output_text.setPlainText(f"Exported dataset to {out_path}")

    def _on_open_report(self) -> None:
        row = self._selected_row()
        if row is None:
            QMessageBox.information(self, "Open report", "Select a session first.")
            return
        module = _import_session_report()
        if module is None:
            self.output_text.setPlainText(
                "The HTML report is not available in this build: the vpsych.reports."
                "session_report module has not been integrated yet."
            )
            return
        try:
            render = module.render_session_report
            session_path = paths.session_dir(
                row.participant_id, row.session_id, self._state.data_root
            )
            report_path = render(session_path)
        except Exception as exc:
            self.output_text.setPlainText(f"Could not render the HTML report: {exc}")
            return
        webbrowser.open(str(report_path))
