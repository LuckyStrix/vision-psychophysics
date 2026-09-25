"""Data-screen logic: cross-participant session listing and validation-report formatting.

Session browsing goes through `vpsych.data.catalog`/`vpsych.data.dataset`
(the data layer) exclusively -- this module never reads or writes dataset
files itself, it only composes and formats what those modules return.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from vpsych.data import catalog, dataset, paths
from vpsych.data.schemas import TestSummary
from vpsych.data.validate import ValidationIssue, ValidationReport

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SessionRow:
    """One row in the Data screen's session browser.

    Attributes:
        participant_id: The session's participant.
        session_id: The session's ID.
        status: Raw session status string.
        started_utc: Raw start timestamp string (ISO 8601), or `None`.
        calibration_grade: Luminance grade of the calibration used, or `None`.
        color_grade: Color grade of the calibration used, or `None`.
    """

    participant_id: str
    session_id: str
    status: str
    started_utc: str | None
    calibration_grade: str | None
    color_grade: str | None


def list_all_sessions(root: Path | None = None) -> list[SessionRow]:
    """List every session across every participant, newest first per participant.

    Args:
        root: Data root, or `None` to use `vpsych.data.paths.data_root()`.

    Returns:
        `SessionRow`s for every participant's sessions, participants in
        `participant_id` order (as returned by
        `vpsych.data.dataset.list_participants`).
    """
    rows: list[SessionRow] = []
    for participant in dataset.list_participants(root):
        for s in catalog.sessions_for_participant(participant.participant_id, root):
            rows.append(
                SessionRow(
                    participant_id=participant.participant_id,
                    session_id=str(s.get("session_id", "")),
                    status=str(s.get("status", "")),
                    started_utc=(
                        str(s["started_utc"]) if s.get("started_utc") is not None else None
                    ),
                    calibration_grade=(
                        str(s["calibration_grade"])
                        if s.get("calibration_grade") is not None
                        else None
                    ),
                    color_grade=(
                        str(s["color_grade"]) if s.get("color_grade") is not None else None
                    ),
                )
            )
    return rows


def list_summaries_for_session(
    participant_id: str, session_id: str, root: Path | None = None
) -> list[TestSummary]:
    """List every test summary recorded for one session.

    Args:
        participant_id: The session's participant, `sub-XXXX`.
        session_id: The session ID, `ses-YYYYMMDDTHHMMSS`.
        root: Data root, or `None` to use `vpsych.data.paths.data_root()`.

    Returns:
        Parsed `TestSummary`s found under the session's `beh/` directory,
        in filename order. A summary file that fails to parse is skipped
        (use `vpsych.data.validate` to surface that as an error).
    """
    beh_dir = paths.beh_dir(participant_id, session_id, root)
    if not beh_dir.exists():
        return []
    summaries: list[TestSummary] = []
    for summary_path in sorted(beh_dir.glob("*_summary.json")):
        try:
            summaries.append(
                TestSummary.model_validate_json(summary_path.read_text(encoding="utf-8"))
            )
        except Exception:
            log.warning("Skipping unreadable summary %s", summary_path, exc_info=True)
            continue
    return summaries


_SEVERITY_ORDER = {"error": 0, "warning": 1, "info": 2}


@dataclass(frozen=True)
class ValidationIssueViewModel:
    """Display-ready form of one `ValidationIssue`."""

    severity: str
    code: str
    message: str
    path: str | None


def format_validation_report(report: ValidationReport) -> list[ValidationIssueViewModel]:
    """Format a `ValidationReport`'s issues for display, most severe first.

    Args:
        report: The validation report to format.

    Returns:
        View models sorted error, then warning, then info (stable within
        each severity).
    """
    indexed: list[tuple[int, ValidationIssue]] = list(enumerate(report.issues))
    indexed.sort(key=lambda pair: (_SEVERITY_ORDER[pair[1].severity], pair[0]))
    return [
        ValidationIssueViewModel(severity=i.severity, code=i.code, message=i.message, path=i.path)
        for _, i in indexed
    ]


def validation_summary_text(report: ValidationReport) -> str:
    """One-line summary of a validation report, e.g. `"2 errors, 1 warning"`.

    Args:
        report: The validation report to summarize.

    Returns:
        `"Valid, no issues found."` if `report.ok` and there are no
        warnings either; otherwise a count of errors/warnings/info issues.
    """
    n_errors = len(report.errors)
    n_warnings = len(report.warnings)
    n_info = len([i for i in report.issues if i.severity == "info"])
    if n_errors == 0 and n_warnings == 0 and n_info == 0:
        return "Valid, no issues found."
    parts = []
    if n_errors:
        parts.append(f"{n_errors} error{'s' if n_errors != 1 else ''}")
    if n_warnings:
        parts.append(f"{n_warnings} warning{'s' if n_warnings != 1 else ''}")
    if n_info:
        parts.append(f"{n_info} informational note{'s' if n_info != 1 else ''}")
    return ", ".join(parts)
