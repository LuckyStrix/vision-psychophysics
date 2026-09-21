"""Home-screen display logic: recent-sessions list formatting.

Consumes `vpsych.data.catalog.sessions_for_participant`'s plain dicts
(`session_id`, `status`, `started_utc`, `ended_utc`, `calibration_hash`,
`calibration_grade`, `color_grade`) and turns them into display-ready rows.
Never fabricates a value: a missing field renders as `"Not available"`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

_STATUS_LABELS = {
    "running": "Running / interrupted",
    "complete": "Complete",
    "incomplete": "Incomplete",
    "aborted": "Aborted",
}


def status_label(status: str) -> str:
    """Human-readable label for a session status string."""
    return _STATUS_LABELS.get(status, status)


def _format_timestamp(value: object) -> str:
    if not value:
        return "Not available"
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError:
            return str(value)
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M UTC")
    return str(value)


@dataclass(frozen=True)
class RecentSessionRow:
    """One formatted row in the Home screen's recent-sessions list.

    Attributes:
        session_id: The session's ID.
        status_text: Human-readable status.
        started_text: Formatted start timestamp.
        calibration_grade_text: Luminance grade of the calibration used, or
            `"Not available"`.
    """

    session_id: str
    status_text: str
    started_text: str
    calibration_grade_text: str


def build_recent_sessions(
    sessions: list[dict[str, object]], limit: int = 5
) -> list[RecentSessionRow]:
    """Build display rows for a participant's most recent sessions.

    Args:
        sessions: Session dicts from
            `vpsych.data.catalog.sessions_for_participant` (already ordered
            newest-first).
        limit: Maximum number of rows to return.

    Returns:
        Up to `limit` formatted rows, in the input order.
    """
    rows: list[RecentSessionRow] = []
    for s in sessions[:limit]:
        grade = s.get("calibration_grade")
        rows.append(
            RecentSessionRow(
                session_id=str(s.get("session_id", "Not available")),
                status_text=status_label(str(s.get("status", ""))),
                started_text=_format_timestamp(s.get("started_utc")),
                calibration_grade_text=str(grade) if grade else "Not available",
            )
        )
    return rows


def format_participant_label(
    participant_id: str, year_of_birth: int | None, sex: str | None
) -> str:
    """Format a short participant label for a picker list, with no identifying detail.

    Args:
        participant_id: The `sub-XXXX` ID.
        year_of_birth: Birth year, or `None`.
        sex: Self-reported sex, or `None`.

    Returns:
        e.g. `"sub-0003 (b. 1990, F)"`, or just `participant_id` if neither
        optional field is recorded.
    """
    details = []
    if year_of_birth is not None:
        details.append(f"b. {year_of_birth}")
    if sex:
        details.append(sex)
    if not details:
        return participant_id
    return f"{participant_id} ({', '.join(details)})"
