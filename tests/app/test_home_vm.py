"""Tests for `vpsych.app.viewmodels.home`."""

from __future__ import annotations

from vpsych.app.viewmodels.home import (
    build_recent_sessions,
    format_participant_label,
    status_label,
)


def test_status_label_known_and_unknown() -> None:
    assert status_label("complete") == "Complete"
    assert status_label("weird_status") == "weird_status"


def test_build_recent_sessions_formats_and_limits() -> None:
    sessions = [
        {
            "session_id": f"ses-2026010{i}T000000",
            "status": "complete",
            "started_utc": f"2026-01-0{i}T00:00:00+00:00",
            "calibration_grade": "A",
        }
        for i in range(1, 8)
    ]
    rows = build_recent_sessions(sessions, limit=3)
    assert len(rows) == 3
    assert rows[0].session_id == "ses-20260101T000000"
    assert rows[0].status_text == "Complete"
    assert rows[0].calibration_grade_text == "A"
    assert "2026-01-01" in rows[0].started_text


def test_build_recent_sessions_missing_fields_show_not_available() -> None:
    rows = build_recent_sessions([{"session_id": "ses-1"}])
    assert rows[0].status_text == ""
    assert rows[0].started_text == "Not available"
    assert rows[0].calibration_grade_text == "Not available"


def test_format_participant_label_variants() -> None:
    assert format_participant_label("sub-0001", None, None) == "sub-0001"
    assert format_participant_label("sub-0002", 1990, None) == "sub-0002 (b. 1990)"
    assert format_participant_label("sub-0003", 1990, "F") == "sub-0003 (b. 1990, F)"
    assert format_participant_label("sub-0004", None, "M") == "sub-0004 (M)"
