"""Tests for `vpsych.app.viewmodels.data`."""

from __future__ import annotations

from pathlib import Path

from tests.data.conftest import build_full_session
from vpsych.app.viewmodels.data import (
    format_validation_report,
    list_all_sessions,
    list_summaries_for_session,
    validation_summary_text,
)
from vpsych.data import catalog
from vpsych.data.validate import ValidationIssue, ValidationReport


def test_list_all_sessions_reads_real_data_root(tmp_path: Path) -> None:
    participant_id, session_id, _cal = build_full_session(tmp_path)
    catalog.rebuild_catalog(tmp_path)
    rows = list_all_sessions(tmp_path)
    assert len(rows) == 1
    assert rows[0].participant_id == participant_id
    assert rows[0].session_id == session_id
    assert rows[0].status == "complete"


def test_list_all_sessions_empty_dataset(tmp_path: Path) -> None:
    from vpsych.data import dataset

    dataset.init_dataset(tmp_path)
    assert list_all_sessions(tmp_path) == []


def test_list_summaries_for_session_reads_real_summary(tmp_path: Path) -> None:
    participant_id, session_id, _cal = build_full_session(tmp_path)
    summaries = list_summaries_for_session(participant_id, session_id, tmp_path)
    assert len(summaries) == 1
    assert summaries[0].task_id == "dummy_test"


def test_list_summaries_for_session_missing_beh_dir(tmp_path: Path) -> None:
    assert list_summaries_for_session("sub-0001", "ses-20260101T000000", tmp_path) == []


def test_validation_summary_text_ok() -> None:
    assert validation_summary_text(ValidationReport()) == "Valid, no issues found."


def test_validation_summary_text_counts_severities() -> None:
    report = ValidationReport(
        issues=[
            ValidationIssue(severity="error", code="e1", message="m"),
            ValidationIssue(severity="warning", code="w1", message="m"),
            ValidationIssue(severity="warning", code="w2", message="m"),
            ValidationIssue(severity="info", code="i1", message="m"),
        ]
    )
    text = validation_summary_text(report)
    assert "1 error" in text
    assert "2 warnings" in text
    assert "1 informational note" in text


def test_format_validation_report_sorts_errors_first() -> None:
    report = ValidationReport(
        issues=[
            ValidationIssue(severity="info", code="i1", message="m"),
            ValidationIssue(severity="error", code="e1", message="m"),
        ]
    )
    vms = format_validation_report(report)
    assert vms[0].severity == "error"
    assert vms[1].severity == "info"
