"""Tests for vpsych.data.reanalyze."""

from __future__ import annotations

import json
from pathlib import Path

from vpsych.data import paths, reanalyze

from .conftest import build_full_session


def test_reanalyze_reproduces_summary_exactly(
    tmp_path: Path, registered_dummy_test: object
) -> None:
    pid, sid, _ = build_full_session(tmp_path)
    session_dir = paths.session_dir(pid, sid, tmp_path)

    results = reanalyze.reanalyze_session(session_dir, write=False)
    assert len(results) == 1
    result = results[0]
    assert result.task_id == "dummy_test"
    assert result.old_summary is not None
    assert result.matches
    assert result.differing_fields == []
    assert result.new_summary == result.old_summary


def test_reanalyze_write_creates_new_file_without_touching_original(
    tmp_path: Path, registered_dummy_test: object
) -> None:
    pid, sid, _ = build_full_session(tmp_path)
    session_dir = paths.session_dir(pid, sid, tmp_path)
    original_summary_path = paths.summary_json_path(pid, sid, "dummy_test", "OD", 1, tmp_path)
    original_mtime = original_summary_path.stat().st_mtime
    original_bytes = original_summary_path.read_bytes()

    results = reanalyze.reanalyze_session(session_dir, write=True)
    result = results[0]

    assert result.written_path is not None
    assert result.written_path.exists()
    assert result.written_path != original_summary_path
    assert "reanalysis" in result.written_path.name

    # Original file is untouched.
    assert original_summary_path.read_bytes() == original_bytes
    assert original_summary_path.stat().st_mtime == original_mtime

    from vpsych.data.schemas import TestSummary

    written = TestSummary.model_validate_json(result.written_path.read_text(encoding="utf-8"))
    assert written == result.new_summary


def test_reanalyze_no_write_creates_no_file(tmp_path: Path, registered_dummy_test: object) -> None:
    pid, sid, _ = build_full_session(tmp_path)
    session_dir = paths.session_dir(pid, sid, tmp_path)
    beh_dir = session_dir / "beh"
    before = set(beh_dir.iterdir())

    reanalyze.reanalyze_session(session_dir, write=False)

    after = set(beh_dir.iterdir())
    assert before == after


def test_reanalyze_detects_mismatch_when_summary_was_stale(
    tmp_path: Path, registered_dummy_test: object
) -> None:
    pid, sid, _ = build_full_session(tmp_path)
    summary_path = paths.summary_json_path(pid, sid, "dummy_test", "OD", 1, tmp_path)
    summary_path.chmod(0o644)
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    data["estimate"]["value"] = 999.0
    summary_path.write_text(json.dumps(data), encoding="utf-8")

    session_dir = paths.session_dir(pid, sid, tmp_path)
    results = reanalyze.reanalyze_session(session_dir, write=False)
    result = results[0]
    assert not result.matches
    assert "estimate" in result.differing_fields


def test_reanalyze_no_stored_summary(tmp_path: Path, registered_dummy_test: object) -> None:
    pid, sid, _ = build_full_session(tmp_path, write_summary=False)
    session_dir = paths.session_dir(pid, sid, tmp_path)
    results = reanalyze.reanalyze_session(session_dir, write=False)
    result = results[0]
    assert result.old_summary is None
    assert not result.matches
    assert result.differing_fields == []
