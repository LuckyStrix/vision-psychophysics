"""Tests for vpsych.data.catalog: the SQLite index."""

from __future__ import annotations

from pathlib import Path

from vpsych.data import catalog, dataset

from .conftest import build_full_session


def test_rebuild_catalog_indexes_session(tmp_path: Path) -> None:
    pid, sid, cal = build_full_session(tmp_path)
    catalog.rebuild_catalog(tmp_path)

    sessions = catalog.sessions_for_participant(pid, tmp_path)
    assert len(sessions) == 1
    assert sessions[0]["session_id"] == sid
    assert sessions[0]["status"] == "complete"
    assert sessions[0]["calibration_grade"] == cal.luminance_grade
    assert sessions[0]["color_grade"] == cal.color_grade

    history = catalog.task_history(pid, "dummy_test", "OD", tmp_path)
    assert len(history) == 1
    assert history[0]["session_id"] == sid
    assert history[0]["color_grade"] == cal.color_grade


def test_index_session_incremental(tmp_path: Path) -> None:
    pid, sid, _ = build_full_session(tmp_path)
    # No rebuild -- index_session alone should populate the catalog.
    catalog.index_session(pid, sid, tmp_path)
    sessions = catalog.sessions_for_participant(pid, tmp_path)
    assert len(sessions) == 1


def test_index_session_is_idempotent(tmp_path: Path) -> None:
    pid, sid, _ = build_full_session(tmp_path)
    catalog.index_session(pid, sid, tmp_path)
    catalog.index_session(pid, sid, tmp_path)
    sessions = catalog.sessions_for_participant(pid, tmp_path)
    assert len(sessions) == 1  # not duplicated

    history = catalog.task_history(pid, "dummy_test", "OD", tmp_path)
    assert len(history) == 1


def test_rebuild_catalog_equivalence_to_incremental(tmp_path: Path) -> None:
    """Rebuilding from files must produce the same index as incremental indexing."""
    pid, sid, _ = build_full_session(tmp_path)

    catalog.index_session(pid, sid, tmp_path)
    incremental_sessions = catalog.sessions_for_participant(pid, tmp_path)
    incremental_history = catalog.task_history(pid, "dummy_test", "OD", tmp_path)

    # Delete and rebuild from scratch.
    from vpsych.data import paths

    paths.catalog_sqlite_path(tmp_path).unlink()
    catalog.rebuild_catalog(tmp_path)
    rebuilt_sessions = catalog.sessions_for_participant(pid, tmp_path)
    rebuilt_history = catalog.task_history(pid, "dummy_test", "OD", tmp_path)

    assert incremental_sessions == rebuilt_sessions
    assert incremental_history == rebuilt_history


def test_task_history_ordered_oldest_first(tmp_path: Path) -> None:
    dataset.init_dataset(tmp_path)
    participant = dataset.create_participant(tmp_path)

    build_full_session(
        tmp_path, participant_id=participant.participant_id, session_id="ses-20260101T090000"
    )
    build_full_session(
        tmp_path, participant_id=participant.participant_id, session_id="ses-20260201T090000"
    )

    catalog.rebuild_catalog(tmp_path)
    history = catalog.task_history(participant.participant_id, "dummy_test", "OD", tmp_path)
    assert len(history) == 2
    assert history[0]["session_id"] == "ses-20260101T090000"
    assert history[1]["session_id"] == "ses-20260201T090000"


def test_sessions_for_participant_empty(tmp_path: Path) -> None:
    dataset.init_dataset(tmp_path)
    assert catalog.sessions_for_participant("sub-0001", tmp_path) == []


def test_rebuild_catalog_includes_participants(tmp_path: Path) -> None:
    dataset.init_dataset(tmp_path)
    p = dataset.create_participant(tmp_path, year_of_birth=1985, sex="M")
    catalog.rebuild_catalog(tmp_path)

    conn = catalog.connect(tmp_path)
    try:
        row = conn.execute(
            "SELECT * FROM participants WHERE participant_id = ?", (p.participant_id,)
        ).fetchone()
    finally:
        conn.close()
    assert row is not None
    assert row["year_of_birth"] == 1985
    assert row["sex"] == "M"
