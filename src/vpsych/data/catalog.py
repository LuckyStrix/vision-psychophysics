"""SQLite index of participants, sessions, and test summaries, for the UI.

`catalog.sqlite` is a pure index: files under the data root are always the
source of truth, and `rebuild_catalog` regenerates the whole database from
them at any time (e.g. after manual edits, or if the database is deleted or
corrupted). `index_session` does the same for one session incrementally, so
the runner/app can keep the catalog current without a full rebuild after
every session. Uses only the standard library `sqlite3` module.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from vpsych.data import dataset, paths
from vpsych.data.schemas import SessionInfo, TestSummary

_SCHEMA = """
CREATE TABLE IF NOT EXISTS participants (
    participant_id TEXT PRIMARY KEY,
    year_of_birth INTEGER,
    sex TEXT,
    refractive_correction TEXT
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT NOT NULL,
    participant_id TEXT NOT NULL,
    status TEXT NOT NULL,
    started_utc TEXT NOT NULL,
    ended_utc TEXT,
    calibration_hash TEXT NOT NULL,
    calibration_grade TEXT,
    PRIMARY KEY (participant_id, session_id)
);

CREATE TABLE IF NOT EXISTS test_summaries (
    participant_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    eye TEXT NOT NULL,
    run INTEGER NOT NULL,
    value REAL,
    ci_low REAL,
    ci_high REAL,
    ci_level REAL,
    units TEXT,
    n_trials INTEGER,
    n_catch INTEGER,
    catch_lapse_rate REAL,
    flags TEXT NOT NULL,
    calibration_grade TEXT,
    analysis_version TEXT,
    started_utc TEXT NOT NULL,
    PRIMARY KEY (participant_id, session_id, task_id, eye, run)
);

CREATE INDEX IF NOT EXISTS idx_test_summaries_history
    ON test_summaries (participant_id, task_id, eye, started_utc);

CREATE INDEX IF NOT EXISTS idx_sessions_participant
    ON sessions (participant_id, started_utc);
"""


def connect(root: Path | None = None) -> sqlite3.Connection:
    """Open (creating if needed) `catalog.sqlite` at a data root, with the schema ensured.

    Args:
        root: Data root, or `None` to use `vpsych.data.paths.data_root()`.

    Returns:
        A `sqlite3.Connection` with `row_factory` set to `sqlite3.Row` and
        the schema created if it did not already exist.
    """
    root = root or paths.data_root()
    paths.ensure_dir_exists(root)
    conn = sqlite3.connect(str(paths.catalog_sqlite_path(root)))
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def _calibration_grade_for(root: Path, calibration_hash: str) -> str | None:
    try:
        calibration = dataset.load_calibration(calibration_hash, root)
    except (FileNotFoundError, ValueError):
        return None
    return calibration.luminance_grade


def index_session(
    participant_id: str,
    session_id: str,
    root: Path | None = None,
    conn: sqlite3.Connection | None = None,
) -> None:
    """Index (or re-index) one session's `session.json` and summaries into the catalog.

    Idempotent: re-running replaces this session's existing rows.

    Args:
        participant_id: The session's participant, `sub-XXXX`.
        session_id: The session ID, `ses-YYYYMMDDTHHMMSS`.
        root: Data root, or `None` to use `vpsych.data.paths.data_root()`.
        conn: An open connection to reuse (e.g. from `rebuild_catalog`), or
            `None` to open (and close) one just for this call.
    """
    root = root or paths.data_root()
    owns_conn = conn is None
    conn = conn or connect(root)
    try:
        session_json_path = paths.session_json_path(participant_id, session_id, root)
        if not session_json_path.exists():
            return
        session_info = SessionInfo.model_validate_json(
            session_json_path.read_text(encoding="utf-8")
        )
        grade = _calibration_grade_for(root, session_info.calibration_hash)

        conn.execute(
            "DELETE FROM sessions WHERE participant_id = ? AND session_id = ?",
            (participant_id, session_id),
        )
        conn.execute(
            "DELETE FROM test_summaries WHERE participant_id = ? AND session_id = ?",
            (participant_id, session_id),
        )
        conn.execute(
            """
            INSERT INTO sessions
                (session_id, participant_id, status, started_utc, ended_utc,
                 calibration_hash, calibration_grade)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                participant_id,
                session_info.status,
                session_info.started_utc.isoformat(),
                session_info.ended_utc.isoformat() if session_info.ended_utc else None,
                session_info.calibration_hash,
                grade,
            ),
        )

        beh_dir = paths.beh_dir(participant_id, session_id, root)
        if beh_dir.exists():
            for summary_path in sorted(beh_dir.glob("*_summary.json")):
                try:
                    summary = TestSummary.model_validate_json(
                        summary_path.read_text(encoding="utf-8")
                    )
                except Exception:
                    continue
                conn.execute(
                    """
                    INSERT OR REPLACE INTO test_summaries
                        (participant_id, session_id, task_id, eye, run, value, ci_low, ci_high,
                         ci_level, units, n_trials, n_catch, catch_lapse_rate, flags,
                         calibration_grade, analysis_version, started_utc)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        participant_id,
                        session_id,
                        summary.task_id,
                        summary.eye,
                        summary.run,
                        summary.estimate.value,
                        summary.estimate.ci_low,
                        summary.estimate.ci_high,
                        summary.estimate.ci_level,
                        summary.estimate.units,
                        summary.n_trials,
                        summary.n_catch,
                        summary.catch_lapse_rate,
                        json.dumps([f.code for f in summary.quality_flags]),
                        grade,
                        summary.analysis_version,
                        session_info.started_utc.isoformat(),
                    ),
                )

        conn.execute(
            """
            INSERT OR REPLACE INTO participants (participant_id, year_of_birth, sex, refractive_correction)
            VALUES (?, ?, ?, ?)
            """,
            _participant_row(participant_id, root),
        )
        conn.commit()
    finally:
        if owns_conn:
            conn.close()


def _participant_row(
    participant_id: str, root: Path
) -> tuple[str, int | None, str | None, str | None]:
    participant = dataset.get_participant(participant_id, root)
    if participant is None:
        return (participant_id, None, None, None)
    return (
        participant_id,
        participant.year_of_birth,
        participant.sex,
        participant.refractive_correction,
    )


def rebuild_catalog(root: Path | None = None) -> None:
    """Rebuild `catalog.sqlite` from scratch by scanning the data root.

    Drops and recreates every table, then indexes every participant and
    every `sub-*/ses-*` session directory found on disk. Files are always
    the source of truth -- this is safe to run at any time (e.g. after
    manual edits to the data root, or to repair a corrupted catalog).

    Args:
        root: Data root, or `None` to use `vpsych.data.paths.data_root()`.
    """
    root = root or paths.data_root()
    conn = connect(root)
    try:
        conn.executescript(
            "DELETE FROM participants; DELETE FROM sessions; DELETE FROM test_summaries;"
        )
        conn.commit()
        for participant in dataset.list_participants(root):
            conn.execute(
                """
                INSERT OR REPLACE INTO participants
                    (participant_id, year_of_birth, sex, refractive_correction)
                VALUES (?, ?, ?, ?)
                """,
                (
                    participant.participant_id,
                    participant.year_of_birth,
                    participant.sex,
                    participant.refractive_correction,
                ),
            )
        conn.commit()

        for participant_dir in sorted(p for p in root.glob("sub-*") if p.is_dir()):
            for session_dir in sorted(s for s in participant_dir.glob("ses-*") if s.is_dir()):
                index_session(participant_dir.name, session_dir.name, root, conn=conn)
    finally:
        conn.close()


def sessions_for_participant(
    participant_id: str, root: Path | None = None
) -> list[dict[str, object]]:
    """List every indexed session for a participant, newest first.

    Args:
        participant_id: The participant to query.
        root: Data root, or `None` to use `vpsych.data.paths.data_root()`.

    Returns:
        A list of dicts (one per session), each with keys `session_id`,
        `status`, `started_utc`, `ended_utc`, `calibration_hash`,
        `calibration_grade`.
    """
    root = root or paths.data_root()
    conn = connect(root)
    try:
        rows = conn.execute(
            """
            SELECT session_id, status, started_utc, ended_utc, calibration_hash, calibration_grade
            FROM sessions
            WHERE participant_id = ?
            ORDER BY started_utc DESC
            """,
            (participant_id,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def task_history(
    participant_id: str, task_id: str, eye: str, root: Path | None = None
) -> list[dict[str, object]]:
    """List a participant's results for one task/eye over time, oldest first.

    Used to draw the longitudinal comparison plot on the results screen.

    Args:
        participant_id: The participant to query.
        task_id: The `TestSpec.id` to filter on.
        eye: Eye condition to filter on (`"OD"`, `"OS"`, or `"OU"`).
        root: Data root, or `None` to use `vpsych.data.paths.data_root()`.

    Returns:
        A list of dicts (one per test run), ordered by `started_utc`
        ascending, each with keys `session_id`, `run`, `value`, `ci_low`,
        `ci_high`, `ci_level`, `units`, `flags` (a JSON-encoded list of
        quality-flag codes), `calibration_grade`, `analysis_version`,
        `started_utc`.
    """
    root = root or paths.data_root()
    conn = connect(root)
    try:
        rows = conn.execute(
            """
            SELECT session_id, run, value, ci_low, ci_high, ci_level, units, flags,
                   calibration_grade, analysis_version, started_utc
            FROM test_summaries
            WHERE participant_id = ? AND task_id = ? AND eye = ?
            ORDER BY started_utc ASC
            """,
            (participant_id, task_id, eye),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()
