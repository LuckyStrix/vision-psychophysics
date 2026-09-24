"""BIDS-like path layout for the vpsych data root.

The data root lives outside this repository (default `~/vpsych-data`,
overridable with the `VPSYCH_DATA_ROOT` environment variable) and holds
every participant's raw and summarized data:

```
<data_root>/
  dataset_description.json
  participants.tsv
  participants.json
  calibration/cal-<hash>.json
  sub-0001/ses-20260916T103000/
    session.json
    beh/sub-0001_ses-20260916T103000_task-acuity_eye-OD_run-1_trials.tsv
    beh/sub-0001_ses-20260916T103000_task-acuity_eye-OD_run-1_trials.json
    beh/sub-0001_ses-20260916T103000_task-acuity_eye-OD_run-1_summary.json
    beh/sub-0001_ses-20260916T103000_task-acuity_eye-OD_run-1_frames.tsv
    MANIFEST.sha256
  catalog.sqlite
```

This module only builds and validates paths -- it does not read or write
file contents (see `vpsych.data.writer` for that) and does not touch the
filesystem except in `ensure_dir_exists`, so it is trivially headless
testable.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

_PARTICIPANT_ID_RE = re.compile(r"^sub-[0-9]{4}$")
_SESSION_ID_RE = re.compile(r"^ses-[0-9]{8}T[0-9]{6}$")
_TASK_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_EYE_RE = re.compile(r"^(OD|OS|OU)$")

_ENV_VAR = "VPSYCH_DATA_ROOT"
_DEFAULT_DATA_ROOT = "~/vpsych-data"


def _validate(value: str, pattern: re.Pattern[str], kind: str) -> str:
    if not pattern.fullmatch(value):
        raise ValueError(f"Invalid {kind}: {value!r} does not match {pattern.pattern!r}")
    return value


def data_root() -> Path:
    """Return the vpsych data root directory.

    Uses the `VPSYCH_DATA_ROOT` environment variable if set (expanding `~`
    and environment variables in it), otherwise defaults to
    `~/vpsych-data`. Does not create the directory.

    Returns:
        The absolute path to the data root.
    """
    raw = os.environ.get(_ENV_VAR, _DEFAULT_DATA_ROOT)
    return Path(os.path.expanduser(os.path.expandvars(raw))).resolve()


def dataset_description_path(root: Path | None = None) -> Path:
    """Path to `dataset_description.json` at the root of the data tree.

    Args:
        root: Data root to use, or `None` to use :func:`data_root`.

    Returns:
        The path to `dataset_description.json`.
    """
    return (root or data_root()) / "dataset_description.json"


def participants_tsv_path(root: Path | None = None) -> Path:
    """Path to the dataset-wide `participants.tsv` index.

    Args:
        root: Data root to use, or `None` to use :func:`data_root`.

    Returns:
        The path to `participants.tsv`.
    """
    return (root or data_root()) / "participants.tsv"


def participants_json_path(root: Path | None = None) -> Path:
    """Path to the `participants.tsv` sidecar, `participants.json`.

    Args:
        root: Data root to use, or `None` to use :func:`data_root`.

    Returns:
        The path to `participants.json`.
    """
    return (root or data_root()) / "participants.json"


def catalog_sqlite_path(root: Path | None = None) -> Path:
    """Path to the rebuildable UI index database, `catalog.sqlite`.

    Args:
        root: Data root to use, or `None` to use :func:`data_root`.

    Returns:
        The path to `catalog.sqlite`.
    """
    return (root or data_root()) / "catalog.sqlite"


def calibration_dir(root: Path | None = None) -> Path:
    """Directory holding immutable calibration files, `calibration/`.

    Args:
        root: Data root to use, or `None` to use :func:`data_root`.

    Returns:
        The path to the `calibration/` directory.
    """
    return (root or data_root()) / "calibration"


def calibration_path(content_hash: str, root: Path | None = None) -> Path:
    """Path to one immutable calibration file, `calibration/cal-<hash>.json`.

    Args:
        content_hash: The calibration's `Calibration.content_hash()`.
        root: Data root to use, or `None` to use :func:`data_root`.

    Returns:
        The path to `calibration/cal-<content_hash>.json`.
    """
    return calibration_dir(root) / f"cal-{content_hash}.json"


def participant_dir(participant_id: str, root: Path | None = None) -> Path:
    """Directory holding all sessions for one participant, `sub-XXXX/`.

    Args:
        participant_id: Pseudonymous participant ID, `sub-XXXX`.
        root: Data root to use, or `None` to use :func:`data_root`.

    Returns:
        The path to `sub-XXXX/`.

    Raises:
        ValueError: If `participant_id` does not match `sub-[0-9]{4}`.
    """
    _validate(participant_id, _PARTICIPANT_ID_RE, "participant_id")
    return (root or data_root()) / participant_id


def session_dir(participant_id: str, session_id: str, root: Path | None = None) -> Path:
    """Directory holding all files for one session, `sub-XXXX/ses-.../`.

    Args:
        participant_id: Pseudonymous participant ID, `sub-XXXX`.
        session_id: Session ID, `ses-YYYYMMDDTHHMMSS`.
        root: Data root to use, or `None` to use :func:`data_root`.

    Returns:
        The path to `sub-XXXX/ses-YYYYMMDDTHHMMSS/`.

    Raises:
        ValueError: If `participant_id` or `session_id` fail validation.
    """
    _validate(session_id, _SESSION_ID_RE, "session_id")
    return participant_dir(participant_id, root) / session_id


def session_json_path(participant_id: str, session_id: str, root: Path | None = None) -> Path:
    """Path to a session's `session.json`.

    Args:
        participant_id: Pseudonymous participant ID, `sub-XXXX`.
        session_id: Session ID, `ses-YYYYMMDDTHHMMSS`.
        root: Data root to use, or `None` to use :func:`data_root`.

    Returns:
        The path to `session.json` within the session directory.
    """
    return session_dir(participant_id, session_id, root) / "session.json"


def session_manifest_path(participant_id: str, session_id: str, root: Path | None = None) -> Path:
    """Path to a session's integrity manifest, `MANIFEST.sha256`.

    Args:
        participant_id: Pseudonymous participant ID, `sub-XXXX`.
        session_id: Session ID, `ses-YYYYMMDDTHHMMSS`.
        root: Data root to use, or `None` to use :func:`data_root`.

    Returns:
        The path to `MANIFEST.sha256` within the session directory.
    """
    return session_dir(participant_id, session_id, root) / "MANIFEST.sha256"


def beh_dir(participant_id: str, session_id: str, root: Path | None = None) -> Path:
    """Directory holding one session's behavioral data files, `.../beh/`.

    Args:
        participant_id: Pseudonymous participant ID, `sub-XXXX`.
        session_id: Session ID, `ses-YYYYMMDDTHHMMSS`.
        root: Data root to use, or `None` to use :func:`data_root`.

    Returns:
        The path to the session's `beh/` directory.
    """
    return session_dir(participant_id, session_id, root) / "beh"


def _beh_stem(
    participant_id: str,
    session_id: str,
    task_id: str,
    eye: str,
    run: int,
) -> str:
    _validate(participant_id, _PARTICIPANT_ID_RE, "participant_id")
    _validate(session_id, _SESSION_ID_RE, "session_id")
    _validate(task_id, _TASK_ID_RE, "task_id")
    _validate(eye, _EYE_RE, "eye")
    if run < 1:
        raise ValueError(f"run must be >= 1, got {run}")
    return f"{participant_id}_{session_id}_task-{task_id}_eye-{eye}_run-{run}"


def trials_tsv_path(
    participant_id: str,
    session_id: str,
    task_id: str,
    eye: str,
    run: int,
    root: Path | None = None,
) -> Path:
    """Path to one test run's per-trial data file, `..._trials.tsv`.

    Args:
        participant_id: Pseudonymous participant ID, `sub-XXXX`.
        session_id: Session ID, `ses-YYYYMMDDTHHMMSS`.
        task_id: The test's `TestSpec.id` (snake_case).
        eye: Eye tested: `"OD"`, `"OS"`, or `"OU"`.
        run: 1-based run number.
        root: Data root to use, or `None` to use :func:`data_root`.

    Returns:
        The path to `..._trials.tsv` within the session's `beh/` directory.

    Raises:
        ValueError: If any identifier fails validation.
    """
    stem = _beh_stem(participant_id, session_id, task_id, eye, run)
    return beh_dir(participant_id, session_id, root) / f"{stem}_trials.tsv"


def trials_sidecar_path(
    participant_id: str,
    session_id: str,
    task_id: str,
    eye: str,
    run: int,
    root: Path | None = None,
) -> Path:
    """Path to the JSON sidecar for a trials file, `..._trials.json`.

    See :func:`trials_tsv_path` for argument documentation.

    Returns:
        The path to `..._trials.json` within the session's `beh/` directory.
    """
    stem = _beh_stem(participant_id, session_id, task_id, eye, run)
    return beh_dir(participant_id, session_id, root) / f"{stem}_trials.json"


def summary_json_path(
    participant_id: str,
    session_id: str,
    task_id: str,
    eye: str,
    run: int,
    root: Path | None = None,
) -> Path:
    """Path to a test run's summary file, `..._summary.json`.

    See :func:`trials_tsv_path` for argument documentation.

    Returns:
        The path to `..._summary.json` within the session's `beh/` directory.
    """
    stem = _beh_stem(participant_id, session_id, task_id, eye, run)
    return beh_dir(participant_id, session_id, root) / f"{stem}_summary.json"


def frames_tsv_path(
    participant_id: str,
    session_id: str,
    task_id: str,
    eye: str,
    run: int,
    root: Path | None = None,
) -> Path:
    """Path to a test run's frame-interval log, `..._frames.tsv`.

    See :func:`trials_tsv_path` for argument documentation.

    Returns:
        The path to `..._frames.tsv` within the session's `beh/` directory.
    """
    stem = _beh_stem(participant_id, session_id, task_id, eye, run)
    return beh_dir(participant_id, session_id, root) / f"{stem}_frames.tsv"


def ensure_dir_exists(path: Path) -> Path:
    """Create `path` (and parents) if it does not already exist.

    Args:
        path: The directory to create.

    Returns:
        `path`, for chaining.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path
