"""Data-root-level management: dataset init, participants, calibration store.

This module owns the three dataset-wide files/directories that live
alongside (not inside) any one session: `dataset_description.json`,
`participants.tsv` + `participants.json`, and `calibration/`. All writes are
atomic (temp file + fsync + rename) so a reader never observes a
half-written file; `participants.tsv`/`.json` are rewritten wholesale on
every change (the participant list is small -- one row per person, not per
trial -- so this is simpler and just as safe as an append-only log).
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from vpsych.core.calibration.models import Calibration
from vpsych.data import paths
from vpsych.data.schemas import SCHEMA_VERSION, ColumnSidecar, DatasetDescription, Participant
from vpsych.data.writer import _atomic_write_bytes

_PARTICIPANT_ID_PATTERN_DIGITS = 4

#: Column documentation for `participants.tsv`, written to `participants.json`.
PARTICIPANT_COLUMN_SIDECARS: dict[str, ColumnSidecar] = {
    "participant_id": ColumnSidecar(description="Pseudonymous participant ID.", units=None),
    "year_of_birth": ColumnSidecar(
        description="Birth year only (not full date of birth).", units=None
    ),
    "sex": ColumnSidecar(description="Self-reported sex, free-text/short-code.", units=None),
    "refractive_correction": ColumnSidecar(
        description="Refractive correction worn during testing (e.g. glasses, contacts, none).",
        units=None,
    ),
    "notes": ColumnSidecar(
        description="Free-text notes. Must not contain identifying information.", units=None
    ),
}


class ParticipantNotFoundError(KeyError):
    """Raised when a `participant_id` is not found in `participants.tsv`."""


def _git_commit(cwd: Path | None = None) -> str | None:
    """Best-effort `git rev-parse HEAD` for the vpsych checkout, or `None` if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd else Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def init_dataset(
    root: Path | None = None, name: str = "vpsych dataset", force: bool = False
) -> DatasetDescription:
    """Initialize a vpsych data root: create it and write `dataset_description.json`.

    Idempotent by default: if `dataset_description.json` already exists and
    `force` is `False`, the existing description is returned unchanged. Also
    ensures `calibration/` exists and that `participants.tsv`/`.json` exist
    (empty) if not already present.

    Args:
        root: Data root to initialize, or `None` to use
            `vpsych.data.paths.data_root()`.
        name: Human-readable dataset name, used only if the dataset is
            being created (or `force=True`).
        force: If `True`, overwrite an existing `dataset_description.json`
            with a freshly created one.

    Returns:
        The dataset's `DatasetDescription` (existing or newly created).
    """
    import vpsych

    root = root or paths.data_root()
    paths.ensure_dir_exists(root)
    paths.ensure_dir_exists(paths.calibration_dir(root))

    desc_path = paths.dataset_description_path(root)
    if desc_path.exists() and not force:
        return DatasetDescription.model_validate_json(desc_path.read_text(encoding="utf-8"))

    description = DatasetDescription(
        schema_version=SCHEMA_VERSION,
        software_version=vpsych.__version__,
        git_commit=_git_commit(),
        name=name,
        created_utc=datetime.now(timezone.utc),
    )
    _atomic_write_bytes(desc_path, description.model_dump_json(indent=2).encode("utf-8"))

    if not paths.participants_tsv_path(root).exists():
        _write_participants(root, [])

    return description


def load_dataset_description(root: Path | None = None) -> DatasetDescription:
    """Load and validate `dataset_description.json` from a data root.

    Args:
        root: Data root, or `None` to use `vpsych.data.paths.data_root()`.

    Returns:
        The parsed `DatasetDescription`.

    Raises:
        FileNotFoundError: If the data root has not been initialized.
    """
    path = paths.dataset_description_path(root)
    if not path.exists():
        raise FileNotFoundError(f"No dataset_description.json at {path}; call init_dataset first.")
    return DatasetDescription.model_validate_json(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Participants
# ---------------------------------------------------------------------------


def _participant_to_row(p: Participant) -> dict[str, str]:
    row: dict[str, str] = {}
    for name, value in p.model_dump(mode="python").items():
        row[name] = "n/a" if value is None else str(value)
    return row


def _row_to_participant(row: dict[str, str]) -> Participant:
    kwargs: dict[str, object] = {}
    for key, value in row.items():
        if value == "n/a" or value == "":
            kwargs[key] = None
        elif key == "year_of_birth":
            kwargs[key] = int(value)
        else:
            kwargs[key] = value
    return Participant.model_validate(kwargs)


def _write_participants(root: Path, participants: list[Participant]) -> None:
    fieldnames = list(Participant.model_fields.keys())
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for p in sorted(participants, key=lambda p: p.participant_id):
        writer.writerow(_participant_to_row(p))
    _atomic_write_bytes(paths.participants_tsv_path(root), buf.getvalue().encode("utf-8"))

    sidecar = {col: PARTICIPANT_COLUMN_SIDECARS[col].model_dump() for col in fieldnames}
    _atomic_write_bytes(
        paths.participants_json_path(root),
        json.dumps(sidecar, indent=2, sort_keys=True).encode("utf-8"),
    )


def list_participants(root: Path | None = None) -> list[Participant]:
    """List every participant registered in `participants.tsv`.

    Args:
        root: Data root, or `None` to use `vpsych.data.paths.data_root()`.

    Returns:
        Participants in `participant_id` order, or an empty list if
        `participants.tsv` does not exist yet.
    """
    root = root or paths.data_root()
    path = paths.participants_tsv_path(root)
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return [_row_to_participant(row) for row in reader]


def get_participant(participant_id: str, root: Path | None = None) -> Participant | None:
    """Look up one participant by ID.

    Args:
        participant_id: The `sub-XXXX` ID to look up.
        root: Data root, or `None` to use `vpsych.data.paths.data_root()`.

    Returns:
        The matching `Participant`, or `None` if not found.
    """
    for p in list_participants(root):
        if p.participant_id == participant_id:
            return p
    return None


def _next_free_participant_id(existing: list[Participant]) -> str:
    used = {int(p.participant_id.removeprefix("sub-")) for p in existing}
    n = 1
    while n in used:
        n += 1
    return f"sub-{n:04d}"


def create_participant(
    root: Path | None = None,
    year_of_birth: int | None = None,
    sex: str | None = None,
    refractive_correction: str | None = None,
    notes: str | None = None,
) -> Participant:
    """Create a new participant with the next free `sub-NNNN` ID.

    "Next free" fills gaps: if `sub-0001` and `sub-0003` exist, the next
    participant gets `sub-0002`, not `sub-0004`.

    Args:
        root: Data root, or `None` to use `vpsych.data.paths.data_root()`.
        year_of_birth: Birth year only, or `None`.
        sex: Self-reported sex, or `None`.
        refractive_correction: Refractive correction worn during testing,
            or `None`.
        notes: Free-text pseudonymous notes, or `None`.

    Returns:
        The newly created `Participant`.
    """
    root = root or paths.data_root()
    existing = list_participants(root)
    new_id = _next_free_participant_id(existing)
    participant = Participant(
        participant_id=new_id,
        year_of_birth=year_of_birth,
        sex=sex,
        refractive_correction=refractive_correction,
        notes=notes,
    )
    _write_participants(root, [*existing, participant])
    return participant


def update_participant(
    participant_id: str,
    /,
    root: Path | None = None,
    **fields: object,
) -> Participant:
    """Update fields on an existing participant.

    `participant_id` is positional-only so that `participant_id="..."` in
    `**fields` (an attempt to change the ID, which is not allowed) is a
    `ValueError` from this function rather than a `TypeError` from Python's
    own duplicate-keyword-argument check.

    Args:
        participant_id: The `sub-XXXX` ID to update.
        root: Data root, or `None` to use `vpsych.data.paths.data_root()`.
        **fields: New values for any of `year_of_birth`, `sex`,
            `refractive_correction`, `notes` (unspecified fields are left
            unchanged). `participant_id` cannot be changed this way.

    Returns:
        The updated `Participant`.

    Raises:
        ParticipantNotFoundError: If `participant_id` is not registered.
        ValueError: If `fields` includes `"participant_id"`.
    """
    if "participant_id" in fields:
        raise ValueError("participant_id cannot be changed; create a new participant instead.")
    root = root or paths.data_root()
    existing = list_participants(root)
    updated: list[Participant] = []
    found = False
    for p in existing:
        if p.participant_id == participant_id:
            p = Participant.model_validate({**p.model_dump(), **fields})  # validates the update
            found = True
        updated.append(p)
    if not found:
        raise ParticipantNotFoundError(participant_id)
    _write_participants(root, updated)
    result = get_participant(participant_id, root)
    assert result is not None  # just wrote it
    return result


# ---------------------------------------------------------------------------
# Calibration store
# ---------------------------------------------------------------------------


class CalibrationIntegrityError(ValueError):
    """Raised when a loaded calibration file's content hash doesn't match its filename."""


def save_calibration(
    calibration: Calibration, root: Path | None = None, force: bool = False
) -> Path:
    """Store an immutable calibration as `calibration/cal-<hash>.json`.

    If a calibration with the same content hash already exists, this is a
    no-op (calibrations are content-addressed, so an identical calibration
    is already stored) unless `force=True`, which rewrites the file (still
    with the same content, since the hash is derived from the content).

    Args:
        calibration: The calibration to store.
        root: Data root, or `None` to use `vpsych.data.paths.data_root()`.
        force: Rewrite the file even if it already exists.

    Returns:
        The path the calibration was (or already was) stored at.
    """
    root = root or paths.data_root()
    content_hash = calibration.content_hash()
    path = paths.calibration_path(content_hash, root)
    if path.exists() and not force:
        return path
    _atomic_write_bytes(path, calibration.model_dump_json(indent=2).encode("utf-8"))
    with contextlib.suppress(OSError):
        path.chmod(0o444)
    return path


def load_calibration(content_hash: str, root: Path | None = None) -> Calibration:
    """Load a stored calibration and verify its content hash.

    Args:
        content_hash: The calibration's content hash (as used in its
            filename).
        root: Data root, or `None` to use `vpsych.data.paths.data_root()`.

    Returns:
        The loaded, verified `Calibration`.

    Raises:
        FileNotFoundError: If no calibration with that hash is stored.
        CalibrationIntegrityError: If the file's actual content hash does
            not match `content_hash` (the file was modified after being
            written -- calibrations must be immutable).
    """
    root = root or paths.data_root()
    path = paths.calibration_path(content_hash, root)
    if not path.exists():
        raise FileNotFoundError(f"No calibration stored with hash {content_hash!r} at {path}.")
    calibration = Calibration.model_validate_json(path.read_text(encoding="utf-8"))
    actual = calibration.content_hash()
    if actual != content_hash:
        raise CalibrationIntegrityError(
            f"Calibration at {path} has content hash {actual!r}, expected {content_hash!r} "
            "(the file may have been modified after being written)."
        )
    return calibration


def list_calibrations(root: Path | None = None) -> list[Calibration]:
    """List every stored calibration.

    Args:
        root: Data root, or `None` to use `vpsych.data.paths.data_root()`.

    Returns:
        All calibrations found in `calibration/`, sorted by `created_utc`
        ascending. Files that fail to parse or fail hash verification are
        skipped (use `vpsych.data.validate` to surface those as errors).
    """
    root = root or paths.data_root()
    cal_dir = paths.calibration_dir(root)
    if not cal_dir.exists():
        return []
    calibrations: list[Calibration] = []
    for f in sorted(cal_dir.glob("cal-*.json")):
        content_hash = f.stem.removeprefix("cal-")
        try:
            calibrations.append(load_calibration(content_hash, root))
        except (ValueError, OSError):
            continue
    return sorted(calibrations, key=lambda c: c.created_utc)


def latest_calibration(root: Path | None = None) -> Calibration | None:
    """Return the most recently created stored calibration.

    Args:
        root: Data root, or `None` to use `vpsych.data.paths.data_root()`.

    Returns:
        The calibration with the latest `created_utc`, or `None` if none
        are stored.
    """
    calibrations = list_calibrations(root)
    return calibrations[-1] if calibrations else None


__all__ = [
    "CalibrationIntegrityError",
    "ParticipantNotFoundError",
    "create_participant",
    "get_participant",
    "init_dataset",
    "latest_calibration",
    "list_calibrations",
    "list_participants",
    "load_calibration",
    "load_dataset_description",
    "save_calibration",
    "update_participant",
]
