"""Crash-safe session data writer.

`SessionWriter` is the only code path that writes trial/summary/frame data
to a session directory (see `vpsych.data.paths`). It exists to enforce the
project's *no silent data loss* rule: every trial is appended and
flushed/fsynced to disk before the next trial begins, summaries are written
atomically (temp file + rename) so a reader never sees a half-written file,
and an interrupted session is left on disk marked `status: incomplete`
rather than deleted.

Crash safety in practice: if the process is killed (including `os._exit`,
which skips `__exit__`/context-manager cleanup) after `append_trial`
returns, that trial's row is durably on disk -- `fsync` completed before the
call returned. `session.json` is left with `status: "running"` in that case
(never updated to a terminal status), which `vpsych.data.validate` treats as
an expected, non-corrupt outcome rather than an error.
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType

from vpsych.core.trial import TrialRecord
from vpsych.data import paths
from vpsych.data.schemas import ColumnSidecar, SessionInfo, SessionStatus, TestSummary

#: Default value of `SessionWriter.readonly_after_finalize`: whether files
#: are chmod'd to 0o444 (read-only for everyone) once `finalize` completes.
#: Configurable per-instance via the constructor because some test
#: environments (e.g. a tmp_path fixture cleaned up by shutil.rmtree) find
#: read-only files inconvenient; production code should leave this `True`.
DEFAULT_READONLY_AFTER_FINALIZE = True

#: Column documentation for every `TrialRecord` field, used to generate the
#: `_trials.json` sidecar the first time a given trials TSV is written.
#: Keys must exactly match `TrialRecord.model_fields` (checked in
#: `tests/data/test_writer.py`).
TRIAL_COLUMN_SIDECARS: dict[str, ColumnSidecar] = {
    "participant_id": ColumnSidecar(description="Pseudonymous participant ID.", units=None),
    "session_id": ColumnSidecar(description="Session identifier, ses-YYYYMMDDTHHMMSS.", units=None),
    "task_id": ColumnSidecar(description="Test spec id (snake_case).", units=None),
    "task_version": ColumnSidecar(
        description="Test spec semantic version at the time this trial was run.", units=None
    ),
    "run": ColumnSidecar(description="1-based run number.", units=None),
    "eye": ColumnSidecar(
        description="Eye tested.",
        units=None,
        levels={"OD": "Right eye", "OS": "Left eye", "OU": "Both eyes (binocular)"},
    ),
    "block": ColumnSidecar(
        description="Trial block.",
        units=None,
        levels={
            "practice": "Practice trial with feedback, not analyzed.",
            "main": "Main-block trial, no feedback, analyzed.",
        },
    ),
    "trial_index": ColumnSidecar(
        description="0-based trial index within this block and run.", units=None
    ),
    "is_catch": ColumnSidecar(
        description="Whether this was a suprathreshold catch trial (estimates lapse rate).",
        units=None,
        levels={"True": "Catch trial", "False": "Not a catch trial"},
    ),
    "intensity": ColumnSidecar(
        description="Scalar stimulus intensity presented, in intensity_units.",
        units="see intensity_units column",
    ),
    "intensity_units": ColumnSidecar(
        description="Units of the intensity column for this row, e.g. logMAR, log10_contrast.",
        units=None,
    ),
    "stimulus_params": ColumnSidecar(
        description="Full stimulus parameterization (test-specific keys/values), JSON-encoded.",
        units=None,
    ),
    "correct_response": ColumnSidecar(
        description="The response that would have been scored correct for this stimulus.",
        units=None,
    ),
    "response": ColumnSidecar(
        description="The observer's actual response, or n/a if none was given (e.g. timeout).",
        units=None,
    ),
    "correct": ColumnSidecar(
        description="Whether response was scored correct, or n/a if unscored.",
        units=None,
        levels={"True": "Correct", "False": "Incorrect", "n/a": "Unscored (no response)"},
    ),
    "rt_s": ColumnSidecar(
        description="Response time from stimulus onset to the response event, or n/a if none.",
        units="s",
    ),
    "stimulus_onset_s": ColumnSidecar(
        description="Stimulus-onset flip timestamp, on the runner's monotonic clock.", units="s"
    ),
    "n_dropped_frames_trial": ColumnSidecar(
        description="Dropped frames detected during this trial's stimulus presentation.",
        units=None,
    ),
    "procedure_state": ColumnSidecar(
        description=(
            "JSON-encoded snapshot of the driving adaptive procedure's internal state "
            "immediately after this trial."
        ),
        units=None,
    ),
    "timestamp_utc": ColumnSidecar(
        description="Wall-clock UTC timestamp this trial was recorded, ISO 8601.", units=None
    ),
    "rng_seed": ColumnSidecar(
        description="RNG seed in effect for this trial's stochastic choices.", units=None
    ),
}


class SessionAlreadyFinalizedError(RuntimeError):
    """Raised when writing to, or re-finalizing, a session that is already finalized."""


def _fsync_dir(dir_path: Path) -> None:
    """fsync a directory's entry (so a rename/create into it is durable)."""
    fd = os.open(str(dir_path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write `data` to `path` atomically: temp file in the same dir, fsync, rename, fsync dir."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
        _fsync_dir(path.parent)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(tmp_name)
        raise


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class SessionWriter:
    """Context manager that owns crash-safe writes for one session's data.

    Usage::

        with SessionWriter(participant_id, session_id, session_info, root) as writer:
            writer.append_trial(trial_record)
            ...
            writer.write_summary(task_id, eye, run, summary)
            writer.write_frames(task_id, eye, run, intervals_s)
        # __exit__ calls finalize("complete") if no exception was raised, or
        # finalize("incomplete") if one was, unless finalize() was already
        # called explicitly.

    Args:
        participant_id: Pseudonymous participant ID, `sub-XXXX`.
        session_id: Session ID, `ses-YYYYMMDDTHHMMSS`.
        session_info: Initial `SessionInfo` to write at session start
            (`status` should be `"running"`); `session.json` is updated
            again by `finalize`.
        root: Data root to write under, or `None` to use
            `vpsych.data.paths.data_root()`.
        readonly_after_finalize: Whether `finalize` chmods every file in the
            session directory to 0o444 (read-only) once the manifest is
            written. Defaults to `True`; set `False` in test fixtures that
            need to clean up the directory afterward without adjusting
            permissions first.

    Raises:
        SessionAlreadyFinalizedError: On `__enter__`, if `session.json`
            already exists with a terminal (non-`"running"`) status --
            refuses to overwrite a finalized session.
    """

    def __init__(
        self,
        participant_id: str,
        session_id: str,
        session_info: SessionInfo,
        root: Path | None = None,
        readonly_after_finalize: bool = DEFAULT_READONLY_AFTER_FINALIZE,
    ) -> None:
        self.participant_id = participant_id
        self.session_id = session_id
        self.session_info = session_info
        self.root = root
        self.readonly_after_finalize = readonly_after_finalize
        self._finalized = False
        self._written_paths: set[Path] = set()
        self._session_dir = paths.session_dir(participant_id, session_id, root)
        self._session_json_path = paths.session_json_path(participant_id, session_id, root)
        self._manifest_path = paths.session_manifest_path(participant_id, session_id, root)

    def _refuse_if_finalized_on_disk(self) -> None:
        if not self._session_json_path.exists():
            return
        try:
            existing = SessionInfo.model_validate_json(
                self._session_json_path.read_text(encoding="utf-8")
            )
        except Exception:
            # Unparsable session.json: let normal writes proceed (this is
            # not a "finalized" state we can detect); validate() will flag
            # the corrupt file separately.
            return
        if existing.status != "running":
            raise SessionAlreadyFinalizedError(
                f"Session {self.session_id!r} for {self.participant_id!r} is already "
                f"finalized with status {existing.status!r}; refusing to overwrite."
            )

    def __enter__(self) -> SessionWriter:
        self._refuse_if_finalized_on_disk()
        paths.ensure_dir_exists(self._session_dir)
        paths.ensure_dir_exists(paths.beh_dir(self.participant_id, self.session_id, self.root))
        self._write_session_json(self.session_info)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._finalized:
            return
        status: SessionStatus = "complete" if exc_type is None else "incomplete"
        self.finalize(status)

    def _write_session_json(self, info: SessionInfo) -> None:
        data = info.model_dump_json(indent=2).encode("utf-8")
        _atomic_write_bytes(self._session_json_path, data)
        self._written_paths.add(self._session_json_path)

    def append_trial(self, trial: TrialRecord) -> None:
        """Append one trial to the current test run's `_trials.tsv`, flushed and fsynced.

        Must complete (including the fsync) before returning, so that if
        the process is killed immediately afterward, this trial is durably
        on disk. Writes the header row (from `TrialRecord.to_tsv_row()`
        keys) on the first call for a given trials file, and writes the
        `_trials.json` column sidecar (see `TRIAL_COLUMN_SIDECARS`) the
        first time that trials file is created.

        Args:
            trial: The trial record to append.

        Raises:
            SessionAlreadyFinalizedError: If this session has already been
                finalized.
        """
        if self._finalized:
            raise SessionAlreadyFinalizedError("Cannot append a trial to a finalized session.")
        tsv_path = paths.trials_tsv_path(
            self.participant_id, self.session_id, trial.task_id, trial.eye, trial.run, self.root
        )
        row = trial.to_tsv_row()
        paths.ensure_dir_exists(tsv_path.parent)
        write_header = not tsv_path.exists()
        with open(tsv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=list(row.keys()), delimiter="\t", lineterminator="\n"
            )
            if write_header:
                writer.writeheader()
            writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())
        self._written_paths.add(tsv_path)

        sidecar_path = paths.trials_sidecar_path(
            self.participant_id, self.session_id, trial.task_id, trial.eye, trial.run, self.root
        )
        if write_header or not sidecar_path.exists():
            self._write_trials_sidecar(sidecar_path, list(row.keys()))

    def _write_trials_sidecar(self, path: Path, columns: list[str]) -> None:
        sidecar = {
            col: TRIAL_COLUMN_SIDECARS[col].model_dump()
            if col in TRIAL_COLUMN_SIDECARS
            else ColumnSidecar(description=col).model_dump()
            for col in columns
        }
        data = json.dumps(sidecar, indent=2, sort_keys=True).encode("utf-8")
        _atomic_write_bytes(path, data)
        self._written_paths.add(path)

    def write_summary(self, task_id: str, eye: str, run: int, summary: TestSummary) -> None:
        """Atomically write a test run's `..._summary.json`.

        Writes to a temporary file in the same directory and renames it
        into place, so a concurrent reader never observes a partially
        written summary file.

        Args:
            task_id: The test's `TestSpec.id`.
            eye: Eye tested: `"OD"`, `"OS"`, or `"OU"`.
            run: 1-based run number.
            summary: The summary to write.

        Raises:
            SessionAlreadyFinalizedError: If this session has already been
                finalized.
        """
        if self._finalized:
            raise SessionAlreadyFinalizedError("Cannot write a summary to a finalized session.")
        path = paths.summary_json_path(
            self.participant_id, self.session_id, task_id, eye, run, self.root
        )
        data = summary.model_dump_json(indent=2).encode("utf-8")
        _atomic_write_bytes(path, data)
        self._written_paths.add(path)

    def write_frames(self, task_id: str, eye: str, run: int, intervals_s: list[float]) -> None:
        """Append to a test run's frame-interval log, `..._frames.tsv`.

        Args:
            task_id: The test's `TestSpec.id`.
            eye: Eye tested: `"OD"`, `"OS"`, or `"OU"`.
            run: 1-based run number.
            intervals_s: Measured inter-flip intervals for this run, in
                seconds (see `vpsych.core.timing.summarize_frame_intervals`).

        Raises:
            SessionAlreadyFinalizedError: If this session has already been
                finalized.
        """
        if self._finalized:
            raise SessionAlreadyFinalizedError("Cannot write frames to a finalized session.")
        path = paths.frames_tsv_path(
            self.participant_id, self.session_id, task_id, eye, run, self.root
        )
        paths.ensure_dir_exists(path.parent)
        write_header = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t", lineterminator="\n")
            if write_header:
                writer.writerow(["frame_interval_s"])
            for interval in intervals_s:
                writer.writerow([interval])
            f.flush()
            os.fsync(f.fileno())
        self._written_paths.add(path)

    def finalize(self, status: SessionStatus) -> None:
        """Mark the session finished, writing the final `session.json` and `MANIFEST.sha256`.

        Never deletes any file, regardless of `status` -- an aborted or
        incomplete session's partial data remains on disk, marked as such,
        per the project's no-silent-data-loss rule. The manifest is
        computed by walking the session directory on disk (not just the
        files this instance wrote), in `sha256sum -c`-compatible format
        (`<hex digest><two spaces><path relative to the session dir>`).

        Args:
            status: Final status to record: `"complete"`, `"incomplete"`,
                or `"aborted"` (`"running"` is not a valid final status).

        Raises:
            ValueError: If `status == "running"`.
            SessionAlreadyFinalizedError: If this session has already been
                finalized (in this instance, or on disk).
        """
        if status == "running":
            raise ValueError("'running' is not a valid final status for finalize().")
        if self._finalized:
            raise SessionAlreadyFinalizedError("Session has already been finalized.")
        self._refuse_if_finalized_on_disk()

        final_info = self.session_info.model_copy(
            update={"status": status, "ended_utc": datetime.now(timezone.utc)}
        )
        self._write_session_json(final_info)
        self.session_info = final_info

        all_files = sorted(
            p for p in self._session_dir.rglob("*") if p.is_file() and p != self._manifest_path
        )
        manifest_lines = [
            f"{_sha256_file(p)}  {p.relative_to(self._session_dir).as_posix()}\n" for p in all_files
        ]
        _atomic_write_bytes(self._manifest_path, "".join(manifest_lines).encode("utf-8"))
        self._written_paths.add(self._manifest_path)

        self._finalized = True

        if self.readonly_after_finalize:
            for p in sorted({*all_files, self._manifest_path}):
                with contextlib.suppress(OSError):
                    os.chmod(p, 0o444)
