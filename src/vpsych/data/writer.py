"""Crash-safe session data writer.

`SessionWriter` is the only code path that writes trial/summary/frame data
to a session directory (see `vpsych.data.paths`). It exists to enforce the
project's *no silent data loss* rule: every trial is appended and
flushed/fsynced to disk before the next trial begins, summaries are written
atomically (temp file + rename) so a reader never sees a half-written file,
and an interrupted session is left on disk marked `status: incomplete`
rather than deleted.

Not implemented in Phase 0 -- this module freezes the constructor and
method signatures; bodies raise `NotImplementedError` until a later phase
implements the actual file I/O.
"""

from __future__ import annotations

from pathlib import Path
from types import TracebackType

from vpsych.core.trial import TrialRecord
from vpsych.data.schemas import SessionInfo, SessionStatus, TestSummary


class SessionWriter:
    """Context manager that owns crash-safe writes for one session's data.

    Usage::

        with SessionWriter(participant_id, session_id, root) as writer:
            writer.append_trial(trial_record)
            ...
            writer.write_summary(task_id, eye, run, summary)
            writer.write_frames(task_id, eye, run, intervals_s)
        # __exit__ calls finalize() with an appropriate status if not
        # already finalized explicitly.

    Args:
        participant_id: Pseudonymous participant ID, `sub-XXXX`.
        session_id: Session ID, `ses-YYYYMMDDTHHMMSS`.
        session_info: Initial `SessionInfo` to write at session start
            (`status` should be `"running"`); `session.json` is updated
            again by `finalize`.
        root: Data root to write under, or `None` to use
            `vpsych.data.paths.data_root()`.
    """

    def __init__(
        self,
        participant_id: str,
        session_id: str,
        session_info: SessionInfo,
        root: Path | None = None,
    ) -> None:
        self.participant_id = participant_id
        self.session_id = session_id
        self.session_info = session_info
        self.root = root
        raise NotImplementedError(
            "SessionWriter is a Phase-0 interface stub; implementation lands in a later phase."
        )

    def __enter__(self) -> SessionWriter:
        raise NotImplementedError

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        raise NotImplementedError

    def append_trial(self, trial: TrialRecord) -> None:
        """Append one trial to the current test run's `_trials.tsv`, flushed and fsynced.

        Must complete (including the fsync) before returning, so that if
        the process is killed immediately afterward, this trial is durably
        on disk. Writes the header row (from `TrialRecord.to_tsv_row()`
        keys) on the first call for a given trials file.

        Args:
            trial: The trial record to append.
        """
        raise NotImplementedError

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
        """
        raise NotImplementedError

    def write_frames(self, task_id: str, eye: str, run: int, intervals_s: list[float]) -> None:
        """Write a test run's frame-interval log, `..._frames.tsv`.

        Args:
            task_id: The test's `TestSpec.id`.
            eye: Eye tested: `"OD"`, `"OS"`, or `"OU"`.
            run: 1-based run number.
            intervals_s: Measured inter-flip intervals for this run, in
                seconds (see `vpsych.core.timing.summarize_frame_intervals`).
        """
        raise NotImplementedError

    def finalize(self, status: SessionStatus) -> None:
        """Mark the session finished, writing the final `session.json` and `MANIFEST.sha256`.

        Never deletes any file, regardless of `status` -- an aborted or
        incomplete session's partial data remains on disk, marked as such,
        per the project's no-silent-data-loss rule.

        Args:
            status: Final status to record: `"complete"`, `"incomplete"`,
                or `"aborted"` (`"running"` is not a valid final status).
        """
        raise NotImplementedError
