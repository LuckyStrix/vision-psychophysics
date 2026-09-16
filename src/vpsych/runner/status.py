"""Runner <-> UI status contract.

The runner (`vpsych.runner.__main__`, a separate subprocess so Qt never
competes with stimulus timing) periodically writes a `RunnerStatus` to a
JSON file; the PySide6 app polls or watches that file to show run progress
without any IPC beyond the filesystem. The runner's process exit code
additionally reports terminal outcomes via `RunnerExitCode`.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from datetime import datetime, timezone
from enum import IntEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

RunnerState = Literal[
    "starting",
    "calibrating_refresh",
    "instructions",
    "practice",
    "running",
    "break",
    "finished",
    "aborted",
    "error",
]


class RunnerStatus(BaseModel):
    """A snapshot of runner progress, written atomically to a status JSON file.

    Attributes:
        state: Current runner state.
        current_test_index: 0-based index of the test currently running (or
            about to run) within the session plan.
        n_tests: Total number of tests in the session plan.
        task_id: `TestSpec.id` of the current test, or `None` if not yet
            known (e.g. during `"starting"`).
        trial_index: 0-based index of the current trial within the current
            test/block, or `None` if not applicable.
        n_trials_expected: Expected total number of trials for the current
            test/block, or `None` if not yet known (adaptive procedures may
            not have a fixed count).
        message: Short human-readable status message for display in the UI.
        error: Error message if `state == "error"`, otherwise `None`.
        updated_utc: UTC timestamp this status was last written.
    """

    model_config = ConfigDict(frozen=True)

    state: RunnerState = Field(description="Current runner state.")
    current_test_index: int = Field(ge=0, description="0-based index of the current test.")
    n_tests: int = Field(ge=0, description="Total number of tests in the session plan.")
    task_id: str | None = Field(default=None, description="TestSpec.id of the current test.")
    trial_index: int | None = Field(default=None, ge=0, description="0-based current trial index.")
    n_trials_expected: int | None = Field(
        default=None, ge=0, description="Expected total trials for the current test/block."
    )
    message: str = Field(default="", description="Short human-readable status message.")
    error: str | None = Field(default=None, description="Error message if state == 'error'.")
    updated_utc: datetime = Field(description="UTC timestamp this status was last written.")

    @classmethod
    def create(
        cls,
        state: RunnerState,
        current_test_index: int = 0,
        n_tests: int = 0,
        task_id: str | None = None,
        trial_index: int | None = None,
        n_trials_expected: int | None = None,
        message: str = "",
        error: str | None = None,
    ) -> RunnerStatus:
        """Construct a `RunnerStatus` with `updated_utc` set to the current UTC time.

        Convenience constructor so callers don't need to pass a timestamp
        explicitly on every status update.
        """
        return cls(
            state=state,
            current_test_index=current_test_index,
            n_tests=n_tests,
            task_id=task_id,
            trial_index=trial_index,
            n_trials_expected=n_trials_expected,
            message=message,
            error=error,
            updated_utc=datetime.now(timezone.utc),
        )

    def write_atomic(self, path: Path) -> None:
        """Write this status to `path` atomically (temp file + rename).

        Guarantees a concurrent reader (the UI) never observes a partially
        written or truncated status file.

        Args:
            path: Destination path for the status JSON file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(self.model_dump_json(indent=2))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_name, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.remove(tmp_name)
            raise

    @classmethod
    def read(cls, path: Path) -> RunnerStatus:
        """Read and parse a `RunnerStatus` previously written by `write_atomic`.

        Args:
            path: Path to the status JSON file.

        Returns:
            The parsed status.
        """
        with open(path, encoding="utf-8") as f:
            return cls.model_validate(json.load(f))


class RunnerExitCode(IntEnum):
    """Process exit codes the runner subprocess reports on termination.

    The app inspects this alongside the final `RunnerStatus` to decide how
    to present the outcome to the user.
    """

    OK = 0
    """Session completed normally."""

    ERROR = 1
    """An unexpected error occurred; see the final status's `error` field
    and runner logs."""

    ABORTED_BY_USER = 2
    """The user aborted the session (e.g. pressed the escape/quit key)."""

    REFRESH_MISMATCH = 3
    """Measured display refresh rate differed from the calibration's
    `refresh_hz` by more than the runner's tolerance (see the plan: 1%)."""

    REQUIREMENTS_UNMET = 4
    """A planned test's `TestRequirements` were not met by the active
    display/calibration at run time (see
    `vpsych.tests_catalog.base.check_requirements`)."""
