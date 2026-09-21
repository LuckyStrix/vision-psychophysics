"""Run-screen logic: status-file polling, progress text, and exit-code explanations.

The Run screen never opens a stimulus window itself -- it launches
`vpsych-run` as a subprocess (see `vpsych.app.runner_process`) and polls the
`RunnerStatus` file that subprocess writes. This module holds the pure
logic: reading a status file defensively (a poll can race a concurrent
atomic write), turning a status into progress text, and turning a terminal
`RunnerExitCode` into a clear, specific explanation -- never a generic
"something went wrong".
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from vpsych.runner.status import RunnerExitCode, RunnerStatus

#: Poll interval for the Run screen's status-file timer, in milliseconds.
POLL_INTERVAL_MS = 250


def read_status_safely(path: Path) -> RunnerStatus | None:
    """Read a `RunnerStatus` file, tolerating the file not existing yet.

    `RunnerStatus.write_atomic` guarantees a reader never observes a
    partially written file (temp-file-then-rename), so the only expected
    failure mode here is the file not existing yet (the subprocess hasn't
    written its first status update) -- reported as `None`, not an
    exception, so a polling timer can call this every tick without special
    casing.

    Args:
        path: Path to the status JSON file.

    Returns:
        The parsed status, or `None` if the file does not exist (yet).

    Raises:
        json.JSONDecodeError: If the file exists but is not valid JSON --
            this should not happen given atomic writes, so it is allowed to
            propagate rather than being silently swallowed.
    """
    if not path.exists():
        return None
    return RunnerStatus.read(path)


def progress_fraction(status: RunnerStatus) -> float | None:
    """Compute a coarse 0-1 overall session progress fraction, if derivable.

    Combines the current test's index (out of `n_tests`) with within-test
    trial progress (`trial_index` out of `n_trials_expected`) when both are
    known.

    Args:
        status: The current runner status.

    Returns:
        A fraction in `[0, 1]`, or `None` if `n_tests` is 0 (not yet known).
    """
    if status.n_tests <= 0:
        return None
    per_test = 1.0 / status.n_tests
    base = status.current_test_index * per_test
    within_test = 0.0
    if (
        status.trial_index is not None
        and status.n_trials_expected is not None
        and status.n_trials_expected > 0
    ):
        within_test = per_test * min(1.0, status.trial_index / status.n_trials_expected)
    return min(1.0, base + within_test)


_STATE_LABELS = {
    "starting": "Starting",
    "calibrating_refresh": "Measuring display refresh rate",
    "instructions": "Showing instructions",
    "practice": "Practice trials",
    "running": "Running",
    "break": "Break",
    "finished": "Finished",
    "aborted": "Aborted",
    "error": "Error",
}


def progress_text(status: RunnerStatus) -> str:
    """Build a short, plain-language progress line for the current status.

    Args:
        status: The current runner status.

    Returns:
        A one-line status string, e.g. `"Running visual_acuity — trial 12 of
        ~40 (test 1 of 3)"`. Falls back to the state label alone, or
        `status.message`, when task/trial detail isn't available.
    """
    label = _STATE_LABELS.get(status.state, status.state)
    parts = [label]
    if status.task_id:
        parts[0] = f"{label}: {status.task_id}"
    if status.trial_index is not None:
        if status.n_trials_expected:
            parts.append(f"trial {status.trial_index + 1} of ~{status.n_trials_expected}")
        else:
            parts.append(f"trial {status.trial_index + 1}")
    if status.n_tests:
        parts.append(f"(test {status.current_test_index + 1} of {status.n_tests})")
    text = " — ".join(parts) if len(parts) > 1 else parts[0]
    if status.message and status.message not in text:
        text = f"{text}. {status.message}"
    return text


@dataclass(frozen=True)
class ExitOutcome:
    """A terminal run outcome, ready for display.

    Attributes:
        title: Short headline, e.g. `"Session complete"`.
        explanation: Full plain-language explanation of what happened and,
            where applicable, what to do next.
        is_success: Whether the session completed normally
            (`RunnerExitCode.OK`).
        data_retained: Whether any data collected before the outcome is
            still on disk (true for everything except a pre-flight refusal
            that never wrote a session at all).
    """

    title: str
    explanation: str
    is_success: bool
    data_retained: bool


def explain_exit(code: RunnerExitCode, status: RunnerStatus | None) -> ExitOutcome:
    """Turn a terminal `RunnerExitCode` (plus the last known status) into a clear explanation.

    Every `RunnerExitCode` value is handled distinctly and by name -- never
    a generic fallback message -- per the project's rule that run failures
    must be explained clearly. Session data already written to disk is
    never deleted regardless of outcome; this function only describes
    what happened.

    Args:
        code: The runner subprocess's exit code.
        status: The last `RunnerStatus` read before the process exited, if
            any (used for its `error`/`message` detail).

    Returns:
        The constructed `ExitOutcome`.
    """
    detail = status.error if status and status.error else (status.message if status else None)

    if code == RunnerExitCode.OK:
        return ExitOutcome(
            title="Session complete",
            explanation="The session finished normally. Results are available on the Results "
            "screen.",
            is_success=True,
            data_retained=True,
        )
    if code == RunnerExitCode.ABORTED_BY_USER:
        return ExitOutcome(
            title="Session aborted",
            explanation="The session was aborted before it finished (e.g. the participant "
            "pressed the quit key, or the window was closed). Trials recorded before the "
            "abort are kept on disk and marked as an incomplete session.",
            is_success=False,
            data_retained=True,
        )
    if code == RunnerExitCode.REFRESH_MISMATCH:
        return ExitOutcome(
            title="Display refresh rate does not match calibration",
            explanation=(
                "The display's measured refresh rate differed from the calibration's recorded "
                "refresh rate by more than 1%, so the session was refused before any stimuli "
                "were shown. This usually means the OS/driver is running the display at a "
                "different refresh rate than when it was calibrated (e.g. 59.94 Hz vs 60 Hz, or "
                "a different mode entirely). Recalibrate at the current refresh rate, or fix "
                "the display mode to match the calibration, then try again."
                + (f" Detail: {detail}" if detail else "")
            ),
            is_success=False,
            data_retained=False,
        )
    if code == RunnerExitCode.REQUIREMENTS_UNMET:
        return ExitOutcome(
            title="Session requirements not met",
            explanation=(
                "One or more planned tests could not run under the active calibration/display "
                "(for example, a test needing a gamma calibration while the display is "
                "uncalibrated). Nothing was presented to the participant. Review the reasons "
                "below, fix them (often: run the calibration wizard, or lower the required "
                "grade in the session builder), and try again."
                + (f" Reasons: {detail}" if detail else "")
            ),
            is_success=False,
            data_retained=False,
        )
    if code == RunnerExitCode.ERROR:
        return ExitOutcome(
            title="Session error",
            explanation=(
                "An unexpected error stopped the session. Any trials already recorded before "
                "the error remain on disk and the session is marked incomplete, not deleted."
                + (f" Detail: {detail}" if detail else "")
            ),
            is_success=False,
            data_retained=True,
        )
    return ExitOutcome(
        title=f"Unknown exit code ({int(code)})",
        explanation="The runner subprocess exited with an exit code this version of the app "
        "does not recognize. Check the runner logs for detail.",
        is_success=False,
        data_retained=True,
    )


def json_decode_is_transient(path: Path) -> bool:
    """Best-effort check whether a status-file JSON parse failure is a transient read race.

    `RunnerStatus.write_atomic` uses rename-based atomic writes, so a
    genuinely torn read should not be possible on POSIX -- but a poller
    calling `read_status_safely` in a tight loop could still observe the
    file mid-rename on some filesystems/OSes. This helper lets a caller
    that caught a `json.JSONDecodeError` decide whether to retry next tick
    (file still exists, plausibly transient) versus surface an error (file
    now gone, or exists but was never valid -- a real bug).

    Args:
        path: The status file path that failed to parse.

    Returns:
        `True` if the path still exists (worth retrying next poll tick).
    """
    return path.exists()


__all__ = [
    "POLL_INTERVAL_MS",
    "ExitOutcome",
    "explain_exit",
    "json_decode_is_transient",
    "progress_fraction",
    "progress_text",
    "read_status_safely",
]

# Re-exported for convenience so callers need not import `json` themselves
# just to catch the decode error `read_status_safely` may propagate.
JSONDecodeError = json.JSONDecodeError
