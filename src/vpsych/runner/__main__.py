"""Runner subprocess entry point: `python -m vpsych.runner` / the `vpsych-run` console script.

Runs one session end to end in its own process, so a fullscreen PsychoPy
window never has to share a process (or an event loop) with the PySide6
app. Writes trial data via `vpsych.data.writer.SessionWriter` and a
`vpsych.runner.status.RunnerStatus` snapshot to `--status-file` as it goes,
for the app to poll/watch.

`--simulate OBSERVER` runs the same trial loop headless, driven by a
`vpsych.core.observers.SimulatedObserver` instead of a live PsychoPy window
and keyboard, so end-to-end session logic (trial loop, writer, status
updates, exit codes) is exercisable in CI without a display.

Not implemented in Phase 0 -- this module freezes the CLI contract; `main`
parses arguments and documents intended behavior but raises
`NotImplementedError` before actually running a session.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vpsych.runner.status import RunnerExitCode


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the `vpsych-run` / `python -m vpsych.runner` argument parser.

    Returns:
        A configured `argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="vpsych-run",
        description="Run one vpsych session (a subprocess launched by the app, or standalone).",
    )
    parser.add_argument(
        "--session-plan",
        type=Path,
        required=True,
        metavar="PATH",
        help="Path to a session-plan JSON file (see vpsych.data.schemas.SessionPlan).",
    )
    parser.add_argument(
        "--status-file",
        type=Path,
        required=True,
        metavar="PATH",
        help=(
            "Path to write RunnerStatus JSON snapshots to as the session progresses "
            "(see vpsych.runner.status.RunnerStatus.write_atomic)."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        metavar="PATH",
        help="Data root to write session output under (default: VPSYCH_DATA_ROOT env var, or ~/vpsych-data).",
    )
    parser.add_argument(
        "--simulate",
        type=str,
        default=None,
        metavar="OBSERVER",
        help=(
            "Run headless against a named simulated observer instead of a live PsychoPy window "
            "and keyboard (e.g. for CI end-to-end tests). Omit for a normal, real display run."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the `vpsych-run` console script.

    Args:
        argv: Command-line arguments (excluding the program name), or
            `None` to use `sys.argv[1:]`.

    Returns:
        A `RunnerExitCode` value (see `vpsych.runner.status.RunnerExitCode`)
        to use as the process exit code.
    """
    parser = build_arg_parser()
    parser.parse_args(argv)
    raise NotImplementedError(
        "vpsych.runner.__main__.main is a Phase-0 interface stub; session "
        "execution lands in a later phase. Exit codes are defined in "
        f"vpsych.runner.status.RunnerExitCode: {list(RunnerExitCode)}"
    )


if __name__ == "__main__":
    sys.exit(main())
