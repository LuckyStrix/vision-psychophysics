#!/usr/bin/env python3
"""Standalone diagnostic: measure actual display refresh rate, frame jitter, and dropped frames.

Opens a fullscreen PsychoPy window, flips it repeatedly, and reports a
`vpsych.core.timing.FrameTimingStats` summary -- the same summary used to
quality-flag a real session. Intended to be run manually on a test machine
before running real sessions on it, not part of the automated test suite
(it requires a live display). Run it after the monitor has warmed up at
least 15 minutes (see docs/CALIBRATION.md) and with the OS compositor/
vsync settings from docs/SETUP_LINUX.md applied, since both materially
affect the numbers this tool reports.

Usage::

    uv run python tools/timing_check.py --n-frames 300

An optional photodiode check (comparing a flip's actual light-output timing,
measured by a photodiode taped to the screen and read by a DAQ/oscilloscope,
against the OS-reported flip time) is the most direct way to validate that
`win.flip()` timestamps really correspond to light hitting the screen. It is
not implemented here (it needs photodiode-specific hardware this project
does not otherwise depend on); `--photodiode` is accepted and documented as
a placeholder so scripts/docs can refer to a stable flag name, but currently
only prints a note that manual photodiode verification is recommended for
grade-A timing claims, per a Phase 4+ extension.
"""

from __future__ import annotations

import argparse
import sys

from vpsych.core.timing import summarize_frame_intervals


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the `timing_check.py` argument parser.

    Returns:
        A configured `argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-frames", type=int, default=300, help="Number of frames to measure.")
    parser.add_argument(
        "--drop-tolerance",
        type=float,
        default=1.5,
        help="Multiplier on nominal frame interval above which a frame counts as dropped.",
    )
    parser.add_argument(
        "--fullscreen",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Open the measurement window fullscreen (default) or windowed (--no-fullscreen).",
    )
    parser.add_argument(
        "--photodiode",
        action="store_true",
        help=(
            "Documented placeholder only (see module docstring): this tool does not drive "
            "photodiode hardware. Passing this flag just prints a reminder to verify flip "
            "timing with a photodiode manually before relying on grade-A timing claims."
        ),
    )
    return parser


def _run_measurement(n_frames: int, fullscreen: bool) -> tuple[list[float], float]:
    """Open a window, flip `n_frames` times, and return (intervals_s, measured_refresh_hz).

    Imported lazily by `main` (never at module import time), per the
    project's rule that `psychopy` is imported lazily -- this function
    requires a live display and is not exercised by the automated test
    suite.
    """
    from psychopy import visual

    from vpsych.core.timing import measure_refresh

    win = visual.Window(fullscr=fullscreen, units="pix", waitBlanking=True, allowGUI=False)
    win.recordFrameIntervals = True
    try:
        measured_refresh_hz = measure_refresh(win, n_frames=n_frames)

        # A second, plain flip loop so the reported intervals reflect ordinary
        # (non-measurement-harness) flip behavior over the requested count.
        win.frameIntervals = []
        for _ in range(n_frames):
            win.flip()
        intervals_s = list(win.frameIntervals)
    finally:
        win.close()

    return intervals_s, measured_refresh_hz


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Command-line arguments, or `None` to use `sys.argv[1:]`.

    Returns:
        Process exit code (0 on success).
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.photodiode:
        print(
            "--photodiode is a documented placeholder (see tools/timing_check.py's module "
            "docstring): this tool does not itself drive photodiode hardware. For grade-A "
            "confidence in flip timing, tape a photodiode to the screen and confirm its "
            "measured light-onset timing matches win.flip()'s reported timestamps using an "
            "oscilloscope or DAQ, independent of this script.\n"
        )

    print(f"Measuring {args.n_frames} frames (fullscreen={args.fullscreen})...")
    intervals_s, measured_refresh_hz = _run_measurement(args.n_frames, args.fullscreen)

    stats = summarize_frame_intervals(
        intervals_s, refresh_hz=measured_refresh_hz, drop_tolerance=args.drop_tolerance
    )

    print()
    print(f"Measured refresh rate:   {measured_refresh_hz:.3f} Hz")
    print(f"Nominal frame interval:  {stats.expected_ms:.4f} ms")
    print(f"Mean frame interval:     {stats.mean_ms:.4f} ms")
    print(f"Frame interval jitter:   {stats.sd_ms:.4f} ms (sample SD)")
    print(
        f"Dropped frames:          {stats.n_dropped} / {stats.n_frames} "
        f"({stats.dropped_fraction * 100:.2f}%), drop_tolerance={args.drop_tolerance}x nominal"
    )
    print()
    if stats.dropped_fraction > 0.01:
        print(
            "WARNING: dropped-frame fraction exceeds 1%. See docs/SETUP_LINUX.md for "
            "compositor/vsync troubleshooting before running real sessions on this machine."
        )
    else:
        print("Dropped-frame fraction is within the 1% quality-flag threshold.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
