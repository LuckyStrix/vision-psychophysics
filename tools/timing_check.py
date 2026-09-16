#!/usr/bin/env python3
"""Standalone diagnostic: measure actual display refresh rate, frame jitter, and dropped frames.

Opens a fullscreen PsychoPy window, flips it repeatedly, and reports a
`vpsych.core.timing.FrameTimingStats` summary -- the same summary used to
quality-flag a real session. Intended to be run manually on a test machine
before running real sessions on it, not part of the automated test suite
(it requires a live display). An optional photodiode check (comparing a
flip's actual light-output timing against the OS-reported flip time) is a
Phase 4+ extension.

Not implemented in Phase 0.
"""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Command-line arguments, or `None` to use `sys.argv[1:]`.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-frames", type=int, default=300, help="Number of frames to measure.")
    parser.add_argument(
        "--drop-tolerance",
        type=float,
        default=1.5,
        help="Multiplier on nominal frame interval above which a frame counts as dropped.",
    )
    parser.parse_args(argv)
    raise NotImplementedError(
        "tools/timing_check.py is a Phase-0 stub; implemented in a later phase."
    )


if __name__ == "__main__":
    raise SystemExit(main())
