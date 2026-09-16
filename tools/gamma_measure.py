#!/usr/bin/env python3
"""Standalone tool: measure display luminance/gamma with a supported photometer.

Steps the display through a series of gray levels, reads luminance from a
PsychoPy-supported photometer at each level, fits a gamma curve, and writes
a `vpsych.core.calibration.models.GammaCalibration` (method="photometer",
grade A) to be embedded in a new `Calibration`. Requires a live display and
a connected photometer, so it is not part of the automated (headless) test
suite.

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
    parser.add_argument("--n-levels", type=int, default=17, help="Number of gray levels to sample.")
    parser.add_argument(
        "--output", type=str, default=None, help="Path to write the fitted GammaCalibration JSON."
    )
    parser.parse_args(argv)
    raise NotImplementedError(
        "tools/gamma_measure.py is a Phase-0 stub; implemented in a later phase."
    )


if __name__ == "__main__":
    raise SystemExit(main())
