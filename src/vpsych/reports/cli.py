"""Command-line interface for rendering session reports.

Entry point: `vpsych-report` console script.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vpsych.reports.session_report import render_session_report


def main() -> int:
    """CLI entry point for vpsych-report command.

    Usage:
        vpsych-report <session_dir> [--output <output.html>]

    Args:
        session_dir: Path to a session directory (sub-XXXX/ses-YYYYMMDDTHHMMSS/).
        --output: Optional path to write the HTML report to (default: session_dir/report.html).

    Returns:
        0 on success, 1 on error.
    """
    parser = argparse.ArgumentParser(
        description="Render an HTML session report from a vpsych session directory."
    )
    parser.add_argument("session_dir", help="Path to the session directory to report on.")
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Path to write the HTML report to (default: <session_dir>/report.html).",
    )

    args = parser.parse_args()
    session_dir = Path(args.session_dir)

    if not session_dir.exists():
        print(f"Error: session directory not found: {session_dir}", file=sys.stderr)
        return 1

    if not session_dir.is_dir():
        print(f"Error: not a directory: {session_dir}", file=sys.stderr)
        return 1

    try:
        out_path = render_session_report(session_dir, args.output)
        print(f"Report written to: {out_path}")
        return 0
    except Exception as e:
        print(f"Error rendering report: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
