"""PySide6 desktop app entry point: the `vpsych` console script.

The app is the calm, clinical, keyboard-navigable UI described in the
project plan (home / calibration wizard / test catalog / session builder /
pre-test briefing / run / results / data). It never draws stimuli itself --
running a session hands off to `vpsych.runner` in a subprocess (see
`vpsych.runner.__main__`) so Qt's event loop never competes with
frame-locked stimulus presentation, and watches that subprocess's
`vpsych.runner.status.RunnerStatus` file for progress.

Not implemented in Phase 0 -- this module freezes the entry point; `main`
raises `NotImplementedError` until a later phase implements the actual
Qt application (see the plan's Phase 3).
"""

from __future__ import annotations


def main() -> int:
    """Entry point for the `vpsych` console script.

    Returns:
        Process exit code.
    """
    raise NotImplementedError(
        "vpsych.app.main.main is a Phase-0 interface stub; the PySide6 app "
        "lands in a later phase (see the plan's Phase 3)."
    )


if __name__ == "__main__":
    raise SystemExit(main())
