"""PySide6 desktop app entry point: the `vpsych` console script.

The app is the calm, clinical, keyboard-navigable UI described in the
project plan (home / calibration wizard / test catalog / session builder /
run / results / data). It never draws stimuli itself -- running a session
hands off to `vpsych.runner` in a subprocess (see
`vpsych.app.runner_process`) so Qt's event loop never competes with
frame-locked stimulus presentation, and watches that subprocess's
`vpsych.runner.status.RunnerStatus` file for progress.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from PySide6.QtWidgets import QApplication

from vpsych.app import theme
from vpsych.app.main_window import MainWindow
from vpsych.app.state import AppState


def main() -> int:
    """Entry point for the `vpsych` console script.

    Returns:
        Process exit code.
    """
    app = QApplication(sys.argv)
    app.setApplicationName("vpsych")
    app.setStyleSheet(theme.STYLESHEET)

    state = AppState()
    # Session-plan and runner-status files are a transient hand-off to the
    # runner subprocess, not participant data -- kept outside the data root
    # entirely so they never show up as stray files in a dataset scan.
    run_scratch_dir = Path(tempfile.gettempdir()) / "vpsych-app-runs"

    window = MainWindow(state, run_scratch_dir)
    window.resize(1100, 720)
    window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
