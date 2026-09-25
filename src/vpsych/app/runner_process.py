"""Launches `vpsych-run` as a subprocess and polls its status file.

The Qt process never renders stimuli itself: `RunnerProcessController` starts
`vpsych.runner.__main__` (the same code the `vpsych-run` console script
runs) via `QProcess` -- fully asynchronous, so the UI thread is never
blocked waiting on the subprocess -- and polls the small `RunnerStatus` JSON
file on a `QTimer` (see `vpsych.app.viewmodels.run.POLL_INTERVAL_MS`), which
reads a few hundred bytes from local disk per tick, cheap enough not to
matter on the UI thread in practice.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QTimer, Signal

from vpsych.app.viewmodels.run import POLL_INTERVAL_MS, read_status_safely
from vpsych.runner.status import RunnerExitCode, RunnerStatus


def build_runner_command(
    session_plan_path: Path,
    status_file_path: Path,
    data_root: Path | None = None,
    simulate: str | None = None,
) -> tuple[str, list[str]]:
    """Build the `(program, arguments)` pair to launch the runner subprocess with.

    Uses `sys.executable -m vpsych.runner` rather than relying on the
    `vpsych-run` console script being on `PATH` (robust to dev/editable
    installs and test environments); it is the identical entry point
    (`vpsych.runner.__main__.main`).

    Args:
        session_plan_path: Path to the session-plan JSON file.
        status_file_path: Path the runner should write status snapshots to.
        data_root: Data root to pass via `--data-root`, or `None` to let
            the runner use its own default (`VPSYCH_DATA_ROOT`/`~/vpsych-data`).
        simulate: If given, passed as `--simulate` (only ever used by tests/
            demos, never a real participant session).

    Returns:
        `(program, arguments)`, suitable for `QProcess.start`.
    """
    args = [
        "-m",
        "vpsych.runner",
        "--session-plan",
        str(session_plan_path),
        "--status-file",
        str(status_file_path),
    ]
    if data_root is not None:
        args += ["--data-root", str(data_root)]
    if simulate is not None:
        args += ["--simulate", simulate]
    return sys.executable, args


class RunnerProcessController(QObject):
    """Owns the runner subprocess and its status-file polling timer.

    Signals:
        status_updated: Emitted with the latest `RunnerStatus` every time a
            newer one is read from the status file.
        finished: Emitted once, with the process's `RunnerExitCode` (or
            `RunnerExitCode.ERROR` if the process could not even start),
            after the subprocess has exited and one final status read has
            been attempted.
    """

    status_updated = Signal(object)
    finished = Signal(object)

    def __init__(
        self,
        session_plan_path: Path,
        status_file_path: Path,
        data_root: Path | None = None,
        simulate: str | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._status_file_path = status_file_path
        self._last_status: RunnerStatus | None = None
        self._process = QProcess(self)
        program, args = build_runner_command(
            session_plan_path, status_file_path, data_root, simulate
        )
        self._process.setProgram(program)
        self._process.setArguments(args)
        self._process.finished.connect(self._on_process_finished)
        self._process.errorOccurred.connect(self._on_process_error)

        self._timer = QTimer(self)
        self._timer.setInterval(POLL_INTERVAL_MS)
        self._timer.timeout.connect(self._poll_status)

    def start(self) -> None:
        """Start the runner subprocess and begin polling its status file."""
        self._timer.start()
        self._process.start()

    def abort(self) -> None:
        """Ask the runner subprocess to terminate (SIGTERM, then SIGKILL if needed)."""
        if self._process.state() != QProcess.ProcessState.NotRunning:
            self._process.terminate()
            # Escalate to SIGKILL without blocking the UI thread if it ignores SIGTERM.
            QTimer.singleShot(2000, self._kill_if_running)

    def _kill_if_running(self) -> None:
        if self._process.state() != QProcess.ProcessState.NotRunning:
            self._process.kill()

    @property
    def last_status(self) -> RunnerStatus | None:
        """The most recently read `RunnerStatus`, or `None` if none has been read yet."""
        return self._last_status

    def _poll_status(self) -> None:
        try:
            status = read_status_safely(self._status_file_path)
        except Exception:
            # A torn read should not happen given atomic writes; skip this
            # tick and retry next poll rather than crashing the UI thread.
            return
        if status is None:
            return
        if self._last_status is None or status.updated_utc != self._last_status.updated_utc:
            self._last_status = status
            self.status_updated.emit(status)

    def _on_process_error(self, error: QProcess.ProcessError) -> None:
        del error  # detail not needed: _on_process_finished always fires too/instead

    def _on_process_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        self._timer.stop()
        self._poll_status()  # one last read in case a final status was just written
        if exit_status == QProcess.ExitStatus.CrashExit:
            code = RunnerExitCode.ERROR
        else:
            try:
                code = RunnerExitCode(exit_code)
            except ValueError:
                code = RunnerExitCode.ERROR
        self.finished.emit(code)
