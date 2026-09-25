"""Tests for `vpsych.app.runner_process.build_runner_command` (no live subprocess/Qt event loop)."""

from __future__ import annotations

import sys
from pathlib import Path

from vpsych.app.runner_process import build_runner_command


def test_build_runner_command_basic() -> None:
    program, args = build_runner_command(Path("/tmp/plan.json"), Path("/tmp/status.json"))
    assert program == sys.executable
    assert args[:2] == ["-m", "vpsych.runner"]
    assert "--session-plan" in args
    assert str(Path("/tmp/plan.json")) in args
    assert "--status-file" in args
    assert str(Path("/tmp/status.json")) in args
    assert "--data-root" not in args
    assert "--simulate" not in args


def test_build_runner_command_with_data_root_and_simulate() -> None:
    _program, args = build_runner_command(
        Path("/tmp/plan.json"),
        Path("/tmp/status.json"),
        data_root=Path("/tmp/data"),
        simulate="always_correct",
    )
    assert "--data-root" in args
    assert str(Path("/tmp/data")) in args
    assert "--simulate" in args
    assert "always_correct" in args


def test_failed_to_start_reports_an_error_instead_of_hanging(tmp_path: Path) -> None:
    from PySide6.QtCore import QCoreApplication, QProcess, QTimer

    from vpsych.app.runner_process import RunnerProcessController
    from vpsych.runner.status import RunnerExitCode

    app = QCoreApplication.instance() or QCoreApplication([])
    controller = RunnerProcessController(tmp_path / "plan.json", tmp_path / "status.json")
    controller._process.setProgram(str(tmp_path / "no-such-interpreter"))
    codes: list[RunnerExitCode] = []
    controller.finished.connect(codes.append)
    QTimer.singleShot(3000, app.quit)  # safety net so a regression fails instead of hanging
    controller.finished.connect(lambda _c: app.quit())
    controller.start()
    app.exec()
    assert codes == [RunnerExitCode.ERROR]
    assert controller._process.state() == QProcess.ProcessState.NotRunning
