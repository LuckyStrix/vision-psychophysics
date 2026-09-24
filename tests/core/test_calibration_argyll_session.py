"""`ArgyllSpotreadSession` (pty transport) against a fake ``spotread`` script.

The fake replays the transcript captured from a real ColorMunki Photo
(ArgyllCMS 2.3.1, via the sibling calsuite project), including its two prompts
that have no trailing newline, and reads one key at a time from the pty like
the real program. It is a stand-in for the transport, not for the instrument:
the real binary is never run.
"""

from __future__ import annotations

import os
import stat
import sys
import textwrap
from pathlib import Path

import pytest

from vpsych.core.calibration import argyll
from vpsych.core.calibration.argyll import (
    ArgyllCalibrationRequiredError,
    ArgyllNotAvailableError,
    ArgyllReadError,
    ArgyllSpotreadSession,
)

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="needs a POSIX pty")

_FAKE = textwrap.dedent(
    r"""
    import os, sys, time
    try:
        import tty
        if sys.stdin.isatty():
            tty.setcbreak(0)
    except ImportError:
        pass

    def say(s):
        sys.stdout.write(s)
        sys.stdout.flush()

    def key():
        return os.read(0, 1)

    open(os.environ["FAKE_PID_FILE"], "w").write(str(os.getpid()))
    mode = os.environ.get("FAKE_MODE", "ok")
    wrong_tries = int(os.environ.get("FAKE_DIAL_WRONG_TRIES", "0"))
    READY = (
        "\nPlace instrument on spot to be measured,\n"
        "and hit [A-Z] to read white and setup FWA compensation (keyed to letter)\n"
        "Hit ESC or Q to exit, instrument switch or any other key to take a reading: "
    )
    CAL = (
        "\nSpot read needs a calibration before continuing\n"
        "\nSet instrument sensor to calibration position,\n"
        " and then hit any key to continue,\n"
        " or hit Esc or Q to abort: "
    )

    if mode == "no_instrument":
        say("Diagnostic: Unknown, inappropriate or no instrument detected\n")
        sys.exit(1)
    if mode == "hang":
        time.sleep(60)

    say(CAL)
    wrong = 0
    while True:
        if key() in (b"q", b"Q", b""):
            sys.exit(0)
        if wrong >= wrong_tries:
            break
        wrong += 1
        say(CAL)
    say("\nCalibration complete\n")
    say(READY)
    n = 0
    while True:
        if key() in (b"q", b"Q", b"\x1b", b""):
            sys.exit(0)
        n += 1
        say(f"\n Result is XYZ: {n}.000000 {2 * n}.000000 {3 * n}.000000, D50 Lab: 1.0 2.0 3.0\n")
        if mode == "recal_after_result" and n == 2:
            say(CAL)
            key()  # the calibration confirmation, not a reading
            say("\nCalibration complete\n")
        say(READY)
    """
)


@pytest.fixture
def fake_spotread(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    script = tmp_path / "spotread"
    script.write_text(f"#!{sys.executable}\n{_FAKE}")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    pid_file = tmp_path / "pid"
    monkeypatch.setattr(argyll, "SPOTREAD_BIN", str(script))
    monkeypatch.setenv("FAKE_PID_FILE", str(pid_file))
    return pid_file


def _alive(pid_file: Path) -> bool:
    try:
        os.kill(int(pid_file.read_text()), 0)
    except ProcessLookupError:
        return False
    return True


def test_calibrate_then_read_over_a_real_pty(fake_spotread: Path) -> None:
    session = ArgyllSpotreadSession(read_timeout_s=10.0)
    try:
        with pytest.raises(ArgyllCalibrationRequiredError) as exc_info:
            session.open()
        assert "calibration position" in exc_info.value.prompt_text
        session.confirm_calibration()
        assert [session.read_patch().Y for _ in range(3)] == pytest.approx([2.0, 4.0, 6.0])
    finally:
        session.close()
    assert not _alive(fake_spotread)


def test_wrong_dial_position_prompts_again_over_a_real_pty(
    fake_spotread: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_DIAL_WRONG_TRIES", "1")
    session = ArgyllSpotreadSession(read_timeout_s=10.0)
    try:
        with pytest.raises(ArgyllCalibrationRequiredError):
            session.open()
        with pytest.raises(ArgyllCalibrationRequiredError):
            session.confirm_calibration()
        session.confirm_calibration()
        assert pytest.approx(2.0) == session.read_patch().Y
    finally:
        session.close()


def test_recalibration_after_a_reading_is_raised_by_the_next_read(
    fake_spotread: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_MODE", "recal_after_result")
    session = ArgyllSpotreadSession(read_timeout_s=10.0)
    try:
        with pytest.raises(ArgyllCalibrationRequiredError):
            session.open()
        session.confirm_calibration()
        assert pytest.approx(2.0) == session.read_patch().Y
        assert pytest.approx(4.0) == session.read_patch().Y  # returned, not lost
        with pytest.raises(ArgyllCalibrationRequiredError):
            session.read_patch()
        session.confirm_calibration()
        assert pytest.approx(6.0) == session.read_patch().Y
    finally:
        session.close()


def test_failed_open_closes_the_process_and_pty(
    fake_spotread: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_MODE", "no_instrument")
    session = ArgyllSpotreadSession(read_timeout_s=10.0)
    with pytest.raises(ArgyllNotAvailableError):
        session.open()
    assert session._proc is None
    assert session._master_fd is None
    assert not _alive(fake_spotread)


def test_open_timeout_closes_the_process_and_pty(
    fake_spotread: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_MODE", "hang")
    session = ArgyllSpotreadSession(read_timeout_s=1.0)
    with pytest.raises(ArgyllReadError, match="Timed out"):
        session.open()
    assert session._proc is None
    assert not _alive(fake_spotread)


def test_open_twice_is_refused(fake_spotread: Path) -> None:
    session = ArgyllSpotreadSession(read_timeout_s=10.0)
    try:
        with pytest.raises(ArgyllCalibrationRequiredError):
            session.open()
        with pytest.raises(RuntimeError, match="already open"):
            session.open()
    finally:
        session.close()
