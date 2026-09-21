"""Unit tests for vpsych.core.calibration.argyll.

No test in this file shells out to the real `spotread` binary. Line
sequences are either genuine captured output (see `_REAL_NO_INSTRUMENT_OUTPUT`,
captured from a real `spotread -e` run against ArgyllCMS 2.3.1 with no
ColorMunki attached) or synthetic-but-representative fixtures for the
prompt/result lines that need real hardware to observe (see module
docstrings in `argyll.py` for the caveat on those).
"""

from __future__ import annotations

import subprocess

import pytest

from vpsych.core.calibration.argyll import (
    ArgyllCalibrationRequiredError,
    ArgyllNotAvailableError,
    ArgyllReadError,
    ArgyllSpotreadPhotometer,
    SpotReading,
    SpotreadProtocol,
    classify_line,
    detect_instrument,
    parse_result_line,
    reading_to_primary_chromaticity,
    spotread_available,
)
from vpsych.core.calibration.photometer import measure_gamma, measure_gamma_per_channel

# Captured verbatim (stdout+stderr) from a real `spotread -e` invocation
# against a real ArgyllCMS 2.3.1 install with no ColorMunki plugged in.
# spotread errors out before reaching any interactive prompt, so this is the
# one part of a real transcript this build could actually observe.
_REAL_NO_INSTRUMENT_OUTPUT = """\
icoms_set_ser_port: tcgetattr on '/dev/ttyS0' failed with 3 (Input/output error)
icoms_set_ser_port: tcgetattr on '/dev/ttyS0' failed with 3 (Input/output error)
Measure spot values, Version 2.3.1
Author: Graeme W. Gill, licensed under the GPL Version 2 or later
Diagnostic: Unknown, inappropriate or no instrument detected
usage: spotread [-options] [logfile]
 -v                   Verbose mode
"""

# Representative (not device-captured; see argyll.py's module docstring)
# prompt/result lines, modelled on ArgyllCMS's documented spotread
# interaction for emissive spot reads.
_MEASURE_PROMPT_LINE = (
    "Place instrument on spot to be measured, and hit any key to read it,\nor Esc, Q or ^C to exit"
)
_CALIBRATION_PROMPT_LINE = (
    "Set instrument sensor to calibration position, then hit any key to continue,\n"
    "or Esc, Q or ^C to abandon"
)
_RESULT_LINE_LAB = (
    "Result is XYZ: 41.24 21.26 1.93, D50 Lab: 53.24 80.09 67.20, CCT = 1913K (Delta E 22.7)"
)
_RESULT_LINE_YXY = "Result is XYZ: 95.05 100.00 108.91, Yxy: 100.00 0.3127 0.3290"


# ---------------------------------------------------------------------------
# classify_line / parse_result_line / SpotReading
# ---------------------------------------------------------------------------


def test_classify_line_recognizes_real_captured_no_instrument_output() -> None:
    lines = _REAL_NO_INSTRUMENT_OUTPUT.splitlines()
    classes = [classify_line(line) for line in lines]
    assert "no_instrument" in classes


def test_classify_line_recognizes_calibration_prompt() -> None:
    for line in _CALIBRATION_PROMPT_LINE.splitlines():
        if "calibrat" in line.lower():
            assert classify_line(line) == "calibration_prompt"


def test_classify_line_recognizes_measure_prompt() -> None:
    assert classify_line(_MEASURE_PROMPT_LINE.splitlines()[0]) == "measure_prompt"


def test_classify_line_recognizes_result_over_other_categories() -> None:
    assert classify_line(_RESULT_LINE_LAB) == "result"
    assert classify_line(_RESULT_LINE_YXY) == "result"


def test_classify_line_other_for_unrelated_text() -> None:
    assert classify_line("Measure spot values, Version 2.3.1") == "other"


def test_parse_result_line_lab_format() -> None:
    reading = parse_result_line(_RESULT_LINE_LAB)
    assert pytest.approx(41.24) == reading.X
    assert pytest.approx(21.26) == reading.Y
    assert pytest.approx(1.93) == reading.Z
    assert reading.luminance_cdm2 == pytest.approx(21.26)


def test_parse_result_line_yxy_format_still_parses_leading_xyz() -> None:
    """XYZ is always printed first regardless of the -x/-h/-u secondary format flag."""
    reading = parse_result_line(_RESULT_LINE_YXY)
    assert pytest.approx(95.05) == reading.X
    assert pytest.approx(100.00) == reading.Y
    assert pytest.approx(108.91) == reading.Z


def test_parse_result_line_rejects_unrecognized_text() -> None:
    with pytest.raises(ArgyllReadError):
        parse_result_line("not a result line at all")


def test_spot_reading_xy_matches_near_d65_white() -> None:
    reading = SpotReading(X=95.05, Y=100.00, Z=108.91)
    x, y = reading.xy
    assert x == pytest.approx(0.3127, abs=1e-3)
    assert y == pytest.approx(0.3290, abs=1e-3)


def test_spot_reading_xy_degenerate_all_zero_is_safe() -> None:
    assert SpotReading(X=0.0, Y=0.0, Z=0.0).xy == (0.0, 0.0)


def test_reading_to_primary_chromaticity_clamps_negative_luminance_noise() -> None:
    reading = SpotReading(X=0.01, Y=-0.001, Z=0.02)
    primary = reading_to_primary_chromaticity(reading)
    assert primary.Y_cdm2 == 0.0


# ---------------------------------------------------------------------------
# spotread_available / detect_instrument
# ---------------------------------------------------------------------------


def test_spotread_available_reflects_which(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _name: None)
    assert spotread_available() is False
    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/spotread")
    assert spotread_available() is True


def test_detect_instrument_missing_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _name: None)
    result = detect_instrument()
    assert result.available is False
    assert "not found" in result.message.lower()


def test_detect_instrument_no_instrument_from_real_captured_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/spotread")

    def fake_runner(_args: object, _timeout: float) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=["spotread"], returncode=1, stdout=_REAL_NO_INSTRUMENT_OUTPUT, stderr=""
        )

    result = detect_instrument(runner=fake_runner)
    assert result.available is False
    assert "no instrument" in result.message.lower()


def test_detect_instrument_timeout_suggests_instrument_attached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/spotread")

    def fake_runner(_args: object, timeout: float) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd="spotread", timeout=timeout)

    result = detect_instrument(runner=fake_runner)
    assert result.available is True


# ---------------------------------------------------------------------------
# SpotreadProtocol: the fully unit-tested interaction state machine
# ---------------------------------------------------------------------------


class _FakeTransport:
    """Feeds a canned line sequence to `SpotreadProtocol` and records sent keys."""

    def __init__(self, lines: list[str]) -> None:
        self._lines = iter(lines)
        self.sent_keys: list[str] = []

    def next_line(self) -> str | None:
        return next(self._lines, None)

    def send_key(self, ch: str) -> None:
        self.sent_keys.append(ch)


def test_protocol_wait_for_ready_skips_banner_lines() -> None:
    transport = _FakeTransport(
        ["Measure spot values, Version 2.3.1", "Author: ...", _MEASURE_PROMPT_LINE]
    )
    protocol = SpotreadProtocol(transport.next_line, transport.send_key)
    protocol.wait_for_ready()  # must not raise


def test_protocol_wait_for_ready_raises_on_no_instrument() -> None:
    transport = _FakeTransport(_REAL_NO_INSTRUMENT_OUTPUT.splitlines())
    protocol = SpotreadProtocol(transport.next_line, transport.send_key)
    with pytest.raises(ArgyllNotAvailableError):
        protocol.wait_for_ready()


def test_protocol_wait_for_ready_raises_calibration_required_with_prompt_text() -> None:
    transport = _FakeTransport([_CALIBRATION_PROMPT_LINE, _MEASURE_PROMPT_LINE])
    protocol = SpotreadProtocol(transport.next_line, transport.send_key)
    with pytest.raises(ArgyllCalibrationRequiredError) as exc_info:
        protocol.wait_for_ready()
    assert "calibrat" in exc_info.value.prompt_text.lower()


def test_protocol_confirm_calibration_sends_key_and_reaches_ready() -> None:
    transport = _FakeTransport([_CALIBRATION_PROMPT_LINE, _MEASURE_PROMPT_LINE])
    protocol = SpotreadProtocol(transport.next_line, transport.send_key)
    with pytest.raises(ArgyllCalibrationRequiredError):
        protocol.wait_for_ready()
    protocol.confirm_calibration()
    assert transport.sent_keys == [" "]
    # Now ready: a further read_patch() call should not re-block on calibration.
    protocol.wait_for_ready()


def test_protocol_read_patch_happy_path() -> None:
    transport = _FakeTransport([_MEASURE_PROMPT_LINE, _RESULT_LINE_LAB])
    protocol = SpotreadProtocol(transport.next_line, transport.send_key)
    reading = protocol.read_patch()
    assert pytest.approx(21.26) == reading.Y
    assert transport.sent_keys == [" "]


def test_protocol_read_patch_sequence_of_readings() -> None:
    result_lines = [
        "Result is XYZ: 0.40 0.40 0.50, D50 Lab: 6.9 0.1 -0.5, CCT = 6500K (Delta E 0.1)",
        "Result is XYZ: 20.0 21.0 25.0, D50 Lab: 52.9 -1.0 -1.0, CCT = 6500K (Delta E 0.1)",
        "Result is XYZ: 130.0 140.0 150.0, D50 Lab: 91.0 -1.0 0.5, CCT = 6500K (Delta E 0.1)",
    ]
    transport = _FakeTransport([_MEASURE_PROMPT_LINE, *result_lines])
    protocol = SpotreadProtocol(transport.next_line, transport.send_key)
    readings = [protocol.read_patch() for _ in result_lines]
    assert [r.Y for r in readings] == pytest.approx([0.40, 21.0, 140.0])
    assert transport.sent_keys == [" ", " ", " "]


def test_protocol_read_patch_raises_no_instrument_mid_session() -> None:
    transport = _FakeTransport([_MEASURE_PROMPT_LINE, "Diagnostic: no instrument detected"])
    protocol = SpotreadProtocol(transport.next_line, transport.send_key)
    with pytest.raises(ArgyllNotAvailableError):
        protocol.read_patch()


def test_protocol_raises_read_error_when_process_output_ends() -> None:
    transport = _FakeTransport([_MEASURE_PROMPT_LINE])
    protocol = SpotreadProtocol(transport.next_line, transport.send_key)
    with pytest.raises(ArgyllReadError):
        protocol.read_patch()


def test_protocol_raises_read_error_when_line_budget_exhausted() -> None:
    transport = _FakeTransport(["noise"] * 5)
    protocol = SpotreadProtocol(transport.next_line, transport.send_key, max_lines_per_prompt=3)
    with pytest.raises(ArgyllReadError):
        protocol.wait_for_ready()


def test_protocol_quit_never_raises_even_with_broken_sink() -> None:
    def broken_send(_ch: str) -> None:
        raise OSError("pty already closed")

    protocol = SpotreadProtocol(lambda: None, broken_send)
    protocol.quit()  # must not raise


# ---------------------------------------------------------------------------
# ArgyllSpotreadPhotometer + integration with core/calibration/photometer.py
# ---------------------------------------------------------------------------


class _FakeSession:
    """Duck-types `ArgyllSpotreadSession`'s `read_patch()` surface for the adapter."""

    def __init__(self, readings: list[SpotReading]) -> None:
        self._readings = iter(readings)

    def read_patch(self) -> SpotReading:
        return next(self._readings)


def test_argyll_photometer_measure_luminance_delegates_to_session() -> None:
    photometer = ArgyllSpotreadPhotometer(_FakeSession([SpotReading(X=10, Y=12.5, Z=8)]))
    assert photometer.measure_luminance_cdm2("gray", 0.5) == pytest.approx(12.5)


def test_argyll_photometer_measure_xyz_returns_full_reading() -> None:
    photometer = ArgyllSpotreadPhotometer(_FakeSession([SpotReading(X=10, Y=12.5, Z=8)]))
    reading = photometer.measure_xyz("r", 1.0)
    assert (reading.X, reading.Y, reading.Z) == (10, 12.5, 8)


def _synthetic_readings_for_gamma(
    gamma: float, lum_min: float, lum_max: float, n_levels: int, white_xy: tuple[float, float]
) -> list[SpotReading]:
    import numpy as np

    x_w, y_w = white_xy
    readings = []
    for v in np.linspace(0.0, 1.0, n_levels):
        lum = lum_min + (lum_max - lum_min) * (v**gamma)
        readings.append(SpotReading(X=lum * (x_w / y_w), Y=lum, Z=lum * ((1 - x_w - y_w) / y_w)))
    return readings


def test_argyll_photometer_end_to_end_through_measure_gamma() -> None:
    """A fully synthetic but plausible session (fixed near-D65 chromaticity across the
    ramp) driven through `SpotreadProtocol`, feeding `measure_gamma` end to end -- proving
    the argyll parsing layer composes correctly with the existing photometer orchestration
    without reinventing its measurement-sequence logic.
    """
    readings = _synthetic_readings_for_gamma(
        gamma=2.4, lum_min=0.5, lum_max=150.0, n_levels=13, white_xy=(0.3127, 0.3290)
    )
    lines = [_MEASURE_PROMPT_LINE]
    for r in readings:
        lines.append(f"Result is XYZ: {r.X:.4f} {r.Y:.4f} {r.Z:.4f}, D50 Lab: 0 0 0")
    transport = _FakeTransport(lines)
    protocol = SpotreadProtocol(transport.next_line, transport.send_key)

    class _ProtocolSession:
        def read_patch(self) -> SpotReading:
            return protocol.read_patch()

    photometer = ArgyllSpotreadPhotometer(_ProtocolSession())
    cal = measure_gamma(photometer, n_levels=13, channel="gray")
    assert cal.method == "photometer"
    assert cal.grade == "A"
    assert cal.gamma_single == pytest.approx(2.4, rel=1e-3)
    assert cal.lum_min_cdm2 == pytest.approx(0.5, rel=1e-3)
    assert cal.lum_max_cdm2 == pytest.approx(150.0, rel=1e-3)


def test_argyll_photometer_end_to_end_per_channel() -> None:
    """Same idea as above, but through `measure_gamma_per_channel` (3 independent sweeps)."""
    gammas = {"r": 2.1, "g": 2.3, "b": 2.5}
    lines = []
    for channel in ("r", "g", "b"):
        lines.append(_MEASURE_PROMPT_LINE if channel == "r" else "")
        readings = _synthetic_readings_for_gamma(
            gamma=gammas[channel],
            lum_min=0.3,
            lum_max=140.0,
            n_levels=9,
            white_xy=(0.3127, 0.3290),
        )
        for r in readings:
            lines.append(f"Result is XYZ: {r.X:.4f} {r.Y:.4f} {r.Z:.4f}, D50 Lab: 0 0 0")
    lines = [line for line in lines if line]
    transport = _FakeTransport(lines)
    protocol = SpotreadProtocol(transport.next_line, transport.send_key)

    class _ProtocolSession:
        def read_patch(self) -> SpotReading:
            return protocol.read_patch()

    photometer = ArgyllSpotreadPhotometer(_ProtocolSession())
    cal = measure_gamma_per_channel(photometer, n_levels=9)
    assert cal.gamma_r == pytest.approx(2.1, rel=1e-3)
    assert cal.gamma_g == pytest.approx(2.3, rel=1e-3)
    assert cal.gamma_b == pytest.approx(2.5, rel=1e-3)
