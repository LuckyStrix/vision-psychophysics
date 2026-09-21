"""Argyll CMS ``spotread`` photometer backend: emissive spot readings via subprocess.

Drives ``/usr/bin/spotread`` (ArgyllCMS, tested against 2.3.1) in **emissive
spot mode** (``-e``) to read a displayed patch as CIE XYZ, satisfying the
:class:`vpsych.core.calibration.photometer.Photometer` protocol for grade-A
luminance calibration (:mod:`vpsych.core.calibration.photometer`) and also
exposing full XYZ for grade-A measured color primaries
(:mod:`vpsych.core.calibration.color`).

**Two supported instruments**, both driven by the same ``spotread`` binary
(ArgyllCMS auto-detects which is plugged in) but with different flags:

- **X-Rite ColorMunki Photo** (a spectrometer): measures an actual light
  spectrum, so no display-technology correction is needed --
  ``instrument="spectrometer"`` omits ``-y``. It has a sliding calibration
  cover and needs an explicit calibration step (cover closed, instrument
  off the screen) before its first emissive reading in a session; this
  module surfaces that as :class:`ArgyllCalibrationRequiredError` rather
  than silently blocking or guessing when it will happen.
- **X-Rite ColorMunki Display** (a colorimeter): measures via filtered
  photodiodes and needs a display-technology hint to select the right
  correction table -- ``instrument="colorimeter"`` passes ``-y l`` (LCD;
  pass ``display_technology="c"`` for a CRT) so its filter response is
  corrected for the panel type actually being measured.

**Why a pseudo-terminal.** ``spotread`` is an interactive console program:
its "place the instrument and hit any key" prompts read single raw
keystrokes from its *controlling terminal*, not from buffered stdin. Piping
plain ``subprocess.PIPE`` stdin to it does not present as a terminal to
Argyll's console-input layer, so :class:`ArgyllSpotreadSession` allocates a
pseudo-terminal (:mod:`pty`) and attaches the subprocess to its slave side,
exactly as running it from an interactive shell would.

**Why the interaction logic is split from the I/O.** :class:`SpotreadProtocol`
implements the entire read-a-patch / handle-calibration control flow as a
pure state machine over two injected callables (``next_line``, ``send_key``)
and contains no subprocess/pty code itself, so it is exercised in
``tests/core/test_calibration_argyll.py`` with canned line sequences (some
of them genuine ``spotread`` output captured with the ColorMunki *not*
plugged in) -- the test suite never shells out to ``spotread``, per project
policy. :class:`ArgyllSpotreadSession` is the thin, unavoidably
hardware/subprocess-dependent transport that feeds real lines into that same
state machine; see docs/CALIBRATION.md for the live-hardware verification
this class still needs before it can be trusted end to end.
"""

from __future__ import annotations

import contextlib
import os
import re
import select
import shutil
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from vpsych.core.calibration.models import PrimaryChromaticity

#: The ``spotread`` executable name (resolved via ``PATH``).
SPOTREAD_BIN = "spotread"

InstrumentKind = Literal["spectrometer", "colorimeter"]
DisplayTechnology = Literal["l", "c"]
LineClass = Literal["no_instrument", "calibration_prompt", "measure_prompt", "result", "other"]


class ArgyllNotAvailableError(RuntimeError):
    """``spotread`` is missing from ``PATH``, or no supported instrument is attached."""


class ArgyllCalibrationRequiredError(RuntimeError):
    """``spotread`` is waiting at an instrument-calibration prompt.

    Raised (never silently retried) so the caller -- the calibration
    wizard's photometer step -- can show the participant/experimenter the
    real instruction (``prompt_text``, spotread's own wording, not a
    hardcoded guess at it) and only proceed once they confirm the
    instrument has been positioned as asked, via
    :meth:`ArgyllSpotreadSession.confirm_calibration`.

    Attributes:
        prompt_text: The calibration-prompt line spotread printed, verbatim.
    """

    def __init__(self, prompt_text: str) -> None:
        super().__init__("spotread is waiting for an instrument calibration step: " + prompt_text)
        self.prompt_text = prompt_text


class ArgyllReadError(RuntimeError):
    """A spotread session produced output that could not be parsed as a reading."""


# --- Textual output parsing -------------------------------------------------
#
# spotread's primary result line is stable across its `-x`/`-h`/`-u`/plain
# (Lab) output-format flags -- all of them print "Result is XYZ: X Y Z, ..."
# first and only vary the secondary (Lab/Yxy/LCh/Yuv) representation after
# it -- so this module never needs those flags: it parses XYZ directly and
# derives xy chromaticity itself (see `SpotReading.xy`).
_RESULT_RE = re.compile(r"XYZ:\s*(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)")

# Captured verbatim from a real `spotread -e` run (ArgyllCMS 2.3.1) with no
# instrument attached: "Diagnostic: Unknown, inappropriate or no instrument
# detected". Matched case-insensitively and loosely (just "no instrument
# detected") so small wording variants across Argyll versions still match.
_NO_INSTRUMENT_RE = re.compile(r"no instrument detected", re.IGNORECASE)

# Wording for the calibration/measurement prompts is not independently
# verifiable without the physical device (see module docstring), so these
# are deliberately loose, keyword-based patterns rather than an exact
# string match -- robust to the small wording differences between
# instrument types/Argyll point releases, at the cost of only being as
# precise as the keywords "calibrat[e/ion]" and "place instrument"/"spot to
# be measured" actually appearing.
_CALIBRATION_RE = re.compile(r"calibrat", re.IGNORECASE)
_MEASURE_PROMPT_RE = re.compile(
    r"place instrument|spot to be measured|hit.{0,20}key.{0,20}(read|measur)", re.IGNORECASE
)


@dataclass(frozen=True)
class SpotReading:
    """One parsed ``spotread`` emissive-mode measurement: CIE XYZ tristimulus values.

    Attributes:
        X: CIE X tristimulus value.
        Y: CIE Y tristimulus value -- in emissive (``-e``) mode this is an
            absolute luminance in cd/m^2.
        Z: CIE Z tristimulus value.
    """

    X: float
    Y: float
    Z: float

    @property
    def luminance_cdm2(self) -> float:
        """``Y``, interpreted as absolute luminance in cd/m^2 (emissive mode)."""
        return self.Y

    @property
    def xy(self) -> tuple[float, float]:
        """CIE 1931 ``(x, y)`` chromaticity derived from ``X``/``Y``/``Z``.

        Returns ``(0.0, 0.0)`` for a (degenerate) all-zero reading rather
        than raising, since a true black reading is a legitimate
        measurement outcome, just not a meaningful chromaticity.
        """
        total = self.X + self.Y + self.Z
        if total <= 0:
            return (0.0, 0.0)
        return (self.X / total, self.Y / total)


def parse_result_line(line: str) -> SpotReading:
    """Parse one ``spotread`` ``"Result is XYZ: ..."`` line into a :class:`SpotReading`.

    Args:
        line: One line of spotread's stdout.

    Returns:
        The parsed reading.

    Raises:
        ArgyllReadError: If ``line`` does not contain a recognizable
            ``XYZ: <x> <y> <z>`` triple.
    """
    match = _RESULT_RE.search(line)
    if match is None:
        raise ArgyllReadError(f"Could not parse a spotread result line: {line!r}")
    x_str, y_str, z_str = match.groups()
    return SpotReading(X=float(x_str), Y=float(y_str), Z=float(z_str))


def classify_line(line: str) -> LineClass:
    """Classify one line of ``spotread``'s interactive textual output.

    Order matters: a "no instrument" diagnostic and a result line both take
    priority over the (looser) calibration/measurement-prompt keyword
    matches, since e.g. a results logfile header could otherwise coincide
    with prompt wording.

    Args:
        line: One line of spotread's stdout/stderr.

    Returns:
        ``"no_instrument"``, ``"result"``, ``"calibration_prompt"``,
        ``"measure_prompt"``, or ``"other"``.
    """
    if _NO_INSTRUMENT_RE.search(line):
        return "no_instrument"
    if _RESULT_RE.search(line):
        return "result"
    if _CALIBRATION_RE.search(line):
        return "calibration_prompt"
    if _MEASURE_PROMPT_RE.search(line):
        return "measure_prompt"
    return "other"


def reading_to_primary_chromaticity(reading: SpotReading) -> PrimaryChromaticity:
    """Convert a raw :class:`SpotReading` into a :class:`PrimaryChromaticity`.

    Used by the calibration wizard's photometer-driven color step to turn a
    measured R/G/B/white patch reading into the field
    :class:`~vpsych.core.calibration.models.ColorCalibration` expects.

    Args:
        reading: The measured patch reading.

    Returns:
        A `PrimaryChromaticity` with `Y_cdm2` clamped to be non-negative
        (measurement noise near black can otherwise read slightly negative).
    """
    x, y = reading.xy
    return PrimaryChromaticity(x=x, y=max(y, 1e-6), Y_cdm2=max(reading.Y, 0.0))


def spotread_available() -> bool:
    """Whether the ``spotread`` executable can be found on ``PATH``."""
    return shutil.which(SPOTREAD_BIN) is not None


@dataclass(frozen=True)
class InstrumentDetectionResult:
    """Result of a best-effort ``spotread`` instrument probe (see :func:`detect_instrument`).

    Attributes:
        available: Whether a photometer looks usable.
        message: Human-readable detail, always populated, suitable for
            showing directly in the wizard UI.
    """

    available: bool
    message: str


def _run_probe(args: Sequence[str], timeout_s: float) -> subprocess.CompletedProcess[str]:
    """Run ``args`` with empty/closed stdin and capture output as text (real subprocess)."""
    return subprocess.run(list(args), input="", capture_output=True, text=True, timeout=timeout_s)


def detect_instrument(
    timeout_s: float = 5.0,
    runner: Callable[[Sequence[str], float], subprocess.CompletedProcess[str]] | None = None,
) -> InstrumentDetectionResult:
    """Best-effort check for whether ``spotread`` exists and an instrument is attached.

    With no instrument attached, ``spotread -e`` (given closed/empty stdin)
    reliably fails fast with a "no instrument detected" diagnostic before it
    ever reaches an interactive prompt -- confirmed against a real run of
    ArgyllCMS 2.3.1 with no ColorMunki plugged in (see
    ``tests/core/test_calibration_argyll.py`` for the captured fixture).
    With an instrument attached, ``spotread`` instead blocks at its first
    interactive prompt waiting for a keystroke; since this probe gives it
    closed stdin, that shows up as either a timeout (treated here as "likely
    available") or process exit without the "no instrument" diagnostic. This
    distinction should be re-verified once the ColorMunki is physically
    available (see docs/CALIBRATION.md); it has not been exercised against
    real hardware in this build.

    Args:
        timeout_s: How long to wait for the probe process before assuming
            it is blocked at a prompt (and therefore an instrument is
            attached).
        runner: Injected process runner, for testing; defaults to a real
            `subprocess.run` call. Never invoked by the automated test
            suite with the default (real) runner.

    Returns:
        The detection result.
    """
    if not spotread_available():
        return InstrumentDetectionResult(
            False,
            "ArgyllCMS 'spotread' was not found on PATH. Install ArgyllCMS "
            "(see docs/CALIBRATION.md) to use a photometer for grade-A calibration.",
        )
    run = runner or _run_probe
    try:
        proc = run([SPOTREAD_BIN, "-e"], timeout_s)
    except subprocess.TimeoutExpired:
        return InstrumentDetectionResult(
            True,
            "spotread started and is waiting for input rather than exiting immediately -- "
            "an instrument appears to be attached.",
        )
    output = (proc.stdout or "") + (proc.stderr or "")
    if _NO_INSTRUMENT_RE.search(output):
        return InstrumentDetectionResult(
            False,
            "spotread ran but found no instrument. Check the ColorMunki is plugged in via "
            "USB and, on Linux, that udev permissions are set up (see docs/CALIBRATION.md).",
        )
    return InstrumentDetectionResult(
        True, "spotread ran without reporting 'no instrument detected'."
    )


class SpotreadProtocol:
    """Pure ``spotread`` interaction state machine: no subprocess/pty I/O of its own.

    Implements the read-a-patch and handle-calibration control flow by
    reading lines from an injected ``next_line`` callable and writing
    keystrokes via an injected ``send_key`` callable. This indirection is
    what makes the whole interaction logic unit-testable
    (``tests/core/test_calibration_argyll.py``) with canned line sequences
    and no real ``spotread`` process -- only :class:`ArgyllSpotreadSession`
    (below) supplies the real, untested-by-necessity pty-backed versions of
    these two callables.
    """

    def __init__(
        self,
        next_line: Callable[[], str | None],
        send_key: Callable[[str], None],
        max_lines_per_prompt: int = 200,
    ) -> None:
        """Args:
        next_line: Returns the next line of spotread's output (without a
            trailing newline), or `None` if the process has exited/closed
            its output.
        send_key: Sends one keystroke to spotread's (pseudo-)terminal.
        max_lines_per_prompt: Safety bound on how many lines to read while
            waiting for a recognized prompt/result, so a genuinely stuck
            or misbehaving process raises rather than looping forever.
        """
        self._next_line = next_line
        self._send_key = send_key
        self._max_lines = max_lines_per_prompt
        self._ready_to_measure = False

    def _pump_until(self, target: LineClass) -> str:
        for _ in range(self._max_lines):
            line = self._next_line()
            if line is None:
                raise ArgyllReadError(
                    "spotread exited/closed its output before producing a recognizable line."
                )
            cls = classify_line(line)
            if cls == "no_instrument":
                raise ArgyllNotAvailableError(
                    f"No supported photometer/colorimeter was detected by spotread "
                    f"(spotread said: {line.strip()!r})."
                )
            if cls == "calibration_prompt":
                raise ArgyllCalibrationRequiredError(line.strip())
            if cls == target:
                return line
        raise ArgyllReadError(
            f"spotread did not reach a {target!r} prompt within {self._max_lines} lines."
        )

    def wait_for_ready(self) -> None:
        """Consume startup output up to the first measurement prompt.

        Raises:
            ArgyllCalibrationRequiredError: If a calibration prompt is
                reached first; call :meth:`confirm_calibration` once the
                instrument has been positioned as instructed.
            ArgyllNotAvailableError: If spotread reports no instrument.
        """
        if self._ready_to_measure:
            return
        self._pump_until("measure_prompt")
        self._ready_to_measure = True

    def confirm_calibration(self) -> None:
        """Send a keystroke to complete a pending calibration step.

        Call only after `wait_for_ready()` or `read_patch()` raised
        `ArgyllCalibrationRequiredError` and the instrument has been
        physically positioned as that error's `prompt_text` instructed.

        Raises:
            ArgyllCalibrationRequiredError: If another calibration prompt
                follows (e.g. a multi-step calibration sequence) -- call
                this method again once positioned as newly instructed.
            ArgyllNotAvailableError: If spotread reports no instrument.
        """
        self._send_key(" ")
        self._pump_until("measure_prompt")
        self._ready_to_measure = True

    def read_patch(self) -> SpotReading:
        """Trigger one measurement and return the parsed reading.

        Raises:
            ArgyllCalibrationRequiredError: If a calibration prompt is
                encountered (first reading of a session that needs one).
            ArgyllNotAvailableError: If spotread reports no instrument.
            ArgyllReadError: If spotread's output cannot be parsed as a
                reading within the line budget.
        """
        self.wait_for_ready()
        self._send_key(" ")
        line = self._pump_until("result")
        return parse_result_line(line)

    def quit(self) -> None:
        """Best-effort: send spotread's quit key. Never raises."""
        with contextlib.suppress(Exception):
            self._send_key("Q")


class ArgyllSpotreadSession:
    """A real, pty-backed ``spotread`` session driving :class:`SpotreadProtocol`.

    Not exercised by the automated test suite -- it needs the real
    ``spotread`` binary attached to a real pseudo-terminal, and (for a full
    read) real hardware. See the module docstring for why a pty is used
    instead of plain pipes, and :class:`SpotreadProtocol` for the
    interaction logic this class delegates to (which *is* fully unit
    tested).
    """

    def __init__(
        self,
        instrument: InstrumentKind = "spectrometer",
        display_technology: DisplayTechnology | None = "l",
        extra_args: Sequence[str] = (),
        read_timeout_s: float = 30.0,
    ) -> None:
        """Args:
        instrument: `"spectrometer"` (ColorMunki Photo -- no `-y` flag,
            since it measures a real spectrum and needs no per-technology
            correction) or `"colorimeter"` (ColorMunki Display -- passes
            `-y <display_technology>` so its filtered-photodiode response
            is corrected for the panel technology being measured).
        display_technology: `"l"` (LCD, the default -- covers virtually
            all modern flat panels including OLED, which Argyll has no
            separate profile for) or `"c"` (CRT). Ignored for
            `instrument="spectrometer"`.
        extra_args: Additional raw `spotread` arguments, appended after the
            standard ones (e.g. a logfile path for provenance/debugging).
        read_timeout_s: Seconds to wait for each line of output before
            raising `ArgyllReadError`.
        """
        self.instrument = instrument
        self.display_technology = display_technology
        self.extra_args = tuple(extra_args)
        self.read_timeout_s = read_timeout_s
        self._proc: subprocess.Popen[bytes] | None = None
        self._master_fd: int | None = None
        self._buffer = ""
        self._protocol: SpotreadProtocol | None = None

    def command_line(self) -> list[str]:
        """The exact ``spotread`` argv this session launches (for logging/debugging)."""
        args = [SPOTREAD_BIN, "-e"]
        if self.instrument == "colorimeter" and self.display_technology is not None:
            args += ["-y", self.display_technology]
        args += list(self.extra_args)
        return args

    def open(self) -> None:
        """Start ``spotread`` attached to a fresh pseudo-terminal and wait for readiness.

        Raises:
            ArgyllNotAvailableError: If `spotread` is not on `PATH`, or
                reports no instrument.
            ArgyllCalibrationRequiredError: If the instrument needs a
                calibration step before its first reading; call
                `confirm_calibration()` once positioned as instructed.
        """
        if not spotread_available():
            raise ArgyllNotAvailableError(
                "ArgyllCMS 'spotread' was not found on PATH. Install ArgyllCMS "
                "(see docs/CALIBRATION.md)."
            )
        import pty

        master_fd, slave_fd = pty.openpty()
        try:
            self._proc = subprocess.Popen(
                self.command_line(),
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=subprocess.STDOUT,
                close_fds=True,
            )
        finally:
            os.close(slave_fd)
        self._master_fd = master_fd
        self._protocol = SpotreadProtocol(self._next_line, self._send_key)
        self._protocol.wait_for_ready()

    def confirm_calibration(self) -> None:
        """See `SpotreadProtocol.confirm_calibration`."""
        assert self._protocol is not None, "open() must be called before confirm_calibration()"
        self._protocol.confirm_calibration()

    def read_patch(self) -> SpotReading:
        """See `SpotreadProtocol.read_patch`."""
        assert self._protocol is not None, "open() must be called before read_patch()"
        return self._protocol.read_patch()

    def close(self) -> None:
        """Quit spotread and release the subprocess/pty. Safe to call more than once."""
        if self._protocol is not None:
            self._protocol.quit()
        if self._proc is not None:
            with contextlib.suppress(Exception):
                self._proc.terminate()
                self._proc.wait(timeout=5.0)
        if self._master_fd is not None:
            with contextlib.suppress(OSError):
                os.close(self._master_fd)
        self._proc = None
        self._master_fd = None
        self._protocol = None

    def __enter__(self) -> ArgyllSpotreadSession:
        self.open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _next_line(self) -> str | None:
        assert self._master_fd is not None
        deadline = time.monotonic() + self.read_timeout_s
        while "\n" not in self._buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ArgyllReadError("Timed out waiting for spotread output.")
            ready, _, _ = select.select([self._master_fd], [], [], min(remaining, 1.0))
            if not ready:
                continue
            try:
                chunk = os.read(self._master_fd, 4096)
            except OSError:
                # The pty slave closed (process exited): surface whatever's left.
                return self._buffer or None
            if not chunk:
                return self._buffer or None
            self._buffer += chunk.decode("utf-8", errors="replace")
        line, self._buffer = self._buffer.split("\n", 1)
        return line

    def _send_key(self, ch: str) -> None:
        assert self._master_fd is not None
        os.write(self._master_fd, ch.encode())


class ArgyllSpotreadPhotometer:
    """Adapts an `ArgyllSpotreadSession` to the `Photometer` protocol, plus XYZ readings.

    Satisfies `vpsych.core.calibration.photometer.Photometer` (so it drops
    directly into `measure_gamma`/`measure_gamma_per_channel`), and also
    exposes `measure_xyz` for full CIE XYZ, used by the calibration
    wizard's photometer-driven color step to measure R/G/B/white primaries
    (grade A `ColorCalibration`; see `reading_to_primary_chromaticity`).
    """

    def __init__(self, session: ArgyllSpotreadSession) -> None:
        self._session = session

    @property
    def session(self) -> ArgyllSpotreadSession:
        """The underlying session (e.g. to call `confirm_calibration()` or `close()`)."""
        return self._session

    def measure_luminance_cdm2(self, channel: str, level: float) -> float:
        """See `Photometer.measure_luminance_cdm2`. Caller must have already set the display."""
        del channel, level
        return self._session.read_patch().Y

    def measure_xyz(self, channel: str, level: float) -> SpotReading:
        """Like `measure_luminance_cdm2`, but returns the full XYZ reading."""
        del channel, level
        return self._session.read_patch()


def open_argyll_photometer(
    instrument: InstrumentKind = "spectrometer",
    display_technology: DisplayTechnology | None = "l",
) -> ArgyllSpotreadPhotometer:
    """Open a live Argyll ``spotread``-backed photometer.

    Args:
        instrument: `"spectrometer"` (ColorMunki Photo) or `"colorimeter"`
            (ColorMunki Display). See `ArgyllSpotreadSession`.
        display_technology: `"l"` (LCD, default) or `"c"` (CRT); ignored
            for a spectrometer.

    Returns:
        A ready-to-read `ArgyllSpotreadPhotometer` (its session is already
        open and past any calibration step that didn't require a prompt).

    Raises:
        ArgyllNotAvailableError: If `spotread` is missing or no instrument
            is attached.
        ArgyllCalibrationRequiredError: If the instrument needs a
            calibration step before its first reading (typically
            ColorMunki Photo). Catch this, show `prompt_text` to the user,
            have them position the instrument as instructed, and call
            `photometer.session.confirm_calibration()` before reading.
    """
    session = ArgyllSpotreadSession(instrument=instrument, display_technology=display_technology)
    session.open()
    return ArgyllSpotreadPhotometer(session)
