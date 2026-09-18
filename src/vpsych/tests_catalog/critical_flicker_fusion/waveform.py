"""Pure-numpy frame-sampled flicker waveform generation and the intensity transform.

A frame-based display cannot present an arbitrary continuous sinusoid: it
can only hold one luminance value for the whole duration of each refresh
frame. This module generates the actual per-frame luminance sequence a
flickering disc will show (honestly, including the effect of that sampling)
and provides tools to measure how much a given frequency's fundamental
component survives the sampling (`dft_fundamental_amplitude`), rather than
assuming the nominal modulation depth is what the display actually produces.
See `docs/methods/critical_flicker_fusion.md` for the full derivation and
citations.

No `psychopy` import here (pure `numpy`/`math`), per the project's
headless-testability rule; `critical_flicker_fusion/__init__.py`
gamma-linearizes these luminance sequences into drive levels using
`vpsych.core.calibration.gamma` before handing them to PsychoPy.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

WaveformMode = str  # "continuous" | "square_wave" -- see module docstring.


def raised_cosine_envelope(n_frames: int, ramp_frames: int) -> FloatArray:
    """Build a `[0, 1]` amplitude envelope with raised-cosine on/off ramps.

    Args:
        n_frames: Total number of frames in the presentation.
        ramp_frames: Length of each ramp (onset and offset), in frames.
            Clamped to at most `n_frames // 2` so the ramps never overlap.

    Returns:
        A length-`n_frames` array: `1.0` during the plateau, rising/falling
        as `0.5 * (1 - cos(pi * t / ramp_frames))` during the ramps, `1.0`
        everywhere if `ramp_frames <= 0`.
    """
    if n_frames <= 0:
        raise ValueError(f"n_frames must be positive, got {n_frames}")
    envelope = np.ones(n_frames, dtype=np.float64)
    ramp = max(0, min(ramp_frames, n_frames // 2))
    if ramp == 0:
        return envelope
    # `ramp` interior points strictly between 0 (exclusive, the frame before
    # the stimulus) and 1 (exclusive, the first full-amplitude frame), so the
    # plateau's leading edge is not itself flattened into the ramp.
    t = np.linspace(0.0, np.pi, ramp + 2)[1:-1]
    ramp_curve = 0.5 * (1.0 - np.cos(t))
    envelope[:ramp] = ramp_curve
    envelope[n_frames - ramp :] = ramp_curve[::-1]
    return envelope


def frame_luminance_sequence(
    mean_luminance_cdm2: float,
    modulation_depth: float,
    frequency_hz: float,
    refresh_hz: float,
    n_frames: int,
    phase_rad: float = 0.0,
    onset_ramp_frames: int = 0,
) -> FloatArray:
    """Generate the actual per-frame luminance sequence for a sinusoidally flickered disc.

    Each frame `n` (0-indexed) holds the constant luminance

        L_n = Lmean * (1 + m * envelope(n) * sin(2*pi*f*n/R + phi))

    where `R` is the *measured* refresh rate (`refresh_hz`) and `envelope`
    is a raised-cosine on/off ramp (`raised_cosine_envelope`) applied to
    suppress onset/offset transients (see the "Requirements"/"Stimulus"
    section of `docs/methods/critical_flicker_fusion.md` for why an abrupt
    step is itself a detectable, frequency-independent cue at CFF-relevant
    frequencies -- Tyler & Hamer 1990). Frequencies approaching `refresh_hz
    / 2` are severely under-sampled by this frame-based representation; see
    `dft_fundamental_amplitude` to measure how much of `modulation_depth`
    actually survives for a given `(frequency_hz, refresh_hz, n_frames)`.

    Args:
        mean_luminance_cdm2: Mean luminance, in cd/m^2 (Michelson modulation
            is defined around this mean).
        modulation_depth: Nominal Michelson modulation depth `m`, in
            `[0, 1]` (`1.0` = 100% modulation, full excursion between 0 and
            `2 * mean_luminance_cdm2`).
        frequency_hz: Nominal flicker frequency, in Hz.
        refresh_hz: Display refresh rate, in Hz (should be the *measured*
            rate, `display.refresh_hz`, not just the OS-nominal value).
        n_frames: Number of frames to generate.
        phase_rad: Starting phase, in radians.
        onset_ramp_frames: Length of the raised-cosine onset/offset ramp
            applied to the modulation envelope, in frames (`0` = abrupt
            onset, no ramp).

    Returns:
        A length-`n_frames` array of luminance values, in cd/m^2.

    Raises:
        ValueError: If `mean_luminance_cdm2 <= 0`, `modulation_depth` is
            outside `[0, 1]`, `frequency_hz <= 0`, `refresh_hz <= 0`, or
            `n_frames <= 0`.
    """
    if mean_luminance_cdm2 <= 0:
        raise ValueError(f"mean_luminance_cdm2 must be positive, got {mean_luminance_cdm2}")
    if not (0.0 <= modulation_depth <= 1.0):
        raise ValueError(f"modulation_depth must be in [0, 1], got {modulation_depth}")
    if frequency_hz <= 0:
        raise ValueError(f"frequency_hz must be positive, got {frequency_hz}")
    if refresh_hz <= 0:
        raise ValueError(f"refresh_hz must be positive, got {refresh_hz}")
    if n_frames <= 0:
        raise ValueError(f"n_frames must be positive, got {n_frames}")

    n = np.arange(n_frames, dtype=np.float64)
    envelope = raised_cosine_envelope(n_frames, onset_ramp_frames)
    modulation = (
        modulation_depth
        * envelope
        * np.sin(2.0 * np.pi * frequency_hz * n / refresh_hz + phase_rad)
    )
    result: FloatArray = mean_luminance_cdm2 * (1.0 + modulation)
    return result


def steady_luminance_sequence(mean_luminance_cdm2: float, n_frames: int) -> FloatArray:
    """The non-flickering comparison disc: a constant luminance for every frame.

    Args:
        mean_luminance_cdm2: Luminance, in cd/m^2, matched to the flickering
            disc's mean (see `docs/methods/critical_flicker_fusion.md`
            "Requirements": mismatched mean luminance between the two discs
            is a luminance cue independent of flicker).
        n_frames: Number of frames.

    Returns:
        A length-`n_frames` array, every entry equal to `mean_luminance_cdm2`.
    """
    if n_frames <= 0:
        raise ValueError(f"n_frames must be positive, got {n_frames}")
    return np.full(n_frames, float(mean_luminance_cdm2), dtype=np.float64)


def square_wave_frame_luminance_sequence(
    mean_luminance_cdm2: float,
    modulation_depth: float,
    frequency_hz: float,
    refresh_hz: float,
    n_frames: int,
    phase_frames: int = 0,
) -> FloatArray:
    """Generate an exact (alias-free) square-wave flicker sequence.

    Only well-defined when `frequency_hz` divides the refresh rate exactly
    as `refresh_hz / (2 * k)` for a positive integer `k` (see
    `usable_square_wave_frequencies`): each half-cycle is then exactly `k`
    frames long, so every frame is either the high or low luminance level
    with no partial-frame averaging -- unlike the continuous-sinusoid mode,
    this representation has *zero* sampling error for its fundamental
    (verified in `test_critical_flicker_fusion_waveform.py` via
    `dft_fundamental_amplitude`).

    Args:
        mean_luminance_cdm2: Mean luminance, in cd/m^2.
        modulation_depth: Michelson modulation depth `m`, in `[0, 1]`.
        frequency_hz: Flicker frequency, in Hz; must satisfy
            `valid_square_wave_frequency(frequency_hz, refresh_hz)`.
        refresh_hz: Display refresh rate, in Hz.
        n_frames: Number of frames to generate.
        phase_frames: Starting phase, in whole frames (square-wave phase is
            necessarily quantized to frame boundaries).

    Returns:
        A length-`n_frames` array of luminance values, in cd/m^2, each
        exactly `mean_luminance_cdm2 * (1 +/- modulation_depth)`.

    Raises:
        ValueError: If `frequency_hz` is not a valid `refresh_hz / (2k)`
            square-wave frequency, or any other argument is out of range
            (see `frame_luminance_sequence`).
    """
    if mean_luminance_cdm2 <= 0:
        raise ValueError(f"mean_luminance_cdm2 must be positive, got {mean_luminance_cdm2}")
    if not (0.0 <= modulation_depth <= 1.0):
        raise ValueError(f"modulation_depth must be in [0, 1], got {modulation_depth}")
    if n_frames <= 0:
        raise ValueError(f"n_frames must be positive, got {n_frames}")
    if not valid_square_wave_frequency(frequency_hz, refresh_hz):
        raise ValueError(
            f"frequency_hz={frequency_hz} is not a valid square-wave frequency for "
            f"refresh_hz={refresh_hz} (must equal refresh_hz / (2 * k) for integer k)."
        )
    half_period_frames = round(refresh_hz / (2.0 * frequency_hz))
    n = np.arange(n_frames) + phase_frames
    high = (n // half_period_frames) % 2 == 0
    return np.where(
        high,
        mean_luminance_cdm2 * (1.0 + modulation_depth),
        mean_luminance_cdm2 * (1.0 - modulation_depth),
    )


def valid_square_wave_frequency(frequency_hz: float, refresh_hz: float, tol: float = 1e-6) -> bool:
    """Check whether `frequency_hz` is exactly representable as `refresh_hz / (2 * k)`.

    Args:
        frequency_hz: Candidate frequency, in Hz.
        refresh_hz: Display refresh rate, in Hz.
        tol: Relative tolerance for the integer-`k` check.

    Returns:
        `True` if `refresh_hz / (2 * frequency_hz)` is within `tol` of a
        positive integer.
    """
    if frequency_hz <= 0 or refresh_hz <= 0:
        return False
    k = refresh_hz / (2.0 * frequency_hz)
    return bool(k >= 1.0 - tol and abs(k - round(k)) <= tol * max(1.0, k))


def usable_square_wave_frequencies(refresh_hz: float, min_hz: float, max_hz: float) -> list[float]:
    """Enumerate every valid square-wave frequency `refresh_hz / (2k)` in `[min_hz, max_hz]`.

    Args:
        refresh_hz: Display refresh rate, in Hz.
        min_hz: Lowest frequency to include, in Hz.
        max_hz: Highest frequency to include, in Hz.

    Returns:
        Frequencies in ascending order.
    """
    if refresh_hz <= 0:
        raise ValueError(f"refresh_hz must be positive, got {refresh_hz}")
    k_min = max(1, math.floor(refresh_hz / (2.0 * max_hz)))
    k_max = max(1, math.ceil(refresh_hz / (2.0 * min_hz)))
    freqs = []
    for k in range(k_min, k_max + 1):
        f = refresh_hz / (2.0 * k)
        if min_hz - 1e-9 <= f <= max_hz + 1e-9:
            freqs.append(f)
    return sorted({round(f, 9) for f in freqs})


def dft_fundamental_amplitude(
    luminances: FloatArray, frequency_hz: float, refresh_hz: float
) -> float:
    """Measure the actual amplitude of a luminance sequence's component at `frequency_hz`.

    Evaluates the discrete-time Fourier transform of the (mean-removed)
    sequence directly at `frequency_hz` (a Goertzel-style single-frequency
    correlation, not restricted to FFT bin frequencies -- valid whether or
    not `frequency_hz * n_frames / refresh_hz` is an integer):

        A(f) = (2 / N) * | sum_n (L_n - mean(L)) * exp(-2*pi*i*f*n/R) |

    This is the honest, "what did the display actually show" complement to
    `frame_luminance_sequence`'s *nominal* modulation depth: near
    `refresh_hz / 2`, frame-sampling and any onset/offset envelope
    attenuate the delivered fundamental well below the nominal
    `modulation_depth * mean_luminance_cdm2` -- exactly the aliasing/
    sampling-artifact honesty the task requires (see
    `docs/methods/critical_flicker_fusion.md`).

    Args:
        luminances: The frame luminance sequence, in cd/m^2.
        frequency_hz: Frequency to evaluate the fundamental at, in Hz.
        refresh_hz: Display refresh rate, in Hz.

    Returns:
        The measured amplitude, in cd/m^2 (comparable to
        `modulation_depth * mean_luminance_cdm2`, the nominal amplitude).
    """
    n = np.arange(len(luminances), dtype=np.float64)
    phase = 2.0 * np.pi * frequency_hz * n / refresh_hz
    centered = luminances - np.mean(luminances)
    coeff = np.sum(centered * np.exp(-1j * phase))
    return float(2.0 * np.abs(coeff) / len(luminances))


def frequency_to_intensity(frequency_hz: float) -> float:
    """Map a flicker frequency to the QUEST+ intensity dimension `x = -log10(f)`.

    QUEST+ (and every adaptive procedure in this project) assumes
    probability-correct is *increasing* in intensity, but CFF performance
    *decreases* as frequency increases -- higher frequencies are harder to
    detect as flickering. `x = -log10(frequency_hz)` restores the assumed
    monotonic direction: `x` increases as `f` decreases (easier), so
    `p(x)` is increasing in `x` exactly as QUEST+ requires. See
    `docs/methods/critical_flicker_fusion.md` for the full justification
    and `intensity_to_frequency` for the inverse.

    Args:
        frequency_hz: Flicker frequency, in Hz (must be positive).

    Returns:
        The transformed intensity, `-log10(frequency_hz)`.

    Raises:
        ValueError: If `frequency_hz <= 0`.
    """
    if frequency_hz <= 0:
        raise ValueError(f"frequency_hz must be positive, got {frequency_hz}")
    return -math.log10(frequency_hz)


def intensity_to_frequency(intensity: float) -> float:
    """Invert `frequency_to_intensity`: recover a frequency from `x = -log10(f)`.

    Args:
        intensity: The QUEST+ intensity value, `-log10(frequency_hz)`.

    Returns:
        The corresponding frequency, in Hz: `10 ** (-intensity)`.
    """
    return float(10.0 ** (-intensity))
