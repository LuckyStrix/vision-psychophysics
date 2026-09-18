"""Shared grating/letter rendering, gamma linearization, and dithering helpers.

Used only by `vpsych.tests_catalog.contrast_sensitivity_function` and
`vpsych.tests_catalog.letter_contrast_sensitivity` -- both present
contrast-defined stimuli (a sinusoidal grating and Sloan letters,
respectively) on a mid-gray background, linearized against the display's
gamma calibration and noisy-bit dithered (Allard, R., & Faubert, J. (2008).
The noisy-bit method for digital displays: converting a resolution
limitation into a pseudo-resolution. Behavior Research Methods, 40(3),
735-743) so contrasts finer than an 8-bit frame buffer's ~1/255 step are
represented correctly in expectation. Kept here (a private module directly
under `tests_catalog`, per `docs/WRITING_A_TEST.md`'s Phase 2B task) rather
than duplicated in each test package.

This module is pure numpy/pydantic-adjacent code (no `psychopy` import), so
it -- and everything in it -- is exercised directly by headless unit tests
(see `tests/tests_catalog/test_contrast_rendering.py`).

**Shared-code notes for other test implementers** (reported, not fixed,
here -- see `docs/WRITING_A_TEST.md` section on not modifying files outside
one's own package):

1. `vpsych.core.calibration.gamma` has no built-in constructor from a
   *stored* `vpsych.core.calibration.models.GammaCalibration` straight to a
   `GammaChannelModel` -- callers only ever get one back from
   `fit_gamma`/`fit_gamma_lookup` at calibration time itself.
   `gamma_channel_model_from_calibration` below fills that gap locally
   (averaging `gamma_r`/`gamma_g`/`gamma_b` when `gamma_single` is absent,
   since both tests using this module render grayscale, R=G=B, stimuli
   only). A third test needing the same conversion would probably be
   better served by promoting this to `core/calibration/gamma.py` instead
   of copying it again.
2. `vpsych.core.trial_loop.PsychoPyBackend.__init__` does not currently set
   a window gamma ramp at all (no `win.gammaRamp` / monitor gamma-grid call
   anywhere in it). `docs/WRITING_A_TEST.md` section 7 describes gamma
   linearization as happening "at window-creation time... not typically
   inside your test", but that hook does not actually exist yet. Both
   tests using this module therefore do their own full linearization in
   Python -- computing final, already-gamma-correct, already-dithered
   drive levels in `[0, 1]` -- and hand PsychoPy the result directly as a
   raw image array with no further color-space transform, the same pattern
   `tests_catalog._example` uses for its single scalar contrast value. This
   works today because an unconfigured `psychopy.visual.Window` applies no
   additional gamma correction of its own, but it means every
   gamma-dependent test must repeat this logic.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from vpsych.core.calibration.dither import dither_to_uint8
from vpsych.core.calibration.gamma import GammaChannelModel, linearize
from vpsych.core.calibration.models import GammaCalibration

FloatArray = npt.NDArray[np.float64]

# ---------------------------------------------------------------------------
# Gamma linearization
# ---------------------------------------------------------------------------


def gamma_channel_model_from_calibration(gamma_cal: GammaCalibration) -> GammaChannelModel:
    """Build a grayscale `GammaChannelModel` from a stored `GammaCalibration`.

    Both tests in this module render grayscale (R=G=B driven identically)
    stimuli, so a single combined channel model is enough: `gamma_single`
    is used directly when present (the common case for both grade A
    combined-channel photometer fits and grade B psychophysical estimates,
    see `docs/CALIBRATION.md`); otherwise the mean of whichever of
    `gamma_r`/`gamma_g`/`gamma_b` are set is used (documented
    simplification -- see this module's docstring).

    Args:
        gamma_cal: The active calibration's gamma characterization. Must
            not be `method="none"` (callers should have already checked
            `TestRequirements.needs_gamma_calibration` via
            `vpsych.tests_catalog.base.check_requirements`).

    Returns:
        A parametric `GammaChannelModel` usable with `linearize`.

    Raises:
        ValueError: If `gamma_cal.method == "none"`, or neither
            `gamma_single` nor any per-channel gamma is set.
    """
    if gamma_cal.method == "none":
        raise ValueError(
            "Cannot linearize against an uncalibrated (method='none') GammaCalibration; "
            "this test declares TestRequirements.needs_gamma_calibration=True and should "
            "never be constructed against one."
        )
    if gamma_cal.gamma_single is not None:
        gamma = gamma_cal.gamma_single
    else:
        per_channel = [
            v for v in (gamma_cal.gamma_r, gamma_cal.gamma_g, gamma_cal.gamma_b) if v is not None
        ]
        if not per_channel:
            raise ValueError(
                "GammaCalibration has neither gamma_single nor any per-channel gamma set; "
                "cannot build a GammaChannelModel from it."
            )
        gamma = float(np.mean(per_channel))
    return GammaChannelModel(
        lum_min_cdm2=gamma_cal.lum_min_cdm2, lum_max_cdm2=gamma_cal.lum_max_cdm2, gamma=gamma
    )


def mean_luminance_cdm2(model: GammaChannelModel) -> float:
    """The mid-gray background luminance both tests present stimuli against.

    The midpoint of the calibration's `[lum_min_cdm2, lum_max_cdm2]` range
    (drive fraction 0.5), i.e. the "mean luminance" the task description
    refers to.
    """
    return 0.5 * (model.lum_min_cdm2 + model.lum_max_cdm2)


def michelson_drive_from_pattern(
    pattern: FloatArray, contrast: float, model: GammaChannelModel
) -> FloatArray:
    """Map a +-1 grating pattern at a given Michelson contrast to linearized drive values.

    Michelson contrast is defined in the luminance domain around the mean
    luminance (`mean_luminance_cdm2`): target luminance is
    ``L_mean * (1 + contrast * pattern)``. This is converted to a fraction
    of the calibration's full `[Lmin, Lmax]` range and then linearized
    (`vpsych.core.calibration.gamma.linearize`) to a drive value in
    `[0, 1]` -- i.e. what should be written to the (uncorrected, see this
    module's docstring) frame buffer to produce that luminance.

    Args:
        pattern: Values in `[-1, 1]` (e.g. a windowed sinusoidal grating).
        contrast: Michelson contrast, in `[0, 1]`.
        model: The display's linearizing gamma model.

    Returns:
        Linearized drive values, same shape as `pattern`, clipped to
        `[0, 1]` (contrast/envelope combinations that would drive past the
        display's range are clipped rather than raising).
    """
    l_mean = mean_luminance_cdm2(model)
    target_cdm2 = l_mean * (1.0 + contrast * pattern)
    fraction = (target_cdm2 - model.lum_min_cdm2) / (model.lum_max_cdm2 - model.lum_min_cdm2)
    fraction = np.clip(fraction, 0.0, 1.0)
    return np.asarray(linearize(fraction, model), dtype=np.float64)


def weber_drive_from_ink_mask(
    ink_mask: FloatArray, contrast: float, model: GammaChannelModel
) -> FloatArray:
    """Map a dark-letter-on-background ink mask at a given Weber contrast to drive values.

    Background pixels (``ink_mask <= 0.5``) are held at the fixed mid-gray
    fraction 0.5 (`mean_luminance_cdm2`) regardless of `contrast`; ink
    pixels (``ink_mask > 0.5``) are darkened per the Weber contrast
    definition used by the Pelli-Robson chart and its analogues (Pelli,
    D. G., Robson, J. G., & Wilkins, A. J. (1988). The design of a new
    letter chart for measuring contrast sensitivity. Clinical Vision
    Sciences, 2(3), 187-199):

        contrast = (L_background - L_letter) / L_background

    Args:
        ink_mask: Values in `[0, 1]` (or boolean-like), where > 0.5 marks
            an ink (letter-stroke) pixel.
        contrast: Weber contrast, in `[0, 1]` (0 = invisible, 1 = black
            letter against the background luminance).
        model: The display's linearizing gamma model.

    Returns:
        Linearized drive values, same shape as `ink_mask`, clipped to
        `[0, 1]`.
    """
    l_mean = mean_luminance_cdm2(model)
    l_ink = l_mean * (1.0 - contrast)
    target_cdm2 = np.where(np.asarray(ink_mask) > 0.5, l_ink, l_mean)
    fraction = (target_cdm2 - model.lum_min_cdm2) / (model.lum_max_cdm2 - model.lum_min_cdm2)
    fraction = np.clip(fraction, 0.0, 1.0)
    return np.asarray(linearize(fraction, model), dtype=np.float64)


def dither_frame(
    drive_fraction: FloatArray | float, rng: np.random.Generator, bit_depth: int = 8
) -> FloatArray:
    """Noisy-bit dither drive value(s) in `[0, 1]` and rescale back to `[0, 1]`.

    Thin wrapper around `vpsych.core.calibration.dither.dither_to_uint8`
    that converts its `uint8` output back to a `[0, 1]` float array, ready
    to hand to PsychoPy as a raw (already gamma-correct, already quantized)
    drive value. Both tests using this module call this once per stimulus
    frame (not once per trial), refreshing the dither noise every frame so
    the *effective* contrast resolution improves with the number of frames
    averaged (`vpsych.core.calibration.dither.effective_contrast_resolution`,
    with `n_samples` = the number of stimulus frames shown).

    Args:
        drive_fraction: Desired drive value(s) in `[0, 1]`.
        rng: Seeded generator (must be `trial_ctx["rng"]`; see
            `docs/WRITING_A_TEST.md` section 5/7).
        bit_depth: Output bit depth (defaults to 8).

    Returns:
        Dithered drive value(s) in `[0, 1]`, same shape as
        `drive_fraction`.
    """
    levels = dither_to_uint8(drive_fraction, rng, bit_depth=bit_depth)
    return levels.astype(np.float64) / float(2**bit_depth - 1)


# ---------------------------------------------------------------------------
# Grating stimulus (contrast_sensitivity_function)
# ---------------------------------------------------------------------------


def sinusoidal_grating_pattern(
    size_px: int, cycles_per_px: float, orientation_deg: float, phase_rad: float
) -> FloatArray:
    """A `[-1, 1]` sinusoidal luminance pattern, `size_px` x `size_px`.

    Args:
        size_px: Side length of the (square) pattern, in pixels.
        cycles_per_px: Spatial frequency, cycles per pixel (convert from
            cycles/deg via `DisplayGeometry.px_per_deg_at_center`).
        orientation_deg: Grating orientation. `0` produces vertical stripes
            (luminance varies along the horizontal axis); positive values
            rotate clockwise.
        phase_rad: Spatial phase, in radians.

    Returns:
        A `(size_px, size_px)` array of values in `[-1, 1]`.
    """
    half = (size_px - 1) / 2.0
    y, x = np.mgrid[0:size_px, 0:size_px].astype(np.float64)
    x = x - half
    y = y - half
    theta = np.radians(orientation_deg)
    xr = x * np.cos(theta) + y * np.sin(theta)
    pattern = np.sin(2.0 * np.pi * cycles_per_px * xr + phase_rad)
    return np.asarray(pattern, dtype=np.float64)


def raised_cosine_disc_envelope(size_px: int, plateau_frac: float = 0.5) -> FloatArray:
    """A circular raised-cosine (Tukey) spatial envelope, 1 at the center, 0 at the edge.

    A standard smooth circular aperture: full contrast out to
    `plateau_frac` of the patch radius, then a half-cosine ramp down to 0
    at the edge, avoiding the sharp edges (and their spurious high spatial
    frequencies) of a hard-edged disc.

    Args:
        size_px: Side length of the (square) envelope, in pixels.
        plateau_frac: Fraction of the radius (0 at center, 1 at edge) held
            at full contrast before the cosine ramp begins.

    Returns:
        A `(size_px, size_px)` array of values in `[0, 1]`.
    """
    half = (size_px - 1) / 2.0
    y, x = np.mgrid[0:size_px, 0:size_px].astype(np.float64)
    r = np.sqrt((x - half) ** 2 + (y - half) ** 2) / half  # 0 at center, 1 at edge
    env = np.ones_like(r)
    ramp = (r > plateau_frac) & (r <= 1.0)
    env[ramp] = 0.5 * (1.0 + np.cos(np.pi * (r[ramp] - plateau_frac) / (1.0 - plateau_frac)))
    env[r > 1.0] = 0.0
    return np.asarray(env, dtype=np.float64)


def raised_cosine_temporal_envelope(n_frames: int, ramp_frames: int) -> FloatArray:
    """Per-frame contrast multiplier: raised-cosine on/off ramps with a plateau.

    Args:
        n_frames: Total number of stimulus frames.
        ramp_frames: Number of frames for each of the onset/offset ramps
            (clamped to at most `n_frames // 2` so the ramps never overlap).

    Returns:
        A length-`n_frames` array of multipliers in `[0, 1]`: a half-cosine
        ramp from 0 to 1, a plateau at 1, then a half-cosine ramp from 1
        back to 0.
    """
    if n_frames < 1:
        raise ValueError(f"n_frames must be >= 1, got {n_frames}")
    ramp_frames = max(0, min(ramp_frames, n_frames // 2))
    env = np.ones(n_frames, dtype=np.float64)
    if ramp_frames > 0:
        t = np.linspace(0.0, np.pi, ramp_frames)
        up = 0.5 * (1.0 - np.cos(t))
        env[:ramp_frames] = up
        env[n_frames - ramp_frames :] = up[::-1]
    return env


# ---------------------------------------------------------------------------
# Sloan letter stimulus (letter_contrast_sensitivity)
# ---------------------------------------------------------------------------

#: The 10 Sloan letters (Sloan, L. L. (1959). New test charts for the
#: measurement of visual acuity at far and near distances. American Journal
#: of Ophthalmology, 48(6), 807-813), as standardized by the NAS-NRC (1980)
#: Committee on Vision report "Recommended standard procedures for the
#: clinical measurement and specification of visual acuity", and used by
#: the Pelli-Robson chart (Pelli, Robson & Wilkins 1988, see
#: `weber_drive_from_ink_mask`'s docstring).
SLOAN_LETTERS: tuple[str, ...] = ("C", "D", "H", "K", "N", "O", "R", "S", "V", "Z")

#: Coarse 5x5-grid stroke-based approximations of the 10 Sloan letterforms.
#: Authentic Sloan letters are designed on a 5x5 unit grid with stroke
#: width 1/5 of the letter height, but are not simple block/pixel shapes;
#: this project does not bundle Denis Pelli's Sloan font (no verified
#: OFL/permissive license was located for it -- see the module docstring
#: and `docs/methods/letter_contrast_sensitivity.md`), so these grids are a
#: **documented, coarse approximation** of the real letterforms at 5x5
#: resolution, not a faithful reproduction, and should not be treated as
#: clinically equivalent to a real Sloan/Pelli-Robson chart. Each string is
#: one row (top to bottom), '1' = ink, '0' = background.
_SLOAN_5X5_GRIDS: dict[str, tuple[str, str, str, str, str]] = {
    "C": ("01111", "10000", "10000", "10000", "01111"),
    "D": ("11110", "10001", "10001", "10001", "11110"),
    "H": ("10001", "10001", "11111", "10001", "10001"),
    "K": ("10001", "10010", "11100", "10010", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001"),
    "O": ("01110", "10001", "10001", "10001", "01110"),
    "R": ("11110", "10001", "11110", "10100", "10010"),
    "S": ("01111", "10000", "01110", "00001", "11110"),
    "V": ("10001", "10001", "10001", "01010", "00100"),
    "Z": ("11111", "00010", "00100", "01000", "11111"),
}


def render_letter_ink_mask(letter: str, size_px: int) -> FloatArray:
    """Rasterize one Sloan letter's 5x5 stroke grid to a `size_px` x `size_px` ink mask.

    Each of the 5x5 grid's cells becomes a `size_px // 5`-pixel block
    (nearest-neighbour upsampling, matching the "stroke = 1/5 of letter
    height" block-letter convention -- see `_SLOAN_5X5_GRIDS`'s docstring);
    the result is zero-padded and centered if `size_px` is not an exact
    multiple of 5.

    Args:
        letter: One of `SLOAN_LETTERS` (case-insensitive).
        size_px: Side length of the (square) letter bounding box, in
            pixels.

    Returns:
        A `(size_px, size_px)` array of `1.0` (ink) / `0.0` (background).

    Raises:
        KeyError: If `letter` is not one of `SLOAN_LETTERS`.
    """
    grid = _SLOAN_5X5_GRIDS[letter.upper()]
    bits = np.array([[float(c) for c in row] for row in grid], dtype=np.float64)
    reps = max(1, size_px // 5)
    block = np.kron(bits, np.ones((reps, reps), dtype=np.float64))
    out = np.zeros((size_px, size_px), dtype=np.float64)
    b = block.shape[0]
    if b >= size_px:
        offset = (b - size_px) // 2
        out[:, :] = block[offset : offset + size_px, offset : offset + size_px]
    else:
        offset = (size_px - b) // 2
        out[offset : offset + b, offset : offset + b] = block
    return out
