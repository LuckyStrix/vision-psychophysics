"""Per-disc color rendering pipeline and gamut math for the trivector task.

Pipeline for one disc: a target chromaticity ``(u', v')`` (CIE 1976 UCS) plus
a luminance ``Y`` (cd/m^2, drawn per-disc from the noise range) is converted
CIE-1931 ``xyY`` -> ``XYZ`` -> linear RGB (via the calibrated, or
sRGB-assumed, primaries -- :func:`vpsych.core.calibration.color.matrix_from_calibration`)
-> a gamma-linearized drive value per channel
(:func:`vpsych.core.calibration.gamma.linearize`), which is what is actually
sent to the display (after bit-stealing dithering -- see
:mod:`vpsych.core.calibration.dither` -- applied by the caller at draw time).

This module is pure numpy math; no psychopy import, so it is fully
headless-testable (round-trip and gamut properties are checked in
``tests/tests_catalog/test_color_discrimination.py`` without a display).

**Gamut**: a chromaticity/luminance combination is "in gamut" iff the linear
RGB it maps to is in ``[0, 1]^3`` (a small numerical tolerance is applied).
Two gamut queries this test needs beyond a plain per-disc check:

- :func:`max_in_gamut_displacement_uv`: the largest positive displacement
  along a confusion-line direction that stays in gamut *simultaneously* at
  every luminance the noise range can draw (used to compute each trivector
  axis's presentable maximum -- see the package's ``__init__`` module
  docstring).
- :func:`nearest_in_gamut_point`: used only if the CCT default background
  chromaticity itself turns out to be out of gamut for a given display (see
  ``ColorDiscriminationTest._resolve_background_uv``); finds the nearest
  in-gamut point on the segment toward a known-safe reference (the display's
  white point) by bisection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from vpsych.core.calibration.gamma import GammaChannelModel, linearize
from vpsych.core.calibration.models import ColorCalibration, GammaCalibration

FloatArray = npt.NDArray[np.float64]

UV = tuple[float, float]


def uv_prime_to_xy(u_prime: float, v_prime: float) -> tuple[float, float]:
    """Invert CIE 1976 ``(u', v')`` to CIE 1931 ``(x, y)`` chromaticity.

    Closed-form inverse of ``vpsych.core.calibration.color.xy_to_uv_prime``:
    ``x = 9u' / (6u' - 16v' + 12)``, ``y = 4v' / (6u' - 16v' + 12)``.
    """
    denom = 6.0 * u_prime - 16.0 * v_prime + 12.0
    x = 9.0 * u_prime / denom
    y = 4.0 * v_prime / denom
    return x, y


def uv_y_to_xyz(u_prime: float, v_prime: float, y_cdm2: float) -> FloatArray:
    """Convert CIE 1976 ``(u', v')`` chromaticity plus absolute luminance to XYZ.

    Args:
        u_prime: CIE 1976 u' coordinate.
        v_prime: CIE 1976 v' coordinate.
        y_cdm2: Absolute luminance, in cd/m^2 (becomes ``Y``).

    Returns:
        A length-3 array ``[X, Y, Z]``.

    Raises:
        ValueError: If the implied CIE 1931 ``y`` chromaticity is not
            positive (a degenerate/invalid chromaticity).
    """
    x, y = uv_prime_to_xy(u_prime, v_prime)
    if y <= 0.0:
        raise ValueError(
            f"Degenerate chromaticity (u'={u_prime}, v'={v_prime}): implied y={y} <= 0."
        )
    big_x = y_cdm2 / y * x
    big_z = y_cdm2 / y * (1.0 - x - y)
    return np.array([big_x, y_cdm2, big_z], dtype=np.float64)


def channel_gamma_models(
    gamma_cal: GammaCalibration,
) -> tuple[GammaChannelModel, GammaChannelModel, GammaChannelModel]:
    """Build (R, G, B) :class:`GammaChannelModel` instances from a calibration.

    `GammaCalibration` stores a single achromatic ``[lum_min_cdm2,
    lum_max_cdm2]`` luminance range (from a photometer sweep of the whole
    display, not each primary measured in isolation) plus either per-channel
    exponents (``gamma_r``/``gamma_g``/``gamma_b``) or one shared
    ``gamma_single``. This function assumes each channel's own gamma curve
    spans that same shared luminance range -- the standard channel-
    independence assumption used throughout this rendering pipeline (see
    this test's methods doc, "Limitations", for the caveat this implies for
    the achromatic-normalization step below).

    Raises:
        ValueError: If neither `gamma_single` nor all three of
            `gamma_r`/`gamma_g`/`gamma_b` are set (lookup-table gamma models
            are not currently supported by this pipeline).
    """
    lum_min, lum_max = gamma_cal.lum_min_cdm2, gamma_cal.lum_max_cdm2
    g_r = gamma_cal.gamma_r if gamma_cal.gamma_r is not None else gamma_cal.gamma_single
    g_g = gamma_cal.gamma_g if gamma_cal.gamma_g is not None else gamma_cal.gamma_single
    g_b = gamma_cal.gamma_b if gamma_cal.gamma_b is not None else gamma_cal.gamma_single
    if g_r is None or g_g is None or g_b is None:
        raise ValueError(
            "channel_gamma_models needs gamma_single or all of gamma_r/gamma_g/gamma_b set "
            "on the GammaCalibration (lookup-table gamma models are not supported here)."
        )
    return (
        GammaChannelModel(lum_min_cdm2=lum_min, lum_max_cdm2=lum_max, gamma=g_r),
        GammaChannelModel(lum_min_cdm2=lum_min, lum_max_cdm2=lum_max, gamma=g_g),
        GammaChannelModel(lum_min_cdm2=lum_min, lum_max_cdm2=lum_max, gamma=g_b),
    )


def linear_rgb_from_uv_y(
    u_prime: float, v_prime: float, y_cdm2: float, rgb_to_xyz_abs: FloatArray
) -> FloatArray:
    """Map a target chromaticity/luminance to linear RGB (may be outside ``[0, 1]``).

    Args:
        u_prime: Target CIE 1976 u'.
        v_prime: Target CIE 1976 v'.
        y_cdm2: Target absolute luminance, cd/m^2.
        rgb_to_xyz_abs: The display's absolute ``RGB -> XYZ`` matrix (see
            ``vpsych.core.calibration.color.matrix_from_calibration(...,
            absolute=True)``).

    Returns:
        A length-3 linear RGB array; values outside ``[0, 1]`` indicate the
        target is out of gamut (see :func:`is_in_gamut`).
    """
    xyz = uv_y_to_xyz(u_prime, v_prime, y_cdm2)
    result = np.linalg.solve(rgb_to_xyz_abs, xyz)
    return np.asarray(result, dtype=np.float64)


def is_in_gamut(rgb: FloatArray, tol: float = 1e-9) -> bool:
    """Whether a linear RGB triple is representable: every channel in ``[0, 1]`` (+/- `tol`)."""
    return bool(np.all(rgb >= -tol) and np.all(rgb <= 1.0 + tol))


def chromaticity_in_gamut_at_luminances(
    uv: UV, rgb_to_xyz_abs: FloatArray, luminance_values_cdm2: list[float]
) -> bool:
    """Whether `uv` is representable at *every* luminance in `luminance_values_cdm2`.

    A `uv` far enough along a confusion-line direction (in particular toward
    the tritan copunctal point, whose CIE 1931 `y` is negative -- see
    `vpsych.core.calibration.color.COPUNCTAL_POINTS_XY`) can cross outside
    the region where CIE 1931 `y > 0`, at which point
    :func:`uv_y_to_xyz` raises `ValueError` (a degenerate chromaticity, not
    just an out-of-gamut one). Such a point is obviously not representable
    either, so it is treated as "not in gamut" here rather than propagating
    the exception -- this is what lets :func:`max_in_gamut_displacement_uv`'s
    bisection search range past the point where a confusion line leaves the
    valid chromaticity diagram at all.
    """
    u, v = uv
    for y in luminance_values_cdm2:
        try:
            rgb = linear_rgb_from_uv_y(u, v, y, rgb_to_xyz_abs)
        except ValueError:
            return False
        if not is_in_gamut(rgb):
            return False
    return True


def max_in_gamut_displacement_uv(
    background_uv: UV,
    direction_uv: UV,
    rgb_to_xyz_abs: FloatArray,
    luminance_values_cdm2: list[float],
    search_hi_uv: float = 0.5,
    n_bisection: int = 40,
) -> float:
    """Largest positive u'v' displacement along `direction_uv` that stays in gamut everywhere.

    "Everywhere" means simultaneously in gamut at *every* luminance in
    `luminance_values_cdm2` -- the luminance-noise design means a target
    chromaticity is drawn at an unpredictable luminance each trial, so the
    presentable displacement ceiling for an axis must be safe at the whole
    noise range, not just its midpoint (see this package's ``__init__``
    module docstring, "Trivector axes").

    Args:
        background_uv: The background's ``(u', v')`` chromaticity.
        direction_uv: Unit ``(du', dv')`` direction (see
            ``vpsych.core.calibration.color.confusion_line_direction_uv``).
        rgb_to_xyz_abs: The display's absolute ``RGB -> XYZ`` matrix.
        luminance_values_cdm2: Luminance levels (cd/m^2) to check gamut
            safety at.
        search_hi_uv: Upper bound of the bisection search, in real u'v'
            units (0.5 comfortably exceeds any real display's gamut along
            any direction from a near-white background).
        n_bisection: Number of bisection iterations (40 gives sub-1e-11
            precision in u'v' units, far finer than needed).

    Returns:
        The largest safe displacement, in real u'v' units (multiply by
        ``1e4`` for this test's ``x1e-4`` intensity convention).
    """

    def _ok(d: float) -> bool:
        point = (
            background_uv[0] + direction_uv[0] * d,
            background_uv[1] + direction_uv[1] * d,
        )
        return chromaticity_in_gamut_at_luminances(point, rgb_to_xyz_abs, luminance_values_cdm2)

    if _ok(search_hi_uv):
        return search_hi_uv
    lo, hi = 0.0, search_hi_uv
    for _ in range(n_bisection):
        mid = (lo + hi) / 2.0
        if _ok(mid):
            lo = mid
        else:
            hi = mid
    return lo


def nearest_in_gamut_point(
    target_uv: UV,
    safe_uv: UV,
    rgb_to_xyz_abs: FloatArray,
    luminance_values_cdm2: list[float],
    n_bisection: int = 40,
) -> UV:
    """Nearest-to-`target_uv` in-gamut point on the segment from `safe_uv` to `target_uv`.

    `safe_uv` is assumed in gamut (typically the display's white point);
    bisects along the straight line toward `target_uv` for the largest
    fraction that stays in gamut at every luminance in
    `luminance_values_cdm2`. Used only as a fallback when the CCT default
    background chromaticity itself is out of gamut for a given display (an
    edge case expected to essentially never trigger for a real display,
    since the CCT background is very close to typical white points -- see
    the package's ``__init__`` module docstring).

    Returns:
        `target_uv` unchanged if it is already in gamut, otherwise the
        bisected nearest-safe point.
    """
    if chromaticity_in_gamut_at_luminances(target_uv, rgb_to_xyz_abs, luminance_values_cdm2):
        return target_uv
    lo, hi = 0.0, 1.0
    best = safe_uv
    for _ in range(n_bisection):
        mid = (lo + hi) / 2.0
        point = (
            safe_uv[0] + (target_uv[0] - safe_uv[0]) * mid,
            safe_uv[1] + (target_uv[1] - safe_uv[1]) * mid,
        )
        if chromaticity_in_gamut_at_luminances(point, rgb_to_xyz_abs, luminance_values_cdm2):
            lo = mid
            best = point
        else:
            hi = mid
    return best


@dataclass(frozen=True)
class DiscRender:
    """One disc's rendered color: gamma-encoded drive values plus gamut status.

    Attributes:
        drive_rgb: ``(R, G, B)`` gamma-encoded drive values in ``[0, 1]``
            (what should be sent to the display, before dithering).
        linear_rgb: The linear-light RGB fraction this disc's chromaticity/
            luminance maps to, clamped to ``[0, 1]`` (the pre-gamma value
            `drive_rgb` was computed from).
        in_gamut: Whether the *unclamped* linear RGB was already within
            ``[0, 1]`` (i.e. no clamping was needed to render this disc).
    """

    drive_rgb: tuple[float, float, float]
    linear_rgb: FloatArray
    in_gamut: bool


def render_disc(
    u_prime: float,
    v_prime: float,
    y_cdm2: float,
    color_cal: ColorCalibration,
    gamma_cal: GammaCalibration,
    rgb_to_xyz_abs: FloatArray | None = None,
    gamut_tol: float = 1e-9,
) -> DiscRender:
    """Render one disc's target chromaticity/luminance to gamma-encoded drive values.

    Args:
        u_prime: Target CIE 1976 u'.
        v_prime: Target CIE 1976 v'.
        y_cdm2: Target absolute luminance, cd/m^2.
        color_cal: Display color primary characterization.
        gamma_cal: Display luminance/gamma characterization.
        rgb_to_xyz_abs: Precomputed absolute ``RGB -> XYZ`` matrix for
            `color_cal`, or `None` to compute it (callers rendering many
            discs per trial should precompute once and pass it in).
        gamut_tol: Numerical tolerance for the in-gamut check (see
            :func:`is_in_gamut`).

    Returns:
        The rendered :class:`DiscRender`.
    """
    if rgb_to_xyz_abs is None:
        from vpsych.core.calibration.color import matrix_from_calibration

        rgb_to_xyz_abs = matrix_from_calibration(color_cal, absolute=True)
    linear_rgb = linear_rgb_from_uv_y(u_prime, v_prime, y_cdm2, rgb_to_xyz_abs)
    in_gamut = is_in_gamut(linear_rgb, gamut_tol)
    clamped = np.clip(linear_rgb, 0.0, 1.0)
    models = channel_gamma_models(gamma_cal)
    drive = tuple(float(linearize(float(clamped[i]), models[i])) for i in range(3))
    return DiscRender(
        drive_rgb=(drive[0], drive[1], drive[2]), linear_rgb=clamped, in_gamut=in_gamut
    )


def inverse_render_to_uv_y(
    drive_rgb: tuple[float, float, float],
    color_cal: ColorCalibration,
    gamma_cal: GammaCalibration,
    rgb_to_xyz_abs: FloatArray | None = None,
) -> tuple[float, float, float]:
    """Invert :func:`render_disc`: gamma-encoded drive values -> ``(u', v', Y)``.

    Used only for the round-trip unit test (a rendered disc's drive values,
    pushed back through the forward gamma model and primary matrix, should
    reproduce the originally requested chromaticity/luminance to numerical
    precision).

    Returns:
        `(u_prime, v_prime, y_cdm2)`.
    """
    from vpsych.core.calibration.color import matrix_from_calibration, xyz_to_uv_prime

    if rgb_to_xyz_abs is None:
        rgb_to_xyz_abs = matrix_from_calibration(color_cal, absolute=True)
    models = channel_gamma_models(gamma_cal)
    fractions = np.array(
        [
            (models[i].luminance(drive_rgb[i]) - models[i].lum_min_cdm2)
            / (models[i].lum_max_cdm2 - models[i].lum_min_cdm2)
            for i in range(3)
        ],
        dtype=np.float64,
    )
    xyz = rgb_to_xyz_abs @ fractions
    u, v = xyz_to_uv_prime(xyz)
    return float(u), float(v), float(xyz[1])
