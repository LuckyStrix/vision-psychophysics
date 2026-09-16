"""Color-space math: primaries -> RGB<->XYZ, XYZ<->LMS cone space, u'v', confusion lines.

Pure numpy math; no psychopy import (headless-safe). Validated in
``tests/core/test_calibration_color.py`` against the independent
``colour-science`` reference implementation.

**RGB <-> XYZ.** :func:`primaries_to_xyz_matrix` implements the standard
normalized-primary-matrix (NPM) construction (Fairman, Brill & Hemmendinger
1997, "How the CIE 1931 color-matching functions were derived from
Wright-Guild data", *Color Research & Application*, 22(1), 11-23, sec. on
deriving an RGB/XYZ matrix from primaries + white point; the same algorithm
underlies ``colour.models.rgb.derivation.normalised_primary_matrix``): each
primary's chromaticity is expanded to unit-``Y`` tristimulus values, then
scaled so the primaries' weighted sum reproduces the white point's
tristimulus values exactly.

**XYZ <-> LMS.** :func:`xyz_to_lms_matrix` derives a linear CIE-1931-XYZ (2
degree) -> Stockman-Sharpe-cone-space transform by ordinary least squares,
regressing the Stockman & Sharpe (2000, "Spectral sensitivities of the
middle- and long-wavelength sensitive cones derived from measurements in
observers of known genotype", *Vision Research*, 40(13), 1711-1737) 2-degree
cone fundamentals against the CIE (1931) 2-degree standard observer color-
matching functions, both as tabulated in ``colour-science`` (which packages
the standard CVRL-published data tables for each). This reproduces the
construction used for published fixed XYZ->LMS matrices (e.g. Golz &
MacLeod 2003) without hard-coding borrowed coefficients; the input spectral
data, not the transform itself, is the cited external source.

**Confusion lines.** :data:`COPUNCTAL_POINTS_XY` gives the protan/deutan/
tritan copunctal points in CIE 1931 ``(x, y)`` from Vienot, Brettel & Mollon
(1999), "Digital video colourmaps for checking the legibility of displays
by dichromats", *Color Research & Application*, 24(4), 243-252 (Table I),
which are themselves derived from Smith & Pokorny (1975) cone
fundamentals -- the standard reference values used for dichromat confusion-
line simulation.
"""

from __future__ import annotations

import functools
import math

import numpy as np
import numpy.typing as npt

from vpsych.core.calibration.models import ColorCalibration

FloatArray = npt.NDArray[np.float64]

# --- Default sRGB / BT.709 primaries and D65 white point, CIE 1931 2-degree (x, y). ---
SRGB_PRIMARIES_XY: dict[str, tuple[float, float]] = {
    "red": (0.6400, 0.3300),
    "green": (0.3000, 0.6000),
    "blue": (0.1500, 0.0600),
}
SRGB_WHITE_XY: tuple[float, float] = (0.3127, 0.3290)

# Protan/deutan/tritan copunctal points, CIE 1931 (x, y). Vienot, Brettel &
# Mollon (1999), Table I (derived from Smith & Pokorny 1975 cone
# fundamentals).
COPUNCTAL_POINTS_XY: dict[str, tuple[float, float]] = {
    "protan": (0.747, 0.253),
    "deutan": (1.080, -0.080),
    "tritan": (0.171, -0.003),
}


def primaries_to_xyz_matrix(
    primaries_xy: dict[str, tuple[float, float]],
    white_xy: tuple[float, float],
    white_y: float = 1.0,
) -> FloatArray:
    """Build the 3x3 matrix mapping (relative) RGB to XYZ from primaries and a white point.

    Args:
        primaries_xy: CIE 1931 ``(x, y)`` chromaticity of the red, green,
            and blue primaries, keyed ``"red"``, ``"green"``, ``"blue"``.
        white_xy: CIE 1931 ``(x, y)`` chromaticity of the white point (the
            XYZ produced by RGB = (1, 1, 1)).
        white_y: Luminance (``Y``) of the white point. Defaults to ``1.0``
            for relative colorimetry (RGB in ``[0, 1]`` maps to XYZ with
            ``Y`` in ``[0, 1]``); pass an absolute luminance in cd/m^2 for
            an absolute-colorimetry matrix.

    Returns:
        The 3x3 ``RGB -> XYZ`` matrix ``M`` such that ``XYZ = M @ [R, G, B]``.
    """
    columns = []
    for channel in ("red", "green", "blue"):
        x, y = primaries_xy[channel]
        columns.append([x / y, 1.0, (1.0 - x - y) / y])
    primary_matrix = np.array(columns, dtype=np.float64).T  # columns = primary XYZ at unit Y

    xw, yw = white_xy
    xyz_white = np.array(
        [xw / yw * white_y, white_y, (1.0 - xw - yw) / yw * white_y], dtype=np.float64
    )
    scale = np.linalg.solve(primary_matrix, xyz_white)
    result: FloatArray = primary_matrix * scale
    return result


def xyz_to_rgb_matrix(
    primaries_xy: dict[str, tuple[float, float]],
    white_xy: tuple[float, float],
    white_y: float = 1.0,
) -> FloatArray:
    """Inverse of :func:`primaries_to_xyz_matrix`: the ``XYZ -> RGB`` matrix."""
    forward = primaries_to_xyz_matrix(primaries_xy, white_xy, white_y)
    result: FloatArray = np.asarray(np.linalg.inv(forward), dtype=np.float64)
    return result


def matrix_from_calibration(color: ColorCalibration, absolute: bool = False) -> FloatArray:
    """Build the ``RGB -> XYZ`` matrix for a measured/assumed display calibration.

    Args:
        color: The display's color primary characterization.
        absolute: If `True`, scale the matrix so RGB = (1, 1, 1) produces
            the white point's *measured absolute* luminance
            (``color.white.Y_cdm2`` cd/m^2) rather than relative ``Y = 1``.

    Returns:
        The 3x3 ``RGB -> XYZ`` matrix.
    """
    primaries_xy = {
        "red": (color.red.x, color.red.y),
        "green": (color.green.x, color.green.y),
        "blue": (color.blue.x, color.blue.y),
    }
    white_xy = (color.white.x, color.white.y)
    white_y = color.white.Y_cdm2 if absolute else 1.0
    return primaries_to_xyz_matrix(primaries_xy, white_xy, white_y)


def rgb_to_xyz(rgb: FloatArray, matrix: FloatArray) -> FloatArray:
    """Apply an ``RGB -> XYZ`` matrix to one or more RGB triples.

    Args:
        rgb: An array whose last axis has length 3 (a single triple, or a
            batch of any leading shape).
        matrix: A 3x3 ``RGB -> XYZ`` matrix (see
            :func:`primaries_to_xyz_matrix`).

    Returns:
        An array the same shape as ``rgb``, holding XYZ tristimulus values.
    """
    rgb_arr = np.asarray(rgb, dtype=np.float64)
    return rgb_arr @ matrix.T


def xyz_to_rgb(xyz: FloatArray, matrix_inv: FloatArray) -> FloatArray:
    """Apply an ``XYZ -> RGB`` matrix to one or more XYZ triples (see :func:`rgb_to_xyz`)."""
    xyz_arr = np.asarray(xyz, dtype=np.float64)
    return xyz_arr @ matrix_inv.T


def xy_to_uv_prime(x: float | FloatArray, y: float | FloatArray) -> tuple[FloatArray, FloatArray]:
    """Convert CIE 1931 ``(x, y)`` chromaticity to CIE 1976 ``(u', v')`` UCS chromaticity.

    Uses the standard closed-form relation
    ``u' = 4x / (-2x + 12y + 3)``, ``v' = 9y / (-2x + 12y + 3)``.

    Args:
        x: CIE 1931 x chromaticity coordinate(s).
        y: CIE 1931 y chromaticity coordinate(s).

    Returns:
        A ``(u_prime, v_prime)`` tuple, each the same shape as the inputs.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    denom = -2.0 * x_arr + 12.0 * y_arr + 3.0
    u_prime = 4.0 * x_arr / denom
    v_prime = 9.0 * y_arr / denom
    return u_prime, v_prime


def xyz_to_uv_prime(xyz: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Convert XYZ tristimulus values to CIE 1976 ``(u', v')`` UCS chromaticity.

    Args:
        xyz: An array whose last axis has length 3 (X, Y, Z).

    Returns:
        A ``(u_prime, v_prime)`` tuple, each shaped like ``xyz`` minus its
        last axis.
    """
    arr = np.asarray(xyz, dtype=np.float64)
    x_t, y_t, z_t = arr[..., 0], arr[..., 1], arr[..., 2]
    denom = x_t + 15.0 * y_t + 3.0 * z_t
    u_prime = 4.0 * x_t / denom
    v_prime = 9.0 * y_t / denom
    return u_prime, v_prime


@functools.lru_cache(maxsize=1)
def xyz_to_lms_matrix() -> FloatArray:
    """The 3x3 CIE-1931-XYZ (2 degree) -> Stockman-Sharpe LMS cone-space matrix.

    Derived once (and cached) by ordinary least squares, regressing the
    Stockman & Sharpe (2000) 2-degree cone fundamentals against the CIE
    (1931) 2-degree standard observer color-matching functions over their
    common wavelength support (390-830 nm, 1 nm sampling), both loaded from
    ``colour-science``'s bundled standard data tables -- see the module
    docstring for full citations. This function is the only place in
    ``vpsych`` that imports ``colour-science`` outside the test suite (it
    is used here purely as a source of standard tabulated spectral data,
    not as the color-math implementation being validated).

    Returns:
        The 3x3 matrix ``M`` such that ``LMS = M @ [X, Y, Z]``.
    """
    import colour

    cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    cone_fundamentals = colour.MSDS_CMFS["Stockman & Sharpe 2 Degree Cone Fundamentals"]

    wavelengths = np.array(
        sorted(set(cmfs.wavelengths.tolist()) & set(cone_fundamentals.wavelengths.tolist()))
    )
    xyz_samples = np.array([cmfs[w] for w in wavelengths])
    lms_samples = np.array([cone_fundamentals[w] for w in wavelengths])

    solution, _residuals, _rank, _sv = np.linalg.lstsq(xyz_samples, lms_samples, rcond=None)
    result: FloatArray = np.asarray(solution.T, dtype=np.float64)
    return result


def xyz_to_lms(xyz: FloatArray) -> FloatArray:
    """Convert XYZ tristimulus values to Stockman-Sharpe LMS cone excitations.

    Args:
        xyz: An array whose last axis has length 3 (X, Y, Z).

    Returns:
        An array the same shape as ``xyz``, holding (L, M, S) values.
    """
    arr = np.asarray(xyz, dtype=np.float64)
    return arr @ xyz_to_lms_matrix().T


def lms_to_xyz(lms: FloatArray) -> FloatArray:
    """Inverse of :func:`xyz_to_lms`: LMS cone excitations to XYZ tristimulus values."""
    arr = np.asarray(lms, dtype=np.float64)
    matrix_inv = np.linalg.inv(xyz_to_lms_matrix())
    return arr @ matrix_inv.T


def cone_contrast(lms: FloatArray, lms_background: FloatArray) -> FloatArray:
    """Weber cone contrast of a stimulus's LMS against a background's LMS.

    Args:
        lms: Stimulus (L, M, S) cone excitations.
        lms_background: Background (L, M, S) cone excitations (the
            adaptation point), same shape as ``lms`` or broadcastable to it.

    Returns:
        Elementwise ``(lms - lms_background) / lms_background``, i.e.
        ``(dL/L, dM/M, dS/S)``.
    """
    lms_arr = np.asarray(lms, dtype=np.float64)
    bg_arr = np.asarray(lms_background, dtype=np.float64)
    return (lms_arr - bg_arr) / bg_arr


def confusion_line_direction_uv(
    background_uv: tuple[float, float], deficiency: str
) -> tuple[float, float]:
    """Unit vector in CIE 1976 ``(u', v')`` from a background toward a dichromat copunctal point.

    Colors displaced along this direction from ``background_uv`` are
    (to first order) indistinguishable from the background to an observer
    with the corresponding color vision deficiency, since all points on the
    line through a copunctal point and the background stimulate that
    observer's remaining two cone classes identically (the classic
    "confusion line" construction used by trivector color-discrimination
    tests, e.g. the Cambridge Colour Test).

    Args:
        background_uv: The background's ``(u', v')`` chromaticity.
        deficiency: Which confusion lines to use: ``"protan"``,
            ``"deutan"``, or ``"tritan"`` (see :data:`COPUNCTAL_POINTS_XY`).

    Returns:
        A unit ``(du', dv')`` vector pointing from the background toward
        the deficiency's copunctal point.

    Raises:
        KeyError: If ``deficiency`` is not one of ``"protan"``,
            ``"deutan"``, ``"tritan"``.
        ValueError: If ``background_uv`` coincides with the copunctal point
            (a zero-length direction).
    """
    x, y = COPUNCTAL_POINTS_XY[deficiency]
    u_copunctal, v_copunctal = xy_to_uv_prime(x, y)
    du = float(u_copunctal) - background_uv[0]
    dv = float(v_copunctal) - background_uv[1]
    norm = math.hypot(du, dv)
    if norm == 0.0:
        raise ValueError("background_uv coincides with the copunctal point.")
    return (du / norm, dv / norm)


def gamut_check(
    xy: tuple[float, float],
    primaries_xy: dict[str, tuple[float, float]] | None = None,
) -> bool:
    """Check whether a chromaticity lies inside a display's realizable gamut triangle.

    Args:
        xy: The ``(x, y)`` chromaticity to test.
        primaries_xy: The display's primaries, or `None` to use
            :data:`SRGB_PRIMARIES_XY`.

    Returns:
        `True` if ``xy`` lies inside (or on the boundary of) the triangle
        formed by the three primaries' chromaticities -- i.e. some
        non-negative RGB combination (each in ``[0, 1]``, no scaling
        assumed beyond that) can reproduce it; `False` otherwise. Note this
        checks the 2-D chromaticity gamut only, not achievable luminance.
    """
    primaries = primaries_xy or SRGB_PRIMARIES_XY
    p_red = np.array(primaries["red"])
    p_green = np.array(primaries["green"])
    p_blue = np.array(primaries["blue"])
    point = np.array(xy)

    def sign(p1: FloatArray, p2: FloatArray, p3: FloatArray) -> float:
        return float((p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1]))

    d1 = sign(point, p_red, p_green)
    d2 = sign(point, p_green, p_blue)
    d3 = sign(point, p_blue, p_red)

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)
