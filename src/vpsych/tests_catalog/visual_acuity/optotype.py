"""Landolt C optotype geometry and antialiased texture rendering (ISO 8596).

Pure numpy math, no `psychopy` import, so it stays headless-testable per the
project's `core`/`data` rule (extended here to this test's own geometry
module since it has no display dependency either).

**Geometry** (ISO 8596 / EN ISO 8596, "Ophthalmic optics -- Visual acuity
testing -- Standard optotype and its presentation"): a Landolt C's outer
diameter is 5x the gap width, and its stroke width equals the gap width, so
the inner diameter is `5*gap - 2*gap = 3*gap`. This is the standard
proportion also used by the Freiburg Visual Acuity Test (FrACT; Bach, M.
(1996). The Freiburg Visual Acuity test -- automatic measurement of visual
acuity. Optometry and Vision Science, 73(1), 49-53;
https://doi.org/10.1097/00006324-199601000-00008).

**Rendering**: rather than draw the ring as a `psychopy.visual.ShapeStim`
polygon (which needs many vertices to look round and whose native
antialiasing quality varies by backend), :func:`render_landolt_c` builds a
precomputed antialiased *texture*: an exact geometric membership test
(pixel is between the inner/outer radius and outside the gap wedge)
evaluated on a supersampled grid, then downsampled (block-averaged) to the
target texture resolution. Each output texel's value is therefore a
continuous *coverage fraction* in [0, 1] (0 = fully background, 1 = fully
optotype stroke), giving smooth, subpixel-quality edges without relying on
any particular GPU/backend antialiasing behavior. This is one of the two
rendering approaches `docs/WRITING_A_TEST.md`/the Phase 2A task
explicitly allows ("PsychoPy ShapeStim/Circle polygons with multisampling,
or a precomputed antialiased texture").
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

#: Standard Landolt C proportions (ISO 8596): outer diameter = 5x gap,
#: stroke width = 1x gap, inner diameter = outer - 2*stroke = 3x gap.
OUTER_DIAMETER_PER_GAP = 5.0
STROKE_WIDTH_PER_GAP = 1.0
INNER_DIAMETER_PER_GAP = OUTER_DIAMETER_PER_GAP - 2.0 * STROKE_WIDTH_PER_GAP


def landolt_c_geometry(gap_px: float) -> dict[str, float]:
    """Compute a Landolt C's ring geometry (ISO 8596 proportions) from its gap size.

    Args:
        gap_px: Gap width, in pixels (equal to the stroke width by
            definition).

    Returns:
        A dict with `gap_px`, `stroke_width_px`, `outer_diameter_px`,
        `inner_diameter_px`, `outer_radius_px`, `inner_radius_px`, and
        `mean_radius_px` (the ring's centerline radius, used to convert the
        gap's linear width to an angular width).
    """
    outer_diameter_px = OUTER_DIAMETER_PER_GAP * gap_px
    inner_diameter_px = INNER_DIAMETER_PER_GAP * gap_px
    outer_radius_px = outer_diameter_px / 2.0
    inner_radius_px = inner_diameter_px / 2.0
    return {
        "gap_px": gap_px,
        "stroke_width_px": STROKE_WIDTH_PER_GAP * gap_px,
        "outer_diameter_px": outer_diameter_px,
        "inner_diameter_px": inner_diameter_px,
        "outer_radius_px": outer_radius_px,
        "inner_radius_px": inner_radius_px,
        "mean_radius_px": (outer_radius_px + inner_radius_px) / 2.0,
    }


def gap_half_angle_deg(gap_px: float, mean_radius_px: float) -> float:
    """Angular half-width of the gap wedge, in degrees, at the ring's centerline radius.

    The gap is defined to have the same linear width as the stroke (`gap_px`
    exactly, per ISO 8596); converting that linear (chord) width to an
    angular width at the ring's mean radius uses the exact chord relation
    `half_angle = asin((gap_px / 2) / mean_radius_px)` rather than the
    small-angle approximation, since the gap can be a substantial fraction
    of the ring radius for large optotypes.

    Args:
        gap_px: Gap (and stroke) width, in pixels.
        mean_radius_px: The ring's centerline radius, in pixels.

    Returns:
        Half the angular width of the gap wedge, in degrees.
    """
    ratio = (gap_px / 2.0) / mean_radius_px
    ratio = max(-1.0, min(1.0, ratio))
    return math.degrees(math.asin(ratio))


def render_landolt_c(
    texture_size_px: int,
    gap_px: float,
    orientation_deg: float,
    weber_contrast: float = -0.99,
    supersample: int = 4,
) -> np.ndarray:
    """Render a Landolt C as a precomputed antialiased texture.

    Args:
        texture_size_px: Output texture side length, in pixels (the texture
            is square). Should comfortably exceed `outer_diameter_px` so the
            ring isn't clipped.
        gap_px: Gap (and stroke) width, in pixels.
        orientation_deg: Direction the gap opens toward, in degrees,
            measured counterclockwise from the positive x-axis (0 = right,
            90 = up, 180 = left, 270 = down) -- standard mathematical
            convention, matching this test's numpad-key orientation mapping
            (see the package `__init__`).
        weber_contrast: Weber contrast of the optotype stroke against the
            background, `(L_stroke - L_bg) / L_bg`, with the background
            normalized to `L_bg = 1`. Defaults to -0.99 (near-maximum dark
            optotype on a bright background, per the Phase 2A task's high-
            contrast requirement); gamma calibration is not required for
            this test (see the test's `TestRequirements`), so this is an
            uncorrected linear approximation to the true perceptual
            contrast, not gamma-linearized -- adequate given the target
            contrast is deliberately far from any near-threshold value
            where a precise contrast would matter.
        supersample: Antialiasing supersampling factor (the geometric mask
            is evaluated at `supersample`x resolution, then block-averaged
            down to `texture_size_px`). Defaults to 4 (16x oversampling in
            area), which resolves ring/gap edge coverage to about 1/16 px
            in each dimension -- comfortably finer than a single display
            pixel's rendering fidelity for this high-contrast, non-
            hyperacuity optotype (unlike `vernier_acuity`, this test does
            not require exact analytic edges to a fraction of a percent of
            a pixel; see that test's `texture.py` for where that finer
            technique is actually needed).

    Returns:
        A `(texture_size_px, texture_size_px)` array of PsychoPy-style
        color values in `[-1, 1]` (-1 = black, +1 = white; here, `+1` is
        the background and `1 + 2 * weber_contrast` approximates the
        optotype stroke color), suitable for `psychopy.visual.ImageStim`.
    """
    if supersample < 1:
        raise ValueError("supersample must be >= 1")
    geom = landolt_c_geometry(gap_px)
    half_angle_deg = gap_half_angle_deg(gap_px, geom["mean_radius_px"])

    hi_res = texture_size_px * supersample
    # Pixel-center coordinates in a frame centered on the texture, y-up so
    # `orientation_deg` follows the usual mathematical (counterclockwise)
    # convention.
    coords = (np.arange(hi_res, dtype=float) + 0.5) / supersample - texture_size_px / 2.0
    xs, ys = np.meshgrid(coords, -coords)  # -coords: row 0 is the top (y-up)
    r = np.hypot(xs, ys)
    theta_deg = np.degrees(np.arctan2(ys, xs)) % 360.0

    angle_diff = (theta_deg - orientation_deg + 180.0) % 360.0 - 180.0
    in_gap = np.abs(angle_diff) <= half_angle_deg
    in_ring = (r >= geom["inner_radius_px"]) & (r <= geom["outer_radius_px"])
    mask_hi_res = in_ring & ~in_gap

    coverage = (
        mask_hi_res.astype(np.float64)
        .reshape(texture_size_px, supersample, texture_size_px, supersample)
        .mean(axis=(1, 3))
    )

    background_level = 1.0
    stroke_level = background_level * (1.0 + weber_contrast)
    result: np.ndarray = background_level + coverage * (stroke_level - background_level)
    return result


def min_renderable_logmar(px_to_deg_fn: Callable[[float], float], min_gap_px: float = 1.0) -> float:
    """Smallest logMAR this display can render, given a minimum gap size in pixels.

    `logMAR = log10(gap size in arcmin)`; the smallest usable gap is bounded
    by the display's pixel pitch, not by the acuity domain design range --
    see the test package's module docstring for the documented choice of
    `min_gap_px = 1.0` (one physical pixel) as that floor.

    Args:
        px_to_deg_fn: A callable converting a pixel extent to degrees of
            visual angle at the configured viewing distance (normally
            `DisplayGeometry.px_to_deg`; accepted as a plain callable here
            to keep this module free of any dependency beyond numpy).
        min_gap_px: The smallest gap, in pixels, this display is considered
            able to render with a meaningful (non-aliased) antialiased edge.

    Returns:
        The logMAR value corresponding to a gap of exactly `min_gap_px`.
    """
    gap_deg = px_to_deg_fn(min_gap_px)
    gap_arcmin = gap_deg * 60.0
    return float(np.log10(gap_arcmin))
