"""Sub-pixel-accurate vertical line-segment texture rendering for Vernier acuity.

Pure numpy math, no `psychopy` import (headless-testable). Hyperacuity
offsets (a few arcsec to a few arcmin) are routinely a small fraction of a
display pixel at normal viewing distances, so the sub-pixel offset must be
encoded in the *texture*'s edge luminance, not in a stimulus object's
floating-point on-screen position (relying on a backend's own bilinear
texture sampling to reproduce an exact sub-pixel edge is not something this
project wants to depend on -- see :mod:`vernier_acuity`'s module docstring
for how the rendered texture is then positioned only at whole-pixel
coordinates on screen, so this analytic sub-pixel accuracy is not undone by
GPU-side resampling).

**Method**: exact analytic area sampling, not stochastic supersampling. For
a vertical bar of width `width_px` centered at (possibly fractional)
`center_px`, each output column `i` represents the continuous pixel
interval `[i, i+1)`; that column's coverage is the exact overlap length
between the bar's `[left, right]` span and `[i, i+1)`, clipped to `[0, 1]`.
Because this is computed analytically (`np.clip`/`np.minimum`/`np.maximum`
on continuous interval endpoints) rather than by rendering at some finite
supersampling factor and averaging, there is no discretization error at
all: coverage is correct to floating-point precision for *any* sub-pixel
`center_px`, which is what lets :func:`render_vertical_line_texture`'s
centroid reproduce a requested sub-pixel position to a small fraction of a
pixel (see `tests/tests_catalog/test_vernier_acuity.py`'s texture-centroid
unit test, and the Phase 2A task's <=0.02 px tolerance requirement).

**Gamma linearization**: each pixel's *coverage-weighted linear mixture* of
the background and foreground (line) luminance levels -- i.e. the luminance
a photoreceptor pooling light linearly over that pixel's area would
actually integrate (the same "linear spatial summation" principle behind
this project's grade-B psychophysical gamma calibration, see
`docs/CALIBRATION.md`'s half-luminance bisection method, and behind sub-
pixel/antialiased rendering on gamma-nonlinear CRT and LCD displays
generally, e.g. Lloyd, C., Winterbottom, M., Gaska, J., & Williams, L.
(2015). Effects of display pixel pitch and antialiasing on threshold
vernier acuity. Proceedings of the IMAGE Society Annual Conference, Dayton,
OH -- who likewise find that antialiasing filter quality, not raw pixel
pitch alone, sets the achievable Vernier threshold on a fixed-resolution
display) -- is computed first, in linear-light space, then turned into the
correct nonlinear hardware drive level via
`vpsych.core.calibration.gamma.linearize`, applied by this module itself
(the `gamma_model` argument) if given.

This is a deliberate, documented resolution of a real inconsistency: an
earlier version of this module (and of `docs/WRITING_A_TEST.md` section 7)
assumed gamma linearization instead happened via the *window's own* gamma
ramp (`vpsych.core.calibration.gamma.make_gamma_ramp`, applied once at
window-creation time by the backend, not per-trial by a test) -- but
`PsychoPyBackend` never actually sets one, and the *other* three
gamma-dependent tests (`contrast_sensitivity_function`,
`letter_contrast_sensitivity` via `tests_catalog._contrast_rendering`, and
`color_discrimination`) all do their own full linearization in Python,
handing PsychoPy an already-gamma-correct drive-level array directly and
relying on an *unconfigured* window applying no further correction of its
own. Adding a window-level ramp would silently double-correct those three
tests. This module now follows the same, already-working, already-tested
pattern instead, for consistency across the whole battery (see
`docs/WRITING_A_TEST.md` section 7 for the updated, accurate description):
every real test linearizes its own stimulus in Python; `PsychoPyBackend`
intentionally sets no window gamma ramp.
`VernierAcuityTest.spec.requirements.needs_gamma_calibration` stays `True`
(grade B minimum) -- the calibration is still required, just applied here
rather than via a ramp.
"""

from __future__ import annotations

import numpy as np

from vpsych.core.calibration.gamma import GammaChannelModel, linearize


def line_column_coverage(canvas_width_px: int, center_px: float, width_px: float) -> np.ndarray:
    """Exact analytic per-column coverage fraction of a vertical bar.

    Args:
        canvas_width_px: Number of output columns. Column `i` represents
            the continuous interval `[i, i+1)`.
        center_px: The bar's center, in the same continuous coordinate
            frame (may be fractional -- this is exactly the sub-pixel
            position this function exists to encode).
        width_px: The bar's full width, in pixels.

    Returns:
        A `(canvas_width_px,)` array of coverage fractions in `[0, 1]`.
    """
    left = center_px - width_px / 2.0
    right = center_px + width_px / 2.0
    edges = np.arange(canvas_width_px + 1, dtype=np.float64)
    col_left = edges[:-1]
    col_right = edges[1:]
    overlap = np.minimum(col_right, right) - np.maximum(col_left, left)
    result: np.ndarray = np.clip(overlap, 0.0, 1.0)
    return result


def render_vertical_line_texture(
    canvas_width_px: int,
    canvas_height_px: int,
    center_x_px: float,
    width_px: float,
    y_start_px: float,
    y_end_px: float,
    background_level: float = 1.0,
    foreground_level: float = -1.0,
    gamma_model: GammaChannelModel | None = None,
) -> np.ndarray:
    """Render one vertical line segment as a sub-pixel-accurate antialiased texture.

    Horizontal (left/right) edges are area-sampled exactly via
    :func:`line_column_coverage`, which is what encodes the segment's
    sub-pixel horizontal position -- the quantity a Vernier judgment
    actually depends on. Vertical (top/bottom) edges are area-sampled the
    same way along rows for a visually clean segment end, though the
    Vernier task does not require sub-pixel precision there.

    Args:
        canvas_width_px: Output texture width, in pixels.
        canvas_height_px: Output texture height, in pixels.
        center_x_px: The segment's horizontal center, in continuous pixel
            coordinates (fractional; the encoded sub-pixel position).
        width_px: The segment's width (line thickness), in pixels.
        y_start_px: Top edge of the segment, in continuous pixel
            coordinates (row 0 is the texture's top row).
        y_end_px: Bottom edge of the segment, in continuous pixel
            coordinates; must be greater than `y_start_px`.
        background_level: Linear luminance-fraction color value for fully
            uncovered pixels (PsychoPy `[-1, 1]` convention; +1 = white).
        foreground_level: Linear luminance-fraction color value for fully
            covered (line) pixels.
        gamma_model: If given, the coverage-weighted linear mixture is
            gamma-linearized (via `vpsych.core.calibration.gamma.linearize`)
            into the correct hardware drive level before being returned --
            see module docstring's "Gamma linearization" section. `None`
            (the default) returns the raw linear mixture unchanged, e.g. for
            unit tests of the pure coverage geometry.

    Returns:
        A `(canvas_height_px, canvas_width_px)` array of PsychoPy `[-1, 1]`
        color values: gamma-corrected hardware drive levels if `gamma_model`
        was given, otherwise linear luminance-fraction values (see module
        docstring's "Gamma linearization" section).
    """
    if y_end_px <= y_start_px:
        raise ValueError("y_end_px must be greater than y_start_px")
    col_cov = line_column_coverage(canvas_width_px, center_x_px, width_px)
    row_edges = np.arange(canvas_height_px + 1, dtype=np.float64)
    row_top = row_edges[:-1]
    row_bot = row_edges[1:]
    row_cov = np.clip(np.minimum(row_bot, y_end_px) - np.maximum(row_top, y_start_px), 0.0, 1.0)
    coverage = np.outer(row_cov, col_cov)  # (height, width)
    result: np.ndarray = background_level + coverage * (foreground_level - background_level)
    if gamma_model is not None:
        # PsychoPy [-1, 1] linear-luminance-fraction convention -> [0, 1] -> gamma-correct
        # hardware drive level -> back to [-1, 1].
        frac_0_1 = np.clip((result + 1.0) / 2.0, 0.0, 1.0)
        drive = linearize(frac_0_1, gamma_model)
        result = np.asarray(drive) * 2.0 - 1.0
    return result


def texture_centroid_x_px(
    texture_row: np.ndarray, background_level: float = 1.0, foreground_level: float = -1.0
) -> float:
    """Recover a rendered line's sub-pixel x centroid from one row of its luminance profile.

    Inverts :func:`line_column_coverage`'s luminance mixture:
    `coverage_i = (background_level - L_i) / (background_level -
    foreground_level)`, then takes the coverage-weighted centroid of pixel-
    center coordinates `i + 0.5`. Used only by this test's own unit test
    that verifies :func:`render_vertical_line_texture`'s edges encode the
    requested sub-pixel position accurately (see the Phase 2A task's
    requirement that this centroid match the requested position to within
    0.02 px) -- not used by the test's `present()` at runtime.

    Args:
        texture_row: A single row (`(canvas_width_px,)`) of a texture
            returned by `render_vertical_line_texture`, fully inside the
            segment's vertical extent (so every column's value reflects
            only horizontal-edge coverage).
        background_level: Must match the value used to render `texture_row`.
        foreground_level: Must match the value used to render `texture_row`.

    Returns:
        The sub-pixel x centroid, in the same continuous pixel coordinate
        frame as `center_x_px` was specified in.

    Raises:
        ValueError: If `texture_row` has no coverage at all (all
            background), so no centroid can be computed.
    """
    coverage = (background_level - texture_row) / (background_level - foreground_level)
    coverage = np.clip(coverage, 0.0, 1.0)
    idx = np.arange(len(coverage)) + 0.5
    total = float(coverage.sum())
    if total <= 0:
        raise ValueError("texture_row has no coverage; cannot compute a centroid.")
    return float(np.sum(idx * coverage) / total)
