"""Gamma (luminance response) modelling: fitting, linearization, and LUT ramps.

A display's luminance response to a normalized drive level ``v`` in ``[0, 1]``
is modelled either parametrically, as the standard power-law (simple gamma)
function

    L(v) = Lmin + (Lmax - Lmin) * v**gamma

or, when finer fidelity than a single exponent captures, as a monotone
lookup table built from the raw measured points (see
:func:`fit_gamma_lookup`). Both are wrapped in :class:`GammaChannelModel` so
callers (e.g. :mod:`vpsych.core.calibration.photometer`, test stimulus
rendering) can linearize a desired luminance fraction into a drive value
without caring which representation backs a given channel's model.

Two calibration routes are supported, matching
:data:`vpsych.core.calibration.models.GammaMethod`:

- ``"photometer"`` (grade A): :func:`fit_gamma` fits ``gamma``/``Lmin``/
  ``Lmax`` (or :func:`fit_gamma_lookup` builds a monotone LUT) from
  physically measured (drive level, luminance) points.
- ``"psychophysical"`` (grade B): :func:`estimate_gamma_psychophysical`
  estimates ``gamma`` (with a confidence interval, no absolute luminance)
  from a half-luminance bisection procedure with no photometer, following
  the logic that human spatial luminance summation is linear (see its
  docstring for the derivation).

This module is pure math (numpy/scipy only); no display or psychopy import,
per the project's headless-testability rule.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field
from scipy import stats
from scipy.interpolate import PchipInterpolator

from vpsych.core.calibration.models import GammaCalibrationPoint

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class GammaChannelModel:
    """A fitted luminance-response model for one display channel.

    Either ``gamma`` (parametric power-law model) or ``lookup``/
    ``inverse_lookup`` (monotone-interpolated lookup model) is set, never
    both -- see :func:`fit_gamma` (parametric) and :func:`fit_gamma_lookup`
    (lookup-based).

    Attributes:
        lum_min_cdm2: Luminance at drive level 0 (black), in cd/m^2.
        lum_max_cdm2: Luminance at drive level 1 (max white), in cd/m^2.
        gamma: Fitted power-law exponent, or `None` for a lookup-based
            model.
        lookup: Monotone drive-level -> luminance (cd/m^2) interpolator, or
            `None` for a parametric model.
        inverse_lookup: Monotone luminance (cd/m^2) -> drive-level
            interpolator (the inverse of ``lookup``), or `None` for a
            parametric model.
    """

    lum_min_cdm2: float
    lum_max_cdm2: float
    gamma: float | None = None
    lookup: PchipInterpolator | None = None
    inverse_lookup: PchipInterpolator | None = None

    def __post_init__(self) -> None:
        has_gamma = self.gamma is not None
        has_lookup = self.lookup is not None and self.inverse_lookup is not None
        if has_gamma == has_lookup:
            raise ValueError(
                "GammaChannelModel must set exactly one of `gamma` or "
                "`lookup`/`inverse_lookup`, not both or neither."
            )

    def luminance(self, drive_level: float | FloatArray) -> float | FloatArray:
        """Predict luminance (cd/m^2) at a given drive level in [0, 1]."""
        if self.gamma is not None:
            frac = np.asarray(drive_level, dtype=np.float64) ** self.gamma
            result = self.lum_min_cdm2 + (self.lum_max_cdm2 - self.lum_min_cdm2) * frac
            return float(result) if np.isscalar(drive_level) else result
        assert self.lookup is not None
        out = self.lookup(drive_level)
        return float(out) if np.isscalar(drive_level) else out


class GammaFitResult(BaseModel):
    """Result of a parametric power-law gamma fit (see :func:`fit_gamma`).

    Attributes:
        gamma: Fitted power-law exponent.
        lum_min_cdm2: Luminance at drive level 0, in cd/m^2 (from the
            measured point nearest 0, or the fit's implied value).
        lum_max_cdm2: Luminance at drive level 1, in cd/m^2 (from the
            measured point nearest 1, or the fit's implied value).
        r_squared: Coefficient of determination of the log-log linear fit
            used to estimate ``gamma`` (1.0 is a perfect fit); `None` if
            fewer than 2 interior points were available to assess fit
            quality.
    """

    model_config = ConfigDict(frozen=True)

    gamma: float = Field(gt=0, description="Fitted power-law exponent.")
    lum_min_cdm2: float = Field(ge=0, description="Luminance at drive level 0, in cd/m^2.")
    lum_max_cdm2: float = Field(gt=0, description="Luminance at drive level 1, in cd/m^2.")
    r_squared: float | None = Field(
        default=None, description="Coefficient of determination of the log-log gamma fit."
    )


def _lum_min_max_from_points(
    points: Sequence[GammaCalibrationPoint],
) -> tuple[float, float]:
    """Estimate Lmin/Lmax from measured points: exact endpoints if present, else min/max."""
    by_level = sorted(points, key=lambda p: p.input_level)
    lum_min = by_level[0].luminance_cdm2 if by_level[0].input_level == 0.0 else None
    lum_max = by_level[-1].luminance_cdm2 if by_level[-1].input_level == 1.0 else None
    if lum_min is None:
        lum_min = min(p.luminance_cdm2 for p in points)
    if lum_max is None:
        lum_max = max(p.luminance_cdm2 for p in points)
    return float(lum_min), float(lum_max)


def fit_gamma(points: Sequence[GammaCalibrationPoint]) -> GammaFitResult:
    """Fit the power-law gamma model ``L(v) = Lmin + (Lmax - Lmin) * v**gamma``.

    ``Lmin``/``Lmax`` are taken directly from measured points at drive level
    0/1 when present (the usual case for a photometer sweep that includes
    black and full white), otherwise from the minimum/maximum measured
    luminance. ``gamma`` is then estimated by ordinary least squares on the
    linearized (log-log) relationship

        ln((L - Lmin) / (Lmax - Lmin)) = gamma * ln(v)

    using only interior points (``0 < v < 1``), since the endpoints are
    exactly 0 and 1 on the left-hand side and contribute no information to
    the slope (and ``ln(0)`` is undefined).

    Args:
        points: Measured (drive level, luminance) samples. Must include at
            least one interior point (``0 < input_level < 1``) to estimate
            ``gamma``.

    Returns:
        The fitted :class:`GammaFitResult`.

    Raises:
        ValueError: If fewer than 2 points are given, if ``Lmax`` does not
            exceed ``Lmin``, or if no point remains to fit gamma from after
            excluding points at drive level 0/1 and points whose luminance
            equals ``Lmin``/``Lmax`` (which carry no slope information and
            would make the log undefined).
    """
    if len(points) < 2:
        raise ValueError("fit_gamma needs at least 2 measured points.")

    lum_min, lum_max = _lum_min_max_from_points(points)
    if lum_max <= lum_min:
        raise ValueError(f"lum_max ({lum_max}) must exceed lum_min ({lum_min}).")

    # Points that define Lmin/Lmax themselves (whether because input_level is
    # exactly 0/1, or because their luminance happens to be the extremal value
    # used as the Lmin/Lmax fallback) normalize to a fraction of exactly 0 or
    # 1 and carry no slope information (and make ln(frac) undefined), so they
    # are excluded from the fit rather than treated as an error.
    interior = [
        p for p in points if 0.0 < p.input_level < 1.0 and lum_min < p.luminance_cdm2 < lum_max
    ]
    if not interior:
        raise ValueError(
            "fit_gamma needs at least 1 interior point (0 < input_level < 1, and luminance "
            "strictly between the fitted Lmin/Lmax) to estimate gamma."
        )

    x = np.array([math.log(p.input_level) for p in interior])
    fracs = [(p.luminance_cdm2 - lum_min) / (lum_max - lum_min) for p in interior]
    y = np.array([math.log(f) for f in fracs])

    # Least squares through the origin: y = gamma * x.
    gamma = float(np.sum(x * y) / np.sum(x * x))

    r_squared: float | None = None
    if len(interior) >= 2:
        y_pred = gamma * x
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return GammaFitResult(
        gamma=gamma, lum_min_cdm2=lum_min, lum_max_cdm2=lum_max, r_squared=r_squared
    )


def fit_gamma_lookup(points: Sequence[GammaCalibrationPoint]) -> GammaChannelModel:
    """Build a monotone-interpolated lookup gamma model from raw measured points.

    Uses a PCHIP (shape-preserving piecewise cubic Hermite) interpolator in
    both directions, which is monotone by construction given monotone input
    data, so it never predicts a luminance decrease for an increasing drive
    level (unlike a plain cubic spline) -- appropriate for a display
    response curve, which is physically monotone.

    Args:
        points: Measured (drive level, luminance) samples, at least 2,
            with strictly increasing ``input_level`` after deduplication
            and strictly increasing ``luminance_cdm2`` (a non-monotone
            luminance sequence indicates a measurement problem, not a valid
            display response, and is rejected).

    Returns:
        A :class:`GammaChannelModel` with ``lookup``/``inverse_lookup`` set
        (and ``gamma=None``).

    Raises:
        ValueError: If fewer than 2 distinct points are given, or the
            measured luminances are not strictly increasing with drive
            level.
    """
    by_level: dict[float, float] = {}
    for p in points:
        by_level[p.input_level] = p.luminance_cdm2
    levels = np.array(sorted(by_level))
    if len(levels) < 2:
        raise ValueError("fit_gamma_lookup needs at least 2 distinct measured drive levels.")
    lums = np.array([by_level[lv] for lv in levels])

    if np.any(np.diff(lums) <= 0):
        raise ValueError(
            "Measured luminances must be strictly increasing with drive level "
            "for a monotone lookup model."
        )

    lookup = PchipInterpolator(levels, lums, extrapolate=False)
    inverse_lookup = PchipInterpolator(lums, levels, extrapolate=False)
    return GammaChannelModel(
        lum_min_cdm2=float(lums[0]),
        lum_max_cdm2=float(lums[-1]),
        lookup=lookup,
        inverse_lookup=inverse_lookup,
    )


def linearize(
    desired_lum_fraction: float | FloatArray, model: GammaChannelModel
) -> float | FloatArray:
    """Invert the gamma model: find the drive value giving a target luminance fraction.

    Args:
        desired_lum_fraction: Desired luminance, expressed as a fraction of
            the ``[Lmin, Lmax]`` range (``0.0`` = ``Lmin``, ``1.0`` =
            ``Lmax``), i.e. the linear-light intensity a test wants to
            present. Scalar or array, each in ``[0, 1]``.
        model: The channel's fitted gamma model.

    Returns:
        The drive value(s) in ``[0, 1]`` to write to the frame buffer (or
        gamma-ramp input) to produce ``desired_lum_fraction`` of the
        channel's luminance range. Same shape as ``desired_lum_fraction``.

    Raises:
        ValueError: If any value of ``desired_lum_fraction`` is outside
            ``[0, 1]``.
    """
    frac = np.asarray(desired_lum_fraction, dtype=np.float64)
    if np.any(frac < 0.0) or np.any(frac > 1.0):
        raise ValueError("desired_lum_fraction must be in [0, 1].")

    if model.gamma is not None:
        # frac = v**gamma  =>  v = frac**(1/gamma). 0**(1/gamma) == 0 is fine.
        drive = frac ** (1.0 / model.gamma)
    else:
        assert model.inverse_lookup is not None
        target_lum = model.lum_min_cdm2 + frac * (model.lum_max_cdm2 - model.lum_min_cdm2)
        drive = np.clip(model.inverse_lookup(target_lum), 0.0, 1.0)

    return float(drive) if np.isscalar(desired_lum_fraction) else drive


def make_gamma_ramp(
    models: tuple[GammaChannelModel, GammaChannelModel, GammaChannelModel],
    size: int = 256,
) -> FloatArray:
    """Build a per-channel linearizing gamma ramp for a PsychoPy window/monitor.

    Row ``i`` of the returned ramp gives the drive value (per channel) that
    produces the ``i``-th evenly spaced luminance step across that channel's
    ``[Lmin, Lmax]`` range, i.e. it is the lookup table that, applied as
    ``win.gammaRamp`` (or a ``psychopy.monitors.Monitor`` gamma grid),
    linearizes the display so that nominal pixel value ``i`` maps to a
    linear luminance step -- see PsychoPy's ``Window.gammaRamp`` and
    ``monitors.Monitor.setGammaGrid``/``setLineariseMethod`` documentation
    for how the ramp is consumed; this function only computes the ramp
    values, it never touches a live window (no psychopy import).

    Args:
        models: The three channel models, in (red, green, blue) order.
        size: Number of levels in the ramp (256 for an 8-bit LUT).

    Returns:
        A ``(size, 3)`` float array of drive values in ``[0, 1]``, column
        order (red, green, blue).
    """
    if size < 2:
        raise ValueError(f"size must be at least 2, got {size}")
    levels: FloatArray = np.linspace(0.0, 1.0, size, dtype=np.float64)
    columns = [linearize(levels, model) for model in models]
    return np.stack(columns, axis=1)


class GammaPsychophysicalEstimate(BaseModel):
    """Result of a psychophysical (no-photometer) gamma estimate; see
    :func:`estimate_gamma_psychophysical`.

    Attributes:
        gamma: Point estimate of the power-law exponent.
        ci_low: Lower bound of the confidence interval.
        ci_high: Upper bound of the confidence interval.
        ci_level: Nominal coverage of the interval, e.g. 0.95.
        n_levels: Number of half-luminance bisection matches used.
    """

    model_config = ConfigDict(frozen=True)

    gamma: float = Field(gt=0, description="Point estimate of the power-law exponent.")
    ci_low: float = Field(description="Lower confidence bound.")
    ci_high: float = Field(description="Upper confidence bound.")
    ci_level: float = Field(gt=0, lt=1, description="Nominal CI coverage, e.g. 0.95.")
    n_levels: int = Field(ge=1, description="Number of bisection matches used.")


def estimate_gamma_psychophysical(
    matches: Sequence[tuple[float, float]],
    ci_level: float = 0.95,
) -> GammaPsychophysicalEstimate:
    """Estimate display gamma from half-luminance bisection matches, with no photometer.

    Method (grade B, ``method="psychophysical"`` in
    :class:`vpsych.core.calibration.models.GammaCalibration`): at each of
    several target luminance fractions, the observer adjusts a uniform gray
    patch until its brightness matches a fine dithered checkerboard (or
    spatiotemporal noise pattern) in which a known fraction ``p`` of pixels
    are driven to level 1 (max) and the rest to level 0 (black). Because the
    human visual system pools light *linearly* over the fine checkerboard
    (retinal photoreceptors integrate incident luminous energy, not
    gamma-encoded drive values), the checkerboard's perceived luminance
    equals the linear mixture

        L_checker = Lmin + p * (Lmax - Lmin)

    So a bisection match at drive level ``v_match`` implies
    ``L(v_match) = L_checker``, i.e. (using the power-law gamma model
    normalized so ``v_match**gamma`` is the matched fraction of the
    ``[Lmin, Lmax]`` range)

        v_match**gamma = p   =>   gamma = ln(p) / ln(v_match)

    Given several ``(p, v_match)`` pairs, this fits ``gamma`` by ordinary
    least squares through the origin on ``ln(p) = gamma * ln(v_match)`` (the
    same log-log linearization as :func:`fit_gamma`, without needing
    absolute luminance), and reports a confidence interval from the
    residual scatter across levels (Student's t, ``n - 1`` degrees of
    freedom).

    Args:
        matches: Sequence of ``(target_fraction, matched_drive_level)``
            pairs, each strictly inside ``(0, 1)``: ``target_fraction`` is
            the checkerboard's white-pixel fraction ``p`` (equivalently,
            its linear-mixture luminance fraction), and
            ``matched_drive_level`` is the uniform drive value the observer
            judged equally bright.
        ci_level: Nominal confidence-interval coverage, e.g. 0.95.

    Returns:
        The fitted :class:`GammaPsychophysicalEstimate`.

    Raises:
        ValueError: If fewer than 2 matches are given, or any
            ``target_fraction``/``matched_drive_level`` is not strictly
            inside ``(0, 1)``.
    """
    if len(matches) < 2:
        raise ValueError("estimate_gamma_psychophysical needs at least 2 bisection matches.")
    for p, v in matches:
        if not (0.0 < p < 1.0) or not (0.0 < v < 1.0):
            raise ValueError(
                f"Match (p={p}, v_match={v}) must have both values strictly in (0, 1)."
            )

    x = np.array([math.log(v) for _, v in matches])
    y = np.array([math.log(p) for p, _ in matches])

    gamma = float(np.sum(x * y) / np.sum(x * x))

    n = len(matches)
    residuals = y - gamma * x
    # df = n - 1 (1 free parameter: gamma); degenerate (zero-width CI) below that.
    sigma2 = float(np.sum(residuals**2) / (n - 1)) if n > 2 else 0.0
    sum_x2 = float(np.sum(x * x))
    se = math.sqrt(sigma2 / sum_x2) if sum_x2 > 0 else 0.0
    t_crit = stats.t.ppf(0.5 + ci_level / 2.0, df=n - 1)
    margin = t_crit * se

    return GammaPsychophysicalEstimate(
        gamma=gamma,
        ci_low=gamma - margin,
        ci_high=gamma + margin,
        ci_level=ci_level,
        n_levels=n,
    )
