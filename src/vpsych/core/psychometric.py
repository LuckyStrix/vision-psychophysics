"""Psychometric function fitting: MLE, bootstrap confidence intervals, goodness of fit.

A psychometric function relates stimulus intensity to the probability of a
correct (or "yes") response. This module defines the function families used
throughout the suite, maximum-likelihood fitting of their parameters to
trial data, parametric-bootstrap confidence intervals, and a deviance-based
goodness-of-fit statistic.

All fitting operates on trial data already aggregated (or aggregatable) by
intensity level: parallel arrays of tested intensities, number correct, and
number of trials at that intensity. Intensities are expected on whatever
scale the caller chooses -- typically log10 for contrast/coherence-type
dimensions where the underlying psychometric function is closer to linear on
a log axis, or linear for dimensions like logMAR that are already a log
quantity. Callers document which scale they pass via
`PsychometricFunction.intensity_scale`.

`fit_mle` fits by maximum likelihood via `scipy.optimize.minimize`
(L-BFGS-B, bounded, multi-start, with an analytic gradient); `bootstrap_ci`
and `deviance_gof` build on it for parametric-bootstrap confidence intervals
and a Monte-Carlo goodness-of-fit p-value, respectively. See
`docs/METHODS.md`'s "Adaptive procedures and psychometric fitting" section
for the equations, defaults, and citations (Wichmann & Hill, 2001, for the
lapse-rate bound and the Monte-Carlo GOF p-value).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy import stats as _stats
from scipy.optimize import minimize

PsychometricFamily = Literal["weibull", "logistic", "norm_cdf"]
IntensityScale = Literal["log10", "linear"]

#: Lapse-rate bound used when fitting a free lapse parameter (Wichmann & Hill,
#: 2001, "The psychometric function: I. Fitting, sampling, and goodness of
#: fit", Perception & Psychophysics 63(8), 1293-1313). They recommend
#: bounding the lapse rate to a small range, e.g. [0, 0.05], rather than
#: leaving it fully free, because with realistic trial counts an unbounded
#: lapse rate trades off against slope and is poorly identified. We use
#: [0, 0.06] as specified for this project.
LAPSE_RATE_BOUNDS: tuple[float, float] = (0.0, 0.06)


def _base_cdf(family: PsychometricFamily, z: np.ndarray) -> np.ndarray:
    """Evaluate the family's base sigmoid ``F`` at standardized intensity ``z``.

    Every family is parameterized so that ``F(0) == 0.5``: the psychometric
    function's ``threshold`` parameter is therefore *always* the intensity at
    which the base sigmoid is exactly half-way between its 0 and 1
    asymptotes (before the guess/lapse rescaling), regardless of family. This
    keeps ``threshold`` directly comparable across families and matches the
    module-level convention documented above.

    - ``logistic``: the standard logistic CDF, ``1 / (1 + exp(-z))``.
    - ``norm_cdf``: the standard normal CDF, ``Phi(z)``.
    - ``weibull``: ``1 - 2**(-2**z)``, a Gumbel-type extreme-value sigmoid
      that is the classic Weibull-based psychometric function used since
      Watson & Pelli (1983, QUEST) when the intensity axis has already been
      log-transformed (so this family is normally used with
      ``intensity_scale="log10"``). It satisfies ``F(0) == 0.5`` by
      construction (unlike the raw Weibull CDF, whose natural "threshold" is
      the 63.2% point).
    """
    if family == "weibull":
        # Clip before the inner exp2 so extreme z (visited transiently during
        # bounded optimization) can't overflow float64; the outer exp2 of a
        # huge exponent underflows silently to the correct saturating limit.
        z_clipped = np.clip(z, -50.0, 50.0)
        return np.asarray(1.0 - np.exp2(-np.exp2(z_clipped)))
    if family == "logistic":
        return np.asarray(1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0))))
    if family == "norm_cdf":
        return np.asarray(_stats.norm.cdf(z))
    raise ValueError(f"Unknown psychometric family: {family!r}")


def _base_inv_cdf(family: PsychometricFamily, q: np.ndarray) -> np.ndarray:
    """Inverse of :func:`_base_cdf`: standardized intensity ``z`` for base probability ``q``."""
    if family == "weibull":
        return np.asarray(np.log2(-np.log2(1.0 - q)))
    if family == "logistic":
        return np.asarray(np.log(q / (1.0 - q)))
    if family == "norm_cdf":
        return np.asarray(_stats.norm.ppf(q))
    raise ValueError(f"Unknown psychometric family: {family!r}")


class PsychometricFunction(BaseModel):
    """A fitted (or fixed) psychometric function of one intensity dimension.

    The function maps intensity `x` (on `intensity_scale`) to probability
    correct:

        p(x) = guess + (1 - guess - lapse) * F((x - threshold) / slope)

    where `F` is the base sigmoid for `family` (`weibull`'s base function
    for intensities on a log scale, the logistic CDF, or the standard normal
    CDF), `threshold` is the intensity at the function's inflection point,
    and `slope` controls its steepness (interpretation is family-specific;
    document the exact parameterization used at implementation time,
    e.g. matching `questplus`/Prins & Kingdom's Palamedes conventions).

    Attributes:
        family: Which sigmoid family this function uses.
        threshold: Threshold parameter, in `intensity_scale` units.
        slope: Slope (steepness) parameter, in `intensity_scale` units.
        guess: Guess rate (lower asymptote), fixed by task design (e.g.
            `0.5` for 2AFC), in [0, 1).
        lapse: Lapse rate (upper-asymptote miss rate independent of
            intensity), in [0, 1).
        intensity_scale: Whether `threshold`/`slope` (and the `x` passed to
            `p_correct`) are on a `"log10"` or `"linear"` intensity scale.
    """

    model_config = ConfigDict(frozen=True)

    family: PsychometricFamily = Field(description="Sigmoid family.")
    threshold: float = Field(description="Threshold parameter, in intensity_scale units.")
    slope: float = Field(gt=0, description="Slope (steepness) parameter, in intensity_scale units.")
    guess: float = Field(
        ge=0, lt=1, description="Guess rate (lower asymptote), fixed by task design."
    )
    lapse: float = Field(ge=0, lt=1, description="Lapse rate (upper-asymptote miss rate).")
    intensity_scale: IntensityScale = Field(description="Scale threshold/slope are expressed on.")

    def p_correct(self, x: float) -> float:
        """Evaluate the fitted psychometric function at intensity `x`.

        Args:
            x: Stimulus intensity, on `intensity_scale`.

        Returns:
            Predicted probability correct at `x`, in [guess, 1 - lapse].
        """
        z = np.asarray((x - self.threshold) / self.slope, dtype=float)
        f = _base_cdf(self.family, z)
        p = self.guess + (1.0 - self.guess - self.lapse) * f
        return float(p)


class FitResult(BaseModel):
    """Result of fitting a :class:`PsychometricFunction` to trial data by MLE.

    Attributes:
        function: The fitted psychometric function.
        log_likelihood: Log-likelihood of the data under the fitted
            parameters.
        n_trials: Total number of trials the fit was based on.
        converged: Whether the underlying optimizer reported convergence.
        fixed_lapse: Whether the lapse rate was held fixed during fitting
            (per the `fix_lapse` argument to `fit_mle`) rather than
            estimated.
        design_intensities: The distinct intensities the fit was based on
            (echo of `fit_mle`'s `intensities` argument). Not part of the
            original Phase 0 freeze; added so `bootstrap_ci` and
            `deviance_gof` can parametrically resample at the same design
            points without the caller having to pass them again. Defaults to
            an empty tuple for backward compatibility with any code
            constructing `FitResult` directly.
        design_n_total: The trial counts at each of `design_intensities`
            (echo of `fit_mle`'s `n_total` argument). See `design_intensities`.
    """

    model_config = ConfigDict(frozen=True)

    function: PsychometricFunction = Field(description="The fitted psychometric function.")
    log_likelihood: float = Field(description="Log-likelihood of the data under the fit.")
    n_trials: int = Field(ge=0, description="Total number of trials the fit was based on.")
    converged: bool = Field(description="Whether the optimizer reported convergence.")
    fixed_lapse: bool = Field(description="Whether the lapse rate was held fixed during fitting.")
    design_intensities: tuple[float, ...] = Field(
        default_factory=tuple,
        description="Distinct intensities the fit was based on (for resampling).",
    )
    design_n_total: tuple[int, ...] = Field(
        default_factory=tuple,
        description="Trial counts at each of design_intensities (for resampling).",
    )


class BootstrapCIResult(BaseModel):
    """Bootstrap confidence interval for one parameter of a fitted psychometric function.

    Attributes:
        parameter: Name of the parameter this interval describes, e.g.
            `"threshold"`.
        point_estimate: The original (non-bootstrapped) fitted value.
        ci_low: Lower bound of the bootstrap confidence interval.
        ci_high: Upper bound of the bootstrap confidence interval.
        ci_level: Nominal coverage of the interval, e.g. `0.95`.
        n_boot: Number of bootstrap resamples used.
    """

    model_config = ConfigDict(frozen=True)

    parameter: str = Field(description="Name of the parameter this interval describes.")
    point_estimate: float = Field(description="Original (non-bootstrapped) fitted value.")
    ci_low: float = Field(description="Lower confidence bound.")
    ci_high: float = Field(description="Upper confidence bound.")
    ci_level: float = Field(gt=0, lt=1, description="Nominal CI coverage.")
    n_boot: int = Field(gt=0, description="Number of bootstrap resamples used.")


class GoodnessOfFit(BaseModel):
    """Deviance-based goodness-of-fit summary for a psychometric function fit.

    Attributes:
        deviance: The fit's deviance statistic (2x log-likelihood-ratio
            between the fitted model and the saturated model).
        df: Degrees of freedom for the deviance's reference chi-squared
            distribution (number of intensity levels minus number of free
            parameters).
        p_value: p-value of `deviance` under a chi-squared(df) reference
            distribution; a small p-value indicates a poor fit.
    """

    model_config = ConfigDict(frozen=True)

    deviance: float = Field(ge=0, description="Deviance statistic.")
    df: int = Field(ge=0, description="Degrees of freedom for the chi-squared reference.")
    p_value: float = Field(ge=0, le=1, description="p-value under chi-squared(df).")


def _base_pdf(family: PsychometricFamily, z: np.ndarray) -> np.ndarray:
    """Derivative dF/dz of :func:`_base_cdf`'s base sigmoid, used by `_negloglik_grad`."""
    if family == "weibull":
        z_clipped = np.clip(z, -50.0, 50.0)
        u = np.exp2(z_clipped)
        return np.asarray((np.log(2.0) ** 2) * u * np.exp2(-u))
    if family == "logistic":
        f = _base_cdf(family, z)
        return np.asarray(f * (1.0 - f))
    if family == "norm_cdf":
        return np.asarray(_stats.norm.pdf(z))
    raise ValueError(f"Unknown psychometric family: {family!r}")


def _negloglik(
    params: np.ndarray,
    x: np.ndarray,
    k: np.ndarray,
    n: np.ndarray,
    guess: float,
    family: PsychometricFamily,
    free_lapse: bool,
    fix_lapse: float | None,
) -> float:
    """Negative log-likelihood of binomial trial data under the psychometric model."""
    if free_lapse:
        threshold, slope, lapse = params
    else:
        threshold, slope = params
        lapse = fix_lapse if fix_lapse is not None else 0.0
    slope = max(float(slope), 1e-8)
    z = (x - threshold) / slope
    f = _base_cdf(family, z)
    p = guess + (1.0 - guess - lapse) * f
    p = np.clip(p, 1e-10, 1.0 - 1e-10)
    ll = np.sum(k * np.log(p) + (n - k) * np.log(1.0 - p))
    return float(-ll)


def _negloglik_grad(
    params: np.ndarray,
    x: np.ndarray,
    k: np.ndarray,
    n: np.ndarray,
    guess: float,
    family: PsychometricFamily,
    free_lapse: bool,
    fix_lapse: float | None,
) -> np.ndarray:
    """Analytic gradient of `_negloglik`.

    Supplying this to `scipy.optimize.minimize` (instead of letting it fall
    back to finite differences) is a large, measured speedup for this
    project's usage pattern -- `bootstrap_ci`/`deviance_gof` call
    `_refit_single` (which uses this gradient) thousands of times per CI, so
    the per-call cost matters. Derivation: with `p = guess + (1 - guess -
    lapse) * F(z)`, `z = (x - threshold) / slope`, and
    `d(-loglik)/dp = -(k/p) + (n-k)/(1-p)` (the usual binomial score), the
    chain rule gives `dp/dthreshold = -(1-guess-lapse)*f(z)/slope`,
    `dp/dslope = -(1-guess-lapse)*f(z)*z/slope`, and (when lapse is free)
    `dp/dlapse = -F(z)`, where `f = dF/dz` is `_base_pdf`.
    """
    if free_lapse:
        threshold, slope, lapse = params
    else:
        threshold, slope = params
        lapse = fix_lapse if fix_lapse is not None else 0.0
    slope_safe = max(float(slope), 1e-8)
    z = (x - threshold) / slope_safe
    big_f = _base_cdf(family, z)
    small_f = _base_pdf(family, z)
    scale = 1.0 - guess - lapse
    p = np.clip(guess + scale * big_f, 1e-10, 1.0 - 1e-10)

    dnll_dp = -(k / p) + (n - k) / (1.0 - p)
    dp_dthreshold = scale * small_f * (-1.0 / slope_safe)
    dp_dslope = scale * small_f * (-z / slope_safe)

    grad_threshold = float(np.sum(dnll_dp * dp_dthreshold))
    grad_slope = float(np.sum(dnll_dp * dp_dslope))
    if free_lapse:
        dp_dlapse = -big_f
        grad_lapse = float(np.sum(dnll_dp * dp_dlapse))
        return np.array([grad_threshold, grad_slope, grad_lapse])
    return np.array([grad_threshold, grad_slope])


def _compute_bounds(x: np.ndarray) -> tuple[float, float, float, float]:
    """Data-driven optimizer bounds for (threshold_lo, threshold_hi, slope_lo, slope_hi)."""
    x_range = float(np.ptp(x)) if len(x) > 1 else 0.0
    if x_range <= 0:
        x_range = max(abs(float(np.mean(x))), 1.0) * 0.5
    thresh_lo = float(np.min(x) - 2 * x_range - 1e-6)
    thresh_hi = float(np.max(x) + 2 * x_range + 1e-6)
    slope_lo = max(x_range * 1e-3, 1e-6)
    slope_hi = max(x_range * 10.0, 1.0)
    return thresh_lo, thresh_hi, slope_lo, slope_hi


def _refit_single(
    x: np.ndarray,
    k: np.ndarray,
    n: np.ndarray,
    family: PsychometricFamily,
    guess: float,
    fix_lapse: float | None,
    x0: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Single-start MLE refit from `x0`, used for fast bootstrap/Monte-Carlo resampling.

    A single start initialized at a nearby (e.g. the original data's) MLE
    converges reliably for resampled data close to the original, and is much
    faster than the multi-start search `fit_mle` uses for the initial fit --
    this is what keeps `bootstrap_ci`/`deviance_gof` fast enough for
    thousands of resamples.
    """
    free_lapse = fix_lapse is None
    thresh_lo, thresh_hi, slope_lo, slope_hi = _compute_bounds(x)
    threshold0 = float(np.clip(x0[0], thresh_lo, thresh_hi))
    slope0 = float(np.clip(x0[1], slope_lo, slope_hi))
    if free_lapse:
        lapse0 = float(np.clip(x0[2], LAPSE_RATE_BOUNDS[0], LAPSE_RATE_BOUNDS[1]))
        bounds = [(thresh_lo, thresh_hi), (slope_lo, slope_hi), LAPSE_RATE_BOUNDS]
        start = [threshold0, slope0, lapse0]
    else:
        bounds = [(thresh_lo, thresh_hi), (slope_lo, slope_hi)]
        start = [threshold0, slope0]
    result = minimize(
        _negloglik,
        x0=start,
        args=(x, k, n, guess, family, free_lapse, fix_lapse),
        method="L-BFGS-B",
        jac=_negloglik_grad,
        bounds=bounds,
    )
    if free_lapse:
        threshold, slope, lapse = (float(v) for v in result.x)
    else:
        threshold, slope = (float(v) for v in result.x)
        lapse = float(fix_lapse) if fix_lapse is not None else 0.0
    return threshold, slope, lapse


def fit_mle(
    intensities: Sequence[float],
    n_correct: Sequence[int],
    n_total: Sequence[int],
    family: PsychometricFamily,
    guess: float,
    fix_lapse: float | None = None,
    intensity_scale: IntensityScale = "log10",
) -> FitResult:
    """Fit a psychometric function to binomial trial data by maximum likelihood.

    Args:
        intensities: Distinct stimulus intensities tested, on
            `intensity_scale`. Must be the same length as `n_correct` and
            `n_total`.
        n_correct: Number of correct responses at each intensity in
            `intensities`.
        n_total: Number of trials presented at each intensity in
            `intensities`.
        family: Sigmoid family to fit (see :class:`PsychometricFunction`).
        guess: Fixed guess (chance/floor) rate, e.g. `0.5` for 2AFC. Not
            estimated.
        fix_lapse: If given, the lapse rate is held fixed at this value
            during fitting rather than estimated as a free parameter (this
            is common practice when trial counts are too small to estimate
            lapse reliably; use catch-trial data instead in that case, see
            `vpsych.data.schemas.TestSummary.catch_lapse_rate`).
        intensity_scale: Scale `intensities` are expressed on; propagated to
            the fitted `PsychometricFunction.intensity_scale`.

    Returns:
        The fitted function and fit diagnostics.

    Raises:
        ValueError: If input array lengths mismatch, if any `n_correct` is
            greater than the corresponding `n_total`, or if `intensities` is
            empty.
    """
    x = np.asarray(intensities, dtype=float)
    k = np.asarray(n_correct, dtype=float)
    n = np.asarray(n_total, dtype=float)
    if not (len(x) == len(k) == len(n)):
        raise ValueError("intensities, n_correct, and n_total must have the same length")
    if len(x) == 0:
        raise ValueError("intensities must not be empty")
    if np.any(k > n):
        raise ValueError("n_correct cannot exceed n_total at the same intensity")
    if np.any(n < 0) or np.any(k < 0):
        raise ValueError("n_correct and n_total must be non-negative")

    free_lapse = fix_lapse is None
    thresh_lo, thresh_hi, slope_lo, slope_hi = _compute_bounds(x)
    x_range = thresh_hi - thresh_lo  # a conservative proxy for the data spread

    if free_lapse:
        bounds: list[tuple[float, float]] = [
            (thresh_lo, thresh_hi),
            (slope_lo, slope_hi),
            LAPSE_RATE_BOUNDS,
        ]
    else:
        bounds = [(thresh_lo, thresh_hi), (slope_lo, slope_hi)]

    # A modest grid of starting points, not an exhaustive one: with an
    # analytic gradient (see `_negloglik_grad`) L-BFGS-B converges reliably
    # from any of these for the smooth, low-dimensional (2-3 parameter)
    # likelihoods this module fits, so a small grid both avoids local minima
    # in practice and keeps `fit_mle` fast enough to call thousands of times
    # in `bootstrap_ci`/`deviance_gof` (via `_refit_single`, which uses a
    # single start instead -- see that function).
    threshold_starts = np.unique([np.min(x), np.mean(x), np.max(x)])
    slope_starts = [max(x_range / 6, slope_lo), max(x_range / 2, slope_lo)]
    lapse_starts: list[float]
    if free_lapse:
        lapse_starts = [0.01, 0.04]
    else:
        assert fix_lapse is not None
        lapse_starts = [fix_lapse]

    best = None
    for t0 in threshold_starts:
        for s0 in slope_starts:
            for l0 in lapse_starts:
                x0 = [float(t0), float(s0), float(l0)] if free_lapse else [float(t0), float(s0)]
                result = minimize(
                    _negloglik,
                    x0=x0,
                    args=(x, k, n, guess, family, free_lapse, fix_lapse),
                    method="L-BFGS-B",
                    jac=_negloglik_grad,
                    bounds=bounds,
                )
                if best is None or result.fun < best.fun:
                    best = result

    assert best is not None  # loop always runs at least once
    if free_lapse:
        threshold, slope, lapse = (float(v) for v in best.x)
    else:
        threshold, slope = (float(v) for v in best.x)
        lapse = float(fix_lapse) if fix_lapse is not None else 0.0

    function = PsychometricFunction(
        family=family,
        threshold=threshold,
        slope=slope,
        guess=guess,
        lapse=lapse,
        intensity_scale=intensity_scale,
    )
    return FitResult(
        function=function,
        log_likelihood=float(-best.fun),
        n_trials=int(np.sum(n)),
        converged=bool(best.success),
        fixed_lapse=not free_lapse,
        design_intensities=tuple(float(v) for v in x),
        design_n_total=tuple(int(v) for v in n),
    )


def bootstrap_ci(
    fit: FitResult,
    n_boot: int = 1000,
    level: float = 0.95,
    rng: np.random.Generator | None = None,
    parameter: str = "threshold",
) -> BootstrapCIResult:
    """Compute a parametric bootstrap confidence interval for a fitted parameter.

    Resamples binomial trial outcomes from the fitted psychometric function
    at the original design intensities/trial counts, refits by MLE for each
    resample, and takes the empirical percentile interval of the resampled
    parameter estimates.

    Args:
        fit: The original fit to bootstrap around (provides the fitted
            function and the design intensities/trial counts implicitly
            needed to resample -- concrete implementation must retain or
            receive these; see implementation notes for exact call shape).
        n_boot: Number of bootstrap resamples.
        level: Nominal coverage of the interval, e.g. `0.95` for a 95% CI.
        rng: Random generator used to draw resamples (see
            `vpsych.core.rng.make_rng`). If `None`, an implementation may
            create its own (unseeded) generator, which will *not* be
            reproducible -- callers requiring reproducibility must pass one.
        parameter: Which fitted parameter to compute the interval for, e.g.
            `"threshold"` or `"slope"`.

    Returns:
        The bootstrap confidence interval for `parameter`.

    Raises:
        ValueError: If `fit` carries no design data to resample from (i.e.
            `fit.design_intensities` is empty -- this happens only if `fit`
            was constructed by hand rather than returned by `fit_mle`), or if
            `parameter` is not one of `"threshold"`, `"slope"`, `"lapse"`.
    """
    if parameter not in ("threshold", "slope", "lapse"):
        raise ValueError(f"Unknown parameter {parameter!r}; expected threshold/slope/lapse")
    x = np.asarray(fit.design_intensities, dtype=float)
    n = np.asarray(fit.design_n_total, dtype=float)
    if len(x) == 0:
        raise ValueError(
            "fit has no stored design_intensities/design_n_total to resample from "
            "(was it constructed by fit_mle?)"
        )
    if rng is None:
        rng = np.random.default_rng()

    fn = fit.function
    p_hat = np.clip(np.array([fn.p_correct(float(xi)) for xi in x]), 0.0, 1.0)
    fix_lapse = fn.lapse if fit.fixed_lapse else None
    x0 = (fn.threshold, fn.slope, fn.lapse)
    n_int = n.astype(int)

    values = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        k_b = rng.binomial(n_int, p_hat).astype(float)
        threshold_b, slope_b, lapse_b = _refit_single(x, k_b, n, fn.family, fn.guess, fix_lapse, x0)
        values[b] = {"threshold": threshold_b, "slope": slope_b, "lapse": lapse_b}[parameter]

    alpha = 1.0 - level
    ci_low, ci_high = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    point_estimate = float(getattr(fn, parameter))
    return BootstrapCIResult(
        parameter=parameter,
        point_estimate=point_estimate,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        ci_level=level,
        n_boot=n_boot,
    )


def bootstrap_ci_at_p_correct(
    fit: FitResult,
    p_target: float,
    n_boot: int = 1000,
    level: float = 0.95,
    rng: np.random.Generator | None = None,
) -> BootstrapCIResult:
    """Full-refit parametric bootstrap CI for the intensity at a target %-correct.

    Unlike `bootstrap_ci(fit, parameter="threshold")` followed by converting
    just the resulting threshold interval via `intensity_at_p_correct` (a
    documented simplification `ConstantStimuli`/`WeightedStaircase`'s
    `estimate()` both historically used, which holds `slope`/`lapse` fixed
    at their *original* point estimates and so only propagates uncertainty
    in `threshold`), this refits **all three** parameters
    (`threshold`/`slope`/`lapse`) on every resample -- exactly like
    `bootstrap_ci` already does internally (see `_refit_single`) -- and
    evaluates `intensity_at_p_correct` using *that resample's own* fitted
    function, not the original one. This propagates slope/lapse
    resample-to-resample uncertainty into the interval too, which matters
    most for a design (like a staircase) whose trials concentrate narrowly
    around threshold, leaving slope poorly identified: holding slope fixed
    at a single point estimate then understates how uncertain the
    target-percent intensity really is.

    Args:
        fit: The original fit to bootstrap around (must carry
            `design_intensities`/`design_n_total`, i.e. returned by
            `fit_mle`).
        p_target: Target probability correct to evaluate
            `intensity_at_p_correct` at on every resample, e.g. the
            staircase's own `target_p_correct`.
        n_boot: Number of bootstrap resamples.
        level: Nominal coverage of the interval, e.g. `0.95`.
        rng: Random generator used to draw resamples. If `None`, an
            implementation may create its own (unseeded) generator, which
            will *not* be reproducible -- callers requiring reproducibility
            must pass one.

    Returns:
        A `BootstrapCIResult` with `parameter="intensity_at_p_correct"` and
        `point_estimate` the original fit's own
        `intensity_at_p_correct(fit.function, p_target)`.

    Raises:
        ValueError: If `fit` carries no design data to resample from.
    """
    x = np.asarray(fit.design_intensities, dtype=float)
    n = np.asarray(fit.design_n_total, dtype=float)
    if len(x) == 0:
        raise ValueError(
            "fit has no stored design_intensities/design_n_total to resample from "
            "(was it constructed by fit_mle?)"
        )
    if rng is None:
        rng = np.random.default_rng()

    fn = fit.function
    p_hat = np.clip(np.array([fn.p_correct(float(xi)) for xi in x]), 0.0, 1.0)
    fix_lapse = fn.lapse if fit.fixed_lapse else None
    x0 = (fn.threshold, fn.slope, fn.lapse)
    n_int = n.astype(int)

    values = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        k_b = rng.binomial(n_int, p_hat).astype(float)
        threshold_b, slope_b, lapse_b = _refit_single(x, k_b, n, fn.family, fn.guess, fix_lapse, x0)
        resampled_fn = fn.model_copy(
            update={"threshold": threshold_b, "slope": slope_b, "lapse": lapse_b}
        )
        values[b] = intensity_at_p_correct(resampled_fn, p_target)

    alpha = 1.0 - level
    ci_low, ci_high = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    point_estimate = intensity_at_p_correct(fn, p_target)
    return BootstrapCIResult(
        parameter="intensity_at_p_correct",
        point_estimate=point_estimate,
        ci_low=float(min(ci_low, ci_high)),
        ci_high=float(max(ci_low, ci_high)),
        ci_level=level,
        n_boot=n_boot,
    )


def _deviance(k: np.ndarray, n: np.ndarray, p_hat: np.ndarray) -> float:
    """Deviance of fitted probabilities `p_hat` against binomial data `(k, n)`.

    ``D = 2 * sum(k*log(k/(n*p_hat)) + (n-k)*log((n-k)/(n*(1-p_hat))))``, the
    log-likelihood-ratio statistic between the fitted model and the
    saturated model (which fits each intensity level's observed proportion
    exactly). Terms with `k == 0` or `k == n` are defined as 0 (their limit),
    matching the usual convention.
    """
    p_hat = np.clip(p_hat, 1e-10, 1.0 - 1e-10)
    with np.errstate(divide="ignore", invalid="ignore"):
        term1 = np.where(
            k > 0,
            k * np.log(np.divide(k, n * p_hat, out=np.ones_like(k), where=(n * p_hat) > 0)),
            0.0,
        )
        term2 = np.where(
            k < n,
            (n - k)
            * np.log(
                np.divide(n - k, n * (1 - p_hat), out=np.ones_like(k), where=(n * (1 - p_hat)) > 0)
            ),
            0.0,
        )
    return float(2.0 * np.sum(term1 + term2))


def deviance_gof(
    fit: FitResult,
    intensities: Sequence[float],
    n_correct: Sequence[int],
    n_total: Sequence[int],
    n_mc: int = 500,
    rng: np.random.Generator | None = None,
) -> GoodnessOfFit:
    """Compute a deviance-based goodness-of-fit statistic for a psychometric function fit.

    The p-value is a parametric-bootstrap Monte-Carlo p-value (Wichmann &
    Hill, 2001, section on goodness of fit): `n_mc` datasets are simulated
    from the *fitted* model at the original design intensities/trial counts,
    each is refit by MLE, and the p-value is the fraction of simulated
    deviances (each simulated dataset's deviance against its own refit) that
    are at least as large as the observed deviance. This is preferred over
    the nominal chi-squared(df) reference for psychometric-function fits,
    whose asymptotics are often poor at typical trial counts.

    Args:
        fit: The fit to assess.
        intensities: Distinct stimulus intensities tested (must match the
            data the fit was based on).
        n_correct: Number of correct responses at each intensity.
        n_total: Number of trials presented at each intensity.
        n_mc: Number of Monte-Carlo simulated datasets used to compute the
            p-value. Not part of the Phase 0 freeze; added (with a default)
            to support the Monte-Carlo p-value this function documents.
        rng: Random generator used to simulate datasets (see
            `vpsych.core.rng.make_rng`). Not part of the Phase 0 freeze (the
            frozen signature had no way to seed the required Monte-Carlo
            simulation); added (with a default of `None`, meaning an
            unseeded generator) for reproducibility. Callers requiring
            reproducible p-values must pass one.

    Returns:
        The deviance, its degrees of freedom (informational, following the
        nominal chi-squared convention: number of intensity levels minus
        number of free parameters), and the Monte-Carlo p-value.
    """
    x = np.asarray(intensities, dtype=float)
    k = np.asarray(n_correct, dtype=float)
    n = np.asarray(n_total, dtype=float)
    if not (len(x) == len(k) == len(n)):
        raise ValueError("intensities, n_correct, and n_total must have the same length")
    if len(x) == 0:
        raise ValueError("intensities must not be empty")

    fn = fit.function
    p_hat = np.array([fn.p_correct(float(xi)) for xi in x])
    observed_deviance = _deviance(k, n, p_hat)

    n_free_params = 2 + (0 if fit.fixed_lapse else 1)  # threshold, slope, [lapse]
    df = max(int(len(x) - n_free_params), 0)

    if rng is None:
        rng = np.random.default_rng()
    fix_lapse = fn.lapse if fit.fixed_lapse else None
    x0 = (fn.threshold, fn.slope, fn.lapse)
    n_int = n.astype(int)
    p_hat_clipped = np.clip(p_hat, 0.0, 1.0)

    sim_deviances = np.empty(n_mc, dtype=float)
    for i in range(n_mc):
        k_sim = rng.binomial(n_int, p_hat_clipped).astype(float)
        threshold_s, slope_s, lapse_s = _refit_single(
            x, k_sim, n, fn.family, fn.guess, fix_lapse, x0
        )
        z_s = (x - threshold_s) / slope_s
        p_sim_hat = fn.guess + (1.0 - fn.guess - lapse_s) * _base_cdf(fn.family, z_s)
        sim_deviances[i] = _deviance(k_sim, n, p_sim_hat)

    p_value = float(np.mean(sim_deviances >= observed_deviance))
    return GoodnessOfFit(deviance=observed_deviance, df=df, p_value=p_value)


def intensity_at_p_correct(function: PsychometricFunction, p_target: float) -> float:
    """Invert a fitted psychometric function to find the intensity at a target %-correct.

    `PsychometricFunction.threshold` is always the F=0.5 point of the base
    sigmoid (see `_base_cdf`), which generally does *not* equal the
    intensity at a task's conventional criterion (e.g. 75% correct in 2AFC).
    This helper converts between the two: it solves
    ``p_target = guess + (1 - guess - lapse) * F((x - threshold) / slope)``
    for `x`.

    Not part of the original Phase 0 freeze; added because several
    procedures (method of constant stimuli in particular) need to report a
    threshold at a configurable target %-correct rather than always at the
    F=0.5 point.

    Args:
        function: The fitted (or fixed) psychometric function.
        p_target: Target probability correct, in
            `(function.guess, 1 - function.lapse)`.

    Returns:
        The intensity at which `function.p_correct` equals `p_target`, on
        `function.intensity_scale`.
    """
    denom = 1.0 - function.guess - function.lapse
    q = (p_target - function.guess) / denom if denom > 0 else 0.5
    q = float(np.clip(q, 1e-12, 1.0 - 1e-12))
    z = float(_base_inv_cdf(function.family, np.asarray(q)))
    return function.threshold + function.slope * z
