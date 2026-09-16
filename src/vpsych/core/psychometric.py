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

Phase 0 freezes these signatures; `fit_mle`, `bootstrap_ci`, and
`deviance_gof` raise `NotImplementedError` until a later phase implements
them (likely via `scipy.optimize.minimize` on the negative log-likelihood).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

PsychometricFamily = Literal["weibull", "logistic", "norm_cdf"]
IntensityScale = Literal["log10", "linear"]


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
        raise NotImplementedError


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
    """

    model_config = ConfigDict(frozen=True)

    function: PsychometricFunction = Field(description="The fitted psychometric function.")
    log_likelihood: float = Field(description="Log-likelihood of the data under the fit.")
    n_trials: int = Field(ge=0, description="Total number of trials the fit was based on.")
    converged: bool = Field(description="Whether the optimizer reported convergence.")
    fixed_lapse: bool = Field(description="Whether the lapse rate was held fixed during fitting.")


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
    raise NotImplementedError


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
    """
    raise NotImplementedError


def deviance_gof(
    fit: FitResult,
    intensities: Sequence[float],
    n_correct: Sequence[int],
    n_total: Sequence[int],
) -> GoodnessOfFit:
    """Compute a deviance-based goodness-of-fit statistic for a psychometric function fit.

    Args:
        fit: The fit to assess.
        intensities: Distinct stimulus intensities tested (must match the
            data the fit was based on).
        n_correct: Number of correct responses at each intensity.
        n_total: Number of trials presented at each intensity.

    Returns:
        The deviance, its degrees of freedom, and the associated p-value.
    """
    raise NotImplementedError
