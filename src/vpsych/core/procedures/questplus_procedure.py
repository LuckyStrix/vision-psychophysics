"""QUEST+ adaptive procedure, wrapping the `questplus` package.

Implements :class:`vpsych.core.procedures.base.AdaptiveProcedure` for a
single-parameter (threshold) QUEST+ run. QUEST+ (Watson, A. B. (2017). QUEST+:
A general multidimensional Bayesian adaptive psychometric method. Journal of
Vision, 17(3):10) generalizes QUEST to arbitrary psychometric-function
families and stimulus/response spaces by maintaining a full posterior over
parameters and selecting each next stimulus to maximize expected information
gain (minimize expected posterior entropy).

Mapping onto the `questplus` package
-------------------------------------
`questplus.QuestPlus` ships a fixed set of built-in psychometric-function
families: `"weibull"`, `"csf"`, `"norm_cdf"`, `"norm_cdf_2"`,
`"thurstone_scaling"` -- notably, no `"logistic"`. This wrapper therefore
supports `function="weibull"` and `function="norm_cdf"`; `function="logistic"`
raises `ValueError` at construction (use `ConstantStimuli` or
`WeightedStaircase` instead, whose threshold estimation goes through
`vpsych.core.psychometric.fit_mle`, which *does* support all three families).

- `function="weibull"`: maps directly onto `questplus`'s `"weibull"` func,
  whose `threshold`/`slope`/`lapse_rate` parameter names match this
  project's `threshold_values`/`slope_values`/`lapse_rate_values`. The guess
  rate is fixed (not searched) by passing it as a singleton
  `lower_asymptote` parameter-domain entry (`[guess_rate]`) -- `questplus`
  has no separate "fixed nuisance parameter" concept, so a size-1 grid
  dimension is the standard way to pin a parameter's value while still
  letting it appear in every likelihood evaluation.
- `function="norm_cdf"`: maps onto `questplus`'s `"norm_cdf"` func, whose
  parameters are named `mean`/`sd` rather than `threshold`/`slope` (mapped
  here) and which only supports `stim_scale="linear"` (it raises otherwise);
  the `stim_scale` constructor argument is therefore ignored (forced to
  `"linear"`) when `function="norm_cdf"`.

Not part of the Phase 0 freeze: `stim_scale` (questplus needs an actual
scale, not just the free-text `intensity_units` label) and
`param_estimation_method` (`"mean"` or `"mode"`, per the Phase 1a task's
"posterior mean (or mode, configurable)" requirement) are added as optional
keyword arguments with defaults (`"log10"`, `"mean"`) preserving
call-compatibility with the frozen signature.

Criterion conversion pitfall
-----------------------------
`estimate().value` is `questplus`'s own native Weibull `threshold`
parameter -- the intensity at which *its* formula (`function="weibull"`,
`stim_scale="log10"`)

    p(x) = 1 - lapse - (1 - guess - lapse) * exp(-10**(slope * (x - threshold)))

evaluates to `guess + (1 - guess - lapse) * (1 - exp(-1))`, i.e. the
*classic* ~63.2%-of-the-way-from-guess-to-ceiling point. This is a
genuinely different functional form from `vpsych.core.psychometric`'s own
`"weibull"` family (a Gumbel-type sigmoid on radix 2, rescaled so
`F(0) == 0.5` -- see that module's docstring), whose `threshold` sits at
the halfway point instead. **They are not interchangeable.** Passing a
`ThresholdEstimate` from this procedure into
`vpsych.core.psychometric.intensity_at_p_correct` (which assumes the
`vpsych.core.psychometric` family) silently returns the wrong intensity for
any `p_target != questplus`'s own ~63.2% anchor.

To report this procedure's threshold at a different, conventional
%-correct criterion (e.g. 75% for 2AFC, or the "FrACT" midpoint
`guess + 0.5 * (1 - guess - lapse)` used to match `vpsych.core.psychometric`
`F(0) = 0.5`-equivalent thresholds), invert `questplus`'s *own* formula
directly via `questplus_weibull_x_at_p` (module-level function below) or
`QuestPlusProcedure.intensity_at_p_correct` (instance method, reads the
fitted slope/lapse off `self.estimate()`) -- never
`vpsych.core.psychometric.intensity_at_p_correct`. See
`docs/METHODS.md`'s "Adaptive procedures" section for the same pitfall
written up for readers of the methods doc.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
from questplus import QuestPlus

from vpsych.core.procedures.base import ThresholdEstimate


def questplus_weibull_x_at_p(
    threshold: float, slope: float, guess: float, lapse: float, p_target: float
) -> float:
    """Invert `questplus`'s own log10-scale Weibull formula for `x` at a target p-correct.

    `questplus`'s `weibull` psychometric function (`scale="log10"`, Watson &
    Pelli 1983's original QUEST parameterization; see
    `questplus.psychometric_function.weibull`) is

        p(x) = 1 - lapse - (1 - guess - lapse) * exp(-10**(slope * (x - threshold)))

    Solving for `x`:

        x = threshold + log10(-ln((1 - p_target - lapse) / (1 - guess - lapse))) / slope

    This is a *different* functional form from `vpsych.core.psychometric`'s
    own weibull family (rescaled so `F(0) = 0.5`; see that module's
    docstring), so it must be inverted here, directly against `questplus`'s
    own formula, rather than via
    `vpsych.core.psychometric.intensity_at_p_correct`, which assumes that
    other parameterization and would give a silently wrong answer applied
    to a `QuestPlusProcedure` fit -- see this module's "Criterion
    conversion pitfall" docstring section.

    This is the one shared, tested implementation of that inversion; every
    `vpsych` test built on `QuestPlusProcedure` (`visual_acuity`,
    `vernier_acuity`, `letter_contrast_sensitivity`, `color_discrimination`,
    `motion_coherence`, `critical_flicker_fusion`) should use this function
    (typically via `QuestPlusProcedure.intensity_at_p_correct`) rather than
    re-deriving or duplicating it locally.

    Args:
        threshold: `QuestPlusProcedure`'s own (native, ~63.2%-point)
            threshold estimate.
        slope: `QuestPlusProcedure`'s own slope estimate.
        guess: Fixed guess (chance) rate.
        lapse: `QuestPlusProcedure`'s own lapse-rate estimate.
        p_target: Target probability correct to solve for.

    Returns:
        The intensity `x` at which `p(x) == p_target`.
    """
    denom = 1.0 - guess - lapse
    q = (1.0 - p_target - lapse) / denom if denom > 0 else 0.5
    q = min(max(q, 1e-12), 1.0 - 1e-12)
    return threshold + math.log10(-math.log(q)) / slope


def questplus_weibull_report_at_p(
    estimate: ThresholdEstimate, guess: float, p_target: float
) -> tuple[float, float, float]:
    """Convert a `QuestPlusProcedure` `ThresholdEstimate` to the intensity at `p_target` correct.

    Wraps `questplus_weibull_x_at_p` for the common case of converting a whole
    `estimate()` result (point estimate *and* credible interval) at once: the
    CI bounds are shifted by the same amount as the point estimate, holding
    the fitted slope/lapse rate fixed at their point estimates -- the same
    approximation `ConstantStimuli`/`WeightedStaircase` use to convert a
    bootstrap CI on their own natural threshold point to one at an arbitrary
    target proportion correct (see `docs/METHODS.md`'s "Adaptive procedures"
    section).

    Args:
        estimate: A `QuestPlusProcedure.estimate()` result (must carry
            `extra["slope"]`/`extra["lapse_rate"]`, as `QuestPlusProcedure`'s
            own `estimate()` always does).
        guess: Fixed guess (chance) rate.
        p_target: Target probability correct to solve for.

    Returns:
        `(reported_value, ci_low, ci_high)`, all on `estimate`'s own units.
    """
    slope = float(estimate.extra["slope"])
    lapse = float(estimate.extra["lapse_rate"])
    reported = questplus_weibull_x_at_p(estimate.value, slope, guess, lapse, p_target)
    shift = reported - estimate.value
    return reported, estimate.ci_low + shift, estimate.ci_high + shift


PsychometricFamily = Literal["weibull", "logistic", "norm_cdf"]


def _equal_tailed_credible_interval(
    values: np.ndarray, probs: np.ndarray, level: float
) -> tuple[float, float]:
    """Equal-tailed credible interval from a discretized marginal posterior.

    Replaces a normal approximation (`mean +/- z * sd` on the posterior),
    which is a poor fit whenever the marginal posterior is skewed or
    truncated by the parameter grid's edges -- both common for QUEST+ with
    realistic trial counts, and the reason the normal-approximation CI's
    empirical coverage was measured well below its nominal level (see
    `tests/procedures/test_questplus_procedure.py`). Instead this takes the
    `alpha/2` and `1 - alpha/2` quantiles directly off the posterior's
    (normalized) CDF over `values`, linearly interpolating between grid
    points -- valid for any posterior shape, including skewed or
    edge-truncated ones.

    Args:
        values: Sorted or unsorted grid values the posterior is defined
            over (e.g. `QuestPlus.param_domain[key]`).
        probs: Posterior probability mass at each of `values` (e.g.
            `QuestPlus.marginal_posterior[key]`); need not already sum to
            exactly 1 (renormalized here).
        level: Nominal credible-interval coverage, e.g. `0.95`.

    Returns:
        `(ci_low, ci_high)`.
    """
    order = np.argsort(values)
    v = values[order]
    p = probs[order]
    cdf = np.cumsum(p)
    total = cdf[-1]
    if total <= 0:
        return float(v[0]), float(v[-1])
    cdf = cdf / total
    alpha = (1.0 - level) / 2.0
    ci_low = float(np.interp(alpha, cdf, v))
    ci_high = float(np.interp(1.0 - alpha, cdf, v))
    return ci_low, ci_high


def _pad_threshold_grid(values: list[float], pad_frac: float) -> list[float]:
    """Extend an evenly spaced threshold grid past both ends by `pad_frac` of its span."""
    arr = np.asarray(values, dtype=float)
    if pad_frac <= 0 or arr.size < 2:
        return list(values)
    step = float(np.median(np.diff(arr)))
    if step <= 0:
        return list(values)
    n_pad = int(np.ceil(pad_frac * (arr[-1] - arr[0]) / step))
    lo = arr[0] - step * np.arange(n_pad, 0, -1)
    hi = arr[-1] + step * np.arange(1, n_pad + 1)
    return [float(v) for v in np.concatenate([lo, arr, hi])]


class QuestPlusProcedure:
    """QUEST+ single-parameter adaptive procedure.

    Args:
        intensity_values: Candidate stimulus intensity levels QUEST+ may
            select from, in `intensity_units` (e.g. log10 contrast steps).
            Must be non-empty and finite.
        intensity_units: Units of `intensity_values` and of values returned
            by `next_intensity`/passed to `update`, e.g. `"log10_contrast"`.
        threshold_values: Candidate threshold-parameter grid values, in the
            same units/scale as `intensity_values`.
        slope_values: Candidate slope (steepness) parameter grid values.
        guess_rate: Fixed guess (chance/floor) rate of the psychometric
            function, e.g. `0.5` for 2AFC, `0.125` for 8AFC.
        lapse_rate_values: Candidate lapse-rate grid values (upper-asymptote
            miss rate independent of stimulus intensity).
        function: Psychometric function family QUEST+ fits: `"weibull"`,
            `"logistic"`, or `"norm_cdf"` (cumulative normal).
        stim_selection_method: `questplus` stimulus-selection strategy, e.g.
            `"min_entropy"` (default in `questplus`) or `"min_n_entropy"`.
        stim_selection_options: Extra keyword options forwarded to
            `questplus`'s stimulus selection.
        max_trials: Maximum number of trials before `finished` becomes
            `True` regardless of posterior precision, or `None` for no
            fixed cap (caller decides when to stop).
        target_entropy: If given, `finished` becomes `True` once the
            posterior entropy drops below this value (an early-stopping
            criterion for a settled/precise threshold).
        ci_level: Nominal coverage for `estimate().ci_low`/`ci_high`,
            e.g. `0.95`.
        stim_scale: Scale `intensity_values` are on: `"log10"`, `"linear"`,
            or `"dB"`. Not part of the Phase 0 freeze; see module docstring.
            Forced to `"linear"` when `function="norm_cdf"` regardless of
            this argument (a `questplus` restriction).
        param_estimation_method: `"mean"` (default) or `"mode"`; which
            summary of the posterior `estimate()` reports. Not part of the
            Phase 0 freeze; see module docstring.
        threshold_pad_frac: Fraction of the threshold grid's span to extend
            it by on *each* side (same step). The true threshold may lie
            beyond the presentable intensity range (e.g. an observer better
            than the display floor); without padding the posterior piles onto
            the edge grid point, giving a degenerate credible interval
            (`ci_low == ci_high`) and a falsely certain estimate. `0` keeps
            the grid exactly as given.
    """

    def __init__(
        self,
        intensity_values: list[float],
        intensity_units: str,
        threshold_values: list[float],
        slope_values: list[float],
        guess_rate: float,
        lapse_rate_values: list[float],
        function: PsychometricFamily = "weibull",
        stim_selection_method: str = "min_entropy",
        stim_selection_options: dict[str, Any] | None = None,
        max_trials: int | None = None,
        target_entropy: float | None = None,
        ci_level: float = 0.95,
        stim_scale: Literal["log10", "linear", "dB"] = "log10",
        param_estimation_method: Literal["mean", "mode"] = "mean",
        threshold_pad_frac: float = 0.25,
    ) -> None:
        if function == "logistic":
            raise ValueError(
                "QuestPlusProcedure does not support function='logistic': the "
                "questplus package's built-in psychometric functions are "
                "'weibull', 'csf', 'norm_cdf', 'norm_cdf_2', and "
                "'thurstone_scaling' (no logistic-CDF option). Use "
                "function='weibull' or 'norm_cdf', or use ConstantStimuli / "
                "WeightedStaircase (whose fitting goes through "
                "vpsych.core.psychometric.fit_mle) for a logistic fit instead."
            )
        if function not in ("weibull", "norm_cdf"):
            raise ValueError(f"Unsupported function: {function!r}")
        if not intensity_values:
            raise ValueError("intensity_values must be non-empty")
        if not threshold_values or not slope_values or not lapse_rate_values:
            raise ValueError(
                "threshold_values, slope_values, and lapse_rate_values must be non-empty"
            )

        threshold_values = _pad_threshold_grid(threshold_values, threshold_pad_frac)

        self.intensity_values = intensity_values
        self.intensity_units = intensity_units
        self.threshold_values = threshold_values
        self.slope_values = slope_values
        self.guess_rate = guess_rate
        self.lapse_rate_values = lapse_rate_values
        self.function = function
        self.stim_selection_method = stim_selection_method
        self.stim_selection_options = stim_selection_options or {}
        self.max_trials = max_trials
        self.target_entropy = target_entropy
        self.ci_level = ci_level
        self.stim_scale = stim_scale
        self.param_estimation_method = param_estimation_method
        self.finished: bool = False
        self._n_trials = 0

        if function == "weibull":
            self._threshold_key = "threshold"
            self._slope_key = "slope"
            qp_func: Literal["weibull", "norm_cdf"] = "weibull"
            qp_stim_scale: Literal["log10", "linear", "dB"] = self.stim_scale
            param_domain = {
                "threshold": threshold_values,
                "slope": slope_values,
                "lower_asymptote": [guess_rate],
                "lapse_rate": lapse_rate_values,
            }
        else:  # norm_cdf
            self._threshold_key = "mean"
            self._slope_key = "sd"
            qp_func = "norm_cdf"
            qp_stim_scale = "linear"
            param_domain = {
                "mean": threshold_values,
                "sd": slope_values,
                "lower_asymptote": [guess_rate],
                "lapse_rate": lapse_rate_values,
            }

        self._qp = QuestPlus(
            stim_domain={"intensity": intensity_values},
            param_domain=param_domain,
            outcome_domain={"response": ["correct", "incorrect"]},
            func=qp_func,
            stim_scale=qp_stim_scale,
            stim_selection_method=stim_selection_method,
            stim_selection_options=self.stim_selection_options or None,
            param_estimation_method=param_estimation_method,
        )

    def next_intensity(self) -> float:
        stim = self._qp.next_stim
        return float(stim["intensity"])

    def update(self, intensity: float, correct: bool) -> None:
        if self.finished:
            return
        stim = {"intensity": float(intensity)}
        outcome = {"response": "correct" if correct else "incorrect"}
        self._qp.update(stim=stim, outcome=outcome)
        self._n_trials += 1

        if self.max_trials is not None and self._n_trials >= self.max_trials:
            self.finished = True
        if self.target_entropy is not None:
            _ = self._qp.next_stim  # refreshes self._qp.entropy for the new posterior
            if np.isfinite(self._qp.entropy) and self._qp.entropy <= self.target_entropy:
                self.finished = True

    def estimate(self) -> ThresholdEstimate:
        param_est = self._qp.param_estimate
        threshold = float(param_est[self._threshold_key])
        slope_est = float(param_est[self._slope_key])
        lapse_est = float(param_est["lapse_rate"])

        marginal = self._qp.marginal_posterior
        thr_vals = np.asarray(self._qp.param_domain[self._threshold_key], dtype=float)
        thr_probs = np.asarray(marginal[self._threshold_key], dtype=float)
        thr_sd = float(np.sqrt(np.sum(thr_probs * (thr_vals - threshold) ** 2)))
        ci_low, ci_high = _equal_tailed_credible_interval(thr_vals, thr_probs, self.ci_level)

        entropy = float(self._qp.entropy)
        return ThresholdEstimate(
            value=threshold,
            ci_low=ci_low,
            ci_high=ci_high,
            ci_level=self.ci_level,
            units=self.intensity_units,
            method=f"quest_plus_posterior_{self.param_estimation_method}",
            extra={
                "slope": slope_est,
                "lapse_rate": lapse_est,
                "threshold_posterior_sd": thr_sd,
                "n_trials": self._n_trials,
                "posterior_entropy": entropy if np.isfinite(entropy) else None,
                "ci_method": (
                    "equal-tailed credible interval from quantiles of the marginal "
                    "threshold posterior (not a normal approximation)"
                ),
            },
        )

    def intensity_at_p_correct(self, p_target: float) -> tuple[float, float, float]:
        """Report this procedure's threshold at an arbitrary target proportion correct.

        Inverts `questplus`'s own fitted Weibull directly (see
        `questplus_weibull_x_at_p` and this module's "Criterion conversion
        pitfall" docstring section) -- **not**
        `vpsych.core.psychometric.intensity_at_p_correct`, which assumes a
        different, incompatible parameterization and would silently return
        the wrong intensity applied to this procedure's fit.

        Only supported for `function="weibull"` (the `"norm_cdf"` function
        has its own, different formula this inversion doesn't handle).

        Args:
            p_target: Target probability correct, e.g. `0.75` for the
                conventional 2AFC criterion, or
                `self.guess_rate + 0.5 * (1 - self.guess_rate - lapse)` for
                the criterion equivalent to `vpsych.core.psychometric`'s own
                `F(0) = 0.5` convention (the "FrACT" midpoint).

        Returns:
            `(reported_value, ci_low, ci_high)`, on `self.intensity_units`,
            with the credible interval shifted by the same amount as the
            point estimate (see `questplus_weibull_report_at_p`).
        """
        if self.function != "weibull":
            raise NotImplementedError(
                "intensity_at_p_correct only supports function='weibull' "
                f"(questplus's own Weibull inversion), not {self.function!r}"
            )
        return questplus_weibull_report_at_p(self.estimate(), self.guess_rate, p_target)

    def state_dict(self) -> dict[str, Any]:
        marginal = self._qp.marginal_posterior
        means: dict[str, float] = {}
        sds: dict[str, float] = {}
        for name, vals in self._qp.param_domain.items():
            if name == "lower_asymptote":
                continue  # fixed (guess_rate), not an estimated parameter
            vals_arr = np.asarray(vals, dtype=float)
            probs = np.asarray(marginal[name], dtype=float)
            mean = float(np.sum(probs * vals_arr))
            sd = float(np.sqrt(np.sum(probs * (vals_arr - mean) ** 2)))
            means[name] = mean
            sds[name] = sd
        entropy = float(self._qp.entropy)
        return {
            "n_trials": self._n_trials,
            "finished": self.finished,
            "posterior_mean": means,
            "posterior_sd": sds,
            "entropy": entropy if np.isfinite(entropy) else None,
        }
