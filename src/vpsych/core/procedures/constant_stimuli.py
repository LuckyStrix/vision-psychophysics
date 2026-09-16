"""Method of constant stimuli.

Implements :class:`vpsych.core.procedures.base.AdaptiveProcedure`. Presents
a fixed, pre-specified set of intensity levels, each repeated a fixed
number of times, in a randomized order, then fits a psychometric function
to the resulting per-level proportions correct
(:func:`vpsych.core.psychometric.fit_mle`) to estimate threshold. Unlike
QUEST+ or a staircase, intensities are not adapted based on responses --
this method trades efficiency for a full, evenly-sampled psychometric
function suitable for careful goodness-of-fit assessment.

`estimate()` reads the threshold off the fitted psychometric function at
`target_p_correct` (via
:func:`vpsych.core.psychometric.intensity_at_p_correct`), *not* necessarily
at the fitted function's own F=0.5 `threshold` parameter (see
`vpsych.core.psychometric` module docs for that convention). Its confidence
interval is obtained by running
:func:`vpsych.core.psychometric.bootstrap_ci` on the fitted `threshold`
parameter, then converting each bound to the `target_p_correct` intensity
the same way as the point estimate -- i.e. holding the fitted slope and
lapse rate at their point estimates while only the threshold varies. This is
a standard, documented simplification (a full CI that also propagates
slope/lapse resample-to-resample variability into the target-percent
intensity would require re-deriving `intensity_at_p_correct` inside every
bootstrap resample; the chosen approach is exact when `target_p_correct`
happens to equal the F=0.5 point, and a good approximation otherwise for the
typically-modest slope uncertainty at constant-stimuli trial counts).
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from vpsych.core.procedures.base import ThresholdEstimate
from vpsych.core.psychometric import (
    IntensityScale,
    PsychometricFamily,
    bootstrap_ci,
    deviance_gof,
    fit_mle,
    intensity_at_p_correct,
)

_VALID_FAMILIES = ("weibull", "logistic", "norm_cdf")


class ConstantStimuli:
    """Method-of-constant-stimuli procedure.

    Args:
        intensity_levels: The fixed set of intensity levels to present, in
            `intensity_units`. Must be non-empty.
        intensity_units: Units of `intensity_levels`, e.g.
            `"log10_contrast"`, `"coherence_fraction"`.
        n_reps_per_level: Number of presentations of each level.
        rng: Random generator used to shuffle presentation order (see
            `vpsych.core.rng.make_rng`).
        psychometric_family: Family fit to the resulting data to estimate
            threshold, e.g. `"weibull"`, `"logistic"`, `"norm_cdf"` (see
            `vpsych.core.psychometric.PsychometricFunction`).
        guess_rate: Fixed guess (chance/floor) rate of the psychometric
            function, e.g. `0.5` for 2AFC.
        fix_lapse_rate: If given, the lapse rate is fixed to this value
            during fitting rather than estimated freely.
        target_p_correct: Proportion-correct point at which the threshold is
            read off the fitted psychometric function, e.g. `0.75`.
        ci_level: Nominal coverage for `estimate().ci_low`/`ci_high`,
            e.g. `0.95`.
        n_bootstrap: Number of bootstrap resamples used to compute the
            confidence interval (see
            `vpsych.core.psychometric.bootstrap_ci`).
        intensity_scale: Whether `intensity_levels` are on a `"log10"` or
            `"linear"` scale, forwarded to `fit_mle`/the fitted
            `PsychometricFunction`. Not part of the Phase 0 freeze (the
            frozen constructor only carries a free-text `intensity_units`
            label, not a scale); added (default `"log10"`, matching this
            project's typical contrast/coherence-type dimensions) because
            `fit_mle` requires it.
    """

    def __init__(
        self,
        intensity_levels: list[float],
        intensity_units: str,
        n_reps_per_level: int,
        rng: np.random.Generator,
        psychometric_family: str = "weibull",
        guess_rate: float = 0.5,
        fix_lapse_rate: float | None = None,
        target_p_correct: float = 0.75,
        ci_level: float = 0.95,
        n_bootstrap: int = 1000,
        intensity_scale: IntensityScale = "log10",
    ) -> None:
        if not intensity_levels:
            raise ValueError("intensity_levels must be non-empty")
        if n_reps_per_level < 1:
            raise ValueError("n_reps_per_level must be >= 1")
        if psychometric_family not in _VALID_FAMILIES:
            raise ValueError(f"psychometric_family must be one of {_VALID_FAMILIES}")

        self.intensity_levels = list(intensity_levels)
        self.intensity_units = intensity_units
        self.n_reps_per_level = n_reps_per_level
        self.rng = rng
        self.psychometric_family = psychometric_family
        self.guess_rate = guess_rate
        self.fix_lapse_rate = fix_lapse_rate
        self.target_p_correct = target_p_correct
        self.ci_level = ci_level
        self.n_bootstrap = n_bootstrap
        self.intensity_scale: IntensityScale = intensity_scale
        self.finished: bool = False

        sequence = np.repeat(np.array(self.intensity_levels, dtype=float), n_reps_per_level)
        self.rng.shuffle(sequence)
        self._sequence = sequence
        self._idx = 0
        self._responses: dict[float, list[bool]] = {float(lvl): [] for lvl in self.intensity_levels}

    def _nearest_level(self, intensity: float) -> float:
        levels = np.array(self.intensity_levels, dtype=float)
        return float(levels[int(np.argmin(np.abs(levels - intensity)))])

    def next_intensity(self) -> float:
        if self._idx >= len(self._sequence):
            raise RuntimeError("ConstantStimuli exhausted: all trials already presented")
        return float(self._sequence[self._idx])

    def update(self, intensity: float, correct: bool) -> None:
        if self.finished:
            return
        level = self._nearest_level(intensity)
        self._responses[level].append(bool(correct))
        self._idx += 1
        if self._idx >= len(self._sequence):
            self.finished = True

    def estimate(self) -> ThresholdEstimate:
        levels_used = [lvl for lvl in self.intensity_levels if self._responses[float(lvl)]]
        if not levels_used:
            raise RuntimeError("ConstantStimuli.estimate() called before any trials were recorded")
        n_correct = [sum(self._responses[float(lvl)]) for lvl in levels_used]
        n_total = [len(self._responses[float(lvl)]) for lvl in levels_used]

        family = cast(PsychometricFamily, self.psychometric_family)
        fit = fit_mle(
            intensities=levels_used,
            n_correct=n_correct,
            n_total=n_total,
            family=family,
            guess=self.guess_rate,
            fix_lapse=self.fix_lapse_rate,
            intensity_scale=self.intensity_scale,
        )

        value = intensity_at_p_correct(fit.function, self.target_p_correct)
        thr_ci = bootstrap_ci(
            fit, n_boot=self.n_bootstrap, level=self.ci_level, rng=self.rng, parameter="threshold"
        )
        low_fn = fit.function.model_copy(update={"threshold": thr_ci.ci_low})
        high_fn = fit.function.model_copy(update={"threshold": thr_ci.ci_high})
        ci_low = intensity_at_p_correct(low_fn, self.target_p_correct)
        ci_high = intensity_at_p_correct(high_fn, self.target_p_correct)
        if ci_low > ci_high:
            ci_low, ci_high = ci_high, ci_low

        extra: dict[str, Any] = {
            "fit_threshold_f50": fit.function.threshold,
            "slope": fit.function.slope,
            "lapse": fit.function.lapse,
            "log_likelihood": fit.log_likelihood,
            "converged": fit.converged,
            "n_trials": fit.n_trials,
            "n_levels_used": len(levels_used),
            "ci_method": (
                "bootstrap_ci on the F=0.5 threshold, converted to the "
                "target_p_correct intensity holding slope/lapse fixed"
            ),
        }
        if len(levels_used) >= 3:
            # n_mc smaller than deviance_gof's own default (500): estimate()
            # may be called often (e.g. live progress display, or many reps
            # in a validation harness), and 50 Monte-Carlo simulated
            # datasets already gives a usable (if noisier than 500) p-value
            # for a quality-flag use case, at a fraction of the cost.
            gof = deviance_gof(fit, levels_used, n_correct, n_total, n_mc=50, rng=self.rng)
            extra["deviance"] = gof.deviance
            extra["deviance_df"] = gof.df
            extra["deviance_p_value"] = gof.p_value

        return ThresholdEstimate(
            value=float(value),
            ci_low=float(ci_low),
            ci_high=float(ci_high),
            ci_level=self.ci_level,
            units=self.intensity_units,
            method=f"{self.psychometric_family}_mle_p{self.target_p_correct:g}",
            extra=extra,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "n_trials": int(self._idx),
            "n_trials_total": len(self._sequence),
            "finished": self.finished,
            "n_correct_by_level": {
                str(lvl): int(sum(resp)) for lvl, resp in self._responses.items()
            },
            "n_total_by_level": {str(lvl): len(resp) for lvl, resp in self._responses.items()},
        }
