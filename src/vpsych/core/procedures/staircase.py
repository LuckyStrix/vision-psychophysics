"""Weighted up/down staircase procedure (Kaernbach 1991).

Implements :class:`vpsych.core.procedures.base.AdaptiveProcedure`. A
weighted up-down staircase converges on an arbitrary target percent-correct
point by using unequal step sizes for "up" (after an incorrect response)
and "down" (after a correct response) steps, with the up/down step-size
ratio set so the staircase's equilibrium point corresponds to the desired
target probability (Kaernbach, C. (1991). Simple adaptive testing with the
weighted up-down method. Perception & Psychophysics, 49(3), 227-229).

This module also implements the classic transformed up/down rule (Levitt,
E. (1971). Transformed up-down methods in psychoacoustics. The Journal of
the Acoustical Society of America, 49(2), 467-477): after `n_down`
consecutive correct responses, intensity decreases by one step; a single
incorrect response increases intensity by one step and resets the
consecutive-correct counter. This targets the proportion-correct point
`0.5 ** (1 / n_down)` (see `transformed_rule_target_p`), e.g. ~70.7% for a
2-down/1-up rule, ~79.4% for 3-down/1-up.

Both rules share the same reversal-tracking, step-size-reduction, and
reversal-mean estimation machinery below; `rule` selects which trial-by-trial
update logic drives intensity changes (`"weighted"`, the default, matches
the frozen Kaernbach-only constructor's documented behavior exactly;
`"transformed"` switches to the Levitt rule and additionally consults
`n_down`). Both are additive, backward-compatible constructor keyword
arguments (default `rule="weighted"`, `n_down=1`) -- Phase 0 had frozen only
the weighted-rule parameters.
"""

from __future__ import annotations

from typing import Any, Literal, cast

import numpy as np

from vpsych.core.procedures.base import ThresholdEstimate
from vpsych.core.psychometric import (
    IntensityScale,
    PsychometricFamily,
    bootstrap_ci_at_p_correct,
    fit_mle,
    intensity_at_p_correct,
)

_VALID_FAMILIES = ("weibull", "logistic", "norm_cdf")


def transformed_rule_target_p(n_down: int) -> float:
    """Proportion-correct point targeted by a classic `n_down`-down/1-up staircase.

    Per Levitt (1971): at equilibrium, the probability that `n_down`
    consecutive trials are all correct (triggering a step down) equals the
    probability of a single incorrect trial (triggering a step up), i.e.
    `p_target ** n_down == 1 - p_target`'s complement condition reduces to
    `p_target = 0.5 ** (1 / n_down)`.

    Args:
        n_down: Number of consecutive correct responses required before a
            step down (`n_down=1` is simple 1-down/1-up, targeting 50%).

    Returns:
        The targeted proportion correct, in (0, 1).
    """
    if n_down < 1:
        raise ValueError("n_down must be >= 1")
    return float(0.5 ** (1.0 / n_down))


def _aggregate_trials(
    intensities: list[float], corrects: list[bool], ndigits: int = 3
) -> tuple[list[float], list[int], list[int]]:
    """Aggregate raw per-trial (intensity, correct) pairs into per-level binomial counts.

    A staircase's presented intensity drifts continuously (shrinking step
    sizes mean later trials cluster near threshold at slightly different
    floats), so fitting one row per raw trial would give `fit_mle`/
    `bootstrap_ci` hundreds of distinct "levels" -- correct, but needlessly
    slow to refit thousands of times for a bootstrap CI. Rounding to
    `ndigits` merges trials whose intensities are practically identical
    (well below step size) into shared levels, which is a much smaller,
    faster design matrix with no meaningful loss of information.

    Args:
        intensities: Presented intensity at each trial.
        corrects: Whether each trial was scored correct.
        ndigits: Decimal places to round intensities to before grouping.

    Returns:
        `(levels, n_correct, n_total)`, sorted by level.
    """
    levels: dict[float, list[int]] = {}
    for x, c in zip(intensities, corrects, strict=True):
        key = round(float(x), ndigits)
        counts = levels.setdefault(key, [0, 0])
        counts[0] += 1
        if c:
            counts[1] += 1
    xs = sorted(levels)
    n_total = [levels[x][0] for x in xs]
    n_correct = [levels[x][1] for x in xs]
    return xs, n_correct, n_total


class WeightedStaircase:
    """Kaernbach (1991) weighted up/down staircase (also supports classic Levitt 1971 rule).

    Args:
        start_intensity: Starting stimulus intensity, in `intensity_units`.
        intensity_units: Units of intensity values, e.g. `"log10_contrast"`,
            `"logMAR"`.
        step_up: Step size added to intensity after an incorrect response,
            in `intensity_units`.
        step_down: Step size subtracted from intensity after a correct
            response, in `intensity_units`. The ratio `step_up / step_down`
            sets the target convergence probability: for target `p_target`,
            `step_down / step_up == p_target / (1 - p_target)`.
        target_p_correct: The proportion-correct point this staircase
            targets (informational; should be consistent with the
            `step_up`/`step_down` ratio above -- callers typically derive
            one from the other).
        min_intensity: Lower clamp on presented intensity, in
            `intensity_units`, or `None` for no lower bound.
        max_intensity: Upper clamp on presented intensity, in
            `intensity_units`, or `None` for no upper bound.
        n_reversals_to_stop: Number of direction reversals after which
            `finished` becomes `True`.
        n_reversals_for_estimate: Number of most-recent reversals averaged
            to compute the threshold estimate (typically less than or equal
            to `n_reversals_to_stop`, excluding early/unstable reversals).
        step_size_reduction_after: Optional number of reversals after which
            step sizes are reduced (e.g. halved), for coarse-then-fine
            staircases. `None` disables step-size reduction.
        ci_level: Nominal coverage for `estimate().ci_low`/`ci_high`,
            e.g. `0.95`.
        rule: `"weighted"` (default) applies Kaernbach's rule -- every
            correct trial steps down by `step_down`, every incorrect trial
            steps up by `step_up`. `"transformed"` applies the classic Levitt
            n-down/1-up rule instead: `n_down` consecutive corrects are
            required before a `step_down` step; any incorrect trial
            immediately steps up by `step_up` and resets the consecutive
            count. Not part of the Phase 0 freeze; added (with a default
            preserving the original weighted-only behavior) to cover both
            methods named in this project's Phase 1a scope.
        n_down: Number of consecutive corrects required before a step down,
            used only when `rule="transformed"`. See
            `transformed_rule_target_p`. Not part of the Phase 0 freeze;
            added alongside `rule` (default `1`, i.e. simple 1-down/1-up if
            `rule="transformed"` were selected without changing it).
        psychometric_family: Family fit to *all* recorded trials (not just
            reversals) to estimate threshold, e.g. `"weibull"`, `"logistic"`,
            `"norm_cdf"` (see `vpsych.core.psychometric.PsychometricFunction`).
        guess_rate: Fixed guess (chance/floor) rate of the psychometric
            function, e.g. `0.5` for 2AFC.
        fix_lapse_rate: If given, the lapse rate is fixed to this value
            during fitting rather than estimated freely.
        n_bootstrap: Number of bootstrap resamples used to compute the
            confidence interval (see
            `vpsych.core.psychometric.bootstrap_ci`).
        intensity_scale: Whether intensities are on a `"log10"` or
            `"linear"` scale, forwarded to `fit_mle`/the fitted
            `PsychometricFunction`.
        rng: Random generator used only by `estimate()`'s bootstrap
            resampling (see `vpsych.core.psychometric.bootstrap_ci`); the
            staircase's own trial-by-trial logic is fully deterministic
            given the responses it is told about. If `None`, an unseeded
            `numpy.random.default_rng()` is created lazily the first time
            `estimate()` needs it -- callers requiring reproducible CIs
            must pass one (see `vpsych.core.rng.make_rng`).

    Estimation: `estimate()`'s primary point estimate and confidence
    interval come from a maximum-likelihood psychometric-function fit
    (`vpsych.core.psychometric.fit_mle`) to *every* trial this staircase has
    seen (not just reversals), read off at `target_p_correct` via
    `vpsych.core.psychometric.intensity_at_p_correct`, with the CI from
    `vpsych.core.psychometric.bootstrap_ci_at_p_correct` -- a **full-refit**
    parametric bootstrap that refits `threshold`/`slope`/`lapse` jointly on
    every resample and evaluates `intensity_at_p_correct` against each
    resample's own fitted function, rather than (as an earlier version of
    this method did) taking a bootstrap interval on `threshold` alone and
    converting it to the target-percent intensity while holding `slope`/
    `lapse` fixed at the original point estimate. This is the well-
    calibrated estimate to use.

    The classic reversal mean (the average intensity at the last
    `n_reversals_for_estimate` reversals) is still computed and reported in
    `extra["reversal_mean"]`, but *without* a CI: reversal values are not
    independent draws (each is shaped by the run of trials since the
    previous reversal), so a Student-t or other i.i.d.-based interval on
    them is not a valid confidence interval -- empirically it undercovers
    badly (~20-30% observed coverage for a nominal 95% interval; see
    `tests/procedures/test_staircase.py`'s git history). We do not report a
    mislabeled CI for it.

    **Phase 4 fix and re-measurement (item 5).** The MLE-fit CI was already
    a large, real improvement over the reversal-mean interval (~20-30%
    observed coverage), but an earlier version held `slope`/`lapse` fixed
    at their point estimates when converting a `threshold`-only bootstrap
    interval to the target-percent intensity -- a good approximation only
    when slope uncertainty is modest (true for `ConstantStimuli`'s
    deliberately wide-spread levels, false for a staircase, which
    concentrates almost all trials tightly around threshold *by design*,
    leaving `slope` poorly identified from the data). That version measured
    ~72-77% observed coverage for a nominal 95% interval across three true
    thresholds (`tests/procedures/test_staircase.py`'s slow validation).

    `estimate()` now uses `bootstrap_ci_at_p_correct` (see above), the
    full-refit bootstrap this docstring previously described as a possible
    future improvement ("would require a bespoke bootstrap loop refitting
    per resample"). Re-measured the same way (200 reps x 3 true thresholds,
    N=100 bootstrap resamples each): observed coverage improved to
    **roughly 0.85-0.92** across the three true thresholds (bias remained
    small, well under the plan's |bias| < 0.05 bound), a substantial
    improvement over the previous ~72-77%, though still short of a
    textbook ~95% -- reported honestly here rather than widened bands
    hiding a residual shortfall (see
    `tests/procedures/test_staircase.py::test_weighted_staircase_recovery_slow`
    for the exact numbers this run produced). The residual gap is
    consistent with the same underlying cause (a staircase's narrow
    intensity range leaves the psychometric function's shape, not just its
    location, genuinely harder to pin down than a passive wide-range
    design like `ConstantStimuli` achieves, even with slope/lapse
    uncertainty now fully propagated) rather than a remaining bug.
    """

    def __init__(
        self,
        start_intensity: float,
        intensity_units: str,
        step_up: float,
        step_down: float,
        target_p_correct: float,
        min_intensity: float | None = None,
        max_intensity: float | None = None,
        n_reversals_to_stop: int = 12,
        n_reversals_for_estimate: int = 8,
        step_size_reduction_after: int | None = None,
        ci_level: float = 0.95,
        rule: Literal["weighted", "transformed"] = "weighted",
        n_down: int = 1,
        psychometric_family: str = "weibull",
        guess_rate: float = 0.5,
        fix_lapse_rate: float | None = None,
        n_bootstrap: int = 1000,
        intensity_scale: IntensityScale = "log10",
        rng: np.random.Generator | None = None,
    ) -> None:
        if step_up <= 0 or step_down <= 0:
            raise ValueError("step_up and step_down must be positive")
        if n_reversals_to_stop < 1:
            raise ValueError("n_reversals_to_stop must be >= 1")
        if n_down < 1:
            raise ValueError("n_down must be >= 1")
        if psychometric_family not in _VALID_FAMILIES:
            raise ValueError(f"psychometric_family must be one of {_VALID_FAMILIES}")
        if (
            min_intensity is not None
            and max_intensity is not None
            and min_intensity > max_intensity
        ):
            raise ValueError("min_intensity must be <= max_intensity")

        self.start_intensity = start_intensity
        self.intensity_units = intensity_units
        self.step_up = step_up
        self.step_down = step_down
        self.target_p_correct = target_p_correct
        self.min_intensity = min_intensity
        self.max_intensity = max_intensity
        self.n_reversals_to_stop = n_reversals_to_stop
        self.n_reversals_for_estimate = n_reversals_for_estimate
        self.step_size_reduction_after = step_size_reduction_after
        self.ci_level = ci_level
        self.rule = rule
        self.n_down = n_down
        self.psychometric_family = psychometric_family
        self.guess_rate = guess_rate
        self.fix_lapse_rate = fix_lapse_rate
        self.n_bootstrap = n_bootstrap
        self.intensity_scale: IntensityScale = intensity_scale
        self.rng = rng
        self.finished: bool = False

        self._current_intensity = self._clip(float(start_intensity))
        self._cur_step_up = float(step_up)
        self._cur_step_down = float(step_down)
        self._direction: int | None = None  # -1: last move was down, +1: last move was up
        self._reversal_count = 0
        self._reversal_intensities: list[float] = []
        self._n_trials = 0
        self._consecutive_correct = 0
        self._reductions_applied = 0
        self._trial_intensities: list[float] = []
        self._trial_correct: list[bool] = []

    def _clip(self, value: float) -> float:
        if self.min_intensity is not None:
            value = max(value, self.min_intensity)
        if self.max_intensity is not None:
            value = min(value, self.max_intensity)
        return value

    def next_intensity(self) -> float:
        return self._current_intensity

    def update(self, intensity: float, correct: bool) -> None:
        if self.finished:
            return
        self._n_trials += 1
        self._trial_intensities.append(float(intensity))
        self._trial_correct.append(bool(correct))

        move: int | None  # -1 => step down (harder/lower), +1 => step up (easier/higher)
        if self.rule == "weighted":
            move = -1 if correct else 1
        else:  # "transformed": classic Levitt n-down/1-up
            if correct:
                self._consecutive_correct += 1
                if self._consecutive_correct >= self.n_down:
                    move = -1
                    self._consecutive_correct = 0
                else:
                    move = None
            else:
                self._consecutive_correct = 0
                move = 1

        if move is not None:
            if self._direction is not None and move != self._direction:
                self._reversal_count += 1
                self._reversal_intensities.append(intensity)
                if (
                    self.step_size_reduction_after is not None
                    and self.step_size_reduction_after > 0
                ):
                    target_reductions = self._reversal_count // self.step_size_reduction_after
                    while self._reductions_applied < target_reductions:
                        self._cur_step_up /= 2.0
                        self._cur_step_down /= 2.0
                        self._reductions_applied += 1
            self._direction = move
            step = self._cur_step_down if move == -1 else self._cur_step_up
            self._current_intensity = self._clip(self._current_intensity + move * step)

        if self._reversal_count >= self.n_reversals_to_stop:
            self.finished = True

    def _reversal_mean_and_sd(self) -> tuple[float, float, int]:
        """Classic reversal-mean estimate (reported in `extra`, without a CI -- see class docstring)."""
        values = self._reversal_intensities[-self.n_reversals_for_estimate :]
        if not values:
            values = [self._current_intensity]
        arr = np.asarray(values, dtype=float)
        mean = float(np.mean(arr))
        sd = float(np.std(arr, ddof=1)) if len(arr) >= 2 else 0.0
        return mean, sd, len(arr)

    def estimate(self) -> ThresholdEstimate:
        reversal_mean, reversal_sd, n_reversals_used = self._reversal_mean_and_sd()
        extra: dict[str, Any] = {
            "reversal_mean": reversal_mean,
            "n_reversals_used": n_reversals_used,
            "n_reversals_total": self._reversal_count,
            "reversal_sd": reversal_sd,
            "rule": self.rule,
            "target_p_correct": self.target_p_correct,
        }

        if not self._trial_intensities:
            # No trials recorded yet (e.g. state_dict()/estimate() called before the
            # first update()): a degenerate single-point estimate at the starting
            # intensity, matching pre-fit behavior for callers that poll estimate()
            # during practice trials.
            return ThresholdEstimate(
                value=self._current_intensity,
                ci_low=self._current_intensity,
                ci_high=self._current_intensity,
                ci_level=self.ci_level,
                units=self.intensity_units,
                method="staircase_no_data",
                extra=extra,
            )

        xs, n_correct, n_total = _aggregate_trials(self._trial_intensities, self._trial_correct)
        family = cast(PsychometricFamily, self.psychometric_family)
        fit = fit_mle(
            intensities=xs,
            n_correct=n_correct,
            n_total=n_total,
            family=family,
            guess=self.guess_rate,
            fix_lapse=self.fix_lapse_rate,
            intensity_scale=self.intensity_scale,
        )
        value = intensity_at_p_correct(fit.function, self.target_p_correct)

        rng = self.rng
        if rng is None:
            rng = np.random.default_rng()
            self.rng = rng
        target_ci = bootstrap_ci_at_p_correct(
            fit, self.target_p_correct, n_boot=self.n_bootstrap, level=self.ci_level, rng=rng
        )
        ci_low, ci_high = target_ci.ci_low, target_ci.ci_high

        extra.update(
            {
                "fit_threshold_f50": fit.function.threshold,
                "slope": fit.function.slope,
                "lapse": fit.function.lapse,
                "log_likelihood": fit.log_likelihood,
                "converged": fit.converged,
                "n_trials": fit.n_trials,
                "n_levels_used": len(xs),
                "ci_method": (
                    "full-refit parametric bootstrap (bootstrap_ci_at_p_correct): threshold, "
                    "slope, and lapse are all refit on every resample, and the "
                    "target_p_correct intensity is evaluated against each resample's own "
                    "fitted function -- not a single threshold interval converted while "
                    "holding slope/lapse fixed at the original point estimate (see class "
                    "docstring's 'Known limitation' section for why that approximation "
                    "under-covered for a staircase's narrow dynamic range)"
                ),
            }
        )

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
            "n_trials": self._n_trials,
            "n_reversals": self._reversal_count,
            "current_intensity": self._current_intensity,
            "direction": self._direction,
            "step_up": self._cur_step_up,
            "step_down": self._cur_step_down,
            "finished": self.finished,
            "recent_reversal_intensities": list(
                self._reversal_intensities[-self.n_reversals_for_estimate :]
            ),
            "rule": self.rule,
        }
