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

from typing import Any, Literal

import numpy as np
from scipy import stats as _stats  # type: ignore[import-untyped]

from vpsych.core.procedures.base import ThresholdEstimate


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

    Estimation: the threshold estimate is the mean of the intensities at the
    last `n_reversals_for_estimate` reversals, with its CI computed via a
    Student-t interval on those reversal values (`mean +/- t_(n-1,
    1-alpha/2) * SEM`) -- appropriate here because the number of reversal
    values used for the estimate is typically small (a handful to ~10),
    where the t distribution's heavier tails are more honest than a normal
    approximation. (A bootstrap of the reversal values is a documented
    alternative -- either is defensible for this small-n, non-independent
    use case; we use the t interval for simplicity and determinism.)

    Known limitation: reversal values are *not* independent draws (each is
    shaped by the run of trials since the previous reversal), so treating
    them as i.i.d. for the Student-t SEM understates the true uncertainty --
    in simulation this interval's empirical coverage runs well below its
    nominal level (see `tests/procedures/test_staircase.py`). Use it as a
    rough indicator of estimate spread, not a calibrated confidence
    interval; where a well-calibrated CI matters, prefer
    `QuestPlusProcedure` or `ConstantStimuli` (whose CIs come from
    `vpsych.core.psychometric.bootstrap_ci` on independent binomial trial
    outcomes).
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
    ) -> None:
        if step_up <= 0 or step_down <= 0:
            raise ValueError("step_up and step_down must be positive")
        if n_reversals_to_stop < 1:
            raise ValueError("n_reversals_to_stop must be >= 1")
        if n_down < 1:
            raise ValueError("n_down must be >= 1")
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

    def estimate(self) -> ThresholdEstimate:
        values = self._reversal_intensities[-self.n_reversals_for_estimate :]
        if not values:
            values = [self._current_intensity]
        arr = np.asarray(values, dtype=float)
        mean = float(np.mean(arr))
        n = len(arr)
        sd = 0.0
        ci_low = ci_high = mean
        if n >= 2:
            sd = float(np.std(arr, ddof=1))
            sem = sd / np.sqrt(n)
            if sem > 0:
                t_crit = float(_stats.t.ppf(0.5 + self.ci_level / 2, df=n - 1))
                ci_low = mean - t_crit * sem
                ci_high = mean + t_crit * sem

        return ThresholdEstimate(
            value=mean,
            ci_low=float(ci_low),
            ci_high=float(ci_high),
            ci_level=self.ci_level,
            units=self.intensity_units,
            method="staircase_reversal_mean",
            extra={
                "n_reversals_used": n,
                "n_reversals_total": self._reversal_count,
                "reversal_sd": sd,
                "rule": self.rule,
                "target_p_correct": self.target_p_correct,
            },
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
