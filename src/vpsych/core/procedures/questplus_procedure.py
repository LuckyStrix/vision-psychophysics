"""QUEST+ adaptive procedure, wrapping the `questplus` package.

Implements :class:`vpsych.core.procedures.base.AdaptiveProcedure` for a
single-parameter (threshold) QUEST+ run. QUEST+ (Watson 2017) generalizes
QUEST to arbitrary psychometric-function families and stimulus/response
spaces by maintaining a full posterior over parameters and selecting each
next stimulus to maximize expected information gain.

Not implemented in Phase 0 -- this module freezes the constructor signature
and documents intended behavior; `next_intensity`/`update`/`estimate` raise
`NotImplementedError` until a later phase implements them against
`questplus.QuestPlus`.
"""

from __future__ import annotations

from typing import Any, Literal

from vpsych.core.procedures.base import ThresholdEstimate

PsychometricFamily = Literal["weibull", "logistic", "norm_cdf"]


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
    ) -> None:
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
        self.finished: bool = False
        raise NotImplementedError(
            "QuestPlusProcedure is a Phase-0 interface stub; implementation "
            "lands in a later phase (wraps questplus.QuestPlus)."
        )

    def next_intensity(self) -> float:
        raise NotImplementedError

    def update(self, intensity: float, correct: bool) -> None:
        raise NotImplementedError

    def estimate(self) -> ThresholdEstimate:
        raise NotImplementedError

    def state_dict(self) -> dict[str, Any]:
        raise NotImplementedError
