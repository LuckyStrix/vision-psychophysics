"""Weighted up/down staircase procedure (Kaernbach 1991).

Implements :class:`vpsych.core.procedures.base.AdaptiveProcedure`. A
weighted up-down staircase converges on an arbitrary target percent-correct
point by using unequal step sizes for "up" (after an incorrect response)
and "down" (after a correct response) steps, with the up/down step-size
ratio set so the staircase's equilibrium point corresponds to the desired
target probability (Kaernbach, C. (1991). Simple adaptive testing with the
weighted up-down method. Perception & Psychophysics, 49(3), 227-229).

Not implemented in Phase 0 -- this module freezes the constructor signature
and documents intended behavior; methods raise `NotImplementedError` until a
later phase implements the staircase logic and reversal-based threshold
estimation.
"""

from __future__ import annotations

from typing import Any

from vpsych.core.procedures.base import ThresholdEstimate


class WeightedStaircase:
    """Kaernbach (1991) weighted up/down staircase.

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
    ) -> None:
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
        self.finished: bool = False
        raise NotImplementedError(
            "WeightedStaircase is a Phase-0 interface stub; implementation lands in a later phase."
        )

    def next_intensity(self) -> float:
        raise NotImplementedError

    def update(self, intensity: float, correct: bool) -> None:
        raise NotImplementedError

    def estimate(self) -> ThresholdEstimate:
        raise NotImplementedError

    def state_dict(self) -> dict[str, Any]:
        raise NotImplementedError
