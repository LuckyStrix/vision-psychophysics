"""Method of constant stimuli.

Implements :class:`vpsych.core.procedures.base.AdaptiveProcedure`. Presents
a fixed, pre-specified set of intensity levels, each repeated a fixed
number of times, in a randomized order, then fits a psychometric function
to the resulting per-level proportions correct
(:func:`vpsych.core.psychometric.fit_mle`) to estimate threshold. Unlike
QUEST+ or a staircase, intensities are not adapted based on responses --
this method trades efficiency for a full, evenly-sampled psychometric
function suitable for careful goodness-of-fit assessment.

Not implemented in Phase 0 -- this module freezes the constructor signature
and documents intended behavior; methods raise `NotImplementedError` until a
later phase implements trial sequencing and the fit-based estimate.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vpsych.core.procedures.base import ThresholdEstimate


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
    ) -> None:
        self.intensity_levels = intensity_levels
        self.intensity_units = intensity_units
        self.n_reps_per_level = n_reps_per_level
        self.rng = rng
        self.psychometric_family = psychometric_family
        self.guess_rate = guess_rate
        self.fix_lapse_rate = fix_lapse_rate
        self.target_p_correct = target_p_correct
        self.ci_level = ci_level
        self.n_bootstrap = n_bootstrap
        self.finished: bool = False
        raise NotImplementedError(
            "ConstantStimuli is a Phase-0 interface stub; implementation lands in a later phase."
        )

    def next_intensity(self) -> float:
        raise NotImplementedError

    def update(self, intensity: float, correct: bool) -> None:
        raise NotImplementedError

    def estimate(self) -> ThresholdEstimate:
        raise NotImplementedError

    def state_dict(self) -> dict[str, Any]:
        raise NotImplementedError
