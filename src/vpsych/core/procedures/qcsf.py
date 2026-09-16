"""Quick Contrast Sensitivity Function (qCSF) procedure (Lesmes et al. 2010).

Implements :class:`vpsych.core.procedures.base.MultiParamProcedure`. qCSF
jointly estimates the four parameters of a truncated log-parabola model of
the contrast sensitivity function (peak gain, peak spatial frequency,
bandwidth, and low-frequency truncation) using a QUEST+-style Bayesian
adaptive procedure over a 2D stimulus space (spatial frequency x contrast),
selecting each trial's stimulus to maximize expected information gain about
the joint parameter posterior (Lesmes, L. A., Lu, Z.-L., Baek, J., &
Albright, T. D. (2010). Bayesian adaptive estimation of the contrast
sensitivity function: The quick CSF method. Journal of Vision, 10(3):17).

Not implemented in Phase 0 -- this module freezes the constructor signature
and documents intended behavior; methods raise `NotImplementedError` until a
later phase implements the joint posterior and stimulus selection (likely
built on `questplus`'s multi-dimensional QUEST+ support).
"""

from __future__ import annotations

from typing import Any

from vpsych.core.procedures.base import MultiParamProcedure, ThresholdEstimate


class QCSF(MultiParamProcedure):
    """qCSF adaptive procedure over the truncated log-parabola CSF model.

    Stimuli are specified as dicts with keys `"spatial_frequency_cpd"`
    (cycles per degree of visual angle) and `"contrast"` (Michelson
    contrast, unitless in [0, 1]).

    Args:
        spatial_frequency_values_cpd: Candidate spatial-frequency levels the
            procedure may select from, in cycles per degree.
        contrast_values: Candidate contrast levels the procedure may select
            from, unitless Michelson contrast in (0, 1].
        peak_gain_values: Candidate grid of peak sensitivity (gain)
            parameter values (log10 sensitivity units).
        peak_freq_values_cpd: Candidate grid of peak spatial-frequency
            parameter values, in cycles per degree.
        bandwidth_values_octaves: Candidate grid of bandwidth parameter
            values (full width at half maximum), in octaves.
        low_freq_truncation_values: Candidate grid of low-frequency
            truncation (log10 sensitivity) parameter values.
        guess_rate: Fixed guess (chance/floor) rate of the psychometric
            function relating contrast to detection probability at a given
            spatial frequency, e.g. `0.5` for 2AFC, `0.25` for 4AFC.
        lapse_rate: Fixed lapse rate assumed during fitting/selection.
        max_trials: Maximum number of trials before `finished` becomes
            `True` regardless of posterior precision, or `None` for no
            fixed cap.
        ci_level: Nominal coverage for summary-estimate confidence
            intervals, e.g. `0.95`.
    """

    def __init__(
        self,
        spatial_frequency_values_cpd: list[float],
        contrast_values: list[float],
        peak_gain_values: list[float],
        peak_freq_values_cpd: list[float],
        bandwidth_values_octaves: list[float],
        low_freq_truncation_values: list[float],
        guess_rate: float,
        lapse_rate: float = 0.02,
        max_trials: int | None = None,
        ci_level: float = 0.95,
    ) -> None:
        self.spatial_frequency_values_cpd = spatial_frequency_values_cpd
        self.contrast_values = contrast_values
        self.peak_gain_values = peak_gain_values
        self.peak_freq_values_cpd = peak_freq_values_cpd
        self.bandwidth_values_octaves = bandwidth_values_octaves
        self.low_freq_truncation_values = low_freq_truncation_values
        self.guess_rate = guess_rate
        self.lapse_rate = lapse_rate
        self.max_trials = max_trials
        self.ci_level = ci_level
        self._finished: bool = False
        raise NotImplementedError(
            "QCSF is a Phase-0 interface stub; implementation lands in a later phase."
        )

    @property
    def finished(self) -> bool:
        return self._finished

    def next_stimulus(self) -> dict[str, float]:
        raise NotImplementedError

    def update(self, stimulus: dict[str, float], correct: bool) -> None:
        raise NotImplementedError

    def estimate(self) -> ThresholdEstimate:
        raise NotImplementedError

    def state_dict(self) -> dict[str, Any]:
        raise NotImplementedError
