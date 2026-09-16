"""Simulated observers for validating procedures and tests without a human.

CI validates every adaptive procedure and test by running many simulated
sessions against an observer with a *known* ground-truth threshold, then
checking the procedure recovers that threshold without bias and with
correct confidence-interval coverage (see the plan's Verification section:
500 simulated runs, |bias| < 0.05 log units, ~95% CI coverage). This module
defines the observer contract those simulations use.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from vpsych.core.procedures.qcsf import log_contrast_sensitivity
from vpsych.core.psychometric import PsychometricFunction


@runtime_checkable
class SimulatedObserver(Protocol):
    """Protocol for a simulated observer that responds to presented stimuli.

    A test's trial loop (or a validation harness in `tests/`) calls
    `respond` in place of collecting a real keypress, so the same trial loop
    code path is exercised whether the response comes from a human or a
    simulated observer (see `vpsych.runner.__main__`'s `--simulate` option).
    """

    def respond(self, stimulus: dict[str, Any], rng: np.random.Generator) -> Any:
        """Produce a simulated response to a presented stimulus.

        Args:
            stimulus: The stimulus parameters for this trial (test-specific
                keys, e.g. `{"intensity": -1.2, "correct_side": "left"}` or
                `{"spatial_frequency_cpd": 2.0, "contrast": 0.1}`).
            rng: Random generator to use for any stochastic response
                behavior (see `vpsych.core.rng.make_rng`); must be the only
                source of randomness used, so simulated sessions are
                reproducible from a logged seed.

        Returns:
            A response value in whatever format the calling test's
            `score(response, stimulus_params)` expects (e.g. a chosen
            alternative label, direction, or boolean).
        """
        ...


class PsychometricObserver:
    """Simulated observer that responds according to a known psychometric function.

    Used to validate that adaptive procedures and psychometric-function
    fitting recover a *known* ground-truth threshold: construct with a
    `true_function` whose `threshold` is fixed by the test, run many
    simulated sessions/procedures against it, and confirm the fitted or
    procedure-estimated threshold clusters around `true_function.threshold`
    without systematic bias.

    Args:
        true_function: The ground-truth psychometric function this observer
            responds according to. `respond` reads the presented stimulus's
            intensity, evaluates `true_function.p_correct(intensity)`, and
            emits a correct response with that probability (and otherwise
            emits a uniformly random incorrect alternative among the
            remaining `n_afc - 1` choices).
        n_afc: Number of alternative-forced-choice response options (e.g.
            `2` for 2AFC, `8` for 8AFC Landolt-C orientation). Must be at
            least 2 and should be consistent with `true_function.guess`
            (typically `guess == 1 / n_afc`).
    """

    def __init__(self, true_function: PsychometricFunction, n_afc: int) -> None:
        if n_afc < 2:
            raise ValueError("n_afc must be >= 2")
        self.true_function = true_function
        self.n_afc = n_afc

    def respond(self, stimulus: dict[str, Any], rng: np.random.Generator) -> Any:
        """Respond according to `true_function.p_correct(stimulus["intensity"])`.

        Reads `stimulus["intensity"]` (required) and, optionally,
        `stimulus["correct_alternative"]` (the label of the correct
        response among `self.n_afc` alternatives; defaults to `0`) and
        `stimulus["alternatives"]` (the full list of alternative labels;
        defaults to `list(range(self.n_afc))`).

        Draws exactly one `rng.random()` call to decide correct/incorrect,
        and -- only on an incorrect trial -- one `rng.integers()` call to
        pick uniformly among the remaining `n_afc - 1` alternatives, so
        response sequences are fully reproducible from a seeded `rng`.

        Returns:
            `stimulus["correct_alternative"]` on a correct trial, or a
            uniformly random other element of `alternatives` on an
            incorrect trial.
        """
        intensity = float(stimulus["intensity"])
        p_correct = self.true_function.p_correct(intensity)
        is_correct = bool(rng.random() < p_correct)

        correct_answer = stimulus.get("correct_alternative", 0)
        alternatives = stimulus.get("alternatives", list(range(self.n_afc)))

        if is_correct:
            return correct_answer

        wrong_alternatives = [a for a in alternatives if a != correct_answer]
        if not wrong_alternatives:
            return correct_answer
        idx = int(rng.integers(len(wrong_alternatives)))
        return wrong_alternatives[idx]


class CSFObserver:
    """Simulated observer for qCSF validation with a known ground-truth CSF.

    Responds according to the same truncated log-parabola CSF model and
    fixed-slope Weibull-family psychometric function that
    `vpsych.core.procedures.qcsf.QCSF` fits (see
    `vpsych.core.procedures.qcsf.log_contrast_sensitivity` and that module's
    docstring for the model and the psychometric-function convention), so
    validation tests can construct an observer with known ground-truth CSF
    parameters and confirm `QCSF` recovers them (see the plan's
    verification section: AULCSF within 0.1 log units, averaged over runs).

    Args:
        peak_gain_log10: True peak sensitivity (log10 units).
        peak_freq_cpd: True peak spatial frequency, cycles per degree.
        bandwidth_octaves: True bandwidth (full width at half maximum),
            octaves.
        low_freq_truncation_log10: True low-frequency truncation depth,
            log10 units.
        n_afc: Number of alternative-forced-choice response options.
        slope: Fixed psychometric-function slope (log10 contrast units);
            should normally match the `psychometric_slope` the `QCSF`
            instance under test was constructed with.
        lapse_rate: Fixed lapse rate; should normally match the `QCSF`
            instance under test's `lapse_rate`.
    """

    def __init__(
        self,
        peak_gain_log10: float,
        peak_freq_cpd: float,
        bandwidth_octaves: float,
        low_freq_truncation_log10: float,
        n_afc: int,
        slope: float = 0.35,
        lapse_rate: float = 0.02,
    ) -> None:
        if n_afc < 2:
            raise ValueError("n_afc must be >= 2")
        self.peak_gain_log10 = peak_gain_log10
        self.peak_freq_cpd = peak_freq_cpd
        self.bandwidth_octaves = bandwidth_octaves
        self.low_freq_truncation_log10 = low_freq_truncation_log10
        self.n_afc = n_afc
        self.slope = slope
        self.lapse_rate = lapse_rate
        self.guess_rate = 1.0 / n_afc

    def p_correct(self, spatial_frequency_cpd: float, contrast: float) -> float:
        """True probability correct at a given spatial frequency and contrast."""
        log_cs = log_contrast_sensitivity(
            spatial_frequency_cpd,
            self.peak_gain_log10,
            self.peak_freq_cpd,
            self.bandwidth_octaves,
            self.low_freq_truncation_log10,
        )
        log10_threshold_contrast = -float(log_cs)
        z = (np.log10(contrast) - log10_threshold_contrast) / self.slope
        z = float(np.clip(z, -50.0, 50.0))
        f = 1.0 - 2.0 ** (-(2.0**z))
        p = self.guess_rate + (1.0 - self.guess_rate - self.lapse_rate) * f
        return float(np.clip(p, 0.0, 1.0))

    def respond(self, stimulus: dict[str, Any], rng: np.random.Generator) -> Any:
        """Respond according to `p_correct` at `stimulus`'s frequency/contrast.

        Reads `stimulus["spatial_frequency_cpd"]` and `stimulus["contrast"]`
        (both required). See `PsychometricObserver.respond` for the
        `correct_alternative`/`alternatives` convention and RNG-usage
        guarantee, which this method follows identically.
        """
        freq = float(stimulus["spatial_frequency_cpd"])
        contrast = float(stimulus["contrast"])
        p_correct = self.p_correct(freq, contrast)
        is_correct = bool(rng.random() < p_correct)

        correct_answer = stimulus.get("correct_alternative", 0)
        alternatives = stimulus.get("alternatives", list(range(self.n_afc)))

        if is_correct:
            return correct_answer

        wrong_alternatives = [a for a in alternatives if a != correct_answer]
        if not wrong_alternatives:
            return correct_answer
        idx = int(rng.integers(len(wrong_alternatives)))
        return wrong_alternatives[idx]
