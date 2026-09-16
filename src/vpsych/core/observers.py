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
        self.true_function = true_function
        self.n_afc = n_afc

    def respond(self, stimulus: dict[str, Any], rng: np.random.Generator) -> Any:
        raise NotImplementedError
