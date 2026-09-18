"""Per-axis simulated observer for the trivector color-discrimination task.

`vpsych.runner.__main__`'s `--simulate`/`--simulate-config` machinery
(`parse_simulated_observer_spec`/`build_simulated_observer`/
`_simulated_observer_from_json`) only dispatches two observer *kinds*,
`"psychometric"` (a single ground-truth `PsychometricFunction`) and `"csf"`
-- neither can express three independent per-axis ground-truth thresholds
keyed by `stimulus["axis"]`, and `_simulated_observer_from_json`'s `kind`
dispatch is a closed `if/elif` in `vpsych.runner.__main__.build_simulated_observer`
that cannot be extended from outside that module. This is a real limitation
of the runner's current simulated-observer configuration surface for any
multi-axis test (reported upstream; see this test's package-level report).

`TrivectorObserver` is therefore not resolvable via `--simulate-config`'s
CLI/JSON path at all. It is fully usable, however, by constructing it
directly and passing it to `vpsych.runner.__main__.run_session(...,
simulated_observer=...)` -- exactly how
`tests/tests_catalog/test_color_discrimination.py`'s simulated end-to-end
recovery test exercises it (see that test for a worked example), which is
the documented way to drive this test's own three-axis validation without a
CLI change.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vpsych.core.psychometric import PsychometricFunction
from vpsych.tests_catalog.color_discrimination.discs import RESPONSE_KEYS
from vpsych.tests_catalog.color_discrimination.procedure import AXES


class TrivectorObserver:
    """Simulated observer with an independent ground-truth threshold per confusion axis.

    Implements the `vpsych.core.observers.SimulatedObserver` protocol (plus
    `decide_correct`, the decoupled correct/incorrect decision path every
    built-in observer in `vpsych.core.observers` also implements -- see that
    module's docstring).

    Args:
        functions: A `PsychometricFunction` (family `"weibull"`,
            `intensity_scale="log10"`, `guess=0.25` matching this test's
            4AFC design) for each of `AXES` (exactly those three keys).
    """

    def __init__(self, functions: dict[str, PsychometricFunction]) -> None:
        if set(functions) != set(AXES):
            raise ValueError(
                f"functions must have exactly the keys {AXES}, got {sorted(functions)}"
            )
        self.functions = functions

    def decide_correct(self, stimulus: dict[str, Any], rng: np.random.Generator) -> bool:
        """Decide correct/incorrect for a presented stimulus, per its own axis's ground truth.

        Reads `stimulus["axis"]` (0, 1, or 2 -- or a float that rounds to
        one of those) and `stimulus["intensity"]` (log10 displacement, in
        that axis's own units), both required.
        """
        axis_idx = round(float(stimulus["axis"]))
        axis = AXES[axis_idx]
        p_correct = self.functions[axis].p_correct(float(stimulus["intensity"]))
        return bool(rng.random() < p_correct)

    def respond(self, stimulus: dict[str, Any], rng: np.random.Generator) -> Any:
        """Respond with a gap-orientation key, via `decide_correct` (see that method).

        Provided for full `SimulatedObserver` protocol compliance; this
        test's `present()` uses the decoupled `decide_correct` +
        `PsychophysicalTest.simulated_response` path instead (see
        `docs/WRITING_A_TEST.md`, section 9), so `respond` itself is not on
        this test's own hot path, only exercised by generic protocol checks.
        """
        is_correct = self.decide_correct(stimulus, rng)
        correct_response = stimulus.get("correct_response", RESPONSE_KEYS[0])
        if is_correct:
            return correct_response
        alternatives = [k for k in RESPONSE_KEYS if k != correct_response]
        idx = int(rng.integers(len(alternatives)))
        return alternatives[idx]
