"""Per-axis simulated observer for the trivector color-discrimination task.

`vpsych.runner.__main__`'s `--simulate`/`--simulate-config` machinery
(`parse_simulated_observer_spec`/`build_simulated_observer`/
`_simulated_observer_from_json`) originally dispatched only two observer
*kinds*, `"psychometric"` (a single ground-truth `PsychometricFunction`)
and `"csf"` -- neither can express three independent per-axis ground-truth
thresholds keyed by `stimulus["axis"]`, and `build_simulated_observer`'s
`kind` dispatch used to be a closed `if/elif` that could not be extended
from outside `vpsych.runner.__main__` at all, so `TrivectorObserver` (and
any future multi-dimensional observer) was only usable by constructing it
directly and injecting it via `run_session(..., simulated_observer=...)`,
bypassing `--simulate`/`--simulate-config` entirely.

That dispatch is now a registry (`vpsych.runner.__main__
.register_simulated_observer_kind`); this module registers `"trivector"` as
an import-time side effect below (mirroring `@register_test`'s own
import-time registration pattern), so `TrivectorObserver` is resolvable
through the normal CLI/JSON path like any built-in kind --
`"trivector:threshold_protan=<f>,threshold_deutan=<f>,threshold_tritan=<f>"`
(optionally with per-axis `slope_<axis>`/`guess_<axis>`/`lapse_<axis>`, or
shared `slope=`/`guess=`/`lapse=` fallbacks) -- see
`_build_trivector_observer` below. `color_discrimination/__init__.py`
imports this module (for exactly this registration side effect), so the
registration happens whenever `vpsych.tests_catalog.base.discover_tests()`
walks the catalog, the same guarantee `@register_test` itself relies on.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vpsych.core.psychometric import PsychometricFunction
from vpsych.runner.__main__ import register_simulated_observer_kind
from vpsych.tests_catalog.color_discrimination.discs import RESPONSE_KEYS
from vpsych.tests_catalog.color_discrimination.procedure import AXES

#: Defaults for `_build_trivector_observer`'s optional per-axis parameters,
#: matching this package's own `GUESS_RATE`/default slope/lapse grids (see
#: `color_discrimination/__init__.py`; not imported directly here to avoid
#: a circular import, since that module imports this one).
_DEFAULT_GUESS_RATE = 0.25
_DEFAULT_SLOPE = 0.4
_DEFAULT_LAPSE_RATE = 0.02


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


def _build_trivector_observer(params: dict[str, float]) -> TrivectorObserver:
    """Build a `TrivectorObserver` from a flat `dict[str, float]` params (see module docstring).

    Required: `threshold_<axis>` for each of `AXES` ("protan", "deutan",
    "tritan"). Optional per-axis `slope_<axis>`/`guess_<axis>`/
    `lapse_<axis>`, falling back to shared `slope=`/`guess=`/`lapse=`
    (further falling back to `_DEFAULT_SLOPE`/`_DEFAULT_GUESS_RATE`/
    `_DEFAULT_LAPSE_RATE` if neither is given), matching
    `build_simulated_observer`'s own `params.get(key, default)` convention
    for its built-in kinds.
    """
    functions: dict[str, PsychometricFunction] = {}
    missing = [f"threshold_{axis}" for axis in AXES if f"threshold_{axis}" not in params]
    if missing:
        raise ValueError(f"trivector observer spec is missing required parameter(s) {missing}.")
    shared_slope = params.get("slope", _DEFAULT_SLOPE)
    shared_guess = params.get("guess", _DEFAULT_GUESS_RATE)
    shared_lapse = params.get("lapse", _DEFAULT_LAPSE_RATE)
    for axis in AXES:
        functions[axis] = PsychometricFunction(
            family="weibull",
            threshold=params[f"threshold_{axis}"],
            slope=params.get(f"slope_{axis}", shared_slope),
            guess=params.get(f"guess_{axis}", shared_guess),
            lapse=params.get(f"lapse_{axis}", shared_lapse),
            intensity_scale="log10",
        )
    return TrivectorObserver(functions)


register_simulated_observer_kind("trivector", _build_trivector_observer)
