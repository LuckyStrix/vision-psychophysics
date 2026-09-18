"""`MultiParamProcedure` interleaving three independent QUEST+ procedures.

The Cambridge Colour Test's trivector design measures a discrimination
threshold along three confusion-line axes (protan, deutan, tritan) in one
interleaved run, choosing which axis to test at random each trial (Regan,
Reffin & Mollon 1994). `vpsych.tests_catalog.base.PsychophysicalTest
.make_procedure` returns a single `AdaptiveProcedure | MultiParamProcedure`,
and neither `vpsych.core.trial_loop.TrialLoop` nor either single-procedure
type supports interleaving three independent procedures out of the box --
this module is the small composite `MultiParamProcedure` this test needs,
implemented entirely within this package (no changes to
`vpsych.core.procedures`).

`TrivectorProcedure` wraps three `QuestPlusProcedure` instances (one per
axis in `AXES` order), each running independently on its own intensity/
threshold/slope grid (axis grids can differ in extent, since each
confusion-line axis's gamut-limited displacement ceiling is generally
different -- see the package's ``__init__`` module for how those grids are
built). `next_stimulus()` draws the axis uniformly at random (via the
caller-supplied `rng`, which should be the same `rng` object the owning
test's `present()` uses via `trial_ctx["rng"]`, per this project's "rng is
the only source of randomness, and is logged" convention) and delegates to
that axis's `next_intensity()`; `update()` routes the outcome back to the
axis recorded in the stimulus dict's `"axis"` field, so
`vpsych.tests_catalog.base.PsychophysicalTest.summarize`'s "replay a fresh
procedure over recorded trials" pattern
(see `docs/WRITING_A_TEST.md`, section 10) works without needing to
re-derive which axis a historical trial belonged to.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vpsych.core.procedures.base import MultiParamProcedure, ThresholdEstimate
from vpsych.core.procedures.questplus_procedure import QuestPlusProcedure

#: The three confusion-line axes, in the fixed order this procedure encodes
#: as the stimulus dict's integer `"axis"` field (0 = protan, 1 = deutan,
#: 2 = tritan).
AXES: tuple[str, str, str] = ("protan", "deutan", "tritan")


class TrivectorProcedure(MultiParamProcedure):
    """Interleaves three per-axis `QuestPlusProcedure` instances.

    Args:
        axis_procedures: A `QuestPlusProcedure` for each of `AXES` (exactly
            those three keys, no more, no fewer). Each should be constructed
            with `max_trials=None` -- overall stopping is controlled by this
            wrapper's own `max_trials_total`, not any individual axis
            procedure's own trial count (an axis that is drawn less often
            than others, purely by chance, should not stop early on its
            own).
        rng: Seeded generator used *only* to choose which axis to test each
            trial (`next_stimulus`); should be the same object the owning
            test's `present()` receives via `trial_ctx["rng"]`, so the axis
            sequence is reproducible from the session's logged seed and its
            draws interleave deterministically with any other draws
            `present()` itself makes.
        max_trials_total: Total number of trials (summed across all three
            axes) before `finished` becomes `True`.
        ci_level: Nominal coverage for the combined estimate's/each axis
            estimate's confidence interval, e.g. `0.95`.

    `next_stimulus()` returns `{"axis": <0, 1, or 2>, "intensity": <log10
    displacement, in the presented axis's own units>}`; `update()` reads
    `stimulus["axis"]` to route the outcome to the correct axis's
    `QuestPlusProcedure.update`.
    """

    def __init__(
        self,
        axis_procedures: dict[str, QuestPlusProcedure],
        rng: np.random.Generator,
        max_trials_total: int,
        ci_level: float = 0.95,
    ) -> None:
        if set(axis_procedures) != set(AXES):
            raise ValueError(
                f"axis_procedures must have exactly the keys {AXES}, got {sorted(axis_procedures)}"
            )
        if max_trials_total < 1:
            raise ValueError(f"max_trials_total must be >= 1, got {max_trials_total}")
        self.axis_procedures = axis_procedures
        self.rng = rng
        self.max_trials_total = max_trials_total
        self.ci_level = ci_level
        self._n_trials = 0
        self._finished = False
        self._last_axis: int | None = None

    @property
    def finished(self) -> bool:
        return self._finished

    def next_stimulus(self) -> dict[str, float]:
        axis_idx = int(self.rng.integers(len(AXES)))
        axis = AXES[axis_idx]
        intensity = self.axis_procedures[axis].next_intensity()
        self._last_axis = axis_idx
        return {"axis": float(axis_idx), "intensity": intensity}

    def update(self, stimulus: dict[str, float], correct: bool) -> None:
        if self._finished:
            return
        axis_idx = round(stimulus["axis"])
        axis = AXES[axis_idx]
        self.axis_procedures[axis].update(float(stimulus["intensity"]), correct)
        self._n_trials += 1
        if self._n_trials >= self.max_trials_total:
            self._finished = True

    def estimate(self) -> ThresholdEstimate:
        """Combined trivector estimate: geometric mean of the three axis thresholds.

        Each axis's `QuestPlusProcedure.estimate().value` is already, by
        this project's psychometric-function convention (every family
        satisfies `F(0) == 0.5`, see `docs/METHODS.md`), the intensity at
        `p_correct` exactly halfway between the guess rate and `1 -
        lapse_rate` -- so no further conversion via
        `vpsych.core.psychometric.intensity_at_p_correct` is needed to reach
        that documented criterion; see this test's
        `docs/methods/color_discrimination.md` for the derivation and how
        it differs from the original CCT's 11-reversal staircase criterion.

        The primary (combined) `value` is the *arithmetic* mean of the three
        axes' log10-displacement thresholds -- equivalently, the
        *geometric* mean of their linear (x1e-4 u'v') displacement
        thresholds, since `10**mean(log10(t_i)) == geomean(t_i)`. `ci_low`/
        `ci_high` are the same mean taken over each axis's own credible
        bounds (a documented approximation, not a fully joint interval).
        Full per-axis thresholds/CIs/slopes/lapse rates are in `extra`.
        """
        if self._n_trials == 0:
            raise RuntimeError(
                "TrivectorProcedure.estimate() called before any trials were recorded"
            )

        per_axis_estimate = {axis: self.axis_procedures[axis].estimate() for axis in AXES}
        mean_log10 = float(np.mean([e.value for e in per_axis_estimate.values()]))
        ci_low = float(np.mean([e.ci_low for e in per_axis_estimate.values()]))
        ci_high = float(np.mean([e.ci_high for e in per_axis_estimate.values()]))

        return ThresholdEstimate(
            value=mean_log10,
            ci_low=ci_low,
            ci_high=ci_high,
            ci_level=self.ci_level,
            units="log10_uv_displacement_x1e4",
            method="trivector_geometric_mean_of_axis_quest_plus_thresholds",
            extra={
                "n_trials": self._n_trials,
                "combination": (
                    "value is the arithmetic mean of the three axes' log10-displacement "
                    "thresholds, i.e. the geometric mean of their linear (u'v' x1e-4) "
                    "displacement thresholds; ci_low/ci_high are the same mean taken over "
                    "each axis's own credible bound (not a joint interval)."
                ),
                "per_axis": {
                    axis: {
                        "threshold_log10_uv_x1e4": est.value,
                        "threshold_uv_x1e4": float(10.0**est.value),
                        "ci_low_log10_uv_x1e4": est.ci_low,
                        "ci_high_log10_uv_x1e4": est.ci_high,
                        "ci_low_uv_x1e4": float(10.0**est.ci_low),
                        "ci_high_uv_x1e4": float(10.0**est.ci_high),
                        "slope": est.extra.get("slope"),
                        "lapse_rate": est.extra.get("lapse_rate"),
                        "n_trials": self.axis_procedures[axis].state_dict()["n_trials"],
                    }
                    for axis, est in per_axis_estimate.items()
                },
            },
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "n_trials": self._n_trials,
            "finished": self._finished,
            "last_axis": self._last_axis,
            "per_axis": {axis: proc.state_dict() for axis, proc in self.axis_procedures.items()},
        }
