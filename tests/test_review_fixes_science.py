"""Regressions for the second review round: estimation, timeouts, geometry sanity."""

from __future__ import annotations

import numpy as np
import pandas as pd

from vpsych.app.viewmodels.calibration_wizard import GeometryStepState
from vpsych.core.procedures.questplus_procedure import QuestPlusProcedure
from vpsych.data.quality import check_timeouts
from vpsych.tests_catalog.base import split_scored_trials


def _proc(pad: float = 0.25) -> QuestPlusProcedure:
    grid = [float(v) for v in np.linspace(0.0, 1.3, 14)]
    return QuestPlusProcedure(
        intensity_values=grid,
        intensity_units="logMAR",
        threshold_values=grid,
        slope_values=[1.0, 3.0],
        guess_rate=0.125,
        lapse_rate_values=[0.02],
        threshold_pad_frac=pad,
    )


def test_perfect_observer_gets_a_non_degenerate_credible_interval() -> None:
    p = _proc()
    for _ in range(30):
        p.update(p.next_intensity(), True)
    est = p.estimate()
    assert est.ci_low < est.ci_high
    assert est.ci_low <= est.value <= est.ci_high


def test_threshold_grid_is_padded_beyond_the_intensity_range() -> None:
    p = _proc()
    assert min(p.threshold_values) < 0.0 and max(p.threshold_values) > 1.3
    assert _proc(pad=0.0).threshold_values[0] == 0.0


def test_split_scored_trials_drops_timeouts_but_keeps_catch_out() -> None:
    main = pd.DataFrame(
        {
            "trial_index": [2, 0, 1, 3],
            "is_catch": [False, False, False, True],
            "response": [90, None, 180, None],
        }
    )
    scored, n_timeouts = split_scored_trials(main)
    assert list(scored["trial_index"]) == [1, 2]
    assert n_timeouts == 1


def test_many_timeouts_flagged_few_not() -> None:
    assert check_timeouts(1, 30) == []
    assert check_timeouts(6, 30)[0].code == "many_timeouts"


def test_height_warning_catches_an_unmeasured_default() -> None:
    g = GeometryStepState()
    g.width_px, g.height_px, g.width_cm, g.height_cm = 1920, 1200, 34.4, 30.0
    assert g.derived_height_cm() is not None and abs(g.derived_height_cm() - 21.5) < 0.1
    assert g.height_warning() is not None
    g.height_cm = 21.5
    assert g.height_warning() is None
