"""Regressions for the third review round (fixing the second review's own findings)."""

from __future__ import annotations

import numpy as np

from vpsych.core.trial_loop import _contrasting_text_color


def test_contrasting_text_color_picks_black_on_white_and_white_on_black() -> None:
    assert _contrasting_text_color(1.0) == (-1.0, -1.0, -1.0)
    assert _contrasting_text_color((1.0, 1.0, 1.0)) == (-1.0, -1.0, -1.0)
    assert _contrasting_text_color(-1.0) == (1.0, 1.0, 1.0)
    assert _contrasting_text_color((-1.0, -1.0, -1.0)) == (1.0, 1.0, 1.0)


def test_qcsf_ci_always_contains_the_point_estimate_even_for_near_chance_data() -> None:
    from vpsych.core.procedures.qcsf import QCSF

    rng = np.random.default_rng(0)
    proc = QCSF(**QCSF.default_grids(), guess_rate=0.5, lapse_rate=0.02)
    # Near-chance responses: a diffuse posterior is exactly the regime where the plug-in
    # point estimate and the percentile-of-grid CI can disagree enough to put the point
    # estimate outside its own reported interval.
    for _ in range(15):
        stim = proc.next_stimulus()
        proc.update(stim, bool(rng.random() < 0.5))
    est = proc.estimate()
    assert est.ci_low <= est.value <= est.ci_high
