"""Scientific validation and unit tests for vpsych.core.procedures.questplus_procedure."""

from __future__ import annotations

import json

import numpy as np
import pytest

from vpsych.core.observers import PsychometricObserver
from vpsych.core.procedures.base import ThresholdEstimate
from vpsych.core.procedures.questplus_procedure import (
    QuestPlusProcedure,
    questplus_weibull_report_at_p,
    questplus_weibull_x_at_p,
)
from vpsych.core.psychometric import PsychometricFunction

TRUE_SLOPE = 0.3
GUESS = 0.5
LAPSE = 0.02
TRUE_THRESHOLDS = [-1.4, -1.0, -0.5]  # >=3 true thresholds, per the plan's verification section
INTENSITY_VALUES = list(np.linspace(-2.5, 0.5, 61))
THRESHOLD_VALUES = list(np.linspace(-2.2, 0.2, 25))
SLOPE_VALUES = list(np.linspace(0.1, 0.8, 8))
LAPSE_VALUES = [0.0, 0.02, 0.04]


def _true_function(threshold: float) -> PsychometricFunction:
    return PsychometricFunction(
        family="weibull",
        threshold=threshold,
        slope=TRUE_SLOPE,
        guess=GUESS,
        lapse=LAPSE,
        intensity_scale="log10",
    )


def _make_proc(max_trials: int = 48) -> QuestPlusProcedure:
    return QuestPlusProcedure(
        intensity_values=INTENSITY_VALUES,
        intensity_units="log10_contrast",
        threshold_values=THRESHOLD_VALUES,
        slope_values=SLOPE_VALUES,
        guess_rate=GUESS,
        lapse_rate_values=LAPSE_VALUES,
        function="weibull",
        max_trials=max_trials,
    )


def _run(proc: QuestPlusProcedure, obs: PsychometricObserver, rng: np.random.Generator) -> None:
    while not proc.finished:
        x = proc.next_intensity()
        resp = obs.respond({"intensity": x, "correct_alternative": 0}, rng)
        proc.update(x, resp == 0)


def test_logistic_family_rejected() -> None:
    with pytest.raises(ValueError, match="logistic"):
        QuestPlusProcedure(
            intensity_values=INTENSITY_VALUES,
            intensity_units="x",
            threshold_values=THRESHOLD_VALUES,
            slope_values=SLOPE_VALUES,
            guess_rate=GUESS,
            lapse_rate_values=LAPSE_VALUES,
            function="logistic",
        )


def test_constructor_rejects_empty_domains() -> None:
    with pytest.raises(ValueError):
        QuestPlusProcedure(
            intensity_values=[],
            intensity_units="x",
            threshold_values=THRESHOLD_VALUES,
            slope_values=SLOPE_VALUES,
            guess_rate=GUESS,
            lapse_rate_values=LAPSE_VALUES,
        )


def test_state_dict_works_before_any_update() -> None:
    """The trial loop calls state_dict() on every trial, including practice
    trials before the first update() -- must never raise."""
    proc = _make_proc(max_trials=30)
    state = proc.state_dict()
    json.dumps(state)
    assert state["n_trials"] == 0
    assert state["finished"] is False
    # estimate() before any data is well-defined here (the prior mean).
    est = proc.estimate()
    assert est.ci_low <= est.value <= est.ci_high


def test_finishes_at_max_trials() -> None:
    fn = _true_function(-1.0)
    obs = PsychometricObserver(fn, n_afc=2)
    rng = np.random.default_rng(0)
    proc = _make_proc(max_trials=30)
    _run(proc, obs, rng)
    assert proc.finished
    assert proc.state_dict()["n_trials"] == 30


def test_target_entropy_stops_early() -> None:
    fn = _true_function(-1.0)
    obs = PsychometricObserver(fn, n_afc=2)
    rng = np.random.default_rng(1)
    proc = QuestPlusProcedure(
        intensity_values=INTENSITY_VALUES,
        intensity_units="log10_contrast",
        threshold_values=THRESHOLD_VALUES,
        slope_values=SLOPE_VALUES,
        guess_rate=GUESS,
        lapse_rate_values=LAPSE_VALUES,
        max_trials=200,
        target_entropy=5.5,
    )
    _run(proc, obs, rng)
    assert proc.finished
    assert proc.state_dict()["n_trials"] < 200


@pytest.mark.parametrize("true_threshold", TRUE_THRESHOLDS)
def test_recovery_fast(true_threshold: float) -> None:
    fn = _true_function(true_threshold)
    obs = PsychometricObserver(fn, n_afc=2)
    rng = np.random.default_rng(1000 + TRUE_THRESHOLDS.index(true_threshold))
    biases = []
    for _ in range(4):
        proc = _make_proc(max_trials=25)
        _run(proc, obs, rng)
        biases.append(proc.estimate().value - true_threshold)
    assert abs(float(np.mean(biases))) < 0.35


@pytest.mark.slow
@pytest.mark.parametrize("true_threshold", TRUE_THRESHOLDS)
def test_recovery_slow(true_threshold: float) -> None:
    fn = _true_function(true_threshold)
    obs = PsychometricObserver(fn, n_afc=2)
    rng = np.random.default_rng(2000 + TRUE_THRESHOLDS.index(true_threshold))
    biases = []
    covered = 0
    n_reps = 200
    for _ in range(n_reps):
        proc = _make_proc(max_trials=48)
        _run(proc, obs, rng)
        est = proc.estimate()
        biases.append(est.value - true_threshold)
        if est.ci_low <= true_threshold <= est.ci_high:
            covered += 1
    assert abs(float(np.mean(biases))) < 0.05
    coverage = covered / n_reps
    # Upper-bounded loosely (not tightly at ~0.99): the equal-tailed credible
    # interval taken from quantiles of the marginal threshold posterior can be
    # a touch conservative at these trial counts -- a wider interval than
    # nominal is a benign, non-flaky outcome, unlike under-coverage, so it
    # isn't penalized tightly here.
    assert 0.88 <= coverage <= 1.0


def test_norm_cdf_family_runs() -> None:
    fn = PsychometricFunction(
        family="norm_cdf",
        threshold=-1.0,
        slope=0.3,
        guess=GUESS,
        lapse=LAPSE,
        intensity_scale="linear",
    )
    obs = PsychometricObserver(fn, n_afc=2)
    rng = np.random.default_rng(2)
    proc = QuestPlusProcedure(
        intensity_values=INTENSITY_VALUES,
        intensity_units="log10_contrast",
        threshold_values=THRESHOLD_VALUES,
        slope_values=SLOPE_VALUES,
        guess_rate=GUESS,
        lapse_rate_values=LAPSE_VALUES,
        function="norm_cdf",
        max_trials=25,
    )
    _run(proc, obs, rng)
    est = proc.estimate()
    assert est.value == pytest.approx(-1.0, abs=0.5)


def test_estimate_ci_contains_point() -> None:
    fn = _true_function(-1.0)
    obs = PsychometricObserver(fn, n_afc=2)
    rng = np.random.default_rng(3)
    proc = _make_proc(max_trials=30)
    _run(proc, obs, rng)
    est = proc.estimate()
    assert est.ci_low <= est.value <= est.ci_high


def test_state_dict_is_compact_and_json_serializable() -> None:
    fn = _true_function(-1.0)
    obs = PsychometricObserver(fn, n_afc=2)
    rng = np.random.default_rng(4)
    proc = _make_proc(max_trials=20)
    _run(proc, obs, rng)
    state = proc.state_dict()
    json.dumps(state)
    # Compact: mean/SD per free parameter (threshold, slope, lapse_rate), not
    # the full posterior array over the (25 x 8 x 3)-point grid.
    assert set(state["posterior_mean"].keys()) == {"threshold", "slope", "lapse_rate"}
    assert all(isinstance(v, float) for v in state["posterior_mean"].values())
    json.dumps(proc.estimate().model_dump())


def test_determinism_same_seed_same_trial_sequence() -> None:
    fn = _true_function(-1.0)

    def run(seed: int) -> list[float]:
        obs = PsychometricObserver(fn, n_afc=2)
        rng = np.random.default_rng(seed)
        proc = _make_proc(max_trials=25)
        intensities = []
        while not proc.finished:
            x = proc.next_intensity()
            intensities.append(x)
            resp = obs.respond({"intensity": x, "correct_alternative": 0}, rng)
            proc.update(x, resp == 0)
        return intensities

    seq1 = run(123)
    seq2 = run(123)
    assert seq1 == seq2


# --- Criterion conversion (questplus_weibull_x_at_p / report_at_p / method) ---


def test_questplus_weibull_x_at_p_recovers_threshold_at_its_own_natural_point() -> None:
    """At `p_target` == questplus's own ~63.2%-of-range anchor, `x == threshold` exactly."""
    threshold, slope, guess, lapse = -1.0, 0.4, 0.5, 0.02
    natural_p = guess + (1.0 - guess - lapse) * (1.0 - np.exp(-1.0))
    x = questplus_weibull_x_at_p(threshold, slope, guess, lapse, natural_p)
    assert x == pytest.approx(threshold, abs=1e-9)


def test_questplus_weibull_x_at_p_matches_questplus_own_formula() -> None:
    """Round-trips: invert for x at p, then evaluate questplus's own forward formula at x."""
    threshold, slope, guess, lapse = -0.8, 0.35, 0.25, 0.03
    for p_target in (0.4, 0.5, 0.6, 0.75, 0.9):
        x = questplus_weibull_x_at_p(threshold, slope, guess, lapse, p_target)
        p_forward = (
            1.0 - lapse - (1.0 - guess - lapse) * np.exp(-(10.0 ** (slope * (x - threshold))))
        )
        assert p_forward == pytest.approx(p_target, abs=1e-6)


def test_questplus_weibull_x_at_p_differs_from_vpsych_own_intensity_at_p_correct() -> None:
    """The whole point of item 1: these two inversions are NOT interchangeable."""
    from vpsych.core.psychometric import intensity_at_p_correct

    threshold, slope, guess, lapse = -1.0, 0.4, 0.5, 0.02
    p_target = 0.75
    x_native = questplus_weibull_x_at_p(threshold, slope, guess, lapse, p_target)
    fn = PsychometricFunction(
        family="weibull",
        threshold=threshold,
        slope=slope,
        guess=guess,
        lapse=lapse,
        intensity_scale="log10",
    )
    x_vpsych_family = intensity_at_p_correct(fn, p_target)
    assert x_native != pytest.approx(x_vpsych_family, abs=1e-3)


def test_questplus_weibull_report_at_p_shifts_ci_by_same_amount_as_point() -> None:
    est = ThresholdEstimate(
        value=-1.0,
        ci_low=-1.3,
        ci_high=-0.7,
        ci_level=0.95,
        units="log10_contrast",
        method="quest_plus_posterior_mean",
        extra={"slope": 0.4, "lapse_rate": 0.02},
    )
    reported, ci_low, ci_high = questplus_weibull_report_at_p(est, guess=0.5, p_target=0.75)
    shift = reported - est.value
    assert ci_low == pytest.approx(est.ci_low + shift)
    assert ci_high == pytest.approx(est.ci_high + shift)
    assert ci_low <= reported <= ci_high


def test_procedure_intensity_at_p_correct_recovers_true_threshold_at_native_anchor() -> None:
    """Sanity: the method's result at the native anchor p matches raw estimate().value."""
    fn = _true_function(-1.0)
    obs = PsychometricObserver(fn, n_afc=2)
    rng = np.random.default_rng(7)
    proc = _make_proc(max_trials=40)
    _run(proc, obs, rng)
    est = proc.estimate()
    lapse = float(est.extra["lapse_rate"])
    natural_p = GUESS + (1.0 - GUESS - lapse) * (1.0 - np.exp(-1.0))
    reported, ci_low, ci_high = proc.intensity_at_p_correct(natural_p)
    assert reported == pytest.approx(est.value, abs=1e-6)
    assert ci_low == pytest.approx(est.ci_low, abs=1e-6)
    assert ci_high == pytest.approx(est.ci_high, abs=1e-6)
    # And a different p_target genuinely gives a different reported value, since
    # slope != 0 -- guards against a no-op stub implementation.
    reported_75, _, _ = proc.intensity_at_p_correct(0.75)
    assert reported_75 != pytest.approx(reported, abs=1e-6)


def test_procedure_intensity_at_p_correct_rejects_norm_cdf() -> None:
    proc = QuestPlusProcedure(
        intensity_values=INTENSITY_VALUES,
        intensity_units="x",
        threshold_values=THRESHOLD_VALUES,
        slope_values=SLOPE_VALUES,
        guess_rate=GUESS,
        lapse_rate_values=LAPSE_VALUES,
        function="norm_cdf",
    )
    with pytest.raises(NotImplementedError):
        proc.intensity_at_p_correct(0.75)
