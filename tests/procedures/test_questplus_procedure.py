"""Scientific validation and unit tests for vpsych.core.procedures.questplus_procedure."""

from __future__ import annotations

import json

import numpy as np
import pytest

from vpsych.core.observers import PsychometricObserver
from vpsych.core.procedures.questplus_procedure import QuestPlusProcedure
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
    assert 0.88 <= coverage <= 0.99


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
