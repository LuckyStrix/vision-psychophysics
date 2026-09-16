"""Scientific validation and unit tests for vpsych.core.procedures.constant_stimuli."""

from __future__ import annotations

import json

import numpy as np
import pytest

from vpsych.core.observers import PsychometricObserver
from vpsych.core.procedures.constant_stimuli import ConstantStimuli
from vpsych.core.psychometric import PsychometricFunction, intensity_at_p_correct

TRUE_SLOPE = 0.3
GUESS = 0.5
LAPSE = 0.02
TARGET_P = 0.75
TRUE_THRESHOLDS = [-1.4, -1.0, -0.5]  # >=3 true thresholds, per the plan's verification section
LEVELS_OFFSETS = np.linspace(-0.8, 0.8, 7)


def _true_function(threshold: float) -> PsychometricFunction:
    return PsychometricFunction(
        family="weibull",
        threshold=threshold,
        slope=TRUE_SLOPE,
        guess=GUESS,
        lapse=LAPSE,
        intensity_scale="log10",
    )


def _run(cs: ConstantStimuli, obs: PsychometricObserver, rng: np.random.Generator) -> None:
    while not cs.finished:
        x = cs.next_intensity()
        resp = obs.respond({"intensity": x, "correct_alternative": 0}, rng)
        cs.update(x, resp == 0)


def _make_cs(
    true_threshold: float, rng: np.random.Generator, n_reps: int, n_boot: int
) -> ConstantStimuli:
    levels = [float(true_threshold + off) for off in LEVELS_OFFSETS]
    return ConstantStimuli(
        intensity_levels=levels,
        intensity_units="log10_contrast",
        n_reps_per_level=n_reps,
        rng=rng,
        guess_rate=GUESS,
        target_p_correct=TARGET_P,
        n_bootstrap=n_boot,
    )


def test_state_dict_works_before_any_update() -> None:
    """The trial loop calls state_dict() on every trial, including practice
    trials before the first update() -- must never raise."""
    rng = np.random.default_rng(11)
    cs = _make_cs(-1.0, rng, n_reps=5, n_boot=20)
    state = cs.state_dict()
    json.dumps(state)
    assert state["n_trials"] == 0
    assert state["finished"] is False
    # estimate() legitimately cannot produce a fit with zero data -- it
    # should fail loudly and cleanly (RuntimeError), not crash obscurely.
    with pytest.raises(RuntimeError, match="before any trials"):
        cs.estimate()


def test_runs_expected_number_of_trials_and_finishes() -> None:
    fn = _true_function(-1.0)
    obs = PsychometricObserver(fn, n_afc=2)
    rng = np.random.default_rng(0)
    cs = _make_cs(-1.0, rng, n_reps=20, n_boot=50)
    n = 0
    while not cs.finished:
        x = cs.next_intensity()
        resp = obs.respond({"intensity": x, "correct_alternative": 0}, rng)
        cs.update(x, resp == 0)
        n += 1
    assert n == 20 * len(LEVELS_OFFSETS)


@pytest.mark.parametrize("true_threshold", TRUE_THRESHOLDS)
def test_recovery_fast(true_threshold: float) -> None:
    fn = _true_function(true_threshold)
    obs = PsychometricObserver(fn, n_afc=2)
    target_x = intensity_at_p_correct(fn, TARGET_P)
    rng = np.random.default_rng(1000 + TRUE_THRESHOLDS.index(true_threshold))
    biases = []
    for _ in range(3):
        cs = _make_cs(true_threshold, rng, n_reps=15, n_boot=25)
        _run(cs, obs, rng)
        biases.append(cs.estimate().value - target_x)
    assert abs(float(np.mean(biases))) < 0.3


@pytest.mark.slow
@pytest.mark.parametrize("true_threshold", TRUE_THRESHOLDS)
def test_recovery_slow(true_threshold: float) -> None:
    fn = _true_function(true_threshold)
    obs = PsychometricObserver(fn, n_afc=2)
    target_x = intensity_at_p_correct(fn, TARGET_P)
    rng = np.random.default_rng(2000 + TRUE_THRESHOLDS.index(true_threshold))
    biases = []
    covered = 0
    n_reps = 200
    for _ in range(n_reps):
        cs = _make_cs(true_threshold, rng, n_reps=15, n_boot=80)
        _run(cs, obs, rng)
        est = cs.estimate()
        biases.append(est.value - target_x)
        if est.ci_low <= target_x <= est.ci_high:
            covered += 1
    assert abs(float(np.mean(biases))) < 0.05
    coverage = covered / n_reps
    # n_bootstrap=80 (vs. bootstrap_ci's own 1000+ default) trades CI
    # precision for a slow test that finishes in minutes rather than tens of
    # minutes: at n_reps=200, the coverage estimate itself has a binomial SE
    # of ~0.025-0.03 even before accounting for the extra Monte Carlo noise
    # n_boot=80 adds to each individual CI, so a wide band (0.75, not the
    # ~0.88 that's appropriate for the full-precision bootstrap_ci coverage
    # check in test_psychometric.py) is needed to avoid flaking on ordinary
    # sampling variation (observed 0.82-0.92 across true_threshold/seeds
    # during development) while still catching a badly miscalibrated CI.
    assert 0.75 <= coverage <= 0.99


def test_estimate_ci_contains_point() -> None:
    fn = _true_function(-1.0)
    obs = PsychometricObserver(fn, n_afc=2)
    rng = np.random.default_rng(5)
    cs = _make_cs(-1.0, rng, n_reps=20, n_boot=50)
    _run(cs, obs, rng)
    est = cs.estimate()
    assert est.ci_low <= est.value <= est.ci_high


def test_estimate_before_any_trials_raises() -> None:
    rng = np.random.default_rng(6)
    cs = _make_cs(-1.0, rng, n_reps=5, n_boot=20)
    with pytest.raises(RuntimeError, match="before any trials"):
        cs.estimate()


def test_next_intensity_after_finished_raises() -> None:
    fn = _true_function(-1.0)
    obs = PsychometricObserver(fn, n_afc=2)
    rng = np.random.default_rng(7)
    cs = _make_cs(-1.0, rng, n_reps=2, n_boot=20)
    _run(cs, obs, rng)
    with pytest.raises(RuntimeError, match="exhausted"):
        cs.next_intensity()


def test_constructor_rejects_bad_arguments() -> None:
    rng = np.random.default_rng(8)
    with pytest.raises(ValueError, match="non-empty"):
        ConstantStimuli(intensity_levels=[], intensity_units="x", n_reps_per_level=5, rng=rng)
    with pytest.raises(ValueError, match="n_reps_per_level"):
        ConstantStimuli(
            intensity_levels=[0.0, 1.0], intensity_units="x", n_reps_per_level=0, rng=rng
        )
    with pytest.raises(ValueError, match="psychometric_family"):
        ConstantStimuli(
            intensity_levels=[0.0, 1.0],
            intensity_units="x",
            n_reps_per_level=5,
            rng=rng,
            psychometric_family="bogus",
        )


def test_state_dict_is_json_serializable() -> None:
    fn = _true_function(-1.0)
    obs = PsychometricObserver(fn, n_afc=2)
    rng = np.random.default_rng(9)
    cs = _make_cs(-1.0, rng, n_reps=10, n_boot=30)
    x = cs.next_intensity()
    resp = obs.respond({"intensity": x, "correct_alternative": 0}, rng)
    cs.update(x, resp == 0)
    json.dumps(cs.state_dict())
    _run(cs, obs, rng)
    json.dumps(cs.estimate().model_dump())


def test_determinism_same_seed_same_trial_sequence() -> None:
    def run(seed: int) -> list[float]:
        rng = np.random.default_rng(seed)
        cs = _make_cs(-1.0, rng, n_reps=10, n_boot=20)
        intensities = []
        while not cs.finished:
            x = cs.next_intensity()
            intensities.append(x)
            cs.update(x, correct=True)
        return intensities

    seq1 = run(123)
    seq2 = run(123)
    assert seq1 == seq2
