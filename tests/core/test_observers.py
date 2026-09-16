"""Unit tests for vpsych.core.observers: PsychometricObserver and CSFObserver."""

from __future__ import annotations

import numpy as np
import pytest

from vpsych.core.observers import CSFObserver, PsychometricObserver
from vpsych.core.psychometric import PsychometricFunction


def _true_function() -> PsychometricFunction:
    return PsychometricFunction(
        family="weibull", threshold=-1.0, slope=0.3, guess=0.5, lapse=0.02, intensity_scale="log10"
    )


def test_psychometric_observer_rejects_n_afc_below_2() -> None:
    with pytest.raises(ValueError, match="n_afc"):
        PsychometricObserver(_true_function(), n_afc=1)


def test_psychometric_observer_response_rate_matches_p_correct() -> None:
    fn = _true_function()
    obs = PsychometricObserver(fn, n_afc=2)
    rng = np.random.default_rng(0)
    x = -1.0  # at threshold -> p = 0.5 + 0.48*0.5 = 0.74
    expected_p = fn.p_correct(x)
    n = 4000
    n_correct = sum(
        1 for _ in range(n) if obs.respond({"intensity": x, "correct_alternative": 0}, rng) == 0
    )
    observed_p = n_correct / n
    assert observed_p == pytest.approx(expected_p, abs=0.03)


def test_psychometric_observer_incorrect_picks_among_other_alternatives() -> None:
    fn = PsychometricFunction(
        family="weibull", threshold=10.0, slope=0.3, guess=0.25, lapse=0.0, intensity_scale="log10"
    )
    obs = PsychometricObserver(fn, n_afc=4)
    rng = np.random.default_rng(1)
    # Intensity far below threshold -> ~guess rate -> mostly incorrect responses.
    responses = {
        obs.respond(
            {"intensity": -5.0, "correct_alternative": 0, "alternatives": [0, 1, 2, 3]}, rng
        )
        for _ in range(200)
    }
    assert responses <= {0, 1, 2, 3}
    assert len(responses) > 1  # should see more than one wrong alternative across many draws


def test_psychometric_observer_is_deterministic_given_seeded_rng() -> None:
    fn = _true_function()
    obs1 = PsychometricObserver(fn, n_afc=2)
    obs2 = PsychometricObserver(fn, n_afc=2)
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    seq1 = [obs1.respond({"intensity": -1.0, "correct_alternative": 0}, rng1) for _ in range(50)]
    seq2 = [obs2.respond({"intensity": -1.0, "correct_alternative": 0}, rng2) for _ in range(50)]
    assert seq1 == seq2


def test_csf_observer_rejects_n_afc_below_2() -> None:
    with pytest.raises(ValueError, match="n_afc"):
        CSFObserver(
            peak_gain_log10=1.5,
            peak_freq_cpd=3.0,
            bandwidth_octaves=3.0,
            low_freq_truncation_log10=1.0,
            n_afc=1,
        )


def test_csf_observer_p_correct_peaks_near_peak_frequency() -> None:
    obs = CSFObserver(
        peak_gain_log10=1.5,
        peak_freq_cpd=3.0,
        bandwidth_octaves=3.0,
        low_freq_truncation_log10=1.0,
        n_afc=2,
    )
    # At a fixed, moderate contrast, sensitivity (and thus p_correct) should
    # be higher near the peak frequency than far above it.
    contrast = 0.05
    p_at_peak = obs.p_correct(spatial_frequency_cpd=3.0, contrast=contrast)
    p_far_above = obs.p_correct(spatial_frequency_cpd=20.0, contrast=contrast)
    assert p_at_peak > p_far_above


def test_csf_observer_response_rate_matches_p_correct() -> None:
    obs = CSFObserver(
        peak_gain_log10=1.5,
        peak_freq_cpd=3.0,
        bandwidth_octaves=3.0,
        low_freq_truncation_log10=1.0,
        n_afc=2,
    )
    stim = {"spatial_frequency_cpd": 3.0, "contrast": 0.03, "correct_alternative": 0}
    expected_p = obs.p_correct(3.0, 0.03)
    rng = np.random.default_rng(2)
    n = 3000
    n_correct = sum(1 for _ in range(n) if obs.respond(stim, rng) == 0)
    observed_p = n_correct / n
    assert observed_p == pytest.approx(expected_p, abs=0.03)


def test_psychometric_observer_decide_correct_matches_respond() -> None:
    """decide_correct and respond must draw RNG identically (same is-correct decision)."""
    fn = _true_function()
    obs = PsychometricObserver(fn, n_afc=2)
    stim = {"intensity": -1.0, "correct_alternative": 0}
    for seed in range(10):
        rng_a = np.random.default_rng(seed)
        rng_b = np.random.default_rng(seed)
        is_correct = obs.decide_correct(stim, rng_a)
        response = obs.respond(stim, rng_b)
        assert is_correct == (response == 0)


def test_psychometric_observer_decide_correct_rate_matches_p_correct() -> None:
    fn = _true_function()
    obs = PsychometricObserver(fn, n_afc=2)
    rng = np.random.default_rng(11)
    x = -1.0
    expected_p = fn.p_correct(x)
    n = 4000
    n_correct = sum(1 for _ in range(n) if obs.decide_correct({"intensity": x}, rng))
    assert n_correct / n == pytest.approx(expected_p, abs=0.03)


def test_csf_observer_decide_correct_matches_respond() -> None:
    obs = CSFObserver(
        peak_gain_log10=1.5,
        peak_freq_cpd=3.0,
        bandwidth_octaves=3.0,
        low_freq_truncation_log10=1.0,
        n_afc=2,
    )
    stim = {"spatial_frequency_cpd": 3.0, "contrast": 0.03, "correct_alternative": 0}
    for seed in range(10):
        rng_a = np.random.default_rng(seed)
        rng_b = np.random.default_rng(seed)
        is_correct = obs.decide_correct(stim, rng_a)
        response = obs.respond(stim, rng_b)
        assert is_correct == (response == 0)


def test_csf_observer_is_deterministic_given_seeded_rng() -> None:
    obs1 = CSFObserver(
        peak_gain_log10=1.5,
        peak_freq_cpd=3.0,
        bandwidth_octaves=3.0,
        low_freq_truncation_log10=1.0,
        n_afc=2,
    )
    obs2 = CSFObserver(
        peak_gain_log10=1.5,
        peak_freq_cpd=3.0,
        bandwidth_octaves=3.0,
        low_freq_truncation_log10=1.0,
        n_afc=2,
    )
    stim = {"spatial_frequency_cpd": 3.0, "contrast": 0.03, "correct_alternative": 0}
    rng1 = np.random.default_rng(7)
    rng2 = np.random.default_rng(7)
    seq1 = [obs1.respond(stim, rng1) for _ in range(50)]
    seq2 = [obs2.respond(stim, rng2) for _ in range(50)]
    assert seq1 == seq2
