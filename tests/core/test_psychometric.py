"""Scientific validation and unit tests for vpsych.core.psychometric.

Fit-recovery/coverage tests come in matched slow/fast pairs: the slow
variant (`@pytest.mark.slow`, excluded from the default run via
`-m "not slow"` in pyproject.toml) uses >=200 repetitions per the plan's
verification requirement (|bias| < 0.05 log10 units, ~95% CI coverage,
checked in the 88-99% band over >=200 reps); the fast variant uses far fewer
repetitions purely as an always-on smoke test that the machinery isn't
badly broken, with looser tolerances appropriate to its much higher sampling
noise.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from vpsych.core.psychometric import (
    FitResult,
    PsychometricFunction,
    bootstrap_ci,
    deviance_gof,
    fit_mle,
    intensity_at_p_correct,
)

TRUE_THRESHOLD = -1.0
TRUE_SLOPE = 0.3
GUESS = 0.5
LAPSE = 0.02
LEVELS = np.linspace(-1.8, -0.2, 7)
N_PER_LEVEL = 50


def _simulate(
    rng: np.random.Generator,
    family: str,
    threshold: float = TRUE_THRESHOLD,
    slope: float = TRUE_SLOPE,
    guess: float = GUESS,
    lapse: float = LAPSE,
    levels: np.ndarray = LEVELS,
    n_per_level: int = N_PER_LEVEL,
) -> tuple[list[float], list[int], list[int]]:
    true_fn = PsychometricFunction(
        family=family,  # type: ignore[arg-type]
        threshold=threshold,
        slope=slope,
        guess=guess,
        lapse=lapse,
        intensity_scale="log10",
    )
    n_correct = [int(rng.binomial(n_per_level, true_fn.p_correct(float(x)))) for x in levels]
    n_total = [n_per_level] * len(levels)
    return list(levels), n_correct, n_total


def _recovery_run(n_reps: int, n_boot: int, family: str, seed: int) -> tuple[float, float]:
    """Return (mean bias, CI coverage fraction) over `n_reps` simulated datasets."""
    rng = np.random.default_rng(seed)
    biases = []
    covered = 0
    for _ in range(n_reps):
        levels, n_correct, n_total = _simulate(rng, family)
        fit = fit_mle(levels, n_correct, n_total, family=family, guess=GUESS)  # type: ignore[arg-type]
        biases.append(fit.function.threshold - TRUE_THRESHOLD)
        ci = bootstrap_ci(fit, n_boot=n_boot, level=0.95, rng=rng, parameter="threshold")
        if ci.ci_low <= TRUE_THRESHOLD <= ci.ci_high:
            covered += 1
    return float(np.mean(biases)), covered / n_reps


def _bias_only(n_reps: int, family: str, seed: int) -> float:
    """Mean threshold bias over `n_reps` simulated datasets (no bootstrap -- much cheaper)."""
    rng = np.random.default_rng(seed)
    biases = []
    for _ in range(n_reps):
        levels, n_correct, n_total = _simulate(rng, family)
        fit = fit_mle(levels, n_correct, n_total, family=family, guess=GUESS)  # type: ignore[arg-type]
        biases.append(fit.function.threshold - TRUE_THRESHOLD)
    return float(np.mean(biases))


@pytest.mark.parametrize("family", ["weibull", "logistic", "norm_cdf"])
def test_fit_mle_recovery_bias_fast(family: str) -> None:
    """Cheap smoke test: fit_mle only (no bootstrap), few reps. See the slow variant below
    for the real bias/coverage check (>=200 reps, |bias|<0.05, 88-99% coverage)."""
    bias = _bias_only(n_reps=15, family=family, seed=1)
    assert abs(bias) < 0.15


def test_bootstrap_ci_covers_true_threshold_smoke() -> None:
    """Cheap smoke test that bootstrap_ci runs end-to-end and is roughly centered correctly."""
    rng = np.random.default_rng(20)
    n_reps = 10
    covered = 0
    for _ in range(n_reps):
        levels, n_correct, n_total = _simulate(rng, "weibull")
        fit = fit_mle(levels, n_correct, n_total, family="weibull", guess=GUESS)
        ci = bootstrap_ci(fit, n_boot=40, level=0.95, rng=rng, parameter="threshold")
        assert ci.ci_low <= ci.point_estimate <= ci.ci_high
        if ci.ci_low <= TRUE_THRESHOLD <= ci.ci_high:
            covered += 1
    assert covered / n_reps >= 0.4  # loose smoke bound; see the slow variant for the real check


@pytest.mark.slow
@pytest.mark.parametrize("family", ["weibull", "logistic", "norm_cdf"])
def test_fit_mle_recovery_slow(family: str) -> None:
    bias, coverage = _recovery_run(n_reps=200, n_boot=200, family=family, seed=2)
    assert abs(bias) < 0.05
    assert 0.88 <= coverage <= 0.99


def test_fit_mle_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        fit_mle([0.0, 1.0], [1], [1, 1], family="weibull", guess=0.5)


def test_fit_mle_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        fit_mle([], [], [], family="weibull", guess=0.5)


def test_fit_mle_rejects_n_correct_exceeding_n_total() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        fit_mle([0.0, 1.0], [5, 1], [3, 3], family="weibull", guess=0.5)


def test_fit_mle_fixed_lapse_is_reported() -> None:
    rng = np.random.default_rng(3)
    levels, n_correct, n_total = _simulate(rng, "weibull")
    fit = fit_mle(levels, n_correct, n_total, family="weibull", guess=0.5, fix_lapse=0.02)
    assert fit.fixed_lapse is True
    assert fit.function.lapse == pytest.approx(0.02)


def test_fit_mle_free_lapse_is_bounded() -> None:
    rng = np.random.default_rng(4)
    levels, n_correct, n_total = _simulate(rng, "weibull")
    fit = fit_mle(levels, n_correct, n_total, family="weibull", guess=0.5, fix_lapse=None)
    assert fit.fixed_lapse is False
    assert 0.0 <= fit.function.lapse <= 0.06


def test_fit_mle_is_deterministic() -> None:
    rng = np.random.default_rng(5)
    levels, n_correct, n_total = _simulate(rng, "weibull")
    fit1 = fit_mle(levels, n_correct, n_total, family="weibull", guess=0.5)
    fit2 = fit_mle(levels, n_correct, n_total, family="weibull", guess=0.5)
    assert fit1.function.threshold == fit2.function.threshold
    assert fit1.function.slope == fit2.function.slope


def test_bootstrap_ci_is_deterministic_given_seeded_rng() -> None:
    rng = np.random.default_rng(6)
    levels, n_correct, n_total = _simulate(rng, "weibull")
    fit = fit_mle(levels, n_correct, n_total, family="weibull", guess=0.5)
    ci1 = bootstrap_ci(fit, n_boot=100, rng=np.random.default_rng(42))
    ci2 = bootstrap_ci(fit, n_boot=100, rng=np.random.default_rng(42))
    assert ci1.ci_low == ci2.ci_low
    assert ci1.ci_high == ci2.ci_high


def test_bootstrap_ci_rejects_unknown_parameter() -> None:
    rng = np.random.default_rng(7)
    levels, n_correct, n_total = _simulate(rng, "weibull")
    fit = fit_mle(levels, n_correct, n_total, family="weibull", guess=0.5)
    with pytest.raises(ValueError, match="Unknown parameter"):
        bootstrap_ci(fit, parameter="bogus")


def test_bootstrap_ci_rejects_fit_without_design_data() -> None:
    fn = PsychometricFunction(
        family="weibull", threshold=-1.0, slope=0.3, guess=0.5, lapse=0.02, intensity_scale="log10"
    )
    fit = FitResult(
        function=fn, log_likelihood=-10.0, n_trials=100, converged=True, fixed_lapse=False
    )
    with pytest.raises(ValueError, match="no stored design"):
        bootstrap_ci(fit)


def test_deviance_gof_p_value_in_unit_interval() -> None:
    rng = np.random.default_rng(8)
    levels, n_correct, n_total = _simulate(rng, "weibull")
    fit = fit_mle(levels, n_correct, n_total, family="weibull", guess=0.5)
    gof = deviance_gof(fit, levels, n_correct, n_total, n_mc=100, rng=np.random.default_rng(9))
    assert 0.0 <= gof.p_value <= 1.0
    assert gof.deviance >= 0.0
    assert gof.df == len(levels) - 3  # threshold, slope, lapse all free


def test_deviance_gof_flags_a_badly_misspecified_fit() -> None:
    """A fit forced far from the data should show a small (or at least not large) p-value."""
    rng = np.random.default_rng(10)
    levels, n_correct, n_total = _simulate(rng, "weibull", n_per_level=200)
    bad_fn = PsychometricFunction(
        family="weibull", threshold=2.0, slope=0.05, guess=0.5, lapse=0.0, intensity_scale="log10"
    )
    bad_fit = FitResult(
        function=bad_fn,
        log_likelihood=-1.0,
        n_trials=sum(n_total),
        converged=True,
        fixed_lapse=True,
        design_intensities=tuple(levels),
        design_n_total=tuple(n_total),
    )
    gof = deviance_gof(bad_fit, levels, n_correct, n_total, n_mc=100, rng=np.random.default_rng(11))
    assert gof.p_value < 0.1


def test_intensity_at_p_correct_matches_threshold_at_f50() -> None:
    fn = PsychometricFunction(
        family="weibull", threshold=-1.0, slope=0.3, guess=0.0, lapse=0.0, intensity_scale="log10"
    )
    # With guess=lapse=0, p=0.5 is exactly F(0)=0.5, i.e. x == threshold.
    x = intensity_at_p_correct(fn, 0.5)
    assert x == pytest.approx(fn.threshold, abs=1e-9)


def test_intensity_at_p_correct_is_monotonic_in_p() -> None:
    fn = PsychometricFunction(
        family="logistic", threshold=-1.0, slope=0.3, guess=0.5, lapse=0.02, intensity_scale="log10"
    )
    x_low = intensity_at_p_correct(fn, 0.6)
    x_high = intensity_at_p_correct(fn, 0.9)
    assert x_high > x_low


def test_intensity_at_p_correct_round_trips_through_p_correct() -> None:
    fn = PsychometricFunction(
        family="norm_cdf",
        threshold=-0.5,
        slope=0.4,
        guess=0.25,
        lapse=0.03,
        intensity_scale="log10",
    )
    x = intensity_at_p_correct(fn, 0.7)
    assert fn.p_correct(x) == pytest.approx(0.7, abs=1e-6)


@pytest.mark.parametrize("family", ["weibull", "logistic", "norm_cdf"])
def test_base_sigmoid_is_half_at_threshold(family: str) -> None:
    fn = PsychometricFunction(
        family=family,  # type: ignore[arg-type]
        threshold=-1.0,
        slope=0.3,
        guess=0.0,
        lapse=0.0,
        intensity_scale="log10",
    )
    assert fn.p_correct(fn.threshold) == pytest.approx(0.5, abs=1e-9)


def test_fit_result_and_derived_models_are_json_serializable() -> None:
    rng = np.random.default_rng(12)
    levels, n_correct, n_total = _simulate(rng, "weibull")
    fit = fit_mle(levels, n_correct, n_total, family="weibull", guess=0.5)
    ci = bootstrap_ci(fit, n_boot=50, rng=rng)
    gof = deviance_gof(fit, levels, n_correct, n_total, n_mc=50, rng=rng)

    json.loads(fit.model_dump_json())
    json.loads(ci.model_dump_json())
    json.loads(gof.model_dump_json())
