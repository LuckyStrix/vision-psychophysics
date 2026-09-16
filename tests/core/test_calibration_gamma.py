"""Unit tests for vpsych.core.calibration.gamma: fitting, linearization, ramps, psychophysical estimation."""

from __future__ import annotations

import numpy as np
import pytest

from vpsych.core.calibration.gamma import (
    GammaChannelModel,
    estimate_gamma_psychophysical,
    fit_gamma,
    fit_gamma_lookup,
    linearize,
    make_gamma_ramp,
)
from vpsych.core.calibration.models import GammaCalibrationPoint


def _synthetic_points(
    gamma: float, lum_min: float = 0.5, lum_max: float = 150.0, n: int = 9
) -> list[GammaCalibrationPoint]:
    levels = np.linspace(0.0, 1.0, n)
    points = []
    for v in levels:
        lum = lum_min + (lum_max - lum_min) * (v**gamma)
        points.append(GammaCalibrationPoint(input_level=float(v), luminance_cdm2=float(lum)))
    return points


@pytest.mark.parametrize("true_gamma", [1.0, 1.8, 2.2, 2.6])
def test_fit_gamma_recovers_exponent(true_gamma: float) -> None:
    points = _synthetic_points(true_gamma)
    fit = fit_gamma(points)
    assert fit.gamma == pytest.approx(true_gamma, rel=1e-6)
    assert fit.lum_min_cdm2 == pytest.approx(0.5)
    assert fit.lum_max_cdm2 == pytest.approx(150.0)
    assert fit.r_squared == pytest.approx(1.0, abs=1e-9)


def test_fit_gamma_uses_min_max_when_endpoints_missing() -> None:
    # No point at input_level exactly 0 or 1: Lmin/Lmax fall back to the min/max
    # of measured luminance (here, the points at 0.05 and 0.95), and the
    # remaining strictly-interior points still fit gamma normally.
    points = [
        GammaCalibrationPoint(input_level=0.05, luminance_cdm2=1.0),
        GammaCalibrationPoint(input_level=0.25, luminance_cdm2=10.0),
        GammaCalibrationPoint(input_level=0.5, luminance_cdm2=40.0),
        GammaCalibrationPoint(input_level=0.75, luminance_cdm2=90.0),
        GammaCalibrationPoint(input_level=0.95, luminance_cdm2=120.0),
    ]
    fit = fit_gamma(points)
    assert fit.lum_min_cdm2 == 1.0
    assert fit.lum_max_cdm2 == 120.0


def test_fit_gamma_requires_at_least_two_points() -> None:
    with pytest.raises(ValueError):
        fit_gamma([GammaCalibrationPoint(input_level=0.5, luminance_cdm2=10.0)])


def test_fit_gamma_requires_interior_point() -> None:
    points = [
        GammaCalibrationPoint(input_level=0.0, luminance_cdm2=0.5),
        GammaCalibrationPoint(input_level=1.0, luminance_cdm2=150.0),
    ]
    with pytest.raises(ValueError, match="interior point"):
        fit_gamma(points)


def test_linearize_parametric_round_trip() -> None:
    model = GammaChannelModel(lum_min_cdm2=0.5, lum_max_cdm2=150.0, gamma=2.2)
    for frac in (0.0, 0.1, 0.5, 0.9, 1.0):
        drive = linearize(frac, model)
        recovered_lum = model.luminance(drive)
        recovered_frac = (recovered_lum - 0.5) / (150.0 - 0.5)
        assert recovered_frac == pytest.approx(frac, abs=1e-9)


def test_linearize_array_input() -> None:
    model = GammaChannelModel(lum_min_cdm2=0.0, lum_max_cdm2=100.0, gamma=2.0)
    fracs = np.array([0.0, 0.25, 1.0])
    drive = linearize(fracs, model)
    assert isinstance(drive, np.ndarray)
    np.testing.assert_allclose(drive, np.sqrt(fracs))


def test_linearize_rejects_out_of_range() -> None:
    model = GammaChannelModel(lum_min_cdm2=0.0, lum_max_cdm2=100.0, gamma=2.0)
    with pytest.raises(ValueError):
        linearize(1.5, model)
    with pytest.raises(ValueError):
        linearize(-0.1, model)


def test_gamma_channel_model_requires_exactly_one_representation() -> None:
    with pytest.raises(ValueError):
        GammaChannelModel(lum_min_cdm2=0.0, lum_max_cdm2=100.0)  # neither gamma nor lookup
    lookup = fit_gamma_lookup(_synthetic_points(2.2))
    with pytest.raises(ValueError):
        GammaChannelModel(
            lum_min_cdm2=0.0,
            lum_max_cdm2=100.0,
            gamma=2.2,
            lookup=lookup.lookup,
            inverse_lookup=lookup.inverse_lookup,
        )


def test_fit_gamma_lookup_monotone_and_round_trips() -> None:
    points = _synthetic_points(2.2, n=11)
    model = fit_gamma_lookup(points)
    assert model.gamma is None
    levels = np.linspace(0.0, 1.0, 50)
    lums = model.luminance(levels)
    assert np.all(np.diff(lums) > 0)  # monotone increasing

    # linearize should approximately invert luminance() at the measured points.
    for p in points:
        frac = (p.luminance_cdm2 - model.lum_min_cdm2) / (model.lum_max_cdm2 - model.lum_min_cdm2)
        drive = linearize(frac, model)
        assert drive == pytest.approx(p.input_level, abs=1e-6)


def test_fit_gamma_lookup_rejects_non_monotone_luminance() -> None:
    points = [
        GammaCalibrationPoint(input_level=0.0, luminance_cdm2=1.0),
        GammaCalibrationPoint(input_level=0.5, luminance_cdm2=50.0),
        GammaCalibrationPoint(input_level=1.0, luminance_cdm2=30.0),  # decreases!
    ]
    with pytest.raises(ValueError, match="increasing"):
        fit_gamma_lookup(points)


def test_fit_gamma_lookup_requires_two_distinct_levels() -> None:
    with pytest.raises(ValueError):
        fit_gamma_lookup([GammaCalibrationPoint(input_level=0.5, luminance_cdm2=10.0)])


def test_make_gamma_ramp_shape_and_range() -> None:
    r = GammaChannelModel(lum_min_cdm2=0.0, lum_max_cdm2=100.0, gamma=2.2)
    g = GammaChannelModel(lum_min_cdm2=0.0, lum_max_cdm2=110.0, gamma=2.0)
    b = GammaChannelModel(lum_min_cdm2=0.0, lum_max_cdm2=90.0, gamma=2.4)
    ramp = make_gamma_ramp((r, g, b), size=256)
    assert ramp.shape == (256, 3)
    assert np.all(ramp >= 0.0) and np.all(ramp <= 1.0)
    # First row (level 0) should map to drive 0; last row (level 1) to drive 1.
    np.testing.assert_allclose(ramp[0], [0.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(ramp[-1], [1.0, 1.0, 1.0], atol=1e-9)


def test_make_gamma_ramp_rejects_too_small_size() -> None:
    model = GammaChannelModel(lum_min_cdm2=0.0, lum_max_cdm2=100.0, gamma=2.2)
    with pytest.raises(ValueError):
        make_gamma_ramp((model, model, model), size=1)


def _synthetic_bisection_matches(
    true_gamma: float, fractions: list[float]
) -> list[tuple[float, float]]:
    return [(p, p ** (1.0 / true_gamma)) for p in fractions]


@pytest.mark.parametrize("true_gamma", [1.0, 1.8, 2.2])
def test_estimate_gamma_psychophysical_recovers_exponent(true_gamma: float) -> None:
    fractions = [0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9]
    matches = _synthetic_bisection_matches(true_gamma, fractions)
    result = estimate_gamma_psychophysical(matches)
    assert result.gamma == pytest.approx(true_gamma, rel=1e-6)
    assert result.ci_low <= result.gamma <= result.ci_high
    assert result.n_levels == len(fractions)


def test_estimate_gamma_psychophysical_ci_widens_with_noise() -> None:
    rng = np.random.default_rng(0)
    fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    true_gamma = 2.2
    clean = _synthetic_bisection_matches(true_gamma, fractions)
    noisy = [(p, min(0.999, max(0.001, v + rng.normal(0, 0.02)))) for p, v in clean]
    clean_result = estimate_gamma_psychophysical(clean)
    noisy_result = estimate_gamma_psychophysical(noisy)
    clean_width = clean_result.ci_high - clean_result.ci_low
    noisy_width = noisy_result.ci_high - noisy_result.ci_low
    assert noisy_width > clean_width


def test_estimate_gamma_psychophysical_requires_two_matches() -> None:
    with pytest.raises(ValueError):
        estimate_gamma_psychophysical([(0.5, 0.7)])


def test_estimate_gamma_psychophysical_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        estimate_gamma_psychophysical([(0.5, 0.7), (1.0, 0.5)])
    with pytest.raises(ValueError):
        estimate_gamma_psychophysical([(0.5, 0.7), (0.5, 0.0)])


def test_estimate_gamma_psychophysical_exact_two_points_zero_width_ci() -> None:
    # With exactly 2 points the fit has 0 residual df for variance estimation;
    # implementation should not raise and should return a degenerate (zero-width) CI.
    matches = _synthetic_bisection_matches(2.0, [0.3, 0.7])
    result = estimate_gamma_psychophysical(matches)
    assert result.ci_low == pytest.approx(result.gamma)
    assert result.ci_high == pytest.approx(result.gamma)


def test_gamma_channel_model_luminance_scalar_and_array() -> None:
    model = GammaChannelModel(lum_min_cdm2=1.0, lum_max_cdm2=101.0, gamma=1.0)
    assert model.luminance(0.5) == pytest.approx(51.0)
    arr = model.luminance(np.array([0.0, 1.0]))
    np.testing.assert_allclose(arr, [1.0, 101.0])


def test_fit_gamma_rejects_bad_normalized_luminance() -> None:
    # An interior point whose luminance exceeds lum_max (from endpoints) is invalid.
    points = [
        GammaCalibrationPoint(input_level=0.0, luminance_cdm2=0.0),
        GammaCalibrationPoint(input_level=0.5, luminance_cdm2=200.0),
        GammaCalibrationPoint(input_level=1.0, luminance_cdm2=100.0),
    ]
    with pytest.raises(ValueError):
        fit_gamma(points)
