"""Unit tests for vpsych.core.calibration.dither: noisy-bit dithering."""

from __future__ import annotations

import numpy as np
import pytest

from vpsych.core.calibration.dither import dither_to_uint8, effective_contrast_resolution


@pytest.mark.parametrize("intensity", [0.0, 0.1, 0.25, 0.5, 0.7331, 0.999, 1.0])
def test_dither_mean_reproduces_target_within_tolerance(intensity: float) -> None:
    rng = np.random.default_rng(0)
    n = 200_000
    samples = dither_to_uint8(np.full(n, intensity), rng)
    mean_fraction = samples.mean() / 255.0
    assert mean_fraction == pytest.approx(intensity, abs=1e-3)


def test_dither_output_dtype_and_range() -> None:
    rng = np.random.default_rng(1)
    samples = dither_to_uint8(np.linspace(0.0, 1.0, 1000), rng)
    assert samples.dtype == np.uint8
    assert samples.min() >= 0
    assert samples.max() <= 255


def test_dither_scalar_input() -> None:
    rng = np.random.default_rng(2)
    out = dither_to_uint8(0.5, rng)
    assert out.dtype == np.uint8
    assert 0 <= int(out) <= 255


def test_dither_is_stochastic_not_constant() -> None:
    rng = np.random.default_rng(3)
    samples = dither_to_uint8(np.full(5000, 0.5), rng)
    # A single 8-bit level (127 or 128) cannot represent 0.5 exactly; dithering
    # should produce more than one distinct output value.
    assert len(np.unique(samples)) > 1


def test_dither_clips_out_of_range_intensity() -> None:
    rng = np.random.default_rng(4)
    samples = dither_to_uint8(np.array([-0.5, 1.5]), rng)
    assert samples[0] == 0
    assert samples[1] == 255


def test_dither_reproducible_with_same_rng_state() -> None:
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    a = dither_to_uint8(np.full(100, 0.3), rng1)
    b = dither_to_uint8(np.full(100, 0.3), rng2)
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("bit_depth", [1, 4, 6, 8])
def test_dither_bit_depth_max_level(bit_depth: int) -> None:
    rng = np.random.default_rng(5)
    samples = dither_to_uint8(np.full(10000, 1.0), rng, bit_depth=bit_depth)
    assert samples.max() == 2**bit_depth - 1


def test_dither_invalid_bit_depth_raises() -> None:
    rng = np.random.default_rng(6)
    with pytest.raises(ValueError):
        dither_to_uint8(0.5, rng, bit_depth=0)
    with pytest.raises(ValueError):
        dither_to_uint8(0.5, rng, bit_depth=9)


def test_dither_preserves_shape() -> None:
    rng = np.random.default_rng(7)
    image = np.random.default_rng(8).uniform(0, 1, size=(4, 5, 3))
    out = dither_to_uint8(image, rng)
    assert out.shape == (4, 5, 3)


@pytest.mark.parametrize("n_samples", [1, 4, 16, 100])
def test_effective_contrast_resolution_scales_with_sqrt_n(n_samples: int) -> None:
    base = effective_contrast_resolution(1, bit_depth=8)
    res = effective_contrast_resolution(n_samples, bit_depth=8)
    assert res == pytest.approx(base / (n_samples**0.5))


def test_effective_contrast_resolution_finer_than_native_lsb_for_n_gt_1() -> None:
    native_lsb = 1.0 / 255.0
    assert effective_contrast_resolution(4, bit_depth=8) < native_lsb


def test_effective_contrast_resolution_invalid_n_raises() -> None:
    with pytest.raises(ValueError):
        effective_contrast_resolution(0)
    with pytest.raises(ValueError):
        effective_contrast_resolution(-1)
