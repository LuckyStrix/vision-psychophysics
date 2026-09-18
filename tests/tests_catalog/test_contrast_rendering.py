"""Headless numpy tests for `vpsych.tests_catalog._contrast_rendering`.

Shared by `contrast_sensitivity_function` and `letter_contrast_sensitivity`
(see that module's docstring); these tests cover the rendering math (gamma
linearization, Michelson/Weber contrast mapping, dithering accuracy, the
grating/envelope/letter rasterization) independent of either test's plugin
wiring, and satisfy the "rendering math (headless numpy)" and dithering
accuracy requirements from `docs/WRITING_A_TEST.md`.
"""

from __future__ import annotations

import numpy as np
import pytest

from vpsych.core.calibration.gamma import GammaChannelModel
from vpsych.core.calibration.models import GammaCalibration
from vpsych.tests_catalog._contrast_rendering import (
    SLOAN_LETTERS,
    dither_frame,
    gamma_channel_model_from_calibration,
    mean_luminance_cdm2,
    michelson_drive_from_pattern,
    raised_cosine_disc_envelope,
    raised_cosine_temporal_envelope,
    render_letter_ink_mask,
    sinusoidal_grating_pattern,
    weber_drive_from_ink_mask,
)


def _model() -> GammaChannelModel:
    return GammaChannelModel(lum_min_cdm2=0.5, lum_max_cdm2=120.5, gamma=2.2)


# ---------------------------------------------------------------------------
# gamma_channel_model_from_calibration
# ---------------------------------------------------------------------------


def test_gamma_channel_model_from_calibration_uses_gamma_single() -> None:
    cal = GammaCalibration(
        method="photometer", gamma_single=2.3, lum_min_cdm2=0.4, lum_max_cdm2=110.0
    )
    model = gamma_channel_model_from_calibration(cal)
    assert model.gamma == pytest.approx(2.3)
    assert model.lum_min_cdm2 == pytest.approx(0.4)
    assert model.lum_max_cdm2 == pytest.approx(110.0)


def test_gamma_channel_model_from_calibration_averages_per_channel() -> None:
    cal = GammaCalibration(
        method="photometer",
        gamma_r=2.0,
        gamma_g=2.2,
        gamma_b=2.4,
        lum_min_cdm2=0.3,
        lum_max_cdm2=100.0,
    )
    model = gamma_channel_model_from_calibration(cal)
    assert model.gamma == pytest.approx(2.2)


def test_gamma_channel_model_from_calibration_rejects_none() -> None:
    cal = GammaCalibration(method="none", lum_min_cdm2=0.0, lum_max_cdm2=100.0)
    with pytest.raises(ValueError, match="uncalibrated"):
        gamma_channel_model_from_calibration(cal)


# ---------------------------------------------------------------------------
# Michelson / Weber contrast mapping + dithering accuracy
# ---------------------------------------------------------------------------


def test_mean_luminance_is_range_midpoint() -> None:
    model = _model()
    assert mean_luminance_cdm2(model) == pytest.approx((0.5 + 120.5) / 2.0)


def test_michelson_drive_zero_contrast_is_background() -> None:
    model = _model()
    pattern = np.array([[-1.0, 0.0, 1.0]])
    drive = michelson_drive_from_pattern(pattern, contrast=0.0, model=model)
    # Zero contrast: every pixel should linearize to the same background drive
    # regardless of the (irrelevant, since contrast=0) pattern value.
    assert np.allclose(drive, drive[0, 0])


def test_michelson_drive_full_contrast_spans_black_to_white() -> None:
    model = GammaChannelModel(lum_min_cdm2=0.0, lum_max_cdm2=100.0, gamma=2.2)
    pattern = np.array([-1.0, 1.0])
    drive = michelson_drive_from_pattern(pattern, contrast=1.0, model=model)
    assert drive[0] == pytest.approx(0.0, abs=1e-9)
    assert drive[1] == pytest.approx(1.0, abs=1e-9)


def test_weber_drive_background_pixels_unaffected_by_contrast() -> None:
    model = _model()
    mask = np.array([[0.0, 1.0]])
    for contrast in (0.1, 0.5, 0.9):
        drive = weber_drive_from_ink_mask(mask, contrast, model)
        background_fraction = (mean_luminance_cdm2(model) - model.lum_min_cdm2) / (
            model.lum_max_cdm2 - model.lum_min_cdm2
        )
        expected_bg_drive = background_fraction ** (1.0 / model.gamma)
        assert drive[0, 0] == pytest.approx(expected_bg_drive, rel=1e-6)


def test_weber_drive_darkens_ink_as_contrast_increases() -> None:
    model = _model()
    mask = np.array([[1.0]])
    low = weber_drive_from_ink_mask(mask, 0.1, model)[0, 0]
    high = weber_drive_from_ink_mask(mask, 0.9, model)[0, 0]
    assert high < low  # higher Weber contrast -> darker letter -> lower drive


def test_dither_to_uint8_mean_reproduces_low_contrast_target() -> None:
    """Headless dithering-accuracy check (docs/WRITING_A_TEST.md): the mean
    rendered luminance over many dither samples reproduces a 0.2% contrast
    target within tolerance, even though 0.2% is far below the raw 1/255
    (~0.39%) 8-bit step."""
    model = GammaChannelModel(lum_min_cdm2=0.0, lum_max_cdm2=100.0, gamma=2.2)
    contrast = 0.002  # 0.2%
    pattern = np.array([1.0])  # a single bright-side pixel of a grating
    target_drive = michelson_drive_from_pattern(pattern, contrast, model)[0]

    rng = np.random.default_rng(12345)
    n_samples = 20000
    samples = np.array([dither_frame(target_drive, rng)[()] for _ in range(n_samples)])
    mean_drive = float(samples.mean())

    # Effective resolution after averaging n_samples dithered draws (Allard &
    # Faubert 2008); the empirical mean should land well within a handful of
    # effective steps of the exact target.
    from vpsych.core.calibration.dither import effective_contrast_resolution

    tolerance = 5.0 * effective_contrast_resolution(n_samples)
    assert abs(mean_drive - target_drive) < tolerance


def test_dither_frame_is_reproducible_from_seed() -> None:
    rng1 = np.random.default_rng(7)
    rng2 = np.random.default_rng(7)
    arr = np.linspace(0.0, 1.0, 50)
    assert np.array_equal(dither_frame(arr, rng1), dither_frame(arr, rng2))


# ---------------------------------------------------------------------------
# Grating pattern / envelopes
# ---------------------------------------------------------------------------


def test_sinusoidal_grating_pattern_bounded_and_correct_frequency() -> None:
    size_px = 128
    cycles_per_px = 4.0 / size_px  # 4 cycles across the patch
    pattern = sinusoidal_grating_pattern(size_px, cycles_per_px, orientation_deg=0.0, phase_rad=0.0)
    assert pattern.shape == (size_px, size_px)
    assert pattern.max() <= 1.0 + 1e-9
    assert pattern.min() >= -1.0 - 1e-9
    # Count zero-crossings along one row: ~2 per cycle for a sinusoid.
    row = pattern[size_px // 2, :]
    crossings = np.sum(np.diff(np.sign(row)) != 0)
    assert 6 <= crossings <= 10  # ~8 for 4 cycles


def test_raised_cosine_disc_envelope_center_one_edge_zero() -> None:
    env = raised_cosine_disc_envelope(101, plateau_frac=0.5)
    center = env[50, 50]
    corner = env[0, 0]
    assert center == pytest.approx(1.0)
    assert corner == pytest.approx(0.0, abs=1e-6)
    assert np.all(env >= 0.0) and np.all(env <= 1.0)


def test_raised_cosine_temporal_envelope_ramps_and_plateau() -> None:
    env = raised_cosine_temporal_envelope(n_frames=30, ramp_frames=6)
    assert env[0] == pytest.approx(0.0, abs=1e-9)
    assert env[2] < env[5]  # monotonically increasing through the ramp
    assert env[5] == pytest.approx(1.0)  # ramp reaches the plateau value exactly
    assert np.all(env[6:24] == pytest.approx(1.0))
    assert env[-1] == pytest.approx(0.0, abs=1e-9)
    assert np.all((env >= 0.0) & (env <= 1.0))


def test_raised_cosine_temporal_envelope_clamps_long_ramp() -> None:
    # ramp_frames larger than half of n_frames should not raise or overlap oddly.
    env = raised_cosine_temporal_envelope(n_frames=10, ramp_frames=100)
    assert env.shape == (10,)
    assert np.all((env >= 0.0) & (env <= 1.0))


# ---------------------------------------------------------------------------
# Sloan letters
# ---------------------------------------------------------------------------


def test_all_sloan_letters_render_without_error_and_are_nonempty() -> None:
    for letter in SLOAN_LETTERS:
        mask = render_letter_ink_mask(letter, size_px=50)
        assert mask.shape == (50, 50)
        assert set(np.unique(mask)).issubset({0.0, 1.0})
        assert mask.sum() > 0  # every letter has some ink


def test_render_letter_ink_mask_unknown_letter_raises() -> None:
    with pytest.raises(KeyError):
        render_letter_ink_mask("A", size_px=25)


def test_render_letter_ink_mask_handles_non_multiple_of_five_size() -> None:
    mask = render_letter_ink_mask("O", size_px=37)
    assert mask.shape == (37, 37)
