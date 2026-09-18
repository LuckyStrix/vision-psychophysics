"""Headless unit tests for the pure-numpy CFF frame-sampled waveform generator.

Per `docs/WRITING_A_TEST.md` and the Phase 2D task: the frame luminance
sequence generator (mean equals Lmean exactly per presentation within
tolerance; amplitude via DFT; square-wave mode frequencies), plus the
intensity transform round-trip.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vpsych.tests_catalog.critical_flicker_fusion.waveform import (
    dft_fundamental_amplitude,
    frame_luminance_sequence,
    frequency_to_intensity,
    intensity_to_frequency,
    raised_cosine_envelope,
    square_wave_frame_luminance_sequence,
    steady_luminance_sequence,
    usable_square_wave_frequencies,
    valid_square_wave_frequency,
)


def test_frame_luminance_mean_equals_lmean_for_integer_cycle_count() -> None:
    """With an exact integer number of cycles and no envelope, the mean is exact."""
    refresh_hz = 60.0
    frequency_hz = 6.0  # 10 frames/cycle
    n_frames = 60  # exactly 6 cycles
    seq = frame_luminance_sequence(
        mean_luminance_cdm2=50.0,
        modulation_depth=1.0,
        frequency_hz=frequency_hz,
        refresh_hz=refresh_hz,
        n_frames=n_frames,
        onset_ramp_frames=0,
    )
    assert seq.mean() == pytest.approx(50.0, abs=1e-9)


def test_frame_luminance_mean_close_to_lmean_within_tolerance_generally() -> None:
    """Even without an integer cycle count, the mean stays close to Lmean."""
    seq = frame_luminance_sequence(
        mean_luminance_cdm2=40.0,
        modulation_depth=0.8,
        frequency_hz=7.3,
        refresh_hz=60.0,
        n_frames=61,
        onset_ramp_frames=0,
    )
    # Bounded by (amplitude / n_frames) roughly -- generous tolerance here.
    assert seq.mean() == pytest.approx(40.0, abs=2.0)


def test_frame_luminance_respects_bounds() -> None:
    seq = frame_luminance_sequence(
        mean_luminance_cdm2=50.0,
        modulation_depth=1.0,
        frequency_hz=5.0,
        refresh_hz=60.0,
        n_frames=60,
    )
    assert seq.min() >= 0.0 - 1e-9
    assert seq.max() <= 100.0 + 1e-9


def test_steady_luminance_sequence_is_constant() -> None:
    seq = steady_luminance_sequence(42.0, 30)
    assert len(seq) == 30
    assert np.all(seq == 42.0)


def test_raised_cosine_envelope_edges_and_plateau() -> None:
    env = raised_cosine_envelope(n_frames=20, ramp_frames=5)
    assert env[0] < env[4] < env[5]
    assert env[5] == pytest.approx(1.0)
    assert env[-1] < env[-2]
    assert np.all(env >= 0.0) and np.all(env <= 1.0 + 1e-9)


def test_raised_cosine_envelope_zero_ramp_is_flat() -> None:
    env = raised_cosine_envelope(n_frames=10, ramp_frames=0)
    assert np.all(env == 1.0)


def test_dft_fundamental_amplitude_recovers_known_sinusoid() -> None:
    refresh_hz = 60.0
    frequency_hz = 5.0
    mean = 50.0
    modulation_depth = 0.6
    n_frames = 120  # 10 full cycles -> exact bin alignment
    seq = frame_luminance_sequence(
        mean_luminance_cdm2=mean,
        modulation_depth=modulation_depth,
        frequency_hz=frequency_hz,
        refresh_hz=refresh_hz,
        n_frames=n_frames,
        onset_ramp_frames=0,
    )
    amplitude = dft_fundamental_amplitude(seq, frequency_hz, refresh_hz)
    assert amplitude == pytest.approx(modulation_depth * mean, rel=1e-6)


def test_dft_fundamental_amplitude_catastrophically_attenuated_at_exact_nyquist() -> None:
    """At exactly refresh_hz/2 with zero starting phase, sampling aliases the signal to zero.

    `sin(2*pi*(R/2)*n/R + 0) = sin(pi*n) = 0` for every integer `n`: the
    frame-sampled sequence is perfectly flat (no visible flicker at all)
    despite 100% nominal modulation -- the sharpest possible illustration
    of "aliasing and sampling artifacts appear as f approaches R/2"
    (see the module docstring and docs/methods/critical_flicker_fusion.md).
    """
    refresh_hz = 60.0
    frequency_hz = refresh_hz / 2.0
    mean = 50.0
    modulation_depth = 1.0
    seq = frame_luminance_sequence(
        mean_luminance_cdm2=mean,
        modulation_depth=modulation_depth,
        frequency_hz=frequency_hz,
        refresh_hz=refresh_hz,
        n_frames=60,
        phase_rad=0.0,
        onset_ramp_frames=0,
    )
    np.testing.assert_allclose(seq, mean, atol=1e-9)
    amplitude = dft_fundamental_amplitude(seq, frequency_hz, refresh_hz)
    nominal = modulation_depth * mean
    assert amplitude < 0.01 * nominal


def test_dft_fundamental_amplitude_attenuated_by_onset_ramp_window() -> None:
    """A large onset/offset ramp relative to the trial removes real energy at the fundamental."""
    refresh_hz = 60.0
    frequency_hz = 5.0
    mean = 50.0
    modulation_depth = 1.0
    n_frames = 30
    seq_no_ramp = frame_luminance_sequence(
        mean_luminance_cdm2=mean,
        modulation_depth=modulation_depth,
        frequency_hz=frequency_hz,
        refresh_hz=refresh_hz,
        n_frames=n_frames,
        onset_ramp_frames=0,
    )
    seq_heavy_ramp = frame_luminance_sequence(
        mean_luminance_cdm2=mean,
        modulation_depth=modulation_depth,
        frequency_hz=frequency_hz,
        refresh_hz=refresh_hz,
        n_frames=n_frames,
        onset_ramp_frames=14,  # nearly the whole trial is ramp
    )
    amp_no_ramp = dft_fundamental_amplitude(seq_no_ramp, frequency_hz, refresh_hz)
    amp_heavy_ramp = dft_fundamental_amplitude(seq_heavy_ramp, frequency_hz, refresh_hz)
    assert amp_heavy_ramp < 0.9 * amp_no_ramp


def test_valid_square_wave_frequency() -> None:
    refresh_hz = 120.0
    assert valid_square_wave_frequency(60.0, refresh_hz)  # k=1
    assert valid_square_wave_frequency(30.0, refresh_hz)  # k=2
    assert valid_square_wave_frequency(20.0, refresh_hz)  # k=3
    assert not valid_square_wave_frequency(25.0, refresh_hz)
    assert not valid_square_wave_frequency(0.0, refresh_hz)


def test_usable_square_wave_frequencies_are_all_valid_and_sorted() -> None:
    refresh_hz = 120.0
    freqs = usable_square_wave_frequencies(refresh_hz, min_hz=2.0, max_hz=40.0)
    assert freqs == sorted(freqs)
    assert len(freqs) > 1
    for f in freqs:
        assert valid_square_wave_frequency(f, refresh_hz)
        assert 2.0 - 1e-6 <= f <= 40.0 + 1e-6


def test_square_wave_frame_luminance_sequence_exact_levels() -> None:
    refresh_hz = 120.0
    frequency_hz = 20.0  # k=3, half-period = 3 frames
    mean = 50.0
    modulation_depth = 1.0
    seq = square_wave_frame_luminance_sequence(mean, modulation_depth, frequency_hz, refresh_hz, 12)
    high = mean * (1 + modulation_depth)
    low = mean * (1 - modulation_depth)
    assert set(np.round(seq, 9)) <= {round(high, 9), round(low, 9)}
    # Half-period blocks of exactly 3 frames each.
    np.testing.assert_allclose(seq[:3], high)
    np.testing.assert_allclose(seq[3:6], low)


def test_square_wave_rejects_invalid_frequency() -> None:
    with pytest.raises(ValueError, match="square-wave"):
        square_wave_frame_luminance_sequence(50.0, 1.0, 25.0, 120.0, 12)


def test_intensity_transform_round_trip() -> None:
    for f in (2.0, 5.5, 20.0, 40.0, 59.9):
        x = frequency_to_intensity(f)
        f2 = intensity_to_frequency(x)
        assert f2 == pytest.approx(f, rel=1e-12)


def test_intensity_transform_is_monotonic_with_performance_direction() -> None:
    """Higher frequency (harder) must map to a lower intensity x (QUEST+ needs p increasing in x)."""
    x_low_freq = frequency_to_intensity(2.0)
    x_high_freq = frequency_to_intensity(40.0)
    assert x_low_freq > x_high_freq


def test_frequency_to_intensity_rejects_nonpositive() -> None:
    with pytest.raises(ValueError):
        frequency_to_intensity(0.0)
    with pytest.raises(ValueError):
        frequency_to_intensity(-1.0)


def test_intensity_to_frequency_matches_formula() -> None:
    x = -1.0
    assert intensity_to_frequency(x) == pytest.approx(10.0**1.0)
    assert intensity_to_frequency(x) == pytest.approx(math.pow(10.0, -x))
