"""Headless unit tests for the pure-numpy RDK dot-field physics.

Per `docs/WRITING_A_TEST.md` and the Phase 2D task: exact fraction of signal
dots, signal-dot displacement vector, noise-dot speed distribution,
lifetime replotting, aperture containment, and seed determinism.
"""

from __future__ import annotations

import numpy as np
import pytest

from vpsych.tests_catalog.motion_coherence.dots import (
    check_displacement_dmax,
    init_dot_field,
    n_dots_for_density,
    step_dot_field,
)


def test_n_dots_for_density_matches_area() -> None:
    # 10 deg aperture -> area = pi * 5^2 = 78.54 deg^2, at 2 dots/deg^2 -> ~157 dots.
    n = n_dots_for_density(aperture_diameter_deg=10.0, density_per_deg2=2.0)
    expected = round(np.pi * 25.0 * 2.0)
    assert n == expected


def test_init_dot_field_contains_all_dots_and_stages_ages() -> None:
    rng = np.random.default_rng(0)
    state = init_dot_field(n_dots=500, aperture_radius_deg=5.0, lifetime_frames=10, rng=rng)
    radii = np.sqrt(np.sum(state.positions_deg**2, axis=1))
    assert np.all(radii <= 5.0)
    assert state.ages_frames.min() >= 0
    assert state.ages_frames.max() < 10
    # Staggered, not all zero.
    assert len(np.unique(state.ages_frames)) > 1


def test_exact_fraction_of_signal_dots() -> None:
    rng = np.random.default_rng(1)
    n = 200
    state = init_dot_field(n_dots=n, aperture_radius_deg=10.0, lifetime_frames=1000, rng=rng)
    coherence = 0.3
    _new_state, is_signal = step_dot_field(
        state,
        rng,
        coherence=coherence,
        direction_deg=0.0,
        speed_deg_per_s=1.0,
        dt_s=1.0 / 60.0,
        aperture_radius_deg=10.0,
        lifetime_frames=1000,
    )
    assert int(np.sum(is_signal)) == round(coherence * n)


def test_signal_dot_displacement_vector_is_exact() -> None:
    rng = np.random.default_rng(2)
    n = 50
    # Large aperture and lifetime so nothing gets replotted this frame.
    state = init_dot_field(n_dots=n, aperture_radius_deg=1000.0, lifetime_frames=1_000_000, rng=rng)
    speed = 5.0
    dt_s = 1.0 / 60.0
    direction_deg = 37.0
    new_state, is_signal = step_dot_field(
        state,
        rng,
        coherence=1.0,  # force everyone to be a signal dot
        direction_deg=direction_deg,
        speed_deg_per_s=speed,
        dt_s=dt_s,
        aperture_radius_deg=1000.0,
        lifetime_frames=1_000_000,
    )
    assert np.all(is_signal)
    displacement = new_state.positions_deg - state.positions_deg
    expected_step = speed * dt_s
    theta = np.radians(direction_deg)
    expected_vec = np.array([np.cos(theta), np.sin(theta)]) * expected_step
    np.testing.assert_allclose(displacement, np.tile(expected_vec, (n, 1)), atol=1e-10)


def test_noise_dot_speed_is_constant_direction_is_random() -> None:
    rng = np.random.default_rng(3)
    n = 300
    state = init_dot_field(n_dots=n, aperture_radius_deg=1000.0, lifetime_frames=1_000_000, rng=rng)
    speed = 3.0
    dt_s = 1.0 / 85.0
    new_state, is_signal = step_dot_field(
        state,
        rng,
        coherence=0.0,  # force everyone to be a noise dot
        direction_deg=0.0,
        speed_deg_per_s=speed,
        dt_s=dt_s,
        aperture_radius_deg=1000.0,
        lifetime_frames=1_000_000,
    )
    assert not np.any(is_signal)
    displacement = new_state.positions_deg - state.positions_deg
    magnitudes = np.sqrt(np.sum(displacement**2, axis=1))
    expected_magnitude = speed * dt_s
    np.testing.assert_allclose(magnitudes, expected_magnitude, atol=1e-10)
    # Directions should not all be identical (random direction algorithm).
    angles = np.arctan2(displacement[:, 1], displacement[:, 0])
    assert len(np.unique(np.round(angles, 6))) > 1


def test_lifetime_replotting_resets_age_and_moves_dot() -> None:
    rng = np.random.default_rng(4)
    n = 20
    lifetime = 3
    state = init_dot_field(n_dots=n, aperture_radius_deg=1000.0, lifetime_frames=lifetime, rng=rng)
    # Force every dot to start at age lifetime - 1 so the next step ages them out.
    state.ages_frames[:] = lifetime - 1
    new_state, _is_signal = step_dot_field(
        state,
        rng,
        coherence=0.0,
        direction_deg=0.0,
        speed_deg_per_s=1.0,
        dt_s=1.0 / 60.0,
        aperture_radius_deg=1000.0,
        lifetime_frames=lifetime,
    )
    assert np.all(new_state.ages_frames == 0)


def test_aperture_containment_after_step() -> None:
    rng = np.random.default_rng(5)
    n = 200
    radius = 3.0
    state = init_dot_field(n_dots=n, aperture_radius_deg=radius, lifetime_frames=5, rng=rng)
    for _ in range(20):
        state, _is_signal = step_dot_field(
            state,
            rng,
            coherence=0.5,
            direction_deg=90.0,
            speed_deg_per_s=8.0,  # fast enough that many dots would exit without replotting
            dt_s=1.0 / 60.0,
            aperture_radius_deg=radius,
            lifetime_frames=5,
        )
        radii = np.sqrt(np.sum(state.positions_deg**2, axis=1))
        assert np.all(radii <= radius + 1e-9)


def test_seed_determinism() -> None:
    def run(seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        state = init_dot_field(n_dots=100, aperture_radius_deg=5.0, lifetime_frames=8, rng=rng)
        for _ in range(15):
            state, _is_signal = step_dot_field(
                state,
                rng,
                coherence=0.4,
                direction_deg=180.0,
                speed_deg_per_s=5.0,
                dt_s=1.0 / 60.0,
                aperture_radius_deg=5.0,
                lifetime_frames=8,
            )
        return state.positions_deg

    a = run(42)
    b = run(42)
    c = run(43)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_step_dot_field_rejects_invalid_coherence() -> None:
    rng = np.random.default_rng(0)
    state = init_dot_field(n_dots=5, aperture_radius_deg=5.0, lifetime_frames=5, rng=rng)
    with pytest.raises(ValueError, match="coherence"):
        step_dot_field(
            state,
            rng,
            coherence=1.5,
            direction_deg=0.0,
            speed_deg_per_s=1.0,
            dt_s=1.0 / 60.0,
            aperture_radius_deg=5.0,
            lifetime_frames=5,
        )


def test_check_displacement_dmax_flags_fast_dense_field() -> None:
    # Fast dots in a dense (tightly spaced) field -> displacement should exceed half spacing.
    exceeds, displacement_deg, spacing_deg = check_displacement_dmax(
        speed_deg_per_s=20.0, dt_s=1.0 / 60.0, density_per_deg2=50.0
    )
    assert exceeds is True
    assert displacement_deg > 0.5 * spacing_deg


def test_check_displacement_dmax_passes_for_slow_dense_field() -> None:
    exceeds, displacement_deg, spacing_deg = check_displacement_dmax(
        speed_deg_per_s=5.0, dt_s=1.0 / 60.0, density_per_deg2=2.0
    )
    assert exceeds is False
    assert displacement_deg <= 0.5 * spacing_deg
