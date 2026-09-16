"""Unit tests for vpsych.core.rng: seeded, logged RNG creation."""

from __future__ import annotations

import numpy as np

from vpsych.core.rng import make_rng


def test_make_rng_returns_generator_and_seed() -> None:
    rng, seed_used = make_rng(42)
    assert isinstance(rng, np.random.Generator)
    assert seed_used == 42


def test_make_rng_is_reproducible_with_explicit_seed() -> None:
    rng1, seed1 = make_rng(123)
    rng2, seed2 = make_rng(123)
    assert seed1 == seed2 == 123
    draws1 = rng1.standard_normal(10)
    draws2 = rng2.standard_normal(10)
    assert np.array_equal(draws1, draws2)


def test_make_rng_none_seed_draws_fresh_entropy() -> None:
    _rng1, seed1 = make_rng(None)
    _rng2, seed2 = make_rng(None)
    assert isinstance(seed1, int)
    assert isinstance(seed2, int)
    # Astronomically unlikely to collide; guards against a constant stub.
    assert seed1 != seed2


def test_make_rng_reported_seed_reconstructs_stream() -> None:
    rng1, seed_used = make_rng(None)
    draws1 = rng1.standard_normal(5)
    rng2, _ = make_rng(seed_used)
    draws2 = rng2.standard_normal(5)
    assert np.array_equal(draws1, draws2)
