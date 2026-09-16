"""Seeded, logged random number generation.

Every stochastic decision in a session (trial-order randomization, stimulus
jitter, catch-trial placement, simulated-observer responses in tests) must
flow from a :class:`numpy.random.Generator` created here, so the seed that
produced a session is always known and logged in
:class:`vpsych.data.schemas.SessionInfo` / :class:`vpsych.core.trial.TrialRecord`,
making runs exactly reproducible.
"""

from __future__ import annotations

import numpy as np


def make_rng(seed: int | None = None) -> tuple[np.random.Generator, int]:
    """Create a seeded NumPy random Generator and report the seed actually used.

    If ``seed`` is ``None``, a fresh, unpredictable seed is drawn from the
    OS entropy source (via :class:`numpy.random.SeedSequence`) so the caller
    can log it and reproduce the run later.

    Args:
        seed: Non-negative integer seed to use, or ``None`` to draw a fresh
            seed from OS entropy.

    Returns:
        A ``(generator, seed_used)`` tuple: a PCG64-backed
        :class:`numpy.random.Generator` ready to use, and the integer seed
        that was actually used to construct it (echo of ``seed`` when
        provided, otherwise the freshly drawn value). Log ``seed_used``
        wherever this RNG's output affects recorded data.
    """
    if seed is None:
        seed_sequence = np.random.SeedSequence()
        seed_used = int(seed_sequence.entropy)  # type: ignore[arg-type]
    else:
        seed_used = int(seed)
        seed_sequence = np.random.SeedSequence(seed_used)
    generator = np.random.Generator(np.random.PCG64(seed_sequence))
    return generator, seed_used
