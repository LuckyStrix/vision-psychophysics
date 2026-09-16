"""Bit-stealing (noisy-bit) dithering: render sub-LSB contrasts on an 8-bit display.

An ordinary 8-bit-per-channel frame buffer can only represent 256 distinct
drive levels per channel, i.e. contrast steps no finer than ~1/255. Many
threshold measurements (contrast sensitivity, color discrimination) need
finer steps than that. Allard & Faubert (2008), "The noisy-bit method for
digital displays: converting a resolution limitation into a
pseudo-resolution," *Behavior Research Methods*, 40(3), 735-743, describe a
stochastic-rounding ("noisy-bit") method: instead of always rounding a
desired intensity to the nearest representable 8-bit level, round to a
level chosen probabilistically so its *expected value* equals the desired
(sub-LSB-precision) intensity exactly. Averaged over many frames (temporal
dithering) or many neighbouring pixels (spatial dithering), the display
reproduces the desired intensity far more precisely than a single 8-bit
sample could -- this module implements that mapping and a helper to report
the resulting effective contrast resolution.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt


def dither_to_uint8(
    intensity: npt.NDArray[np.float64] | float,
    rng: np.random.Generator,
    bit_depth: int = 8,
) -> npt.NDArray[np.uint8]:
    """Convert a float intensity image to dithered per-channel unsigned integer levels.

    Implements Allard & Faubert's (2008) noisy-bit method: each element of
    ``intensity`` (normalized to ``[0, 1]``) is scaled to
    ``[0, 2**bit_depth - 1]`` and stochastically rounded -- a uniform random
    threshold in ``[0, 1)`` decides whether to round down or up -- so that
    ``E[output] == scaled`` exactly (unbiased quantization) rather than the
    systematic bias of ordinary (nearest or floor) rounding. A single
    dithered frame looks noisy at the pixel level, but the *mean* intensity
    over many independent draws (successive frames, or neighbouring pixels
    using independent noise) converges to the requested value far more
    precisely than the underlying bit depth alone would allow (see
    :func:`effective_contrast_resolution`).

    Args:
        intensity: Desired intensity, normalized to ``[0, 1]`` (0 = black,
            1 = maximum output), as a scalar or an array of any shape (e.g.
            an ``(H, W)`` or ``(H, W, 3)`` image).
        rng: Seeded random generator supplying the per-element dither noise
            (see :func:`vpsych.core.rng.make_rng`); must be the only source
            of randomness so dithered frames are reproducible from a logged
            seed.
        bit_depth: Output bit depth per channel; the returned levels span
            ``[0, 2**bit_depth - 1]``. Defaults to 8 (a standard 8-bit
            frame buffer channel).

    Returns:
        An array the same shape as ``intensity`` (or a 0-d array for a
        scalar input), dtype ``uint8``, with values in
        ``[0, 2**bit_depth - 1]``.

    Raises:
        ValueError: If ``bit_depth`` is not in ``[1, 8]`` (the return dtype
            is ``uint8``, which cannot represent more than 8 bits per
            element).
    """
    if not (1 <= bit_depth <= 8):
        raise ValueError(f"bit_depth must be in [1, 8] for a uint8 result, got {bit_depth}")

    values = np.asarray(intensity, dtype=np.float64)
    clipped = np.clip(values, 0.0, 1.0)
    max_level = float(2**bit_depth - 1)
    scaled = clipped * max_level

    noise = rng.uniform(0.0, 1.0, size=scaled.shape)
    dithered = np.floor(scaled + noise)
    dithered = np.clip(dithered, 0.0, max_level)
    return dithered.astype(np.uint8)


def effective_contrast_resolution(n_samples: int, bit_depth: int = 8) -> float:
    """Report the smallest intensity step reliably resolvable after averaging dithered samples.

    Stochastic rounding (:func:`dither_to_uint8`) converts quantization
    error into approximately uniformly-distributed noise with standard
    deviation ``1 / (2 * sqrt(3))`` of one quantization step (the standard
    deviation of a ``Uniform(-0.5, 0.5)`` step-sized error). Averaging
    ``n_samples`` *independent* dithered draws of the same nominal
    intensity -- across successive frames (temporal dithering) and/or
    independent neighbouring pixels (spatial dithering) -- reduces that
    noise's standard deviation by ``sqrt(n_samples)`` (central limit
    theorem), giving an effective pseudo-resolution finer than the
    underlying bit depth by that same factor.

    Args:
        n_samples: Number of independent dithered samples averaged
            together (e.g. frames shown within a stimulus presentation, or
            pixels in a spatial dither neighbourhood). Must be positive.
        bit_depth: The underlying per-channel bit depth being dithered
            (see :func:`dither_to_uint8`).

    Returns:
        The effective resolvable intensity step, as a fraction of the full
        ``[0, 1]`` intensity range (smaller is finer resolution). Equal to
        ``1 / ((2**bit_depth - 1) * sqrt(n_samples))``.

    Raises:
        ValueError: If ``n_samples`` is not positive.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    max_level = 2**bit_depth - 1
    return float(1.0 / (max_level * math.sqrt(n_samples)))
