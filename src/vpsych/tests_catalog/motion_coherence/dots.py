"""Pure-numpy random-dot kinematogram (RDK) dot-field update.

This module is the entire physics of the motion-coherence stimulus,
deliberately factored out of the PsychoPy-driving test class so it is
headless-testable and so exactly what was displayed on a given frame can be
logged/reproduced from a seed alone (per the project's headless-testability
and reproducibility rules -- see `CONTRIBUTING.md`). `MotionCoherenceTest`
uses `step_dot_field` to compute dot positions every frame and feeds those
positions straight into a `psychopy.visual.ElementArrayStim`; it never
relies on PsychoPy's own `DotStim` motion logic, which is not unit-testable
independent of a live window.

Design decisions (see `docs/methods/motion_coherence.md` for the full
citations and discussion):

- **Signal selection ("white noise")**: on every frame, an *exact* count of
  ``round(coherence * n_dots)`` dots is freshly, independently chosen (via
  `rng.permutation`) to be "signal" dots for that frame only -- not a fixed
  subset held for the whole trial. This is the "white noise"/independent
  signal-selection scheme discussed by Pilly & Seitz (2009, *Vision
  Research*, 49(13), 1599-1612): reselecting afresh each frame prevents an
  observer from tracking a persistent subset of dots as individuated
  objects across the whole trial, which a fixed-signal-dot scheme would
  allow and which would inflate coherence sensitivity for reasons unrelated
  to the global motion percept the task intends to measure.
- **Noise algorithm**: "random direction" (Scase, Braddick, & Raymond,
  1996, *Vision Research*, 36(18), 2579-2586, who compared this against
  "random position" and "random walk" schemes): every frame, each
  non-signal dot is assigned a fresh, independent, uniformly random
  direction and moves at the *same speed* as a signal dot. This keeps a
  well-defined, testable per-frame displacement magnitude for every dot
  (signal or noise) -- unlike "random position" (full re-plotting, which
  produces discontinuous jumps with no fixed speed) -- while still
  guaranteeing that no single noise dot carries a consistent net direction
  a local motion detector could integrate across frames (each frame's
  direction is independent of the last). Scase et al. found "random
  position" gives the purest measure of global-motion sensitivity (least
  contaminated by local, dot-level motion energy), which is a documented
  trade-off of this choice -- "random direction" is simpler to implement
  and log exactly, and is the scheme used in most later human/monkey RDK
  work descending from Newsome & Paré (1988, *Journal of Neuroscience*,
  8(6), 2201-2211) and Britten, Shadlen, Newsome, & Movshon (1992, *Journal
  of Neuroscience*, 12(12), 4745-4765).
- **Limited dot lifetime**: every dot is replotted at a uniformly random
  position after `lifetime_frames` frames (regardless of whether it also
  left the aperture), reducing the ability to track any single dot for the
  whole trial (Scase et al. 1996). Initial ages are staggered uniformly
  across `[0, lifetime_frames)` (`init_dot_field`) so replotting events are
  spread across frames rather than all dots refreshing in synchrony (which
  would itself be a salient, coherence-independent transient).
- **Edge handling**: a dot whose step would carry it outside the circular
  aperture is replotted at a fresh uniformly random position *inside* the
  aperture (not wrapped to the opposite edge). Wrapping along the motion
  axis is a common alternative, but on a *circular* aperture "the opposite
  side along the direction of motion" is not simply the mirrored edge (the
  chord length through the aperture varies with the dot's off-axis
  position), so a naive wrap either breaks containment or introduces a
  position-dependent bias in wrap timing that could act as a coherence-
  independent, direction-correlated density cue near the aperture edge
  (Scase et al. 1996 raise exactly this class of concern about
  edge-handling artifacts in RDK design; Pilly & Seitz 2009's review makes
  the same point). Random replotting sacrifices a small amount of
  local-density continuity but introduces no directional cue.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

NOISE_ALGORITHM = "random_direction"
"""Name of the noise-dot algorithm implemented here, per Scase, Braddick, &
Raymond (1996) -- logged verbatim into `stimulus_params` by the test class."""

SIGNAL_SELECTION = "white_noise"
"""Name of the signal-dot selection scheme implemented here (afresh every
frame, not a fixed subset) -- logged verbatim into `stimulus_params`."""

EDGE_HANDLING = "random_replot_in_aperture"
"""Name of the aperture-edge-handling scheme implemented here (replot at a
random in-aperture position, not wrap) -- logged verbatim into
`stimulus_params`."""

DIRECTION_ANGLES_DEG: dict[str, float] = {"right": 0.0, "left": 180.0}
"""2AFC signal-direction labels mapped to a standard math angle (degrees,
counter-clockwise from the positive x/rightward screen axis)."""


@dataclass(frozen=True)
class DotFieldState:
    """A random-dot field's positions and per-dot ages at one point in time.

    Attributes:
        positions_deg: `(n_dots, 2)` array of `(x, y)` dot positions, in
            degrees of visual angle relative to the aperture center.
        ages_frames: `(n_dots,)` array of the number of frames each dot has
            been alive since its last replot (0 immediately after a
            replot).
    """

    positions_deg: FloatArray
    ages_frames: IntArray


def n_dots_for_density(aperture_diameter_deg: float, density_per_deg2: float) -> int:
    """Compute the (rounded) dot count for a circular aperture at a given density.

    Args:
        aperture_diameter_deg: Aperture diameter, in degrees of visual
            angle.
        density_per_deg2: Dot density, in dots per square degree.

    Returns:
        The number of dots, rounded to the nearest integer, at least 1.
    """
    radius = aperture_diameter_deg / 2.0
    area_deg2 = np.pi * radius * radius
    return max(1, round(area_deg2 * density_per_deg2))


def _random_positions_in_aperture(
    n: int, aperture_radius_deg: float, rng: np.random.Generator
) -> FloatArray:
    """Draw `n` positions uniformly distributed over a filled disc of the given radius."""
    if n == 0:
        return np.zeros((0, 2), dtype=np.float64)
    # Uniform-in-area sampling of a disc: radius ~ R * sqrt(U), angle ~ U(0, 2pi).
    r = aperture_radius_deg * np.sqrt(rng.random(n))
    theta = rng.random(n) * 2.0 * np.pi
    return np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)


def init_dot_field(
    n_dots: int,
    aperture_radius_deg: float,
    lifetime_frames: int,
    rng: np.random.Generator,
) -> DotFieldState:
    """Construct an initial dot field with staggered ages.

    Args:
        n_dots: Number of dots.
        aperture_radius_deg: Aperture radius, in degrees.
        lifetime_frames: Dot lifetime, in frames (must be >= 1); initial
            ages are drawn uniformly from `[0, lifetime_frames)` so dots do
            not all reach end-of-life (and replot) simultaneously.
        rng: Seeded random generator (the only source of randomness).

    Returns:
        The initial `DotFieldState`.

    Raises:
        ValueError: If `n_dots < 0`, `aperture_radius_deg <= 0`, or
            `lifetime_frames < 1`.
    """
    if n_dots < 0:
        raise ValueError(f"n_dots must be >= 0, got {n_dots}")
    if aperture_radius_deg <= 0:
        raise ValueError(f"aperture_radius_deg must be positive, got {aperture_radius_deg}")
    if lifetime_frames < 1:
        raise ValueError(f"lifetime_frames must be >= 1, got {lifetime_frames}")

    positions = _random_positions_in_aperture(n_dots, aperture_radius_deg, rng)
    ages = rng.integers(0, lifetime_frames, size=n_dots) if n_dots else np.zeros(0, dtype=np.int64)
    return DotFieldState(positions_deg=positions, ages_frames=ages.astype(np.int64))


def step_dot_field(
    state: DotFieldState,
    rng: np.random.Generator,
    *,
    coherence: float,
    direction_deg: float,
    speed_deg_per_s: float,
    dt_s: float,
    aperture_radius_deg: float,
    lifetime_frames: int,
) -> tuple[DotFieldState, npt.NDArray[np.bool_]]:
    """Advance a dot field by exactly one frame.

    Pure function: given the same `state`, `rng` state, and parameters, this
    always produces the same result (seed determinism). All randomness
    (signal-dot selection, noise-dot directions, replot positions) is drawn
    from `rng`, in a fixed order, so the sequence of dot fields for a whole
    trial is fully reproducible from the trial's seed.

    Update order, per frame:

    1. **Signal selection** (white noise, see module docstring): choose
       exactly `round(coherence * n_dots)` dots, freshly, via
       `rng.permutation`, to be this frame's signal dots.
    2. **Displacement**: signal dots move by
       `speed_deg_per_s * dt_s` in `direction_deg`; noise dots move by the
       same distance in an independent, freshly drawn uniform random
       direction each ("random direction" algorithm).
    3. **Ages** increment by 1 frame.
    4. **Replot**: any dot that (a) has reached `lifetime_frames` or (b) has
       left the circular aperture after the displacement step is replotted
       at a fresh uniformly random in-aperture position, with age reset to
       0.

    Args:
        state: The dot field's state before this frame.
        rng: Seeded random generator (the only source of randomness).
        coherence: Fraction of dots that are signal dots this frame, in
            `[0, 1]`.
        direction_deg: Signal motion direction, in degrees (standard math
            angle, counter-clockwise from the positive x-axis) -- see
            `DIRECTION_ANGLES_DEG`.
        speed_deg_per_s: Dot speed, in degrees of visual angle per second
            (same for signal and noise dots).
        dt_s: Frame duration, in seconds (`1 / display.refresh_hz`,
            measured -- see `check_displacement_dmax`).
        aperture_radius_deg: Aperture radius, in degrees.
        lifetime_frames: Dot lifetime, in frames.

    Returns:
        `(new_state, is_signal)`: the updated `DotFieldState`, and a
        `(n_dots,)` boolean array marking which dots were this frame's
        signal dots (before any lifetime/edge replot -- a replotted dot's
        `is_signal` entry still reflects whether it was selected as signal
        for its pre-replot displacement this frame).

    Raises:
        ValueError: If `coherence` is outside `[0, 1]`.
    """
    if not (0.0 <= coherence <= 1.0):
        raise ValueError(f"coherence must be in [0, 1], got {coherence}")

    n = state.positions_deg.shape[0]
    is_signal = np.zeros(n, dtype=bool)
    n_signal = round(coherence * n)
    if n_signal > 0 and n > 0:
        signal_idx = rng.permutation(n)[:n_signal]
        is_signal[signal_idx] = True

    displacement = speed_deg_per_s * dt_s
    signal_theta = np.radians(direction_deg)
    signal_step = np.array([np.cos(signal_theta), np.sin(signal_theta)]) * displacement

    noise_theta = rng.random(n) * 2.0 * np.pi
    noise_steps = np.stack([np.cos(noise_theta), np.sin(noise_theta)], axis=1) * displacement

    steps = np.where(is_signal[:, None], signal_step[None, :], noise_steps)
    new_positions = state.positions_deg + steps
    new_ages = state.ages_frames + 1

    radii = np.sqrt(np.sum(new_positions**2, axis=1))
    outside = radii > aperture_radius_deg
    end_of_life = new_ages >= lifetime_frames
    needs_replot = outside | end_of_life

    n_replot = int(np.sum(needs_replot))
    if n_replot > 0:
        replot_positions = _random_positions_in_aperture(n_replot, aperture_radius_deg, rng)
        new_positions[needs_replot] = replot_positions
        new_ages[needs_replot] = 0

    return DotFieldState(positions_deg=new_positions, ages_frames=new_ages), is_signal


def check_displacement_dmax(
    speed_deg_per_s: float,
    dt_s: float,
    density_per_deg2: float,
) -> tuple[bool, float, float]:
    """Check per-frame dot displacement against Braddick's `d_max` correspondence limit.

    Braddick, O. (1974). A short-range process in apparent motion. *Vision
    Research*, 14(7), 519-527, describes a short-range motion-correspondence
    process that fails once successive positions of a moving element are
    displaced by more than roughly half the mean spacing between elements --
    beyond that, the visual system can no longer reliably solve the frame-
    to-frame correspondence problem for individual dots (spatial aliasing:
    a dot's post-displacement position is roughly as close to a
    *different* dot's pre-displacement position as to its own). Exceeding
    this limit corrupts the very motion signal the coherence manipulation
    is supposed to control.

    Mean dot spacing is approximated as `1 / sqrt(density_per_deg2)` (the
    side length of a square cell containing one dot at the given area
    density).

    Args:
        speed_deg_per_s: Dot speed, in degrees per second.
        dt_s: Frame duration, in seconds (from the *measured* refresh rate).
        density_per_deg2: Dot density, in dots per square degree.

    Returns:
        `(exceeds_dmax, displacement_deg, spacing_deg)`: whether the
        per-frame displacement exceeds half the mean dot spacing, the
        displacement itself (degrees), and the mean spacing (degrees).

    Raises:
        ValueError: If `density_per_deg2` is not positive.
    """
    if density_per_deg2 <= 0:
        raise ValueError(f"density_per_deg2 must be positive, got {density_per_deg2}")
    displacement_deg = speed_deg_per_s * dt_s
    spacing_deg = 1.0 / np.sqrt(density_per_deg2)
    exceeds = displacement_deg > 0.5 * spacing_deg
    return bool(exceeds), float(displacement_deg), float(spacing_deg)
