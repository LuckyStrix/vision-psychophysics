"""Procedural disc-field stimulus for the trivector color-discrimination task.

The Cambridge Colour Test (CCT; Regan, Reffin & Mollon 1994; Mollon & Reffin
1989) embeds a Landolt-C target in a field of discs of random position and
size, each independently assigned a random luminance from a fixed range. All
discs share the same *chromaticity* except those falling inside the C-shaped
target region, which carry the trial's target chromaticity instead -- the
per-disc luminance randomization ("luminance noise") is what forces the
gap-orientation judgment to depend on chromatic, not luminance or edge,
information (a purely-luminance-defined edge would give the game away).

This module generates that field procedurally: :func:`generate_disc_field`
places non-overlapping discs of random diameter (within a documented range)
by rejection sampling (random sequential adsorption) within a square field,
and :func:`is_in_target_region` classifies a disc by whether its *center*
falls within the Landolt-C ring (a documented convention -- discs are not
clipped/split at the ring boundary, they are simply the same disc as
everywhere else in the field, and only their assigned chromaticity differs
by which side of the boundary their center lands on).

Landolt-C geometry follows the standard optotype convention: stroke width
equals ``outer_diameter / 5``, so the ring spans from ``outer_diameter/2 -
outer_diameter/5`` (inner radius) to ``outer_diameter/2`` (outer radius).
The gap is specified as an arc length (degrees of visual angle) at the ring's
mean radius, converted to an angular half-width via the small-angle relation
``half_angle_rad = (gap_deg / 2) / mean_radius_deg`` (accurate to within a
fraction of a percent for the gap-to-radius ratios this test uses, e.g. ~0.6
at CCT's 4.3 deg / 1.0 deg defaults).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: 4AFC gap orientations and the direction (degrees, standard math convention:
#: 0 = right/+x, 90 = up/+y, counterclockwise) each corresponds to.
ORIENTATION_GAP_ANGLE_DEG: dict[str, float] = {
    "right": 0.0,
    "up": 90.0,
    "left": 180.0,
    "down": 270.0,
}

#: Response keys accepted for the 4AFC gap-orientation judgment, in the order
#: matching `ORIENTATION_GAP_ANGLE_DEG`'s natural reading.
RESPONSE_KEYS: list[str] = ["up", "down", "left", "right"]


@dataclass(frozen=True)
class Disc:
    """One disc placed in the stimulus field.

    Attributes:
        x_deg: Horizontal offset from the field center, in degrees of visual
            angle.
        y_deg: Vertical offset from the field center, in degrees of visual
            angle.
        diameter_deg: Disc diameter, in degrees of visual angle.
        is_target: Whether this disc's center falls inside the Landolt-C
            target region (see :func:`is_in_target_region`) and should
            therefore be rendered with the trial's target chromaticity
            rather than the background chromaticity.
    """

    x_deg: float
    y_deg: float
    diameter_deg: float
    is_target: bool


def is_in_target_region(
    x_deg: float,
    y_deg: float,
    outer_diameter_deg: float,
    stroke_width_deg: float,
    gap_deg: float,
    orientation: str,
) -> bool:
    """Whether a point falls inside the Landolt-C ring, excluding the gap sector.

    Args:
        x_deg: Horizontal offset from the ring's center, in degrees.
        y_deg: Vertical offset from the ring's center, in degrees.
        outer_diameter_deg: Outer diameter of the C, in degrees.
        stroke_width_deg: Ring stroke width, in degrees (conventionally
            ``outer_diameter_deg / 5``).
        gap_deg: Gap width, as an arc length at the ring's mean radius, in
            degrees.
        orientation: Which side the gap is on: ``"up"``, ``"down"``,
            ``"left"``, or ``"right"`` (see `ORIENTATION_GAP_ANGLE_DEG`).

    Returns:
        `True` if `(x_deg, y_deg)` is within the ring band (between the
        inner and outer radius) and outside the gap's angular sector.

    Raises:
        KeyError: If `orientation` is not one of the four recognized values.
    """
    gap_center_deg = ORIENTATION_GAP_ANGLE_DEG[orientation]  # validates orientation
    radius = math.hypot(x_deg, y_deg)
    outer_r = outer_diameter_deg / 2.0
    inner_r = outer_r - stroke_width_deg
    if not (inner_r <= radius <= outer_r):
        return False
    mean_r = (inner_r + outer_r) / 2.0
    gap_half_angle_deg = math.degrees((gap_deg / 2.0) / mean_r)
    angle_deg = math.degrees(math.atan2(y_deg, x_deg)) % 360.0
    diff = (angle_deg - gap_center_deg + 180.0) % 360.0 - 180.0
    return abs(diff) > gap_half_angle_deg


def generate_disc_field(
    rng: np.random.Generator,
    field_size_deg: float,
    min_diameter_deg: float,
    max_diameter_deg: float,
    outer_diameter_deg: float,
    stroke_width_deg: float,
    gap_deg: float,
    orientation: str,
    max_attempts: int = 6000,
) -> list[Disc]:
    """Generate a non-overlapping random disc field, via random sequential adsorption.

    Repeatedly samples a candidate disc (uniform random center within the
    field, uniform random diameter in `[min_diameter_deg, max_diameter_deg]`)
    and accepts it only if it does not overlap any previously accepted disc;
    stops after `max_attempts` proposals regardless of how many were
    accepted (later proposals are increasingly likely to be rejected as the
    field fills -- this is the standard random sequential adsorption /
    "jamming limit" behavior, and is expected, not a bug).

    Args:
        rng: Seeded random generator (the *only* source of randomness used,
            for reproducibility from a logged seed -- per this project's
            rng convention).
        field_size_deg: Side length of the square field, in degrees of
            visual angle, centered on the origin.
        min_diameter_deg: Minimum disc diameter, in degrees.
        max_diameter_deg: Maximum disc diameter, in degrees.
        outer_diameter_deg: Outer diameter of the embedded Landolt C, in
            degrees (see :func:`is_in_target_region`).
        stroke_width_deg: Landolt-C ring stroke width, in degrees.
        gap_deg: Landolt-C gap width, in degrees.
        orientation: Which side the gap is on (see
            :func:`is_in_target_region`).
        max_attempts: Maximum number of placement proposals.

    Returns:
        The accepted discs, in the order they were placed.

    Raises:
        ValueError: If `min_diameter_deg` is not positive, or exceeds
            `max_diameter_deg`.
    """
    if min_diameter_deg <= 0.0:
        raise ValueError(f"min_diameter_deg must be positive, got {min_diameter_deg}")
    if max_diameter_deg < min_diameter_deg:
        raise ValueError(
            f"max_diameter_deg ({max_diameter_deg}) must be >= min_diameter_deg "
            f"({min_diameter_deg})"
        )

    half_field = field_size_deg / 2.0
    xs: list[float] = []
    ys: list[float] = []
    radii: list[float] = []
    discs: list[Disc] = []

    for _ in range(max_attempts):
        diameter = float(rng.uniform(min_diameter_deg, max_diameter_deg))
        radius = diameter / 2.0
        if radius >= half_field:
            continue  # disc would not fit in the field at all; skip this proposal
        x = float(rng.uniform(-half_field + radius, half_field - radius))
        y = float(rng.uniform(-half_field + radius, half_field - radius))

        overlaps = False
        for xi, yi, ri in zip(xs, ys, radii, strict=True):
            min_sep = ri + radius
            if (xi - x) ** 2 + (yi - y) ** 2 < min_sep * min_sep:
                overlaps = True
                break
        if overlaps:
            continue

        xs.append(x)
        ys.append(y)
        radii.append(radius)
        is_target = is_in_target_region(
            x,
            y,
            outer_diameter_deg=outer_diameter_deg,
            stroke_width_deg=stroke_width_deg,
            gap_deg=gap_deg,
            orientation=orientation,
        )
        discs.append(Disc(x_deg=x, y_deg=y, diameter_deg=diameter, is_target=is_target))

    return discs


def field_coverage_fraction(discs: list[Disc], field_size_deg: float) -> float:
    """Fraction of the square field's area covered by disc area (discs may not overlap, so this
    is a simple sum, never exceeding 1)."""
    area = sum(math.pi * (d.diameter_deg / 2.0) ** 2 for d in discs)
    return float(area / (field_size_deg**2))
