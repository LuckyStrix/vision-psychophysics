"""Display geometry: exact cm <-> px <-> deg conversions and refresh limits.

This module is pure math (no psychopy, no display server access) so it is
safe to import and exercise in headless CI. All quantities are named with
explicit units:

- ``_px``   pixels
- ``_cm``   centimetres
- ``_deg``  degrees of visual angle
- ``_hz``   hertz (cycles or frames per second)
- ``_cpd``  cycles per degree of visual angle
- ``_ms``   milliseconds

Angle conversions use the exact trigonometric relationship between a linear
extent on the screen and the visual angle it subtends at the eye, *not* the
small-angle approximation. For a linear extent ``x_cm`` centred on, and
perpendicular to, the line of sight at viewing distance ``d_cm``:

    theta_deg = 2 * degrees(atan((x_cm / 2) / d_cm))

and inversely:

    x_cm = 2 * d_cm * tan(radians(theta_deg) / 2)

This is exact for a flat display viewed with the extent centred on the
observer's line of sight; it is a good approximation elsewhere on the screen
for typical lab viewing distances and stimulus sizes, and callers doing
extreme-eccentricity work should account for that themselves.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field, computed_field


class DisplayGeometry(BaseModel):
    """Physical and pixel geometry of a display, plus viewing distance.

    All of a session's spatial-frequency and size calculations derive from
    this model. It is normally constructed once during the calibration
    geometry step and stored (as part of :class:`vpsych.core.calibration.models.Calibration`)
    alongside the session so later reanalysis uses the exact same geometry.

    Attributes:
        width_px: Horizontal display resolution, in pixels.
        height_px: Vertical display resolution, in pixels.
        width_cm: Physical width of the active display area, in centimetres.
        height_cm: Physical height of the active display area, in centimetres.
        viewing_distance_cm: Eye-to-screen distance, in centimetres, measured
            along the line of sight to the screen centre.
        refresh_hz: Nominal (OS-reported or measured) refresh rate, in hertz.
    """

    model_config = ConfigDict(frozen=True)

    width_px: int = Field(gt=0, description="Horizontal resolution in pixels.")
    height_px: int = Field(gt=0, description="Vertical resolution in pixels.")
    width_cm: float = Field(gt=0, description="Physical screen width in centimetres.")
    height_cm: float = Field(gt=0, description="Physical screen height in centimetres.")
    viewing_distance_cm: float = Field(gt=0, description="Eye-to-screen distance in centimetres.")
    refresh_hz: float = Field(gt=0, description="Display refresh rate in hertz.")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def px_per_cm(self) -> float:
        """Horizontal pixel density, in pixels per centimetre.

        Uses the horizontal axis (``width_px`` / ``width_cm``) as the
        reference density. Callers needing the vertical density should
        compute ``height_px / height_cm`` directly; the two normally agree
        to within display manufacturing tolerance for square pixels.
        """
        return self.width_px / self.width_cm

    def deg_to_px(self, deg: float) -> float:
        """Convert a visual angle to a linear pixel extent at the configured viewing distance.

        Uses the exact relation ``x_cm = 2 * d_cm * tan(radians(deg) / 2)``,
        then scales by ``px_per_cm``. Valid for extents centred on, and
        perpendicular to, the line of sight.

        Args:
            deg: Visual angle, in degrees.

        Returns:
            The equivalent linear extent, in pixels (may be fractional).
        """
        x_cm = 2.0 * self.viewing_distance_cm * math.tan(math.radians(deg) / 2.0)
        return x_cm * self.px_per_cm

    def px_to_deg(self, px: float) -> float:
        """Convert a linear pixel extent to the visual angle it subtends.

        Inverse of :meth:`deg_to_px`, using the exact relation
        ``theta_deg = 2 * degrees(atan((x_cm / 2) / d_cm))``.

        Args:
            px: Linear extent, in pixels.

        Returns:
            The subtended visual angle, in degrees.
        """
        x_cm = px / self.px_per_cm
        return 2.0 * math.degrees(math.atan((x_cm / 2.0) / self.viewing_distance_cm))

    @computed_field  # type: ignore[prop-decorator]
    @property
    def px_per_deg_at_center(self) -> float:
        """Pixel density at the display centre, in pixels per degree of visual angle.

        Computed as ``deg_to_px(1.0)``: the number of pixels subtended by one
        degree of visual angle at the configured viewing distance. This is
        the reference value used for Nyquist and stimulus-sizing
        calculations; it is only exact at the point on the screen intersected
        by the line of sight (the display centre for a centred observer) and
        decreases somewhat toward the edges of a flat screen because the
        projection is not equi-angular.
        """
        return self.deg_to_px(1.0)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def nyquist_cpd(self) -> float:
        """Nyquist limit of the display, in cycles per degree of visual angle.

        The display cannot faithfully represent a grating with a spatial
        frequency at or above this value (two pixels per cycle is the
        theoretical limit); tests should warn or refuse when a requested
        spatial frequency approaches or exceeds it. Equal to
        ``px_per_deg_at_center / 2``.
        """
        return self.px_per_deg_at_center / 2.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def max_flicker_hz(self) -> float:
        """Highest unambiguous flicker frequency the display can present, in hertz.

        Equal to ``refresh_hz / 2`` (the temporal Nyquist limit): a flicker
        alternating every frame produces a signal at ``refresh_hz / 2``, and
        no higher true flicker rate can be rendered without aliasing.
        """
        return self.refresh_hz / 2.0

    def frames_for_ms(self, ms: float) -> int:
        """Convert a target duration to a whole number of refresh frames.

        Rounds to the nearest frame (ties round to even, per
        :func:`round`) and clamps to a minimum of 1 frame, since a
        zero-frame stimulus cannot be presented. Per the project rule that
        *all stimulus durations are specified in frames, never seconds*,
        this is the conversion point where a design duration in
        milliseconds is translated into the frame count actually used to
        drive presentation.

        Args:
            ms: Target duration, in milliseconds.

        Returns:
            The number of refresh frames closest to ``ms``, at least 1.
        """
        frame_ms = 1000.0 / self.refresh_hz
        n = round(ms / frame_ms)
        return max(1, int(n))

    def ms_for_frames(self, n: int) -> float:
        """Convert a whole number of refresh frames to a duration in milliseconds.

        Args:
            n: Number of refresh frames.

        Returns:
            The nominal duration of ``n`` frames, in milliseconds, at the
            configured ``refresh_hz``.
        """
        frame_ms = 1000.0 / self.refresh_hz
        return n * frame_ms
