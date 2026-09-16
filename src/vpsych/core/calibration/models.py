"""Calibration data models: geometry, gamma, color, environment, and grading.

A :class:`Calibration` is the immutable record of everything needed to trust
a session's data: display geometry, luminance/gamma characterization, color
primaries, an environment checklist, and letter grades summarizing how the
luminance and color characterizations were obtained. Calibrations are
content-addressed (see :meth:`Calibration.content_hash`) and stored as
``cal-<sha256>.json``; sessions reference a calibration by that hash so
reanalysis always knows exactly which calibration produced a given session's
data.

Grading rules (used by :class:`Calibration` and enforced by
:func:`vpsych.tests_catalog.base.check_requirements`):

- Luminance grade: ``photometer`` -> ``A``, ``psychophysical`` -> ``B``,
  ``none`` -> ``C``.
- Color grade: ``measured`` -> ``A``, ``srgb_assumed`` -> ``C``.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from vpsych.core.display import DisplayGeometry

LuminanceGrade = Literal["A", "B", "C"]
ColorGrade = Literal["A", "B", "C"]
GammaMethod = Literal["photometer", "psychophysical", "none"]
ColorMethod = Literal["measured", "srgb_assumed"]


def grade_luminance(method: GammaMethod) -> LuminanceGrade:
    """Map a gamma-calibration method to its luminance quality grade.

    Args:
        method: How the display's luminance/gamma was characterized:
            ``"photometer"`` (a photometer or colorimeter measured
            luminance at several gray levels and a gamma curve was fit),
            ``"psychophysical"`` (a dithered-checkerboard flicker/minimum-
            motion match against uniform gray, with no physical light
            measurement), or ``"none"`` (uncalibrated; sRGB gamma assumed).

    Returns:
        ``"A"`` for ``"photometer"``, ``"B"`` for ``"psychophysical"``,
        ``"C"`` for ``"none"``.
    """
    mapping: dict[GammaMethod, LuminanceGrade] = {
        "photometer": "A",
        "psychophysical": "B",
        "none": "C",
    }
    return mapping[method]


def grade_color(method: ColorMethod) -> ColorGrade:
    """Map a color-calibration method to its color quality grade.

    Args:
        method: How the display's color primaries were characterized:
            ``"measured"`` (primaries measured with a spectroradiometer or
            colorimeter) or ``"srgb_assumed"`` (nominal sRGB primaries
            assumed, uncalibrated).

    Returns:
        ``"A"`` for ``"measured"``, ``"C"`` for ``"srgb_assumed"``.
    """
    mapping: dict[ColorMethod, ColorGrade] = {"measured": "A", "srgb_assumed": "C"}
    return mapping[method]


class GammaCalibrationPoint(BaseModel):
    """A single measured (input level, luminance) sample used to fit a gamma curve.

    Attributes:
        input_level: Requested output level, normalized to [0, 1] (0 = black,
            1 = maximum white) for the channel being measured.
        luminance_cdm2: Measured luminance at that input level, in
            candelas per square metre (cd/m^2).
    """

    model_config = ConfigDict(frozen=True)

    input_level: float = Field(ge=0, le=1, description="Requested normalized output level [0, 1].")
    luminance_cdm2: float = Field(ge=0, description="Measured luminance in cd/m^2.")


class GammaCalibration(BaseModel):
    """Luminance/gamma characterization of the display.

    Attributes:
        method: How this characterization was obtained; see
            :func:`grade_luminance` for the grading rule it implies.
        gamma_r: Fitted gamma exponent for the red channel (None if not
            fit per-channel or method is ``"none"``).
        gamma_g: Fitted gamma exponent for the green channel.
        gamma_b: Fitted gamma exponent for the blue channel.
        gamma_single: A single fitted gamma exponent used for all channels,
            when per-channel gammas were not estimated separately.
        lum_min_cdm2: Measured (or assumed) luminance at input level 0
            (black), in cd/m^2.
        lum_max_cdm2: Measured (or assumed) luminance at input level 1
            (maximum white), in cd/m^2.
        measured_points: Optional raw (input level, luminance) samples used
            to fit the gamma curve, for provenance/re-fitting.
    """

    model_config = ConfigDict(frozen=True)

    method: GammaMethod = Field(description="How luminance/gamma was characterized.")
    gamma_r: float | None = Field(
        default=None, gt=0, description="Fitted red-channel gamma exponent."
    )
    gamma_g: float | None = Field(
        default=None, gt=0, description="Fitted green-channel gamma exponent."
    )
    gamma_b: float | None = Field(
        default=None, gt=0, description="Fitted blue-channel gamma exponent."
    )
    gamma_single: float | None = Field(
        default=None, gt=0, description="Single gamma exponent applied to all channels."
    )
    lum_min_cdm2: float = Field(ge=0, description="Luminance at input level 0 (black), in cd/m^2.")
    lum_max_cdm2: float = Field(gt=0, description="Luminance at input level 1 (white), in cd/m^2.")
    measured_points: list[GammaCalibrationPoint] | None = Field(
        default=None, description="Raw (input level, luminance) samples used to fit the curve."
    )

    @property
    def grade(self) -> LuminanceGrade:
        """Luminance quality grade implied by ``method`` (see :func:`grade_luminance`)."""
        return grade_luminance(self.method)


class PrimaryChromaticity(BaseModel):
    """CIE xyY chromaticity and luminance of one display primary or the white point.

    Attributes:
        x: CIE 1931 x chromaticity coordinate.
        y: CIE 1931 y chromaticity coordinate.
        Y_cdm2: Luminance (CIE Y, in absolute units of cd/m^2) of this
            primary at maximum output (or of the white point at nominal
            white).
    """

    model_config = ConfigDict(frozen=True)

    x: float = Field(ge=0, le=1, description="CIE 1931 x chromaticity coordinate.")
    y: float = Field(gt=0, le=1, description="CIE 1931 y chromaticity coordinate.")
    Y_cdm2: float = Field(ge=0, description="Luminance in cd/m^2.")


class ColorCalibration(BaseModel):
    """Color primary characterization of the display.

    Attributes:
        method: How the primaries were obtained; see :func:`grade_color`
            for the grading rule it implies.
        red: Red primary chromaticity/luminance.
        green: Green primary chromaticity/luminance.
        blue: Blue primary chromaticity/luminance.
        white: White point chromaticity/luminance.
    """

    model_config = ConfigDict(frozen=True)

    method: ColorMethod = Field(description="How color primaries were characterized.")
    red: PrimaryChromaticity = Field(description="Red primary, CIE xyY.")
    green: PrimaryChromaticity = Field(description="Green primary, CIE xyY.")
    blue: PrimaryChromaticity = Field(description="Blue primary, CIE xyY.")
    white: PrimaryChromaticity = Field(description="White point, CIE xyY.")

    @property
    def grade(self) -> ColorGrade:
        """Color quality grade implied by ``method`` (see :func:`grade_color`)."""
        return grade_color(self.method)


class EnvironmentChecklist(BaseModel):
    """Self-reported environment checks completed before a session.

    Attributes:
        room_lighting_controlled: Room lighting is dim/controlled and free
            of glare on the screen.
        monitor_warmed_up: The monitor has been powered on for at least 15
            minutes before calibration/testing.
        night_light_disabled: OS-level blue-light/night-light color
            adjustment is turned off.
        hdr_disabled: OS-level HDR/auto-brightness/adaptive-contrast display
            modes are turned off.
        notes: Free-text notes about the environment (e.g. ambient
            luminance reading), if any.
    """

    model_config = ConfigDict(frozen=True)

    room_lighting_controlled: bool = Field(
        description="Room lighting is dim/controlled and free of glare on the screen."
    )
    monitor_warmed_up: bool = Field(
        description="Monitor has been powered on for at least 15 minutes."
    )
    night_light_disabled: bool = Field(description="OS night-light/blue-light filter is off.")
    hdr_disabled: bool = Field(description="OS HDR/adaptive-brightness display modes are off.")
    notes: str | None = Field(default=None, description="Free-text environment notes.")


class Calibration(BaseModel):
    """The complete, immutable calibration record referenced by a session.

    Content-addressed: two calibrations with identical field values (other
    than transient fields excluded from hashing) hash to the same value, so
    accidental duplicate calibrations collapse to one file, and any edit
    produces a new hash rather than mutating history in place.

    Attributes:
        created_utc: UTC timestamp when this calibration was created.
        geometry: Display geometry (see
            :class:`vpsych.core.display.DisplayGeometry`).
        gamma: Luminance/gamma characterization.
        color: Color primary characterization.
        environment: Environment checklist completed at calibration time.
        software_version: ``vpsych`` version string that produced this
            calibration.
        notes: Free-text notes about this calibration.
    """

    model_config = ConfigDict(frozen=True)

    created_utc: datetime = Field(description="UTC timestamp this calibration was created.")
    geometry: DisplayGeometry = Field(description="Display geometry at calibration time.")
    gamma: GammaCalibration = Field(description="Luminance/gamma characterization.")
    color: ColorCalibration = Field(description="Color primary characterization.")
    environment: EnvironmentChecklist = Field(description="Environment checklist.")
    software_version: str = Field(
        description="vpsych version string that produced this calibration."
    )
    notes: str | None = Field(default=None, description="Free-text notes.")

    @field_validator("created_utc")
    @classmethod
    def _require_timezone_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("created_utc must be timezone-aware (UTC)")
        return v

    @property
    def luminance_grade(self) -> LuminanceGrade:
        """Luminance quality grade, derived from ``gamma.method``."""
        return self.gamma.grade

    @property
    def color_grade(self) -> ColorGrade:
        """Color quality grade, derived from ``color.method``."""
        return self.color.grade

    def _canonical_json(self) -> bytes:
        """Serialize this calibration to canonical (sorted-key, no-whitespace) JSON bytes."""
        data = self.model_dump(mode="json")
        return json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def content_hash(self) -> str:
        """Compute the content-addressed hash identifying this calibration.

        The hash is the SHA-256 digest (hex-encoded) of this calibration's
        canonical JSON serialization (all fields, including
        ``created_utc``; two calibrations differing only in creation
        timestamp are considered distinct calibrations, since they may
        reflect re-measurement at a different time). This is the ``<hash>``
        used in the ``cal-<hash>.json`` filename and referenced by sessions.

        Returns:
            A 64-character lowercase hex SHA-256 digest.
        """
        return hashlib.sha256(self._canonical_json()).hexdigest()

    def is_stale(self, max_age_days: int = 30, now: datetime | None = None) -> bool:
        """Check whether this calibration is older than ``max_age_days``.

        Args:
            max_age_days: Maximum age, in days, before a calibration is
                considered stale. Defaults to 30.
            now: Reference time to compare against; defaults to the current
                UTC time. Must be timezone-aware if provided.

        Returns:
            ``True`` if ``now - created_utc`` exceeds ``max_age_days``.
        """
        reference = now if now is not None else datetime.now(timezone.utc)
        age_days = (reference - self.created_utc).total_seconds() / 86400.0
        return age_days > max_age_days
