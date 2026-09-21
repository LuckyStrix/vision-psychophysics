"""Calibration wizard state machine: geometry, gamma, color, environment -> `Calibration`.

Pure, non-Qt state for the four wizard steps described in the project plan.
The wizard screen (`vpsych.app.screens.calibration_wizard`) owns a single
`CalibrationWizardState`, mutates it as the participant/experimenter moves
through steps, and calls `build_calibration` at the end to assemble the
immutable `Calibration` record that gets saved via
`vpsych.data.dataset.save_calibration`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from vpsych.core.calibration.gamma import (
    GammaFitResult,
    GammaPsychophysicalEstimate,
    estimate_gamma_psychophysical,
    fit_gamma,
)
from vpsych.core.calibration.geometry import (
    build_display_geometry,
    estimate_display_width_cm,
)
from vpsych.core.calibration.models import (
    Calibration,
    ColorCalibration,
    EnvironmentChecklist,
    GammaCalibration,
    GammaCalibrationPoint,
    PrimaryChromaticity,
)
from vpsych.core.display import DisplayGeometry

#: Nominal sRGB/BT.709 primaries + D65 white, used for grade-C color.
_SRGB_PRIMARIES = {
    "red": PrimaryChromaticity(x=0.640, y=0.330, Y_cdm2=100.0),
    "green": PrimaryChromaticity(x=0.300, y=0.600, Y_cdm2=100.0),
    "blue": PrimaryChromaticity(x=0.150, y=0.060, Y_cdm2=100.0),
    "white": PrimaryChromaticity(x=0.3127, y=0.3290, Y_cdm2=100.0),
}

#: Nominal sRGB gamma (~2.2), used when no calibration is performed (grade C).
SRGB_NOMINAL_GAMMA = 2.2
#: Nominal luminance range assumed for an uncalibrated (grade C) display.
SRGB_NOMINAL_LUM_MIN_CDM2 = 0.3
SRGB_NOMINAL_LUM_MAX_CDM2 = 250.0


class WizardStepError(ValueError):
    """Raised when a wizard step's inputs are incomplete or invalid."""


@dataclass
class GeometryStepState:
    """Wizard state for the geometry step.

    Attributes:
        width_px: Horizontal resolution, in pixels (from OS query or manual entry).
        height_px: Vertical resolution, in pixels.
        refresh_hz: Refresh rate, in Hz.
        card_width_px: Pixel width of the on-screen rectangle matched to a
            real ID-1 card, or `None` if using a direct ruler measurement.
        width_cm: Physical screen width, in cm -- either entered directly,
            or filled in from `card_width_px` via `resolve_width_cm`.
        height_cm: Physical screen height, in cm (entered directly; the
            card-match method only estimates width).
        viewing_distance_cm: Eye-to-screen distance, in cm.
        resolution_source: `"os_detected"` or `"manual"`, for provenance
            display only.
    """

    width_px: int | None = None
    height_px: int | None = None
    refresh_hz: float | None = None
    card_width_px: float | None = None
    width_cm: float | None = None
    height_cm: float | None = None
    viewing_distance_cm: float | None = None
    resolution_source: str = "manual"

    def resolve_width_cm(self) -> float | None:
        """Compute `width_cm` from the credit-card match, if one was entered.

        Returns:
            The estimated width, or the already-entered `width_cm` if no
            card match was given, or `None` if neither is available.
        """
        if self.card_width_px is not None and self.width_px is not None:
            return estimate_display_width_cm(self.card_width_px, self.width_px)
        return self.width_cm

    def build(self) -> DisplayGeometry:
        """Build a `DisplayGeometry` from this step's state.

        Returns:
            The constructed geometry.

        Raises:
            WizardStepError: If any required field is missing.
        """
        width_cm = self.resolve_width_cm()
        missing = [
            name
            for name, value in (
                ("width_px", self.width_px),
                ("height_px", self.height_px),
                ("refresh_hz", self.refresh_hz),
                ("width_cm", width_cm),
                ("height_cm", self.height_cm),
                ("viewing_distance_cm", self.viewing_distance_cm),
            )
            if value is None
        ]
        if missing:
            raise WizardStepError(f"Geometry step is missing: {', '.join(missing)}.")
        assert self.width_px is not None
        assert self.height_px is not None
        assert self.refresh_hz is not None
        assert width_cm is not None
        assert self.height_cm is not None
        assert self.viewing_distance_cm is not None
        return build_display_geometry(
            width_px=self.width_px,
            height_px=self.height_px,
            width_cm=width_cm,
            height_cm=self.height_cm,
            viewing_distance_cm=self.viewing_distance_cm,
            refresh_hz=self.refresh_hz,
        )


GammaMode = str  # "photometer" | "psychophysical" | "none"


@dataclass
class GammaStepState:
    """Wizard state for the gamma (luminance) step.

    Attributes:
        mode: Which method was used: `"photometer"`, `"psychophysical"`, or
            `"none"` (skipped).
        photometer_points: Measured (drive level, luminance) points, if
            `mode == "photometer"`.
        psychophysical_matches: `(target_fraction, matched_drive_level)`
            bisection matches, if `mode == "psychophysical"`.
    """

    mode: GammaMode = "none"
    photometer_points: list[GammaCalibrationPoint] = field(default_factory=list)
    psychophysical_matches: list[tuple[float, float]] = field(default_factory=list)

    def build(self) -> GammaCalibration:
        """Build a `GammaCalibration` from this step's state.

        Returns:
            The constructed calibration (grade A/B/C per `mode`).

        Raises:
            WizardStepError: If `mode` needs data that hasn't been entered,
                or the fit itself fails (e.g. too few points).
        """
        if self.mode == "photometer":
            if len(self.photometer_points) < 2:
                raise WizardStepError(
                    "Photometer gamma measurement needs at least 2 measured points."
                )
            try:
                fit: GammaFitResult = fit_gamma(self.photometer_points)
            except ValueError as exc:
                raise WizardStepError(str(exc)) from exc
            return GammaCalibration(
                method="photometer",
                gamma_single=fit.gamma,
                lum_min_cdm2=fit.lum_min_cdm2,
                lum_max_cdm2=fit.lum_max_cdm2,
                measured_points=list(self.photometer_points),
            )
        if self.mode == "psychophysical":
            if len(self.psychophysical_matches) < 2:
                raise WizardStepError(
                    "Psychophysical gamma estimate needs at least 2 bisection matches."
                )
            try:
                estimate: GammaPsychophysicalEstimate = estimate_gamma_psychophysical(
                    self.psychophysical_matches
                )
            except ValueError as exc:
                raise WizardStepError(str(exc)) from exc
            return GammaCalibration(
                method="psychophysical",
                gamma_single=estimate.gamma,
                lum_min_cdm2=SRGB_NOMINAL_LUM_MIN_CDM2,
                lum_max_cdm2=SRGB_NOMINAL_LUM_MAX_CDM2,
            )
        return GammaCalibration(
            method="none",
            gamma_single=SRGB_NOMINAL_GAMMA,
            lum_min_cdm2=SRGB_NOMINAL_LUM_MIN_CDM2,
            lum_max_cdm2=SRGB_NOMINAL_LUM_MAX_CDM2,
        )


@dataclass
class ColorStepState:
    """Wizard state for the color step.

    Attributes:
        measured: Whether a measured (grade A) color calibration was
            performed; if `False`, nominal sRGB primaries are assumed
            (grade C).
        red, green, blue, white: Measured primaries/white point, required
            if `measured` is `True`.
    """

    measured: bool = False
    red: PrimaryChromaticity | None = None
    green: PrimaryChromaticity | None = None
    blue: PrimaryChromaticity | None = None
    white: PrimaryChromaticity | None = None

    def build(self) -> ColorCalibration:
        """Build a `ColorCalibration` from this step's state.

        Returns:
            The constructed calibration (grade A if `measured`, else
            grade C with nominal sRGB primaries).

        Raises:
            WizardStepError: If `measured` is `True` but any primary/white
                point is missing.
        """
        if not self.measured:
            return ColorCalibration(method="srgb_assumed", **_SRGB_PRIMARIES)
        missing = [
            name
            for name, value in (
                ("red", self.red),
                ("green", self.green),
                ("blue", self.blue),
                ("white", self.white),
            )
            if value is None
        ]
        if missing:
            raise WizardStepError(f"Measured color step is missing: {', '.join(missing)}.")
        assert self.red is not None
        assert self.green is not None
        assert self.blue is not None
        assert self.white is not None
        return ColorCalibration(
            method="measured", red=self.red, green=self.green, blue=self.blue, white=self.white
        )


@dataclass
class CalibrationWizardState:
    """Top-level wizard state: one instance per wizard run.

    Attributes:
        geometry: Geometry step state.
        gamma: Gamma step state.
        color: Color step state.
        environment: Environment checklist state, or `None` if not yet
            completed.
        notes: Free-text notes for the final `Calibration` (must stay
            pseudonymous -- caller's responsibility, not enforced here).
    """

    geometry: GeometryStepState = field(default_factory=GeometryStepState)
    gamma: GammaStepState = field(default_factory=GammaStepState)
    color: ColorStepState = field(default_factory=ColorStepState)
    environment: EnvironmentChecklist | None = None
    notes: str | None = None

    def build_calibration(self, software_version: str, now: datetime | None = None) -> Calibration:
        """Assemble the immutable `Calibration` from every completed step.

        Args:
            software_version: `vpsych` package version to record.
            now: Timestamp to use for `created_utc`, or `None` for the
                current UTC time.

        Returns:
            The constructed `Calibration`, ready to save via
            `vpsych.data.dataset.save_calibration`.

        Raises:
            WizardStepError: If the geometry step is incomplete, or the
                environment checklist was never completed.
        """
        if self.environment is None:
            raise WizardStepError("Environment checklist must be completed before saving.")
        return Calibration(
            created_utc=now if now is not None else datetime.now(timezone.utc),
            geometry=self.geometry.build(),
            gamma=self.gamma.build(),
            color=self.color.build(),
            environment=self.environment,
            software_version=software_version,
            notes=self.notes,
        )


def environment_checklist_complete(checklist: EnvironmentChecklist | None) -> bool:
    """Whether every boolean item on the environment checklist is checked.

    Args:
        checklist: The checklist to check, or `None`.

    Returns:
        `False` if `checklist` is `None` or any boolean item is `False`.
    """
    if checklist is None:
        return False
    return (
        checklist.room_lighting_controlled
        and checklist.monitor_warmed_up
        and checklist.night_light_disabled
        and checklist.hdr_disabled
    )


def environment_checklist_warnings(checklist: EnvironmentChecklist | None) -> list[str]:
    """Human-readable warnings for every unchecked environment-checklist item.

    Args:
        checklist: The checklist to check, or `None` (treated as nothing
            checked).

    Returns:
        One warning per unmet item.
    """
    items = {
        "room_lighting_controlled": "Room lighting is not confirmed dim/glare-free.",
        "monitor_warmed_up": "Monitor warm-up (15+ minutes) is not confirmed.",
        "night_light_disabled": "OS night-light/blue-light filter is not confirmed disabled.",
        "hdr_disabled": "OS HDR/adaptive-brightness is not confirmed disabled.",
    }
    if checklist is None:
        return list(items.values())
    warnings = []
    for attr, message in items.items():
        if not getattr(checklist, attr):
            warnings.append(message)
    return warnings
