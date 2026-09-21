"""Calibration display strings: grade labels, staleness, and what grade C rules out.

Pure functions over `vpsych.core.calibration.models.Calibration` (and the
test catalog), used by the Home screen's calibration badge and the
Calibration wizard's summary step. No Qt import.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from vpsych.core.calibration.models import Calibration, ColorGrade, LuminanceGrade
from vpsych.core.display import DisplayGeometry
from vpsych.tests_catalog.base import (
    PsychophysicalTest,
    TestSpec,
    check_requirements,
    visible_tests,
)

#: Default staleness threshold shown in the UI; mirrors `Calibration.is_stale`'s default.
DEFAULT_STALE_DAYS = 30

_LUMINANCE_GRADE_LABELS: dict[LuminanceGrade, str] = {
    "A": "A — photometer measured",
    "B": "B — psychophysical estimate",
    "C": "C — uncalibrated (sRGB assumed)",
}

_COLOR_GRADE_LABELS: dict[ColorGrade, str] = {
    "A": "A — measured primaries",
    "B": "B — n/a for color",
    "C": "C — sRGB assumed",
}


def luminance_grade_label(grade: LuminanceGrade) -> str:
    """Human-readable label for a luminance grade, e.g. `"A — photometer measured"`."""
    return _LUMINANCE_GRADE_LABELS[grade]


def color_grade_label(grade: ColorGrade) -> str:
    """Human-readable label for a color grade, e.g. `"C — sRGB assumed"`."""
    return _COLOR_GRADE_LABELS[grade]


def format_age_days(calibration: Calibration, now: datetime | None = None) -> str:
    """Format a calibration's age as a short human string, e.g. `"3 days ago"`.

    Args:
        calibration: The calibration to report the age of.
        now: Reference time, or `None` for the current UTC time.

    Returns:
        `"today"`, `"1 day ago"`, or `"<n> days ago"`.
    """
    reference = now if now is not None else datetime.now(timezone.utc)
    age_days = int((reference - calibration.created_utc).total_seconds() // 86400)
    if age_days <= 0:
        return "today"
    if age_days == 1:
        return "1 day ago"
    return f"{age_days} days ago"


@dataclass(frozen=True)
class CalibrationBadgeViewModel:
    """Everything the Home screen's calibration badge needs to render.

    Attributes:
        has_calibration: Whether any calibration is stored at all.
        luminance_grade_text: Label for the luminance grade, or `"Not available"`.
        color_grade_text: Label for the color grade, or `"Not available"`.
        age_text: Human-readable age string, or `"Not available"`.
        is_stale: Whether the calibration is older than `stale_after_days`.
        stale_warning: Warning text to show if `is_stale`, else `None`.
        summary_text: One-line overall summary for the badge.
    """

    has_calibration: bool
    luminance_grade_text: str
    color_grade_text: str
    age_text: str
    is_stale: bool
    stale_warning: str | None
    summary_text: str


def calibration_badge(
    calibration: Calibration | None,
    now: datetime | None = None,
    stale_after_days: int = DEFAULT_STALE_DAYS,
) -> CalibrationBadgeViewModel:
    """Build the Home screen's calibration status badge view model.

    Args:
        calibration: The active (most recent) calibration, or `None` if the
            participant/data root has never been calibrated.
        now: Reference time for staleness, or `None` for current UTC time.
        stale_after_days: Age threshold for the stale warning.

    Returns:
        The constructed `CalibrationBadgeViewModel`. Never invents values:
        every field is `"Not available"` or a plain-language placeholder
        when `calibration is None`.
    """
    if calibration is None:
        return CalibrationBadgeViewModel(
            has_calibration=False,
            luminance_grade_text="Not available",
            color_grade_text="Not available",
            age_text="Not available",
            is_stale=False,
            stale_warning=None,
            summary_text="No calibration on record. Run the calibration wizard before testing.",
        )
    is_stale = calibration.is_stale(max_age_days=stale_after_days, now=now)
    stale_warning = (
        f"This calibration is more than {stale_after_days} days old. Consider recalibrating "
        "before starting a new session."
        if is_stale
        else None
    )
    summary = (
        f"Luminance grade {calibration.luminance_grade}, color grade {calibration.color_grade} "
        f"— calibrated {format_age_days(calibration, now)}."
    )
    return CalibrationBadgeViewModel(
        has_calibration=True,
        luminance_grade_text=luminance_grade_label(calibration.luminance_grade),
        color_grade_text=color_grade_label(calibration.color_grade),
        age_text=format_age_days(calibration, now),
        is_stale=is_stale,
        stale_warning=stale_warning,
        summary_text=summary,
    )


@dataclass(frozen=True)
class GradeCLimitation:
    """One test that a grade-C (or otherwise insufficient) calibration currently rules out.

    Attributes:
        task_id: The blocked test's `TestSpec.id`.
        name: The blocked test's display name.
        reasons: Human-readable reasons it cannot run (verbatim from
            `check_requirements`).
    """

    task_id: str
    name: str
    reasons: list[str]


def grade_c_limitations(
    display: DisplayGeometry,
    calibration: Calibration | None,
    tests: list[type[PsychophysicalTest]] | None = None,
) -> list[GradeCLimitation]:
    """List every visible test that the given (grade-C or absent) calibration rules out.

    Used by the calibration wizard's summary step to plainly show "what
    grade C rules out" rather than leaving it abstract.

    Args:
        display: Display geometry to check tests against.
        calibration: The calibration under consideration (e.g. the one just
            produced by the wizard), or `None`.
        tests: Test classes to check, or `None` to use
            `vpsych.tests_catalog.base.visible_tests()`.

    Returns:
        One `GradeCLimitation` per test with at least one unmet requirement,
        in catalog order. Empty if every visible test can run.
    """
    candidates = tests if tests is not None else visible_tests()
    limitations: list[GradeCLimitation] = []
    for test_cls in candidates:
        spec: TestSpec = test_cls.spec
        reasons = check_requirements(spec.requirements, display, calibration)
        if reasons:
            limitations.append(GradeCLimitation(task_id=spec.id, name=spec.name, reasons=reasons))
    return limitations
