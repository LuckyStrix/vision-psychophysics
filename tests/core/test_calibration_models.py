"""Unit tests for vpsych.core.calibration.models: grading and content hashing."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from vpsych.core.calibration.models import (
    Calibration,
    ColorCalibration,
    EnvironmentChecklist,
    GammaCalibration,
    PrimaryChromaticity,
    grade_color,
    grade_luminance,
)
from vpsych.core.display import DisplayGeometry


def _geometry() -> DisplayGeometry:
    return DisplayGeometry(
        width_px=1920,
        height_px=1080,
        width_cm=53.13,
        height_cm=29.88,
        viewing_distance_cm=57.0,
        refresh_hz=60.0,
    )


def _gamma(method: str = "photometer") -> GammaCalibration:
    return GammaCalibration(
        method=method,  # type: ignore[arg-type]
        gamma_single=2.2,
        lum_min_cdm2=0.3,
        lum_max_cdm2=120.0,
    )


def _srgb_primaries() -> dict[str, PrimaryChromaticity]:
    return {
        "red": PrimaryChromaticity(x=0.64, y=0.33, Y_cdm2=22.0),
        "green": PrimaryChromaticity(x=0.30, y=0.60, Y_cdm2=72.0),
        "blue": PrimaryChromaticity(x=0.15, y=0.06, Y_cdm2=6.0),
        "white": PrimaryChromaticity(x=0.3127, y=0.3290, Y_cdm2=120.0),
    }


def _color(method: str = "measured") -> ColorCalibration:
    return ColorCalibration(method=method, **_srgb_primaries())  # type: ignore[arg-type]


def _environment() -> EnvironmentChecklist:
    return EnvironmentChecklist(
        room_lighting_controlled=True,
        monitor_warmed_up=True,
        night_light_disabled=True,
        hdr_disabled=True,
    )


def _calibration(**overrides: object) -> Calibration:
    kwargs: dict[str, object] = {
        "created_utc": datetime(2026, 9, 1, tzinfo=timezone.utc),
        "geometry": _geometry(),
        "gamma": _gamma(),
        "color": _color(),
        "environment": _environment(),
        "software_version": "0.1.0",
    }
    kwargs.update(overrides)
    return Calibration(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("method", "expected"),
    [("photometer", "A"), ("psychophysical", "B"), ("none", "C")],
)
def test_grade_luminance(method: str, expected: str) -> None:
    assert grade_luminance(method) == expected  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("method", "expected"),
    [("measured", "A"), ("srgb_assumed", "C")],
)
def test_grade_color(method: str, expected: str) -> None:
    assert grade_color(method) == expected  # type: ignore[arg-type]


def test_calibration_luminance_and_color_grade_properties() -> None:
    cal = _calibration(gamma=_gamma("psychophysical"), color=_color("srgb_assumed"))
    assert cal.luminance_grade == "B"
    assert cal.color_grade == "C"


def test_content_hash_is_deterministic() -> None:
    cal1 = _calibration()
    cal2 = _calibration()
    assert cal1.content_hash() == cal2.content_hash()


def test_content_hash_is_64_char_hex() -> None:
    h = _calibration().content_hash()
    assert len(h) == 64
    int(h, 16)  # raises if not valid hex


def test_content_hash_changes_with_field_change() -> None:
    cal1 = _calibration()
    cal2 = _calibration(notes="different")
    assert cal1.content_hash() != cal2.content_hash()


def test_content_hash_changes_with_timestamp() -> None:
    cal1 = _calibration(created_utc=datetime(2026, 9, 1, tzinfo=timezone.utc))
    cal2 = _calibration(created_utc=datetime(2026, 9, 2, tzinfo=timezone.utc))
    assert cal1.content_hash() != cal2.content_hash()


def test_created_utc_requires_timezone() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        _calibration(created_utc=datetime(2026, 9, 1))


def test_is_stale_false_when_fresh() -> None:
    cal = _calibration(created_utc=datetime(2026, 9, 1, tzinfo=timezone.utc))
    now = datetime(2026, 9, 5, tzinfo=timezone.utc)
    assert cal.is_stale(max_age_days=30, now=now) is False


def test_is_stale_true_when_old() -> None:
    cal = _calibration(created_utc=datetime(2026, 1, 1, tzinfo=timezone.utc))
    now = datetime(2026, 9, 16, tzinfo=timezone.utc)
    assert cal.is_stale(max_age_days=30, now=now) is True


def test_is_stale_boundary() -> None:
    created = datetime(2026, 1, 1, tzinfo=timezone.utc)
    cal = _calibration(created_utc=created)
    just_under = created + timedelta(days=29, hours=23)
    just_over = created + timedelta(days=30, hours=1)
    assert cal.is_stale(max_age_days=30, now=just_under) is False
    assert cal.is_stale(max_age_days=30, now=just_over) is True


def test_is_stale_defaults_now_to_current_time() -> None:
    cal = _calibration(created_utc=datetime(2000, 1, 1, tzinfo=timezone.utc))
    assert cal.is_stale() is True


def test_calibration_is_immutable() -> None:
    cal = _calibration()
    with pytest.raises(Exception):  # noqa: B017 - pydantic frozen model raises
        cal.notes = "mutate"  # type: ignore[misc]
