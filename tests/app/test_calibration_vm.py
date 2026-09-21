"""Tests for `vpsych.app.viewmodels.calibration`."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from tests.data.conftest import make_calibration, make_display
from vpsych.app.viewmodels.calibration import (
    calibration_badge,
    color_grade_label,
    format_age_days,
    grade_c_limitations,
    luminance_grade_label,
)
from vpsych.core.calibration.models import GammaCalibration


def test_calibration_badge_no_calibration_shows_not_available() -> None:
    vm = calibration_badge(None)
    assert vm.has_calibration is False
    assert vm.luminance_grade_text == "Not available"
    assert vm.color_grade_text == "Not available"
    assert vm.age_text == "Not available"
    assert vm.is_stale is False
    assert vm.stale_warning is None
    assert "No calibration" in vm.summary_text


def test_calibration_badge_fresh_calibration() -> None:
    now = datetime(2026, 9, 21, tzinfo=timezone.utc)
    cal = make_calibration(now=now)
    vm = calibration_badge(cal, now=now)
    assert vm.has_calibration is True
    assert vm.is_stale is False
    assert vm.stale_warning is None
    assert luminance_grade_label(cal.luminance_grade) == vm.luminance_grade_text
    assert color_grade_label(cal.color_grade) == vm.color_grade_text
    assert vm.age_text == "today"


def test_calibration_badge_stale_calibration() -> None:
    created = datetime(2026, 1, 1, tzinfo=timezone.utc)
    now = created + timedelta(days=45)
    cal = make_calibration(now=created)
    vm = calibration_badge(cal, now=now, stale_after_days=30)
    assert vm.is_stale is True
    assert vm.stale_warning is not None
    assert "30" in vm.stale_warning


def test_format_age_days_today_and_days_ago() -> None:
    created = datetime(2026, 1, 1, tzinfo=timezone.utc)
    cal = make_calibration(now=created)
    assert format_age_days(cal, now=created) == "today"
    assert format_age_days(cal, now=created + timedelta(days=1)) == "1 day ago"
    assert format_age_days(cal, now=created + timedelta(days=5)) == "5 days ago"


def test_grade_c_limitations_none_calibration_blocks_gamma_requiring_tests(
    registered_dummy_test,
) -> None:
    from vpsych.tests_catalog.base import TestRequirements, TestSpec

    strict_test = registered_dummy_test
    original_spec = strict_test.spec
    try:
        strict_test.spec = TestSpec(
            id="dummy_test",
            name="Dummy Test",
            version="1.0.0",
            domain="acuity",
            description_participant="n/a",
            description_technical="n/a",
            measures="n/a",
            output_units="logMAR",
            estimated_minutes=1.0,
            allowed_eyes=["OU"],
            requirements=TestRequirements(needs_gamma_calibration=True),
            params_model=original_spec.params_model,
        )
        limitations = grade_c_limitations(make_display(), None, tests=[strict_test])
        assert len(limitations) == 1
        assert limitations[0].task_id == "dummy_test"
        assert any("gamma calibration" in r for r in limitations[0].reasons)
    finally:
        strict_test.spec = original_spec


def test_grade_c_limitations_empty_when_calibrated(registered_dummy_test) -> None:
    cal = make_calibration()
    limitations = grade_c_limitations(make_display(), cal, tests=[registered_dummy_test])
    assert limitations == []


def test_gamma_calibration_grade_mapping() -> None:
    assert GammaCalibration(method="photometer", lum_min_cdm2=0, lum_max_cdm2=1).grade == "A"
    assert GammaCalibration(method="psychophysical", lum_min_cdm2=0, lum_max_cdm2=1).grade == "B"
    assert GammaCalibration(method="none", lum_min_cdm2=0, lum_max_cdm2=1).grade == "C"
