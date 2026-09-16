"""Tests for vpsych.data.quality quality-flag computation."""

from __future__ import annotations

from vpsych.data.quality import (
    check_calibration_quality,
    check_catch_lapse_rate,
    check_dropped_frames,
    check_goodness_of_fit,
    check_threshold_at_range_edge,
    check_trial_count,
    compute_quality_flags,
)


def test_catch_lapse_rate_below_threshold_no_flag() -> None:
    assert check_catch_lapse_rate(0.05, n_catch=20) == []


def test_catch_lapse_rate_warning() -> None:
    flags = check_catch_lapse_rate(0.15, n_catch=20)
    assert len(flags) == 1
    assert flags[0].severity == "warning"
    assert flags[0].code == "high_catch_lapse_rate"


def test_catch_lapse_rate_critical() -> None:
    flags = check_catch_lapse_rate(0.25, n_catch=20)
    assert len(flags) == 1
    assert flags[0].severity == "critical"


def test_catch_lapse_rate_zero_catch_trials_no_flag() -> None:
    assert check_catch_lapse_rate(0.5, n_catch=0) == []


def test_dropped_frames_below_threshold_no_flag() -> None:
    assert check_dropped_frames(0.005) == []


def test_dropped_frames_above_threshold_warns() -> None:
    flags = check_dropped_frames(0.02)
    assert len(flags) == 1
    assert flags[0].code == "excess_dropped_frames"
    assert flags[0].severity == "warning"


def test_threshold_at_range_edge_low() -> None:
    flags = check_threshold_at_range_edge(threshold=0.01, range_min=0.0, range_max=1.0)
    assert len(flags) == 1
    assert flags[0].code == "threshold_at_range_edge"


def test_threshold_at_range_edge_high() -> None:
    flags = check_threshold_at_range_edge(threshold=0.99, range_min=0.0, range_max=1.0)
    assert len(flags) == 1


def test_threshold_mid_range_no_flag() -> None:
    assert check_threshold_at_range_edge(threshold=0.5, range_min=0.0, range_max=1.0) == []


def test_threshold_degenerate_range_no_flag() -> None:
    assert check_threshold_at_range_edge(threshold=0.5, range_min=1.0, range_max=1.0) == []


def test_calibration_quality_good_no_flag() -> None:
    assert check_calibration_quality("A", calibration_age_days=1) == []


def test_calibration_quality_low_grade_flag() -> None:
    flags = check_calibration_quality("C", calibration_age_days=1)
    assert any(f.code == "low_calibration_grade" for f in flags)


def test_calibration_quality_stale_flag() -> None:
    flags = check_calibration_quality("A", calibration_age_days=40)
    assert any(f.code == "stale_calibration" for f in flags)


def test_calibration_quality_both_flags() -> None:
    flags = check_calibration_quality("C", calibration_age_days=40)
    codes = {f.code for f in flags}
    assert codes == {"low_calibration_grade", "stale_calibration"}


def test_goodness_of_fit_none_no_flag() -> None:
    assert check_goodness_of_fit(None) == []


def test_goodness_of_fit_poor() -> None:
    flags = check_goodness_of_fit(0.01)
    assert len(flags) == 1
    assert flags[0].code == "poor_gof"


def test_goodness_of_fit_good_no_flag() -> None:
    assert check_goodness_of_fit(0.5) == []


def test_trial_count_sufficient_no_flag() -> None:
    assert check_trial_count(50, min_trials=40) == []


def test_trial_count_insufficient_flag() -> None:
    flags = check_trial_count(10, min_trials=40)
    assert len(flags) == 1
    assert flags[0].code == "too_few_trials"


def test_compute_quality_flags_aggregates_everything() -> None:
    flags = compute_quality_flags(
        catch_lapse_rate=0.25,
        n_catch=20,
        dropped_fraction=0.02,
        threshold=0.01,
        range_min=0.0,
        range_max=1.0,
        luminance_grade="C",
        calibration_age_days=40,
        gof_p_value=0.01,
        n_trials=10,
        min_trials=40,
    )
    codes = {f.code for f in flags}
    assert codes == {
        "high_catch_lapse_rate",
        "excess_dropped_frames",
        "threshold_at_range_edge",
        "low_calibration_grade",
        "stale_calibration",
        "poor_gof",
        "too_few_trials",
    }


def test_compute_quality_flags_skips_omitted_optional_checks() -> None:
    flags = compute_quality_flags(catch_lapse_rate=0.0, n_catch=10, dropped_fraction=0.0)
    assert flags == []
