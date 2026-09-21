"""Tests for `vpsych.app.viewmodels.results`."""

from __future__ import annotations

from vpsych.app.viewmodels.results import (
    build_quality_flag_view_models,
    build_result_view_model,
    format_threshold,
    severity_label,
    severity_marker,
)
from vpsych.core.procedures.base import ThresholdEstimate
from vpsych.data.schemas import QualityFlag, TestSummary


def test_format_threshold_formats_value_ci_and_units() -> None:
    text = format_threshold(-0.123456, -0.2, -0.05, 0.95, "logMAR")
    assert "logMAR" in text
    assert "95%" in text
    assert "-0.123" in text


def test_severity_marker_and_label_are_distinct_per_severity() -> None:
    markers = {severity_marker(s) for s in ("info", "warning", "critical")}
    labels = {severity_label(s) for s in ("info", "warning", "critical")}
    assert len(markers) == 3
    assert len(labels) == 3


def test_build_quality_flag_view_models_sorts_critical_first() -> None:
    flags = [
        QualityFlag(code="a", severity="info", message="a"),
        QualityFlag(code="b", severity="critical", message="b"),
        QualityFlag(code="c", severity="warning", message="c"),
    ]
    vms = build_quality_flag_view_models(flags)
    assert [v.severity for v in vms] == ["critical", "warning", "info"]
    assert vms[0].code == "b"


def _make_summary(**overrides: object) -> TestSummary:
    kwargs: dict[str, object] = {
        "task_id": "visual_acuity",
        "task_version": "1.0.0",
        "eye": "OD",
        "run": 1,
        "estimate": ThresholdEstimate(
            value=-0.1, ci_low=-0.2, ci_high=0.0, ci_level=0.95, units="logMAR", method="mean"
        ),
        "fit_params": {},
        "gof": {},
        "quality_flags": [],
        "n_trials": 40,
        "n_catch": 4,
        "catch_lapse_rate": 0.05,
        "frame_stats": {},
        "analysis_version": "1.0.0",
    }
    kwargs.update(overrides)
    return TestSummary(**kwargs)  # type: ignore[arg-type]


def test_build_result_view_model_formats_everything_from_real_values() -> None:
    summary = _make_summary(
        quality_flags=[QualityFlag(code="poor_gof", severity="warning", message="Poor fit.")]
    )
    vm = build_result_view_model(summary)
    assert vm.task_id == "visual_acuity"
    assert vm.eye == "OD"
    assert vm.run == 1
    assert "logMAR" in vm.threshold_text
    assert vm.n_trials == 40
    assert vm.n_catch == 4
    assert vm.catch_lapse_rate_text == "5.0%"
    assert vm.analysis_version == "1.0.0"
    assert len(vm.flags) == 1
    assert vm.flags[0].message == "Poor fit."
