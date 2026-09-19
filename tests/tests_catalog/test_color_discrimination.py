"""Tests for vpsych.tests_catalog.color_discrimination: the CCT-style trivector test.

Mirrors the "required tests" pattern in `docs/WRITING_A_TEST.md` (see
`tests/tests_catalog/test_example.py` for the canonical instance of each):
requirements check, response scoring, disc packing properties, color
round-trip/gamut, composite-procedure routing/state, `summarize()` on a
fixture (plus reanalysis-style reproducibility), simulated end-to-end
recovery of three distinct per-axis thresholds, and a slow bias/coverage
sweep of `TrivectorProcedure` (this test's own composite procedure, which
does not live under `core/procedures` and so is not covered by any other
test file).
"""

from __future__ import annotations

import itertools
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from vpsych.core.calibration.color import matrix_from_calibration
from vpsych.core.calibration.models import (
    Calibration,
    ColorCalibration,
    EnvironmentChecklist,
    GammaCalibration,
    PrimaryChromaticity,
)
from vpsych.core.display import DisplayGeometry
from vpsych.core.psychometric import PsychometricFunction
from vpsych.runner.__main__ import build_arg_parser, run_session
from vpsych.runner.status import RunnerExitCode
from vpsych.tests_catalog import base as catalog_base
from vpsych.tests_catalog.base import check_requirements
from vpsych.tests_catalog.color_discrimination import (
    AXES,
    DEFAULT_BACKGROUND_UV,
    ColorDiscriminationParams,
    ColorDiscriminationTest,
    colorspace,
    discs,
)
from vpsych.tests_catalog.color_discrimination.observer import TrivectorObserver
from vpsych.tests_catalog.color_discrimination.procedure import TrivectorProcedure

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _display() -> DisplayGeometry:
    return DisplayGeometry(
        width_px=1920,
        height_px=1080,
        width_cm=53.13,
        height_cm=29.88,
        viewing_distance_cm=57.0,
        refresh_hz=60.0,
    )


def _color_calibration(method: str = "measured") -> ColorCalibration:
    return ColorCalibration(
        method=method,  # type: ignore[arg-type]
        red=PrimaryChromaticity(x=0.64, y=0.33, Y_cdm2=22.0),
        green=PrimaryChromaticity(x=0.30, y=0.60, Y_cdm2=72.0),
        blue=PrimaryChromaticity(x=0.15, y=0.06, Y_cdm2=6.0),
        white=PrimaryChromaticity(x=0.3127, y=0.3290, Y_cdm2=120.0),
    )


def _calibration(color_method: str = "measured") -> Calibration:
    return Calibration(
        created_utc=datetime(2026, 9, 1, tzinfo=timezone.utc),
        geometry=_display(),
        gamma=GammaCalibration(
            method="photometer", gamma_single=2.2, lum_min_cdm2=0.3, lum_max_cdm2=120.0
        ),
        color=_color_calibration(color_method),
        environment=EnvironmentChecklist(
            room_lighting_controlled=True,
            monitor_warmed_up=True,
            night_light_disabled=True,
            hdr_disabled=True,
        ),
        software_version="0.1.0",
    )


def _make_test(
    calibration: Calibration | None = None, params: ColorDiscriminationParams | None = None
) -> ColorDiscriminationTest:
    return ColorDiscriminationTest(
        params=params or ColorDiscriminationParams(),
        display=_display(),
        calibration=calibration or _calibration(),
        rng=np.random.default_rng(0),
    )


# ---------------------------------------------------------------------------
# 1. Requirements check
# ---------------------------------------------------------------------------


def test_requirements_met_with_gamma_calibration_only() -> None:
    """Only a gamma calibration (grade B+) is required -- sRGB-assumed color is fine."""
    cal = _calibration(color_method="srgb_assumed")
    reasons = check_requirements(ColorDiscriminationTest.spec.requirements, _display(), cal)
    assert reasons == []


def test_requirements_unmet_without_any_calibration() -> None:
    reasons = check_requirements(ColorDiscriminationTest.spec.requirements, _display(), None)
    assert any("gamma" in r.lower() for r in reasons)


def test_is_registered_and_visible() -> None:
    catalog_base.discover_tests()
    assert catalog_base.get_test("color_discrimination") is ColorDiscriminationTest
    assert ColorDiscriminationTest.spec.hidden is False
    assert ColorDiscriminationTest.spec.id in {t.spec.id for t in catalog_base.visible_tests()}


# ---------------------------------------------------------------------------
# 2. Score correctness
# ---------------------------------------------------------------------------


def test_score_correct_and_incorrect() -> None:
    test = _make_test()
    stim = {"correct_response": "up"}
    assert test.score("up", stim) is True
    assert test.score("down", stim) is False
    assert test.score(None, stim) is False


# ---------------------------------------------------------------------------
# 3. Disc packing properties
# ---------------------------------------------------------------------------


def _generate(seed: int, orientation: str = "up") -> list[discs.Disc]:
    rng = np.random.default_rng(seed)
    return discs.generate_disc_field(
        rng,
        field_size_deg=4.3 * 1.3,
        min_diameter_deg=0.15,
        max_diameter_deg=0.45,
        outer_diameter_deg=4.3,
        stroke_width_deg=4.3 / 5.0,
        gap_deg=1.0,
        orientation=orientation,
    )


def test_disc_field_is_non_overlapping() -> None:
    field = _generate(seed=1)
    assert len(field) > 20
    for a, b in itertools.combinations(field, 2):
        dist = ((a.x_deg - b.x_deg) ** 2 + (a.y_deg - b.y_deg) ** 2) ** 0.5
        min_sep = a.diameter_deg / 2.0 + b.diameter_deg / 2.0
        assert dist >= min_sep - 1e-9


def test_disc_field_coverage_is_substantial() -> None:
    field = _generate(seed=2)
    coverage = discs.field_coverage_fraction(field, 4.3 * 1.3)
    # Random sequential adsorption of equal-ish circles typically jams
    # somewhere in the 0.3-0.55 range; just check it isn't a near-empty or
    # impossible (>1) field.
    assert 0.2 < coverage < 0.9


def test_disc_field_deterministic_by_seed() -> None:
    field_a = _generate(seed=42)
    field_b = _generate(seed=42)
    assert field_a == field_b
    field_c = _generate(seed=43)
    assert field_a != field_c


def test_disc_field_has_target_and_background_discs() -> None:
    field = _generate(seed=5)
    n_target = sum(1 for d in field if d.is_target)
    assert 0 < n_target < len(field)


def test_is_in_target_region_respects_orientation() -> None:
    # A point at the ring's mean radius, straight up, is in the target
    # region for every orientation except "up" (whose gap sits there).
    outer, stroke, gap = 4.3, 4.3 / 5.0, 1.0
    x, y = 0.0, (outer / 2.0 + (outer / 2.0 - stroke)) / 2.0
    assert discs.is_in_target_region(x, y, outer, stroke, gap, "down") is True
    assert discs.is_in_target_region(x, y, outer, stroke, gap, "up") is False


def test_generate_disc_field_rejects_bad_diameter_range() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="positive"):
        discs.generate_disc_field(
            rng,
            field_size_deg=5.0,
            min_diameter_deg=0.0,
            max_diameter_deg=0.5,
            outer_diameter_deg=4.3,
            stroke_width_deg=0.86,
            gap_deg=1.0,
            orientation="up",
        )
    with pytest.raises(ValueError, match=">="):
        discs.generate_disc_field(
            rng,
            field_size_deg=5.0,
            min_diameter_deg=0.5,
            max_diameter_deg=0.1,
            outer_diameter_deg=4.3,
            stroke_width_deg=0.86,
            gap_deg=1.0,
            orientation="up",
        )


# ---------------------------------------------------------------------------
# 4. Color round-trip and gamut
# ---------------------------------------------------------------------------


def test_render_round_trip_matches_within_tolerance() -> None:
    color_cal = _color_calibration()
    gamma_cal = GammaCalibration(
        method="photometer", gamma_single=2.2, lum_min_cdm2=0.3, lum_max_cdm2=120.0
    )
    matrix = matrix_from_calibration(color_cal, absolute=True)
    u, v, y = 0.1977, 0.4689, 12.0
    render = colorspace.render_disc(u, v, y, color_cal, gamma_cal, rgb_to_xyz_abs=matrix)
    assert render.in_gamut is True
    u2, v2, y2 = colorspace.inverse_render_to_uv_y(
        render.drive_rgb, color_cal, gamma_cal, rgb_to_xyz_abs=matrix
    )
    assert abs(u2 - u) < 1e-4
    assert abs(v2 - v) < 1e-4
    assert abs(y2 - y) < 1e-4 * y


def test_out_of_gamut_chromaticity_is_clamped_and_flagged() -> None:
    color_cal = _color_calibration()
    gamma_cal = GammaCalibration(
        method="photometer", gamma_single=2.2, lum_min_cdm2=0.3, lum_max_cdm2=120.0
    )
    matrix = matrix_from_calibration(color_cal, absolute=True)
    # Far outside the sRGB-ish triangle at this luminance.
    render = colorspace.render_disc(0.6, 0.55, 12.0, color_cal, gamma_cal, rgb_to_xyz_abs=matrix)
    assert render.in_gamut is False
    assert np.all(render.linear_rgb >= 0.0) and np.all(render.linear_rgb <= 1.0)


def test_max_in_gamut_displacement_is_positive_and_bounded() -> None:
    color_cal = _color_calibration()
    matrix = matrix_from_calibration(color_cal, absolute=True)
    direction = (1.0, 0.0)
    max_d = colorspace.max_in_gamut_displacement_uv(
        DEFAULT_BACKGROUND_UV, direction, matrix, [8.0, 13.0, 18.0], search_hi_uv=0.5
    )
    assert 0.0 < max_d < 0.5
    # A displacement just past the found maximum must not be in gamut anymore
    # (bisection actually found the boundary, not an arbitrary safe value).
    just_over = (
        DEFAULT_BACKGROUND_UV[0] + direction[0] * (max_d + 1e-3),
        DEFAULT_BACKGROUND_UV[1] + direction[1] * (max_d + 1e-3),
    )
    assert not colorspace.chromaticity_in_gamut_at_luminances(just_over, matrix, [8.0, 13.0, 18.0])


def test_background_falls_back_when_out_of_gamut() -> None:
    """A display whose max luminance barely covers the noise range's top forces a fallback."""
    display = _display()
    white_y = 18.0
    cal = Calibration(
        created_utc=datetime(2026, 9, 1, tzinfo=timezone.utc),
        geometry=display,
        gamma=GammaCalibration(
            method="photometer", gamma_single=2.2, lum_min_cdm2=0.3, lum_max_cdm2=white_y
        ),
        color=ColorCalibration(
            method="measured",
            red=PrimaryChromaticity(x=0.64, y=0.33, Y_cdm2=white_y * 0.2),
            green=PrimaryChromaticity(x=0.30, y=0.60, Y_cdm2=white_y * 0.6),
            blue=PrimaryChromaticity(x=0.15, y=0.06, Y_cdm2=white_y * 0.05),
            white=PrimaryChromaticity(x=0.3127, y=0.3290, Y_cdm2=white_y),
        ),
        environment=EnvironmentChecklist(
            room_lighting_controlled=True,
            monitor_warmed_up=True,
            night_light_disabled=True,
            hdr_disabled=True,
        ),
        software_version="0.1.0",
    )
    test = _make_test(calibration=cal)
    assert test.background_fallback_used is True
    assert test.background_uv != DEFAULT_BACKGROUND_UV


def test_background_not_fallen_back_on_a_normal_display() -> None:
    test = _make_test()
    assert test.background_fallback_used is False
    assert test.background_uv == DEFAULT_BACKGROUND_UV


def test_axis_max_displacement_is_computed_per_axis_and_differs() -> None:
    test = _make_test()
    maxima = test._axis_max_displacement_x1e4
    assert set(maxima) == set(AXES)
    assert all(v > 0 for v in maxima.values())
    # The three axes need not (and, for a typical sRGB-ish gamut, generally
    # do not) share the same gamut-limited ceiling.
    assert len({round(v, 3) for v in maxima.values()}) > 1


# ---------------------------------------------------------------------------
# 5. Composite procedure routing and state
# ---------------------------------------------------------------------------


def test_make_procedure_builds_a_trivector_procedure() -> None:
    test = _make_test(params=ColorDiscriminationParams(max_trials=30))
    proc = test.make_procedure()
    assert isinstance(proc, TrivectorProcedure)
    assert set(proc.axis_procedures) == set(AXES)


def test_trivector_procedure_rejects_bad_axis_keys() -> None:
    test = _make_test(params=ColorDiscriminationParams(max_trials=30))
    proc = test.make_procedure()
    bad = {k: v for k, v in proc.axis_procedures.items() if k != "tritan"}
    with pytest.raises(ValueError, match=r"protan.*deutan.*tritan"):
        TrivectorProcedure(axis_procedures=bad, rng=np.random.default_rng(0), max_trials_total=10)


def test_trivector_procedure_routes_updates_to_the_correct_axis() -> None:
    test = _make_test(params=ColorDiscriminationParams(max_trials=30))
    proc = test.make_procedure()
    rng = np.random.default_rng(0)

    axis_trial_counts_before = {
        axis: sub.state_dict()["n_trials"] for axis, sub in proc.axis_procedures.items()
    }
    assert set(axis_trial_counts_before.values()) == {0}

    for _ in range(9):
        stim = proc.next_stimulus()
        assert set(stim) == {"axis", "intensity"}
        axis_idx = round(stim["axis"])
        axis = AXES[axis_idx]
        n_before = {a: sub.state_dict()["n_trials"] for a, sub in proc.axis_procedures.items()}
        proc.update(stim, bool(rng.random() < 0.5))
        n_after = {a: sub.state_dict()["n_trials"] for a, sub in proc.axis_procedures.items()}
        # Only the drawn axis's sub-procedure should have advanced; every
        # other axis's own trial count must be exactly unchanged.
        assert n_after[axis] == n_before[axis] + 1
        for other_axis in AXES:
            if other_axis != axis:
                assert n_after[other_axis] == n_before[other_axis]

    assert proc.state_dict()["n_trials"] == 9


def test_trivector_procedure_finishes_at_max_trials_total() -> None:
    test = _make_test(params=ColorDiscriminationParams(max_trials=30))
    proc = test.make_procedure()
    rng = np.random.default_rng(0)
    n = 0
    while not proc.finished:
        stim = proc.next_stimulus()
        proc.update(stim, bool(rng.random() < 0.5))
        n += 1
        assert n <= 30  # safety: must not run away past the configured budget
    assert n == 30
    assert proc.state_dict()["finished"] is True


def test_trivector_procedure_state_dict_is_json_serializable() -> None:
    test = _make_test(params=ColorDiscriminationParams(max_trials=30))
    proc = test.make_procedure()
    rng = np.random.default_rng(0)
    for _ in range(5):
        stim = proc.next_stimulus()
        proc.update(stim, bool(rng.random() < 0.5))
    json.dumps(proc.state_dict())  # must not raise


def test_trivector_procedure_estimate_before_any_trial_raises() -> None:
    test = _make_test(params=ColorDiscriminationParams(max_trials=30))
    proc = test.make_procedure()
    with pytest.raises(RuntimeError, match="before any trials"):
        proc.estimate()


def test_trivector_procedure_estimate_combines_per_axis_geometrically() -> None:
    test = _make_test(params=ColorDiscriminationParams(max_trials=30))
    proc = test.make_procedure()
    rng = np.random.default_rng(1)
    while not proc.finished:
        stim = proc.next_stimulus()
        proc.update(stim, bool(rng.random() < 0.6))
    est = proc.estimate()
    per_axis = est.extra["per_axis"]
    assert set(per_axis) == set(AXES)
    expected_value = float(np.mean([per_axis[axis]["threshold_log10_uv_x1e4"] for axis in AXES]))
    assert est.value == pytest.approx(expected_value)
    assert est.units == "log10_uv_displacement_x1e4"


# ---------------------------------------------------------------------------
# 6. summarize() on a fixture, plus reproducibility
# ---------------------------------------------------------------------------


def _trials_fixture(test: ColorDiscriminationTest, n_per_axis: int = 15) -> pd.DataFrame:
    """A fixed, deterministic trials table spanning all three axes plus some catch trials.

    Intensities must land exactly on each axis's QUEST+ intensity grid --
    `questplus.QuestPlus.update` looks the presented stimulus up by exact
    value (`xarray`'s `.sel()`, no tolerance), so an arbitrary derived float
    (even one mathematically "in range") raises `KeyError` unless it happens
    to coincide with a grid point.
    """
    rows: list[dict[str, Any]] = []
    trial_index = 0
    probe_procedure = test.make_procedure()
    intensities = {
        axis: probe_procedure.axis_procedures[axis].intensity_values[10] for axis in AXES
    }
    for axis_idx, axis in enumerate(AXES):
        for i in range(n_per_axis):
            correct = i % 5 != 0  # 80% correct, deterministic
            rows.append(
                {
                    "block": "main",
                    "is_catch": False,
                    "trial_index": trial_index,
                    "intensity": intensities[axis],
                    "correct": correct,
                    "eye": "OU",
                    "n_dropped_frames_trial": 0,
                    "stimulus_params": {"axis": axis_idx, "axis_name": axis},
                }
            )
            trial_index += 1
    for _i in range(4):
        rows.append(
            {
                "block": "main",
                "is_catch": True,
                "trial_index": trial_index,
                "intensity": intensities["protan"],
                "correct": True,
                "eye": "OU",
                "n_dropped_frames_trial": 0,
                "stimulus_params": {"axis": 0, "axis_name": "protan"},
            }
        )
        trial_index += 1
    return pd.DataFrame(rows)


def test_summarize_on_fixture_is_deterministic_and_well_formed() -> None:
    test = _make_test(params=ColorDiscriminationParams(max_trials=45))
    df = _trials_fixture(test)
    summary1 = test.summarize(df)
    summary2 = test.summarize(df)
    assert summary1.model_dump() == summary2.model_dump()
    assert summary1.n_trials == 45
    assert summary1.n_catch == 4
    assert summary1.catch_lapse_rate == pytest.approx(0.0)
    assert summary1.estimate.ci_low <= summary1.estimate.value <= summary1.estimate.ci_high
    assert set(summary1.estimate.extra["per_axis"]) == set(AXES)


def test_summarize_reproducible_by_a_fresh_test_instance() -> None:
    """Simulates `vpsych reanalyze`: a brand-new test instance (fresh rng) must reproduce the
    exact same summary from the same trials table, since summarize() must be a pure function
    of `trials` (see docs/WRITING_A_TEST.md section 10)."""
    test_a = _make_test(params=ColorDiscriminationParams(max_trials=45), calibration=_calibration())
    df = _trials_fixture(test_a)
    summary_a = test_a.summarize(df)

    test_b = ColorDiscriminationTest(
        params=ColorDiscriminationParams(max_trials=45),
        display=_display(),
        calibration=_calibration(),
        rng=np.random.default_rng(999),  # different seed: must not matter for summarize()
    )
    summary_b = test_b.summarize(df)
    assert summary_a.model_dump() == summary_b.model_dump()


def test_summarize_flags_high_catch_lapse_rate() -> None:
    test = _make_test(params=ColorDiscriminationParams(max_trials=45))
    df = _trials_fixture(test)
    df.loc[df["is_catch"], "correct"] = False
    summary = test.summarize(df)
    assert summary.catch_lapse_rate == pytest.approx(1.0)
    assert any(
        f.code == "high_catch_lapse_rate" and f.severity == "critical"
        for f in summary.quality_flags
    )


def test_summarize_flags_unmeasured_color_calibration_as_critical() -> None:
    test = _make_test(
        calibration=_calibration(color_method="srgb_assumed"),
        params=ColorDiscriminationParams(max_trials=45),
    )
    df = _trials_fixture(test)
    summary = test.summarize(df)
    flags = [f for f in summary.quality_flags if f.code == "color_calibration_not_measured"]
    assert len(flags) == 1
    assert flags[0].severity == "critical"
    assert "not research-grade" in flags[0].message


def test_summarize_flags_measured_color_calibration_as_info() -> None:
    test = _make_test(
        calibration=_calibration(color_method="measured"),
        params=ColorDiscriminationParams(max_trials=45),
    )
    df = _trials_fixture(test)
    summary = test.summarize(df)
    flags = [f for f in summary.quality_flags if f.code == "color_calibration_not_measured"]
    assert len(flags) == 1
    assert flags[0].severity == "info"


def test_summarize_logs_calibration_grades_in_fit_params() -> None:
    test = _make_test(params=ColorDiscriminationParams(max_trials=45))
    df = _trials_fixture(test)
    summary = test.summarize(df)
    assert summary.fit_params["color_grade"] == "A"
    assert summary.fit_params["luminance_grade"] == "A"
    assert "background_uv" in summary.fit_params
    assert "axis_max_displacement_uv_x1e4" in summary.fit_params


# ---------------------------------------------------------------------------
# 7. Simulated end-to-end recovery of 3 different per-axis thresholds
# ---------------------------------------------------------------------------


class _FakeWriter:
    def __init__(self) -> None:
        self.trials: list[Any] = []
        self.summaries: list[Any] = []
        self.frames: list[Any] = []
        self.finalized_status: str | None = None

    def append_trial(self, trial: Any) -> None:
        self.trials.append(trial)

    def write_summary(self, task_id: str, eye: str, run: int, summary: Any) -> None:
        self.summaries.append(summary)

    def write_frames(self, task_id: str, eye: str, run: int, intervals_s: list[float]) -> None:
        self.frames.append(intervals_s)

    def finalize(self, status: str) -> None:
        self.finalized_status = status


def _write_calibration(data_root: Path, cal: Calibration) -> str:
    cal_dir = data_root / "calibration"
    cal_dir.mkdir(parents=True, exist_ok=True)
    h = cal.content_hash()
    (cal_dir / f"cal-{h}.json").write_text(cal.model_dump_json(), encoding="utf-8")
    return h


def test_simulated_end_to_end_recovery_of_three_distinct_thresholds(tmp_path: Path) -> None:
    """Drives a real run_session() with a per-axis TrivectorObserver (a protan-like elevated
    threshold, a typical deutan threshold, and a low tritan threshold) and checks the recovered
    per-axis thresholds track their distinct ground truths.

    Goes through the normal `--simulate` CLI-spec path (`parse_simulated_observer_spec` ->
    `build_simulated_observer`'s registry, see `vpsych.runner.__main__
    .register_simulated_observer_kind` and observer.py's module docstring), rather than
    injecting a directly-constructed observer via `run_session`'s `simulated_observer=` kwarg
    -- once a real limitation (a closed if/elif that only vpsych.runner.__main__ could extend),
    now exercised as a registered "trivector" kind like any built-in one.
    """
    catalog_base.discover_tests()
    data_root = tmp_path / "data"
    cal = _calibration()
    _write_calibration(data_root, cal)

    true_thresholds = {"protan": 2.3, "deutan": 1.6, "tritan": 1.1}  # log10 uv x1e-4 units
    simulate_spec = "trivector:" + ",".join(
        f"threshold_{axis}={v}" for axis, v in true_thresholds.items()
    )

    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "participant_id": "sub-0001",
                "tests": [
                    {
                        "task_id": "color_discrimination",
                        "eye": "OU",
                        "params": {"max_trials": 240},
                        "viewing_distance_cm": 57.0,
                    }
                ],
                "ordering": "fixed",
                "seed": 7,
            }
        ),
        encoding="utf-8",
    )

    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--session-plan",
            str(plan_path),
            "--status-file",
            str(tmp_path / "status.json"),
            "--data-root",
            str(data_root),
            "--simulate",
            simulate_spec,
        ]
    )
    writer = _FakeWriter()
    exit_code = run_session(args, writer_factory=lambda *a: writer)
    status = (
        (tmp_path / "status.json").read_text(encoding="utf-8")
        if exit_code != RunnerExitCode.OK
        else ""
    )
    assert exit_code == RunnerExitCode.OK, status
    assert len(writer.summaries) == 1

    summary = writer.summaries[0]
    per_axis = summary.estimate.extra["per_axis"]
    for axis, true_value in true_thresholds.items():
        recovered = per_axis[axis]["threshold_log10_uv_x1e4"]
        assert abs(recovered - true_value) < 1.0  # loose smoke-level tolerance, see _example

    # The three recovered thresholds should preserve the ground truth's
    # relative ordering (protan highest, tritan lowest) -- a coarser but
    # more robust check than each axis's absolute tolerance alone.
    assert (
        per_axis["protan"]["threshold_log10_uv_x1e4"]
        > per_axis["tritan"]["threshold_log10_uv_x1e4"]
    )


# ---------------------------------------------------------------------------
# 8. Slow bias/coverage sweep of TrivectorProcedure
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_trivector_procedure_recovery_bias_slow() -> None:
    """Per-axis threshold bias, averaged over many simulated runs.

    `TrivectorProcedure` lives entirely in this package (not
    `core/procedures`), so its own bias/coverage characterization belongs
    here rather than in `tests/procedures/` (see docs/WRITING_A_TEST.md
    section 11, item 3). A pilot sweep (40 reps, 180 total trials -- i.e.
    ~60 trials/axis on average, since the axis is drawn at random each
    trial) measured per-axis bias of about 0.13-0.18 log10 units (SD about
    0.13-0.17), somewhat higher than qCSF's ~0.04-0.08 at 100+ trials,
    consistent with substantially fewer trials per axis here. This test
    uses fewer reps than that pilot (for runtime) and a correspondingly
    looser bound to avoid flaking on ordinary sampling variation while
    still catching a much larger, genuinely broken bias.
    """
    test = _make_test(params=ColorDiscriminationParams(max_trials=180))
    true_thresholds = {"protan": 2.2, "deutan": 1.7, "tritan": 1.3}
    funcs = {
        axis: PsychometricFunction(
            family="weibull",
            threshold=v,
            slope=0.4,
            guess=0.25,
            lapse=0.02,
            intensity_scale="log10",
        )
        for axis, v in true_thresholds.items()
    }
    observer = TrivectorObserver(funcs)
    rng = np.random.default_rng(7)

    n_reps = 30
    diffs = {axis: [] for axis in AXES}
    for _ in range(n_reps):
        proc = test.make_procedure()
        while not proc.finished:
            stim = proc.next_stimulus()
            correct = observer.decide_correct(stim, rng)
            proc.update(stim, correct)
        est = proc.estimate()
        for axis in AXES:
            diffs[axis].append(
                est.extra["per_axis"][axis]["threshold_log10_uv_x1e4"] - true_thresholds[axis]
            )

    for axis in AXES:
        mean_bias = float(np.mean(diffs[axis]))
        assert abs(mean_bias) < 0.35, f"{axis} bias {mean_bias:.3f} exceeds bound"


# ---------------------------------------------------------------------------
# 9. Display smoke test
# ---------------------------------------------------------------------------


class _FakeKeyboard:
    """Minimal duck-typed stand-in for psychopy.hardware.keyboard.Keyboard, for the smoke test.

    Method/argument names match PsychoPy's own (non-snake_case) API exactly,
    since `present()` calls them polymorphically on whatever
    `trial_ctx["keyboard"]` is.
    """

    def clearEvents(self) -> None:  # noqa: N802
        pass

    def waitKeys(  # noqa: N802
        self,
        keyList: list[str] | None = None,  # noqa: N803
        timeStamped: bool = True,  # noqa: N803
    ) -> list[Any]:
        del keyList, timeStamped
        # Use PsychoPy's own global clock (the same one win.flip() timestamps
        # come from) so the fabricated "keypress" timestamp is always after
        # the stimulus-onset flip time, giving a non-negative rt_s -- a bare
        # 0.0 here would (correctly) fail PresentedTrial's rt_s >= 0
        # validation once run alongside a real window's monotonic clock.
        from psychopy.core import getTime

        return [("up", getTime())]


@pytest.mark.display
def test_present_renders_on_a_real_window() -> None:
    """Opens a real (small, non-fullscreen) PsychoPy window and exercises the actual
    present() real-display code path (disc-field generation, per-disc color rendering,
    dithering, ElementArrayStim construction, frame flips) end to end, with a stubbed
    keyboard standing in for a live keypress."""
    from psychopy import visual

    win = visual.Window(size=(800, 600), fullscr=False, units="pix", allowGUI=False)
    try:
        display = DisplayGeometry(
            width_px=800,
            height_px=600,
            width_cm=30.0,
            height_cm=22.5,
            viewing_distance_cm=57.0,
            refresh_hz=60.0,
        )
        test = ColorDiscriminationTest(
            params=ColorDiscriminationParams(
                max_trials=30, stimulus_duration_ms=50.0, fixation_ms=16.0, iti_ms=16.0
            ),
            display=display,
            calibration=_calibration(),
            rng=np.random.default_rng(0),
        )
        test.build_stimuli(win)
        trial_ctx: dict[str, Any] = {
            "timeline": None,
            "rng": np.random.default_rng(1),
            "block": "main",
            "is_catch": False,
            "trial_index": 0,
            "keyboard": _FakeKeyboard(),
            "simulated_observer": None,
        }
        presented = test.present(win, test.make_catch_trial_intensity(), trial_ctx)
        assert presented.response == "up"
        assert presented.n_dropped_frames == 0
    finally:
        win.close()
