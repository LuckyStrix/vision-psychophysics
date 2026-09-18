"""Tests for vpsych.tests_catalog.contrast_sensitivity_function (qCSF).

Follows the "required tests" pattern from `docs/WRITING_A_TEST.md`:
requirements checks, score correctness, frequency-grid construction
(rendering-adjacent logic specific to this test), `summarize()` on a
fixture (+ reproducibility), a fast simulated end-to-end recovery check via
the real runner, and a `@pytest.mark.slow` recovery check over several true
CSF parameter sets. Rendering math and dithering accuracy are covered in
`tests/tests_catalog/test_contrast_rendering.py` (shared by this test and
`letter_contrast_sensitivity`). A `@pytest.mark.display` smoke test opens a
real PsychoPy window.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from vpsych.core.calibration.models import (
    Calibration,
    ColorCalibration,
    EnvironmentChecklist,
    GammaCalibration,
    PrimaryChromaticity,
)
from vpsych.core.display import DisplayGeometry
from vpsych.core.observers import CSFObserver
from vpsych.core.procedures.qcsf import (
    DEFAULT_PSYCHOMETRIC_SLOPE,
    log_contrast_sensitivity,
)
from vpsych.runner.__main__ import build_arg_parser, run_session
from vpsych.runner.status import RunnerExitCode
from vpsych.tests_catalog import base as catalog_base
from vpsych.tests_catalog.base import check_requirements
from vpsych.tests_catalog.contrast_sensitivity_function import (
    ContrastSensitivityFunctionTest,
    CSFParams,
)


def _display(
    *, width_px: int = 1920, width_cm: float = 53.13, viewing_distance_cm: float = 200.0
) -> DisplayGeometry:
    return DisplayGeometry(
        width_px=width_px,
        height_px=1080,
        width_cm=width_cm,
        height_cm=29.88,
        viewing_distance_cm=viewing_distance_cm,
        refresh_hz=60.0,
    )


def _calibration() -> Calibration:
    return Calibration(
        created_utc=datetime(2026, 9, 1, tzinfo=timezone.utc),
        geometry=_display(),
        gamma=GammaCalibration(
            method="photometer", gamma_single=2.2, lum_min_cdm2=0.3, lum_max_cdm2=120.0
        ),
        color=ColorCalibration(
            method="measured",
            red=PrimaryChromaticity(x=0.64, y=0.33, Y_cdm2=22.0),
            green=PrimaryChromaticity(x=0.30, y=0.60, Y_cdm2=72.0),
            blue=PrimaryChromaticity(x=0.15, y=0.06, Y_cdm2=6.0),
            white=PrimaryChromaticity(x=0.3127, y=0.3290, Y_cdm2=120.0),
        ),
        environment=EnvironmentChecklist(
            room_lighting_controlled=True,
            monitor_warmed_up=True,
            night_light_disabled=True,
            hdr_disabled=True,
        ),
        software_version="0.1.0",
    )


def _make_test(
    *, quick: bool = False, max_trials: int = 60, display: DisplayGeometry | None = None
) -> ContrastSensitivityFunctionTest:
    return ContrastSensitivityFunctionTest(
        params=CSFParams(max_trials=max_trials, quick=quick),
        display=display or _display(),
        calibration=_calibration(),
        rng=np.random.default_rng(0),
    )


# ---------------------------------------------------------------------------
# 1. Requirements check
# ---------------------------------------------------------------------------


def test_requirements_met_with_gamma_b_and_sufficient_distance() -> None:
    reasons = check_requirements(
        ContrastSensitivityFunctionTest.spec.requirements, _display(), _calibration()
    )
    assert reasons == []


def test_requirements_unmet_without_calibration() -> None:
    reasons = check_requirements(
        ContrastSensitivityFunctionTest.spec.requirements, _display(), None
    )
    assert any("gamma calibration" in r for r in reasons)


def test_requirements_unmet_too_close() -> None:
    reasons = check_requirements(
        ContrastSensitivityFunctionTest.spec.requirements,
        _display(viewing_distance_cm=10.0),
        _calibration(),
    )
    assert any("viewing distance" in r for r in reasons)


def test_is_registered_and_visible() -> None:
    catalog_base.discover_tests()
    assert catalog_base.get_test("contrast_sensitivity_function") is ContrastSensitivityFunctionTest
    assert ContrastSensitivityFunctionTest.spec.hidden is False
    assert "contrast_sensitivity_function" in {t.spec.id for t in catalog_base.visible_tests()}


# ---------------------------------------------------------------------------
# 2. Score correctness
# ---------------------------------------------------------------------------


def test_score_correct_and_incorrect() -> None:
    test = _make_test()
    stim = {"correct_response": "left"}
    assert test.score("left", stim) is True
    assert test.score("right", stim) is False
    assert test.score(None, stim) is False


# ---------------------------------------------------------------------------
# 3. Frequency-grid construction (this test's own display-dependent logic)
# ---------------------------------------------------------------------------


def test_frequency_grid_lower_bound_gives_two_cycles_at_default_size() -> None:
    test = _make_test()
    assert test._freq_low_cpd == pytest.approx(2.0 / 4.0)  # default grating_diameter_deg=4.0
    assert min(test._freq_grid) >= test._freq_low_cpd - 1e-9


def test_frequency_grid_upper_bound_is_quarter_of_nyquist() -> None:
    display = _display()
    test = _make_test(display=display)
    assert test._freq_high_cpd == pytest.approx(display.nyquist_cpd / 2.0)
    assert max(test._freq_grid) <= test._freq_high_cpd + 1e-9


def test_reduced_coverage_flagged_for_a_close_display() -> None:
    close_display = _display(viewing_distance_cm=100.0)
    test = _make_test(display=close_display)
    # At 100 cm on this display, nyquist/2 is just under 16 cpd -- reduced coverage.
    assert test._freq_high_cpd < 16.0
    assert test._reduced_coverage is True


def test_full_coverage_not_flagged_for_a_generous_display() -> None:
    far_display = _display(viewing_distance_cm=300.0)
    test = _make_test(display=far_display)
    assert test._freq_high_cpd >= 16.0
    assert test._reduced_coverage is False


def test_make_catch_trial_intensity_is_low_frequency_high_contrast() -> None:
    test = _make_test()
    catch = test.make_catch_trial_intensity()
    assert catch["contrast"] == pytest.approx(test.params.catch_contrast)
    assert catch["spatial_frequency_cpd"] == pytest.approx(test.params.catch_spatial_frequency_cpd)
    assert catch["intensity"] == pytest.approx(np.log10(test.params.catch_contrast))


def test_quick_param_selects_100_trials() -> None:
    quick_test = _make_test(quick=True, max_trials=999)
    assert quick_test._effective_max_trials == 100
    normal_test = _make_test(quick=False, max_trials=60)
    assert normal_test._effective_max_trials == 60


# ---------------------------------------------------------------------------
# 4. Simulated end-to-end recovery via the runner
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


def _true_aulcsf(true_params: dict[str, float], f_lo: float = 1.0, f_hi: float = 18.0) -> float:
    log10_f = np.linspace(np.log10(f_lo), np.log10(f_hi), 200)
    f = 10.0**log10_f
    log_cs = log_contrast_sensitivity(f, **true_params)
    return float(np.trapezoid(log_cs, x=log10_f))


def _run_session_with_csf_observer(
    tmp_path: Path, true_params: dict[str, float], max_trials: int, seed: int
) -> Any:
    catalog_base.discover_tests()
    data_root = tmp_path / f"data_{seed}"
    _write_calibration(data_root, _calibration())

    plan_path = tmp_path / f"plan_{seed}.json"
    plan_path.write_text(
        json.dumps(
            {
                "participant_id": "sub-0001",
                "tests": [
                    {
                        "task_id": "contrast_sensitivity_function",
                        "eye": "OU",
                        "params": {"max_trials": max_trials},
                        "viewing_distance_cm": 200.0,
                    }
                ],
                "ordering": "fixed",
                "seed": seed,
            }
        ),
        encoding="utf-8",
    )

    spec = (
        f"csf:peak_gain={true_params['peak_gain_log10']},"
        f"peak_freq={true_params['peak_freq_cpd']},"
        f"bandwidth={true_params['bandwidth_octaves']},"
        f"low_freq_truncation={true_params['low_freq_truncation_log10']}"
    )
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--session-plan",
            str(plan_path),
            "--status-file",
            str(tmp_path / f"status_{seed}.json"),
            "--data-root",
            str(data_root),
            "--simulate",
            spec,
        ]
    )
    writer = _FakeWriter()
    exit_code = run_session(args, writer_factory=lambda *a: writer)
    assert exit_code == RunnerExitCode.OK
    assert len(writer.summaries) == 1
    return writer.summaries[0]


TRUE_PARAMS = {
    "peak_gain_log10": 1.6,
    "peak_freq_cpd": 3.0,
    "bandwidth_octaves": 3.0,
    "low_freq_truncation_log10": 1.0,
}


def test_simulated_end_to_end_recovery_via_runner(tmp_path: Path) -> None:
    """Smoke-level recovery check (loose tolerance, modest trial count) -- the
    properly-powered bias/coverage validation lives in
    tests/procedures/test_qcsf.py, not duplicated here (docs/WRITING_A_TEST.md
    section 11)."""
    summary = _run_session_with_csf_observer(tmp_path, TRUE_PARAMS, max_trials=150, seed=42)
    true_aulcsf = _true_aulcsf(TRUE_PARAMS)
    assert abs(summary.estimate.value - true_aulcsf) < 0.5


def test_quick_run_flags_bias_quality_concern(tmp_path: Path) -> None:
    catalog_base.discover_tests()
    data_root = tmp_path / "data_quick"
    _write_calibration(data_root, _calibration())
    plan_path = tmp_path / "plan_quick.json"
    plan_path.write_text(
        json.dumps(
            {
                "participant_id": "sub-0001",
                "tests": [
                    {
                        "task_id": "contrast_sensitivity_function",
                        "eye": "OU",
                        "params": {"quick": True},
                        "viewing_distance_cm": 200.0,
                    }
                ],
                "ordering": "fixed",
                "seed": 7,
            }
        ),
        encoding="utf-8",
    )
    spec = "csf:peak_gain=1.6,peak_freq=3.0,bandwidth=3.0,low_freq_truncation=1.0"
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--session-plan",
            str(plan_path),
            "--status-file",
            str(tmp_path / "status_quick.json"),
            "--data-root",
            str(data_root),
            "--simulate",
            spec,
        ]
    )
    writer = _FakeWriter()
    exit_code = run_session(args, writer_factory=lambda *a: writer)
    assert exit_code == RunnerExitCode.OK
    summary = writer.summaries[0]
    assert summary.n_trials == 100
    assert any(f.code == "quick_csf_run" for f in summary.quality_flags)


@pytest.mark.slow
def test_recovery_slow_over_several_true_parameter_sets(tmp_path: Path) -> None:
    """Recovery within tolerance over several true CSF parameter sets, through
    the full test/runner pipeline (not just the bare QCSF procedure, which
    tests/procedures/test_qcsf.py already validates with much larger n)."""
    parameter_sets = [
        {
            "peak_gain_log10": 1.6,
            "peak_freq_cpd": 3.0,
            "bandwidth_octaves": 3.0,
            "low_freq_truncation_log10": 1.0,
        },
        {
            "peak_gain_log10": 1.2,
            "peak_freq_cpd": 1.5,
            "bandwidth_octaves": 2.5,
            "low_freq_truncation_log10": 0.5,
        },
        {
            "peak_gain_log10": 2.0,
            "peak_freq_cpd": 5.0,
            "bandwidth_octaves": 3.5,
            "low_freq_truncation_log10": 1.5,
        },
    ]
    n_reps = 3
    for i, params in enumerate(parameter_sets):
        true_aulcsf = _true_aulcsf(params)
        diffs = []
        for rep in range(n_reps):
            summary = _run_session_with_csf_observer(
                tmp_path, params, max_trials=200, seed=1000 * i + rep
            )
            diffs.append(summary.estimate.value - true_aulcsf)
        mean_diff = float(np.mean(diffs))
        assert abs(mean_diff) < 0.4, f"parameter set {i}: mean bias {mean_diff:.3f}"


# ---------------------------------------------------------------------------
# 5. summarize() on a fixture (+ reproducibility) and csf_curve()
# ---------------------------------------------------------------------------


def _trials_fixture(
    test: ContrastSensitivityFunctionTest, n_main: int = 40, n_catch: int = 4
) -> pd.DataFrame:
    obs = CSFObserver(n_afc=2, slope=DEFAULT_PSYCHOMETRIC_SLOPE, lapse_rate=0.02, **TRUE_PARAMS)
    procedure = test.make_procedure()
    rng = np.random.default_rng(9)
    rows = []
    for i in range(n_main):
        stim = procedure.next_stimulus()
        is_correct = obs.decide_correct(
            {"spatial_frequency_cpd": stim["spatial_frequency_cpd"], "contrast": stim["contrast"]},
            rng,
        )
        procedure.update(stim, is_correct)
        rows.append(
            {
                "block": "main",
                "is_catch": False,
                "trial_index": i,
                "intensity": stim["intensity"],
                "correct": is_correct,
                "eye": "OU",
                "n_dropped_frames_trial": 0,
                "stimulus_params": {
                    "spatial_frequency_cpd": stim["spatial_frequency_cpd"],
                    "contrast": stim["contrast"],
                },
            }
        )
    catch = test.make_catch_trial_intensity()
    for j in range(n_catch):
        rows.append(
            {
                "block": "main",
                "is_catch": True,
                "trial_index": n_main + j,
                "intensity": catch["intensity"],
                "correct": True,
                "eye": "OU",
                "n_dropped_frames_trial": 0,
                "stimulus_params": {
                    "spatial_frequency_cpd": catch["spatial_frequency_cpd"],
                    "contrast": catch["contrast"],
                },
            }
        )
    return pd.DataFrame(rows)


def test_summarize_on_fixture_is_deterministic_and_well_formed() -> None:
    test = _make_test(max_trials=40)
    df = _trials_fixture(test)
    summary1 = test.summarize(df)
    summary2 = test.summarize(df)
    assert summary1.model_dump() == summary2.model_dump()
    assert summary1.n_trials == 40
    assert summary1.n_catch == 4
    assert summary1.estimate.ci_low <= summary1.estimate.value <= summary1.estimate.ci_high
    assert "posterior_mean" in summary1.estimate.extra
    assert "log_cs_at_frequencies_cpd" in summary1.estimate.extra
    assert "frequency_range_tested_cpd" in summary1.estimate.extra
    assert "cutoff_spatial_frequency_cpd" in summary1.estimate.extra


def test_summarize_matches_fresh_test_instance_reanalysis() -> None:
    """Simulates vpsych reanalyze: a *fresh* test instance (same params/display/
    calibration, no shared in-memory state) must reproduce the identical summary."""
    test1 = _make_test(max_trials=40)
    df = _trials_fixture(test1)
    summary1 = test1.summarize(df)

    test2 = _make_test(max_trials=40)  # fresh instance, fresh rng
    summary2 = test2.summarize(df)
    assert summary1.model_dump() == summary2.model_dump()


def test_summarize_flags_high_catch_lapse_rate() -> None:
    test = _make_test(max_trials=40)
    df = _trials_fixture(test)
    df.loc[df["is_catch"], "correct"] = False
    summary = test.summarize(df)
    assert summary.catch_lapse_rate == pytest.approx(1.0)
    assert any(f.code == "high_catch_lapse_rate" for f in summary.quality_flags)


def test_csf_curve_is_reproducible_and_well_shaped() -> None:
    test = _make_test(max_trials=40)
    df = _trials_fixture(test)
    curve1 = test.csf_curve(df, n_points=20)
    curve2 = test.csf_curve(df, n_points=20)
    assert curve1 == curve2  # fixed local seed -> exact reproducibility
    assert len(curve1["spatial_frequency_cpd"]) == 20
    for _f, lo, mean, hi in zip(
        curve1["spatial_frequency_cpd"],
        curve1["log10_cs_ci_low"],
        curve1["log10_cs_mean"],
        curve1["log10_cs_ci_high"],
        strict=True,
    ):
        assert lo <= mean <= hi


# ---------------------------------------------------------------------------
# 6. Display smoke test (real PsychoPy window, fake keyboard)
# ---------------------------------------------------------------------------


class _FakeKeyboard:
    """Stands in for `psychopy.hardware.keyboard.Keyboard`: returns a canned key instantly."""

    def __init__(self, key: str) -> None:
        self._key = key
        self._returned = False

    def clearEvents(self) -> None:
        self._returned = False

    def waitKeys(
        self, maxWait: float | None = None, keyList: list[str] | None = None, timeStamped: bool = True
    ) -> list[tuple[str, float]]:
        return [(self._key, 0.05)]

    def getKeys(
        self, keyList: list[str] | None = None, timeStamped: bool = True
    ) -> list[tuple[str, float]]:
        if self._returned:
            return []
        self._returned = True
        return [(self._key, 0.05)]


@pytest.mark.display
def test_display_smoke_two_trials() -> None:
    """Opens a real PsychoPy window and presents 2 trials with a fake keyboard."""
    from psychopy import visual

    win = visual.Window(size=(400, 300), fullscr=False, allowGUI=False)
    try:
        test = _make_test(max_trials=2)
        test.build_stimuli(win)
        rng = np.random.default_rng(0)
        for trial_index in range(2):
            trial_ctx = {
                "timeline": test._timeline(),
                "rng": rng,
                "block": "main",
                "is_catch": False,
                "trial_index": trial_index,
                "keyboard": _FakeKeyboard("left"),
                "simulated_observer": None,
            }
            stim = test.make_catch_trial_intensity()
            presented = test.present(win, stim, trial_ctx)
            assert presented.response in test.response_keys()
            assert presented.stimulus_onset_s >= 0.0
    finally:
        win.close()
