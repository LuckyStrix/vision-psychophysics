"""Tests for vpsych.tests_catalog.motion_coherence: the RDK direction-discrimination test.

Follows the "required tests" pattern in `docs/WRITING_A_TEST.md`: a
requirements check, response-scoring correctness, simulated end-to-end
recovery via the real runner, `summarize()` on a fixture, and a
`@pytest.mark.display` smoke test. Pure dot-field-physics unit tests live in
`test_motion_coherence_dots.py`.
"""

from __future__ import annotations

import json
import math
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
from vpsych.core.procedures.questplus_procedure import questplus_weibull_x_at_p
from vpsych.core.psychometric import intensity_at_p_correct
from vpsych.runner.__main__ import build_arg_parser, parse_simulated_observer_spec, run_session
from vpsych.runner.status import RunnerExitCode
from vpsych.tests_catalog import base as catalog_base
from vpsych.tests_catalog.base import check_requirements
from vpsych.tests_catalog.motion_coherence import (
    DEFAULT_INTENSITY_VALUES,
    MIN_REFRESH_HZ,
    MotionCoherenceParams,
    MotionCoherenceTest,
)


def _display(refresh_hz: float = 60.0) -> DisplayGeometry:
    return DisplayGeometry(
        width_px=1920,
        height_px=1080,
        width_cm=53.13,
        height_cm=29.88,
        viewing_distance_cm=57.0,
        refresh_hz=refresh_hz,
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


# ---------------------------------------------------------------------------
# 1. Requirements check
# ---------------------------------------------------------------------------


def test_requirements_met_at_60hz() -> None:
    reasons = check_requirements(MotionCoherenceTest.spec.requirements, _display(60.0), None)
    assert reasons == []


def test_requirements_unmet_below_60hz() -> None:
    reasons = check_requirements(MotionCoherenceTest.spec.requirements, _display(30.0), None)
    assert any("refresh rate" in r for r in reasons)


def test_min_refresh_is_60hz() -> None:
    assert MIN_REFRESH_HZ == 60.0
    assert MotionCoherenceTest.spec.requirements.min_refresh_hz == 60.0


def test_is_registered_and_visible() -> None:
    catalog_base.discover_tests()
    assert catalog_base.get_test("motion_coherence") is MotionCoherenceTest
    assert MotionCoherenceTest.spec.hidden is False
    assert MotionCoherenceTest.spec.id in {t.spec.id for t in catalog_base.visible_tests()}


# ---------------------------------------------------------------------------
# 2. Score correctness
# ---------------------------------------------------------------------------


def test_score_correct_and_incorrect() -> None:
    test = MotionCoherenceTest(
        params=MotionCoherenceParams(),
        display=_display(),
        calibration=None,
        rng=np.random.default_rng(0),
    )
    stim = {"correct_response": "left"}
    assert test.score("left", stim) is True
    assert test.score("right", stim) is False
    assert test.score(None, stim) is False


# ---------------------------------------------------------------------------
# 3. Simulated end-to-end recovery via the runner
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


def test_simulated_end_to_end_recovery_via_runner(tmp_path: Path) -> None:
    catalog_base.discover_tests()
    data_root = tmp_path / "data"
    _write_calibration(data_root, _calibration())

    true_threshold_log10 = math.log10(0.30)  # 30% coherence
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "participant_id": "sub-0001",
                "tests": [
                    {
                        "task_id": "motion_coherence",
                        "eye": "OU",
                        "params": {"max_trials": 80},
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
            f"psychometric:threshold={true_threshold_log10},slope=0.3,lapse=0.02",
        ]
    )
    writer = _FakeWriter()
    exit_code = run_session(args, writer_factory=lambda *a: writer)
    assert exit_code == RunnerExitCode.OK
    assert len(writer.summaries) == 1

    summary = writer.summaries[0]
    observer = parse_simulated_observer_spec(
        f"psychometric:threshold={true_threshold_log10},slope=0.3,lapse=0.02"
    )
    target_log10 = intensity_at_p_correct(observer.true_function, 0.75)
    achieved_log10 = math.log10(summary.estimate.value / 100.0)
    # Generous tolerance in log10 units (a percent-scale comparison is
    # dominated by the nonlinear log->linear mapping near the domain's high
    # end): smoke-level check, not the properly-powered bias/coverage
    # validation (that lives in tests/procedures).
    assert abs(achieved_log10 - target_log10) < 0.5
    assert summary.estimate.units == "coherence_percent"


# ---------------------------------------------------------------------------
# 4. summarize() on a fixture
# ---------------------------------------------------------------------------


def _trials_fixture(catch_correct: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    rows = []
    intensity = DEFAULT_INTENSITY_VALUES[12]  # an exact QUEST+ grid point, mid-domain
    for i in range(40):
        correct = bool(rng.random() < 0.8)
        rows.append(
            {
                "block": "main",
                "is_catch": False,
                "trial_index": i,
                "intensity": intensity,
                "correct": correct,
                "eye": "OU",
                "n_dropped_frames_trial": 0,
            }
        )
    for i in range(4):
        rows.append(
            {
                "block": "main",
                "is_catch": True,
                "trial_index": 40 + i,
                "intensity": 0.0,
                "correct": catch_correct,
                "eye": "OU",
                "n_dropped_frames_trial": 0,
            }
        )
    return pd.DataFrame(rows)


def test_summarize_on_fixture_is_deterministic_and_well_formed() -> None:
    test = MotionCoherenceTest(
        params=MotionCoherenceParams(max_trials=40),
        display=_display(),
        calibration=None,
        rng=np.random.default_rng(0),
    )
    df = _trials_fixture()
    summary1 = test.summarize(df)
    summary2 = test.summarize(df)
    assert summary1.model_dump() == summary2.model_dump()
    assert summary1.n_trials == 40
    assert summary1.n_catch == 4
    assert summary1.catch_lapse_rate == pytest.approx(0.0)
    assert summary1.estimate.ci_low <= summary1.estimate.value <= summary1.estimate.ci_high
    assert summary1.estimate.units == "coherence_percent"
    # Coherence percent must stay within the tested [1, 100] range.
    assert 0.0 < summary1.estimate.value <= 100.0


def test_summarize_flags_high_catch_lapse_rate() -> None:
    test = MotionCoherenceTest(
        params=MotionCoherenceParams(max_trials=40),
        display=_display(),
        calibration=None,
        rng=np.random.default_rng(0),
    )
    df = _trials_fixture(catch_correct=False)
    summary = test.summarize(df)
    assert summary.catch_lapse_rate == pytest.approx(1.0)
    assert any(f.code == "high_catch_lapse_rate" for f in summary.quality_flags)


def test_summarize_flags_excess_dropped_frames() -> None:
    test = MotionCoherenceTest(
        params=MotionCoherenceParams(max_trials=40),
        display=_display(),
        calibration=None,
        rng=np.random.default_rng(0),
    )
    df = _trials_fixture()
    df.loc[df.index[0], "n_dropped_frames_trial"] = len(df)  # force >1% dropped
    summary = test.summarize(df)
    assert any(f.code == "excess_dropped_frames" for f in summary.quality_flags)


def test_reanalysis_matches_75pct_correct_conversion() -> None:
    """The reported estimate should be the 75%-correct point on questplus's OWN fitted
    Weibull (see QuestPlusProcedure's "Criterion conversion pitfall" docstring), not
    vpsych.core.psychometric's incompatible F(0)=0.5-anchored family -- using the latter
    to invert a QuestPlusProcedure fit is exactly the item-1 bug this test guards against.
    """
    test = MotionCoherenceTest(
        params=MotionCoherenceParams(max_trials=40),
        display=_display(),
        calibration=None,
        rng=np.random.default_rng(0),
    )
    df = _trials_fixture()
    summary = test.summarize(df)
    raw_threshold_log10 = summary.estimate.extra["raw_questplus_native_threshold_log10_coherence"]
    slope = summary.estimate.extra["slope"]
    lapse = summary.estimate.extra["lapse_rate"]
    expected_log10 = questplus_weibull_x_at_p(raw_threshold_log10, slope, 0.5, lapse, 0.75)
    expected_percent = 10.0**expected_log10 * 100.0
    assert summary.estimate.value == pytest.approx(expected_percent, rel=1e-9)


# ---------------------------------------------------------------------------
# Pure-function sanity: domain bounds
# ---------------------------------------------------------------------------


def test_intensity_domain_is_1_to_100_percent_coherence() -> None:
    assert min(DEFAULT_INTENSITY_VALUES) == pytest.approx(math.log10(0.01))
    assert max(DEFAULT_INTENSITY_VALUES) == pytest.approx(math.log10(1.0))


# ---------------------------------------------------------------------------
# Slow: bias/coverage of this test's own summarize() pipeline
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("true_log10", [-1.5, -1.0, -0.5])
def test_summarize_bias_and_coverage_over_many_simulated_runs(true_log10: float) -> None:
    """Bias/coverage of *this test's* 75%-correct conversion + percent reporting.

    Ground truth is defined directly in `questplus`'s OWN Weibull
    parameterization (see `QuestPlusProcedure`'s "Criterion conversion
    pitfall" docstring and `visual_acuity`/`vernier_acuity`'s
    identically-structured validation tests) -- **not**
    `vpsych.core.psychometric.PsychometricObserver`, which uses a different,
    incompatible sigmoid family. Using the `vpsych.core.psychometric` family
    as ground truth here, as an earlier version of this test did, conflates
    two independent effects: (1) the item-1 criterion-conversion bug (fixed:
    `summarize()` now inverts `questplus`'s own fitted curve via
    `QuestPlusProcedure.intensity_at_p_correct` rather than
    `vpsych.core.psychometric.intensity_at_p_correct`), and (2) an
    unrelated family-mismatch artifact from generating data under one
    sigmoid family while fitting another. Native-family ground truth
    isolates (1), which is what this test is actually meant to check.

    Not a re-validation of QUEST+ itself (that belongs to
    tests/procedures/test_questplus_procedure.py, per docs/WRITING_A_TEST.md
    section 11) -- this drives many independent simulated QUEST+ runs
    directly (bypassing the trial loop/window for speed).

    Measured empirically (N=40/true value, 50 trials/run, run with
    `OPENBLAS_NUM_THREADS=1`) after the item-1 fix: mean bias ranges from
    about +0.42 log10-coherence units at true_log10=-1.5 down to about
    -0.11 at true_log10=-0.5 (CI coverage 0.93-0.98, close to nominal
    0.95) -- see docs/methods/motion_coherence.md "Validation" for the full
    before/after table and discussion. The coverage fix (from the
    criterion-conversion bug) is real and substantial; the remaining
    point-estimate bias reflects this test's default shallow slope grid
    (`DEFAULT_SLOPE_VALUES`, centered on 0.3) combined with a modest
    50-trial budget amplifying slope-estimation noise when extrapolating
    from questplus's own ~80%-of-range native anchor (guess=0.5) out to the
    75% conventional criterion -- a genuine statistical/design property of
    this procedure's defaults, not a parameterization bug. This test guards
    against a gross regression, not a tight calibration target.
    """
    display = _display()
    n_runs = 40
    slope_true = 0.3
    lapse_true = 0.02
    guess = 0.5

    def p_correct(x: float) -> float:
        return (
            1.0
            - lapse_true
            - (1.0 - guess - lapse_true) * math.exp(-(10.0 ** (slope_true * (x - true_log10))))
        )

    biases = []
    coverages = []
    for seed in range(n_runs):
        rng = np.random.default_rng(4000 + seed)
        test = MotionCoherenceTest(
            params=MotionCoherenceParams(max_trials=50),
            display=display,
            calibration=None,
            rng=np.random.default_rng(0),
        )
        procedure = test.make_procedure()
        while not procedure.finished:
            x = procedure.next_intensity()
            correct = bool(rng.random() < p_correct(x))
            procedure.update(x, correct)
        reported, ci_low, ci_high = procedure.intensity_at_p_correct(0.75)
        true_at_75 = questplus_weibull_x_at_p(true_log10, slope_true, guess, lapse_true, 0.75)
        biases.append(reported - true_at_75)
        coverages.append(ci_low <= true_at_75 <= ci_high)

    mean_bias = float(np.mean(biases))
    coverage_rate = float(np.mean(coverages))
    # Generous bounds around the measured values above (small-N Monte Carlo
    # noise on both bias and coverage): this guards against a gross
    # regression in the conversion pipeline, not a tight calibration check.
    assert abs(mean_bias) < 0.6, f"mean bias {mean_bias:.3f} log10 units is too large"
    assert coverage_rate >= 0.75, f"empirical coverage {coverage_rate:.2f} far below nominal 0.95"


# ---------------------------------------------------------------------------
# Display smoke test
# ---------------------------------------------------------------------------


@pytest.mark.display
def test_display_smoke_two_trials() -> None:
    """Runs 2 real trials through a real PsychoPy window with a fake keyboard."""
    from psychopy import visual

    from vpsych.core.timing import measure_refresh

    win = visual.Window(size=(800, 600), fullscr=False, allowGUI=False, units="pix")
    try:
        measured_hz = measure_refresh(win, n_frames=30)
        display = DisplayGeometry(
            width_px=800,
            height_px=600,
            width_cm=30.0,
            height_cm=22.5,
            viewing_distance_cm=57.0,
            refresh_hz=measured_hz,
        )
        test = MotionCoherenceTest(
            params=MotionCoherenceParams(max_trials=2, duration_ms=150.0),
            display=display,
            calibration=None,
            rng=np.random.default_rng(0),
        )
        test.build_stimuli(win)

        class _FakeKeyboard:
            # Method/argument names match psychopy.hardware.keyboard.Keyboard's
            # real API exactly (present() calls these directly), not this
            # project's own naming convention.
            class clock:  # noqa: N801
                @staticmethod
                def reset() -> None:
                    pass

            def clearEvents(self) -> None:  # noqa: N802
                pass

            def waitKeys(  # noqa: N802
                self,
                keyList: list[str],  # noqa: N803
                waitRelease: bool,  # noqa: N803
                clear: bool,
            ) -> list[Any]:
                class _Key:
                    name = keyList[0]
                    rt = 0.3

                return [_Key()]

        from vpsych.core.trial import TrialTimeline

        timeline = TrialTimeline(
            fixation_frames=5, stimulus_frames=display.frames_for_ms(150.0), iti_frames=5
        )
        for _ in range(2):
            trial_ctx: dict[str, Any] = {
                "timeline": timeline,
                "rng": np.random.default_rng(1),
                "block": "main",
                "is_catch": False,
                "trial_index": 0,
                "keyboard": _FakeKeyboard(),
                "simulated_observer": None,
            }
            presented = test.present(win, 0.0, trial_ctx)
            assert presented.response in ("left", "right")
        print(f"motion_coherence display smoke: measured refresh = {measured_hz:.2f} Hz")
    finally:
        win.close()
