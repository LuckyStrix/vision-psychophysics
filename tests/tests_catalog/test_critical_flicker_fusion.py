"""Tests for vpsych.tests_catalog.critical_flicker_fusion.

Follows the "required tests" pattern in `docs/WRITING_A_TEST.md`: a
requirements check (including that a 60 Hz display is rejected), response-
scoring correctness, simulated end-to-end recovery via the real runner
(including the display-limited lower-bound case, with the simulated
observer expressed in the transformed intensity), `summarize()` on a
fixture, and a `@pytest.mark.display` smoke test. Pure waveform-generator
unit tests live in `test_critical_flicker_fusion_waveform.py`.
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
from vpsych.core.psychometric import PsychometricFunction, intensity_at_p_correct
from vpsych.runner.__main__ import build_arg_parser, parse_simulated_observer_spec, run_session
from vpsych.runner.status import RunnerExitCode
from vpsych.tests_catalog import base as catalog_base
from vpsych.tests_catalog.base import check_requirements
from vpsych.tests_catalog.critical_flicker_fusion import (
    MIN_REFRESH_HZ,
    USABLE_CEILING_DIVISOR,
    CFFParams,
    CriticalFlickerFusionTest,
)
from vpsych.tests_catalog.critical_flicker_fusion.waveform import (
    frequency_to_intensity,
    intensity_to_frequency,
)


def _display(refresh_hz: float = 120.0) -> DisplayGeometry:
    return DisplayGeometry(
        width_px=1920,
        height_px=1080,
        width_cm=53.13,
        height_cm=29.88,
        viewing_distance_cm=57.0,
        refresh_hz=refresh_hz,
    )


def _calibration(method: str = "photometer") -> Calibration:
    return Calibration(
        created_utc=datetime(2026, 9, 1, tzinfo=timezone.utc),
        geometry=_display(),
        gamma=GammaCalibration(
            method=method,  # type: ignore[arg-type]
            gamma_single=2.2,
            lum_min_cdm2=0.3,
            lum_max_cdm2=120.0,
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


def test_requirements_met_at_120hz_grade_a() -> None:
    reasons = check_requirements(
        CriticalFlickerFusionTest.spec.requirements, _display(120.0), _calibration("photometer")
    )
    assert reasons == []


def test_requirements_reject_60hz_display() -> None:
    reasons = check_requirements(
        CriticalFlickerFusionTest.spec.requirements, _display(60.0), _calibration("photometer")
    )
    assert any("refresh rate" in r for r in reasons)


def test_min_refresh_is_120hz() -> None:
    assert MIN_REFRESH_HZ == 120.0
    assert CriticalFlickerFusionTest.spec.requirements.min_refresh_hz == 120.0


def test_requirements_reject_uncalibrated_display() -> None:
    reasons = check_requirements(CriticalFlickerFusionTest.spec.requirements, _display(120.0), None)
    assert any("gamma calibration" in r for r in reasons)


def test_requirements_reject_grade_c_calibration() -> None:
    reasons = check_requirements(
        CriticalFlickerFusionTest.spec.requirements, _display(120.0), _calibration("none")
    )
    assert any("luminance grade" in r for r in reasons)


def test_requirements_accept_grade_b_calibration() -> None:
    reasons = check_requirements(
        CriticalFlickerFusionTest.spec.requirements, _display(120.0), _calibration("psychophysical")
    )
    assert reasons == []


def test_is_registered_and_visible() -> None:
    catalog_base.discover_tests()
    assert catalog_base.get_test("critical_flicker_fusion") is CriticalFlickerFusionTest
    assert CriticalFlickerFusionTest.spec.hidden is False
    assert CriticalFlickerFusionTest.spec.id in {t.spec.id for t in catalog_base.visible_tests()}


# ---------------------------------------------------------------------------
# 2. Score correctness
# ---------------------------------------------------------------------------


def test_score_correct_and_incorrect() -> None:
    test = CriticalFlickerFusionTest(
        params=CFFParams(),
        display=_display(),
        calibration=_calibration(),
        rng=np.random.default_rng(0),
    )
    stim = {"correct_response": "left"}
    assert test.score("left", stim) is True
    assert test.score("right", stim) is False
    assert test.score(None, stim) is False


def test_usable_ceiling_hz() -> None:
    test = CriticalFlickerFusionTest(
        params=CFFParams(),
        display=_display(120.0),
        calibration=_calibration(),
        rng=np.random.default_rng(0),
    )
    assert test.usable_ceiling_hz() == pytest.approx(120.0 / USABLE_CEILING_DIVISOR)


def test_intensity_values_span_matches_domain() -> None:
    test = CriticalFlickerFusionTest(
        params=CFFParams(min_frequency_hz=2.0),
        display=_display(120.0),
        calibration=_calibration(),
        rng=np.random.default_rng(0),
    )
    values = test._intensity_values()
    assert min(values) == pytest.approx(frequency_to_intensity(test.usable_ceiling_hz()))
    assert max(values) == pytest.approx(frequency_to_intensity(2.0))


def test_catch_trial_intensity_is_easiest() -> None:
    test = CriticalFlickerFusionTest(
        params=CFFParams(),
        display=_display(120.0),
        calibration=_calibration(),
        rng=np.random.default_rng(0),
    )
    assert test.make_catch_trial_intensity() == max(test._intensity_values())


def test_square_wave_mode_grid_is_all_valid_frequencies() -> None:
    from vpsych.tests_catalog.critical_flicker_fusion.waveform import (
        valid_square_wave_frequency,
    )

    test = CriticalFlickerFusionTest(
        params=CFFParams(waveform_mode="square_wave"),
        display=_display(120.0),
        calibration=_calibration(),
        rng=np.random.default_rng(0),
    )
    values = test._intensity_values()
    for x in values:
        f = intensity_to_frequency(x)
        assert valid_square_wave_frequency(f, 120.0)


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


def _run_session(tmp_path: Path, simulate_spec: str, max_trials: int = 80) -> Any:
    catalog_base.discover_tests()
    data_root = tmp_path / "data"
    _write_calibration(data_root, _calibration())

    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "participant_id": "sub-0001",
                "tests": [
                    {
                        "task_id": "critical_flicker_fusion",
                        "eye": "OU",
                        "params": {"max_trials": max_trials},
                        "viewing_distance_cm": 57.0,
                    }
                ],
                "ordering": "fixed",
                "seed": 11,
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
    assert exit_code == RunnerExitCode.OK
    assert len(writer.summaries) == 1
    return writer.summaries[0]


def test_simulated_end_to_end_recovery_via_runner(tmp_path: Path) -> None:
    """The simulated observer's true threshold is expressed in the transformed intensity
    x = -log10(frequency_hz), per the task requirement."""
    true_x = -1.0  # corresponds to a "true CFF-ish" region around 10 Hz
    spec = f"psychometric:threshold={true_x},slope=0.25,lapse=0.02"
    summary = _run_session(tmp_path, spec)

    observer = parse_simulated_observer_spec(spec)
    target_x = intensity_at_p_correct(observer.true_function, 0.75)
    target_hz = intensity_to_frequency(target_x)

    assert summary.estimate.units == "hz"
    if not any(f.code == "cff_display_limited" for f in summary.quality_flags):
        # Generous smoke-level tolerance (few trials); log-frequency units.
        import math

        assert abs(math.log10(summary.estimate.value) - math.log10(target_hz)) < 0.5


def test_simulated_end_to_end_display_limited_case(tmp_path: Path) -> None:
    """An observer who is essentially always correct, even at the hardest tested frequency,
    should yield a display-limited lower bound, not a fabricated point estimate.

    Needs a larger trial budget than a typical smoke test: QUEST+'s
    min-entropy stimulus selection resolves the true domain edge only very
    slowly for a near-ceiling-everywhere observer (see `summarize`'s
    docstring in `vpsych.tests_catalog.critical_flicker_fusion` and
    `docs/methods/critical_flicker_fusion.md`), so a modest 60-trial budget
    is not enough to reliably trigger the display-limited path.
    """
    # A true threshold far below the tested domain (very high, easily seen
    # frequency) with a shallow slope: near-ceiling performance everywhere.
    spec = "psychometric:threshold=-3.0,slope=0.05,lapse=0.0"
    summary = _run_session(tmp_path, spec, max_trials=300)

    assert any(f.code == "cff_display_limited" for f in summary.quality_flags)
    flag = next(f for f in summary.quality_flags if f.code == "cff_display_limited")
    assert flag.severity == "critical"
    assert summary.estimate.extra.get("display_limited") is True
    # value/ci pinned to the usable ceiling, not an extrapolated fabrication.
    assert summary.estimate.value == pytest.approx(summary.estimate.ci_low)
    assert summary.estimate.value == pytest.approx(summary.estimate.ci_high)


# ---------------------------------------------------------------------------
# 4. summarize() on a fixture
# ---------------------------------------------------------------------------


def _base_stimulus_params(refresh_hz: float, ceiling_hz: float) -> dict[str, Any]:
    return {
        "refresh_hz": refresh_hz,
        "usable_ceiling_hz": ceiling_hz,
        "amplitude_attenuated": False,
    }


def _trials_fixture(test: CriticalFlickerFusionTest, catch_correct: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(9)
    values = test._intensity_values()
    intensity = values[len(values) // 2]
    refresh_hz = test.display.refresh_hz
    ceiling_hz = test.usable_ceiling_hz()
    rows = []
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
                "stimulus_params": _base_stimulus_params(refresh_hz, ceiling_hz),
            }
        )
    for i in range(4):
        rows.append(
            {
                "block": "main",
                "is_catch": True,
                "trial_index": 40 + i,
                "intensity": max(values),
                "correct": catch_correct,
                "eye": "OU",
                "n_dropped_frames_trial": 0,
                "stimulus_params": _base_stimulus_params(refresh_hz, ceiling_hz),
            }
        )
    return pd.DataFrame(rows)


def test_summarize_on_fixture_is_deterministic_and_well_formed() -> None:
    test = CriticalFlickerFusionTest(
        params=CFFParams(max_trials=40),
        display=_display(120.0),
        calibration=_calibration(),
        rng=np.random.default_rng(0),
    )
    df = _trials_fixture(test)
    summary1 = test.summarize(df)
    summary2 = test.summarize(df)
    assert summary1.model_dump() == summary2.model_dump()
    assert summary1.n_trials == 40
    assert summary1.n_catch == 4
    assert summary1.catch_lapse_rate == pytest.approx(0.0)
    assert summary1.estimate.units == "hz"
    assert any(f.code == "lcd_response_time_limitation" for f in summary1.quality_flags)


def test_summarize_flags_high_catch_lapse_rate() -> None:
    test = CriticalFlickerFusionTest(
        params=CFFParams(max_trials=40),
        display=_display(120.0),
        calibration=_calibration(),
        rng=np.random.default_rng(0),
    )
    df = _trials_fixture(test, catch_correct=False)
    summary = test.summarize(df)
    assert summary.catch_lapse_rate == pytest.approx(1.0)
    assert any(f.code == "high_catch_lapse_rate" for f in summary.quality_flags)


def test_summarize_display_limited_case_on_fixture() -> None:
    """All trials correct even at the hardest (highest-frequency) grid point -> lower bound.

    Uses 200 trials, not the usual 40: see `test_simulated_end_to_end_display_limited_case`
    and `docs/methods/critical_flicker_fusion.md` for why a modest trial count isn't enough
    to resolve this case (QUEST+'s posterior converges to the domain edge only slowly for a
    near-ceiling-everywhere observer).
    """
    test = CriticalFlickerFusionTest(
        params=CFFParams(max_trials=200),
        display=_display(120.0),
        calibration=_calibration(),
        rng=np.random.default_rng(0),
    )
    values = test._intensity_values()
    hardest = min(values)  # highest frequency = usable ceiling
    refresh_hz = test.display.refresh_hz
    ceiling_hz = test.usable_ceiling_hz()
    rows = [
        {
            "block": "main",
            "is_catch": False,
            "trial_index": i,
            "intensity": hardest,
            "correct": True,
            "eye": "OU",
            "n_dropped_frames_trial": 0,
            "stimulus_params": _base_stimulus_params(refresh_hz, ceiling_hz),
        }
        for i in range(200)
    ]
    df = pd.DataFrame(rows)
    summary = test.summarize(df)
    assert any(f.code == "cff_display_limited" for f in summary.quality_flags)
    assert summary.estimate.value == pytest.approx(ceiling_hz)
    assert summary.estimate.extra["display_limited"] is True


def test_reanalysis_matches_75pct_correct_conversion() -> None:
    test = CriticalFlickerFusionTest(
        params=CFFParams(max_trials=40),
        display=_display(120.0),
        calibration=_calibration(),
        rng=np.random.default_rng(0),
    )
    df = _trials_fixture(test)
    summary = test.summarize(df)
    if summary.estimate.extra.get("display_limited"):
        pytest.skip("fixture landed in the display-limited regime; covered separately")
    raw_threshold_x = summary.estimate.extra["threshold_f0.5_neg_log10_hz"]
    slope = summary.estimate.extra["slope"]
    lapse = summary.estimate.extra["lapse_rate"]
    fn = PsychometricFunction(
        family="weibull",
        threshold=raw_threshold_x,
        slope=slope,
        guess=0.5,
        lapse=lapse,
        intensity_scale="log10",
    )
    expected_x = intensity_at_p_correct(fn, 0.75)
    expected_hz = intensity_to_frequency(expected_x)
    assert summary.estimate.value == pytest.approx(expected_hz, rel=1e-9)


# ---------------------------------------------------------------------------
# Display smoke test
# ---------------------------------------------------------------------------


@pytest.mark.display
def test_display_smoke_two_trials() -> None:
    """Runs 2 real trials through a real PsychoPy window with a fake keyboard."""
    from psychopy import core as psychopy_core
    from psychopy import visual

    from vpsych.core.timing import measure_refresh
    from vpsych.core.trial import TrialTimeline

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
        test = CriticalFlickerFusionTest(
            params=CFFParams(max_trials=2, duration_ms=150.0, onset_ramp_ms=0.0),
            display=display,
            calibration=_calibration(),
            rng=np.random.default_rng(0),
        )
        test.build_stimuli(win)

        class _FakeKeyboard:
            # Method/argument names match psychopy.hardware.keyboard.Keyboard's
            # real API exactly (present() calls these directly), not this
            # project's own naming convention.
            def clearEvents(self) -> None:  # noqa: N802
                pass

            def waitKeys(  # noqa: N802
                self,
                keyList: list[str],  # noqa: N803
                timeStamped: bool,  # noqa: N803
            ) -> list[tuple[str, float]]:
                return [(keyList[0], psychopy_core.getTime())]

        timeline = TrialTimeline(
            fixation_frames=5, stimulus_frames=display.frames_for_ms(150.0), iti_frames=5
        )
        intensity = test.make_catch_trial_intensity()
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
            presented = test.present(win, intensity, trial_ctx)
            assert presented.response in ("left", "right")
        print(f"critical_flicker_fusion display smoke: measured refresh = {measured_hz:.2f} Hz")
    finally:
        win.close()
