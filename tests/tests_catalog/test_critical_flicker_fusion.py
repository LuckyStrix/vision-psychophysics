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


def _display(refresh_hz: float = 240.0) -> DisplayGeometry:
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


def test_requirements_met_at_240hz_grade_a() -> None:
    reasons = check_requirements(
        CriticalFlickerFusionTest.spec.requirements, _display(240.0), _calibration("photometer")
    )
    assert reasons == []


def test_requirements_reject_60hz_display() -> None:
    reasons = check_requirements(
        CriticalFlickerFusionTest.spec.requirements, _display(60.0), _calibration("photometer")
    )
    assert any("refresh rate" in r for r in reasons)


def test_requirements_reject_120hz_display() -> None:
    """120 Hz's continuous-mode usable ceiling (40 Hz) is below the realistic human CFF
    range (~30-60 Hz) -- this test's own simulated validation shows most runs land in the
    display-limited regime even at true thresholds well under the nominal 40 Hz ceiling
    (see docs/methods/critical_flicker_fusion.md "Requirements"/"Validation"), so 120 Hz
    is rejected outright rather than silently returning an unreliable estimate."""
    reasons = check_requirements(
        CriticalFlickerFusionTest.spec.requirements, _display(120.0), _calibration("photometer")
    )
    assert any("refresh rate" in r for r in reasons)


def test_min_refresh_is_240hz() -> None:
    assert MIN_REFRESH_HZ == 240.0
    assert CriticalFlickerFusionTest.spec.requirements.min_refresh_hz == 240.0


def test_requirements_reject_uncalibrated_display() -> None:
    reasons = check_requirements(CriticalFlickerFusionTest.spec.requirements, _display(240.0), None)
    assert any("gamma calibration" in r for r in reasons)


def test_requirements_reject_grade_c_calibration() -> None:
    reasons = check_requirements(
        CriticalFlickerFusionTest.spec.requirements, _display(240.0), _calibration("none")
    )
    assert any("luminance grade" in r for r in reasons)


def test_requirements_accept_grade_b_calibration() -> None:
    reasons = check_requirements(
        CriticalFlickerFusionTest.spec.requirements, _display(240.0), _calibration("psychophysical")
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
    # slope=1.5 is within the new DEFAULT_SLOPE_VALUES grid (0.5-6.0) -- realistic for a
    # near-threshold 2AFC psychometric function; the old slope=0.25 ground truth here
    # predates the slope-grid fix and was itself an unrealistically shallow value.
    spec = f"psychometric:threshold={true_x},slope=1.5,lapse=0.02"
    summary = _run_session(tmp_path, spec)

    observer = parse_simulated_observer_spec(spec)
    target_x = intensity_at_p_correct(observer.true_function, 0.75)
    target_hz = intensity_to_frequency(target_x)

    assert summary.estimate.units == "hz"
    if not any(f.code == "cff_display_limited" for f in summary.quality_flags):
        # Generous smoke-level tolerance (few trials); log-frequency units.
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
    """The reported estimate should be the 75%-correct point on questplus's OWN fitted
    Weibull (see QuestPlusProcedure's "Criterion conversion pitfall" docstring), not
    vpsych.core.psychometric's incompatible F(0)=0.5-anchored family -- using the latter
    to invert a QuestPlusProcedure fit is exactly the item-1 bug this test guards against.
    """
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
    raw_threshold_x = summary.estimate.extra["raw_questplus_native_threshold_neg_log10_hz"]
    slope = summary.estimate.extra["slope"]
    lapse = summary.estimate.extra["lapse_rate"]
    expected_x = questplus_weibull_x_at_p(raw_threshold_x, slope, 0.5, lapse, 0.75)
    expected_hz = intensity_to_frequency(expected_x)
    assert summary.estimate.value == pytest.approx(expected_hz, rel=1e-9)


# ---------------------------------------------------------------------------
# Slow: bias/coverage of this test's own summarize() pipeline
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize(
    "true_freq_hz,slope_true,max_abs_bias,min_coverage,max_display_limited_rate",
    [
        # Realistic thresholds (15-45 Hz, not the old 3-5 Hz) x several true slopes
        # spanning the plausible range, INCLUDING values (1.5, 3.0) that were entirely
        # outside the old domain-derived grid (~0.03-0.8 at this refresh) -- see
        # DEFAULT_SLOPE_VALUES's docstring. At refresh=240 Hz (ceiling 80 Hz) these true
        # frequencies sit well clear of the display-limited edge band (edges near ~46 Hz;
        # see docs/methods/critical_flicker_fusion.md "Display-limited detection").
        (15.0, 1.5, 0.3, 0.7, 0.5),
        (20.0, 1.5, 0.3, 0.7, 0.5),
        (30.0, 1.5, 0.3, 0.7, 0.5),
        (20.0, 3.0, 0.3, 0.7, 0.5),
        # A genuinely shallow (beta=0.6) observer: documents the honest, NOT-fully-fixed
        # residual limitation -- the 75%-criterion conversion divides the fitted-slope
        # offset by the slope itself, so a shallow true slope amplifies both bias and
        # display-limited rate even with a grid that comfortably brackets it. This case
        # is asserted with looser bounds, not silently dropped -- see
        # docs/methods/critical_flicker_fusion.md "Validation" for the measured table.
        (20.0, 0.6, 0.5, 0.7, 0.5),
    ],
)
def test_summarize_bias_and_coverage_over_many_simulated_runs(
    true_freq_hz: float,
    slope_true: float,
    max_abs_bias: float,
    min_coverage: float,
    max_display_limited_rate: float,
) -> None:
    """Bias/coverage/display-limited-rate of *this test's* summarize() pipeline.

    Ground truth is defined directly in `questplus`'s OWN Weibull
    parameterization (see `QuestPlusProcedure`'s "Criterion conversion
    pitfall" docstring and `visual_acuity`/`vernier_acuity`'s
    identically-structured validation tests) -- **not**
    `vpsych.core.psychometric.PsychometricObserver`, which uses a
    different, incompatible sigmoid family (that conflates the criterion-
    conversion fix with an unrelated family-mismatch artifact; see git
    history / docs/methods/critical_flicker_fusion.md for the earlier,
    confounded version of this test).

    Not a re-validation of QUEST+ itself (that belongs to
    tests/procedures/test_questplus_procedure.py, per docs/WRITING_A_TEST.md
    section 11) -- drives many independent simulated QUEST+ runs directly
    through the real `CriticalFlickerFusionTest.make_procedure()`/
    `summarize()` pipeline (bypassing the trial loop/window for speed).

    **Phase 4 slope-grid fix.** An earlier version of `make_procedure()`
    derived `slope_values` from the *width of the tested domain*
    (`linspace(span*0.02, span*0.5, 6)`), giving slopes of roughly 0.03-0.8
    at this refresh -- values far shallower than any realistic human
    psychometric-function slope (see `DEFAULT_SLOPE_VALUES`'s docstring).
    Combined with this test's *own* earlier validation always simulating
    `slope_true=0.25` (inside that narrow, accidental grid) and silently
    `continue`-ing past every display-limited run without reporting the
    rate, the published bias/coverage figures were both parameter-matched
    to pass and selection-biased. Measured with the *old* buggy grid at
    realistic slopes (ad hoc, not itself checked in): true_f=20 Hz,
    slope_true=1.5 -> **100% of runs display-limited** (a spurious verdict:
    an artificially shallow slope fit pushes the 75%-point toward the
    domain's hard edge regardless of the true threshold -- see
    `DEFAULT_SLOPE_VALUES`'s docstring for the mechanism).

    Measured with the fixed grid (`DEFAULT_SLOPE_VALUES`, N=20/case, 50
    trials/run, refresh=240 Hz, run with `OPENBLAS_NUM_THREADS=1`) -- see
    docs/methods/critical_flicker_fusion.md "Validation" for the full
    table:

    - true_f=15 Hz, slope=1.5: bias +0.042, coverage 0.95, 0/20 display-limited
    - true_f=20 Hz, slope=1.5: bias +0.065, coverage 1.00, 0/20 display-limited
    - true_f=30 Hz, slope=1.5: bias +0.044, coverage 1.00, 1/20 display-limited
    - true_f=20 Hz, slope=3.0: bias -0.010, coverage 0.90, 0/20 display-limited
    - true_f=20 Hz, slope=0.6: bias +0.225, coverage 0.94, 3/20 display-limited (the
      documented, still-degraded-but-no-longer-catastrophic shallow-slope regime)

    This test guards against a gross regression (e.g. the grid becoming
    domain-derived again, or the display-limited rate silently exploding),
    not a tight calibration target -- bounds below are generous margins
    around the measured values, chosen per-case (looser for the known-
    degraded shallow-slope case) rather than one blanket threshold.
    """
    display = _display(240.0)
    cal = _calibration()
    n_runs = 20
    lapse_true = 0.02
    guess = 0.5
    true_x = frequency_to_intensity(true_freq_hz)

    def p_correct(x: float) -> float:
        return (
            1.0
            - lapse_true
            - (1.0 - guess - lapse_true) * math.exp(-(10.0 ** (slope_true * (x - true_x))))
        )

    biases = []
    coverages = []
    n_display_limited = 0
    for seed in range(n_runs):
        rng = np.random.default_rng(6000 + seed)
        test = CriticalFlickerFusionTest(
            params=CFFParams(max_trials=50),
            display=display,
            calibration=cal,
            rng=np.random.default_rng(0),
        )
        procedure = test.make_procedure()
        rows = []
        i = 0
        while not procedure.finished:
            x = procedure.next_intensity()
            correct = bool(rng.random() < p_correct(x))
            procedure.update(x, correct)
            rows.append(
                {
                    "block": "main",
                    "is_catch": False,
                    "trial_index": i,
                    "intensity": x,
                    "correct": correct,
                    "eye": "OU",
                    "n_dropped_frames_trial": 0,
                    "stimulus_params": {"amplitude_attenuated": False},
                }
            )
            i += 1
        summary = test.summarize(pd.DataFrame(rows))
        # Display-limited runs are ALWAYS counted toward n_display_limited (and thus
        # the reported rate below) -- they are excluded only from the Hz-bias/coverage
        # statistics themselves, because a display-limited run reports a censored lower
        # bound, not a point estimate comparable to the true 75%-point in Hz. This is
        # the honest accounting the old version lacked (it discarded these runs from
        # the denominator entirely, with no rate ever asserted or reported).
        if summary.estimate.extra.get("display_limited"):
            n_display_limited += 1
            continue
        est_x = frequency_to_intensity(summary.estimate.value)
        ci_low_x = frequency_to_intensity(summary.estimate.ci_low)
        ci_high_x = frequency_to_intensity(summary.estimate.ci_high)
        true_at_75 = questplus_weibull_x_at_p(true_x, slope_true, guess, lapse_true, 0.75)
        biases.append(est_x - true_at_75)
        # x = -log10(f) is decreasing in f, so the Hz-domain CI bounds swap order in x.
        coverages.append(min(ci_low_x, ci_high_x) <= true_at_75 <= max(ci_low_x, ci_high_x))

    display_limited_rate = n_display_limited / n_runs
    assert display_limited_rate <= max_display_limited_rate, (
        f"true_f={true_freq_hz}Hz slope={slope_true}: {n_display_limited}/{n_runs} "
        f"({display_limited_rate:.0%}) runs landed in the display-limited regime -- "
        "likely a regression (a spuriously shallow slope fit pushing the fitted point "
        "toward the domain edge; see DEFAULT_SLOPE_VALUES's docstring)"
    )
    assert biases, (
        f"true_f={true_freq_hz}Hz slope={slope_true}: every run "
        f"({n_display_limited}/{n_runs}) was display-limited -- no comparable runs left "
        "to assess bias/coverage"
    )
    mean_bias = float(np.mean(biases))
    coverage_rate = float(np.mean(coverages))
    assert abs(mean_bias) < max_abs_bias, (
        f"true_f={true_freq_hz}Hz slope={slope_true}: mean bias {mean_bias:.3f} "
        f"log10(Hz)-equivalent units (display_limited_rate={display_limited_rate:.0%}) "
        "is too large"
    )
    assert coverage_rate >= min_coverage, (
        f"true_f={true_freq_hz}Hz slope={slope_true}: empirical coverage "
        f"{coverage_rate:.2f} (display_limited_rate={display_limited_rate:.0%}) is too "
        "far below nominal 0.95"
    )


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
