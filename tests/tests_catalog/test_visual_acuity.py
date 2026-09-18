"""Tests for vpsych.tests_catalog.visual_acuity: Landolt C acuity (FrACT method)."""

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
from vpsych.core.psychometric import intensity_at_p_correct
from vpsych.runner.__main__ import build_arg_parser, parse_simulated_observer_spec, run_session
from vpsych.runner.status import RunnerExitCode
from vpsych.tests_catalog import base as catalog_base
from vpsych.tests_catalog.base import check_requirements
from vpsych.tests_catalog.visual_acuity import (
    VisualAcuityParams,
    VisualAcuityTest,
    _questplus_weibull_x_at_p,
)
from vpsych.tests_catalog.visual_acuity.optotype import (
    gap_half_angle_deg,
    landolt_c_geometry,
    min_renderable_logmar,
    render_landolt_c,
)


def _display(viewing_distance_cm: float = 200.0) -> DisplayGeometry:
    return DisplayGeometry(
        width_px=1920,
        height_px=1080,
        width_cm=53.13,
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


def _make_test(display: DisplayGeometry | None = None, **param_overrides: Any) -> VisualAcuityTest:
    return VisualAcuityTest(
        params=VisualAcuityParams(**param_overrides),
        display=display or _display(),
        calibration=None,
        rng=np.random.default_rng(0),
    )


# ---------------------------------------------------------------------------
# 1. Requirements check
# ---------------------------------------------------------------------------


def test_requirements_are_met_by_a_bare_display_and_no_calibration() -> None:
    """No requirements are declared (gamma calibration not needed for this test)."""
    reasons = check_requirements(VisualAcuityTest.spec.requirements, _display(), None)
    assert reasons == []


def test_is_registered_and_visible() -> None:
    catalog_base.discover_tests()
    assert catalog_base.get_test("visual_acuity") is VisualAcuityTest
    assert VisualAcuityTest.spec.hidden is False
    assert VisualAcuityTest.spec.id in {t.spec.id for t in catalog_base.visible_tests()}


# ---------------------------------------------------------------------------
# 2. Score correctness
# ---------------------------------------------------------------------------


def test_score_correct_and_incorrect() -> None:
    test = _make_test()
    stim = {"correct_response": 90}
    assert test.score(90, stim) is True
    assert test.score(180, stim) is False
    assert test.score(None, stim) is False


def test_response_keys_8afc_are_numpad_only() -> None:
    test = _make_test(n_orientations=8)
    keys = test.response_keys()
    assert len(keys) == 8
    assert set(keys) == {"num1", "num2", "num3", "num4", "num6", "num7", "num8", "num9"}


def test_response_keys_4afc_include_arrows_and_numpad() -> None:
    test = _make_test(n_orientations=4)
    keys = set(test.response_keys())
    assert {"up", "down", "left", "right", "num2", "num4", "num6", "num8"} == keys


def test_simulated_response_returns_correct_or_a_valid_alternative() -> None:
    test = _make_test(n_orientations=8)
    rng = np.random.default_rng(1)
    stim = {"correct_response": 90}
    assert test.simulated_response(True, stim, rng) == 90
    for _ in range(20):
        wrong = test.simulated_response(False, stim, rng)
        assert wrong != 90
        assert wrong in test._angles


# ---------------------------------------------------------------------------
# 3. Stimulus-parameter math: optotype geometry, px conversions
# ---------------------------------------------------------------------------


def test_landolt_c_geometry_iso_8596_proportions() -> None:
    geom = landolt_c_geometry(gap_px=10.0)
    assert geom["outer_diameter_px"] == pytest.approx(50.0)
    assert geom["inner_diameter_px"] == pytest.approx(30.0)
    assert geom["stroke_width_px"] == pytest.approx(10.0)
    assert geom["outer_radius_px"] == pytest.approx(25.0)
    assert geom["inner_radius_px"] == pytest.approx(15.0)
    assert geom["mean_radius_px"] == pytest.approx(20.0)


def test_gap_half_angle_deg_matches_exact_chord_relation() -> None:
    half_angle = gap_half_angle_deg(gap_px=10.0, mean_radius_px=20.0)
    assert half_angle == pytest.approx(math.degrees(math.asin(0.25)))


def test_render_landolt_c_gap_faces_requested_orientation() -> None:
    tex = render_landolt_c(64, gap_px=8.0, orientation_deg=0.0, supersample=4)
    center = 32
    # Gap (background level, 1.0) at the requested orientation (right, index increasing x).
    assert np.all(tex[center, 40:56] == pytest.approx(1.0))
    # Stroke (near-black) present on the opposite side and top/bottom.
    assert tex[center, 8:24].min() < 0.5
    assert tex[8:24, center].min() < 0.5


def test_render_landolt_c_weber_contrast_sets_stroke_level() -> None:
    tex = render_landolt_c(64, gap_px=8.0, orientation_deg=90.0, supersample=4)
    assert tex.max() == pytest.approx(1.0)
    assert tex.min() == pytest.approx(1.0 + (-0.99), abs=1e-6)


def test_min_renderable_logmar_matches_gap_px_conversion() -> None:
    display = _display()
    floor = min_renderable_logmar(display.px_to_deg, min_gap_px=1.0)
    gap_arcmin = 10.0**floor
    # Round trip: converting that logMAR back to px should give ~1 px.
    assert display.deg_to_px(gap_arcmin / 60.0) == pytest.approx(1.0, abs=1e-6)


def test_domain_is_clipped_by_display_resolution_at_short_viewing_distance() -> None:
    """At a very short viewing distance, the display can't render very sharp gaps."""
    test = _make_test(display=_display(viewing_distance_cm=40.0))
    floor = min_renderable_logmar(test.display.px_to_deg, test.params.min_gap_px)
    assert test._logmar_min == pytest.approx(floor)
    assert test._logmar_min > VisualAcuityParams().logmar_domain_min


def test_present_logs_rendered_gap_px_matching_deg_to_px(monkeypatch: pytest.MonkeyPatch) -> None:
    """stimulus_params['gap_px'] must equal display.deg_to_px(gap_arcmin / 60)."""
    from vpsych.core.observers import PsychometricObserver
    from vpsych.core.psychometric import PsychometricFunction

    test = _make_test()
    fn = PsychometricFunction(
        family="weibull",
        threshold=0.3,
        slope=0.25,
        guess=0.125,
        lapse=0.02,
        intensity_scale="linear",
    )
    observer = PsychometricObserver(fn, n_afc=8)
    rng = np.random.default_rng(2)

    class _Timeline:
        stimulus_frames = 6

    trial_ctx: dict[str, Any] = {
        "rng": rng,
        "timeline": _Timeline(),
        "keyboard": None,
        "simulated_observer": observer,
        "block": "main",
        "is_catch": False,
        "trial_index": 0,
    }
    logmar = 0.4
    test.present(None, logmar, trial_ctx)
    params = trial_ctx["stimulus_params"]
    expected_gap_px = test.display.deg_to_px((10.0**logmar) / 60.0)
    assert params["gap_px"] == pytest.approx(expected_gap_px)
    assert params["outer_diameter_px"] == pytest.approx(5.0 * expected_gap_px)


# ---------------------------------------------------------------------------
# 4. summarize() on a fixture
# ---------------------------------------------------------------------------


def _trials_fixture(
    test: VisualAcuityTest, n_main: int = 40, n_catch: int = 4, catch_correct: bool = True
) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    grid = test.make_procedure().intensity_values
    intensity = grid[len(grid) // 2]  # a value guaranteed to be on the procedure's own grid
    rows = []
    for i in range(n_main):
        correct = bool(rng.random() < 0.7)
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
    for i in range(n_catch):
        rows.append(
            {
                "block": "main",
                "is_catch": True,
                "trial_index": n_main + i,
                "intensity": 1.3,
                "correct": catch_correct,
                "eye": "OU",
                "n_dropped_frames_trial": 0,
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
    assert summary1.catch_lapse_rate == pytest.approx(0.0)
    assert summary1.estimate.units == "logMAR"
    assert "decimal_acuity" in summary1.fit_params
    assert "snellen_20" in summary1.fit_params
    assert "snellen_6" in summary1.fit_params
    decimal_acuity = summary1.fit_params["decimal_acuity"]
    assert decimal_acuity == pytest.approx(10.0 ** (-summary1.estimate.value))


def test_summarize_flags_high_catch_lapse_rate() -> None:
    test = _make_test(max_trials=40)
    df = _trials_fixture(test, catch_correct=False)
    summary = test.summarize(df)
    assert summary.catch_lapse_rate == pytest.approx(1.0)
    assert any(f.code == "high_catch_lapse_rate" for f in summary.quality_flags)


def test_summarize_flags_display_resolution_limited_when_threshold_near_floor() -> None:
    """At a very short viewing distance, an easily-achieved threshold should sit near the
    display's own pixel-bounded floor and trigger the dynamic display-resolution flag."""
    display = _display(viewing_distance_cm=40.0)
    test = _make_test(display=display, max_trials=40)
    floor = min_renderable_logmar(display.px_to_deg, test.params.min_gap_px)
    rng = np.random.default_rng(7)
    rows = []
    for i in range(40):
        # Present near the floor and answer correctly often, so the procedure's
        # estimate is pulled toward (and clipped near) the domain minimum.
        rows.append(
            {
                "block": "main",
                "is_catch": False,
                "trial_index": i,
                "intensity": test._logmar_min,
                "correct": bool(rng.random() < 0.85),
                "eye": "OU",
                "n_dropped_frames_trial": 0,
            }
        )
    df = pd.DataFrame(rows)
    summary = test.summarize(df)
    assert floor == pytest.approx(test._logmar_min)
    assert any(f.code == "display_resolution_limited" for f in summary.quality_flags)


# ---------------------------------------------------------------------------
# 5. Simulated end-to-end recovery via the runner
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
    """Smoke-level check (loose tolerance, few trials), mirroring _example's own runner test.

    The `psychometric:` simulate spec's `slope` is on `vpsych.core.psychometric`'s own
    (F(0)=0.5-rescaled) weibull scale, a genuinely different parameterization from what
    `QuestPlusProcedure`/`questplus` fits internally (see `visual_acuity`'s module
    docstring) -- so this only checks the recovered threshold is in the right ballpark of
    a target computed against the *observer's own* function, not a tight recovery check
    (that belongs in `core/procedures`'s own tests, per `docs/WRITING_A_TEST.md` section 11).
    """
    catalog_base.discover_tests()
    data_root = tmp_path / "data"
    _write_calibration(data_root, _calibration())

    true_threshold = 0.3
    spec = f"psychometric:threshold={true_threshold},slope=0.25,lapse=0.02,guess=0.125,n_afc=8"
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "participant_id": "sub-0001",
                "tests": [
                    {
                        "task_id": "visual_acuity",
                        "eye": "OU",
                        "params": {"max_trials": 50},
                        "viewing_distance_cm": 200.0,
                    }
                ],
                "ordering": "fixed",
                "seed": 42,
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
            spec,
        ]
    )
    writer = _FakeWriter()
    exit_code = run_session(args, writer_factory=lambda *a: writer)
    assert exit_code == RunnerExitCode.OK
    assert len(writer.summaries) == 1

    summary = writer.summaries[0]
    observer = parse_simulated_observer_spec(spec)
    fn = observer.true_function
    target_p = fn.guess + 0.5 * (1.0 - fn.guess - fn.lapse)
    target = intensity_at_p_correct(fn, target_p)
    assert abs(summary.estimate.value - target) < 0.6


# ---------------------------------------------------------------------------
# 6. Slow: recovery bias/coverage over several true thresholds (native family)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("true_threshold", [-0.1, 0.3, 0.7])
def test_recovery_slow_bias_and_coverage(true_threshold: float) -> None:
    """Bias/coverage against a ground-truth observer defined directly in questplus's OWN
    Weibull parameterization (the family QuestPlusProcedure actually fits), so this
    isolates this test's own domain/grid/criterion-conversion choices from any family
    mismatch with vpsych.core.observers.PsychometricObserver (see the runner-based test
    above for that looser, family-mismatched smoke check)."""
    display = _display(viewing_distance_cm=200.0)
    n_reps = 20
    slope_true = 3.0
    lapse_true = 0.02
    guess = 0.125

    def p_correct(x: float) -> float:
        return (
            1.0
            - lapse_true
            - (1.0 - guess - lapse_true) * math.exp(-(10.0 ** (slope_true * (x - true_threshold))))
        )

    biases = []
    covered = 0
    for seed in range(n_reps):
        rng = np.random.default_rng(1000 + seed)
        test = _make_test(display=display, max_trials=40)
        proc = test.make_procedure()
        while not proc.finished:
            x = proc.next_intensity()
            correct = bool(rng.random() < p_correct(x))
            proc.update(x, correct)
        est = proc.estimate()
        target_p = guess + 0.5 * (1.0 - guess - est.extra["lapse_rate"])
        reported, ci_low, ci_high, _ = test._fract_criterion_threshold(
            est.value, est.ci_low, est.ci_high, est.extra["slope"], est.extra["lapse_rate"]
        )
        true_at_criterion = _questplus_weibull_x_at_p(
            true_threshold, slope_true, guess, lapse_true, target_p
        )
        biases.append(reported - true_at_criterion)
        if ci_low <= true_at_criterion <= ci_high:
            covered += 1

    mean_bias = float(np.mean(biases))
    coverage = covered / n_reps
    # Loose bounds appropriate to n_reps=20 (a fast "slow" test, not a full 200-500 run
    # scientific validation sweep) -- see docs/methods/visual_acuity.md for the honestly-
    # reported numbers from a larger sweep.
    assert abs(mean_bias) < 0.25
    assert coverage >= 0.5


# ---------------------------------------------------------------------------
# 7. Display smoke test (excluded by default; `uv run pytest -m display`)
# ---------------------------------------------------------------------------


class _FakeKeyboard:
    """Minimal stand-in for psychopy.hardware.keyboard.Keyboard, for a display smoke test.

    Returns `key_name` on the first `getKeys()` call after each `clearEvents()`, then
    nothing until cleared again -- enough for `present()`'s per-frame response-polling
    loop to terminate immediately without a real keypress.
    """

    def __init__(self, key_name: str) -> None:
        self._key_name = key_name
        self._armed = True

    def clearEvents(self) -> None:  # noqa: N802 -- mimics psychopy.hardware.keyboard.Keyboard
        self._armed = True

    def getKeys(  # noqa: N802 -- mimics psychopy.hardware.keyboard.Keyboard
        self,
        keyList: list[str] | None = None,  # noqa: N803
        waitRelease: bool = False,  # noqa: N803
    ) -> list[Any]:
        del keyList, waitRelease
        if not self._armed:
            return []
        self._armed = False

        class _Key:
            name = self._key_name
            rt = 0.3

        return [_Key()]


@pytest.mark.display
def test_display_smoke_two_trials() -> None:
    """Opens a real PsychoPy window, builds stimuli, and presents 2 trials with a fake
    keyboard. Excluded by default (`-m "not display"`); run explicitly with
    `uv run pytest -m display` when a display is available."""
    from psychopy import visual

    win = visual.Window(size=(800, 600), fullscr=False, units="pix", allowGUI=False)
    try:
        test = _make_test(max_trials=4)
        test.build_stimuli(win)
        proc = test.make_procedure()
        rng = np.random.default_rng(0)

        class _Timeline:
            fixation_frames = 2
            stimulus_frames = 2
            response_timeout_frames = None
            iti_frames = 2

        for _ in range(2):
            intensity = proc.next_intensity()
            trial_ctx: dict[str, Any] = {
                "timeline": _Timeline(),
                "rng": rng,
                "block": "main",
                "is_catch": False,
                "trial_index": 0,
                "keyboard": _FakeKeyboard("num8"),
                "simulated_observer": None,
            }
            presented = test.present(win, intensity, trial_ctx)
            assert presented.response in test._angles or presented.response is None
            correct = test.score(presented.response, trial_ctx["stimulus_params"])
            proc.update(intensity, correct)
    finally:
        win.close()
