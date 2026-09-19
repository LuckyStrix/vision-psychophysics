"""Tests for vpsych.tests_catalog.vernier_acuity: Vernier (hyperacuity) offset detection."""

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
from vpsych.tests_catalog.vernier_acuity import (
    VernierAcuityParams,
    VernierAcuityTest,
)
from vpsych.tests_catalog.vernier_acuity.texture import (
    line_column_coverage,
    render_vertical_line_texture,
    texture_centroid_x_px,
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


def _calibration_grade(method: str = "photometer") -> Calibration:
    return Calibration(
        created_utc=datetime(2026, 9, 1, tzinfo=timezone.utc),
        geometry=_display(),
        gamma=GammaCalibration(
            method=method, gamma_single=2.2, lum_min_cdm2=0.3, lum_max_cdm2=120.0
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


def _make_test(display: DisplayGeometry | None = None, **param_overrides: Any) -> VernierAcuityTest:
    return VernierAcuityTest(
        params=VernierAcuityParams(**param_overrides),
        display=display or _display(),
        calibration=None,
        rng=np.random.default_rng(0),
    )


# ---------------------------------------------------------------------------
# 1. Requirements check
# ---------------------------------------------------------------------------


def test_requirements_need_gamma_calibration_grade_b() -> None:
    reqs = VernierAcuityTest.spec.requirements
    assert reqs.needs_gamma_calibration is True
    assert reqs.min_luminance_grade == "B"

    # Unmet: no calibration at all.
    reasons = check_requirements(reqs, _display(), None)
    assert reasons != []

    # Unmet: grade C (uncalibrated gamma method).
    reasons_c = check_requirements(reqs, _display(), _calibration_grade(method="none"))
    assert reasons_c != []

    # Met: grade A (photometer) satisfies a grade-B-minimum requirement.
    reasons_a = check_requirements(reqs, _display(), _calibration_grade(method="photometer"))
    assert reasons_a == []

    # Met: grade B (psychophysical) exactly satisfies the requirement.
    reasons_b = check_requirements(reqs, _display(), _calibration_grade(method="psychophysical"))
    assert reasons_b == []


def test_is_registered_and_visible() -> None:
    catalog_base.discover_tests()
    assert catalog_base.get_test("vernier_acuity") is VernierAcuityTest
    assert VernierAcuityTest.spec.hidden is False
    assert VernierAcuityTest.spec.id in {t.spec.id for t in catalog_base.visible_tests()}


# ---------------------------------------------------------------------------
# 2. Score correctness
# ---------------------------------------------------------------------------


def test_score_correct_and_incorrect() -> None:
    test = _make_test()
    stim = {"correct_response": "left"}
    assert test.score("left", stim) is True
    assert test.score("right", stim) is False
    assert test.score(None, stim) is False


def test_response_keys_are_arrows() -> None:
    assert set(_make_test().response_keys()) == {"left", "right"}


# ---------------------------------------------------------------------------
# 3. Stimulus-parameter math: sub-pixel texture centroid, px conversions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("center", [10.0, 10.123, 10.5, 10.987, 15.003, 9.9999, 3.0007, 27.5])
def test_texture_centroid_matches_requested_subpixel_position(center: float) -> None:
    """Phase 2A requirement: centroid of the linearized luminance profile must equal the
    requested sub-pixel position to within 0.02 px."""
    tex = render_vertical_line_texture(
        canvas_width_px=30,
        canvas_height_px=10,
        center_x_px=center,
        width_px=2.0,
        y_start_px=2.0,
        y_end_px=8.0,
    )
    recovered = texture_centroid_x_px(tex[5])
    assert abs(recovered - center) < 0.02


def test_line_column_coverage_integrates_to_bar_width() -> None:
    cov = line_column_coverage(canvas_width_px=20, center_px=10.3, width_px=3.0)
    assert cov.sum() == pytest.approx(3.0)
    assert cov.min() >= 0.0
    assert cov.max() <= 1.0


def test_render_vertical_line_texture_rejects_degenerate_y_range() -> None:
    with pytest.raises(ValueError):
        render_vertical_line_texture(
            10, 10, center_x_px=5.0, width_px=1.0, y_start_px=5.0, y_end_px=5.0
        )


def test_present_logs_offset_px_matching_deg_to_px() -> None:
    from vpsych.core.observers import PsychometricObserver
    from vpsych.core.psychometric import PsychometricFunction

    test = _make_test()
    fn = PsychometricFunction(
        family="weibull",
        threshold=1.0,
        slope=0.3,
        guess=0.5,
        lapse=0.02,
        intensity_scale="log10",
    )
    observer = PsychometricObserver(fn, n_afc=2)
    rng = np.random.default_rng(3)

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
    log_offset = 1.2
    test.present(None, log_offset, trial_ctx)
    params = trial_ctx["stimulus_params"]
    expected_offset_arcsec = 10.0**log_offset
    expected_offset_px = test.display.deg_to_px(expected_offset_arcsec / 3600.0)
    assert params["offset_px"] == pytest.approx(expected_offset_px)
    assert params["offset_arcsec"] == pytest.approx(expected_offset_arcsec)
    assert params["correct_response"] in ("left", "right")


def test_canvas_size_is_fixed_across_the_intensity_domain() -> None:
    """The stimulus's overall apparent size must not vary trial to trial (see module
    docstring: it would otherwise be usable as a spurious cue)."""
    test = _make_test()
    w1, h1 = test._canvas_width_px, test._canvas_height_px
    tex_small = test._render_stimulus_texture(offset_px=0.5, direction=1)
    tex_large = test._render_stimulus_texture(
        offset_px=test.display.deg_to_px((10.0**test.params.log_offset_domain_max) / 3600.0),
        direction=-1,
    )
    assert tex_small.shape == (h1, w1)
    assert tex_large.shape == (h1, w1)


# ---------------------------------------------------------------------------
# 4. summarize() on a fixture
# ---------------------------------------------------------------------------


def _trials_fixture(
    test: VernierAcuityTest, n_main: int = 40, n_catch: int = 4, catch_correct: bool = True
) -> pd.DataFrame:
    rng = np.random.default_rng(9)
    grid = test.make_procedure().intensity_values
    intensity = grid[len(grid) // 2]
    rows = []
    for i in range(n_main):
        correct = bool(rng.random() < 0.75)
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
                "intensity": test.params.log_offset_domain_max,
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
    assert summary1.estimate.units == "arcsec"
    assert summary1.estimate.value > 0.0
    assert summary1.estimate.ci_low <= summary1.estimate.value <= summary1.estimate.ci_high


def test_summarize_flags_high_catch_lapse_rate() -> None:
    test = _make_test(max_trials=40)
    df = _trials_fixture(test, catch_correct=False)
    summary = test.summarize(df)
    assert summary.catch_lapse_rate == pytest.approx(1.0)
    assert any(f.code == "high_catch_lapse_rate" for f in summary.quality_flags)


def test_summarize_flags_short_viewing_distance() -> None:
    test = _make_test(display=_display(viewing_distance_cm=50.0), max_trials=40)
    df = _trials_fixture(test)
    summary = test.summarize(df)
    assert any(f.code == "short_viewing_distance" for f in summary.quality_flags)


def test_summarize_flags_near_rendering_limit() -> None:
    """Force a very small reported threshold (near/at the domain minimum, 1 arcsec) at a
    long viewing distance, so the equivalent pixel offset is comfortably below 0.1 px."""
    test = _make_test(display=_display(viewing_distance_cm=500.0), max_trials=40)
    rng = np.random.default_rng(11)
    grid = test.make_procedure().intensity_values
    intensity = grid[0]  # smallest offset in the domain
    rows = []
    for i in range(40):
        rows.append(
            {
                "block": "main",
                "is_catch": False,
                "trial_index": i,
                "intensity": intensity,
                "correct": bool(rng.random() < 0.9),
                "eye": "OU",
                "n_dropped_frames_trial": 0,
            }
        )
    df = pd.DataFrame(rows)
    summary = test.summarize(df)
    assert any(f.code == "near_rendering_limit" for f in summary.quality_flags)


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
    """Smoke-level check, mirroring _example's own runner test and visual_acuity's --
    the `psychometric:` observer's `slope` is on vpsych.core.psychometric's own scale, a
    different parameterization from what QuestPlusProcedure fits internally, so this
    checks only that the recovered arcsec threshold is in the right ballpark (compared in
    log10 space, since arcsec spans orders of magnitude) of a target computed against the
    observer's own function."""
    catalog_base.discover_tests()
    data_root = tmp_path / "data"
    _write_calibration(data_root, _calibration_grade())

    true_log_offset = 1.2
    spec = f"psychometric:threshold={true_log_offset},slope=0.3,lapse=0.02,guess=0.5,n_afc=2"
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "participant_id": "sub-0001",
                "tests": [
                    {
                        "task_id": "vernier_acuity",
                        "eye": "OU",
                        "params": {"max_trials": 60},
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
    target_log_offset = intensity_at_p_correct(observer.true_function, 0.75)
    reported_log_offset = math.log10(summary.estimate.value)
    assert abs(reported_log_offset - target_log_offset) < 0.6


# ---------------------------------------------------------------------------
# 6. Slow: recovery bias/coverage over several true thresholds (native family)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("true_log_offset", [0.5, 1.0, 1.5])
def test_recovery_slow_bias_and_coverage(true_log_offset: float) -> None:
    """Bias/coverage against a ground-truth observer defined directly in questplus's OWN
    Weibull parameterization -- see visual_acuity's identically-structured test for why."""
    display = _display(viewing_distance_cm=150.0)
    n_reps = 20
    slope_true = 3.0
    lapse_true = 0.02
    guess = 0.5

    def p_correct(x: float) -> float:
        return (
            1.0
            - lapse_true
            - (1.0 - guess - lapse_true) * math.exp(-(10.0 ** (slope_true * (x - true_log_offset))))
        )

    biases = []
    covered = 0
    for seed in range(n_reps):
        rng = np.random.default_rng(2000 + seed)
        test = _make_test(display=display, max_trials=60)
        proc = test.make_procedure()
        while not proc.finished:
            x = proc.next_intensity()
            correct = bool(rng.random() < p_correct(x))
            proc.update(x, correct)
        est = proc.estimate()
        reported = questplus_weibull_x_at_p(
            est.value, est.extra["slope"], guess, est.extra["lapse_rate"], 0.75
        )
        shift = reported - est.value
        ci_low = est.ci_low + shift
        ci_high = est.ci_high + shift
        true_at_75 = questplus_weibull_x_at_p(true_log_offset, slope_true, guess, lapse_true, 0.75)
        biases.append(reported - true_at_75)
        if ci_low <= true_at_75 <= ci_high:
            covered += 1

    mean_bias = float(np.mean(biases))
    coverage = covered / n_reps
    assert abs(mean_bias) < 0.15
    assert coverage >= 0.5


# ---------------------------------------------------------------------------
# 7. Display smoke test (excluded by default; `uv run pytest -m display`)
# ---------------------------------------------------------------------------


class _FakeKeyboard:
    """Minimal stand-in for psychopy.hardware.keyboard.Keyboard, for a display smoke test.

    Returns `key_name` on the first `getKeys()` call after each `clearEvents()`, then
    nothing until cleared again.
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
                "keyboard": _FakeKeyboard("left"),
                "simulated_observer": None,
            }
            presented = test.present(win, intensity, trial_ctx)
            assert presented.response in ("left", "right", None)
            correct = test.score(presented.response, trial_ctx["stimulus_params"])
            proc.update(intensity, correct)
    finally:
        win.close()
