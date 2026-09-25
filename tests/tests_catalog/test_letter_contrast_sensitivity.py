"""Tests for vpsych.tests_catalog.letter_contrast_sensitivity (Pelli-Robson analogue).

Follows the "required tests" pattern from `docs/WRITING_A_TEST.md`:
requirements checks, score correctness (including the "exact letter
identity only" confusion-scoring rule), `summarize()` on a fixture (+
reproducibility), a fast simulated end-to-end recovery check via the real
runner, and a `@pytest.mark.slow` recovery check over several true
threshold values. Rendering math and dithering accuracy are covered in
`tests/tests_catalog/test_contrast_rendering.py` (shared with
`contrast_sensitivity_function`). A `@pytest.mark.display` smoke test opens
a real PsychoPy window.
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
from vpsych.core.observers import PsychometricObserver
from vpsych.core.psychometric import PsychometricFunction, intensity_at_p_correct
from vpsych.runner.__main__ import build_arg_parser, run_session
from vpsych.runner.status import RunnerExitCode
from vpsych.tests_catalog import base as catalog_base
from vpsych.tests_catalog._contrast_rendering import SLOAN_LETTERS
from vpsych.tests_catalog.base import check_requirements
from vpsych.tests_catalog.letter_contrast_sensitivity import (
    DEFAULT_INTENSITY_VALUES,
    LetterContrastSensitivityTest,
    LetterCSParams,
)


def _display() -> DisplayGeometry:
    return DisplayGeometry(
        width_px=1920,
        height_px=1080,
        width_cm=53.13,
        height_cm=29.88,
        viewing_distance_cm=100.0,
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


def _make_test(max_trials: int = 40) -> LetterContrastSensitivityTest:
    return LetterContrastSensitivityTest(
        params=LetterCSParams(max_trials=max_trials),
        display=_display(),
        calibration=_calibration(),
        rng=np.random.default_rng(0),
    )


# ---------------------------------------------------------------------------
# 1. Requirements check
# ---------------------------------------------------------------------------


def test_requirements_met_with_gamma_calibration() -> None:
    reasons = check_requirements(
        LetterContrastSensitivityTest.spec.requirements, _display(), _calibration()
    )
    assert reasons == []


def test_requirements_unmet_without_calibration() -> None:
    reasons = check_requirements(LetterContrastSensitivityTest.spec.requirements, _display(), None)
    assert any("gamma calibration" in r for r in reasons)


def test_requirements_unmet_grade_c() -> None:
    grade_c_cal = _calibration().model_copy(
        update={"gamma": GammaCalibration(method="none", lum_min_cdm2=0.0, lum_max_cdm2=100.0)}
    )
    reasons = check_requirements(
        LetterContrastSensitivityTest.spec.requirements, _display(), grade_c_cal
    )
    assert reasons  # uncalibrated -> unmet


def test_is_registered_and_visible() -> None:
    catalog_base.discover_tests()
    assert catalog_base.get_test("letter_contrast_sensitivity") is LetterContrastSensitivityTest
    assert LetterContrastSensitivityTest.spec.hidden is False
    assert "letter_contrast_sensitivity" in {t.spec.id for t in catalog_base.visible_tests()}


# ---------------------------------------------------------------------------
# 2. Score correctness (exact identity only -- no partial credit)
# ---------------------------------------------------------------------------


def test_score_exact_match_only() -> None:
    test = _make_test()
    stim = {"correct_response": "o"}
    assert test.score("o", stim) is True
    assert test.score("c", stim) is False  # visually similar letter -- still wrong
    assert test.score(None, stim) is False


def test_response_keys_are_the_ten_sloan_letters_lowercase() -> None:
    test = _make_test()
    assert set(test.response_keys()) == {letter.lower() for letter in SLOAN_LETTERS}
    assert len(test.response_keys()) == 10


def test_make_catch_trial_intensity_is_high_contrast() -> None:
    test = _make_test()
    catch = test.make_catch_trial_intensity()
    assert catch == pytest.approx(np.log10(test.params.catch_contrast))
    # Catch intensity should sit near the top of the candidate intensity grid
    # (suprathreshold, comfortably detectable) -- within its top decile.
    top_decile = max(DEFAULT_INTENSITY_VALUES) - 0.1 * (
        max(DEFAULT_INTENSITY_VALUES) - min(DEFAULT_INTENSITY_VALUES)
    )
    assert catch >= top_decile


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


def _run_session_with_psychometric_observer(
    tmp_path: Path, true_threshold: float, max_trials: int, seed: int, slope: float = 0.4
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
                        "task_id": "letter_contrast_sensitivity",
                        "eye": "OU",
                        "params": {"max_trials": max_trials},
                        "viewing_distance_cm": 100.0,
                    }
                ],
                "ordering": "fixed",
                "seed": seed,
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
            str(tmp_path / f"status_{seed}.json"),
            "--data-root",
            str(data_root),
            "--simulate",
            f"psychometric:threshold={true_threshold},slope={slope},lapse=0.02,guess=0.1,n_afc=10",
        ]
    )
    writer = _FakeWriter()
    exit_code = run_session(args, writer_factory=lambda *a: writer)
    assert exit_code == RunnerExitCode.OK
    assert len(writer.summaries) == 1
    return writer.summaries[0]


def test_simulated_end_to_end_recovery_via_runner(tmp_path: Path) -> None:
    true_threshold = -1.2  # log10 Weber contrast, F(0)=0.5 crossing
    summary = _run_session_with_psychometric_observer(
        tmp_path, true_threshold, max_trials=60, seed=1
    )
    # Recovered log CS should be in the right ballpark of -true_threshold
    # (generous tolerance: 60 trials, smoke-level check only -- see
    # tests/procedures/test_questplus_procedure.py for the properly-powered
    # recovery/coverage validation, per docs/WRITING_A_TEST.md section 11).
    assert abs(summary.estimate.value - (-true_threshold)) < 1.0


@pytest.mark.slow
def test_recovery_slow_over_several_true_thresholds(tmp_path: Path) -> None:
    """Recovery within tolerance over several true threshold values, through
    the full test/runner pipeline (QUEST+'s own bias/coverage validation
    lives in tests/procedures/test_questplus_procedure.py with much larger
    n; not duplicated here per docs/WRITING_A_TEST.md section 11)."""
    true_thresholds = [-2.0, -1.2, -0.5]
    n_reps = 5
    for i, true_threshold in enumerate(true_thresholds):
        diffs = []
        for rep in range(n_reps):
            summary = _run_session_with_psychometric_observer(
                tmp_path, true_threshold, max_trials=40, seed=2000 * i + rep
            )
            diffs.append(summary.estimate.value - (-true_threshold))
        mean_diff = float(np.mean(diffs))
        assert abs(mean_diff) < 0.5, f"threshold {true_threshold}: mean bias {mean_diff:.3f}"


# ---------------------------------------------------------------------------
# 4. summarize() on a fixture (+ reproducibility)
# ---------------------------------------------------------------------------


def _trials_fixture(
    test: LetterContrastSensitivityTest, n_main: int = 30, n_catch: int = 3
) -> pd.DataFrame:
    true_fn = PsychometricFunction(
        family="weibull", threshold=-1.2, slope=0.4, guess=0.1, lapse=0.02, intensity_scale="log10"
    )
    obs = PsychometricObserver(true_fn, n_afc=10)
    procedure = test.make_procedure()
    rng = np.random.default_rng(11)
    rows = []
    for i in range(n_main):
        intensity = procedure.next_intensity()
        is_correct = obs.decide_correct({"intensity": intensity}, rng)
        procedure.update(intensity, is_correct)
        rows.append(
            {
                "block": "main",
                "is_catch": False,
                "trial_index": i,
                "intensity": intensity,
                "correct": is_correct,
                "eye": "OU",
                "n_dropped_frames_trial": 0,
            }
        )
    catch_intensity = test.make_catch_trial_intensity()
    for j in range(n_catch):
        rows.append(
            {
                "block": "main",
                "is_catch": True,
                "trial_index": n_main + j,
                "intensity": catch_intensity,
                "correct": True,
                "eye": "OU",
                "n_dropped_frames_trial": 0,
            }
        )
    return pd.DataFrame(rows)


def test_summarize_on_fixture_is_deterministic_and_well_formed() -> None:
    test = _make_test(max_trials=30)
    df = _trials_fixture(test)
    summary1 = test.summarize(df)
    summary2 = test.summarize(df)
    assert summary1.model_dump() == summary2.model_dump()
    assert summary1.n_trials == 30
    assert summary1.n_catch == 3
    assert summary1.estimate.ci_low <= summary1.estimate.value <= summary1.estimate.ci_high
    assert summary1.estimate.units == "log10_contrast_sensitivity"
    assert "threshold_criterion" in summary1.estimate.extra
    assert "threshold_log10_weber_contrast" in summary1.estimate.extra


def test_summarize_matches_fresh_test_instance_reanalysis() -> None:
    test1 = _make_test(max_trials=30)
    df = _trials_fixture(test1)
    summary1 = test1.summarize(df)

    test2 = _make_test(max_trials=30)  # fresh instance
    summary2 = test2.summarize(df)
    assert summary1.model_dump() == summary2.model_dump()


def test_log_cs_ci_bounds_correctly_flip_sign_and_order() -> None:
    """log CS = -contrast_threshold, so ci_low/ci_high must swap under negation."""
    test = _make_test(max_trials=30)
    df = _trials_fixture(test)
    summary = test.summarize(df)
    raw_low = -summary.estimate.ci_high
    raw_high = -summary.estimate.ci_low
    assert raw_low <= summary.estimate.extra["threshold_log10_weber_contrast"] <= raw_high


def test_summarize_flags_high_catch_lapse_rate() -> None:
    test = _make_test(max_trials=30)
    df = _trials_fixture(test)
    df.loc[df["is_catch"], "correct"] = False
    summary = test.summarize(df)
    assert summary.catch_lapse_rate == pytest.approx(1.0)
    assert any(f.code == "high_catch_lapse_rate" for f in summary.quality_flags)


# ---------------------------------------------------------------------------
# Sanity: intensity_at_p_correct is consistent with the documented criterion
# ---------------------------------------------------------------------------


def test_threshold_criterion_is_not_75_percent_point() -> None:
    """Sanity-checks the docs/methods claim that this test's F(0)=0.5 threshold
    criterion differs from the conventional 75%-correct point used elsewhere
    in this suite (e.g. 2AFC tests)."""
    fn = PsychometricFunction(
        family="weibull", threshold=-1.2, slope=0.4, guess=0.1, lapse=0.02, intensity_scale="log10"
    )
    p_at_threshold = fn.p_correct(-1.2)
    seventy_five_point = intensity_at_p_correct(fn, 0.75)
    assert p_at_threshold != pytest.approx(0.75)
    assert seventy_five_point != pytest.approx(-1.2)


# ---------------------------------------------------------------------------
# 5. Display smoke test (real PsychoPy window, fake keyboard)
# ---------------------------------------------------------------------------


class _FakeClock:
    def reset(self) -> None:
        pass


class _FakeKeyPress:
    """Mimics psychopy.hardware.keyboard.KeyPress: `.name` and onset-relative `.rt`."""

    def __init__(self, name: str, rt: float = 0.3) -> None:
        self.name = name
        self.rt = rt


class _FakeKeyboard:
    """Stands in for `psychopy.hardware.keyboard.Keyboard`: returns a canned key instantly.

    Returns KeyPress-like objects (`.name`, onset-relative `.rt`), like the real Keyboard.
    """

    def __init__(self, key: str) -> None:
        self._key = key
        self._returned = False

    clock = _FakeClock()

    def clearEvents(self) -> None:  # noqa: N802
        self._returned = False

    def getKeys(  # noqa: N802
        self,
        keyList: list[str] | None = None,  # noqa: N803
        waitRelease: bool = True,  # noqa: N803
        clear: bool = True,
    ) -> list[_FakeKeyPress]:
        if self._returned:
            return []
        self._returned = True
        return [_FakeKeyPress(self._key)]


@pytest.mark.display
def test_display_smoke_two_trials() -> None:
    """Opens a real PsychoPy window and presents 2 trials with a fake keyboard.

    Uses minimal frame counts (not the test's normal defaults) purely to
    keep this manual smoke test fast; see
    `test_contrast_sensitivity_function.py::test_display_smoke_two_trials`'s
    docstring for why.
    """
    from psychopy import visual

    fast_params = LetterCSParams(
        max_trials=2, fixation_duration_ms=1.0, response_timeout_ms=50.0, iti_ms=1.0
    )
    test = LetterContrastSensitivityTest(
        params=fast_params,
        display=_display(),
        calibration=_calibration(),
        rng=np.random.default_rng(0),
    )
    win = visual.Window(size=(400, 300), fullscr=False, allowGUI=False)
    try:
        test.build_stimuli(win)
        rng = np.random.default_rng(0)
        for trial_index in range(2):
            trial_ctx = {
                "timeline": test._timeline(),
                "rng": rng,
                "block": "main",
                "is_catch": False,
                "trial_index": trial_index,
                "keyboard": _FakeKeyboard("o"),
                "simulated_observer": None,
            }
            intensity = test.make_catch_trial_intensity()
            presented = test.present(win, intensity, trial_ctx)
            assert presented.response in test.response_keys()
            assert presented.stimulus_onset_s >= 0.0
            assert presented.rt_s is not None and presented.rt_s >= 0.0
    finally:
        win.close()
