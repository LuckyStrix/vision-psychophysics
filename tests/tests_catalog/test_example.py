"""Tests for vpsych.tests_catalog._example: the canonical example test plugin.

This exercises the "required tests" pattern `docs/WRITING_A_TEST.md` asks
every real test implementation to have: a requirements check, response
scoring correctness, simulated end-to-end recovery via the real runner, and
`summarize()` on a fixed trials fixture. `tests/integration/test_end_to_end.py`
additionally runs this same test through the runner as a real subprocess.
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
from vpsych.core.psychometric import intensity_at_p_correct
from vpsych.runner.__main__ import build_arg_parser, parse_simulated_observer_spec, run_session
from vpsych.runner.status import RunnerExitCode
from vpsych.tests_catalog import base as catalog_base
from vpsych.tests_catalog._example import ExampleContrastParams, ExampleContrastTest
from vpsych.tests_catalog.base import check_requirements


def _display() -> DisplayGeometry:
    return DisplayGeometry(
        width_px=1920,
        height_px=1080,
        width_cm=53.13,
        height_cm=29.88,
        viewing_distance_cm=57.0,
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


# ---------------------------------------------------------------------------
# 1. Requirements check
# ---------------------------------------------------------------------------


def test_requirements_are_met_by_a_bare_display_and_no_calibration() -> None:
    """This example declares no requirements, so it should always be runnable."""
    reasons = check_requirements(ExampleContrastTest.spec.requirements, _display(), None)
    assert reasons == []


def test_is_registered_and_hidden() -> None:
    catalog_base.discover_tests()
    assert catalog_base.get_test("example_contrast_2afc") is ExampleContrastTest
    assert ExampleContrastTest.spec.hidden is True
    assert ExampleContrastTest.spec.id not in {t.spec.id for t in catalog_base.visible_tests()}


# ---------------------------------------------------------------------------
# 2. Score correctness
# ---------------------------------------------------------------------------


def test_score_correct_and_incorrect() -> None:
    test = ExampleContrastTest(
        params=ExampleContrastParams(),
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
    """Runs the real run_session() (in-process, real trial loop, fake writer) driven by a
    --simulate psychometric: observer, and checks the recovered threshold is in the right
    ballpark of the known simulated truth."""
    catalog_base.discover_tests()
    data_root = tmp_path / "data"
    _write_calibration(data_root, _calibration())

    true_threshold = -1.0
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "participant_id": "sub-0001",
                "tests": [
                    {
                        "task_id": "example_contrast_2afc",
                        "eye": "OU",
                        "params": {"max_trials": 60},
                        "viewing_distance_cm": 57.0,
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
            f"psychometric:threshold={true_threshold},slope=0.3,lapse=0.02",
        ]
    )
    writer = _FakeWriter()
    exit_code = run_session(args, writer_factory=lambda *a: writer)
    assert exit_code == RunnerExitCode.OK
    assert len(writer.summaries) == 1

    summary = writer.summaries[0]
    observer = parse_simulated_observer_spec(
        f"psychometric:threshold={true_threshold},slope=0.3,lapse=0.02"
    )
    target = intensity_at_p_correct(observer.true_function, 0.75)  # type: ignore[attr-defined]
    # Generous tolerance: only 60 trials, and QUEST+ threshold reporting isn't
    # necessarily at the 75%-correct point -- this just checks it's in the
    # right ballpark, not tightly calibrated (see test_questplus_procedure.py
    # for the properly-powered recovery/coverage validation).
    assert abs(summary.estimate.value - target) < 1.0


# ---------------------------------------------------------------------------
# 4. summarize() on a fixture
# ---------------------------------------------------------------------------


def _trials_fixture() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    rows = []
    for i in range(30):
        intensity = -1.0
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
    for i in range(3):
        rows.append(
            {
                "block": "main",
                "is_catch": True,
                "trial_index": 30 + i,
                "intensity": 0.2,
                "correct": True,
                "eye": "OU",
                "n_dropped_frames_trial": 0,
            }
        )
    return pd.DataFrame(rows)


def test_summarize_on_fixture_is_deterministic_and_well_formed() -> None:
    test = ExampleContrastTest(
        params=ExampleContrastParams(max_trials=30),
        display=_display(),
        calibration=None,
        rng=np.random.default_rng(0),
    )
    df = _trials_fixture()
    summary1 = test.summarize(df)
    summary2 = test.summarize(df)
    assert summary1.model_dump() == summary2.model_dump()
    assert summary1.n_trials == 30
    assert summary1.n_catch == 3
    assert summary1.catch_lapse_rate == pytest.approx(0.0)
    assert summary1.estimate.ci_low <= summary1.estimate.value <= summary1.estimate.ci_high


def test_summarize_flags_high_catch_lapse_rate() -> None:
    test = ExampleContrastTest(
        params=ExampleContrastParams(max_trials=30),
        display=_display(),
        calibration=None,
        rng=np.random.default_rng(0),
    )
    df = _trials_fixture()
    df.loc[df["is_catch"], "correct"] = False  # miss every catch trial
    summary = test.summarize(df)
    assert summary.catch_lapse_rate == pytest.approx(1.0)
    assert any(f.code == "high_catch_lapse_rate" for f in summary.quality_flags)
