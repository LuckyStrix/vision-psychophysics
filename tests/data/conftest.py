"""Shared fixtures and helpers for vpsych.data tests."""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel

from vpsych.core.calibration.models import (
    Calibration,
    ColorCalibration,
    EnvironmentChecklist,
    GammaCalibration,
    PrimaryChromaticity,
)
from vpsych.core.display import DisplayGeometry
from vpsych.core.procedures.base import ThresholdEstimate
from vpsych.core.trial import TrialRecord
from vpsych.data import dataset
from vpsych.data.schemas import PlannedTest, SessionInfo, SessionPlan, TestSummary
from vpsych.data.writer import SessionWriter
from vpsych.tests_catalog import base as tests_catalog_base
from vpsych.tests_catalog.base import (
    PsychophysicalTest,
    TestRequirements,
    TestSpec,
)


@pytest.fixture(autouse=True)
def _restore_tmp_path_permissions(tmp_path: Path) -> Iterator[None]:
    """Make everything under `tmp_path` writable again after each test.

    `SessionWriter.finalize` (and `dataset.save_calibration`) chmod their
    files to 0o444 by default, which is the whole point of the tests in
    this package -- but pytest's own `tmp_path` cleanup needs to delete
    those files afterward, and directory entries with no write/execute bit
    can't be removed. This restores normal permissions after every test,
    regardless of outcome.
    """
    yield
    for p in tmp_path.rglob("*"):
        with contextlib.suppress(OSError):
            os.chmod(p, 0o755 if p.is_dir() else 0o644)
    with contextlib.suppress(OSError):
        os.chmod(tmp_path, 0o755)


def make_display() -> DisplayGeometry:
    return DisplayGeometry(
        width_px=1920,
        height_px=1080,
        width_cm=53.0,
        height_cm=30.0,
        viewing_distance_cm=57.0,
        refresh_hz=60.0,
    )


def make_environment() -> EnvironmentChecklist:
    return EnvironmentChecklist(
        room_lighting_controlled=True,
        monitor_warmed_up=True,
        night_light_disabled=True,
        hdr_disabled=True,
    )


def make_calibration(now: datetime | None = None) -> Calibration:
    gamma = GammaCalibration(method="none", lum_min_cdm2=0.5, lum_max_cdm2=150.0)
    color = ColorCalibration(
        method="srgb_assumed",
        red=PrimaryChromaticity(x=0.64, y=0.33, Y_cdm2=30),
        green=PrimaryChromaticity(x=0.3, y=0.6, Y_cdm2=90),
        blue=PrimaryChromaticity(x=0.15, y=0.06, Y_cdm2=10),
        white=PrimaryChromaticity(x=0.3127, y=0.329, Y_cdm2=130),
    )
    return Calibration(
        created_utc=now or datetime.now(timezone.utc),
        geometry=make_display(),
        gamma=gamma,
        color=color,
        environment=make_environment(),
        software_version="0.1.0",
    )


def make_session_plan(
    task_id: str = "dummy_test", eye: str = "OD", participant_id: str = "sub-0001"
) -> SessionPlan:
    return SessionPlan(
        participant_id=participant_id,
        tests=[PlannedTest(task_id=task_id, eye=eye, params={"n": 5}, viewing_distance_cm=57.0)],
        ordering="fixed",
        seed=42,
    )


def make_session_info(
    participant_id: str,
    session_id: str,
    calibration_hash: str,
    status: str = "running",
    **overrides: Any,
) -> SessionInfo:
    kwargs: dict[str, Any] = {
        "session_id": session_id,
        "participant_id": participant_id,
        "plan": make_session_plan(),
        "calibration_hash": calibration_hash,
        "display": make_display(),
        "os_info": "linux-test",
        "python_version": "3.10.0",
        "psychopy_version": "n/a",
        "software_version": "0.1.0",
        "status": status,
        "started_utc": datetime.now(timezone.utc),
        "environment": make_environment(),
    }
    kwargs.update(overrides)
    return SessionInfo(**kwargs)


def make_trial(
    participant_id: str = "sub-0001",
    session_id: str = "ses-20260916T103000",
    task_id: str = "dummy_test",
    run: int = 1,
    eye: str = "OD",
    block: str = "main",
    trial_index: int = 0,
    is_catch: bool = False,
    intensity: float = 0.1,
    correct: bool | None = True,
    **overrides: Any,
) -> TrialRecord:
    kwargs: dict[str, Any] = {
        "participant_id": participant_id,
        "session_id": session_id,
        "task_id": task_id,
        "task_version": "1.0.0",
        "run": run,
        "eye": eye,
        "block": block,
        "trial_index": trial_index,
        "is_catch": is_catch,
        "intensity": intensity,
        "intensity_units": "logMAR",
        "stimulus_params": {"orientation_deg": 90},
        "correct_response": "left",
        "response": "left",
        "correct": correct,
        "rt_s": 0.3,
        "stimulus_onset_s": float(trial_index),
        "n_dropped_frames_trial": 0,
        "procedure_state": {"step": trial_index},
        "timestamp_utc": datetime.now(timezone.utc),
        "rng_seed": 42,
    }
    kwargs.update(overrides)
    return TrialRecord(**kwargs)


class DummyParams(BaseModel):
    n: int = 5


class DummyTest(PsychophysicalTest):
    """A minimal, deterministic `PsychophysicalTest` used only by tests.

    `summarize` is a pure function of its trials DataFrame (mean main-block
    intensity as the "threshold"), so reanalysis is expected to reproduce
    its stored summary exactly.
    """

    spec = TestSpec(
        id="dummy_test",
        name="Dummy Test",
        version="1.0.0",
        domain="acuity",
        description_participant="A dummy test for the test suite.",
        description_technical="A dummy test for the test suite.",
        measures="Mean intensity of main-block trials.",
        output_units="logMAR",
        estimated_minutes=1.0,
        allowed_eyes=["OD", "OS", "OU"],
        requirements=TestRequirements(),
        params_model=DummyParams,
    )

    def __init__(
        self,
        params: BaseModel,
        display: DisplayGeometry,
        calibration: Calibration,
        rng: np.random.Generator,
    ) -> None:
        self.params = params
        self.display = display
        self.calibration = calibration
        self.rng = rng

    def make_procedure(self) -> Any:
        raise NotImplementedError

    def make_catch_trial_intensity(self) -> float:
        return 1.0

    def build_stimuli(self, win: Any) -> dict[str, Any]:
        return {}

    def present(self, win: Any, intensity_or_stimulus: Any, trial_ctx: dict[str, Any]) -> Any:
        raise NotImplementedError

    def response_keys(self) -> list[str]:
        return ["left", "right"]

    def score(self, response: Any, stimulus_params: dict[str, Any]) -> bool:
        return bool(response == stimulus_params.get("correct"))

    def instructions(self) -> str:
        return "Dummy instructions."

    def summarize(self, trials: pd.DataFrame) -> TestSummary:
        main = trials[trials["block"] == "main"]
        catch = trials[trials["is_catch"]]
        n_catch = len(catch)
        catch_lapse = float((~catch["correct"].astype(bool)).mean()) if n_catch else 0.0
        value = float(main["intensity"].mean()) if len(main) else 0.0
        return TestSummary(
            task_id="dummy_test",
            task_version="1.0.0",
            eye=str(trials["eye"].iloc[0]) if len(trials) else "OD",
            run=int(trials["run"].iloc[0]) if len(trials) else 1,
            estimate=ThresholdEstimate(
                value=value,
                ci_low=value - 0.1,
                ci_high=value + 0.1,
                ci_level=0.95,
                units="logMAR",
                method="mean",
            ),
            fit_params={},
            gof={},
            quality_flags=[],
            n_trials=len(main),
            n_catch=n_catch,
            catch_lapse_rate=catch_lapse,
            frame_stats={},
            analysis_version="dummy-1.0.0",
        )


@pytest.fixture
def registered_dummy_test() -> Any:
    """Register `DummyTest` in the `tests_catalog` registry for the duration of one test."""
    if "dummy_test" not in tests_catalog_base._REGISTRY:
        tests_catalog_base.register_test(DummyTest)
    yield DummyTest
    tests_catalog_base._REGISTRY.pop("dummy_test", None)


def build_full_session(
    root: Path,
    n_trials: int = 5,
    n_catch: int = 1,
    participant_id: str | None = None,
    session_id: str = "ses-20260916T103000",
    write_summary: bool = True,
) -> tuple[str, str, Calibration]:
    """Set up a complete dataset/participant/calibration/session on disk, for tests.

    Returns:
        `(participant_id, session_id, calibration)` for the freshly written
        session, finalized as `"complete"`.
    """
    dataset.init_dataset(root)
    if participant_id is None:
        participant = dataset.create_participant(root)
        participant_id = participant.participant_id
    calibration = make_calibration()
    dataset.save_calibration(calibration, root)

    session_info = make_session_info(participant_id, session_id, calibration.content_hash())
    n_main = n_trials - n_catch
    with SessionWriter(participant_id, session_id, session_info, root) as writer:
        for i in range(n_main):
            writer.append_trial(
                make_trial(
                    participant_id=participant_id,
                    session_id=session_id,
                    trial_index=i,
                    intensity=0.1 * i,
                    is_catch=False,
                    correct=True,
                )
            )
        for j in range(n_catch):
            writer.append_trial(
                make_trial(
                    participant_id=participant_id,
                    session_id=session_id,
                    trial_index=n_main + j,
                    intensity=1.0,
                    is_catch=True,
                    correct=True,
                )
            )
        writer.write_frames("dummy_test", "OD", 1, [0.0166, 0.0167, 0.0165])

        if write_summary:
            from vpsych.data import paths as _paths
            from vpsych.data.tsv import read_trials_tsv

            trials_path = _paths.trials_tsv_path(
                participant_id, session_id, "dummy_test", "OD", 1, root
            )
            df = read_trials_tsv(trials_path)
            summary = DummyTest(
                DummyParams(), make_display(), calibration, np.random.default_rng(0)
            ).summarize(df)
            writer.write_summary("dummy_test", "OD", 1, summary)

    return participant_id, session_id, calibration
