"""Unit/integration tests for vpsych.runner.__main__: the session runner, --simulate path.

Runs `run_session` end to end against a temporarily registered dummy test,
an in-memory fake `Writer`, and `--simulate always_correct`, so no display
or real `SessionWriter`/photometer is needed. Also covers session-plan
loading (participant_id/calibration_hash conventions -- see the module
docstring in `vpsych.runner.__main__`), requirements-unmet and error exit
paths, and the SIGINT/SIGTERM abort-signal wiring.
"""

from __future__ import annotations

import json
import signal
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
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
from vpsych.data.schemas import TestSummary
from vpsych.runner import status as status_module
from vpsych.runner.__main__ import (
    RunnerExitCode,
    Writer,
    _install_signal_handlers,
    _SessionAbortedError,
    build_arg_parser,
    load_session_plan,
    run_session,
)
from vpsych.tests_catalog import base as catalog_base
from vpsych.tests_catalog.base import (
    PresentedTrial,
    PsychophysicalTest,
    TestRequirements,
    TestSpec,
)


class _RunnerDummyParams(BaseModel):
    pass


class _RunnerDummyProcedure:
    intensity_units = "log10_contrast"

    def __init__(self, n_trials: int = 5) -> None:
        self.n_trials = n_trials
        self.n_updates = 0
        self.finished = False

    def next_intensity(self) -> float:
        return -1.0

    def update(self, intensity: float, correct: bool) -> None:
        self.n_updates += 1
        if self.n_updates >= self.n_trials:
            self.finished = True

    def estimate(self) -> ThresholdEstimate:
        return ThresholdEstimate(
            value=-1.0,
            ci_low=-1.2,
            ci_high=-0.8,
            ci_level=0.95,
            units=self.intensity_units,
            method="dummy",
        )

    def state_dict(self) -> dict[str, Any]:
        return {"n_updates": self.n_updates}


class _RunnerDummyTest(PsychophysicalTest):
    spec = TestSpec(
        id="runner_dummy_test",
        name="Runner Dummy Test",
        version="0.1.0",
        domain="contrast",
        description_participant="Press left or right.",
        description_technical="A minimal dummy task for runner integration tests.",
        measures="Nothing real.",
        output_units="log10_contrast",
        estimated_minutes=1.0,
        allowed_eyes=["OU"],
        requirements=TestRequirements(),
        citations=[],
        params_model=_RunnerDummyParams,
    )

    def __init__(
        self, params: BaseModel, display: Any, calibration: Any, rng: np.random.Generator
    ) -> None:
        self.params = params
        self.display = display
        self.calibration = calibration
        self.rng = rng
        self._n = 0

    def make_procedure(self) -> _RunnerDummyProcedure:
        return _RunnerDummyProcedure(n_trials=5)

    def make_catch_trial_intensity(self) -> float:
        return 0.0

    def build_stimuli(self, win: Any) -> dict[str, Any]:
        return {}

    def present(
        self, win: Any, intensity_or_stimulus: Any, trial_ctx: dict[str, Any]
    ) -> PresentedTrial:
        self._n += 1
        rng = trial_ctx["rng"]
        correct_side = "left" if rng.random() < 0.5 else "right"
        stimulus_params = {"correct_response": correct_side}
        trial_ctx["stimulus_params"] = stimulus_params
        trial_ctx["correct_response"] = correct_side
        observer = trial_ctx["simulated_observer"]
        response = observer.respond(stimulus_params, rng)
        return PresentedTrial(
            response=response,
            rt_s=0.3,
            stimulus_onset_s=float(self._n),
            n_dropped_frames=0,
            frame_intervals_s=[1 / 60.0] * 5,
        )

    def response_keys(self) -> list[str]:
        return ["left", "right"]

    def score(self, response: Any, stimulus_params: dict[str, Any]) -> bool:
        return bool(response == stimulus_params["correct_response"])

    def instructions(self) -> str:
        return "Press left or right."

    def summarize(self, trials: Any) -> TestSummary:
        return TestSummary(
            task_id=self.spec.id,
            task_version=self.spec.version,
            eye="OU",
            run=1,
            estimate=ThresholdEstimate(
                value=-1.0,
                ci_low=-1.2,
                ci_high=-0.8,
                ci_level=0.95,
                units="log10_contrast",
                method="dummy",
            ),
            n_trials=len(trials),
            n_catch=int(trials["is_catch"].sum()) if len(trials) else 0,
            catch_lapse_rate=0.0,
            analysis_version="0.0.0",
        )


@pytest.fixture
def _registered_dummy_test() -> Any:
    catalog_base._REGISTRY["runner_dummy_test"] = _RunnerDummyTest
    try:
        yield _RunnerDummyTest
    finally:
        catalog_base._REGISTRY.pop("runner_dummy_test", None)


class _FakeRunnerWriter:
    def __init__(self) -> None:
        self.trials: list[TrialRecord] = []
        self.summaries: list[tuple[str, str, int, TestSummary]] = []
        self.frames: list[tuple[str, str, int, list[float]]] = []
        self.finalized_status: str | None = None

    def append_trial(self, trial: TrialRecord) -> None:
        self.trials.append(trial)

    def write_summary(self, task_id: str, eye: str, run: int, summary: TestSummary) -> None:
        self.summaries.append((task_id, eye, run, summary))

    def write_frames(self, task_id: str, eye: str, run: int, intervals_s: list[float]) -> None:
        self.frames.append((task_id, eye, run, list(intervals_s)))

    def finalize(self, status: str) -> None:
        self.finalized_status = status


def _geometry() -> DisplayGeometry:
    return DisplayGeometry(
        width_px=1920,
        height_px=1080,
        width_cm=53.13,
        height_cm=29.88,
        viewing_distance_cm=57.0,
        refresh_hz=60.0,
    )


def _calibration(created_utc: datetime | None = None) -> Calibration:
    return Calibration(
        created_utc=created_utc or datetime(2026, 9, 1, tzinfo=timezone.utc),
        geometry=_geometry(),
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


def _write_calibration(data_root: Path, cal: Calibration) -> str:
    cal_dir = data_root / "calibration"
    cal_dir.mkdir(parents=True, exist_ok=True)
    h = cal.content_hash()
    (cal_dir / f"cal-{h}.json").write_text(cal.model_dump_json(), encoding="utf-8")
    return h


def _write_plan(
    path: Path, *, participant_id: str | None = "sub-0001", calibration_hash: str | None = None
) -> None:
    raw: dict[str, Any] = {
        "tests": [
            {"task_id": "runner_dummy_test", "eye": "OU", "params": {}, "viewing_distance_cm": 57.0}
        ],
        "ordering": "fixed",
        "seed": 123,
    }
    if participant_id is not None:
        raw["participant_id"] = participant_id
    if calibration_hash is not None:
        raw["calibration_hash"] = calibration_hash
    path.write_text(json.dumps(raw), encoding="utf-8")


def _args(
    tmp_path: Path, plan_path: Path, data_root: Path, simulate: str | None = "always_correct"
) -> Any:
    parser = build_arg_parser()
    argv = [
        "--session-plan",
        str(plan_path),
        "--status-file",
        str(tmp_path / "status.json"),
        "--data-root",
        str(data_root),
    ]
    if simulate is not None:
        argv += ["--simulate", simulate]
    return parser.parse_args(argv)


def test_load_session_plan_requires_participant_id(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    _write_plan(plan_path, participant_id=None)
    with pytest.raises(ValueError, match="participant_id"):
        load_session_plan(plan_path, tmp_path)


def test_load_session_plan_missing_calibration_hash_raises(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    _write_plan(plan_path, calibration_hash="deadbeef" * 8)
    with pytest.raises(FileNotFoundError):
        load_session_plan(plan_path, tmp_path)


def test_load_session_plan_picks_most_recent_calibration(tmp_path: Path) -> None:
    older = _calibration(datetime(2026, 1, 1, tzinfo=timezone.utc))
    newer = _calibration(datetime(2026, 6, 1, tzinfo=timezone.utc))
    _write_calibration(tmp_path, older)
    newer_hash = _write_calibration(tmp_path, newer)

    plan_path = tmp_path / "plan.json"
    _write_plan(plan_path)
    _plan, participant_id, calibration = load_session_plan(plan_path, tmp_path)
    assert participant_id == "sub-0001"
    assert calibration is not None
    assert calibration.content_hash() == newer_hash


def test_load_session_plan_no_calibration_returns_none(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    _write_plan(plan_path)
    _plan, _pid, calibration = load_session_plan(plan_path, tmp_path)
    assert calibration is None


def test_load_session_plan_explicit_hash_used(tmp_path: Path) -> None:
    cal = _calibration()
    h = _write_calibration(tmp_path, cal)
    plan_path = tmp_path / "plan.json"
    _write_plan(plan_path, calibration_hash=h)
    _plan, _pid, calibration = load_session_plan(plan_path, tmp_path)
    assert calibration is not None
    assert calibration.content_hash() == h


def test_run_session_no_calibration_is_requirements_unmet(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    _write_plan(plan_path)
    data_root = tmp_path / "data"
    args = _args(tmp_path, plan_path, data_root)
    exit_code = run_session(args, writer_factory=lambda *a: _FakeRunnerWriter())
    assert exit_code == RunnerExitCode.REQUIREMENTS_UNMET


def test_run_session_unregistered_test_is_requirements_unmet(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_calibration(data_root, _calibration())
    plan_path = tmp_path / "plan.json"
    _write_plan(plan_path)  # references runner_dummy_test, NOT registered in this test
    args = _args(tmp_path, plan_path, data_root)
    exit_code = run_session(args, writer_factory=lambda *a: _FakeRunnerWriter())
    assert exit_code == RunnerExitCode.REQUIREMENTS_UNMET


def test_run_session_happy_path_simulated(tmp_path: Path, _registered_dummy_test: Any) -> None:
    data_root = tmp_path / "data"
    _write_calibration(data_root, _calibration())
    plan_path = tmp_path / "plan.json"
    _write_plan(plan_path)
    args = _args(tmp_path, plan_path, data_root)

    fake_writer = _FakeRunnerWriter()
    exit_code = run_session(args, writer_factory=lambda *a: fake_writer)

    assert exit_code == RunnerExitCode.OK
    assert fake_writer.finalized_status == "complete"
    assert len(fake_writer.trials) > 0
    assert len(fake_writer.summaries) == 1
    assert len(fake_writer.frames) == 1

    final_status = status_module.RunnerStatus.read(tmp_path / "status.json")
    assert final_status.state == "finished"


def test_run_session_writer_exception_is_caught(
    tmp_path: Path, _registered_dummy_test: Any
) -> None:
    data_root = tmp_path / "data"
    _write_calibration(data_root, _calibration())
    plan_path = tmp_path / "plan.json"
    _write_plan(plan_path)
    args = _args(tmp_path, plan_path, data_root)

    def _broken_writer_factory(*_a: Any) -> Any:
        raise RuntimeError("writer not implemented yet")

    exit_code = run_session(args, writer_factory=_broken_writer_factory)
    assert exit_code == RunnerExitCode.ERROR
    final_status = status_module.RunnerStatus.read(tmp_path / "status.json")
    assert final_status.state == "error"
    assert final_status.error is not None


def test_signal_handler_raises_session_aborted() -> None:
    restore = _install_signal_handlers()
    try:
        handler = signal.getsignal(signal.SIGINT)
        with pytest.raises(_SessionAbortedError):
            handler(signal.SIGINT, None)  # type: ignore[misc]
    finally:
        restore()


def test_writer_protocol_satisfied_by_fake() -> None:
    assert isinstance(_FakeRunnerWriter(), Writer)


def test_exit_code_int_values_unchanged() -> None:
    assert int(RunnerExitCode.OK) == 0
    assert int(RunnerExitCode.ERROR) == 1
    assert int(RunnerExitCode.ABORTED_BY_USER) == 2
    assert int(RunnerExitCode.REFRESH_MISMATCH) == 3
    assert int(RunnerExitCode.REQUIREMENTS_UNMET) == 4
