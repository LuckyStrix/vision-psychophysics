"""Unit/integration tests for vpsych.runner.__main__: the session runner, --simulate path.

Runs `run_session` end to end against a temporarily registered dummy test,
an in-memory fake `Writer`, and `--simulate always_correct`, so no display
or real `SessionWriter`/photometer is needed. Also covers session-plan
loading (`SessionPlan.participant_id`/`calibration_hash` -- see the module
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
from pydantic import BaseModel, ValidationError

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
    build_simulated_observer,
    load_session_plan,
    load_simulate_config,
    parse_simulated_observer_spec,
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
    with pytest.raises(ValidationError, match="participant_id"):
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


def test_parse_simulated_observer_spec_always_correct() -> None:
    from vpsych.runner.__main__ import _AlwaysCorrectObserver

    obs = parse_simulated_observer_spec("always_correct")
    assert isinstance(obs, _AlwaysCorrectObserver)


def test_parse_simulated_observer_spec_unknown_builtin_raises() -> None:
    with pytest.raises(ValueError, match="Unknown simulated observer"):
        parse_simulated_observer_spec("not_a_real_observer")


def test_parse_simulated_observer_spec_psychometric() -> None:
    from vpsych.core.observers import PsychometricObserver

    obs = parse_simulated_observer_spec("psychometric:threshold=-1.0,slope=3.5,lapse=0.02")
    assert isinstance(obs, PsychometricObserver)
    assert obs.true_function.threshold == pytest.approx(-1.0)
    assert obs.true_function.slope == pytest.approx(3.5)
    assert obs.true_function.lapse == pytest.approx(0.02)
    assert obs.true_function.guess == pytest.approx(0.5)  # default
    assert obs.n_afc == 2  # default


def test_parse_simulated_observer_spec_psychometric_missing_required_raises() -> None:
    with pytest.raises(ValueError, match="threshold"):
        parse_simulated_observer_spec("psychometric:slope=3.5")


def test_parse_simulated_observer_spec_csf() -> None:
    from vpsych.core.observers import CSFObserver

    obs = parse_simulated_observer_spec(
        "csf:peak_gain=1.6,peak_freq=3.0,bandwidth=3.0,low_freq_truncation=1.0"
    )
    assert isinstance(obs, CSFObserver)
    assert obs.peak_gain_log10 == pytest.approx(1.6)
    assert obs.peak_freq_cpd == pytest.approx(3.0)
    assert obs.bandwidth_octaves == pytest.approx(3.0)
    assert obs.low_freq_truncation_log10 == pytest.approx(1.0)


def test_parse_simulated_observer_spec_csf_missing_required_raises() -> None:
    with pytest.raises(ValueError, match="missing required"):
        parse_simulated_observer_spec("csf:peak_gain=1.6")


def test_build_simulated_observer_unknown_kind_raises() -> None:
    with pytest.raises(ValueError, match="Unknown simulated observer kind"):
        build_simulated_observer("bogus", {})


def test_register_simulated_observer_kind_extends_dispatch() -> None:
    """A registered kind (see color_discrimination.observer's "trivector" for a real example)
    becomes resolvable through build_simulated_observer/parse_simulated_observer_spec like a
    built-in kind, without modifying vpsych.runner.__main__ itself."""
    from vpsych.runner.__main__ import _AlwaysCorrectObserver, register_simulated_observer_kind

    calls: list[dict[str, float]] = []

    def _builder(params: dict[str, float]) -> Any:
        calls.append(params)
        return _AlwaysCorrectObserver()

    register_simulated_observer_kind("test_only_kind_xyz", _builder)
    try:
        obs = build_simulated_observer("test_only_kind_xyz", {"a": 1.0})
        assert isinstance(obs, _AlwaysCorrectObserver)
        assert calls == [{"a": 1.0}]

        obs2 = parse_simulated_observer_spec("test_only_kind_xyz:a=2.0")
        assert isinstance(obs2, _AlwaysCorrectObserver)
        assert calls[-1] == {"a": 2.0}

        with pytest.raises(ValueError, match="already registered"):
            register_simulated_observer_kind("test_only_kind_xyz", _builder)
        with pytest.raises(ValueError, match="built-in kind"):
            register_simulated_observer_kind("psychometric", _builder)
    finally:
        from vpsych.runner import __main__ as _runner_main

        del _runner_main._REGISTERED_SIMULATED_OBSERVER_KINDS["test_only_kind_xyz"]


def test_parse_simulated_observer_spec_malformed_pair_raises() -> None:
    with pytest.raises(ValueError, match="Malformed"):
        parse_simulated_observer_spec("psychometric:threshold")


def test_load_simulate_config_string_and_object_entries(tmp_path: Path) -> None:
    from vpsych.core.observers import CSFObserver, PsychometricObserver

    config_path = tmp_path / "simulate_config.json"
    config_path.write_text(
        json.dumps(
            {
                "acuity": "psychometric:threshold=-1.0,slope=3.5,lapse=0.02",
                "csf_task": {
                    "kind": "csf",
                    "peak_gain": 1.6,
                    "peak_freq": 3.0,
                    "bandwidth": 3.0,
                    "low_freq_truncation": 1.0,
                },
            }
        ),
        encoding="utf-8",
    )
    observers = load_simulate_config(config_path)
    assert set(observers) == {"acuity", "csf_task"}
    assert isinstance(observers["acuity"], PsychometricObserver)
    assert isinstance(observers["csf_task"], CSFObserver)


def test_load_simulate_config_requires_json_object(tmp_path: Path) -> None:
    config_path = tmp_path / "bad_config.json"
    config_path.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        load_simulate_config(config_path)


class _PsychometricDummyProcedure:
    """Presents a fixed intensity for many trials -- enough to statistically tell apart
    two different --simulate-config observers by their resulting main-block correct rate."""

    intensity_units = "log10_contrast"

    def __init__(self, n_trials: int = 120, fixed_intensity: float = -1.0) -> None:
        self.n_trials = n_trials
        self.fixed_intensity = fixed_intensity
        self.n_updates = 0
        self.finished = False

    def next_intensity(self) -> float:
        return self.fixed_intensity

    def update(self, intensity: float, correct: bool) -> None:
        self.n_updates += 1
        if self.n_updates >= self.n_trials:
            self.finished = True

    def estimate(self) -> ThresholdEstimate:
        return ThresholdEstimate(
            value=self.fixed_intensity,
            ci_low=self.fixed_intensity - 0.1,
            ci_high=self.fixed_intensity + 0.1,
            ci_level=0.95,
            units=self.intensity_units,
            method="dummy",
        )

    def state_dict(self) -> dict[str, Any]:
        return {"n_updates": self.n_updates}


def _psychometric_dummy_spec(task_id: str) -> TestSpec:
    return TestSpec(
        id=task_id,
        name="Runner Psychometric Dummy Test",
        version="0.1.0",
        domain="contrast",
        description_participant="n/a",
        description_technical="Dummy task exercising --simulate psychometric:/csf: specs.",
        measures="Nothing real.",
        output_units="log10_contrast",
        estimated_minutes=1.0,
        allowed_eyes=["OU"],
        requirements=TestRequirements(),
        citations=[],
        params_model=_RunnerDummyParams,
    )


class _RunnerPsychometricDummyTest(PsychophysicalTest):
    """Demonstrates the decoupled decide_correct + simulated_response response-mapping
    path (see vpsych.core.trial_loop's module docstring), robust to any --simulate
    observer kind (always_correct, psychometric, csf), unlike _RunnerDummyTest above
    (which only works with always_correct). `spec` is set per-subclass (below) since
    `TestSpec.id` -- what TrialRecord.task_id actually records -- is a class attribute,
    not derived from the registry key a test happens to be registered under."""

    def __init__(
        self, params: BaseModel, display: Any, calibration: Any, rng: np.random.Generator
    ) -> None:
        self.params = params
        self._n = 0

    def make_procedure(self) -> _PsychometricDummyProcedure:
        return _PsychometricDummyProcedure()

    def make_catch_trial_intensity(self) -> float:
        return 5.0  # suprathreshold: far above any threshold used in these tests

    def build_stimuli(self, win: Any) -> dict[str, Any]:
        return {}

    def present(
        self, win: Any, intensity_or_stimulus: Any, trial_ctx: dict[str, Any]
    ) -> PresentedTrial:
        self._n += 1
        rng = trial_ctx["rng"]
        intensity = float(intensity_or_stimulus)
        correct_side = "left" if rng.random() < 0.5 else "right"
        stimulus_params = {
            "correct_response": correct_side,
            "intensity": intensity,
            "spatial_frequency_cpd": 3.0,
            "contrast": 10.0**intensity,
        }
        trial_ctx["stimulus_params"] = stimulus_params
        trial_ctx["correct_response"] = correct_side
        observer = trial_ctx["simulated_observer"]
        is_correct = observer.decide_correct(stimulus_params, rng)
        response = self.simulated_response(is_correct, stimulus_params, rng)
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
        return "n/a"

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


class _RunnerPsyEasyTest(_RunnerPsychometricDummyTest):
    spec = _psychometric_dummy_spec("runner_psy_easy")


class _RunnerPsyHardTest(_RunnerPsychometricDummyTest):
    spec = _psychometric_dummy_spec("runner_psy_hard")


def test_run_session_simulate_config_dispatches_per_task_observer(tmp_path: Path) -> None:
    """Two tasks, two different --simulate-config psychometric specs: the easy task
    (threshold matches the presented intensity exactly) should score correct far more
    often than the hard task (threshold very far from the presented intensity), proving
    the runner is actually using a different, per-task observer for each -- not the
    same one for the whole session."""
    catalog_base._REGISTRY["runner_psy_easy"] = _RunnerPsyEasyTest
    catalog_base._REGISTRY["runner_psy_hard"] = _RunnerPsyHardTest
    try:
        data_root = tmp_path / "data"
        _write_calibration(data_root, _calibration())
        plan_path = tmp_path / "plan.json"
        raw = {
            "participant_id": "sub-0001",
            "tests": [
                {
                    "task_id": "runner_psy_easy",
                    "eye": "OU",
                    "params": {},
                    "viewing_distance_cm": 57.0,
                },
                {
                    "task_id": "runner_psy_hard",
                    "eye": "OU",
                    "params": {},
                    "viewing_distance_cm": 57.0,
                },
            ],
            "ordering": "fixed",
            "seed": 123,
        }
        plan_path.write_text(json.dumps(raw), encoding="utf-8")

        config_path = tmp_path / "simulate_config.json"
        config_path.write_text(
            json.dumps(
                {
                    # Threshold exactly at the fixed presented intensity (-1.0): easy.
                    "runner_psy_easy": "psychometric:threshold=-1.0,slope=3.5,lapse=0.02",
                    # Threshold far above the presented intensity: near chance.
                    "runner_psy_hard": "psychometric:threshold=5.0,slope=3.5,lapse=0.02",
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
                "--simulate-config",
                str(config_path),
            ]
        )
        fake_writer = _FakeRunnerWriter()
        exit_code = run_session(args, writer_factory=lambda *a: fake_writer)
        assert exit_code == RunnerExitCode.OK

        by_task: dict[str, list[bool]] = {"runner_psy_easy": [], "runner_psy_hard": []}
        for trial in fake_writer.trials:
            if trial.block == "main" and not trial.is_catch:
                by_task[trial.task_id].append(bool(trial.correct))

        easy_rate = sum(by_task["runner_psy_easy"]) / len(by_task["runner_psy_easy"])
        hard_rate = sum(by_task["runner_psy_hard"]) / len(by_task["runner_psy_hard"])
        assert easy_rate > 0.65  # ~0.74 expected (guess=0.5, lapse=0.02, at threshold)
        assert hard_rate < 0.65  # ~0.5 expected (near chance, threshold far from intensity)
        assert easy_rate - hard_rate > 0.1
    finally:
        catalog_base._REGISTRY.pop("runner_psy_easy", None)
        catalog_base._REGISTRY.pop("runner_psy_hard", None)


def test_run_session_simulate_config_missing_task_without_default_errors(tmp_path: Path) -> None:
    catalog_base._REGISTRY["runner_psy_easy"] = _RunnerPsyEasyTest
    try:
        data_root = tmp_path / "data"
        _write_calibration(data_root, _calibration())
        plan_path = tmp_path / "plan.json"
        raw = {
            "participant_id": "sub-0001",
            "tests": [
                {
                    "task_id": "runner_psy_easy",
                    "eye": "OU",
                    "params": {},
                    "viewing_distance_cm": 57.0,
                }
            ],
            "ordering": "fixed",
            "seed": 1,
        }
        plan_path.write_text(json.dumps(raw), encoding="utf-8")
        # An empty config covers no tasks, and no --simulate default is given.
        config_path = tmp_path / "simulate_config.json"
        config_path.write_text(json.dumps({}), encoding="utf-8")

        parser = build_arg_parser()
        args = parser.parse_args(
            [
                "--session-plan",
                str(plan_path),
                "--status-file",
                str(tmp_path / "status.json"),
                "--data-root",
                str(data_root),
                "--simulate-config",
                str(config_path),
            ]
        )
        exit_code = run_session(args, writer_factory=lambda *a: _FakeRunnerWriter())
        assert exit_code == RunnerExitCode.ERROR
        status = status_module.RunnerStatus.read(tmp_path / "status.json")
        assert status.state == "error"
    finally:
        catalog_base._REGISTRY.pop("runner_psy_easy", None)


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
