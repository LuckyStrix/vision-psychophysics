"""Unit tests for vpsych.core.trial_loop: the generic trial protocol, headless via SimulatedBackend.

Uses a minimal dummy `PsychophysicalTest`, dummy `AdaptiveProcedure`/
`MultiParamProcedure` implementations, and an in-memory fake writer, so the
trial loop's own logic (protocol ordering, catch-trial placement, practice
not updating the procedure, crash-safe writes, abort handling, status
updates) is fully exercised with no display.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import BaseModel

from vpsych.core.procedures.base import MultiParamProcedure, ThresholdEstimate
from vpsych.core.trial import TrialRecord, TrialTimeline
from vpsych.core.trial_loop import (
    SimulatedBackend,
    TrialLoop,
    TrialLoopConfig,
    TrialWriter,
    _scalar_intensity,
)
from vpsych.runner.status import RunnerStatus
from vpsych.tests_catalog.base import (
    PresentedTrial,
    PsychophysicalTest,
    TestRequirements,
    TestSpec,
)


class _DummyParams(BaseModel):
    pass


class _DummyTest(PsychophysicalTest):
    spec = TestSpec(
        id="dummy_test",
        name="Dummy Test",
        version="0.1.0",
        domain="contrast",
        description_participant="Press left or right.",
        description_technical="A minimal 2AFC dummy task for trial-loop tests.",
        measures="Nothing real.",
        output_units="log10_contrast_from_spec",
        estimated_minutes=1.0,
        allowed_eyes=["OU"],
        requirements=TestRequirements(),
        citations=[],
        params_model=_DummyParams,
    )

    def __init__(
        self, params: BaseModel, display: Any, calibration: Any, rng: np.random.Generator
    ) -> None:
        self.params = params
        self.display = display
        self.calibration = calibration
        self.rng = rng
        self._n_present = 0

    def make_procedure(self) -> Any:
        raise NotImplementedError("tests construct the procedure directly")

    def make_catch_trial_intensity(self) -> float:
        return 0.0

    def build_stimuli(self, win: Any) -> dict[str, Any]:
        return {}

    def present(
        self, win: Any, intensity_or_stimulus: float | dict[str, float], trial_ctx: dict[str, Any]
    ) -> PresentedTrial:
        self._n_present += 1
        rng = trial_ctx["rng"]
        correct_side = "left" if rng.random() < 0.5 else "right"
        if isinstance(intensity_or_stimulus, int | float):
            stimulus_params: dict[str, Any] = {"intensity": intensity_or_stimulus}
        else:
            stimulus_params = dict(intensity_or_stimulus)
        stimulus_params["correct_response"] = correct_side
        trial_ctx["stimulus_params"] = stimulus_params
        trial_ctx["correct_response"] = correct_side

        observer = trial_ctx.get("simulated_observer")
        assert observer is not None, "_DummyTest only supports the SimulatedBackend path"
        response = observer.respond(stimulus_params, rng)
        return PresentedTrial(
            response=response,
            rt_s=0.4,
            stimulus_onset_s=float(self._n_present),
            n_dropped_frames=0,
            frame_intervals_s=[1 / 60.0] * 5,
        )

    def response_keys(self) -> list[str]:
        return ["left", "right"]

    def score(self, response: Any, stimulus_params: dict[str, Any]) -> bool:
        return bool(response == stimulus_params["correct_response"])

    def instructions(self) -> str:
        return "Press left or right."

    def summarize(self, trials: Any) -> Any:
        raise NotImplementedError


class _AlwaysCorrectObserver:
    def respond(self, stimulus: dict[str, Any], rng: np.random.Generator) -> Any:
        del rng
        return stimulus["correct_response"]


class _ProbabilisticObserver:
    def __init__(self, p_correct: float) -> None:
        self.p_correct = p_correct

    def respond(self, stimulus: dict[str, Any], rng: np.random.Generator) -> Any:
        correct = stimulus["correct_response"]
        if rng.random() < self.p_correct:
            return correct
        return "left" if correct == "right" else "right"


class _DummyAdaptiveProcedure:
    intensity_units = "log10_contrast_from_proc"

    def __init__(self, n_trials: int) -> None:
        self.n_trials = n_trials
        self.n_updates = 0
        self.finished = False
        self.history: list[tuple[float, bool]] = []

    def next_intensity(self) -> float:
        return -1.0 + 0.01 * self.n_updates

    def update(self, intensity: float, correct: bool) -> None:
        self.n_updates += 1
        self.history.append((intensity, correct))
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


class _DummyMultiParamProcedure(MultiParamProcedure):
    def __init__(self, n_trials: int) -> None:
        self.n_trials = n_trials
        self.n_updates = 0

    def next_stimulus(self) -> dict[str, float]:
        return {"intensity": -1.0, "spatial_frequency_cpd": 2.0}

    def update(self, stimulus: dict[str, float], correct: bool) -> None:
        self.n_updates += 1

    @property
    def finished(self) -> bool:
        return self.n_updates >= self.n_trials

    def estimate(self) -> ThresholdEstimate:
        return ThresholdEstimate(
            value=-1.0,
            ci_low=-1.2,
            ci_high=-0.8,
            ci_level=0.95,
            units="log10_contrast",
            method="dummy",
        )

    def state_dict(self) -> dict[str, Any]:
        return {"n_updates": self.n_updates}


class _FakeWriter:
    def __init__(self) -> None:
        self.trials: list[TrialRecord] = []
        self.frames: list[tuple[str, str, int, list[float]]] = []

    def append_trial(self, trial: TrialRecord) -> None:
        self.trials.append(trial)

    def write_frames(self, task_id: str, eye: str, run: int, intervals_s: list[float]) -> None:
        self.frames.append((task_id, eye, run, list(intervals_s)))


def _timeline() -> TrialTimeline:
    return TrialTimeline(fixation_frames=10, stimulus_frames=5, iti_frames=10)


def _make_loop(
    procedure: Any,
    *,
    backend: SimulatedBackend | None = None,
    writer: _FakeWriter | None = None,
    config: TrialLoopConfig | None = None,
    status_path: Path | None = None,
) -> tuple[TrialLoop, _FakeWriter]:
    test = _DummyTest(_DummyParams(), display=None, calibration=None, rng=np.random.default_rng(0))
    writer = writer if writer is not None else _FakeWriter()
    backend = backend if backend is not None else SimulatedBackend(_AlwaysCorrectObserver())
    loop = TrialLoop(
        test,
        procedure,
        backend,
        writer,
        participant_id="sub-0001",
        session_id="ses-20260916T103000",
        eye="OU",
        run_number=1,
        timeline=_timeline(),
        rng=np.random.default_rng(123),
        rng_seed=123,
        config=config,
        status_path=status_path,
    )
    return loop, writer


def test_basic_run_counts_and_blocks() -> None:
    procedure = _DummyAdaptiveProcedure(n_trials=20)
    config = TrialLoopConfig(n_practice_trials=3, catch_trial_probability=0.10)
    loop, _writer = _make_loop(procedure, config=config)
    result = loop.run()

    assert not result.aborted
    practice = [r for r in result.trial_records if r.block == "practice"]
    main = [r for r in result.trial_records if r.block == "main"]
    assert len(practice) == 3
    assert len(main) == result.n_catch + result.n_main_scored
    assert result.n_main_scored == 20
    assert procedure.n_updates == 20
    # Practice trials must NOT have updated the procedure.
    assert procedure.n_updates == result.n_main_scored


def test_writer_receives_every_trial_in_order() -> None:
    procedure = _DummyAdaptiveProcedure(n_trials=10)
    loop, writer = _make_loop(procedure, config=TrialLoopConfig(n_practice_trials=2))
    result = loop.run()
    assert writer.trials == result.trial_records
    assert len(writer.frames) == 1
    task_id, eye, run, intervals = writer.frames[0]
    assert task_id == "dummy_test"
    assert eye == "OU"
    assert run == 1
    assert len(intervals) > 0


def test_catch_trials_not_in_first_n_and_not_adjacent() -> None:
    procedure = _DummyAdaptiveProcedure(n_trials=200)
    config = TrialLoopConfig(
        n_practice_trials=0, catch_trial_probability=0.10, min_main_trials_before_catch=3
    )
    loop, _writer = _make_loop(procedure, config=config)
    result = loop.run()

    main = [r for r in result.trial_records if r.block == "main"]
    for r in main[:3]:
        assert not r.is_catch
    for prev, cur in itertools.pairwise(main):
        if prev.is_catch:
            assert not cur.is_catch


def test_catch_trial_fraction_is_approximately_target() -> None:
    procedure = _DummyAdaptiveProcedure(n_trials=1000)
    config = TrialLoopConfig(n_practice_trials=0, catch_trial_probability=0.10)
    loop, _writer = _make_loop(procedure, config=config)
    result = loop.run()
    total_main = result.n_catch + result.n_main_scored
    fraction = result.n_catch / total_main
    assert 0.03 < fraction < 0.15  # loose bound: adjacency/first-3 restrictions suppress it a bit


def test_catch_trials_use_make_catch_trial_intensity() -> None:
    procedure = _DummyAdaptiveProcedure(n_trials=200)
    config = TrialLoopConfig(n_practice_trials=0, catch_trial_probability=0.15)
    loop, _writer = _make_loop(procedure, config=config)
    result = loop.run()
    catch_records = [r for r in result.trial_records if r.is_catch]
    assert catch_records  # sanity: some catches happened
    assert all(r.intensity == 0.0 for r in catch_records)


def test_catch_trials_do_not_call_update() -> None:
    procedure = _DummyAdaptiveProcedure(n_trials=50)
    config = TrialLoopConfig(n_practice_trials=0, catch_trial_probability=0.20)
    loop, _writer = _make_loop(procedure, config=config)
    result = loop.run()
    # n_updates only counts non-catch main trials; if catch trials updated the
    # procedure this would exceed result.n_main_scored.
    assert procedure.n_updates == result.n_main_scored


def test_intensity_units_prefers_procedure_over_spec() -> None:
    procedure = _DummyAdaptiveProcedure(n_trials=3)
    loop, _writer = _make_loop(procedure, config=TrialLoopConfig(n_practice_trials=0))
    result = loop.run()
    assert all(r.intensity_units == "log10_contrast_from_proc" for r in result.trial_records)


def test_intensity_units_falls_back_to_spec_for_multi_param() -> None:
    procedure = _DummyMultiParamProcedure(n_trials=3)
    loop, _writer = _make_loop(procedure, config=TrialLoopConfig(n_practice_trials=0))
    result = loop.run()
    assert all(r.intensity_units == "log10_contrast_from_spec" for r in result.trial_records)


def test_multi_param_procedure_extracts_scalar_intensity() -> None:
    procedure = _DummyMultiParamProcedure(n_trials=5)
    loop, _writer = _make_loop(procedure, config=TrialLoopConfig(n_practice_trials=0))
    result = loop.run()
    non_catch = [r for r in result.trial_records if not r.is_catch]
    assert all(r.intensity == -1.0 for r in non_catch)
    assert all(r.stimulus_params["spatial_frequency_cpd"] == 2.0 for r in non_catch)


def test_scalar_intensity_requires_intensity_key() -> None:
    with pytest.raises(ValueError, match="intensity"):
        _scalar_intensity({"spatial_frequency_cpd": 2.0}, "units")


def test_scalar_intensity_passthrough_for_float() -> None:
    assert _scalar_intensity(-1.5, "units") == -1.5


def test_abort_path_stops_early_and_marks_aborted() -> None:
    backend = SimulatedBackend(_AlwaysCorrectObserver(), abort_after_trials=5)
    procedure = _DummyAdaptiveProcedure(n_trials=100)
    loop, writer = _make_loop(
        procedure, backend=backend, config=TrialLoopConfig(n_practice_trials=3)
    )
    result = loop.run()
    assert result.aborted
    assert len(result.trial_records) < 100
    # Everything presented before the abort must still have been written.
    assert writer.trials == result.trial_records


def test_status_file_written(tmp_path: Path) -> None:
    procedure = _DummyAdaptiveProcedure(n_trials=5)
    status_path = tmp_path / "status.json"
    loop, _writer = _make_loop(
        procedure, config=TrialLoopConfig(n_practice_trials=1), status_path=status_path
    )
    result = loop.run()
    assert not result.aborted
    status = RunnerStatus.read(status_path)
    assert status.state == "finished"
    assert status.task_id == "dummy_test"


def test_trial_writer_protocol_satisfied_by_fake_writer() -> None:
    writer = _FakeWriter()
    assert isinstance(writer, TrialWriter)


def test_max_main_trials_safety_raises_for_procedure_that_never_finishes() -> None:
    class _NeverFinishes(_DummyAdaptiveProcedure):
        def update(self, intensity: float, correct: bool) -> None:
            self.n_updates += 1
            # never sets self.finished = True

    procedure = _NeverFinishes(n_trials=10**9)
    config = TrialLoopConfig(n_practice_trials=0, max_main_trials_safety=25)
    loop, _writer = _make_loop(procedure, config=config)
    with pytest.raises(RuntimeError, match="max_main_trials_safety"):
        loop.run()


def test_demo_trial_not_recorded() -> None:
    procedure = _DummyAdaptiveProcedure(n_trials=1)
    loop, writer = _make_loop(procedure, config=TrialLoopConfig(n_practice_trials=0))
    result = loop.run()
    # 1 main trial expected, demo trial must not appear in records/writer.
    assert len(result.trial_records) == 1
    assert len(writer.trials) == 1


def test_break_message_triggered(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    backend = SimulatedBackend(_AlwaysCorrectObserver())
    monkeypatch.setattr(backend, "show_message", lambda text: calls.append(text))
    procedure = _DummyAdaptiveProcedure(n_trials=10)
    config = TrialLoopConfig(n_practice_trials=0, break_every_n_trials=4)
    loop, _writer = _make_loop(procedure, backend=backend, config=config)
    loop.run()
    assert any("break" in c.lower() for c in calls)
