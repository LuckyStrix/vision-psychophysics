"""The generic trial loop that runs one test's full protocol.

Every test follows the same protocol (see the plan's "Trial protocol"
section and :mod:`vpsych.core.trial`): an instruction screen, a demo trial,
a practice block with feedback (not analyzed), a main block with ~10%
suprathreshold catch trials, self-paced breaks, then done. :class:`TrialLoop`
implements that protocol once, generically, driving any
:class:`~vpsych.tests_catalog.base.PsychophysicalTest` and any
:class:`~vpsych.core.procedures.base.AdaptiveProcedure` or
:class:`~vpsych.core.procedures.base.MultiParamProcedure` through it, and
persisting every trial via a writer *before* the next trial starts (the
project's no-silent-data-loss rule).

The same loop drives two interchangeable "presentation backends" (see
:class:`PresentationBackend`): a real :class:`PsychoPyBackend` (fullscreen
window, hardware-timestamped keyboard) and a headless
:class:`SimulatedBackend` (a :class:`~vpsych.core.observers.SimulatedObserver`
stands in for the human, with fabricated perfect frame timing) -- so this
module's own logic, and any test's trial protocol, is fully exercisable in
CI with no display.

## The ``trial_ctx`` contract

`vpsych.tests_catalog.base.PsychophysicalTest.present` is frozen with the
signature ``present(self, win, intensity_or_stimulus, trial_ctx)``, and its
docstring says trial_ctx's *keys are defined by the trial loop
implementation* -- this module is that implementation, so it defines the
contract below. A test's ``present()`` should treat ``trial_ctx`` as
input **and** output:

Input keys (set by :class:`TrialLoop` before calling ``present``):

- ``"timeline"`` (:class:`~vpsych.core.trial.TrialTimeline`): frame-count
  durations for this trial's phases.
- ``"rng"`` (:class:`numpy.random.Generator`): seeded RNG for this trial's
  stochastic stimulus choices (e.g. which side the target appears on).
- ``"block"`` (``"practice"`` or ``"main"``), ``"is_catch"`` (`bool`),
  ``"trial_index"`` (`int`): bookkeeping, for tests that vary behavior by
  block (e.g. practice-only cues).
- ``"keyboard"``: a live ``psychopy.hardware.keyboard.Keyboard`` when
  driven by :class:`PsychoPyBackend`, or `None` when driven by
  :class:`SimulatedBackend`.
- ``"simulated_observer"``: a
  :class:`~vpsych.core.observers.SimulatedObserver` when driven by
  :class:`SimulatedBackend`, or `None` for :class:`PsychoPyBackend`. A
  test's ``present()`` should, when this is not `None`, skip drawing to
  ``win`` (which is `None` in that case -- see
  :attr:`PresentationBackend.win`) entirely and instead call
  ``simulated_observer.respond(stimulus_params, rng)`` to obtain a
  simulated response, fabricating a plausible ``stimulus_onset_s`` and
  perfect ``frame_intervals_s`` (see :class:`SimulatedBackend` for a
  worked example, used by this module's own tests via a minimal dummy
  test).

Output keys (a test's ``present()`` must set before returning, since
:class:`~vpsych.tests_catalog.base.PresentedTrial` itself does not carry
them, but :class:`~vpsych.core.trial.TrialRecord` needs them):

- ``"stimulus_params"`` (`dict[str, Any]`): the full stimulus
  parameterization actually presented this trial.
- ``"correct_response"`` (`Any`): the response that would have been scored
  correct for this trial's stimulus.

:class:`TrialLoop` reads these back out of ``trial_ctx`` after ``present()``
returns (defaulting to ``{}``/`None` if a test does not set them, so a
minimal/dummy test still runs, at the cost of an uninformative
``stimulus_params``/``correct_response`` in the resulting record).

For a :class:`~vpsych.core.procedures.base.MultiParamProcedure`-driven test
(e.g. qCSF), ``intensity_or_stimulus`` passed to ``present()`` is a
``dict[str, float]`` (from ``next_stimulus()``), but
:class:`~vpsych.core.trial.TrialRecord.intensity` is always a scalar. This
module additionally requires such a test to include a ``"intensity"`` key
in that stimulus dict (the chosen scalar summary dimension, e.g. contrast
for qCSF) -- see :func:`_scalar_intensity`. This is a convention this
module introduces to close a gap in the frozen
:class:`~vpsych.tests_catalog.base.PsychophysicalTest` interface (which
does not otherwise specify how a multi-parameter stimulus reduces to
`TrialRecord`'s single scalar ``intensity`` column); it should be treated
as provisional until confirmed against the first real
:class:`MultiParamProcedure` test implementation (qCSF, Phase 2).
"""

from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from vpsych.core.observers import SimulatedObserver
from vpsych.core.procedures.base import AdaptiveProcedure, MultiParamProcedure
from vpsych.core.trial import Block, Eye, TrialRecord, TrialTimeline
from vpsych.runner.status import RunnerState, RunnerStatus
from vpsych.tests_catalog.base import PresentedTrial, PsychophysicalTest


@runtime_checkable
class TrialWriter(Protocol):
    """Minimal write surface :class:`TrialLoop` needs from a session writer.

    Structurally compatible with (satisfied by) any object implementing
    `vpsych.data.writer.SessionWriter`'s corresponding methods -- see
    ``vpsych.runner.__main__``'s own, broader ``Writer`` protocol, which a
    real ``SessionWriter`` (or an in-memory test fake) satisfies and which
    in turn satisfies this narrower protocol.
    """

    def append_trial(self, trial: TrialRecord) -> None:
        """Append one trial, flushed/fsynced, before the next trial starts."""
        ...

    def write_frames(self, task_id: str, eye: str, run: int, intervals_s: list[float]) -> None:
        """Write this run's accumulated frame-interval log."""
        ...


class PresentationBackend(ABC):
    """Abstracts how a trial's stimulus is presented and a response collected.

    :class:`TrialLoop` interacts with a test only through
    `PsychophysicalTest.present(win, intensity_or_stimulus, trial_ctx)`; a
    backend's job is to supply the right ``win`` and ``trial_ctx`` for its
    presentation mode (see the module docstring's ``trial_ctx`` contract),
    and to provide the small set of protocol-level UI actions (instruction/
    break screens, practice feedback, abort detection) every test protocol
    needs regardless of test content.
    """

    @property
    @abstractmethod
    def win(self) -> Any:
        """The window-like object passed to `PsychophysicalTest.present` (or `None`)."""
        raise NotImplementedError

    @abstractmethod
    def build_trial_ctx(
        self,
        *,
        timeline: TrialTimeline,
        rng: np.random.Generator,
        block: Block,
        is_catch: bool,
        trial_index: int,
    ) -> dict[str, Any]:
        """Build the `trial_ctx` dict for one trial (see the module docstring's contract)."""
        raise NotImplementedError

    @abstractmethod
    def show_message(self, text: str) -> None:
        """Display a self-paced instruction/break screen; return once the observer continues."""
        raise NotImplementedError

    @abstractmethod
    def show_feedback(self, correct: bool) -> None:
        """Briefly show correct/incorrect feedback (practice block only)."""
        raise NotImplementedError

    @abstractmethod
    def check_abort(self) -> bool:
        """Return `True` if the observer has requested (and confirmed) aborting the session."""
        raise NotImplementedError


class SimulatedBackend(PresentationBackend):
    """Headless backend: a :class:`SimulatedObserver` stands in for the human.

    No window, no keyboard, no wall-clock waiting: `win` is `None`, feedback
    and messages are no-ops, and frame timing a test fabricates in its
    `present()` (per the `trial_ctx` contract) should use whole, un-dropped
    frames -- this backend does not itself fabricate timing, since only the
    test knows its own stimulus frame count.

    Args:
        observer: The simulated observer whose ``respond`` a test's
            `present()` should call (via `trial_ctx["simulated_observer"]`)
            in place of collecting a real keypress.
        abort_after_trials: If set, :meth:`check_abort` returns `True`
            starting from the trial with this 0-based index (counting every
            `build_trial_ctx` call, i.e. every presented trial including
            practice/demo) -- for exercising the trial loop's abort path in
            tests, without any real keyboard/dialog.
    """

    def __init__(self, observer: SimulatedObserver, abort_after_trials: int | None = None) -> None:
        self._observer = observer
        self._abort_after_trials = abort_after_trials
        self._trials_presented = 0

    @property
    def win(self) -> Any:
        return None

    def build_trial_ctx(
        self,
        *,
        timeline: TrialTimeline,
        rng: np.random.Generator,
        block: Block,
        is_catch: bool,
        trial_index: int,
    ) -> dict[str, Any]:
        self._trials_presented += 1
        return {
            "timeline": timeline,
            "rng": rng,
            "block": block,
            "is_catch": is_catch,
            "trial_index": trial_index,
            "keyboard": None,
            "simulated_observer": self._observer,
        }

    def show_message(self, text: str) -> None:
        del text  # no-op: nothing to display headless

    def show_feedback(self, correct: bool) -> None:
        del correct  # no-op

    def check_abort(self) -> bool:
        if self._abort_after_trials is None:
            return False
        return self._trials_presented > self._abort_after_trials


class PsychoPyBackend(PresentationBackend):
    """Real backend: a fullscreen PsychoPy window and a hardware-timestamped keyboard.

    Imports ``psychopy`` lazily (inside `__init__`, never at module import
    time), per the project's headless-testability rule -- this class is not
    exercised by the automated (headless) test suite.

    Args:
        display: Display geometry to open the window at (resolution/
            refresh; used for the fullscreen window size).
        fullscreen: Whether to open the window fullscreen. Defaults to
            `True`; set `False` only for manual/debugging runs.
    """

    def __init__(self, display: Any, fullscreen: bool = True) -> None:
        from psychopy import visual
        from psychopy.hardware import keyboard

        self._win = visual.Window(
            size=(display.width_px, display.height_px),
            fullscr=fullscreen,
            units="pix",
            waitBlanking=True,
            allowGUI=False,
        )
        self._win.recordFrameIntervals = True
        self._keyboard = keyboard.Keyboard()
        self._message_stim = visual.TextStim(self._win, text="", wrapWidth=1200)
        self._feedback_stim = visual.TextStim(self._win, text="", height=40)

    @property
    def win(self) -> Any:
        return self._win

    def build_trial_ctx(
        self,
        *,
        timeline: TrialTimeline,
        rng: np.random.Generator,
        block: Block,
        is_catch: bool,
        trial_index: int,
    ) -> dict[str, Any]:
        return {
            "timeline": timeline,
            "rng": rng,
            "block": block,
            "is_catch": is_catch,
            "trial_index": trial_index,
            "keyboard": self._keyboard,
            "simulated_observer": None,
        }

    def show_message(self, text: str) -> None:
        self._message_stim.text = text + "\n\n(press space to continue)"
        self._keyboard.clearEvents()
        while True:
            self._message_stim.draw()
            self._win.flip()
            if self._keyboard.getKeys(keyList=["space"]):
                return
            if self.check_abort():
                return

    def show_feedback(self, correct: bool) -> None:
        self._feedback_stim.text = "Correct" if correct else "Incorrect"
        self._feedback_stim.color = "green" if correct else "red"
        for _ in range(30):
            self._feedback_stim.draw()
            self._win.flip()

    def check_abort(self) -> bool:
        if not self._keyboard.getKeys(keyList=["escape"]):
            return False
        from psychopy import gui

        dlg = gui.Dlg(title="Abort session?")
        dlg.addText("Escape was pressed. Abort the session? Data collected so far is kept.")
        dlg.show()
        return bool(dlg.OK)

    def close(self) -> None:
        """Close the underlying window and release its resources."""
        with contextlib.suppress(Exception):
            self._win.close()


class TrialLoopConfig(BaseModel):
    """Tunable parameters of the generic trial protocol.

    Attributes:
        n_practice_trials: Number of practice-block trials (feedback given,
            not analyzed, procedure not updated).
        catch_trial_probability: Target fraction of main-block trials that
            are suprathreshold catch trials (~0.10 per the plan);
            eligibility is further restricted so catch trials are never in
            the first ``min_main_trials_before_catch`` trials nor adjacent
            to another catch trial, so the realized fraction is
            approximately, not exactly, this value.
        min_main_trials_before_catch: No catch trial is placed among the
            first this many main-block trials.
        break_every_n_trials: Show a self-paced break after every this many
            main-block trials, or `None` to disable breaks.
        max_main_trials_safety: Hard cap on main-block trials, guarding
            against a misbehaving procedure whose ``finished`` never
            becomes `True`; exceeding it raises `RuntimeError`.
    """

    model_config = ConfigDict(frozen=True)

    n_practice_trials: int = Field(default=3, ge=0)
    catch_trial_probability: float = Field(default=0.10, ge=0.0, lt=1.0)
    min_main_trials_before_catch: int = Field(default=3, ge=0)
    break_every_n_trials: int | None = Field(default=40, gt=0)
    max_main_trials_safety: int = Field(default=2000, gt=0)


class TrialLoopResult(BaseModel):
    """Everything :class:`TrialLoop.run` collected while running one test.

    Attributes:
        trial_records: Every persisted trial (practice + main, including
            catch trials), in presentation order.
        frame_intervals_s: All measured (or fabricated) inter-flip
            intervals across every presented trial, already written to the
            writer via ``write_frames`` by the time this is returned.
        aborted: `True` if the observer aborted the session partway
            through (see `PresentationBackend.check_abort`); trials
            presented before the abort are still in `trial_records` and
            already durably written.
        n_catch: Number of main-block catch trials presented.
        n_main_scored: Number of main-block, non-catch trials presented
            (i.e. trials that updated the driving procedure).
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    trial_records: list[TrialRecord] = Field(default_factory=list)
    frame_intervals_s: list[float] = Field(default_factory=list)
    aborted: bool = False
    n_catch: int = Field(ge=0, default=0)
    n_main_scored: int = Field(ge=0, default=0)


def _next_value(procedure: AdaptiveProcedure | MultiParamProcedure) -> float | dict[str, float]:
    if isinstance(procedure, MultiParamProcedure):
        return procedure.next_stimulus()
    return procedure.next_intensity()


def _update(
    procedure: AdaptiveProcedure | MultiParamProcedure,
    value: float | dict[str, float],
    correct: bool,
) -> None:
    if isinstance(procedure, MultiParamProcedure):
        assert isinstance(value, dict)
        procedure.update(value, correct)
    else:
        assert isinstance(value, int | float)
        procedure.update(float(value), correct)


def _scalar_intensity(value: float | dict[str, float], intensity_units: str) -> float:
    """Reduce a presented intensity/stimulus to `TrialRecord.intensity`'s required scalar.

    See the module docstring's ``trial_ctx`` contract section: for a
    `MultiParamProcedure`-driven stimulus dict, this requires an
    ``"intensity"`` key.
    """
    if isinstance(value, dict):
        if "intensity" not in value:
            raise ValueError(
                "A MultiParamProcedure's next_stimulus() dict must include an 'intensity' key "
                f"(the scalar summary dimension to log, in units {intensity_units!r}); got keys "
                f"{sorted(value)}."
            )
        return float(value["intensity"])
    return float(value)


class TrialLoop:
    """Runs one test's full protocol: instructions, demo, practice, main block, breaks.

    Args:
        test: The configured test instance to run.
        procedure: The adaptive procedure driving trial-by-trial intensity
            selection (an `AdaptiveProcedure`) or stimulus selection (a
            `MultiParamProcedure`).
        backend: The presentation backend (`PsychoPyBackend` or
            `SimulatedBackend`) supplying `win`/`trial_ctx` and protocol UI.
        writer: Where to persist each trial (see `TrialWriter`).
        participant_id: Pseudonymous participant ID, `sub-XXXX`.
        session_id: Session ID, `ses-YYYYMMDDTHHMMSS`.
        eye: Eye condition under test.
        run_number: 1-based run number for this test within the session.
        timeline: Frame-count phase durations to use for every trial (a
            test may build its own per-trial timeline internally if it
            needs to vary this; this is the default handed to it via
            `trial_ctx["timeline"]`).
        rng: Seeded generator for this test's own stochastic choices
            (catch-trial placement); a per-trial child/derived generator is
            not created here -- the same `rng` is handed to every trial via
            `trial_ctx["rng"]` (draws are sequential and therefore still
            fully reproducible from `rng_seed`).
        rng_seed: The seed that produced `rng` (see
            `vpsych.core.rng.make_rng`), logged verbatim on every
            `TrialRecord`.
        config: Protocol tuning parameters; defaults per
            `TrialLoopConfig`.
        status_path: If given, `RunnerStatus` snapshots are written here as
            the protocol progresses (see `vpsych.runner.status.RunnerStatus`).
        current_test_index: 0-based index of this test in the session plan,
            for status snapshots.
        n_tests: Total number of tests in the session plan, for status
            snapshots.
    """

    def __init__(
        self,
        test: PsychophysicalTest,
        procedure: AdaptiveProcedure | MultiParamProcedure,
        backend: PresentationBackend,
        writer: TrialWriter,
        *,
        participant_id: str,
        session_id: str,
        eye: Eye,
        run_number: int,
        timeline: TrialTimeline,
        rng: np.random.Generator,
        rng_seed: int,
        config: TrialLoopConfig | None = None,
        status_path: Path | None = None,
        current_test_index: int = 0,
        n_tests: int = 1,
    ) -> None:
        self.test = test
        self.procedure = procedure
        self.backend = backend
        self.writer = writer
        self.participant_id = participant_id
        self.session_id = session_id
        self.eye = eye
        # Named `run_number` (not `run`) to avoid shadowing the `run()` method
        # below -- an instance attribute named `run` would hide it entirely.
        self.run_number = run_number
        self.timeline = timeline
        self.rng = rng
        self.rng_seed = rng_seed
        self.config = config or TrialLoopConfig()
        self.status_path = status_path
        self.current_test_index = current_test_index
        self.n_tests = n_tests

        intensity_units = getattr(procedure, "intensity_units", None)
        self.intensity_units: str = intensity_units or test.spec.output_units

    def _write_status(
        self,
        state: RunnerState,
        *,
        trial_index: int | None = None,
        n_trials_expected: int | None = None,
        message: str = "",
    ) -> None:
        if self.status_path is None:
            return
        RunnerStatus.create(
            state=state,
            current_test_index=self.current_test_index,
            n_tests=self.n_tests,
            task_id=self.test.spec.id,
            trial_index=trial_index,
            n_trials_expected=n_trials_expected,
            message=message,
        ).write_atomic(self.status_path)

    def _present_and_build_record(
        self, *, block: Block, is_catch: bool, trial_index: int, value: float | dict[str, float]
    ) -> tuple[TrialRecord, bool, list[float]]:
        """Present one trial and build its `TrialRecord`. Returns (record, correct, intervals_s)."""
        trial_ctx = self.backend.build_trial_ctx(
            timeline=self.timeline,
            rng=self.rng,
            block=block,
            is_catch=is_catch,
            trial_index=trial_index,
        )
        presented: PresentedTrial = self.test.present(self.backend.win, value, trial_ctx)

        stimulus_params = trial_ctx.get("stimulus_params", {})
        correct_response = trial_ctx.get("correct_response")
        correct = self.test.score(presented.response, stimulus_params)

        record = TrialRecord(
            participant_id=self.participant_id,
            session_id=self.session_id,
            task_id=self.test.spec.id,
            task_version=self.test.spec.version,
            run=self.run_number,
            eye=self.eye,
            block=block,
            trial_index=trial_index,
            is_catch=is_catch,
            intensity=_scalar_intensity(value, self.intensity_units),
            intensity_units=self.intensity_units,
            stimulus_params=stimulus_params,
            correct_response=correct_response,
            response=presented.response,
            correct=correct,
            rt_s=presented.rt_s,
            stimulus_onset_s=presented.stimulus_onset_s,
            n_dropped_frames_trial=presented.n_dropped_frames,
            procedure_state=self.procedure.state_dict(),
            timestamp_utc=datetime.now(timezone.utc),
            rng_seed=self.rng_seed,
        )
        return record, correct, list(presented.frame_intervals_s)

    def run(self) -> TrialLoopResult:
        """Run the full protocol for this test and return the collected result.

        Order: instructions -> demo trial (not recorded) -> practice block
        (recorded with ``block="practice"``, feedback shown, procedure NOT
        updated) -> main block (recorded with ``block="main"``, ~10% catch
        trials interspersed, procedure updated on non-catch trials only,
        self-paced breaks every `TrialLoopConfig.break_every_n_trials`
        trials) -> done. Every trial's `TrialRecord` is appended via
        `writer.append_trial` immediately, before the next trial starts.

        Returns:
            The collected `TrialLoopResult`. If the observer aborts
            partway through (`PresentationBackend.check_abort`), returns
            immediately with `TrialLoopResult.aborted=True` and whatever
            trials were completed (and already durably written) so far.
        """
        trial_records: list[TrialRecord] = []
        all_intervals: list[float] = []

        self._write_status("instructions", message=f"Instructions: {self.test.spec.name}")
        self.backend.show_message(self.test.instructions())
        if self.backend.check_abort():
            return self._finish(trial_records, all_intervals, aborted=True, n_catch=0, n_main=0)

        # Demo trial: shown, not recorded, not scored against the procedure.
        demo_value = self.test.make_catch_trial_intensity()
        self._present_and_build_record(
            block="practice", is_catch=False, trial_index=0, value=demo_value
        )

        # Practice block: recorded, feedback shown, procedure NOT updated.
        self._write_status("practice", n_trials_expected=self.config.n_practice_trials)
        for i in range(self.config.n_practice_trials):
            if self.backend.check_abort():
                return self._finish(trial_records, all_intervals, aborted=True, n_catch=0, n_main=0)
            practice_value = self.test.make_catch_trial_intensity()
            record, correct, intervals = self._present_and_build_record(
                block="practice", is_catch=False, trial_index=i, value=practice_value
            )
            self.writer.append_trial(record)
            trial_records.append(record)
            all_intervals.extend(intervals)
            self.backend.show_feedback(correct)
            self._write_status(
                "practice", trial_index=i, n_trials_expected=self.config.n_practice_trials
            )

        # Main block.
        self._write_status("running")
        n_catch = 0
        n_main_scored = 0
        last_was_catch = False
        main_trial_index = 0
        while not self.procedure.finished:
            if main_trial_index >= self.config.max_main_trials_safety:
                raise RuntimeError(
                    f"TrialLoop exceeded max_main_trials_safety="
                    f"{self.config.max_main_trials_safety} without the procedure finishing; "
                    "this indicates a misbehaving procedure (finished never became True)."
                )
            if self.backend.check_abort():
                return self._finish(
                    trial_records,
                    all_intervals,
                    aborted=True,
                    n_catch=n_catch,
                    n_main=n_main_scored,
                )

            eligible_for_catch = (
                main_trial_index >= self.config.min_main_trials_before_catch and not last_was_catch
            )
            is_catch = eligible_for_catch and bool(
                self.rng.random() < self.config.catch_trial_probability
            )

            if is_catch:
                value: float | dict[str, float] = self.test.make_catch_trial_intensity()
            else:
                value = _next_value(self.procedure)

            record, correct, intervals = self._present_and_build_record(
                block="main", is_catch=is_catch, trial_index=main_trial_index, value=value
            )
            self.writer.append_trial(record)
            trial_records.append(record)
            all_intervals.extend(intervals)

            if is_catch:
                n_catch += 1
            else:
                _update(self.procedure, value, correct)
                n_main_scored += 1

            last_was_catch = is_catch
            main_trial_index += 1

            self._write_status("running", trial_index=main_trial_index)

            if (
                self.config.break_every_n_trials is not None
                and main_trial_index % self.config.break_every_n_trials == 0
                and not self.procedure.finished
            ):
                self._write_status("break", trial_index=main_trial_index)
                self.backend.show_message("Take a short break. Press space when ready to continue.")

        return self._finish(
            trial_records, all_intervals, aborted=False, n_catch=n_catch, n_main=n_main_scored
        )

    def _finish(
        self,
        trial_records: list[TrialRecord],
        all_intervals: list[float],
        *,
        aborted: bool,
        n_catch: int,
        n_main: int,
    ) -> TrialLoopResult:
        self.writer.write_frames(self.test.spec.id, self.eye, self.run_number, all_intervals)
        self._write_status(
            "aborted" if aborted else "finished",
            message="Session aborted by observer." if aborted else "Test complete.",
        )
        return TrialLoopResult(
            trial_records=trial_records,
            frame_intervals_s=all_intervals,
            aborted=aborted,
            n_catch=n_catch,
            n_main_scored=n_main,
        )
