"""Plugin interface every perceptual test implements, plus the test registry.

Each test (acuity, contrast sensitivity, color discrimination, motion
coherence, vernier acuity, critical flicker fusion, ...) lives in its own
subpackage of `tests_catalog` and registers a `PsychophysicalTest` subclass
with `@register_test`. The app, runner, and reanalysis code depend only on
this interface, never on a specific test's internals, so tests can be added
in parallel without touching shared code.

`psychopy` is never imported at module import time here -- only inside
methods that are actually given a live `win` (a PsychoPy `Window`), per the
project's headless-testability rule. This module and `TestSpec`/
`TestRequirements` must import and be inspectable (e.g. to build the test
catalog UI, or to run `check_requirements`) with no display present.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from vpsych.core.calibration.models import Calibration, ColorGrade, LuminanceGrade
from vpsych.core.display import DisplayGeometry
from vpsych.core.procedures.base import AdaptiveProcedure, MultiParamProcedure
from vpsych.data.schemas import TestSummary

if TYPE_CHECKING:
    import pandas as pd

Domain = Literal["acuity", "contrast", "color", "motion", "temporal", "hyperacuity"]

_GRADE_RANK: dict[str, int] = {"A": 0, "B": 1, "C": 2}


class TestRequirements(BaseModel):
    """Prerequisites a test needs from the active display and calibration.

    Used by `check_requirements` to decide whether a test can currently be
    run "research-grade", and by the test-catalog UI to disable a test card
    and explain why.

    Attributes:
        needs_gamma_calibration: Whether a luminance/gamma calibration is
            required at all (grade C, i.e. `method="none"`, does not
            satisfy this).
        needs_color_calibration: Whether a color primaries calibration
            better than the sRGB-assumed default is required.
        min_luminance_grade: Minimum acceptable luminance grade (`"A"`,
            `"B"`, or `"C"`; `"A"` is strictest), or `None` if no minimum.
        min_color_grade: Minimum acceptable color grade, or `None` if no
            minimum.
        min_viewing_distance_cm: Minimum required viewing distance, in
            centimetres, or `None` if unconstrained (some tests need
            distance to keep required spatial frequencies under the
            display's Nyquist limit, see
            `DisplayGeometry.nyquist_cpd`).
        min_refresh_hz: Minimum required display refresh rate, in hertz, or
            `None` if unconstrained (relevant for flicker/motion tests).
        max_required_cpd: The highest spatial frequency, in cycles per
            degree, this test needs to present at the configured viewing
            distance, or `None` if not spatial-frequency-limited. Checked
            against `DisplayGeometry.nyquist_cpd`.
    """

    model_config = ConfigDict(frozen=True)

    needs_gamma_calibration: bool = Field(
        default=False, description="Whether a luminance/gamma calibration is required."
    )
    needs_color_calibration: bool = Field(
        default=False, description="Whether a measured color calibration is required."
    )
    min_luminance_grade: LuminanceGrade | None = Field(
        default=None, description="Minimum acceptable luminance grade."
    )
    min_color_grade: ColorGrade | None = Field(
        default=None, description="Minimum acceptable color grade."
    )
    min_viewing_distance_cm: float | None = Field(
        default=None, gt=0, description="Minimum required viewing distance, in cm."
    )
    min_refresh_hz: float | None = Field(
        default=None, gt=0, description="Minimum required display refresh rate, in Hz."
    )
    max_required_cpd: float | None = Field(
        default=None, gt=0, description="Highest spatial frequency this test needs, in cycles/deg."
    )


class TestSpec(BaseModel):
    """Static metadata describing one perceptual test, independent of any run.

    Attributes:
        id: Stable snake_case identifier, e.g. `"visual_acuity"`. Used as
            `TrialRecord.task_id` and as the registry key.
        name: Human-readable display name, e.g. "Visual Acuity (Landolt C)".
        version: Semantic version of this test's implementation
            (`TrialRecord.task_version` records this per trial so changes to
            a test's method don't silently mix with old data at reanalysis).
        domain: Which perceptual domain this test belongs to.
        description_participant: Plain-language description shown to the
            participant before running the test (no jargon).
        description_technical: Technical description for the results/docs
            audience (method, citation-level detail).
        measures: Short description of the quantity this test estimates,
            e.g. "Letter acuity threshold".
        output_units: Units of the test's primary threshold estimate, e.g.
            `"logMAR"`, `"log10_contrast"`, `"coherence_fraction"`,
            `"arcsec"`, `"hz"`.
        estimated_minutes: Expected wall-clock duration, in minutes, shown
            in the session builder.
        allowed_eyes: Which eye conditions this test supports testing.
        requirements: Calibration/display prerequisites (see
            :class:`TestRequirements`).
        citations: Literature citations backing this test's method, as
            free-text strings (e.g. `"Bach, M. (1996). The Freiburg Visual
            Acuity Test..."`), also included in `docs/METHODS.md`.
        params_model: The pydantic model class describing this test's
            configurable parameters (e.g. starting intensity, number of
            trials); instantiated per `SessionPlan.PlannedTest.params`.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$", description="Stable snake_case identifier.")
    name: str = Field(description="Human-readable display name.")
    version: str = Field(
        pattern=r"^\d+\.\d+\.\d+$", description="Semantic version of this test's implementation."
    )
    domain: Domain = Field(description="Perceptual domain this test belongs to.")
    description_participant: str = Field(description="Plain-language description for participants.")
    description_technical: str = Field(
        description="Technical description (method, citation-level)."
    )
    measures: str = Field(description="Short description of the estimated quantity.")
    output_units: str = Field(description="Units of the primary threshold estimate.")
    estimated_minutes: float = Field(gt=0, description="Expected wall-clock duration, in minutes.")
    allowed_eyes: list[Literal["OD", "OS", "OU"]] = Field(
        description="Which eye conditions this test supports."
    )
    requirements: TestRequirements = Field(description="Calibration/display prerequisites.")
    citations: list[str] = Field(default_factory=list, description="Literature citations.")
    params_model: type[BaseModel] = Field(
        description="Pydantic model class for this test's params."
    )


def check_requirements(
    requirements: TestRequirements,
    display: DisplayGeometry,
    calibration: Calibration | None,
) -> list[str]:
    """Check a test's requirements against the active display and calibration.

    Args:
        requirements: The test's declared prerequisites.
        display: The active display geometry.
        calibration: The active calibration, or `None` if uncalibrated.

    Returns:
        A list of human-readable reasons the requirements are *not* met
        (empty list means all requirements are satisfied). Each reason is a
        complete sentence suitable for display directly in the test-catalog
        UI, e.g. `"Needs a gamma calibration (currently uncalibrated)."`.
    """
    reasons: list[str] = []

    if requirements.needs_gamma_calibration and (
        calibration is None or calibration.gamma.method == "none"
    ):
        reasons.append("Needs a gamma calibration (currently uncalibrated).")

    if requirements.needs_color_calibration and (
        calibration is None or calibration.color.method != "measured"
    ):
        reasons.append(
            "Needs a measured color calibration (currently sRGB-assumed or uncalibrated)."
        )

    if requirements.min_luminance_grade is not None:
        if calibration is None:
            reasons.append(
                f"Needs luminance grade {requirements.min_luminance_grade} or better "
                "(currently uncalibrated)."
            )
        elif (
            _GRADE_RANK[calibration.luminance_grade] > _GRADE_RANK[requirements.min_luminance_grade]
        ):
            reasons.append(
                f"Needs luminance grade {requirements.min_luminance_grade} or better "
                f"(currently grade {calibration.luminance_grade})."
            )

    if requirements.min_color_grade is not None:
        if calibration is None:
            reasons.append(
                f"Needs color grade {requirements.min_color_grade} or better "
                "(currently uncalibrated)."
            )
        elif _GRADE_RANK[calibration.color_grade] > _GRADE_RANK[requirements.min_color_grade]:
            reasons.append(
                f"Needs color grade {requirements.min_color_grade} or better "
                f"(currently grade {calibration.color_grade})."
            )

    if (
        requirements.min_viewing_distance_cm is not None
        and display.viewing_distance_cm < requirements.min_viewing_distance_cm
    ):
        reasons.append(
            f"Needs a viewing distance of at least {requirements.min_viewing_distance_cm:g} cm "
            f"(currently {display.viewing_distance_cm:g} cm)."
        )

    if requirements.min_refresh_hz is not None and display.refresh_hz < requirements.min_refresh_hz:
        reasons.append(
            f"Needs a display refresh rate of at least {requirements.min_refresh_hz:g} Hz "
            f"(currently {display.refresh_hz:g} Hz)."
        )

    if (
        requirements.max_required_cpd is not None
        and requirements.max_required_cpd >= display.nyquist_cpd
    ):
        reasons.append(
            f"Needs to present {requirements.max_required_cpd:g} cycles/deg, at or above this "
            f"display's Nyquist limit of {display.nyquist_cpd:g} cycles/deg at the configured "
            "viewing distance."
        )

    return reasons


class PresentedTrial(BaseModel):
    """Outcome of presenting one trial's stimulus and collecting a response.

    Returned by `PsychophysicalTest.present`; the trial loop combines this
    with procedure/session bookkeeping to build a full
    `vpsych.core.trial.TrialRecord`.

    Attributes:
        response: The observer's response, or `None` if no response was
            given (e.g. a response timeout). Representation is test-specific
            (e.g. a direction label or orientation index).
        rt_s: Response time, in seconds, from stimulus onset to the
            response event, or `None` if no response was given.
        stimulus_onset_s: Timestamp of the stimulus-onset flip, in seconds,
            on the runner's monotonic clock.
        n_dropped_frames: Number of dropped frames detected during this
            trial's stimulus presentation.
        frame_intervals_s: Raw measured inter-flip intervals during the
            stimulus phase, in seconds, for logging to the session's
            `_frames.tsv` (see `vpsych.core.timing.summarize_frame_intervals`).
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    response: Any = Field(default=None, description="The observer's response, or None.")
    rt_s: float | None = Field(default=None, ge=0, description="Response time in seconds.")
    stimulus_onset_s: float = Field(description="Stimulus-onset flip time, in seconds.")
    n_dropped_frames: int = Field(ge=0, description="Dropped frames during stimulus presentation.")
    frame_intervals_s: list[float] = Field(
        default_factory=list, description="Raw measured inter-flip intervals, in seconds."
    )


class PsychophysicalTest(ABC):
    """Abstract base class every perceptual test implements.

    A concrete subclass is registered with `@register_test` and is
    constructed once per test run by the runner:

        test = SomeTest(params=..., display=..., calibration=..., rng=...)
        procedure = test.make_procedure()
        win = ...  # a psychopy Window, created by the runner
        stimuli = test.build_stimuli(win)
        while not procedure.finished:
            intensity = procedure.next_intensity()
            presented = test.present(win, intensity, trial_ctx)
            correct = test.score(presented.response, stimulus_params)
            procedure.update(intensity, correct)
        summary = test.summarize(trials_df)

    Attributes:
        spec: Class-level :class:`TestSpec` describing this test; set by
            the subclass as a class variable.
    """

    spec: TestSpec

    @abstractmethod
    def __init__(
        self,
        params: BaseModel,
        display: DisplayGeometry,
        calibration: Calibration,
        rng: np.random.Generator,
    ) -> None:
        """Construct a test instance for one run.

        Args:
            params: Instance of `spec.params_model` with this run's
                configured parameters.
            display: Active display geometry.
            calibration: Active calibration (callers should have already
                checked `check_requirements(spec.requirements, display,
                calibration)` is empty).
            rng: Seeded random generator for this test's stochastic choices
                (trial order, catch-trial placement, jitter), from
                `vpsych.core.rng.make_rng`.
        """
        raise NotImplementedError

    @abstractmethod
    def make_procedure(self) -> AdaptiveProcedure | MultiParamProcedure:
        """Construct and return the adaptive procedure driving this test's trials.

        Returns:
            A configured procedure instance (e.g. a `QuestPlusProcedure` for
            most tests, or a `QCSF` for the contrast sensitivity function
            test).
        """
        raise NotImplementedError

    @abstractmethod
    def make_catch_trial_intensity(self) -> float:
        """Return a suprathreshold intensity to use for a catch trial.

        Catch trials (about 10% of main-block trials, per the shared trial
        protocol) use an intensity well above threshold so a correct
        response is nearly certain; a miss on a catch trial contributes to
        the estimated lapse rate.

        Returns:
            An intensity value, in the same `intensity_units` as the
            driving procedure, expected to be answered correctly by an
            attentive observer.
        """
        raise NotImplementedError

    @abstractmethod
    def build_stimuli(self, win: Any) -> dict[str, Any]:
        """Construct and return this test's reusable PsychoPy stimulus objects.

        Called once per run, after the PsychoPy window is open. Must import
        `psychopy` lazily inside this method (or methods it calls), never at
        module import time, so this module stays importable headless.

        Args:
            win: A `psychopy.visual.Window` the stimuli are drawn into.

        Returns:
            A dict of named stimulus objects (test-specific keys) reused
            across trials by `present`.
        """
        raise NotImplementedError

    @abstractmethod
    def present(
        self, win: Any, intensity_or_stimulus: float | dict[str, float], trial_ctx: dict[str, Any]
    ) -> PresentedTrial:
        """Present one trial's stimulus and collect the observer's response.

        Drives the fixation -> stimulus -> response phases (see
        `vpsych.core.trial.TrialPhase`) entirely in frames, using a response
        collector supplied via `trial_ctx` (e.g. a
        `psychopy.hardware.keyboard.Keyboard` instance) to get
        hardware-timestamped responses.

        Args:
            win: The `psychopy.visual.Window` stimuli are drawn into.
            intensity_or_stimulus: The scalar intensity (for
                `AdaptiveProcedure`-driven tests) or stimulus parameter dict
                (for `MultiParamProcedure`-driven tests) to present this
                trial.
            trial_ctx: Additional per-trial context supplied by the trial
                loop (timeline, response collector, RNG, and similar); keys
                are defined by the trial loop implementation.

        Returns:
            The presented trial's outcome.
        """
        raise NotImplementedError

    @abstractmethod
    def response_keys(self) -> list[str]:
        """Return the keyboard keys this test accepts as responses.

        Returns:
            A list of key names (as recognized by
            `psychopy.hardware.keyboard.Keyboard`), e.g. `["left", "right"]`.
        """
        raise NotImplementedError

    @abstractmethod
    def score(self, response: Any, stimulus_params: dict[str, Any]) -> bool:
        """Score whether a response was correct for the given stimulus.

        Args:
            response: The observer's response, as returned in
                `PresentedTrial.response`.
            stimulus_params: The stimulus parameters the response is scored
                against (test-specific).

        Returns:
            `True` if the response was correct, `False` otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def instructions(self) -> str:
        """Return the participant-facing instruction text shown before this test.

        Returns:
            Plain-language instruction text (no jargon), shown on the
            pre-test briefing screen alongside an animated example.
        """
        raise NotImplementedError

    @abstractmethod
    def summarize(self, trials: pd.DataFrame) -> TestSummary:
        """Compute this test's summary (threshold, CI, fit, quality flags) from its trials.

        Args:
            trials: All trials for this test run (main block only should
                typically inform the threshold estimate; practice and catch
                trials are included in the frame for quality-flag purposes
                such as catch-trial lapse rate, but callers should filter
                as appropriate -- see column definitions in
                `vpsych.core.trial.TrialRecord`).

        Returns:
            The computed summary, ready to be written by
            `vpsych.data.writer.SessionWriter.write_summary`.
        """
        raise NotImplementedError


_REGISTRY: dict[str, type[PsychophysicalTest]] = {}


def register_test(cls: type[PsychophysicalTest]) -> type[PsychophysicalTest]:
    """Class decorator that registers a `PsychophysicalTest` subclass by its spec id.

    Args:
        cls: The `PsychophysicalTest` subclass to register. Must define
            `cls.spec` (a `TestSpec` instance) before this decorator runs.

    Returns:
        `cls`, unmodified, so this can be used as a plain decorator.

    Raises:
        ValueError: If a test with the same `spec.id` is already registered
            (catches accidental id collisions between tests).
    """
    test_id = cls.spec.id
    if test_id in _REGISTRY:
        raise ValueError(
            f"A test with id {test_id!r} is already registered "
            f"({_REGISTRY[test_id].__module__}.{_REGISTRY[test_id].__name__})"
        )
    _REGISTRY[test_id] = cls
    return cls


def get_test(test_id: str) -> type[PsychophysicalTest]:
    """Look up a registered test class by its spec id.

    Args:
        test_id: The `TestSpec.id` to look up.

    Returns:
        The registered `PsychophysicalTest` subclass.

    Raises:
        KeyError: If no test with `test_id` is registered.
    """
    try:
        return _REGISTRY[test_id]
    except KeyError:
        raise KeyError(
            f"No test registered with id {test_id!r}. Known ids: {sorted(_REGISTRY)}"
        ) from None


def all_tests() -> list[type[PsychophysicalTest]]:
    """Return every currently registered test class.

    Returns:
        A list of registered `PsychophysicalTest` subclasses, in
        registration order.
    """
    return list(_REGISTRY.values())
