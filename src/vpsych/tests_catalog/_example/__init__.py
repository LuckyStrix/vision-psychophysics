"""A minimal, complete example test plugin: a 2AFC contrast-detection task.

This is **not** a real perceptual test -- it is registered (`hidden=True`,
so it is runnable end to end via `get_test`/the runner but never listed in
a participant-facing UI) purely so it can be exercised by:

- `tests/integration/test_end_to_end.py`, which needs a small but realistic
  test driven by `QuestPlusProcedure` to validate the full runner/data-layer
  pipeline against something more representative than a throwaway dummy.
- `docs/WRITING_A_TEST.md`, the guide for implementing the 7 real perceptual
  tests, which walks through this file section by section.

Every convention a real test should follow is demonstrated here: the
`trial_ctx` input/output keys, the `"intensity"` key for `MultiParamProcedure`
(not used by this particular test, which uses the simpler `AdaptiveProcedure`
path, but documented in the module docstring anyway), frames-only timing,
lazy `psychopy` imports, using calibration (gamma/dithering/deg<->px), catch
trials, `summarize`/quality flags, and the `simulated_response` hook. See
`docs/WRITING_A_TEST.md` for the narrated walkthrough.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from vpsych.core.procedures.questplus_procedure import QuestPlusProcedure
from vpsych.core.timing import presentation_timing
from vpsych.data.quality import compute_quality_flags
from vpsych.data.schemas import TestSummary
from vpsych.tests_catalog.base import (
    PresentedTrial,
    PsychophysicalTest,
    TestRequirements,
    TestSpec,
    register_test,
)

#: Grid defaults kept small on purpose: this test exists to be *fast* in CI
#: (the end-to-end integration test runs it as a real subprocess), not to be
#: scientifically well-powered. A real test's grids should be sized the way
#: `vpsych.core.procedures.qcsf.QCSF.default_grids()` documents its own --
#: informed by the actual dynamic range/precision the measurement needs.
DEFAULT_INTENSITY_VALUES = [float(v) for v in np.linspace(-2.2, 0.2, 25)]
DEFAULT_THRESHOLD_VALUES = [float(v) for v in np.linspace(-2.0, 0.0, 11)]
DEFAULT_SLOPE_VALUES = [float(v) for v in np.linspace(0.15, 0.6, 4)]
DEFAULT_LAPSE_RATE_VALUES = [0.0, 0.02, 0.04]
GUESS_RATE = 0.5  # 2AFC chance rate.


class ExampleContrastParams(BaseModel):
    """This test's configurable parameters (`SessionPlan.PlannedTest.params`).

    Every field has a default so `{}` is a valid `params` dict; a session
    plan overriding `max_trials` (as the end-to-end test does, to keep a
    full subprocess run fast) is the intended way to tune trial count per
    session rather than hardcoding it in the test.
    """

    model_config = ConfigDict(frozen=True)

    max_trials: int = Field(default=30, ge=1, description="QUEST+ trial budget, main block.")
    grating_size_deg: float = Field(
        default=4.0, gt=0, description="Grating patch diameter, in degrees of visual angle."
    )
    spatial_frequency_cpd: float = Field(
        default=2.0, gt=0, description="Grating spatial frequency, cycles per degree."
    )


@register_test
class ExampleContrastTest(PsychophysicalTest):
    """2AFC contrast-detection task: "which interval/side had the grating?"

    A stripped-down illustration of the plugin interface's shape -- not a
    validated perceptual test. See the module docstring.
    """

    spec = TestSpec(
        id="example_contrast_2afc",
        name="Example Contrast Detection (2AFC)",
        version="0.1.0",
        domain="contrast",
        description_participant=(
            "A faint striped patch will appear on the left or right side of the screen. "
            "Press the arrow key for the side it appeared on."
        ),
        description_technical=(
            "2AFC contrast-detection threshold via QUEST+ (Watson 2017) on a fixed-slope "
            "Weibull psychometric function. Example/documentation test, not a validated "
            "perceptual measure -- see vpsych.tests_catalog._example's module docstring."
        ),
        measures="Contrast detection threshold (illustrative only).",
        output_units="log10_contrast",
        estimated_minutes=1.0,
        allowed_eyes=["OU"],
        requirements=TestRequirements(),  # no calibration required for this illustrative task
        citations=["Watson, A. B. (2017). QUEST+. Journal of Vision, 17(3):10."],
        params_model=ExampleContrastParams,
        hidden=True,  # registered and runnable, but never shown in a participant-facing UI
    )

    def __init__(
        self,
        params: BaseModel,
        display: Any,
        calibration: Any,
        rng: np.random.Generator,
    ) -> None:
        assert isinstance(params, ExampleContrastParams)
        self.params = params
        self.display = display
        self.calibration = calibration
        self.rng = rng
        self._n_presented = 0
        # build_stimuli()'s return value is discarded by the caller (see that
        # method's docstring on PsychophysicalTest) -- store on self instead.
        self._stims: dict[str, Any] = {}

    def make_procedure(self) -> QuestPlusProcedure:
        return QuestPlusProcedure(
            intensity_values=DEFAULT_INTENSITY_VALUES,
            intensity_units="log10_contrast",
            threshold_values=DEFAULT_THRESHOLD_VALUES,
            slope_values=DEFAULT_SLOPE_VALUES,
            guess_rate=GUESS_RATE,
            lapse_rate_values=DEFAULT_LAPSE_RATE_VALUES,
            function="weibull",
            max_trials=self.params.max_trials,
        )

    def make_catch_trial_intensity(self) -> float:
        # The highest (easiest) contrast on the grid: a suprathreshold probe
        # an attentive observer should essentially always get right.
        return max(DEFAULT_INTENSITY_VALUES)

    def build_stimuli(self, win: Any) -> dict[str, Any]:
        if win is None:
            # SimulatedBackend: no window, nothing to build (see this method's
            # docstring on PsychophysicalTest).
            return {}

        # Real display: import psychopy lazily, never at module import time,
        # so this whole package stays importable headless (discover_tests()
        # imports every test subpackage unconditionally, including in CI).
        from psychopy import visual

        size_px = self.display.deg_to_px(self.params.grating_size_deg)
        grating = visual.GratingStim(
            win,
            sf=self.params.spatial_frequency_cpd / self.display.px_per_deg_at_center,
            size=(size_px, size_px),
            mask="gauss",
            contrast=0.0,
        )
        fixation = visual.TextStim(win, text="+", height=20)
        self._stims = {"grating": grating, "fixation": fixation}
        return self._stims

    def _draw_grating(self, contrast: float, side: str) -> None:
        """Position and set the grating's (dithered, gamma-correct) contrast, then draw it."""
        grating = self._stims["grating"]
        offset_px = self.display.deg_to_px(self.params.grating_size_deg)
        grating.pos = (-offset_px, 0) if side == "left" else (offset_px, 0)
        # Bit-stealing dither the requested contrast before handing it to
        # PsychoPy, so contrasts below 1/255 are still presented with the
        # right *expected* value (see vpsych.core.calibration.dither); a
        # real test also applies calibration.gamma's inverse-gamma lookup
        # (vpsych.core.calibration.gamma.linearize) when setting the
        # window's own gamma ramp at window-creation time (once, not
        # per-trial) so `contrast` here means linear, perceptual contrast.
        from vpsych.core.calibration.dither import dither_to_uint8

        dithered_level = int(dither_to_uint8(np.clip(contrast, 0.0, 1.0), self.rng))
        grating.contrast = dithered_level / 255.0
        grating.draw()

    def present(
        self, win: Any, intensity_or_stimulus: Any, trial_ctx: dict[str, Any]
    ) -> PresentedTrial:
        self._n_presented += 1
        rng = trial_ctx["rng"]
        intensity = float(intensity_or_stimulus)  # log10 contrast
        contrast = 10.0**intensity
        correct_side = "left" if rng.random() < 0.5 else "right"

        # Output keys every present() must set on trial_ctx (see
        # vpsych.core.trial_loop's trial_ctx contract):
        stimulus_params: dict[str, Any] = {
            "correct_response": correct_side,
            "intensity": intensity,
            "contrast": contrast,
        }
        trial_ctx["stimulus_params"] = stimulus_params
        trial_ctx["correct_response"] = correct_side

        observer = trial_ctx["simulated_observer"]
        if observer is not None:
            # Simulated path: decouple the observer's correct/incorrect
            # decision from this test's own response representation (see
            # vpsych.core.observers.SimulatedObserver's docstring and
            # PsychophysicalTest.simulated_response).
            is_correct = observer.decide_correct(stimulus_params, rng)
            response = self.simulated_response(is_correct, stimulus_params, rng)
            timeline = trial_ctx["timeline"]
            return PresentedTrial(
                response=response,
                rt_s=0.3,
                stimulus_onset_s=float(self._n_presented),
                n_dropped_frames=0,
                # Fabricated perfect timing at the nominal refresh; a real
                # simulated run has no display to measure actual intervals
                # from (see SimulatedBackend's docstring).
                frame_intervals_s=[1.0 / 60.0] * max(timeline.stimulus_frames, 1),
            )

        # Real display path: frames-only timing throughout (never
        # time.sleep()) -- draw fixation for fixation_frames, then the
        # grating for stimulus_frames, then collect a hardware-timestamped
        # keypress via trial_ctx["keyboard"].
        timeline = trial_ctx["timeline"]
        keyboard = trial_ctx["keyboard"]
        fixation = self._stims["fixation"]
        for _ in range(timeline.fixation_frames):
            fixation.draw()
            win.flip()

        keyboard.clearEvents()
        onset_s = None
        flip_times: list[float] = []
        for frame in range(timeline.stimulus_frames):
            self._draw_grating(contrast, correct_side)
            flip_time = win.flip()
            flip_times.append(flip_time)
            if frame == 0:
                onset_s = flip_time
                keyboard.clock.reset()

        keys = keyboard.waitKeys(keyList=["left", "right"], waitRelease=False, clear=False)
        response, rt_s = (keys[0].name, keys[0].rt) if keys else (None, None)

        for _ in range(timeline.iti_frames):
            win.flip()

        intervals_s, n_dropped = presentation_timing(flip_times, self.display.refresh_hz)

        return PresentedTrial(
            response=response,
            rt_s=rt_s,
            stimulus_onset_s=onset_s or 0.0,
            n_dropped_frames=n_dropped,
            frame_intervals_s=intervals_s,
        )

    def response_keys(self) -> list[str]:
        return ["left", "right"]

    def score(self, response: Any, stimulus_params: dict[str, Any]) -> bool:
        return bool(response == stimulus_params["correct_response"])

    def instructions(self) -> str:
        return self.spec.description_participant

    def summarize(self, trials: pd.DataFrame) -> TestSummary:
        """Recompute the threshold deterministically from the raw trials.

        Replays a fresh `QuestPlusProcedure` over every main-block,
        non-catch trial (in `trial_index` order), rather than trusting any
        live in-memory procedure state -- this is what makes `vpsych
        reanalyze` reproduce this summary exactly from `..._trials.tsv`
        alone (see `vpsych.data.reanalyze`), since `QuestPlusProcedure`'s
        `update`/`estimate` involve no randomness at all.
        """
        main = trials[trials["block"] == "main"]
        non_catch = main[~main["is_catch"]].sort_values("trial_index")
        catch = main[main["is_catch"]]

        procedure = self.make_procedure()
        for _, row in non_catch.iterrows():
            procedure.update(float(row["intensity"]), bool(row["correct"]))
        estimate = procedure.estimate()

        n_catch = len(catch)
        catch_lapse_rate = float((~catch["correct"]).mean()) if n_catch else 0.0
        dropped_fraction = (
            float(trials["n_dropped_frames_trial"].sum()) / len(trials) if len(trials) else 0.0
        )

        quality_flags = compute_quality_flags(
            catch_lapse_rate=catch_lapse_rate,
            n_catch=n_catch,
            dropped_fraction=dropped_fraction,
            threshold=estimate.value,
            range_min=min(DEFAULT_INTENSITY_VALUES),
            range_max=max(DEFAULT_INTENSITY_VALUES),
            n_trials=len(non_catch),
            min_trials=self.params.max_trials,
        )

        eye = str(trials["eye"].iloc[0]) if len(trials) else "OU"
        return TestSummary(
            task_id=self.spec.id,
            task_version=self.spec.version,
            eye=eye,  # read from the data, not hardcoded (any of OD/OS/OU)
            run=1,
            estimate=estimate,
            fit_params={},
            gof={},
            quality_flags=quality_flags,
            n_trials=len(non_catch),
            n_catch=n_catch,
            catch_lapse_rate=catch_lapse_rate,
            analysis_version="0.1.0",
        )
