"""Visual acuity: Landolt C recognition acuity via QUEST+ over logMAR.

A FrACT-style (Bach, M. (1996). The Freiburg Visual Acuity test --
automatic measurement of visual acuity. Optometry and Vision Science,
73(1), 49-53, https://doi.org/10.1097/00006324-199601000-00008; Bach, M.
(2007). The Freiburg Visual Acuity Test -- variability unchanged by
post-hoc re-analysis. Graefe's Archive for Clinical and Experimental
Ophthalmology, 245(7), 965-971, https://doi.org/10.1007/s00417-006-0474-4)
Landolt C acuity measurement: the participant reports which of 8 (or 4)
directions a ring's gap opens toward, over a QUEST+ (Watson, A. B. (2017).
QUEST+: A general multidimensional Bayesian adaptive psychometric method.
Journal of Vision, 17(3):10, https://doi.org/10.1167/17.3.10) run on
`logMAR = log10(gap size in arcmin)`. Optotype proportions follow ISO 8596
/ EN ISO 8596 (see `optotype.py`).

**Threshold criterion**: FrACT's own criterion (Bach 1996, Methods) reports
threshold at the intensity giving the probability correct exactly midway
between the guess rate and 100% correct (equivalently, halfway between
chance and ceiling on the psychometric function's rising limb). This
module reproduces that criterion as `guess_rate + 0.5 * (1 - guess_rate -
lapse_rate)`, inverted against `questplus`'s own Weibull parameterization
(Watson & Pelli 1983's original QUEST form,
`p(x) = 1 - lapse - (1-guess-lapse) * exp(-10**(slope*(x-threshold)))`; see
`questplus.psychometric_function.weibull`). This is a genuinely different
functional form from `vpsych.core.psychometric`'s own weibull family
(rescaled so its `threshold` sits at the `F(0) = 0.5` point -- see
`docs/METHODS.md`'s "Psychometric function families" section) whose
`threshold` sits instead at the *classic* Weibull inflection (`1 -
exp(-1) ~= 63.2%` of the way from guess to ceiling) -- so this module
inverts `questplus`'s own formula directly, via the shared, tested
`vpsych.core.procedures.questplus_procedure.QuestPlusProcedure
.intensity_at_p_correct` (which itself wraps `questplus_weibull_x_at_p`),
rather than reusing `vpsych.core.psychometric.intensity_at_p_correct`
(which assumes the other, differently-parameterized family and would give
a silently wrong answer here). See `_fract_criterion_threshold` below.

**Domain and display limits**: the nominal logMAR domain (`-0.5` to `1.3`
by default, per the Phase 2A task) is clipped at construction time to what
the configured display can actually render: the smallest usable gap is
bounded below by `min_gap_px` (default 1.0 physical pixel -- see
`VisualAcuityParams.min_gap_px`'s docstring for why), and the largest
usable gap is bounded above by what fits on screen. `summarize` also
raises a `display_resolution_limited` quality flag if the measured
threshold ends up within 0.1 logMAR of that pixel-bounded floor -- see
that method's docstring, and `docs/methods/visual_acuity.md` for the
worked example of the viewing distance a typical display needs to avoid
this.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from vpsych.core.procedures.base import ThresholdEstimate
from vpsych.core.procedures.questplus_procedure import QuestPlusProcedure
from vpsych.core.timing import presentation_timing
from vpsych.data.quality import compute_quality_flags
from vpsych.data.schemas import QualityFlag, TestSummary
from vpsych.tests_catalog.base import (
    PresentedTrial,
    PsychophysicalTest,
    TestRequirements,
    TestSpec,
    register_test,
)
from vpsych.tests_catalog.visual_acuity.optotype import (
    landolt_c_geometry,
    min_renderable_logmar,
    render_landolt_c,
)

#: Numpad-key orientation mapping for 8AFC, degrees measured counterclockwise
#: from the positive x-axis (0 = right, 90 = up, ...) matching the physical
#: layout of a standard numeric keypad (8 = up, 2 = down, 4 = left, 6 =
#: right, and the four diagonals at the corners).
KEY_BY_ANGLE_8: dict[int, str] = {
    0: "num_6",
    45: "num_9",
    90: "num_8",
    135: "num_7",
    180: "num_4",
    225: "num_1",
    270: "num_2",
    315: "num_3",
}

#: For 4AFC, both the 4 cardinal numpad keys and the 4 arrow keys are
#: accepted (per the Phase 2A task: "also arrow keys for the 4 cardinal
#: directions if the orientation count is set to 4").
KEY_BY_ANGLE_4: dict[int, str] = {0: "num_6", 90: "num_8", 180: "num_4", 270: "num_2"}
ARROW_KEY_BY_ANGLE_4: dict[int, str] = {0: "right", 90: "up", 180: "left", 270: "down"}

#: `questplus`'s log10-scale Weibull slope is the classic Weibull "beta" shape
#: exponent applied to the *linear* (untransformed) stimulus ratio: passing
#: `x = log10(gap_arcmin)` with `stim_scale="log10"` makes
#: `10**(slope*(x-threshold)) == (gap_arcmin / threshold_gap_arcmin)**slope`
#: exactly (see `questplus_weibull_x_at_p`'s docstring) -- i.e. this *is*
#: the standard literature Weibull-on-linear-MAR acuity model, just driven
#: adaptively on the convenient log axis (Watson & Pelli 1983's QUEST
#: design). Typical human psychometric slopes for Landolt C/letter acuity
#: tasks fall around beta ~= 3 (comparable to contrast-detection tasks;
#: `questplus`'s own default `slope=3.5`); this grid spans a wide range
#: around that (0.5-6) to comfortably bracket both very shallow and very
#: steep observers without materially affecting how much of the domain the
#: rising limb of the function occupies (verified: at beta ~1-5, the
#: function sweeps from near guess rate to near ceiling within a domain of
#: about 1.8 logMAR units, matching the difference between
#: `logmar_domain_min` and `logmar_domain_max`).
DEFAULT_SLOPE_VALUES = [float(v) for v in np.linspace(0.5, 6.0, 6)]
DEFAULT_LAPSE_RATE_VALUES = [0.0, 0.02, 0.04]
DEFAULT_N_INTENSITY_LEVELS = 25
DEFAULT_N_THRESHOLD_LEVELS = 17


class VisualAcuityParams(BaseModel):
    """Configurable parameters for the Landolt C visual acuity test.

    Attributes:
        max_trials: QUEST+ main-block trial budget. FrACT-style acuity
            estimates typically stabilize within a few dozen trials
            (Bach 1996); the Phase 2A task's default is "about 30".
        n_practice_trials: Number of unscored practice trials (with
            feedback) run before the main block.
        n_orientations: `8` (default, 8AFC, guess rate 1/8) or `4` (4AFC,
            guess rate 1/4, also accepts arrow keys).
        logmar_domain_min: Nominal smallest (sharpest) logMAR to allow,
            before display-resolution clipping (see module docstring).
        logmar_domain_max: Nominal largest (coarsest) logMAR to allow,
            before on-screen-fit clipping (see module docstring).
        min_gap_px: Smallest gap, in pixels, this test will present,
            regardless of `logmar_domain_min` -- see this field's own
            longer docstring below for the antialiasing rationale.
        weber_contrast: Weber contrast of the optotype against the
            background, `(L_stroke - L_bg) / L_bg`. Defaults to -0.99 (near
            -100%, i.e. a very dark optotype on a bright background), per
            the Phase 2A task; gamma calibration is not required for this
            test (see `TestRequirements`), so this is an uncorrected linear
            approximation rather than a photometrically exact contrast --
            adequate at this deliberately extreme, far-from-threshold
            contrast (see `optotype.render_landolt_c`'s docstring).
        supersample: Antialiasing supersampling factor for the optotype
            texture (see `optotype.render_landolt_c`).
        max_response_ms: Maximum time to wait for a response after
            stimulus onset, in milliseconds, before the trial times out
            with no response (converted to frames via
            `DisplayGeometry.frames_for_ms`). Defaults to 30000 (30 s),
            matching FrACT's own generous per-trial cap (Bach 1996): the
            stimulus itself normally stays visible until response (see
            `fixed_stimulus_duration_ms`).
        fixed_stimulus_duration_ms: If set, the optotype is shown for
            exactly this many milliseconds (converted to frames) and then
            replaced by a blank screen for the remainder of the response
            window, instead of remaining visible until response. `None`
            (the default) reproduces FrACT's usual "stimulus visible until
            response" behavior.
        fixation_ms: Fixation cross duration before the optotype, in
            milliseconds.
        iti_ms: Inter-trial interval duration, in milliseconds.
        catch_logmar: Suprathreshold logMAR to use for catch trials, or
            `None` (the default) to use this instance's clipped
            `logmar_domain_max` -- a large, easy-to-see optotype -- per
            the Phase 2A task's "a large fixed size" option (simpler and
            more robust than tracking a live running threshold estimate;
            matches the convention `tests_catalog._example` uses for its
            own catch trials).
    """

    model_config = ConfigDict(frozen=True)

    max_trials: int = Field(default=30, ge=1, description="QUEST+ trial budget, main block.")
    n_practice_trials: int = Field(default=5, ge=0, description="Unscored practice trials.")
    n_orientations: Literal[4, 8] = Field(
        default=8, description="Number of gap-orientation alternatives (4AFC or 8AFC)."
    )
    logmar_domain_min: float = Field(
        default=-0.5, description="Nominal smallest (sharpest) logMAR, before display clipping."
    )
    logmar_domain_max: float = Field(
        default=1.3, description="Nominal largest (coarsest) logMAR, before on-screen clipping."
    )
    min_gap_px: float = Field(
        default=1.0,
        gt=0,
        description=(
            "Smallest optotype gap this test will present, in pixels. Documented choice: below "
            "about 1 physical pixel, the gap's antialiased edge coverage collapses toward a "
            "single texel's worth of gradient (see optotype.render_landolt_c), so the rendered "
            "gap can no longer be reliably distinguished from a fully closed ring even with "
            "supersampled antialiasing -- there is no meaningful sub-pixel gap detail left to "
            "render, unlike vernier_acuity's hyperacuity offsets (which encode position, not a "
            "geometric feature's presence/absence, and so remain meaningful far below 1 px). "
            "1.0 px is therefore treated as the display's floor for this test, not a tunable "
            "precision knob."
        ),
    )
    weber_contrast: float = Field(
        default=-0.99, lt=0.0, ge=-1.0, description="Weber contrast of optotype vs. background."
    )
    supersample: int = Field(default=4, ge=1, description="Optotype texture antialiasing factor.")
    max_response_ms: float = Field(
        default=30000.0, gt=0, description="Max time to wait for a response, in ms."
    )
    fixed_stimulus_duration_ms: float | None = Field(
        default=None,
        gt=0,
        description="If set, optotype visible for exactly this long, then blanked.",
    )
    fixation_ms: float = Field(default=500.0, ge=0, description="Fixation duration, in ms.")
    iti_ms: float = Field(default=500.0, ge=0, description="Inter-trial interval, in ms.")
    catch_logmar: float | None = Field(
        default=None, description="Suprathreshold logMAR for catch trials, or None for domain max."
    )


@register_test
class VisualAcuityTest(PsychophysicalTest):
    """Landolt C recognition acuity (FrACT method) via QUEST+ over logMAR."""

    spec = TestSpec(
        id="visual_acuity",
        name="Visual Acuity (Landolt C)",
        version="1.0.0",
        domain="acuity",
        description_participant=(
            "A ring with a small gap on one side will appear on the screen. Press the number-pad "
            "key that points toward the gap (for example, the 8 key if the gap points straight "
            "up). The ring will keep getting harder to see as you go -- just do your best guess "
            "when you're not sure."
        ),
        description_technical=(
            "Landolt C recognition acuity threshold via QUEST+ (Watson 2017) on a Weibull "
            "psychometric function over logMAR = log10(gap size in arcmin). ISO 8596 optotype "
            "proportions (outer diameter = 5x gap, stroke width = gap). Default 8AFC (guess rate "
            "1/8); reports threshold at the FrACT criterion (Bach 1996): p-correct midway between "
            "guess rate and 1 - lapse rate."
        ),
        measures="Landolt C recognition acuity threshold (logMAR).",
        output_units="logMAR",
        estimated_minutes=3.0,
        allowed_eyes=["OD", "OS", "OU"],
        requirements=TestRequirements(),  # gamma calibration not required (see module docstring)
        citations=[
            "Bach, M. (1996). The Freiburg Visual Acuity test -- automatic measurement of visual "
            "acuity. Optometry and Vision Science, 73(1), 49-53.",
            "Bach, M. (2007). The Freiburg Visual Acuity Test -- variability unchanged by "
            "post-hoc re-analysis. Graefe's Archive for Clinical and Experimental Ophthalmology, "
            "245(7), 965-971.",
            "ISO 8596:2017. Ophthalmic optics -- Visual acuity testing -- Standard optotype and "
            "its presentation.",
            "Watson, A. B. (2017). QUEST+: A general multidimensional Bayesian adaptive "
            "psychometric method. Journal of Vision, 17(3):10.",
        ],
        params_model=VisualAcuityParams,
    )

    def __init__(
        self,
        params: BaseModel,
        display: Any,
        calibration: Any,
        rng: np.random.Generator,
    ) -> None:
        assert isinstance(params, VisualAcuityParams)
        self.params = params
        self.display = display
        self.calibration = calibration
        self.rng = rng
        self._n_presented = 0
        self._stims: dict[str, Any] = {}

        if params.n_orientations == 8:
            self._angles: list[int] = [0, 45, 90, 135, 180, 225, 270, 315]
            self._angle_by_key: dict[str, int] = {
                key: angle for angle, key in KEY_BY_ANGLE_8.items()
            }
        else:
            self._angles = [0, 90, 180, 270]
            self._angle_by_key = {key: angle for angle, key in KEY_BY_ANGLE_4.items()}
            self._angle_by_key.update({key: angle for angle, key in ARROW_KEY_BY_ANGLE_4.items()})
        self.guess_rate = 1.0 / params.n_orientations

        # Display-clipped domain (see module docstring).
        floor_logmar = min_renderable_logmar(display.px_to_deg, params.min_gap_px)
        self._logmar_min = max(params.logmar_domain_min, floor_logmar)
        max_fit_px = 0.9 * min(display.width_px, display.height_px)
        max_gap_px_on_screen = max_fit_px / landolt_c_geometry(1.0)["outer_diameter_px"]
        fit_logmar_max = float(np.log10(display.px_to_deg(max_gap_px_on_screen) * 60.0))
        self._logmar_max = min(params.logmar_domain_max, fit_logmar_max)
        if self._logmar_max <= self._logmar_min:
            # Degenerate/extreme display geometry: fall back to a minimal
            # 1-logMAR-unit window above the floor rather than raising, so
            # construction never fails outright for an unusual display.
            self._logmar_max = self._logmar_min + 1.0

    def make_procedure(self) -> QuestPlusProcedure:
        intensity_values = [
            float(v)
            for v in np.linspace(self._logmar_min, self._logmar_max, DEFAULT_N_INTENSITY_LEVELS)
        ]
        threshold_values = [
            float(v)
            for v in np.linspace(self._logmar_min, self._logmar_max, DEFAULT_N_THRESHOLD_LEVELS)
        ]
        return QuestPlusProcedure(
            intensity_values=intensity_values,
            intensity_units="logMAR",
            threshold_values=threshold_values,
            slope_values=DEFAULT_SLOPE_VALUES,
            guess_rate=self.guess_rate,
            lapse_rate_values=DEFAULT_LAPSE_RATE_VALUES,
            function="weibull",
            max_trials=self.params.max_trials,
            # stim_scale defaults to "log10": questplus's own Weibull formula (Watson &
            # Pelli 1983's p = 1 - lapse - (1-guess-lapse)*exp(-10**(slope*(x-threshold))))
            # is the correct form precisely when x/threshold are *already* expressed in
            # log10 units of the underlying physical quantity -- true here, since
            # logMAR = log10(gap size in arcmin) already is that log10 quantity. Do not
            # confuse this with vpsych.core.psychometric.PsychometricFunction's own,
            # differently-parameterized "intensity_scale" label (see docs/METHODS.md's
            # "Adaptive procedures" section), which this module does not use directly.
        )

    def make_catch_trial_intensity(self) -> float:
        return (
            self.params.catch_logmar if self.params.catch_logmar is not None else self._logmar_max
        )

    def build_stimuli(self, win: Any) -> dict[str, Any]:
        if win is None:
            return {}
        from psychopy import visual

        fixation = visual.TextStim(win, text="+", height=20, color=-1.0)
        self._stims = {"fixation": fixation}
        return self._stims

    def _build_optotype_stim(self, win: Any, gap_px: float, orientation_deg: float) -> Any:
        from psychopy import visual

        geom = landolt_c_geometry(gap_px)
        texture_size_px = max(8, int(np.ceil(geom["outer_diameter_px"] * 1.2)))
        texture = render_landolt_c(
            texture_size_px,
            gap_px,
            orientation_deg,
            weber_contrast=self.params.weber_contrast,
            supersample=self.params.supersample,
        )
        return visual.ImageStim(
            win,
            # render_landolt_c returns rows top-first, but PsychoPy/OpenGL treats
            # array row 0 as the bottom of the texture; flip so the gap appears
            # at the orientation the numpad mapping expects.
            image=np.flipud(texture),
            size=(texture_size_px, texture_size_px),
            units="pix",
            interpolate=True,
        )

    def present(
        self, win: Any, intensity_or_stimulus: Any, trial_ctx: dict[str, Any]
    ) -> PresentedTrial:
        self._n_presented += 1
        rng = trial_ctx["rng"]
        logmar = float(intensity_or_stimulus)
        gap_arcmin = 10.0**logmar
        gap_deg = gap_arcmin / 60.0
        gap_px = self.display.deg_to_px(gap_deg)
        orientation = int(self._angles[int(rng.integers(len(self._angles)))])
        geom = landolt_c_geometry(gap_px)

        stimulus_params: dict[str, Any] = {
            "correct_response": orientation,
            "intensity": logmar,
            "gap_arcmin": gap_arcmin,
            "gap_px": gap_px,
            "outer_diameter_px": geom["outer_diameter_px"],
            "stroke_width_px": geom["stroke_width_px"],
            "orientation_deg": orientation,
            "weber_contrast": self.params.weber_contrast,
        }
        trial_ctx["stimulus_params"] = stimulus_params
        trial_ctx["correct_response"] = orientation

        observer = trial_ctx["simulated_observer"]
        if observer is not None:
            is_correct = observer.decide_correct(stimulus_params, rng)
            response = self.simulated_response(is_correct, stimulus_params, rng)
            timeline = trial_ctx["timeline"]
            return PresentedTrial(
                response=response,
                rt_s=0.6,
                stimulus_onset_s=float(self._n_presented),
                n_dropped_frames=0,
                frame_intervals_s=[1.0 / 60.0] * max(timeline.stimulus_frames, 1),
            )

        keyboard = trial_ctx["keyboard"]
        fixation = self._stims["fixation"]
        fixation_frames = self.display.frames_for_ms(self.params.fixation_ms)
        iti_frames = self.display.frames_for_ms(self.params.iti_ms)
        max_response_frames = self.display.frames_for_ms(self.params.max_response_ms)
        stim_only_frames = (
            self.display.frames_for_ms(self.params.fixed_stimulus_duration_ms)
            if self.params.fixed_stimulus_duration_ms is not None
            else None
        )
        accepted_keys = list(self.response_keys())

        for _ in range(fixation_frames):
            fixation.draw()
            win.flip()

        optotype = self._build_optotype_stim(win, gap_px, float(orientation))
        keyboard.clearEvents()

        onset_s: float | None = None
        response_key: str | None = None
        rt_s: float | None = None
        flip_times: list[float] = []
        for frame in range(max_response_frames):
            if stim_only_frames is None or frame < stim_only_frames:
                optotype.draw()
            flip_time = win.flip()
            flip_times.append(flip_time)
            if frame == 0:
                onset_s = flip_time
                keyboard.clearEvents()
                keyboard.clock.reset()
            keys = keyboard.getKeys(keyList=accepted_keys, waitRelease=False)
            if keys:
                response_key = keys[0].name
                rt_s = keys[0].rt if keys[0].rt is not None else (flip_time - (onset_s or 0.0))
                break

        response = self._angle_by_key.get(response_key) if response_key is not None else None

        for _ in range(iti_frames):
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
        return list(self._angle_by_key.keys())

    def simulated_response(
        self, correct: bool, stimulus_params: dict[str, Any], rng: np.random.Generator
    ) -> Any:
        correct_response = stimulus_params["correct_response"]
        if correct:
            return correct_response
        alternatives = [a for a in self._angles if a != correct_response]
        idx = int(rng.integers(len(alternatives)))
        return alternatives[idx]

    def score(self, response: Any, stimulus_params: dict[str, Any]) -> bool:
        if response is None:
            return False
        return bool(response == stimulus_params["correct_response"])

    def instructions(self) -> str:
        return self.spec.description_participant

    def _fract_criterion_threshold(
        self, procedure: QuestPlusProcedure, raw_estimate: ThresholdEstimate, lapse: float
    ) -> tuple[float, float, float, float]:
        """Convert QUEST+'s native-Weibull threshold to the FrACT criterion (see module docstring).

        Returns `(reported_threshold, reported_ci_low, reported_ci_high, target_p_correct)`.
        Delegates to `QuestPlusProcedure.intensity_at_p_correct`, the shared,
        tested inversion of `questplus`'s own Weibull formula (see that
        method's docstring for the CI-shift approximation it uses).
        """
        target_p = self.guess_rate + 0.5 * (1.0 - self.guess_rate - lapse)
        reported, ci_low, ci_high = procedure.intensity_at_p_correct(target_p)
        return reported, ci_low, ci_high, target_p

    def summarize(self, trials: pd.DataFrame) -> TestSummary:
        main = trials[trials["block"] == "main"]
        non_catch = main[~main["is_catch"]].sort_values("trial_index")
        catch = main[main["is_catch"]]

        procedure = self.make_procedure()
        for _, row in non_catch.iterrows():
            procedure.update(float(row["intensity"]), bool(row["correct"]))
        raw_estimate = procedure.estimate()

        slope = float(raw_estimate.extra["slope"])
        lapse = float(raw_estimate.extra["lapse_rate"])
        reported, ci_low, ci_high, target_p = self._fract_criterion_threshold(
            procedure, raw_estimate, lapse
        )
        decimal_acuity = float(10.0 ** (-reported))
        snellen_denominator_20 = 20.0 / decimal_acuity
        snellen_denominator_6 = 6.0 / decimal_acuity

        n_catch = len(catch)
        catch_lapse_rate = float((~catch["correct"]).mean()) if n_catch else 0.0
        dropped_fraction = (
            float(trials["n_dropped_frames_trial"].sum()) / len(trials) if len(trials) else 0.0
        )

        quality_flags: list[QualityFlag] = compute_quality_flags(
            catch_lapse_rate=catch_lapse_rate,
            n_catch=n_catch,
            dropped_fraction=dropped_fraction,
            threshold=reported,
            range_min=self._logmar_min,
            range_max=self._logmar_max,
            n_trials=len(non_catch),
            min_trials=self.params.max_trials,
        )
        # Dynamic display-resolution check (Phase 2A task requirement): flag when
        # the measured threshold sits within 0.1 logMAR of the smallest gap this
        # display/viewing-distance combination can render.
        floor_logmar = min_renderable_logmar(self.display.px_to_deg, self.params.min_gap_px)
        if abs(reported - floor_logmar) <= 0.1:
            quality_flags = [
                *quality_flags,
                QualityFlag(
                    code="display_resolution_limited",
                    severity="warning",
                    message=(
                        f"The measured threshold ({reported:.3f} logMAR) is within 0.1 logMAR of "
                        f"this display's smallest renderable gap ({floor_logmar:.3f} logMAR at "
                        f"{self.params.min_gap_px:g} px, {self.display.viewing_distance_cm:g} cm "
                        "viewing distance). Threshold limited by display resolution: increase "
                        "viewing distance."
                    ),
                ),
            ]

        eye = str(trials["eye"].iloc[0]) if len(trials) else "OU"
        return TestSummary(
            task_id=self.spec.id,
            task_version=self.spec.version,
            eye=eye,
            run=1,
            estimate=raw_estimate.model_copy(
                update={
                    "value": reported,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "method": "quest_plus_posterior_mean_at_fract_criterion",
                    "extra": {
                        **raw_estimate.extra,
                        "raw_questplus_native_threshold_logmar": raw_estimate.value,
                        "target_p_correct": target_p,
                        "criterion": (
                            "p-correct midway between guess rate and 1 - lapse rate "
                            "(Bach 1996 FrACT criterion)"
                        ),
                    },
                }
            ),
            fit_params={
                "decimal_acuity": decimal_acuity,
                "snellen_20": f"20/{snellen_denominator_20:.1f}",
                "snellen_6": f"6/{snellen_denominator_6:.1f}",
                "slope": slope,
                "lapse_rate": lapse,
                "guess_rate": self.guess_rate,
                "target_p_correct": target_p,
                "logmar_domain_min": self._logmar_min,
                "logmar_domain_max": self._logmar_max,
            },
            gof={},
            quality_flags=quality_flags,
            n_trials=len(non_catch),
            n_catch=n_catch,
            catch_lapse_rate=catch_lapse_rate,
            analysis_version="1.0.0",
        )
