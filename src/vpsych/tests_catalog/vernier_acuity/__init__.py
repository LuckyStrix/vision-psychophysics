"""Vernier acuity (hyperacuity): 2AFC lateral offset detection via QUEST+.

Two short, high-contrast vertical line segments are shown one above the
other, separated by a small vertical gap; the lower segment is offset
slightly left or right of the upper one, and the participant reports which
way. Vernier offset thresholds are the classic demonstration of
*hyperacuity*: observers reliably detect offsets far smaller than the
photoreceptor/cone spacing that limits ordinary (Landolt C/letter) acuity
(Westheimer, G. (1975). Editorial: Visual acuity and hyperacuity.
Investigative Ophthalmology, 14(8), 570-572; Westheimer, G. (1979). The
spatial sense of the eye. Investigative Ophthalmology & Visual Science,
18(9), 893-912; McKee, S. P., & Westheimer, G. (1978). Improvement in
vernier acuity with practice. Perception & Psychophysics, 24(3), 258-262,
https://doi.org/10.3758/BF03204247), typically 3-10 arcsec in normal
observers under good conditions -- a fraction of a display pixel at normal
viewing distances, which is why this test's rendering (see `texture.py`)
is built around sub-pixel-accurate, gamma-linearized antialiasing rather
than whole-pixel stimulus positioning.

**Procedure**: QUEST+ (Watson, A. B. (2017). QUEST+: A general
multidimensional Bayesian adaptive psychometric method. Journal of Vision,
17(3):10) on `log10(offset in arcsec)`, 2AFC (guess rate 0.5), Weibull
psychometric function. `summarize` reports the offset at 75% correct, in
arcsec, with a credible interval -- see `_questplus_weibull_x_at_p` (this
test's own module, mirroring `visual_acuity`'s identically-named helper;
duplicated rather than imported from shared code, per the Phase 2A task's
instruction not to modify `core/`, since this is `questplus`'s own Weibull
parameterization, not `vpsych.core.psychometric`'s -- see that helper's
docstring for the full derivation).

**Sub-pixel rendering and gamma linearization**: see `texture.py`'s module
docstring for the full method and citations. In short: the offset is
encoded as an exact, analytically area-sampled edge-coverage texture (not
GPU/backend antialiasing, and not stochastic supersampling), and the
texture's per-pixel values are the coverage-weighted *linear* luminance
mixture of foreground/background -- correct display output for this
depends on the window's own gamma ramp having already linearized hardware
luminance from a real calibration (`docs/WRITING_A_TEST.md` section 7),
which is why this test declares `needs_gamma_calibration=True` (grade B
minimum, i.e. either a photometer or the psychophysical half-luminance
bisection method -- see `docs/CALIBRATION.md`).

**Position jitter**: the whole two-segment stimulus is displaced by a small
random amount each trial (`position_jitter_arcmin`) so absolute screen
position can't be used as a response cue. This jitter is rounded to whole
pixels (the `ImageStim`'s position, not the texture) so it never perturbs
the texture's own analytically-encoded sub-pixel offset -- see `present`'s
inline comment for why sub-pixel jitter would be self-defeating here.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from vpsych.core.procedures.questplus_procedure import QuestPlusProcedure
from vpsych.data.quality import compute_quality_flags
from vpsych.data.schemas import QualityFlag, TestSummary
from vpsych.tests_catalog.base import (
    PresentedTrial,
    PsychophysicalTest,
    TestRequirements,
    TestSpec,
    register_test,
)
from vpsych.tests_catalog.vernier_acuity.texture import render_vertical_line_texture

GUESS_RATE = 0.5  # 2AFC chance rate.
DEFAULT_LAPSE_RATE_VALUES = [0.0, 0.02, 0.04]
DEFAULT_N_INTENSITY_LEVELS = 25
DEFAULT_N_THRESHOLD_LEVELS = 15
#: `questplus`'s log10-scale Weibull slope is the classic Weibull "beta" shape
#: exponent on the *linear* stimulus ratio (offset_arcsec / threshold_arcsec)
#: -- see `visual_acuity`'s identically-derived `DEFAULT_SLOPE_VALUES`
#: comment for the full reasoning; a Vernier psychometric function is
#: typically steep (small dynamic range between threshold and ceiling), so
#: this grid leans toward somewhat higher beta than the acuity test's.
DEFAULT_SLOPE_VALUES = [float(v) for v in np.linspace(1.0, 8.0, 6)]


def _questplus_weibull_x_at_p(
    threshold: float, slope: float, guess: float, lapse: float, p_target: float
) -> float:
    """Invert `questplus`'s own log10-scale Weibull formula for `x` at a target p-correct.

    Identical derivation to `vpsych.tests_catalog.visual_acuity`'s helper of
    the same name (duplicated locally rather than imported, since it is not
    shared/core code and both tests were built independently and in
    parallel -- see that module's docstring for the full derivation):

        p(x) = 1 - lapse - (1 - guess - lapse) * exp(-10**(slope * (x - threshold)))
        x = threshold + log10(-ln((1 - p_target - lapse) / (1 - guess - lapse))) / slope
    """
    denom = 1.0 - guess - lapse
    q = (1.0 - p_target - lapse) / denom if denom > 0 else 0.5
    q = min(max(q, 1e-12), 1.0 - 1e-12)
    return threshold + math.log10(-math.log(q)) / slope


class VernierAcuityParams(BaseModel):
    """Configurable parameters for the Vernier (hyperacuity) offset test.

    Attributes:
        max_trials: QUEST+ main-block trial budget.
        n_practice_trials: Number of unscored practice trials (with
            feedback) run before the main block.
        line_length_arcmin: Length of each line segment, in arcmin. Default
            15, within the 10-20 arcmin range cited by Westheimer (1979)/
            McKee & Westheimer (1978) for classic 2-line Vernier stimuli.
        line_width_arcmin: Width (thickness) of each line segment, in
            arcmin. Default 1.
        gap_arcmin: Vertical gap between the two segments, in arcmin.
            Default 3, within the 2-4 arcmin range typically used (a gap
            that is too small risks the segments visually merging; too
            large weakens the positional comparison).
        position_jitter_arcmin: Half-width of the uniform random jitter
            applied to the whole stimulus's on-screen position each trial,
            in arcmin, so absolute position can't be used as a cue.
            Rounded to whole pixels at render time (see module docstring).
        weber_contrast: Weber contrast of the line segments against the
            background. Defaults to -0.99 (near-maximum, high contrast, per
            the Phase 2A task).
        log_offset_domain_min: Nominal smallest `log10(offset_arcsec)` QUEST+
            may present. Default 0.0 (1 arcsec).
        log_offset_domain_max: Nominal largest `log10(offset_arcsec)` QUEST+
            may present. Default 2.5 (~316 arcsec).
        stimulus_duration_ms: Duration the stimulus is visible, in
            milliseconds (converted to frames). Default 175 (within the
            150-200 ms range that limits involuntary eye movements/
            saccades while remaining within the temporal integration window
            classic Vernier studies used; Westheimer, G., & McKee, S. P.
            (1977). Integration regions for visual hyperacuity. Vision
            Research, 17(1), 89-93, https://doi.org/10.1016/0042-6989(77)90209-9,
            report Vernier thresholds improving with stimulus duration up to
            about 100 ms and then plateauing).
        response_timeout_ms: Maximum time to wait for a response after the
            stimulus is blanked, in milliseconds (converted to frames).
        fixation_ms: Fixation cross duration before the stimulus, in ms.
        iti_ms: Inter-trial interval duration, in ms.
        catch_log_offset: Suprathreshold `log10(offset_arcsec)` for catch
            trials, or `None` (default) to use this instance's
            `log_offset_domain_max` (a large, easy offset).
    """

    model_config = ConfigDict(frozen=True)

    max_trials: int = Field(default=40, ge=1, description="QUEST+ trial budget, main block.")
    n_practice_trials: int = Field(default=5, ge=0, description="Unscored practice trials.")
    line_length_arcmin: float = Field(default=15.0, gt=0, description="Segment length, arcmin.")
    line_width_arcmin: float = Field(default=1.0, gt=0, description="Segment width, arcmin.")
    gap_arcmin: float = Field(default=3.0, gt=0, description="Vertical gap between segments.")
    position_jitter_arcmin: float = Field(
        default=4.0, ge=0, description="Half-width of per-trial position jitter, arcmin."
    )
    weber_contrast: float = Field(
        default=-0.99, lt=0.0, ge=-1.0, description="Weber contrast of lines vs. background."
    )
    log_offset_domain_min: float = Field(
        default=0.0, description="Smallest log10(offset in arcsec) QUEST+ may present."
    )
    log_offset_domain_max: float = Field(
        default=2.5, description="Largest log10(offset in arcsec) QUEST+ may present."
    )
    stimulus_duration_ms: float = Field(
        default=175.0, gt=0, description="Stimulus presentation duration, ms."
    )
    response_timeout_ms: float = Field(
        default=5000.0, gt=0, description="Max time to wait for a response after blank, ms."
    )
    fixation_ms: float = Field(default=500.0, ge=0, description="Fixation duration, ms.")
    iti_ms: float = Field(default=500.0, ge=0, description="Inter-trial interval, ms.")
    catch_log_offset: float | None = Field(
        default=None, description="Suprathreshold log10(offset_arcsec) for catch trials."
    )


@register_test
class VernierAcuityTest(PsychophysicalTest):
    """Vernier (hyperacuity) lateral-offset detection via QUEST+ on log10(offset_arcsec)."""

    spec = TestSpec(
        id="vernier_acuity",
        name="Vernier Acuity (Hyperacuity)",
        version="1.0.0",
        domain="hyperacuity",
        description_participant=(
            "You'll see two short lines, one above the other. The bottom line will be shifted "
            "slightly to the left or right of the top one. Press the left or right arrow key to "
            "say which way it's shifted. The shift gets smaller and harder to see as you go -- "
            "just do your best guess when you're not sure."
        ),
        description_technical=(
            "2AFC Vernier (lateral offset) hyperacuity threshold via QUEST+ (Watson 2017) on a "
            "Weibull psychometric function over log10(offset in arcsec). Guess rate 0.5. "
            "Sub-pixel-accurate, gamma-linearized rendering (see texture.py); reports offset at "
            "75% correct."
        ),
        measures="Vernier (lateral offset) hyperacuity threshold, in arcsec.",
        output_units="arcsec",
        estimated_minutes=3.0,
        allowed_eyes=["OD", "OS", "OU"],
        requirements=TestRequirements(
            needs_gamma_calibration=True,
            min_luminance_grade="B",
        ),
        citations=[
            "Westheimer, G. (1975). Editorial: Visual acuity and hyperacuity. Investigative "
            "Ophthalmology, 14(8), 570-572.",
            "Westheimer, G. (1979). The spatial sense of the eye. Investigative Ophthalmology & "
            "Visual Science, 18(9), 893-912.",
            "McKee, S. P., & Westheimer, G. (1978). Improvement in vernier acuity with practice. "
            "Perception & Psychophysics, 24(3), 258-262.",
            "Westheimer, G., & McKee, S. P. (1977). Integration regions for visual hyperacuity. "
            "Vision Research, 17(1), 89-93.",
            "Lloyd, C., Winterbottom, M., Gaska, J., & Williams, L. (2015). Effects of display "
            "pixel pitch and antialiasing on threshold vernier acuity. Proceedings of the IMAGE "
            "Society Annual Conference, Dayton, OH.",
            "Watson, A. B. (2017). QUEST+: A general multidimensional Bayesian adaptive "
            "psychometric method. Journal of Vision, 17(3):10.",
        ],
        params_model=VernierAcuityParams,
    )

    def __init__(
        self,
        params: BaseModel,
        display: Any,
        calibration: Any,
        rng: np.random.Generator,
    ) -> None:
        assert isinstance(params, VernierAcuityParams)
        self.params = params
        self.display = display
        self.calibration = calibration
        self.rng = rng
        self._n_presented = 0
        self._stims: dict[str, Any] = {}

        self._length_px = display.deg_to_px(params.line_length_arcmin / 60.0)
        self._width_px = display.deg_to_px(params.line_width_arcmin / 60.0)
        self._gap_px = display.deg_to_px(params.gap_arcmin / 60.0)
        self._jitter_px = display.deg_to_px(params.position_jitter_arcmin / 60.0)

        # Fixed canvas size across trials (see module docstring: keeping the
        # stimulus's overall apparent size constant avoids a size-based cue).
        # Width sized to fit the domain's largest possible offset.
        max_offset_arcsec = 10.0**params.log_offset_domain_max
        max_offset_px = display.deg_to_px(max_offset_arcsec / 3600.0)
        margin_px = 4.0
        self._canvas_width_px = int(np.ceil(2.0 * max_offset_px + self._width_px + 2 * margin_px))
        self._canvas_height_px = int(np.ceil(2.0 * self._length_px + self._gap_px + 2 * margin_px))

    def make_procedure(self) -> QuestPlusProcedure:
        intensity_values = [
            float(v)
            for v in np.linspace(
                self.params.log_offset_domain_min,
                self.params.log_offset_domain_max,
                DEFAULT_N_INTENSITY_LEVELS,
            )
        ]
        threshold_values = [
            float(v)
            for v in np.linspace(
                self.params.log_offset_domain_min,
                self.params.log_offset_domain_max,
                DEFAULT_N_THRESHOLD_LEVELS,
            )
        ]
        return QuestPlusProcedure(
            intensity_values=intensity_values,
            intensity_units="log10_offset_arcsec",
            threshold_values=threshold_values,
            slope_values=DEFAULT_SLOPE_VALUES,
            guess_rate=GUESS_RATE,
            lapse_rate_values=DEFAULT_LAPSE_RATE_VALUES,
            function="weibull",
            max_trials=self.params.max_trials,
            # stim_scale defaults to "log10" -- correct here since intensity IS already
            # log10(offset_arcsec); see visual_acuity's identical reasoning.
        )

    def make_catch_trial_intensity(self) -> float:
        return (
            self.params.catch_log_offset
            if self.params.catch_log_offset is not None
            else self.params.log_offset_domain_max
        )

    def build_stimuli(self, win: Any) -> dict[str, Any]:
        if win is None:
            return {}
        from psychopy import visual

        fixation = visual.TextStim(win, text="+", height=20, color=-1.0)
        self._stims = {"fixation": fixation}
        return self._stims

    def _render_stimulus_texture(self, offset_px: float, direction: int) -> np.ndarray:
        """Build the two-segment Vernier texture for one trial.

        The upper segment is fixed at the canvas's horizontal center; the
        lower segment is centered at `center + direction * offset_px`
        (`direction` is +1 for "right", -1 for "left"). Both segments'
        edges are area-sampled (see `texture.render_vertical_line_texture`)
        so `offset_px` (which is generally sub-pixel) is encoded exactly.
        """
        background_level = 1.0
        foreground_level = background_level * (1.0 + self.params.weber_contrast)
        center_x = self._canvas_width_px / 2.0
        top_y_start = self._canvas_height_px / 2.0 - self._gap_px / 2.0 - self._length_px
        top_y_end = self._canvas_height_px / 2.0 - self._gap_px / 2.0
        bottom_y_start = self._canvas_height_px / 2.0 + self._gap_px / 2.0
        bottom_y_end = bottom_y_start + self._length_px

        upper = render_vertical_line_texture(
            self._canvas_width_px,
            self._canvas_height_px,
            center_x_px=center_x,
            width_px=self._width_px,
            y_start_px=top_y_start,
            y_end_px=top_y_end,
            background_level=background_level,
            foreground_level=foreground_level,
        )
        lower = render_vertical_line_texture(
            self._canvas_width_px,
            self._canvas_height_px,
            center_x_px=center_x + direction * offset_px,
            width_px=self._width_px,
            y_start_px=bottom_y_start,
            y_end_px=bottom_y_end,
            background_level=background_level,
            foreground_level=foreground_level,
        )
        # Combine: a pixel belongs to whichever segment covers it (segments
        # don't overlap vertically, since gap_px > 0 separates them), so the
        # darker (more-covered) of the two per pixel is the correct combine.
        combined: np.ndarray = np.minimum(upper, lower)
        return combined

    def present(
        self, win: Any, intensity_or_stimulus: Any, trial_ctx: dict[str, Any]
    ) -> PresentedTrial:
        self._n_presented += 1
        rng = trial_ctx["rng"]
        log_offset = float(intensity_or_stimulus)
        offset_arcsec = 10.0**log_offset
        offset_px = self.display.deg_to_px(offset_arcsec / 3600.0)
        direction = 1 if rng.random() < 0.5 else -1
        correct_side = "right" if direction == 1 else "left"
        jitter_px = float(round(rng.uniform(-self._jitter_px, self._jitter_px)))

        stimulus_params: dict[str, Any] = {
            "correct_response": correct_side,
            "intensity": log_offset,
            "offset_arcsec": offset_arcsec,
            "offset_px": offset_px,
            "direction": correct_side,
            "position_jitter_px": jitter_px,
        }
        trial_ctx["stimulus_params"] = stimulus_params
        trial_ctx["correct_response"] = correct_side

        observer = trial_ctx["simulated_observer"]
        if observer is not None:
            is_correct = observer.decide_correct(stimulus_params, rng)
            response = self.simulated_response(is_correct, stimulus_params, rng)
            timeline = trial_ctx["timeline"]
            return PresentedTrial(
                response=response,
                rt_s=0.5,
                stimulus_onset_s=float(self._n_presented),
                n_dropped_frames=0,
                frame_intervals_s=[1.0 / 60.0] * max(timeline.stimulus_frames, 1),
            )

        from psychopy import visual

        keyboard = trial_ctx["keyboard"]
        fixation = self._stims["fixation"]
        fixation_frames = self.display.frames_for_ms(self.params.fixation_ms)
        stimulus_frames = self.display.frames_for_ms(self.params.stimulus_duration_ms)
        response_timeout_frames = self.display.frames_for_ms(self.params.response_timeout_ms)
        iti_frames = self.display.frames_for_ms(self.params.iti_ms)

        for _ in range(fixation_frames):
            fixation.draw()
            win.flip()

        texture = self._render_stimulus_texture(offset_px, direction)
        stim = visual.ImageStim(
            win,
            image=texture,
            size=(self._canvas_width_px, self._canvas_height_px),
            # Whole-pixel-only jitter: keeps the ImageStim's own screen-space
            # position aligned to the physical pixel grid, so the analytically
            # encoded sub-pixel offset inside the texture is not perturbed by
            # additional fractional GPU-side resampling (see module docstring).
            pos=(jitter_px, 0.0),
            units="pix",
            interpolate=True,
        )

        onset_s: float | None = None
        for frame in range(stimulus_frames):
            stim.draw()
            flip_time = win.flip()
            if frame == 0:
                onset_s = flip_time
                keyboard.clearEvents()

        response_key: str | None = None
        rt_s: float | None = None
        for _ in range(response_timeout_frames):
            flip_time = win.flip()  # blank response window
            keys = keyboard.getKeys(keyList=self.response_keys(), waitRelease=False)
            if keys:
                response_key = keys[0].name
                rt_s = keys[0].rt if keys[0].rt is not None else (flip_time - (onset_s or 0.0))
                break

        for _ in range(iti_frames):
            win.flip()

        return PresentedTrial(
            response=response_key,
            rt_s=rt_s,
            stimulus_onset_s=onset_s or 0.0,
            n_dropped_frames=0,
            frame_intervals_s=[1.0 / self.display.refresh_hz] * max(stimulus_frames, 1),
        )

    def response_keys(self) -> list[str]:
        return ["left", "right"]

    def score(self, response: Any, stimulus_params: dict[str, Any]) -> bool:
        if response is None:
            return False
        return bool(response == stimulus_params["correct_response"])

    def instructions(self) -> str:
        return self.spec.description_participant

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
        target_p = 0.75
        reported_log_offset = _questplus_weibull_x_at_p(
            raw_estimate.value, slope, GUESS_RATE, lapse, target_p
        )
        shift = reported_log_offset - raw_estimate.value
        ci_low = raw_estimate.ci_low + shift
        ci_high = raw_estimate.ci_high + shift

        offset_arcsec = 10.0**reported_log_offset
        ci_low_arcsec = 10.0**ci_low
        ci_high_arcsec = 10.0**ci_high

        n_catch = len(catch)
        catch_lapse_rate = float((~catch["correct"]).mean()) if n_catch else 0.0
        dropped_fraction = (
            float(trials["n_dropped_frames_trial"].sum()) / len(trials) if len(trials) else 0.0
        )

        quality_flags: list[QualityFlag] = compute_quality_flags(
            catch_lapse_rate=catch_lapse_rate,
            n_catch=n_catch,
            dropped_fraction=dropped_fraction,
            threshold=reported_log_offset,
            range_min=self.params.log_offset_domain_min,
            range_max=self.params.log_offset_domain_max,
            n_trials=len(non_catch),
            min_trials=self.params.max_trials,
        )
        offset_px_at_threshold = self.display.deg_to_px(offset_arcsec / 3600.0)
        if offset_px_at_threshold < 0.1:
            quality_flags = [
                *quality_flags,
                QualityFlag(
                    code="near_rendering_limit",
                    severity="warning",
                    message=(
                        f"The measured threshold ({offset_arcsec:.2f} arcsec, "
                        f"{offset_px_at_threshold:.3f} px equivalent) is below 0.1 px equivalent "
                        "at this display/viewing distance -- close to the practical limit of "
                        "sub-pixel rendering precision even with area-sampled antialiasing. "
                        "Interpret with some caution."
                    ),
                ),
            ]
        min_recommended_viewing_distance_cm = 100.0
        if self.display.viewing_distance_cm < min_recommended_viewing_distance_cm:
            quality_flags = [
                *quality_flags,
                QualityFlag(
                    code="short_viewing_distance",
                    severity="warning",
                    message=(
                        f"Viewing distance ({self.display.viewing_distance_cm:g} cm) is short "
                        f"for a hyperacuity measurement; consider at least "
                        f"{min_recommended_viewing_distance_cm:g} cm to reduce the physical pixel "
                        "size relative to the offsets being measured."
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
                    "value": offset_arcsec,
                    "ci_low": ci_low_arcsec,
                    "ci_high": ci_high_arcsec,
                    "units": "arcsec",
                    "method": "quest_plus_posterior_mean_at_75pct",
                    "extra": {
                        **raw_estimate.extra,
                        "raw_log10_threshold": raw_estimate.value,
                        "reported_log10_threshold": reported_log_offset,
                        "target_p_correct": target_p,
                    },
                }
            ),
            fit_params={
                "slope": slope,
                "lapse_rate": lapse,
                "guess_rate": GUESS_RATE,
                "target_p_correct": target_p,
                "offset_px_at_threshold": offset_px_at_threshold,
            },
            gof={},
            quality_flags=quality_flags,
            n_trials=len(non_catch),
            n_catch=n_catch,
            catch_lapse_rate=catch_lapse_rate,
            analysis_version="1.0.0",
        )
