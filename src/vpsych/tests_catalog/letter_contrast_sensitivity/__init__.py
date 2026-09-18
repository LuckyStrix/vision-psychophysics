"""Letter contrast sensitivity: a Pelli-Robson analogue with procedural Sloan letters.

A 10-alternative-forced-choice letter-identification task measuring log
contrast sensitivity, in the spirit of the Pelli-Robson chart (Pelli, D. G.,
Robson, J. G., & Wilkins, A. J. (1988). The design of a new letter chart for
measuring contrast sensitivity. Clinical Vision Sciences, 2(3), 187-199),
but driven by QUEST+ trial-by-trial rather than the chart's fixed triplets
-- see `docs/methods/letter_contrast_sensitivity.md` for the full write-up,
including how this differs from an actual Pelli-Robson chart score.

Optotypes
---------
The 10 Sloan letters (Sloan, L. L. (1959). New test charts for the
measurement of visual acuity at far and near distances. American Journal of
Ophthalmology, 48(6), 807-813), as standardized by the NAS-NRC (1980)
Committee on Vision report, rendered **procedurally** from coarse 5x5-grid
stroke definitions (see
`vpsych.tests_catalog._contrast_rendering.render_letter_ink_mask` and its
module docstring) rather than from a bundled font. No font bundling was
used: this repository is GPL-3.0 and public, and no verified OFL/permissive
license for Denis Pelli's actual Sloan font was located during this
implementation, so the safer, license-clean path (an explicitly
approximate procedural rendering, documented as such) was taken instead of
risking an unlicensed or ambiguously-licensed font asset.

Task and procedure
-------------------
10AFC letter identification (type the corresponding key); guess rate 0.1.
QUEST+ (`vpsych.core.procedures.questplus_procedure.QuestPlusProcedure`) on
log10 Weber contrast (dark letter against a mid-gray background), gamma
linearized and noisy-bit dithered exactly as in
`contrast_sensitivity_function` (see `_contrast_rendering`'s module
docstring). Default 40 main trials (see `docs/methods/letter_contrast_sensitivity.md`
for the simulated-recovery numbers this is based on). Catch trials use a
high (near-maximal) Weber contrast.

Scoring and confusions
-----------------------
`score()` only ever checks exact letter identity (no partial credit for a
visually similar wrong letter); `TrialRecord.correct_response` and
`TrialRecord.response` (both single-letter strings) are enough to build a
full confusion matrix (response vs. target) from the trials TSV alone --
this test does not compute one itself, so no letter-similarity assumption
is baked into scoring.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from vpsych.core.procedures.base import ThresholdEstimate
from vpsych.core.procedures.questplus_procedure import QuestPlusProcedure
from vpsych.core.trial import TrialTimeline
from vpsych.data.quality import compute_quality_flags
from vpsych.data.schemas import TestSummary
from vpsych.tests_catalog._contrast_rendering import SLOAN_LETTERS
from vpsych.tests_catalog.base import (
    PresentedTrial,
    PsychophysicalTest,
    TestRequirements,
    TestSpec,
    register_test,
)

GUESS_RATE = 0.1  # 10AFC letter identification.

#: Candidate log10 Weber contrast grid QUEST+ selects from. Lower bound
#: (-2.4, i.e. ~0.4% contrast) is set by what noisy-bit dithering can
#: reliably resolve at the response-window frame counts this test uses (see
#: `vpsych.core.calibration.dither.effective_contrast_resolution` and
#: `docs/methods/letter_contrast_sensitivity.md`); upper bound (-0.05, i.e.
#: ~89% contrast) is just short of a maximal, unmistakably-black letter.
DEFAULT_INTENSITY_VALUES = [float(v) for v in np.linspace(-2.4, -0.05, 25)]
DEFAULT_THRESHOLD_VALUES = [float(v) for v in np.linspace(-2.2, -0.1, 15)]
DEFAULT_SLOPE_VALUES = [float(v) for v in np.linspace(0.2, 1.0, 5)]
DEFAULT_LAPSE_RATE_VALUES = [0.0, 0.02, 0.04]


class LetterCSParams(BaseModel):
    """Configurable parameters for the letter contrast sensitivity test.

    Attributes:
        max_trials: QUEST+ main-block trial budget. Default 40 (see
            `docs/methods/letter_contrast_sensitivity.md` for the
            simulated-recovery justification).
        letter_height_deg: Sloan letter height, degrees of visual angle.
            Default 2.8 deg (matches the Pelli-Robson chart's design angle
            at its intended 1 m viewing distance).
        fixation_duration_ms: Fixation-cross duration before the letter,
            in milliseconds.
        response_timeout_ms: Maximum combined viewing+response time: the
            letter remains visible (like a printed chart) until a response
            key is pressed or this timeout elapses.
        iti_ms: Inter-trial interval, in milliseconds.
        catch_contrast: Weber contrast used for catch trials (high,
            comfortably suprathreshold).
    """

    model_config = ConfigDict(frozen=True)

    max_trials: int = Field(default=40, ge=1, description="QUEST+ main-block trial budget.")
    letter_height_deg: float = Field(
        default=2.8, gt=0, description="Sloan letter height, degrees of visual angle."
    )
    fixation_duration_ms: float = Field(
        default=500.0, ge=0, description="Fixation-cross duration before the letter, ms."
    )
    response_timeout_ms: float = Field(
        default=5000.0, gt=0, description="Max combined letter-visible + response time, ms."
    )
    iti_ms: float = Field(default=500.0, ge=0, description="Inter-trial interval, ms.")
    catch_contrast: float = Field(
        default=0.9, gt=0, le=1.0, description="Weber contrast used for catch trials."
    )


def _letter_response_keys() -> list[str]:
    return [letter.lower() for letter in SLOAN_LETTERS]


@register_test
class LetterContrastSensitivityTest(PsychophysicalTest):
    """Pelli-Robson-analogue 10AFC Sloan letter contrast sensitivity, via QUEST+."""

    spec = TestSpec(
        id="letter_contrast_sensitivity",
        name="Letter Contrast Sensitivity",
        version="0.1.0",
        domain="contrast",
        description_participant=(
            "A single letter will appear in the middle of the screen. It may be very faint. "
            "Type the letter you see on the keyboard (only the letters C D H K N O R S V Z are "
            "used) -- make your best guess even if it is very hard to see."
        ),
        description_technical=(
            "10AFC Sloan letter identification, QUEST+ (Watson 2017) on log10 Weber contrast, "
            "in the spirit of the Pelli-Robson chart (Pelli, Robson & Wilkins 1988) but with a "
            "trial-by-trial adaptive procedure rather than the chart's fixed triplets."
        ),
        measures="Letter contrast sensitivity (log10 CS = -log10 threshold Weber contrast).",
        output_units="log10_contrast_sensitivity",
        estimated_minutes=2.5,
        allowed_eyes=["OD", "OS", "OU"],
        requirements=TestRequirements(
            needs_gamma_calibration=True,
            min_luminance_grade="B",
        ),
        citations=[
            "Pelli, D. G., Robson, J. G., & Wilkins, A. J. (1988). The design of a new letter "
            "chart for measuring contrast sensitivity. Clinical Vision Sciences, 2(3), 187-199.",
            "Sloan, L. L. (1959). New test charts for the measurement of visual acuity at far "
            "and near distances. American Journal of Ophthalmology, 48(6), 807-813. "
            "https://doi.org/10.1016/0002-9394(59)90626-9",
            "National Academy of Sciences - National Research Council, Committee on Vision "
            "(1980). Recommended standard procedures for the clinical measurement and "
            "specification of visual acuity. Advances in Ophthalmology, 41, 103-148.",
            "Watson, A. B. (2017). QUEST+: A general multidimensional Bayesian adaptive "
            "psychometric method. Journal of Vision, 17(3):10. https://doi.org/10.1167/17.3.10",
            "Allard, R., & Faubert, J. (2008). The noisy-bit method for digital displays: "
            "converting a resolution limitation into a pseudo-resolution. Behavior Research "
            "Methods, 40(3), 735-743. https://doi.org/10.3758/BRM.40.3.735",
        ],
        params_model=LetterCSParams,
        hidden=False,
    )

    def __init__(
        self,
        params: BaseModel,
        display: Any,
        calibration: Any,
        rng: np.random.Generator,
    ) -> None:
        assert isinstance(params, LetterCSParams)
        self.params = params
        self.display = display
        self.calibration = calibration
        self.rng = rng
        self._n_presented = 0
        self._stims: dict[str, Any] = {}

    def make_procedure(self) -> QuestPlusProcedure:
        return QuestPlusProcedure(
            intensity_values=DEFAULT_INTENSITY_VALUES,
            intensity_units="log10_weber_contrast",
            threshold_values=DEFAULT_THRESHOLD_VALUES,
            slope_values=DEFAULT_SLOPE_VALUES,
            guess_rate=GUESS_RATE,
            lapse_rate_values=DEFAULT_LAPSE_RATE_VALUES,
            function="weibull",
            max_trials=self.params.max_trials,
        )

    def make_catch_trial_intensity(self) -> float:
        return float(np.log10(self.params.catch_contrast))

    def _timeline(self) -> TrialTimeline:
        return TrialTimeline(
            fixation_frames=self.display.frames_for_ms(self.params.fixation_duration_ms),
            stimulus_frames=self.display.frames_for_ms(self.params.response_timeout_ms),
            response_timeout_frames=None,
            iti_frames=self.display.frames_for_ms(self.params.iti_ms),
        )

    def build_stimuli(self, win: Any) -> dict[str, Any]:
        if win is None:
            return {}
        from psychopy import visual

        size_px = round(self.display.deg_to_px(self.params.letter_height_deg))
        fixation = visual.TextStim(win, text="+", height=20)
        letter_image = visual.ImageStim(win, size=(size_px, size_px), units="pix")
        self._stims = {"fixation": fixation, "letter_image": letter_image, "size_px": size_px}
        return self._stims

    def present(
        self, win: Any, intensity_or_stimulus: Any, trial_ctx: dict[str, Any]
    ) -> PresentedTrial:
        self._n_presented += 1
        rng = trial_ctx["rng"]
        intensity = float(intensity_or_stimulus)  # log10 Weber contrast
        contrast = 10.0**intensity
        letter_idx = int(rng.integers(len(SLOAN_LETTERS)))
        letter = SLOAN_LETTERS[letter_idx]
        correct_response = letter.lower()

        stimulus_params: dict[str, Any] = {
            "correct_response": correct_response,
            "intensity": intensity,
            "contrast": contrast,
            "letter": letter,
            "letter_height_deg": self.params.letter_height_deg,
        }
        trial_ctx["stimulus_params"] = stimulus_params
        trial_ctx["correct_response"] = correct_response

        timeline = self._timeline()

        observer = trial_ctx["simulated_observer"]
        if observer is not None:
            is_correct = observer.decide_correct(stimulus_params, rng)
            response = self.simulated_response(is_correct, stimulus_params, rng)
            return PresentedTrial(
                response=response,
                rt_s=0.6,
                stimulus_onset_s=float(self._n_presented),
                n_dropped_frames=0,
                frame_intervals_s=[1.0 / 60.0] * max(timeline.stimulus_frames, 1),
            )

        from vpsych.tests_catalog._contrast_rendering import (
            dither_frame,
            gamma_channel_model_from_calibration,
            render_letter_ink_mask,
            weber_drive_from_ink_mask,
        )

        gamma_model = gamma_channel_model_from_calibration(self.calibration.gamma)
        size_px = self._stims["size_px"]
        ink_mask = render_letter_ink_mask(letter, size_px)

        fixation = self._stims["fixation"]
        image_stim = self._stims["letter_image"]
        keyboard = trial_ctx["keyboard"]

        for _ in range(timeline.fixation_frames):
            fixation.draw()
            win.flip()

        keyboard.clearEvents()
        onset_s = None
        response = None
        rt_s = None
        max_frames = timeline.stimulus_frames
        for frame in range(max_frames):
            # Dither is refreshed every frame the letter is visible (self-paced,
            # like a printed chart), matching contrast_sensitivity_function's
            # convention -- see _contrast_rendering.dither_frame's docstring.
            drive = weber_drive_from_ink_mask(ink_mask, contrast, gamma_model)
            dithered = dither_frame(drive, rng)
            image_stim.image = dithered * 2.0 - 1.0
            image_stim.draw()
            flip_time = win.flip()
            if frame == 0:
                onset_s = flip_time
            keys = keyboard.getKeys(keyList=self.response_keys(), timeStamped=True)
            if keys:
                response, rt_s = keys[0][0], keys[0][1] - (onset_s or 0.0)
                break

        for _ in range(timeline.iti_frames):
            win.flip()

        return PresentedTrial(
            response=response,
            rt_s=rt_s,
            stimulus_onset_s=onset_s or 0.0,
            n_dropped_frames=0,
            frame_intervals_s=[1.0 / self.display.refresh_hz] * max_frames,
        )

    def response_keys(self) -> list[str]:
        return _letter_response_keys()

    def score(self, response: Any, stimulus_params: dict[str, Any]) -> bool:
        return bool(response == stimulus_params["correct_response"])

    def instructions(self) -> str:
        return self.spec.description_participant

    def summarize(self, trials: pd.DataFrame) -> TestSummary:
        """Recompute log contrast sensitivity by replaying a fresh QUEST+ over `trials`.

        Threshold criterion: `QuestPlusProcedure.estimate()` reports the
        fitted Weibull-family psychometric function's F(0)=0.5 crossing (in
        log10 Weber contrast) -- for guess=0.1 and a small lapse rate, this
        corresponds to roughly 55% correct, *not* the 75%-correct point
        conventionally used for 2AFC tasks in this suite, and *not*
        equivalent to a real Pelli-Robson chart's triplet-scoring criterion
        (see `docs/methods/letter_contrast_sensitivity.md`). Log contrast
        sensitivity is then `-1 * (that log10 Weber contrast threshold)`,
        with the CI transformed the same way (note the low/high bounds
        swap sign and order under negation).
        """
        main = trials[trials["block"] == "main"]
        non_catch = main[~main["is_catch"]].sort_values("trial_index")
        catch = main[main["is_catch"]]

        procedure = self.make_procedure()
        for _, row in non_catch.iterrows():
            procedure.update(float(row["intensity"]), bool(row["correct"]))
        contrast_estimate = procedure.estimate()

        log_cs_value = -contrast_estimate.value
        log_cs_ci_low = -contrast_estimate.ci_high
        log_cs_ci_high = -contrast_estimate.ci_low
        estimate = ThresholdEstimate(
            value=log_cs_value,
            ci_low=log_cs_ci_low,
            ci_high=log_cs_ci_high,
            ci_level=contrast_estimate.ci_level,
            units="log10_contrast_sensitivity",
            method=f"neg_{contrast_estimate.method}",
            extra={
                **contrast_estimate.extra,
                "threshold_log10_weber_contrast": contrast_estimate.value,
                "threshold_criterion": (
                    "F(0)=0.5 crossing of the fitted Weibull-family psychometric function "
                    "(vpsych.core.psychometric convention), in log10 Weber contrast; with "
                    "guess=0.1 and a small lapse rate this is approximately the 55%-correct "
                    "point, not 75% and not the Pelli-Robson chart's own triplet criterion."
                ),
            },
        )

        n_catch = len(catch)
        catch_lapse_rate = float((~catch["correct"]).mean()) if n_catch else 0.0
        dropped_fraction = (
            float(trials["n_dropped_frames_trial"].sum()) / len(trials) if len(trials) else 0.0
        )

        quality_flags = compute_quality_flags(
            catch_lapse_rate=catch_lapse_rate,
            n_catch=n_catch,
            dropped_fraction=dropped_fraction,
            threshold=contrast_estimate.value,
            range_min=min(DEFAULT_INTENSITY_VALUES),
            range_max=max(DEFAULT_INTENSITY_VALUES),
            n_trials=len(non_catch),
            min_trials=self.params.max_trials,
        )

        eye = str(trials["eye"].iloc[0]) if len(trials) else "OU"
        return TestSummary(
            task_id=self.spec.id,
            task_version=self.spec.version,
            eye=eye,
            run=1,
            estimate=estimate,
            fit_params={
                "slope_log10_contrast": contrast_estimate.extra.get("slope"),
                "lapse_rate": contrast_estimate.extra.get("lapse_rate"),
            },
            gof={},
            quality_flags=quality_flags,
            n_trials=len(non_catch),
            n_catch=n_catch,
            catch_lapse_rate=catch_lapse_rate,
            analysis_version=self.spec.version,
        )
