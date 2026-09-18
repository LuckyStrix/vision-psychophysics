"""Motion coherence: random-dot kinematogram (RDK) direction-discrimination threshold.

Measures the fraction of dots that must move coherently in one direction
(the rest moving as directionless "noise") for an observer to reliably
report the coherent direction -- a standard global-motion-perception
measure descending from Newsome & Paré (1988, *Journal of Neuroscience*,
8(6), 2201-2211) and Britten, Shadlen, Newsome, & Movshon (1992, *Journal of
Neuroscience*, 12(12), 4745-4765), with the noise-dot algorithm following
Scase, Braddick, & Raymond (1996, *Vision Research*, 36(18), 2579-2586) and
general RDK design guidance from Pilly & Seitz (2009, *Vision Research*,
49(13), 1599-1612). See `docs/methods/motion_coherence.md` for the full
method write-up and `vpsych.tests_catalog.motion_coherence.dots` for the
pure-numpy dot-field physics (rendering never uses PsychoPy's own `DotStim`
motion logic, precisely so this can be headless-tested and exactly logged).

2AFC: left/right arrow keys report the coherent motion direction; guess rate
0.5. QUEST+ drives `log10(coherence)` over `[log10(0.01), log10(1.0)]` (1%
to 100% coherence); catch trials use 100% coherence (the domain maximum).
"""

from __future__ import annotations

import itertools
import math
import warnings
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from vpsych.core.procedures.base import ThresholdEstimate
from vpsych.core.procedures.questplus_procedure import QuestPlusProcedure
from vpsych.core.psychometric import PsychometricFunction, intensity_at_p_correct
from vpsych.core.timing import summarize_frame_intervals
from vpsych.data.quality import compute_quality_flags
from vpsych.data.schemas import TestSummary
from vpsych.tests_catalog.base import (
    PresentedTrial,
    PsychophysicalTest,
    TestRequirements,
    TestSpec,
    register_test,
)
from vpsych.tests_catalog.motion_coherence.dots import (
    DIRECTION_ANGLES_DEG,
    EDGE_HANDLING,
    NOISE_ALGORITHM,
    SIGNAL_SELECTION,
    check_displacement_dmax,
    init_dot_field,
    n_dots_for_density,
    step_dot_field,
)

GUESS_RATE = 0.5  # 2AFC chance rate.

#: log10(coherence fraction) domain: 1%-100% coherence (Newsome & Paré 1988
#: and successors typically report thresholds well within this range; 100%
#: coherence is reserved for catch trials).
COHERENCE_DOMAIN_LOG10 = (math.log10(0.01), math.log10(1.0))
DEFAULT_INTENSITY_VALUES = [
    float(v) for v in np.linspace(COHERENCE_DOMAIN_LOG10[0], COHERENCE_DOMAIN_LOG10[1], 25)
]
DEFAULT_THRESHOLD_VALUES = [
    float(v) for v in np.linspace(COHERENCE_DOMAIN_LOG10[0], COHERENCE_DOMAIN_LOG10[1], 21)
]
DEFAULT_SLOPE_VALUES = [float(v) for v in np.linspace(0.1, 1.0, 6)]
DEFAULT_LAPSE_RATE_VALUES = [0.0, 0.02, 0.04]

#: Minimum refresh this test will run on -- see `docs/methods/motion_coherence.md`
#: ("Requirements"): motion coherence needs a display fast enough that
#: per-frame dot displacement stays well under Braddick's `d_max`
#: correspondence limit at a usable dot speed.
MIN_REFRESH_HZ = 60.0


def _deg_array_to_px(display: Any, deg: np.ndarray) -> np.ndarray:
    """Vectorized equivalent of `DisplayGeometry.deg_to_px` for a whole array of positions.

    `DisplayGeometry.deg_to_px` is scalar-only (built on `math.tan`), but
    every frame needs the whole dot field converted to pixels at once;
    this applies the identical exact trigonometric relation
    (`x_cm = 2 * d_cm * tan(radians(deg) / 2)`, see `vpsych.core.display`)
    with `numpy` trig so it vectorizes, then scales by `px_per_cm`, matching
    `deg_to_px` to floating-point precision for any scalar input.
    """
    x_cm = 2.0 * display.viewing_distance_cm * np.tan(np.radians(deg) / 2.0)
    return np.asarray(x_cm * display.px_per_cm, dtype=np.float64)


class MotionCoherenceParams(BaseModel):
    """Configurable parameters for the motion-coherence test.

    Every field documents its unit; see `docs/methods/motion_coherence.md`
    for citations backing each default.
    """

    model_config = ConfigDict(frozen=True)

    max_trials: int = Field(default=40, ge=1, description="QUEST+ trial budget, main block.")
    aperture_diameter_deg: float = Field(
        default=10.0,
        gt=0,
        description=(
            "Circular RDK aperture diameter, in degrees of visual angle "
            "(Newsome & Paré 1988-style RDK design)."
        ),
    )
    dot_diameter_arcmin: float = Field(
        default=6.0,
        gt=0,
        description="Individual dot diameter, in arcminutes (Newsome & Paré 1988).",
    )
    dot_density_per_deg2: float = Field(
        default=2.0,
        gt=0,
        description=(
            "Dot density, in dots per square degree of aperture area "
            "(within the range surveyed by Pilly & Seitz 2009)."
        ),
    )
    speed_deg_per_s: float = Field(
        default=5.0,
        gt=0,
        description="Signal and noise dot speed, in degrees of visual angle per second.",
    )
    dot_lifetime_frames: int = Field(
        default=10,
        ge=1,
        description=(
            "Maximum dot lifetime before random replotting, in frames "
            "(limited lifetime, Scase, Braddick, & Raymond 1996)."
        ),
    )
    duration_ms: float = Field(
        default=500.0,
        gt=0,
        description="Stimulus (dot motion) presentation duration, in milliseconds.",
    )


@register_test
class MotionCoherenceTest(PsychophysicalTest):
    """2AFC (left/right) RDK direction-discrimination coherence threshold."""

    spec = TestSpec(
        id="motion_coherence",
        name="Motion Coherence (Random-Dot Kinematogram)",
        version="0.1.0",
        domain="motion",
        description_participant=(
            "A cloud of moving dots will appear in the center of the screen. Some dots move "
            "together in one direction (left or right) while others move randomly. Press the "
            "left or right arrow key for the direction the dots, as a whole, appeared to move."
        ),
        description_technical=(
            "2AFC (left/right) global-motion coherence threshold via a random-dot "
            "kinematogram, QUEST+ (Watson 2017) on log10(coherence fraction), fixed-slope "
            "Weibull psychometric function, guess rate 0.5. Noise algorithm: 'random_direction' "
            "(Scase, Braddick, & Raymond 1996). Signal dots reselected afresh every frame "
            "('white noise' selection, Pilly & Seitz 2009). Dot-field physics is a pure numpy "
            "function (vpsych.tests_catalog.motion_coherence.dots), not PsychoPy's DotStim."
        ),
        measures="Motion-coherence direction-discrimination threshold at 75% correct.",
        output_units="coherence_percent",
        estimated_minutes=4.0,
        allowed_eyes=["OU"],
        requirements=TestRequirements(min_refresh_hz=MIN_REFRESH_HZ),
        citations=[
            "Newsome, W. T., & Paré, E. B. (1988). A selective impairment of motion perception "
            "following lesions of the middle temporal visual area (MT). Journal of "
            "Neuroscience, 8(6), 2201-2211.",
            "Britten, K. H., Shadlen, M. N., Newsome, W. T., & Movshon, J. A. (1992). The "
            "analysis of visual motion: a comparison of neuronal and psychophysical "
            "performance. Journal of Neuroscience, 12(12), 4745-4765.",
            "Scase, M. O., Braddick, O. J., & Raymond, J. E. (1996). What is noise for the "
            "motion system? Vision Research, 36(18), 2579-2586.",
            "Pilly, P. K., & Seitz, A. R. (2009). What a difference a parameter makes: a "
            "psychophysical comparison of random dot motion algorithms. Vision Research, "
            "49(13), 1599-1612.",
            "Braddick, O. (1974). A short-range process in apparent motion. Vision Research, "
            "14(7), 519-527.",
            "Watson, A. B. (2017). QUEST+: A general multidimensional Bayesian adaptive "
            "psychometric method. Journal of Vision, 17(3), 10.",
        ],
        params_model=MotionCoherenceParams,
        hidden=False,
    )

    def __init__(
        self,
        params: BaseModel,
        display: Any,
        calibration: Any,
        rng: np.random.Generator,
    ) -> None:
        assert isinstance(params, MotionCoherenceParams)
        self.params = params
        self.display = display
        self.calibration = calibration
        self.rng = rng
        self._n_presented = 0
        self._stims: dict[str, Any] = {}

    def make_procedure(self) -> QuestPlusProcedure:
        return QuestPlusProcedure(
            intensity_values=DEFAULT_INTENSITY_VALUES,
            intensity_units="log10_coherence",
            threshold_values=DEFAULT_THRESHOLD_VALUES,
            slope_values=DEFAULT_SLOPE_VALUES,
            guess_rate=GUESS_RATE,
            lapse_rate_values=DEFAULT_LAPSE_RATE_VALUES,
            function="weibull",
            max_trials=self.params.max_trials,
        )

    def make_catch_trial_intensity(self) -> float:
        # 100% coherence (log10(1.0) == 0.0), the domain maximum -- see the
        # module docstring and docs/methods/motion_coherence.md.
        return max(DEFAULT_INTENSITY_VALUES)

    def build_stimuli(self, win: Any) -> dict[str, Any]:
        if win is None:
            return {}

        from psychopy import visual

        aperture_radius_px = self.display.deg_to_px(self.params.aperture_diameter_deg / 2.0)
        n_dots = n_dots_for_density(
            self.params.aperture_diameter_deg, self.params.dot_density_per_deg2
        )
        dot_size_px = self.display.deg_to_px(self.params.dot_diameter_arcmin / 60.0)
        dots = visual.ElementArrayStim(
            win,
            units="pix",
            nElements=max(n_dots, 1),
            elementTex=None,
            elementMask="circle",
            sizes=dot_size_px,
            xys=np.zeros((max(n_dots, 1), 2)),
            colors=(1.0, 1.0, 1.0),
            colorSpace="rgb",
        )
        fixation = visual.TextStim(win, text="+", height=20)
        self._stims = {
            "dots": dots,
            "fixation": fixation,
            "n_dots": n_dots,
            "aperture_radius_px": aperture_radius_px,
        }
        return self._stims

    def present(
        self, win: Any, intensity_or_stimulus: Any, trial_ctx: dict[str, Any]
    ) -> PresentedTrial:
        self._n_presented += 1
        rng = trial_ctx["rng"]
        intensity = float(intensity_or_stimulus)  # log10 coherence
        coherence = float(10.0**intensity)
        correct_side = "left" if rng.random() < 0.5 else "right"
        direction_deg = DIRECTION_ANGLES_DEG[correct_side]

        aperture_radius_deg = self.params.aperture_diameter_deg / 2.0
        n_dots = n_dots_for_density(
            self.params.aperture_diameter_deg, self.params.dot_density_per_deg2
        )
        dt_s = 1.0 / self.display.refresh_hz
        exceeds_dmax, displacement_deg, spacing_deg = check_displacement_dmax(
            self.params.speed_deg_per_s, dt_s, self.params.dot_density_per_deg2
        )
        if exceeds_dmax:
            warnings.warn(
                f"motion_coherence: per-frame displacement ({displacement_deg:.3f} deg) exceeds "
                f"half the mean dot spacing ({spacing_deg:.3f} deg) -- risk of correspondence "
                "aliasing beyond Braddick's d_max (Braddick 1974). Reduce speed_deg_per_s or "
                "increase dot_density_per_deg2.",
                stacklevel=2,
            )

        duration_frames = self.display.frames_for_ms(self.params.duration_ms)

        stimulus_params: dict[str, Any] = {
            "correct_response": correct_side,
            "intensity": intensity,
            "coherence": coherence,
            "coherence_percent": coherence * 100.0,
            "direction_deg": direction_deg,
            "aperture_diameter_deg": self.params.aperture_diameter_deg,
            "dot_diameter_arcmin": self.params.dot_diameter_arcmin,
            "dot_density_per_deg2": self.params.dot_density_per_deg2,
            "n_dots": n_dots,
            "speed_deg_per_s": self.params.speed_deg_per_s,
            "dot_lifetime_frames": self.params.dot_lifetime_frames,
            "duration_frames": duration_frames,
            "duration_ms_nominal": self.params.duration_ms,
            "refresh_hz": self.display.refresh_hz,
            "noise_algorithm": NOISE_ALGORITHM,
            "signal_selection": SIGNAL_SELECTION,
            "edge_handling": EDGE_HANDLING,
            "displacement_per_frame_deg": displacement_deg,
            "dot_spacing_deg": spacing_deg,
            "dmax_exceeded": exceeds_dmax,
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
                rt_s=0.4,
                stimulus_onset_s=float(self._n_presented),
                n_dropped_frames=0,
                frame_intervals_s=[1.0 / self.display.refresh_hz]
                * max(timeline.stimulus_frames, 1),
            )

        timeline = trial_ctx["timeline"]
        keyboard = trial_ctx["keyboard"]
        fixation = self._stims["fixation"]
        dots = self._stims["dots"]
        for _ in range(timeline.fixation_frames):
            fixation.draw()
            win.flip()

        state = init_dot_field(n_dots, aperture_radius_deg, self.params.dot_lifetime_frames, rng)

        keyboard.clearEvents()
        onset_s = None
        flip_times: list[float] = []
        for frame in range(duration_frames):
            state, _is_signal = step_dot_field(
                state,
                rng,
                coherence=coherence,
                direction_deg=direction_deg,
                speed_deg_per_s=self.params.speed_deg_per_s,
                dt_s=dt_s,
                aperture_radius_deg=aperture_radius_deg,
                lifetime_frames=self.params.dot_lifetime_frames,
            )
            positions_px = _deg_array_to_px(self.display, state.positions_deg)
            dots.xys = positions_px
            dots.draw()
            fixation.draw()
            flip_time = win.flip()
            flip_times.append(flip_time)
            if frame == 0:
                onset_s = flip_time

        keys = keyboard.waitKeys(keyList=["left", "right"], timeStamped=True)
        response, rt_s = (keys[0][0], keys[0][1] - (onset_s or 0.0)) if keys else (None, None)

        for _ in range(timeline.iti_frames):
            win.flip()

        intervals_s = [b - a for a, b in itertools.pairwise(flip_times)]
        stats = summarize_frame_intervals(intervals_s, self.display.refresh_hz)

        return PresentedTrial(
            response=response,
            rt_s=rt_s,
            stimulus_onset_s=onset_s or 0.0,
            n_dropped_frames=stats.n_dropped,
            frame_intervals_s=intervals_s,
        )

    def response_keys(self) -> list[str]:
        return ["left", "right"]

    def score(self, response: Any, stimulus_params: dict[str, Any]) -> bool:
        return bool(response == stimulus_params["correct_response"])

    def instructions(self) -> str:
        return self.spec.description_participant

    def summarize(self, trials: pd.DataFrame) -> TestSummary:
        """Recompute the threshold deterministically from the raw trials (see WRITING_A_TEST.md).

        The QUEST+ posterior threshold is the F=0.5 crossing of the Weibull
        (see `docs/METHODS.md`'s "Psychometric function families" section);
        this converts that to the intensity at 75% correct (2AFC) via
        `intensity_at_p_correct`, and reports it as coherence *percent*
        (`10**x * 100`) rather than log10 coherence, per the task
        requirement.
        """
        main = trials[trials["block"] == "main"]
        non_catch = main[~main["is_catch"]].sort_values("trial_index")
        catch = main[main["is_catch"]]

        procedure = self.make_procedure()
        for _, row in non_catch.iterrows():
            procedure.update(float(row["intensity"]), bool(row["correct"]))
        raw_estimate = procedure.estimate()

        slope = float(raw_estimate.extra["slope"])
        lapse = float(raw_estimate.extra["lapse_rate"])
        fn = PsychometricFunction(
            family="weibull",
            threshold=raw_estimate.value,
            slope=slope,
            guess=GUESS_RATE,
            lapse=lapse,
            intensity_scale="log10",
        )
        target_log10 = intensity_at_p_correct(fn, 0.75)
        offset = target_log10 - raw_estimate.value
        ci_low_log10 = raw_estimate.ci_low + offset
        ci_high_log10 = raw_estimate.ci_high + offset

        def _to_percent(log10_coherence: float) -> float:
            return float(10.0**log10_coherence * 100.0)

        threshold_percent = _to_percent(target_log10)
        ci_low_percent = _to_percent(min(ci_low_log10, ci_high_log10))
        ci_high_percent = _to_percent(max(ci_low_log10, ci_high_log10))

        n_catch = len(catch)
        catch_lapse_rate = float((~catch["correct"]).mean()) if n_catch else 0.0
        dropped_fraction = (
            float(trials["n_dropped_frames_trial"].sum()) / len(trials) if len(trials) else 0.0
        )

        range_min_percent = _to_percent(min(DEFAULT_INTENSITY_VALUES))
        range_max_percent = _to_percent(max(DEFAULT_INTENSITY_VALUES))

        quality_flags = compute_quality_flags(
            catch_lapse_rate=catch_lapse_rate,
            n_catch=n_catch,
            dropped_fraction=dropped_fraction,
            threshold=threshold_percent,
            range_min=range_min_percent,
            range_max=range_max_percent,
            n_trials=len(non_catch),
            min_trials=self.params.max_trials,
        )

        estimate = ThresholdEstimate(
            value=threshold_percent,
            ci_low=ci_low_percent,
            ci_high=ci_high_percent,
            ci_level=raw_estimate.ci_level,
            units="coherence_percent",
            method="quest_plus_posterior_mean_at_75pct_correct",
            extra={
                **raw_estimate.extra,
                "threshold_f0.5_log10_coherence": raw_estimate.value,
                "target_p_correct": 0.75,
            },
        )

        eye = str(trials["eye"].iloc[0]) if len(trials) else "OU"
        return TestSummary(
            task_id=self.spec.id,
            task_version=self.spec.version,
            eye=eye,
            run=1,
            estimate=estimate,
            fit_params={"slope_log10_coherence": slope, "lapse_rate": lapse},
            gof={},
            quality_flags=quality_flags,
            n_trials=len(non_catch),
            n_catch=n_catch,
            catch_lapse_rate=catch_lapse_rate,
            analysis_version="0.1.0",
        )
