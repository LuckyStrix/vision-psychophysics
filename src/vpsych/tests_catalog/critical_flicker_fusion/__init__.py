"""Critical flicker fusion (CFF): 2AFC flicker-detection threshold, in Hz.

Measures the highest temporal frequency at which a flickering disc is still
reliably discriminable from a steady one of the same mean luminance --
Hecht & Shlaer (1936, *Journal of General Physiology*, 19(6), 965-977) and
the Ferry-Porter law (log-linear increase of CFF with luminance; Porter,
T. C. (1902). Contributions to the study of flicker. *Proceedings of the
Royal Society of London*, 70, 313-329). This implementation is deliberately
honest about the limits of a frame-based display: see
`vpsych.tests_catalog.critical_flicker_fusion.waveform` for the frame-sampled
waveform generator and the amplitude-attenuation check, and
`docs/methods/critical_flicker_fusion.md` for the full method write-up,
including why a measured CFF can only ever be reported up to this display's
"usable ceiling" (Tyler, C. W., & Hamer, R. D. (1990). Analysis of visual
modulation sensitivity. IV. Validity of the Ferry-Porter law. *Journal of
the Optical Society of America A*, 7(4), 743-758).

2AFC: left/right arrow keys report which disc (left or right of fixation)
is flickering; guess rate 0.5. QUEST+ drives the transformed intensity
`x = -log10(frequency_hz)` (see `waveform.frequency_to_intensity`) because
QUEST+ assumes p(correct) increases with intensity, but CFF performance
*decreases* with frequency.
"""

from __future__ import annotations

import itertools
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from vpsych.core.calibration.gamma import GammaChannelModel, fit_gamma_lookup, linearize
from vpsych.core.procedures.base import ThresholdEstimate
from vpsych.core.procedures.questplus_procedure import QuestPlusProcedure
from vpsych.core.timing import summarize_frame_intervals
from vpsych.data.quality import compute_quality_flags
from vpsych.data.schemas import QualityFlag, TestSummary
from vpsych.tests_catalog.base import (
    PresentedTrial,
    PsychophysicalTest,
    TestRequirements,
    TestSpec,
    register_test,
)
from vpsych.tests_catalog.critical_flicker_fusion.waveform import (
    dft_fundamental_amplitude,
    frame_luminance_sequence,
    frequency_to_intensity,
    intensity_to_frequency,
    steady_luminance_sequence,
    usable_square_wave_frequencies,
    valid_square_wave_frequency,
)

GUESS_RATE = 0.5  # 2AFC chance rate.

MIN_REFRESH_HZ = 120.0
"""Minimum display refresh required to run this test at all -- see
`docs/methods/critical_flicker_fusion.md` ("Requirements"): CFF in young
observers under good conditions can approach ~60 Hz (Hecht & Shlaer 1936;
higher under high luminance per the Ferry-Porter law), and a frame-based
display cannot present a clean, minimally-aliased sinusoid much above
roughly a third of its refresh rate (see `USABLE_CEILING_DIVISOR` below) --
120 Hz keeps the usable ceiling (120/3 = 40 Hz) comfortably above typical
human CFF values, whereas a 60 Hz display's ceiling (20 Hz) would be
display-limited for most observers before the flicker even feels fast."""

USABLE_CEILING_DIVISOR = 3.0
"""Divisor applied to the measured refresh rate to get the "usable ceiling"
testable frequency in continuous-waveform mode (`refresh_hz / 3`). The
temporal Nyquist limit is `refresh_hz / 2`, but at exactly the Nyquist rate
a sinusoid is sampled at only 2 points per cycle -- alternating between two
arbitrary phase-dependent luminance values with no recognizable sinusoidal
shape, and highly sensitive to exact phase alignment with frame boundaries.
Three samples per cycle (`refresh_hz / 3`) is the traditional rule-of-thumb
floor for a frame-sampled waveform to still resemble its target shape
(consistent with the general sampling-theory point that Nyquist guarantees
recoverability, not perceptual fidelity, at the limit) and leaves comfortable
margin for the per-trial DFT amplitude check
(`waveform.dft_fundamental_amplitude`) to still detect graceful, not
catastrophic, degradation. See `docs/methods/critical_flicker_fusion.md`."""

DEFAULT_MIN_FREQUENCY_HZ = 2.0
"""Lowest tested flicker frequency, in Hz -- comfortably detectable by any
sighted observer (Hecht & Shlaer 1936 report human CFF well above this even
at low luminance), so it anchors the easy end of the QUEST+ domain."""

WaveformMode = Literal["continuous", "square_wave"]

EDGE_TOLERANCE_FRAC = 0.15
"""Fraction of the tested `x = -log10(f)` domain's span, measured from the
hard (ceiling-frequency) edge, within which the fitted 75%-correct point is
still treated as "display-limited" rather than a trustworthy point estimate.
Wider than the generic `vpsych.data.quality.check_threshold_at_range_edge`'s
default 5%, because QUEST+'s min-entropy stimulus selection resolves the
exact edge only very slowly for a near-ceiling-everywhere observer (see
`summarize`'s docstring and `docs/methods/critical_flicker_fusion.md`)."""


class CFFParams(BaseModel):
    """Configurable parameters for the critical-flicker-fusion test."""

    model_config = ConfigDict(frozen=True)

    max_trials: int = Field(default=40, ge=1, description="QUEST+ trial budget, main block.")
    disc_diameter_deg: float = Field(
        default=2.0, gt=0, description="Each disc's diameter, in degrees of visual angle."
    )
    eccentricity_deg: float = Field(
        default=5.0,
        gt=0,
        description="Distance from fixation to each disc's center, in degrees of visual angle.",
    )
    modulation_depth: float = Field(
        default=1.0,
        gt=0,
        le=1.0,
        description="Michelson modulation depth of the flickering disc, in [0, 1] (1.0 = 100%).",
    )
    mean_luminance_fraction: float = Field(
        default=0.5,
        gt=0,
        lt=1.0,
        description=(
            "Mean luminance of both discs, as a fraction of the calibration's [Lmin, Lmax] "
            "range (0.5 = midpoint); converted to cd/m^2 via the active gamma calibration and "
            "logged in stimulus_params."
        ),
    )
    duration_ms: float = Field(
        default=1000.0, gt=0, description="Stimulus presentation duration, in milliseconds."
    )
    onset_ramp_ms: float = Field(
        default=50.0,
        ge=0,
        description=(
            "Raised-cosine onset/offset ramp applied to the modulation envelope, in "
            "milliseconds (0 = abrupt onset). Nonzero by default: an abrupt luminance step is "
            "itself a potent, frequency-independent transient cue (Tyler & Hamer 1990); see "
            "docs/methods/critical_flicker_fusion.md."
        ),
    )
    waveform_mode: WaveformMode = Field(
        default="continuous",
        description=(
            "'continuous': any frequency up to the usable ceiling (refresh_hz / "
            f"{USABLE_CEILING_DIVISOR:g}). 'square_wave': only frequencies refresh_hz / (2k) "
            "for integer k, presented as an exact, alias-free square wave."
        ),
    )
    min_frequency_hz: float = Field(
        default=DEFAULT_MIN_FREQUENCY_HZ, gt=0, description="Lowest tested flicker frequency, Hz."
    )


def _gamma_channel_model(calibration: Any) -> GammaChannelModel:
    """Build a `GammaChannelModel` from the active calibration's luminance/gamma model.

    Prefers a full measured lookup table (`fit_gamma_lookup`) when raw
    measured points are available (finer fidelity than a single power-law
    exponent, per `docs/CALIBRATION.md`); otherwise falls back to the
    parametric power-law model (`gamma_single`/`lum_min_cdm2`/
    `lum_max_cdm2`), which every grade-A/B calibration provides.
    """
    gamma = calibration.gamma
    if gamma.measured_points:
        return fit_gamma_lookup(gamma.measured_points)
    if gamma.gamma_single is None:
        raise ValueError(
            "Calibration.gamma has neither measured_points nor gamma_single; cannot linearize."
        )
    return GammaChannelModel(
        lum_min_cdm2=gamma.lum_min_cdm2, lum_max_cdm2=gamma.lum_max_cdm2, gamma=gamma.gamma_single
    )


def _luminance_to_drive(luminance_cdm2: np.ndarray, model: GammaChannelModel) -> np.ndarray:
    """Gamma-linearize a luminance (cd/m^2) array to normalized [0, 1] drive levels."""
    frac = (luminance_cdm2 - model.lum_min_cdm2) / (model.lum_max_cdm2 - model.lum_min_cdm2)
    frac = np.clip(frac, 0.0, 1.0)
    return np.asarray(linearize(frac, model))


@register_test
class CriticalFlickerFusionTest(PsychophysicalTest):
    """2AFC (left/right) critical-flicker-fusion threshold, in Hz."""

    spec = TestSpec(
        id="critical_flicker_fusion",
        name="Critical Flicker Fusion",
        version="0.1.0",
        domain="temporal",
        description_participant=(
            "Two identical circles will appear on either side of a center cross. One of them "
            "flickers; the other stays steady. Press the left or right arrow key for the side "
            "that flickers. As the flicker gets faster it will get harder to tell -- just do "
            "your best."
        ),
        description_technical=(
            "2AFC (left/right) critical-flicker-fusion threshold via QUEST+ (Watson 2017) on "
            "the transformed intensity x = -log10(frequency_hz) (performance decreases with "
            "frequency; QUEST+ assumes performance increases with intensity), fixed-slope "
            "Weibull psychometric function, guess rate 0.5. Frame-sampled waveform honesty: "
            "per-trial DFT amplitude of the actually-presented luminance sequence is measured "
            "and compared to nominal (vpsych.tests_catalog.critical_flicker_fusion.waveform). "
            "Requires refresh >= 120 Hz and gamma calibration grade B or better (mean luminance "
            "must match exactly between the flickering and steady discs)."
        ),
        measures="Critical flicker-fusion frequency at 75% correct (or a display-limited lower bound).",
        output_units="hz",
        estimated_minutes=4.0,
        allowed_eyes=["OU"],
        requirements=TestRequirements(
            needs_gamma_calibration=True, min_luminance_grade="B", min_refresh_hz=MIN_REFRESH_HZ
        ),
        citations=[
            "Hecht, S., & Shlaer, S. (1936). Intermittent stimulation by light: V. The relation "
            "between intensity and critical frequency for different parts of the spectrum. "
            "Journal of General Physiology, 19(6), 965-977.",
            "Porter, T. C. (1902). Contributions to the study of flicker. Proceedings of the "
            "Royal Society of London, 70, 313-329.",
            "Tyler, C. W., & Hamer, R. D. (1990). Analysis of visual modulation sensitivity. IV. "
            "Validity of the Ferry-Porter law. Journal of the Optical Society of America A, "
            "7(4), 743-758.",
            "Watson, A. B. (2017). QUEST+: A general multidimensional Bayesian adaptive "
            "psychometric method. Journal of Vision, 17(3), 10.",
        ],
        params_model=CFFParams,
        hidden=False,
    )

    def __init__(
        self,
        params: BaseModel,
        display: Any,
        calibration: Any,
        rng: np.random.Generator,
    ) -> None:
        assert isinstance(params, CFFParams)
        self.params = params
        self.display = display
        self.calibration = calibration
        self.rng = rng
        self._n_presented = 0
        self._stims: dict[str, Any] = {}

    def usable_ceiling_hz(self) -> float:
        """The highest testable frequency at the current (measured) refresh rate, in Hz."""
        return float(self.display.refresh_hz) / USABLE_CEILING_DIVISOR

    def _intensity_values(self) -> list[float]:
        ceiling = self.usable_ceiling_hz()
        if self.params.waveform_mode == "square_wave":
            freqs = usable_square_wave_frequencies(
                self.display.refresh_hz, self.params.min_frequency_hz, ceiling
            )
            if len(freqs) < 2:
                raise ValueError(
                    f"square_wave mode has fewer than 2 usable frequencies between "
                    f"{self.params.min_frequency_hz} and {ceiling:g} Hz at "
                    f"refresh_hz={self.display.refresh_hz:g}; widen the range or raise refresh."
                )
            return sorted(frequency_to_intensity(f) for f in freqs)
        n_points = 25
        x_max = frequency_to_intensity(self.params.min_frequency_hz)
        x_min = frequency_to_intensity(ceiling)
        return [float(v) for v in np.linspace(x_min, x_max, n_points)]

    def make_procedure(self) -> QuestPlusProcedure:
        intensity_values = self._intensity_values()
        span = max(intensity_values) - min(intensity_values)
        threshold_values = [
            float(v) for v in np.linspace(min(intensity_values), max(intensity_values), 21)
        ]
        slope_values = [float(v) for v in np.linspace(max(span * 0.02, 1e-3), span * 0.5, 6)]
        return QuestPlusProcedure(
            intensity_values=intensity_values,
            intensity_units="neg_log10_hz",
            threshold_values=threshold_values,
            slope_values=slope_values,
            guess_rate=GUESS_RATE,
            lapse_rate_values=[0.0, 0.02, 0.04],
            function="weibull",
            max_trials=self.params.max_trials,
        )

    def make_catch_trial_intensity(self) -> float:
        # The lowest (easiest) tested frequency -- the *maximum* of the
        # x = -log10(f) domain, since performance is highest there.
        return max(self._intensity_values())

    def build_stimuli(self, win: Any) -> dict[str, Any]:
        if win is None:
            return {}

        from psychopy import visual

        radius_px = self.display.deg_to_px(self.params.disc_diameter_deg / 2.0)
        offset_px = self.display.deg_to_px(self.params.eccentricity_deg)
        left_disc = visual.Circle(
            win, radius=radius_px, pos=(-offset_px, 0), colorSpace="rgb1", fillColor=(0.5, 0.5, 0.5)
        )
        right_disc = visual.Circle(
            win, radius=radius_px, pos=(offset_px, 0), colorSpace="rgb1", fillColor=(0.5, 0.5, 0.5)
        )
        fixation = visual.TextStim(win, text="+", height=20)
        self._stims = {"left_disc": left_disc, "right_disc": right_disc, "fixation": fixation}
        return self._stims

    def present(
        self, win: Any, intensity_or_stimulus: Any, trial_ctx: dict[str, Any]
    ) -> PresentedTrial:
        self._n_presented += 1
        rng = trial_ctx["rng"]
        intensity = float(intensity_or_stimulus)  # -log10(frequency_hz)
        frequency_hz = intensity_to_frequency(intensity)
        refresh_hz = self.display.refresh_hz
        ceiling_hz = self.usable_ceiling_hz()

        if self.params.waveform_mode == "square_wave" and not valid_square_wave_frequency(
            frequency_hz, refresh_hz
        ):
            # Snap to the nearest valid grid frequency (floating-point drift
            # from the intensity round-trip); the QUEST+ grid itself only
            # ever offers valid frequencies (see _intensity_values), so this
            # is a safety net, not the normal path.
            candidates = usable_square_wave_frequencies(
                refresh_hz, self.params.min_frequency_hz, ceiling_hz
            )
            frequency_hz = min(candidates, key=lambda f: abs(f - frequency_hz))

        correct_side = "left" if rng.random() < 0.5 else "right"
        phase_rad = float(rng.random() * 2.0 * np.pi)

        duration_frames = self.display.frames_for_ms(self.params.duration_ms)
        onset_ramp_frames = self.display.frames_for_ms(self.params.onset_ramp_ms)

        gamma_model = _gamma_channel_model(self.calibration)
        mean_luminance_cdm2 = gamma_model.lum_min_cdm2 + self.params.mean_luminance_fraction * (
            gamma_model.lum_max_cdm2 - gamma_model.lum_min_cdm2
        )

        if self.params.waveform_mode == "square_wave":
            from vpsych.tests_catalog.critical_flicker_fusion.waveform import (
                square_wave_frame_luminance_sequence,
            )

            phase_frames = int(rng.integers(0, max(1, round(refresh_hz / (2 * frequency_hz)))))
            flicker_luminances = square_wave_frame_luminance_sequence(
                mean_luminance_cdm2,
                self.params.modulation_depth,
                frequency_hz,
                refresh_hz,
                duration_frames,
                phase_frames=phase_frames,
            )
            phase_rad = float(2.0 * np.pi * phase_frames / round(refresh_hz / (2 * frequency_hz)))
        else:
            flicker_luminances = frame_luminance_sequence(
                mean_luminance_cdm2,
                self.params.modulation_depth,
                frequency_hz,
                refresh_hz,
                duration_frames,
                phase_rad=phase_rad,
                onset_ramp_frames=onset_ramp_frames,
            )
        steady_luminances = steady_luminance_sequence(mean_luminance_cdm2, duration_frames)

        nominal_amplitude_cdm2 = self.params.modulation_depth * mean_luminance_cdm2
        measured_amplitude_cdm2 = dft_fundamental_amplitude(
            flicker_luminances, frequency_hz, refresh_hz
        )
        attenuation_ratio = (
            measured_amplitude_cdm2 / nominal_amplitude_cdm2 if nominal_amplitude_cdm2 > 0 else 1.0
        )
        attenuated = attenuation_ratio < 0.9

        stimulus_params: dict[str, Any] = {
            "correct_response": correct_side,
            "intensity": intensity,
            "frequency_hz": frequency_hz,
            "refresh_hz": refresh_hz,
            "usable_ceiling_hz": ceiling_hz,
            "waveform_mode": self.params.waveform_mode,
            "modulation_depth": self.params.modulation_depth,
            "mean_luminance_cdm2": mean_luminance_cdm2,
            "disc_diameter_deg": self.params.disc_diameter_deg,
            "eccentricity_deg": self.params.eccentricity_deg,
            "duration_frames": duration_frames,
            "duration_ms_nominal": self.params.duration_ms,
            "onset_ramp_frames": onset_ramp_frames,
            "phase_rad": phase_rad,
            "nominal_amplitude_cdm2": nominal_amplitude_cdm2,
            "measured_amplitude_cdm2": measured_amplitude_cdm2,
            "amplitude_attenuation_ratio": attenuation_ratio,
            "amplitude_attenuated": attenuated,
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
                frame_intervals_s=[1.0 / refresh_hz] * max(timeline.stimulus_frames, 1),
            )

        timeline = trial_ctx["timeline"]
        keyboard = trial_ctx["keyboard"]
        fixation = self._stims["fixation"]
        left_disc = self._stims["left_disc"]
        right_disc = self._stims["right_disc"]
        flicker_disc = left_disc if correct_side == "left" else right_disc
        steady_disc = right_disc if correct_side == "left" else left_disc

        flicker_drive = _luminance_to_drive(flicker_luminances, gamma_model)
        steady_drive = _luminance_to_drive(steady_luminances, gamma_model)

        for _ in range(timeline.fixation_frames):
            fixation.draw()
            win.flip()

        keyboard.clearEvents()
        onset_s = None
        flip_times: list[float] = []
        for frame in range(duration_frames):
            flicker_disc.fillColor = (flicker_drive[frame],) * 3
            steady_disc.fillColor = (steady_drive[frame],) * 3
            flicker_disc.draw()
            steady_disc.draw()
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
        stats = summarize_frame_intervals(intervals_s, refresh_hz)

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
        """Recompute the CFF estimate deterministically from the raw trials.

        `QuestPlusProcedure.estimate().value` is `questplus`'s own native
        Weibull threshold on `x = -log10(f)` (the ~63.2%-of-range point in
        *its* parameterization, not the `vpsych.core.psychometric`
        `F(0)=0.5` point -- see `QuestPlusProcedure`'s "Criterion
        conversion pitfall" docstring section). This converts that to the
        75%-correct point via `QuestPlusProcedure.intensity_at_p_correct`,
        which inverts `questplus`'s own formula directly -- *not*
        `vpsych.core.psychometric.intensity_at_p_correct`, which assumes
        the other, incompatible parameterization and would silently give
        the wrong criterion here -- then back to Hz via
        `intensity_to_frequency`. If that 75%-correct frequency is at or
        beyond this run's usable ceiling, reports a **lower bound** instead
        of a point estimate (see the module/class docstrings and
        `docs/methods/critical_flicker_fusion.md`) with a critical quality
        flag -- never a fabricated point value beyond what the display
        could actually test.
        """
        main = trials[trials["block"] == "main"]
        non_catch = main[~main["is_catch"]].sort_values("trial_index")
        catch = main[main["is_catch"]]

        # Rebuild the same intensity grid this run used from self.display
        # (reanalyze.py reconstructs a fresh test instance from the
        # session's recorded plan/calibration, which includes the same
        # display geometry/refresh_hz the original run used -- see
        # WRITING_A_TEST.md section 10 on replaying self.make_procedure()).
        ceiling_hz = self.usable_ceiling_hz()
        intensity_values = self._intensity_values()
        x_min, x_max = min(intensity_values), max(intensity_values)

        procedure = self.make_procedure()
        for _, row in non_catch.iterrows():
            procedure.update(float(row["intensity"]), bool(row["correct"]))
        raw_estimate = procedure.estimate()
        slope = float(raw_estimate.extra["slope"])
        lapse = float(raw_estimate.extra["lapse_rate"])
        target_x, ci_low_x, ci_high_x = procedure.intensity_at_p_correct(0.75)

        freq_75 = intensity_to_frequency(target_x)
        # x = -log10(f) is decreasing in f, so the x-CI bounds swap order in Hz.
        freq_ci_a = intensity_to_frequency(ci_low_x)
        freq_ci_b = intensity_to_frequency(ci_high_x)
        freq_ci_low, freq_ci_high = min(freq_ci_a, freq_ci_b), max(freq_ci_a, freq_ci_b)

        n_catch = len(catch)
        catch_lapse_rate = float((~catch["correct"]).mean()) if n_catch else 0.0
        dropped_fraction = (
            float(trials["n_dropped_frames_trial"].sum()) / len(trials) if len(trials) else 0.0
        )

        quality_flags: list[QualityFlag] = []
        # Display-limited check: is the 75%-correct point within
        # EDGE_TOLERANCE_FRAC of the domain's hard (ceiling-frequency) edge?
        # This is deliberately a *tolerance band*, not exact edge equality:
        # QUEST+'s min-entropy stimulus selection converges only very slowly
        # onto the true edge for a near-ceiling-everywhere observer (adjacent
        # grid points near the edge predict nearly identical probability
        # correct, so resolving between them by minimal-entropy sampling
        # takes far more trials than a typical session budget) -- see
        # docs/methods/critical_flicker_fusion.md "Display limitations" for
        # the worked numerical justification. Erring toward flagging a
        # borderline case as display-limited (rather than reporting an
        # overconfident point estimate right at the edge) is the honest
        # choice the task requires.
        span = x_max - x_min
        display_limited = span > 0 and target_x <= x_min + EDGE_TOLERANCE_FRAC * span
        if display_limited:
            estimate = ThresholdEstimate(
                value=ceiling_hz,
                ci_low=ceiling_hz,
                ci_high=ceiling_hz,
                ci_level=raw_estimate.ci_level,
                units="hz",
                method="quest_plus_display_limited_lower_bound",
                extra={
                    **raw_estimate.extra,
                    "display_limited": True,
                    "raw_freq_75pct_hz": freq_75,
                    "usable_ceiling_hz": ceiling_hz,
                    "note": (
                        "The observer detected flicker reliably even at the highest testable "
                        "frequency; this display cannot establish a finite CFF for this "
                        "observer. Report as CFF > usable_ceiling_hz, not a point estimate."
                    ),
                },
            )
            quality_flags.append(
                QualityFlag(
                    code="cff_display_limited",
                    severity="critical",
                    message=(
                        f"CFF appears to exceed this display's usable ceiling of "
                        f"{ceiling_hz:.1f} Hz (refresh_hz / {USABLE_CEILING_DIVISOR:g}); report "
                        f"as 'CFF > {ceiling_hz:.1f} Hz, display-limited', not a point estimate. "
                        "A higher-refresh display is needed to measure this observer's true CFF."
                    ),
                )
            )
        else:
            estimate = ThresholdEstimate(
                value=freq_75,
                ci_low=freq_ci_low,
                ci_high=freq_ci_high,
                ci_level=raw_estimate.ci_level,
                units="hz",
                method="quest_plus_posterior_mean_at_75pct_correct",
                extra={
                    **raw_estimate.extra,
                    "raw_questplus_native_threshold_neg_log10_hz": raw_estimate.value,
                    "target_p_correct": 0.75,
                },
            )

        # Always-present info flag: LCD pixel response time attenuates
        # high-frequency modulation independent of anything measured here
        # (see docs/methods/critical_flicker_fusion.md "Display limitations").
        quality_flags.append(
            QualityFlag(
                code="lcd_response_time_limitation",
                severity="info",
                message=(
                    "LCD pixel response time attenuates high-frequency luminance modulation in "
                    "a way this software cannot measure from software timing alone. A "
                    "photodiode check (tools/timing_check.py) is recommended to verify actual "
                    "delivered modulation depth at the frequencies used, especially near this "
                    "run's usable ceiling."
                ),
            )
        )
        if any(
            bool(sp.get("amplitude_attenuated"))
            for sp in non_catch["stimulus_params"]
            if isinstance(sp, dict)
        ):
            quality_flags.append(
                QualityFlag(
                    code="waveform_amplitude_attenuated",
                    severity="warning",
                    message=(
                        "At least one trial's frame-sampled fundamental amplitude was "
                        "attenuated by more than 10% relative to the nominal modulation depth "
                        "(frequency approaching the display's sampling limit)."
                    ),
                )
            )

        quality_flags += compute_quality_flags(
            catch_lapse_rate=catch_lapse_rate,
            n_catch=n_catch,
            dropped_fraction=dropped_fraction,
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
            fit_params={"slope_neg_log10_hz": slope, "lapse_rate": lapse},
            gof={},
            quality_flags=quality_flags,
            n_trials=len(non_catch),
            n_catch=n_catch,
            catch_lapse_rate=catch_lapse_rate,
            analysis_version="0.1.0",
        )
