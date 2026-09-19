"""Quick Contrast Sensitivity Function (qCSF): 2AFC grating orientation identification.

Measures the full contrast sensitivity function -- contrast sensitivity
(1/threshold contrast) as a function of spatial frequency -- via the qCSF
Bayesian adaptive method (Lesmes, L. A., Lu, Z.-L., Baek, J., & Albright,
T. D. (2010). Bayesian adaptive estimation of the contrast sensitivity
function: The quick CSF method. Journal of Vision, 10(3):17), using a 2AFC
orientation-identification task (the variant validated for gratings by Hou,
F., Huang, C.-B., Lesmes, L., Feng, L.-X., Tao, L., Zhou, Y.-F., & Lu, Z.-L.
(2010). qCSF in clinical application: Efficient characterization and
classification of contrast sensitivity functions in amblyopia. Investigative
Ophthalmology & Visual Science, 51(10), 5365-5377). See
`docs/methods/contrast_sensitivity_function.md` for the full write-up.

Stimulus
--------
A sinusoidal grating in a raised-cosine circular envelope (see
`vpsych.tests_catalog._contrast_rendering.raised_cosine_disc_envelope`),
tilted +/- `orientation_deg` from vertical (randomized per trial), Michelson
contrast defined in the luminance domain around a mid-gray (mean-luminance)
background, gamma-linearized and noisy-bit dithered every stimulus frame
(see `_contrast_rendering`'s module docstring for why every frame, not once
per trial, and the shared-code notes on `PsychoPyBackend` not yet applying
its own gamma ramp). Spatial phase is randomized per trial and logged.

Spatial frequency grid
-----------------------
`QCSF.default_grids()`'s spatial-frequency grid (0.5-24 cpd, log-spaced) is
intersected, per test instance (i.e. at the actual configured display and
viewing distance), with two limits:

- **Lower limit**, set by stimulus size: the lowest tested frequency must
  still show at least ~2 cycles across the patch (`2 / grating_diameter_deg`
  cycles/deg) -- a stimulus showing fewer cycles than that is not really a
  "grating" any more (edge effects from the spatial envelope dominate). The
  default 4 deg diameter makes this exactly 0.5 cpd, the bottom of
  `QCSF.default_grids()`'s own grid and of the standard 0.5-16 cpd CSF
  range.
- **Upper limit**, a rendering-fidelity criterion: at most
  `display.nyquist_cpd / 2`, i.e. at least 4 pixels per cycle (twice the
  bare 2-px/cycle Nyquist limit `DisplayGeometry.nyquist_cpd` already
  documents), since a sinusoidal grating sampled at only 2 px/cycle is
  already visibly distorted well before the true optical Nyquist limit --
  4 px/cycle is a common, conservative rendering-fidelity margin for
  grating stimuli. If this excludes part of the standard 0.5-16 cpd range,
  a `reduced_csf_frequency_coverage` quality flag documents exactly which
  range was actually tested (see `summarize`).

Procedure
---------
`QCSF` (`vpsych.core.procedures.qcsf`), `max_trials` defaulting to
`RECOMMENDED_MIN_TRIALS` (300, see that module's docstring for the
bias/trial-count validation this is based on); `params.quick=True` instead
uses 100 trials, with a documented `quick_csf_run` quality flag disclosing
the larger expected bias at that trial count. Catch trials present a low
spatial frequency (~1-2 cpd, configurable) at high contrast.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from vpsych.core.procedures.base import ThresholdEstimate
from vpsych.core.procedures.qcsf import QCSF, RECOMMENDED_MIN_TRIALS, log_contrast_sensitivity
from vpsych.core.trial import TrialTimeline
from vpsych.data.quality import compute_quality_flags
from vpsych.data.schemas import QualityFlag, TestSummary
from vpsych.tests_catalog.base import (
    PresentedTrial,
    PsychophysicalTest,
    TestRequirements,
    TestSpec,
    register_test,
)

GUESS_RATE = 0.5  # 2AFC orientation identification.
LAPSE_RATE = 0.02

#: Standard CSF range this test aims to cover at its default requirements
#: (see `TestRequirements` below and `docs/methods/contrast_sensitivity_function.md`).
STANDARD_RANGE_LOW_CPD = 0.5
STANDARD_RANGE_HIGH_CPD = 16.0

#: Minimum viewing distance (documented default is ~2-3 m, see the module
#: docstring's "2-3 m equivalent distance" language) below which a "typical"
#: 1920x1080, ~53 cm-wide display's 4-px/cycle rendering-fidelity ceiling
#: (`DisplayGeometry.nyquist_cpd / 2`) falls short of 16 cpd. At 100 cm on
#: such a display, px_per_deg is ~63 (nyquist/2 ~15.8 cpd, just under the
#: standard range's top); at the recommended 200-300 cm it is ~126-190
#: (nyquist/2 ~31-47 cpd), comfortably covering the full range. This bound
#: is deliberately conservative (research-grade displays/setups vary), and
#: `summarize()`'s `reduced_csf_frequency_coverage` flag reports the actual
#: per-session coverage regardless.
MIN_VIEWING_DISTANCE_CM = 100.0


class CSFParams(BaseModel):
    """Configurable parameters for the qCSF test.

    Attributes:
        max_trials: Main-block trial budget when `quick=False` (the
            default). Should be `>= RECOMMENDED_MIN_TRIALS` for a
            research-grade AULCSF estimate (see `vpsych.core.procedures.qcsf`
            module docstring); values below that are allowed but a
            `too_few_trials` quality flag will fire.
        quick: If `True`, overrides `max_trials` to a fast ~100-trial run
            (ignoring `max_trials`) for demos/piloting. Carries a documented
            `quick_csf_run` quality flag disclosing the larger
            (~-0.06 to -0.08 log unit) AULCSF bias expected at 100 trials.
        grating_diameter_deg: Grating patch diameter, degrees of visual
            angle. Default 4.0 deg: sized so the lowest tested spatial
            frequency (0.5 cpd, the bottom of the standard CSF range)
            still shows exactly 2 cycles across the patch.
        orientation_deg: Tilt magnitude from vertical, in degrees; the sign
            (clockwise/counterclockwise) is randomized per trial and is
            what the 2AFC response discriminates. Default 45 deg (the
            classic +/-45 deg tilt-identification design).
        stimulus_duration_ms: Total stimulus presentation duration
            (envelope onset to offset), in milliseconds. Default 500 ms.
        ramp_duration_ms: Duration of each of the raised-cosine onset/offset
            temporal ramps, in milliseconds (so the plateau at full
            contrast is `stimulus_duration_ms - 2 * ramp_duration_ms`).
            Default 100 ms (300 ms plateau at the default 500 ms total).
        fixation_duration_ms: Fixation-cross duration before the stimulus,
            in milliseconds.
        response_timeout_ms: Maximum time to wait for a response after
            stimulus offset, in milliseconds.
        iti_ms: Inter-trial interval, in milliseconds.
        catch_spatial_frequency_cpd: Spatial frequency used for catch
            trials (~1-2 cpd, a low, easily-visible frequency).
        catch_contrast: Michelson contrast used for catch trials (high,
            comfortably suprathreshold for any observer with normal vision).
    """

    model_config = ConfigDict(frozen=True)

    max_trials: int = Field(
        default=RECOMMENDED_MIN_TRIALS, ge=1, description="Main-block trial budget."
    )
    quick: bool = Field(
        default=False,
        description="Use a fast ~100-trial run instead of max_trials; carries a bias quality flag.",
    )
    grating_diameter_deg: float = Field(
        default=4.0, gt=0, description="Grating patch diameter, degrees of visual angle."
    )
    orientation_deg: float = Field(
        default=45.0, gt=0, le=89.0, description="Tilt magnitude from vertical, degrees."
    )
    stimulus_duration_ms: float = Field(
        default=500.0, gt=0, description="Total stimulus duration (envelope onset to offset), ms."
    )
    ramp_duration_ms: float = Field(
        default=100.0, ge=0, description="Duration of each temporal raised-cosine ramp, ms."
    )
    fixation_duration_ms: float = Field(
        default=500.0, ge=0, description="Fixation-cross duration before the stimulus, ms."
    )
    response_timeout_ms: float = Field(
        default=3000.0, gt=0, description="Max time to wait for a response after offset, ms."
    )
    iti_ms: float = Field(default=500.0, ge=0, description="Inter-trial interval, ms.")
    catch_spatial_frequency_cpd: float = Field(
        default=1.5, gt=0, description="Spatial frequency used for catch trials, cycles/deg."
    )
    catch_contrast: float = Field(
        default=0.5, gt=0, le=1.0, description="Michelson contrast used for catch trials."
    )


def _quick_trial_count() -> int:
    return 100


@register_test
class ContrastSensitivityFunctionTest(PsychophysicalTest):
    """qCSF: 2AFC orientation identification on a contrast/spatial-frequency grid."""

    spec = TestSpec(
        id="contrast_sensitivity_function",
        name="Contrast Sensitivity Function (qCSF)",
        version="0.1.0",
        domain="contrast",
        description_participant=(
            "A faint striped pattern will briefly appear in the middle of the screen, tilted "
            "slightly to the left or right of upright. Press the left arrow key if it tilted "
            "left, or the right arrow key if it tilted right. Some patterns will be very faint "
            "-- just make your best guess."
        ),
        description_technical=(
            "Quick CSF (Lesmes, Lu, Baek & Albright 2010), 2AFC orientation identification "
            "(Hou et al. 2010 variant), Bayesian adaptive estimation of the truncated "
            "log-parabola contrast sensitivity function over spatial frequency x contrast."
        ),
        measures="Contrast sensitivity function (AULCSF).",
        output_units="aulcsf_log10cs_log10cpd",
        estimated_minutes=8.0,
        allowed_eyes=["OD", "OS", "OU"],
        requirements=TestRequirements(
            needs_gamma_calibration=True,
            min_luminance_grade="B",
            min_viewing_distance_cm=MIN_VIEWING_DISTANCE_CM,
            max_required_cpd=STANDARD_RANGE_HIGH_CPD,
        ),
        citations=[
            "Lesmes, L. A., Lu, Z.-L., Baek, J., & Albright, T. D. (2010). Bayesian adaptive "
            "estimation of the contrast sensitivity function: The quick CSF method. Journal of "
            "Vision, 10(3):17. https://doi.org/10.1167/10.3.17",
            "Hou, F., Huang, C.-B., Lesmes, L., Feng, L.-X., Tao, L., Zhou, Y.-F., & Lu, Z.-L. "
            "(2010). qCSF in clinical application: Efficient characterization and classification "
            "of contrast sensitivity functions in amblyopia. Investigative Ophthalmology & Visual "
            "Science, 51(10), 5365-5377. https://doi.org/10.1167/iovs.10-5636",
            "Allard, R., & Faubert, J. (2008). The noisy-bit method for digital displays: "
            "converting a resolution limitation into a pseudo-resolution. Behavior Research "
            "Methods, 40(3), 735-743. https://doi.org/10.3758/BRM.40.3.735",
        ],
        params_model=CSFParams,
        hidden=False,
    )

    def __init__(
        self,
        params: BaseModel,
        display: Any,
        calibration: Any,
        rng: np.random.Generator,
    ) -> None:
        assert isinstance(params, CSFParams)
        self.params = params
        self.display = display
        self.calibration = calibration
        self.rng = rng
        self._n_presented = 0
        self._stims: dict[str, Any] = {}

        default_freqs = np.array(QCSF.default_grids()["spatial_frequency_values_cpd"])
        low_cpd = 2.0 / params.grating_diameter_deg
        high_cpd = display.nyquist_cpd / 2.0  # 4 px/cycle rendering-fidelity criterion.
        if high_cpd <= low_cpd:
            raise ValueError(
                f"This display/viewing-distance combination cannot present even 2 cycles of a "
                f"{params.grating_diameter_deg:g} deg grating at 4 px/cycle (low={low_cpd:.3g} "
                f"cpd, high={high_cpd:.3g} cpd). Increase the viewing distance or grating size."
            )
        selected = default_freqs[(default_freqs >= low_cpd) & (default_freqs <= high_cpd)]
        if selected.size < 3:
            selected = np.geomspace(low_cpd, high_cpd, 6)
        self._freq_grid = [float(v) for v in selected]
        self._freq_low_cpd = float(low_cpd)
        self._freq_high_cpd = float(high_cpd)
        self._reduced_coverage = (
            high_cpd < STANDARD_RANGE_HIGH_CPD or low_cpd > STANDARD_RANGE_LOW_CPD
        )

    @property
    def _effective_max_trials(self) -> int:
        return _quick_trial_count() if self.params.quick else self.params.max_trials

    def make_procedure(self) -> QCSF:
        grids = QCSF.default_grids()
        return QCSF(
            spatial_frequency_values_cpd=self._freq_grid,
            contrast_values=grids["contrast_values"],
            peak_gain_values=grids["peak_gain_values"],
            peak_freq_values_cpd=grids["peak_freq_values_cpd"],
            bandwidth_values_octaves=grids["bandwidth_values_octaves"],
            low_freq_truncation_values=grids["low_freq_truncation_values"],
            guess_rate=GUESS_RATE,
            lapse_rate=LAPSE_RATE,
            max_trials=self._effective_max_trials,
        )

    def make_catch_trial_intensity(self) -> dict[str, float]:
        # PsychophysicalTest.make_catch_trial_intensity's signature is
        # `-> float | dict[str, float]` precisely to support a
        # MultiParamProcedure-driven test like this one, whose catch-trial
        # "intensity" is inherently 2D (spatial frequency + contrast) --
        # matching vpsych.core.trial_loop.TrialLoop.run's own
        # `value: float | dict[str, float]` typing of this call's result.
        # QCSF.update()/present() below both expect this dict shape,
        # matching next_stimulus()'s own return convention.
        contrast = self.params.catch_contrast
        return {
            "spatial_frequency_cpd": self.params.catch_spatial_frequency_cpd,
            "contrast": contrast,
            "intensity": float(np.log10(contrast)),
        }

    def _timeline(self) -> TrialTimeline:
        return TrialTimeline(
            fixation_frames=self.display.frames_for_ms(self.params.fixation_duration_ms),
            stimulus_frames=self.display.frames_for_ms(self.params.stimulus_duration_ms),
            response_timeout_frames=self.display.frames_for_ms(self.params.response_timeout_ms),
            iti_frames=self.display.frames_for_ms(self.params.iti_ms),
        )

    def build_stimuli(self, win: Any) -> dict[str, Any]:
        if win is None:
            return {}
        from psychopy import visual

        size_px = round(self.display.deg_to_px(self.params.grating_diameter_deg))
        fixation = visual.TextStim(win, text="+", height=20)
        grating_image = visual.ImageStim(win, size=(size_px, size_px), units="pix")
        self._stims = {"fixation": fixation, "grating_image": grating_image, "size_px": size_px}
        return self._stims

    def present(
        self, win: Any, intensity_or_stimulus: Any, trial_ctx: dict[str, Any]
    ) -> PresentedTrial:
        self._n_presented += 1
        rng = trial_ctx["rng"]
        stim = intensity_or_stimulus
        freq = float(stim["spatial_frequency_cpd"])
        contrast = float(stim["contrast"])
        side_sign = 1.0 if rng.random() < 0.5 else -1.0
        orientation = side_sign * self.params.orientation_deg
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        correct_response = "right" if side_sign > 0 else "left"

        stimulus_params: dict[str, Any] = {
            "correct_response": correct_response,
            "intensity": float(stim["intensity"]),
            "contrast": contrast,
            "spatial_frequency_cpd": freq,
            "orientation_deg": orientation,
            "phase_rad": phase,
            "grating_diameter_deg": self.params.grating_diameter_deg,
        }
        trial_ctx["stimulus_params"] = stimulus_params
        trial_ctx["correct_response"] = correct_response

        timeline = self._timeline()

        observer = trial_ctx["simulated_observer"]
        if observer is not None:
            obs_stim = {"spatial_frequency_cpd": freq, "contrast": contrast}
            is_correct = observer.decide_correct(obs_stim, rng)
            response = self.simulated_response(is_correct, stimulus_params, rng)
            return PresentedTrial(
                response=response,
                rt_s=0.4,
                stimulus_onset_s=float(self._n_presented),
                n_dropped_frames=0,
                frame_intervals_s=[1.0 / 60.0] * max(timeline.stimulus_frames, 1),
            )

        from vpsych.tests_catalog._contrast_rendering import (
            dither_frame,
            gamma_channel_model_from_calibration,
            michelson_drive_from_pattern,
            raised_cosine_disc_envelope,
            raised_cosine_temporal_envelope,
            sinusoidal_grating_pattern,
        )

        gamma_model = gamma_channel_model_from_calibration(self.calibration.gamma)
        size_px = self._stims["size_px"]
        cycles_per_px = freq / self.display.px_per_deg_at_center
        pattern = sinusoidal_grating_pattern(size_px, cycles_per_px, orientation, phase)
        envelope = raised_cosine_disc_envelope(size_px)
        windowed_pattern = pattern * envelope

        n_frames = timeline.stimulus_frames
        ramp_frames = self.display.frames_for_ms(self.params.ramp_duration_ms)
        temporal_env = raised_cosine_temporal_envelope(n_frames, ramp_frames)

        fixation = self._stims["fixation"]
        image_stim = self._stims["grating_image"]
        keyboard = trial_ctx["keyboard"]

        for _ in range(timeline.fixation_frames):
            fixation.draw()
            win.flip()

        keyboard.clearEvents()
        onset_s = None
        for frame in range(n_frames):
            frame_contrast = contrast * float(temporal_env[frame])
            # Dither is refreshed every stimulus frame (not once per trial),
            # per _contrast_rendering's dither_frame docstring, so effective
            # contrast resolution improves with n_frames.
            drive = michelson_drive_from_pattern(windowed_pattern, frame_contrast, gamma_model)
            dithered = dither_frame(drive, rng)
            image_stim.image = dithered * 2.0 - 1.0  # [0,1] -> [-1,1] for colorSpace="rgb"
            image_stim.draw()
            flip_time = win.flip()
            if frame == 0:
                onset_s = flip_time

        response_timeout_frames = timeline.response_timeout_frames or 1
        max_wait_s = response_timeout_frames / self.display.refresh_hz
        keys = keyboard.waitKeys(maxWait=max_wait_s, keyList=self.response_keys(), timeStamped=True)
        response, rt_s = (keys[0][0], keys[0][1] - (onset_s or 0.0)) if keys else (None, None)

        for _ in range(timeline.iti_frames):
            win.flip()

        return PresentedTrial(
            response=response,
            rt_s=rt_s,
            stimulus_onset_s=onset_s or 0.0,
            n_dropped_frames=0,
            frame_intervals_s=[1.0 / self.display.refresh_hz] * n_frames,
        )

    def response_keys(self) -> list[str]:
        return ["left", "right"]

    def score(self, response: Any, stimulus_params: dict[str, Any]) -> bool:
        return bool(response == stimulus_params["correct_response"])

    def instructions(self) -> str:
        return self.spec.description_participant

    def _replay_procedure(self, trials: pd.DataFrame) -> QCSF:
        """Replay a fresh `QCSF` over `trials`' main, non-catch rows (see `summarize`)."""
        main = trials[trials["block"] == "main"]
        non_catch = main[~main["is_catch"]].sort_values("trial_index")
        procedure = self.make_procedure()
        for _, row in non_catch.iterrows():
            sp = row["stimulus_params"]
            procedure.update(
                {
                    "spatial_frequency_cpd": float(sp["spatial_frequency_cpd"]),
                    "contrast": float(sp["contrast"]),
                },
                bool(row["correct"]),
            )
        return procedure

    def summarize(self, trials: pd.DataFrame) -> TestSummary:
        """Recompute AULCSF deterministically by replaying a fresh QCSF over `trials`.

        Pure function of `trials` (see `docs/WRITING_A_TEST.md` section 10):
        `QCSF.update`/`estimate` use no randomness, and the frequency grid
        (`self._freq_grid`) is itself a deterministic function of
        `self.params`/`self.display`, both reconstructed identically by
        `vpsych.data.reanalyze.reanalyze_session`.
        """
        main = trials[trials["block"] == "main"]
        non_catch = main[~main["is_catch"]].sort_values("trial_index")
        catch = main[main["is_catch"]]

        procedure = self._replay_procedure(trials)
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
            n_trials=len(non_catch),
            min_trials=RECOMMENDED_MIN_TRIALS,
        )
        if self.params.quick:
            quality_flags = [
                *quality_flags,
                QualityFlag(
                    code="quick_csf_run",
                    severity="warning",
                    message=(
                        f"This run used the 'quick' ~{_quick_trial_count()}-trial qCSF mode "
                        f"rather than the recommended {RECOMMENDED_MIN_TRIALS}+ trials. "
                        "The plug-in AULCSF estimate is systematically biased low by about "
                        "0.06-0.08 log units at 100 trials (see "
                        "docs/methods/contrast_sensitivity_function.md and "
                        "vpsych.core.procedures.qcsf's module docstring); treat this estimate "
                        "as indicative, not final."
                    ),
                ),
            ]
        if self._reduced_coverage:
            quality_flags = [
                *quality_flags,
                QualityFlag(
                    code="reduced_csf_frequency_coverage",
                    severity="warning",
                    message=(
                        f"This display/viewing-distance combination could only test "
                        f"{self._freq_low_cpd:.2g}-{self._freq_high_cpd:.2g} cpd (standard range "
                        f"is {STANDARD_RANGE_LOW_CPD:g}-{STANDARD_RANGE_HIGH_CPD:g} cpd). A "
                        "longer viewing distance increases the rendering-fidelity ceiling "
                        "(see TestRequirements.min_viewing_distance_cm)."
                    ),
                ),
            ]

        eye = str(trials["eye"].iloc[0]) if len(trials) else "OU"
        extra = dict(estimate.extra)
        extra["frequency_range_tested_cpd"] = [self._freq_low_cpd, self._freq_high_cpd]
        extra["cutoff_spatial_frequency_cpd"] = self._acuity_cutoff_cpd(
            estimate.extra["posterior_mean"]
        )
        estimate_with_extra = ThresholdEstimate(
            value=estimate.value,
            ci_low=estimate.ci_low,
            ci_high=estimate.ci_high,
            ci_level=estimate.ci_level,
            units=estimate.units,
            method=estimate.method,
            extra=extra,
        )

        return TestSummary(
            task_id=self.spec.id,
            task_version=self.spec.version,
            eye=eye,
            run=1,
            estimate=estimate_with_extra,
            fit_params=estimate.extra.get("posterior_mean", {}),
            gof={},
            quality_flags=quality_flags,
            n_trials=len(non_catch),
            n_catch=n_catch,
            catch_lapse_rate=catch_lapse_rate,
            analysis_version=self.spec.version,
        )

    @staticmethod
    def _acuity_cutoff_cpd(posterior_mean_params: dict[str, float], n_pts: int = 400) -> float:
        """The acuity-equivalent cutoff spatial frequency: where the CSF crosses CS = 1.

        Evaluated from the plug-in CSF at the posterior-mean parameters
        (the same point-estimate convention `QCSF.estimate()` uses for
        AULCSF itself, see that method's docstring), on a dense log-spaced
        grid from 0.1 to 200 cpd; returns the highest frequency in that
        grid with plug-in log10 CS >= 0 (CS >= 1), or the grid's lowest
        frequency if even that is already below 1 (an extremely
        low-sensitivity CSF).
        """
        freqs = np.geomspace(0.1, 200.0, n_pts)
        log_cs = log_contrast_sensitivity(
            freqs,
            posterior_mean_params["peak_gain_log10cs"],
            posterior_mean_params["peak_freq_cpd"],
            posterior_mean_params["bandwidth_octaves"],
            posterior_mean_params["low_freq_truncation_log10"],
        )
        above = np.where(log_cs >= 0.0)[0]
        if above.size == 0:
            return float(freqs[0])
        return float(freqs[above[-1]])

    def csf_curve(
        self,
        trials: pd.DataFrame,
        n_points: int = 50,
        mc_samples: int = 2000,
        mc_seed: int = 20260916,
    ) -> dict[str, Any]:
        """Figure-ready CSF curve (log10 CS vs. cpd) with an approximate credible band.

        For Phase 3 plotting, not the primary AULCSF estimate (`summarize`).
        Replays the same fresh, deterministic procedure `summarize` does,
        reads off its posterior mean/SD for the 4 CSF parameters
        (`QCSF.estimate().extra`), then Monte Carlo samples `mc_samples`
        *independent*-normal draws around those marginal mean/SD pairs to
        report a percentile band at `n_points` log-spaced frequencies
        across the range this run actually tested
        (`self._freq_low_cpd`-`self._freq_high_cpd`).

        This is a **documented approximation**: it treats the 4 CSF
        parameters' marginal posteriors as independent Gaussians, ignoring
        any posterior correlation between them -- unlike `summarize`'s
        AULCSF credible interval, which is exact (computed directly from
        the joint grid posterior). It is intended only for a quick
        visual credible band on a curve plot, not for inference. Uses a
        fixed local seed (`mc_seed`, not `self.rng`), so this is
        reproducible from `trials` alone.

        Returns:
            A dict with `spatial_frequency_cpd`, `log10_cs_mean`,
            `log10_cs_ci_low`, `log10_cs_ci_high` (each a list of floats,
            length `n_points`), and `ci_level`.
        """
        procedure = self._replay_procedure(trials)
        estimate = procedure.estimate()
        means = estimate.extra["posterior_mean"]
        sds = estimate.extra["posterior_sd"]

        freqs = np.geomspace(self._freq_low_cpd, self._freq_high_cpd, n_points)
        mean_curve = log_contrast_sensitivity(
            freqs,
            means["peak_gain_log10cs"],
            means["peak_freq_cpd"],
            means["bandwidth_octaves"],
            means["low_freq_truncation_log10"],
        )

        mc_rng = np.random.default_rng(mc_seed)
        gain_s = mc_rng.normal(
            means["peak_gain_log10cs"], max(sds["peak_gain_log10cs"], 1e-6), mc_samples
        )
        freq_s = np.clip(
            mc_rng.normal(means["peak_freq_cpd"], max(sds["peak_freq_cpd"], 1e-6), mc_samples),
            1e-3,
            None,
        )
        bw_s = np.clip(
            mc_rng.normal(
                means["bandwidth_octaves"], max(sds["bandwidth_octaves"], 1e-6), mc_samples
            ),
            1e-3,
            None,
        )
        trunc_s = np.clip(
            mc_rng.normal(
                means["low_freq_truncation_log10"],
                max(sds["low_freq_truncation_log10"], 1e-6),
                mc_samples,
            ),
            0.0,
            None,
        )

        curves = log_contrast_sensitivity(
            freqs[:, None], gain_s[None, :], freq_s[None, :], bw_s[None, :], trunc_s[None, :]
        )
        ci_low = np.percentile(curves, 2.5, axis=1)
        ci_high = np.percentile(curves, 97.5, axis=1)

        return {
            "spatial_frequency_cpd": freqs.tolist(),
            "log10_cs_mean": mean_curve.tolist(),
            "log10_cs_ci_low": ci_low.tolist(),
            "log10_cs_ci_high": ci_high.tolist(),
            "ci_level": 0.95,
            "method": "independent_normal_monte_carlo_of_posterior_marginals",
        }
