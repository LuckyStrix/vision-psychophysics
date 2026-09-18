"""Color discrimination: a Cambridge Colour Test (CCT)-style trivector task.

Measures chromatic discrimination thresholds along the three "confusion
line" axes (protan, deutan, tritan) using the luminance-noise, Landolt-C
paradigm introduced for computerized dichromacy/anomalous-trichromacy
screening by Mollon & Reffin (1989) and formalized as the Cambridge Colour
Test (CCT) by Regan, Reffin & Mollon (1994); see also Reffin, Astell &
Mollon (1991). Full citations are in `spec.citations` and repeated, with
equations and parameter choices, in `docs/methods/color_discrimination.md`
(the authoritative reference for this test -- this module docstring only
summarizes the design).

**Stimulus** (see `vpsych.tests_catalog.color_discrimination.discs`): a
square field of randomly sized, non-overlapping discs is generated each
trial. Every disc is independently assigned a luminance drawn uniformly from
a fixed range (CCT: roughly 8-18 cd/m^2; this test uses the same range by
default -- see `ColorDiscriminationParams.luminance_min_cdm2`/
`luminance_max_cdm2` -- and logs the actual range used in every trial's
`stimulus_params`, per this test's requirement to log it if it must be
rescaled for a display whose luminance range cannot support it). This
per-disc luminance noise is the whole point of the design: discs inside a
Landolt-C-shaped target region carry the trial's *target* chromaticity while
every other disc carries the *background* chromaticity, but since luminance
is randomized independently per disc regardless of region, an edge- or
luminance-based strategy cannot reveal the gap -- only genuine chromatic
discrimination can.

**Trivector axes** (see
`vpsych.tests_catalog.color_discrimination.colorspace` and
`vpsych.core.calibration.color.confusion_line_direction_uv`): the target
chromaticity is displaced from a fixed background chromaticity (CCT default,
CIE 1976 u'v' ~ (0.1977, 0.4689) -- see `DEFAULT_BACKGROUND_UV`; this is
used directly if in gamut at the configured luminance range, otherwise
`_resolve_background_uv` falls back to the nearest in-gamut point and logs
that it did) along one of three confusion-line directions, each running
from the background toward a copunctal point (protan/deutan/tritan; Vienot,
Brettel & Mollon 1999's Table I values, the same ones
`vpsych.core.calibration.color.COPUNCTAL_POINTS_XY` already uses). The
displacement magnitude is expressed in u'v' units x1e-4 (CCT convention);
its QUEST+ intensity grid runs on a `log10` scale from
`MIN_DISPLACEMENT_UV_X1E4` (5) up to that axis's own gamut-limited maximum
at the configured luminance-noise range (`ColorDiscriminationTest`'s
`_axis_max_displacement_x1e4`, computed once per test instance from the
active calibration -- see `colorspace.max_in_gamut_displacement_uv`), *not*
CCT's own nominal ceiling of ~1100 (that ceiling is itself gamut-limited on
CCT's own CRT hardware; this test computes its own display's limit rather
than assuming that number transfers).

**Interleaving**: the three axes are run as three independent QUEST+
procedures (`vpsych.core.procedures.questplus_procedure.QuestPlusProcedure`),
composed into one `vpsych.core.procedures.base.MultiParamProcedure` by
`vpsych.tests_catalog.color_discrimination.procedure.TrivectorProcedure`
(this test's own small composite procedure -- neither a single
`AdaptiveProcedure` nor the existing `QCSF` supports interleaving three
independent procedures, so this package implements it directly rather than
touching `vpsych.core.procedures`). The axis to test is chosen uniformly at
random each trial, via `trial_ctx["rng"]` (logged in `stimulus_params["axis"]`
every trial).

**4AFC / criterion**: gap orientation is a 4-alternative forced choice (up/
down/left/right arrow keys), guess rate 0.25. Because every psychometric
family this project uses satisfies `F(0) == 0.5` by construction (see
`docs/METHODS.md`'s "Psychometric function families" section), each axis's
raw QUEST+ `threshold` parameter is *already* the intensity at `p_correct`
exactly halfway between the guess rate and `1 - lapse_rate` -- the
documented criterion this test reports (see
`TrivectorProcedure.estimate`'s docstring for the one-line proof), with no
further `intensity_at_p_correct` conversion needed. This differs from CCT's
own per-axis criterion, an 11-reversal 1-up/1-down transformed staircase
(Levitt 1971) targeting the 50%-of-range reversal mean, not a fitted
psychometric-function criterion; `docs/methods/color_discrimination.md`
documents this difference explicitly.

**Calibration dependence**: this test requires only a gamma (luminance)
calibration of at least grade B (`TestRequirements.min_luminance_grade="B"`)
-- a *measured* color calibration is deliberately **not** required
(`needs_color_calibration=False`), since requiring one would make this test
unrunnable for most participants. Running with the sRGB-assumed (grade C)
color default is explicitly supported, but `summarize()` always adds a
`"critical"`-severity quality flag in that case (see
`ColorDiscriminationTest.summarize`) making clear the reported
chromaticities are nominal, not measured, and results are not
research-grade or comparable across displays -- this test's
`description_participant` never claims a diagnosis either way (see
`docs/methods/color_discrimination.md`, "Calibration dependence" and
"Limitations").
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from vpsych.core.calibration.color import (
    confusion_line_direction_uv,
    matrix_from_calibration,
    xy_to_uv_prime,
)
from vpsych.core.calibration.dither import dither_to_uint8
from vpsych.core.procedures.questplus_procedure import QuestPlusProcedure
from vpsych.data.quality import check_threshold_at_range_edge, compute_quality_flags
from vpsych.data.schemas import QualityFlag, TestSummary
from vpsych.tests_catalog.base import (
    PresentedTrial,
    PsychophysicalTest,
    TestRequirements,
    TestSpec,
    register_test,
)
from vpsych.tests_catalog.color_discrimination import colorspace, discs
from vpsych.tests_catalog.color_discrimination.discs import RESPONSE_KEYS, Disc
from vpsych.tests_catalog.color_discrimination.procedure import AXES, TrivectorProcedure

__all__ = [
    "AXES",
    "DEFAULT_BACKGROUND_UV",
    "DEFAULT_LUMINANCE_RANGE_CDM2",
    "MIN_DISPLACEMENT_UV_X1E4",
    "RESPONSE_KEYS",
    "ColorDiscriminationParams",
    "ColorDiscriminationTest",
]

#: CCT default background chromaticity, CIE 1976 u'v' (Regan, Reffin &
#: Mollon 1994; Mollon & Reffin 1989) -- notably very close to the CIE D65
#: white point (u'=0.1978, v'=0.4683), which is why an out-of-gamut
#: background is expected to be a rare edge case in practice (see
#: `ColorDiscriminationTest._resolve_background_uv`).
DEFAULT_BACKGROUND_UV: tuple[float, float] = (0.1977, 0.4689)

#: CCT's per-disc luminance noise range, cd/m^2 (Regan, Reffin & Mollon
#: 1994). Used verbatim unless a display's calibrated range cannot support
#: it (not currently auto-rescaled by this test -- see
#: `ColorDiscriminationParams.luminance_min_cdm2`/`luminance_max_cdm2` to
#: override for a display with a narrower usable range; the chosen range is
#: always logged in `stimulus_params`/`TestSummary.fit_params`).
DEFAULT_LUMINANCE_RANGE_CDM2: tuple[float, float] = (8.0, 18.0)

#: Landolt-C outer diameter, CCT default, degrees of visual angle.
OUTER_DIAMETER_DEG = 4.3

#: Landolt-C gap width, CCT default, degrees of visual angle (arc length at
#: the ring's mean radius -- see `discs.is_in_target_region`).
GAP_DEG = 1.0

#: Field side length as a multiple of `OUTER_DIAMETER_DEG`, giving the C
#: some surrounding disc field rather than filling the whole square exactly.
FIELD_MARGIN_FACTOR = 1.3

#: Default disc diameter range, degrees of visual angle.
MIN_DISC_DIAMETER_DEG = 0.15
MAX_DISC_DIAMETER_DEG = 0.45

#: 4AFC guess rate (chance rate for a 4-alternative gap-orientation judgment).
GUESS_RATE = 0.25

#: Floor of the QUEST+ intensity/threshold grid, u'v' x1e-4 units (CCT uses
#: a comparable floor; below this, sub-pixel chromatic steps are not
#: reliably distinguishable from dithering/quantization noise on an 8-bit
#: display even with bit-stealing dithering -- see
#: `vpsych.core.calibration.dither`).
MIN_DISPLACEMENT_UV_X1E4 = 5.0

#: CCT's own nominal per-axis ceiling, u'v' x1e-4 units -- cited for context
#: only; this test always computes its own gamut-limited per-axis ceiling
#: instead (see the module docstring's "Trivector axes" section) rather than
#: assuming this number transfers to an arbitrary display.
CCT_NOMINAL_MAX_DISPLACEMENT_UV_X1E4 = 1100.0

#: QUEST+ grids: number of intensity/threshold levels per axis, and the
#: fixed slope/lapse-rate grids (log10-displacement units; chosen to span a
#: plausible discrimination-slope range, following the same style as
#: `vpsych.tests_catalog._example`'s illustrative grids -- not derived from
#: a specific published CCT slope estimate, since QUEST+ itself, unlike
#: CCT's staircase, does not require one to be assumed a priori beyond this
#: search grid).
N_INTENSITY_LEVELS = 20
N_THRESHOLD_LEVELS = 10
DEFAULT_SLOPE_VALUES = [0.2, 0.35, 0.5, 0.65, 0.8]
DEFAULT_LAPSE_RATE_VALUES = [0.0, 0.02, 0.04]

#: Number of luminance samples across the noise range used to verify gamut
#: safety (background resolution and per-axis displacement ceiling); see
#: `colorspace.max_in_gamut_displacement_uv`.
DEFAULT_N_LUMINANCE_GAMUT_SAMPLES = 5


class ColorDiscriminationParams(BaseModel):
    """Configurable parameters for `ColorDiscriminationTest`.

    Every field has a default matching the CCT-style design described in
    this package's module docstring, so `{}` is a valid `params` dict.
    """

    model_config = ConfigDict(frozen=True)

    max_trials: int = Field(
        default=180,
        ge=30,
        description="Total QUEST+ trial budget, summed across all three interleaved axes.",
    )
    outer_diameter_deg: float = Field(
        default=OUTER_DIAMETER_DEG, gt=0, description="Landolt-C outer diameter, degrees."
    )
    gap_deg: float = Field(default=GAP_DEG, gt=0, description="Landolt-C gap width, degrees.")
    luminance_min_cdm2: float = Field(
        default=DEFAULT_LUMINANCE_RANGE_CDM2[0],
        gt=0,
        description="Lower bound of the per-disc luminance-noise range, cd/m^2.",
    )
    luminance_max_cdm2: float = Field(
        default=DEFAULT_LUMINANCE_RANGE_CDM2[1],
        gt=0,
        description="Upper bound of the per-disc luminance-noise range, cd/m^2.",
    )
    min_disc_diameter_deg: float = Field(
        default=MIN_DISC_DIAMETER_DEG, gt=0, description="Minimum disc diameter, degrees."
    )
    max_disc_diameter_deg: float = Field(
        default=MAX_DISC_DIAMETER_DEG, gt=0, description="Maximum disc diameter, degrees."
    )
    stimulus_duration_ms: float = Field(
        default=2000.0,
        gt=0,
        description="Stimulus presentation duration, ms (CCT: 2-3 s; converted to frames).",
    )
    fixation_ms: float = Field(default=500.0, ge=0, description="Fixation phase duration, ms.")
    iti_ms: float = Field(default=500.0, ge=0, description="Inter-trial interval duration, ms.")
    n_luminance_gamut_samples: int = Field(
        default=DEFAULT_N_LUMINANCE_GAMUT_SAMPLES,
        ge=2,
        description=(
            "Number of luminance levels sampled across [luminance_min_cdm2, "
            "luminance_max_cdm2] when computing the per-axis gamut-safe displacement ceiling."
        ),
    )

    @property
    def stroke_width_deg(self) -> float:
        """Landolt-C ring stroke width: `outer_diameter_deg / 5` (standard optotype convention)."""
        return self.outer_diameter_deg / 5.0

    @property
    def field_size_deg(self) -> float:
        """Square disc-field side length: `outer_diameter_deg * FIELD_MARGIN_FACTOR`."""
        return self.outer_diameter_deg * FIELD_MARGIN_FACTOR


@register_test
class ColorDiscriminationTest(PsychophysicalTest):
    """Cambridge Colour Test-style trivector color-discrimination task.

    See this module's docstring for the full design; briefly, three
    interleaved QUEST+ procedures (`TrivectorProcedure`) estimate a
    chromatic discrimination threshold (u'v' x1e-4 displacement) along the
    protan, deutan, and tritan confusion-line axes from a 4AFC Landolt-C
    gap-orientation judgment embedded in a randomized-luminance disc field.
    """

    spec = TestSpec(
        id="color_discrimination",
        name="Color Discrimination (Trivector)",
        version="0.1.0",
        domain="color",
        description_participant=(
            "A patch made of many small colored dots will appear, with a gap somewhere on its "
            "ring-shaped edge. Press the arrow key that matches the direction of the gap (up, "
            "down, left, or right) as accurately as you can, even if you are not certain."
        ),
        description_technical=(
            "Cambridge Colour Test-style trivector chromatic discrimination threshold: a "
            "luminance-noise, Landolt-C paradigm (Mollon & Reffin 1989; Regan, Reffin & Mollon "
            "1994) measuring discrimination thresholds along the protan, deutan, and tritan "
            "confusion-line axes via three interleaved QUEST+ (Watson 2017) procedures. 4AFC "
            "gap-orientation judgment, guess rate 0.25. Requires only a gamma (luminance) "
            "calibration; a measured color calibration is not required, but its absence is "
            "flagged as a critical data-quality concern. See "
            "docs/methods/color_discrimination.md for the full method, equations, and "
            "limitations."
        ),
        measures=(
            "Chromatic discrimination threshold (u'v' x1e-4 displacement) along the protan, "
            "deutan, and tritan confusion-line axes."
        ),
        output_units="log10_uv_displacement_x1e4",
        estimated_minutes=6.0,
        allowed_eyes=["OD", "OS", "OU"],
        requirements=TestRequirements(
            needs_gamma_calibration=True,
            needs_color_calibration=False,
            min_luminance_grade="B",
        ),
        citations=[
            "Mollon, J. D., & Reffin, J. P. (1989). A computer-controlled colour vision test "
            "that combines the principles of Chibret and of Stilling. Journal of Physiology, "
            "414, 5P.",
            "Regan, B. C., Reffin, J. P., & Mollon, J. D. (1994). Luminance noise and the rapid "
            "determination of discrimination ellipses in colour deficiency. Vision Research, "
            "34(10), 1279-1299.",
            "Reffin, J. P., Astell, S., & Mollon, J. D. (1991). Trials of a computer-controlled "
            "colour vision test that preserves the advantages of pseudo-isochromatic plates. In "
            "B. Drum, J. D. Mollon, & G. Verriest (Eds.), Colour Vision Deficiencies X (pp. "
            "69-76). Kluwer.",
            "Vienot, F., Brettel, H., & Mollon, J. D. (1999). Digital video colourmaps for "
            "checking the legibility of displays by dichromats. Color Research & Application, "
            "24(4), 243-252.",
            "Watson, A. B. (2017). QUEST+: A general multidimensional Bayesian adaptive "
            "psychometric method. Journal of Vision, 17(3):10.",
        ],
        params_model=ColorDiscriminationParams,
        hidden=False,
    )

    def __init__(
        self,
        params: BaseModel,
        display: Any,
        calibration: Any,
        rng: np.random.Generator,
    ) -> None:
        assert isinstance(params, ColorDiscriminationParams)
        self.params = params
        self.display = display
        self.calibration = calibration
        self.rng = rng
        self._n_presented = 0
        self._stims: dict[str, Any] = {}

        self._rgb_to_xyz_abs = matrix_from_calibration(self.calibration.color, absolute=True)
        self._luminance_samples_cdm2 = [
            float(v)
            for v in np.linspace(
                self.params.luminance_min_cdm2,
                self.params.luminance_max_cdm2,
                self.params.n_luminance_gamut_samples,
            )
        ]
        self.background_uv, self.background_fallback_used = self._resolve_background_uv()
        self._direction_uv: dict[str, tuple[float, float]] = {
            axis: confusion_line_direction_uv(self.background_uv, axis) for axis in AXES
        }
        self._axis_max_displacement_x1e4: dict[str, float] = {
            axis: colorspace.max_in_gamut_displacement_uv(
                self.background_uv,
                direction,
                self._rgb_to_xyz_abs,
                self._luminance_samples_cdm2,
            )
            * 1e4
            for axis, direction in self._direction_uv.items()
        }

    def _resolve_background_uv(self) -> tuple[tuple[float, float], bool]:
        """Use the CCT default background chromaticity if in gamut, else the nearest in-gamut point.

        Returns:
            `(background_uv, fallback_used)`.
        """
        if colorspace.chromaticity_in_gamut_at_luminances(
            DEFAULT_BACKGROUND_UV, self._rgb_to_xyz_abs, self._luminance_samples_cdm2
        ):
            return DEFAULT_BACKGROUND_UV, False

        white = self.calibration.color.white
        white_uv = xy_to_uv_prime(white.x, white.y)
        safe_uv = (float(white_uv[0]), float(white_uv[1]))
        resolved = colorspace.nearest_in_gamut_point(
            DEFAULT_BACKGROUND_UV, safe_uv, self._rgb_to_xyz_abs, self._luminance_samples_cdm2
        )
        return resolved, True

    def _axis_min_displacement_x1e4(self, axis: str) -> float:
        axis_max = self._axis_max_displacement_x1e4[axis]
        return min(MIN_DISPLACEMENT_UV_X1E4, axis_max * 0.5)

    def make_procedure(self) -> TrivectorProcedure:
        axis_procedures: dict[str, QuestPlusProcedure] = {}
        for axis in AXES:
            axis_max = self._axis_max_displacement_x1e4[axis]
            axis_min = self._axis_min_displacement_x1e4(axis)
            if axis_max <= axis_min:
                raise RuntimeError(
                    f"The active display's gamut is too restrictive along the {axis} confusion "
                    f"axis at background u'v'={self.background_uv} and luminance range "
                    f"[{self.params.luminance_min_cdm2}, {self.params.luminance_max_cdm2}] "
                    f"cd/m^2 (max displacement {axis_max:.2f} x1e-4 u'v' units); "
                    "color_discrimination cannot run on this display/calibration."
                )
            log_min, log_max = math.log10(axis_min), math.log10(axis_max)
            intensity_values = [float(v) for v in np.linspace(log_min, log_max, N_INTENSITY_LEVELS)]
            threshold_values = [float(v) for v in np.linspace(log_min, log_max, N_THRESHOLD_LEVELS)]
            axis_procedures[axis] = QuestPlusProcedure(
                intensity_values=intensity_values,
                intensity_units="log10_uv_displacement_x1e4",
                threshold_values=threshold_values,
                slope_values=list(DEFAULT_SLOPE_VALUES),
                guess_rate=GUESS_RATE,
                lapse_rate_values=list(DEFAULT_LAPSE_RATE_VALUES),
                function="weibull",
                max_trials=None,
            )
        return TrivectorProcedure(
            axis_procedures=axis_procedures,
            rng=self.rng,
            max_trials_total=self.params.max_trials,
        )

    def make_catch_trial_intensity(self) -> float:
        """A large, comfortably in-gamut-for-every-axis log10 displacement.

        The axis a catch trial actually uses is chosen at random inside
        `present()` (see the module docstring's "Catch trials" note in
        `docs/methods/color_discrimination.md`), so this value must be safe
        regardless of which axis is drawn -- hence the minimum over all
        three axes' own ceilings, at 90% of that minimum to leave headroom
        against the bisection search's finite precision.
        """
        safe_max = min(self._axis_max_displacement_x1e4.values()) * 0.9
        safe_max = max(safe_max, MIN_DISPLACEMENT_UV_X1E4)
        return float(math.log10(safe_max))

    def build_stimuli(self, win: Any) -> dict[str, Any]:
        if win is None:
            return {}
        from psychopy import visual

        fixation = visual.TextStim(win, text="+", height=self.display.deg_to_px(0.3), units="pix")
        self._stims = {"fixation": fixation}
        return self._stims

    def _render_disc_colors(
        self, disc_list: list[Disc], target_uv: tuple[float, float], rng: np.random.Generator
    ) -> tuple[list[tuple[float, float]], list[float], list[tuple[float, float, float]]]:
        """Per-disc pixel positions (px), diameters (px), and dithered draw colors ([0, 1] rgb1)."""
        xys: list[tuple[float, float]] = []
        sizes_px: list[float] = []
        colors: list[tuple[float, float, float]] = []
        for d in disc_list:
            y_cdm2 = float(
                rng.uniform(self.params.luminance_min_cdm2, self.params.luminance_max_cdm2)
            )
            uv = target_uv if d.is_target else self.background_uv
            render = colorspace.render_disc(
                uv[0],
                uv[1],
                y_cdm2,
                self.calibration.color,
                self.calibration.gamma,
                rgb_to_xyz_abs=self._rgb_to_xyz_abs,
            )
            dithered = dither_to_uint8(np.array(render.drive_rgb, dtype=np.float64), rng)
            colors.append(
                (float(dithered[0]) / 255.0, float(dithered[1]) / 255.0, float(dithered[2]) / 255.0)
            )
            xys.append((self.display.deg_to_px(d.x_deg), self.display.deg_to_px(d.y_deg)))
            sizes_px.append(self.display.deg_to_px(d.diameter_deg))
        return xys, sizes_px, colors

    def present(
        self, win: Any, intensity_or_stimulus: Any, trial_ctx: dict[str, Any]
    ) -> PresentedTrial:
        self._n_presented += 1
        rng = trial_ctx["rng"]

        if isinstance(intensity_or_stimulus, dict):
            axis_idx = round(float(intensity_or_stimulus["axis"]))
            intensity_log10 = float(intensity_or_stimulus["intensity"])
        else:
            # Catch trial: make_catch_trial_intensity() returns a bare float
            # (see its docstring); the axis is chosen at random here.
            axis_idx = int(rng.integers(len(AXES)))
            intensity_log10 = float(intensity_or_stimulus)

        axis = AXES[axis_idx]
        requested_x1e4 = 10.0**intensity_log10
        axis_max_x1e4 = self._axis_max_displacement_x1e4[axis]
        clamped = requested_x1e4 > axis_max_x1e4
        used_x1e4 = min(requested_x1e4, axis_max_x1e4)
        displacement_uv = used_x1e4 * 1e-4
        direction = self._direction_uv[axis]
        target_uv = (
            self.background_uv[0] + direction[0] * displacement_uv,
            self.background_uv[1] + direction[1] * displacement_uv,
        )

        orientation = RESPONSE_KEYS[int(rng.integers(len(RESPONSE_KEYS)))]

        disc_list = discs.generate_disc_field(
            rng,
            field_size_deg=self.params.field_size_deg,
            min_diameter_deg=self.params.min_disc_diameter_deg,
            max_diameter_deg=self.params.max_disc_diameter_deg,
            outer_diameter_deg=self.params.outer_diameter_deg,
            stroke_width_deg=self.params.stroke_width_deg,
            gap_deg=self.params.gap_deg,
            orientation=orientation,
        )
        n_target_discs = sum(1 for d in disc_list if d.is_target)

        stimulus_params: dict[str, Any] = {
            "axis": axis_idx,
            "axis_name": axis,
            "intensity": intensity_log10,
            "requested_displacement_uv_x1e4": requested_x1e4,
            "displacement_uv_x1e4": used_x1e4,
            "clamped": bool(clamped),
            "axis_max_displacement_uv_x1e4": axis_max_x1e4,
            "orientation": orientation,
            "correct_response": orientation,
            "n_discs": len(disc_list),
            "n_target_discs": n_target_discs,
            "target_uv": [target_uv[0], target_uv[1]],
            "background_uv": [self.background_uv[0], self.background_uv[1]],
            "luminance_range_cdm2": [
                self.params.luminance_min_cdm2,
                self.params.luminance_max_cdm2,
            ],
            "color_grade": self.calibration.color_grade,
            "luminance_grade": self.calibration.luminance_grade,
        }
        trial_ctx["stimulus_params"] = stimulus_params
        trial_ctx["correct_response"] = orientation

        observer = trial_ctx["simulated_observer"]
        if observer is not None:
            is_correct = observer.decide_correct(stimulus_params, rng)
            response = self.simulated_response(is_correct, stimulus_params, rng)
            stimulus_frames = self.display.frames_for_ms(self.params.stimulus_duration_ms)
            return PresentedTrial(
                response=response,
                rt_s=0.5,
                stimulus_onset_s=float(self._n_presented),
                n_dropped_frames=0,
                frame_intervals_s=[1.0 / 60.0] * max(stimulus_frames, 1),
            )

        from psychopy import visual

        fixation = self._stims["fixation"]
        fixation_frames = self.display.frames_for_ms(self.params.fixation_ms)
        stimulus_frames = self.display.frames_for_ms(self.params.stimulus_duration_ms)
        iti_frames = self.display.frames_for_ms(self.params.iti_ms)

        for _ in range(fixation_frames):
            fixation.draw()
            win.flip()

        xys, sizes_px, colors = self._render_disc_colors(disc_list, target_uv, rng)
        field = visual.ElementArrayStim(
            win,
            nElements=max(len(disc_list), 1),
            elementTex=None,
            elementMask="circle",
            xys=xys,
            sizes=sizes_px,
            colors=colors,
            colorSpace="rgb1",
            units="pix",
        )

        keyboard = trial_ctx["keyboard"]
        keyboard.clearEvents()
        onset_s = None
        # Stimulus is shown for a fixed duration (CCT: "2-3 s or until
        # response" -- this implementation always shows the full duration,
        # then collects the response from a blank screen, rather than
        # per-frame-polling for an early response; see
        # docs/methods/color_discrimination.md, "Limitations", for why this
        # simplification was chosen).
        for frame in range(stimulus_frames):
            field.draw()
            flip_time = win.flip()
            if frame == 0:
                onset_s = flip_time

        keys = keyboard.waitKeys(keyList=RESPONSE_KEYS, timeStamped=True)
        response, rt_s = (keys[0][0], keys[0][1] - (onset_s or 0.0)) if keys else (None, None)

        for _ in range(iti_frames):
            win.flip()

        return PresentedTrial(
            response=response,
            rt_s=rt_s,
            stimulus_onset_s=onset_s or 0.0,
            n_dropped_frames=0,
            frame_intervals_s=[1.0 / self.display.refresh_hz] * stimulus_frames,
        )

    def response_keys(self) -> list[str]:
        return list(RESPONSE_KEYS)

    def score(self, response: Any, stimulus_params: dict[str, Any]) -> bool:
        return bool(response == stimulus_params["correct_response"])

    def instructions(self) -> str:
        return self.spec.description_participant

    def summarize(self, trials: pd.DataFrame) -> TestSummary:
        """Recompute the trivector estimate deterministically from the raw trials.

        Replays a fresh `TrivectorProcedure` (via `make_procedure`, which is
        a pure function of `self.calibration`/`self.params`, no randomness)
        over every main-block, non-catch trial in `trial_index` order,
        reading the axis each trial actually used from its stored
        `stimulus_params["axis"]` -- so `TrivectorProcedure.next_stimulus`'s
        own random axis draws (which depend on `self.rng`'s live state) are
        never needed for reanalysis, matching `docs/WRITING_A_TEST.md`
        section 10's "pure function of trials alone" requirement.
        """
        main = trials[trials["block"] == "main"]
        non_catch = main[~main["is_catch"]].sort_values("trial_index")
        catch = main[main["is_catch"]]

        procedure = self.make_procedure()
        for _, row in non_catch.iterrows():
            stim_params = row["stimulus_params"]
            stimulus = {"axis": float(stim_params["axis"]), "intensity": float(row["intensity"])}
            procedure.update(stimulus, bool(row["correct"]))
        estimate = procedure.estimate()

        n_catch = len(catch)
        catch_lapse_rate = float((~catch["correct"]).mean()) if n_catch else 0.0
        dropped_fraction = (
            float(trials["n_dropped_frames_trial"].sum()) / len(trials) if len(trials) else 0.0
        )

        quality_flags: list[QualityFlag] = compute_quality_flags(
            catch_lapse_rate=catch_lapse_rate,
            n_catch=n_catch,
            dropped_fraction=dropped_fraction,
            n_trials=len(non_catch),
            min_trials=self.params.max_trials,
        )

        per_axis = estimate.extra.get("per_axis", {})
        for axis in AXES:
            info = per_axis.get(axis)
            if not info:
                continue
            axis_max = self._axis_max_displacement_x1e4[axis]
            axis_min = self._axis_min_displacement_x1e4(axis)
            quality_flags += check_threshold_at_range_edge(
                info["threshold_log10_uv_x1e4"], math.log10(axis_min), math.log10(axis_max)
            )

        color_grade = self.calibration.color_grade
        if color_grade == "C":
            quality_flags.append(
                QualityFlag(
                    code="color_calibration_not_measured",
                    severity="critical",
                    message=(
                        "Color primaries not measured: chromaticities are nominal; results are "
                        "not research-grade and not comparable across displays."
                    ),
                )
            )
        else:
            quality_flags.append(
                QualityFlag(
                    code="color_calibration_not_measured",
                    severity="info",
                    message=(
                        "Display color primaries were measured (grade A); reported chromaticity "
                        "displacements are colorimetrically calibrated for this display."
                    ),
                )
            )

        eye = str(trials["eye"].iloc[0]) if len(trials) else "OU"
        return TestSummary(
            task_id=self.spec.id,
            task_version=self.spec.version,
            eye=eye,
            run=1,
            estimate=estimate,
            fit_params={
                "background_uv": [self.background_uv[0], self.background_uv[1]],
                "background_fallback_used": self.background_fallback_used,
                "luminance_range_cdm2": [
                    self.params.luminance_min_cdm2,
                    self.params.luminance_max_cdm2,
                ],
                "color_grade": color_grade,
                "luminance_grade": self.calibration.luminance_grade,
                "axis_max_displacement_uv_x1e4": dict(self._axis_max_displacement_x1e4),
            },
            gof={},
            quality_flags=quality_flags,
            n_trials=len(non_catch),
            n_catch=n_catch,
            catch_lapse_rate=catch_lapse_rate,
            analysis_version="0.1.0",
        )
