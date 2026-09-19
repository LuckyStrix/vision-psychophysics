# Color discrimination (trivector)

Measures how finely a participant can discriminate small changes in color
that differ only in chromaticity, not in luminance or edge cues, along the
three "confusion line" directions associated with the three classic
inherited color-vision deficiencies (protan, deutan, tritan). This is not a
diagnostic test: it is a chromatic-discrimination threshold measurement, and
its result is only interpretable relative to normative data collected on
specific, calibrated hardware (see "Classification context" below) -- it
never labels a participant as color-deficient or not.

This document is the authoritative reference for
`vpsych.tests_catalog.color_discrimination` (per `docs/WRITING_A_TEST.md`);
`docs/METHODS.md` intentionally keeps only a placeholder heading for this
test and is not edited by this package.

## Method

The design follows the Cambridge Colour Test (CCT): a luminance-noise,
pseudoisochromatic paradigm introduced by Mollon & Reffin (1989) and
formalized with worked psychophysics by Regan, Reffin & Mollon (1994); see
also Reffin, Astell & Mollon (1991) for an earlier trial of the same
principle. A Landolt-C-shaped target region is embedded in a field of
randomly placed, randomly sized discs. Every disc -- inside or outside the
target region -- is independently assigned a luminance drawn uniformly from
a fixed range (the "luminance noise"). Discs inside the target region carry
the trial's *target* chromaticity; every other disc carries the *background*
chromaticity. Because luminance is randomized per disc regardless of region,
neither an edge-detection strategy nor a "find the darker/lighter patch"
strategy can reveal the target -- only a genuine chromatic difference can.
The participant reports which side (up/down/left/right) the Landolt C's gap
is on: a 4-alternative forced choice (4AFC), guess rate 0.25.

Three independent thresholds -- one per confusion-line axis -- are measured
in a single interleaved run using three QUEST+ (Watson 2017) procedures,
composed by this package's own `TrivectorProcedure`
(`vpsych.tests_catalog.color_discrimination.procedure`), a small
`vpsych.core.procedures.base.MultiParamProcedure` implemented entirely
inside this test package (see "Implementation notes" below for why this was
necessary). Each trial, the axis to test is drawn via **block
randomization** (a shuffled bag of one copy of each axis, refilled whenever
exhausted -- see "Interleaving and threshold criterion" below for why this
replaced an earlier uniform-random draw); the QUEST+ procedure for that
axis alone selects the next intensity and is updated with the outcome.
Each axis's psychometric function is `questplus`'s own native Weibull
(fitted internally by `QuestPlusProcedure` -- **not**
`vpsych.core.psychometric`'s own, differently-parameterized `weibull`
family; see "Interleaving and threshold criterion" below and
`vpsych.core.procedures.questplus_procedure`'s "Criterion conversion
pitfall" docstring section for the distinction and why it matters):

```
p(x) = 1 - lapse - (1 - guess - lapse) * exp(-10**(slope * (x - threshold)))
guess = 0.25   (4AFC)
x = log10(displacement), displacement in u'v' x 1e-4 units
```

## Stimulus

- **Landolt C target**: outer diameter 4.3 deg of visual angle, stroke width
  `outer_diameter / 5` (0.86 deg, the standard optotype convention), gap
  width 1.0 deg (CCT defaults;
  `ColorDiscriminationParams.outer_diameter_deg`/`gap_deg`). The gap is
  specified as an arc length at the ring's mean radius, converted to an
  angular half-width via `half_angle_rad = (gap_deg / 2) / mean_radius_deg`.
- **Disc field**: a square field of side `outer_diameter_deg * 1.3` (5.59
  deg by default) is filled by random sequential adsorption -- discs of
  uniform-random diameter in `[0.15, 0.45]` deg (`min_disc_diameter_deg`/
  `max_disc_diameter_deg`) are proposed at uniform-random positions and
  accepted only if they do not overlap any previously placed disc, until a
  fixed attempt budget is exhausted (the standard "jamming limit" behavior
  of this algorithm; typical realized coverage in this project's own
  simulations is roughly 40-55% of the field area, with several hundred
  discs). A disc is classified as "target" or "background" purely by
  whether its *center* falls inside the Landolt-C ring (see
  `vpsych.tests_catalog.color_discrimination.discs.is_in_target_region`);
  discs are not clipped or split at the ring boundary.
- **Luminance noise**: each disc, independently, is assigned a luminance
  drawn uniformly from `[luminance_min_cdm2, luminance_max_cdm2]`, default
  `[8, 18]` cd/m^2 (Regan, Reffin & Mollon 1994's own range). This range is
  used verbatim by default (not automatically rescaled to a display's
  calibrated range); the *actual* range used is always logged, both per
  trial (`stimulus_params["luminance_range_cdm2"]`) and in the run summary
  (`TestSummary.fit_params["luminance_range_cdm2"]`).
- **Presentation**: fixation (500 ms default) -> stimulus (2000 ms default;
  CCT uses 2-3 s) -> response -> ITI (500 ms default), all converted to
  frames via `DisplayGeometry.frames_for_ms` (never timed by the clock).
  Unlike the literal CCT protocol (stimulus removed as soon as either a
  response is given *or* the maximum duration elapses, whichever comes
  first), this implementation always shows the stimulus for its full fixed
  duration and then collects the response from a blank screen -- see
  "Limitations" below for why.
- **Catch trials**: a suprathreshold displacement -- 90% of the *smallest*
  of the three axes' own gamut-limited ceilings (see below), so it is safe
  regardless of which axis a catch trial's random axis draw lands on -- is
  used on a randomly chosen axis (`ColorDiscriminationTest
  .make_catch_trial_intensity`/`present`).

## Trivector axes and the color pipeline

**Background chromaticity**: CIE 1976 `u'v'` ~ (0.1977, 0.4689), the CCT
default (Regan, Reffin & Mollon 1994; Mollon & Reffin 1989) -- notably very
close to the CIE D65 white point (`u'`=0.1978, `v'`=0.4683), by design (a
near-white background keeps all three confusion-line directions roughly
symmetric and comfortably inside a typical display's gamut). If this
chromaticity is not representable at *every* luminance in the configured
noise range on the active display/calibration, the nearest in-gamut point on
the segment toward the display's own white point is used instead (bisection
search, `colorspace.nearest_in_gamut_point`), and
`TestSummary.fit_params["background_fallback_used"]` records whether this
happened; the actual background used is always logged
(`fit_params["background_uv"]`, and per trial in `stimulus_params`).

**Confusion-line directions**: for each of protan, deutan, and tritan, a
unit vector in `u'v'` from the background toward that deficiency's copunctal
point (`vpsych.core.calibration.color.confusion_line_direction_uv`, using
`COPUNCTAL_POINTS_XY` -- Vienot, Brettel & Mollon 1999, Table I, itself
derived from Smith & Pokorny 1975 cone fundamentals). Colors displaced along
this line from the background are, to first order, indistinguishable from
the background to an observer missing that cone class -- the classic
confusion-line construction every trivector test since Mollon & Reffin
(1989) is built on.

**Displacement units**: the intensity dimension driving each axis's QUEST+
procedure is `log10(displacement)`, where `displacement` is the chromatic
distance from the background along that axis's confusion line, in `u'v' x
1e-4` units (the CCT convention). The domain runs from 5 (below which
sub-pixel chromatic steps are not reliably distinguishable from dithering/
quantization noise on an 8-bit display) up to that axis's own **gamut-
limited maximum at the configured luminance-noise range** -- computed once
per test instance (`ColorDiscriminationTest._axis_max_displacement_x1e4`),
*not* assumed to be CCT's own nominal ceiling of ~1100 (a number that is
itself gamut-limited on CCT's specific CRT hardware and calibration, and
does not transfer to an arbitrary display). The per-axis maximum is found by
bisection search (`colorspace.max_in_gamut_displacement_uv`) for the largest
displacement that keeps the resulting linear RGB in `[0, 1]^3`
*simultaneously* at every sampled luminance across the noise range (5
samples by default, `n_luminance_gamut_samples`) -- not just at the range's
midpoint, since the luminance-noise design means any displayed disc could
land at any luminance in that range.

**Per-disc rendering pipeline** (`vpsych.tests_catalog.color_discrimination
.colorspace`), for a disc's target `(u', v', Y)`:

1. `(u', v') -> (x, y)` (CIE 1931), the closed-form inverse of
   `vpsych.core.calibration.color.xy_to_uv_prime`:
   `x = 9u' / (6u' - 16v' + 12)`, `y = 4v' / (6u' - 16v' + 12)`.
2. `(x, y, Y) -> XYZ`: `X = (Y/y) x`, `Z = (Y/y) (1 - x - y)`.
3. `XYZ -> linear RGB`, via the calibrated (or sRGB-assumed) display
   primaries' absolute `RGB -> XYZ` matrix
   (`vpsych.core.calibration.color.matrix_from_calibration(..., absolute=True)`),
   inverted. A **strict per-trial gamut check** follows immediately: every
   disc's linear RGB must be in `[0, 1]^3` to be representable; if a
   requested axis displacement itself would put any disc out of gamut (this
   should not normally happen, since the axis ceiling above is already
   gamut-safe at every noise luminance, but is still checked per render as a
   safety net), it is clamped to the per-axis maximum and
   `stimulus_params["clamped"] = True` is logged. A threshold estimated at
   that ceiling is flagged `"threshold_at_range_edge"` by
   `vpsych.data.quality.check_threshold_at_range_edge`.
4. Each linear RGB channel is independently **gamma-linearized**
   (`vpsych.core.calibration.gamma.linearize`) against the display's own
   `GammaCalibration`, producing the gamma-encoded drive value actually sent
   to the display. `GammaCalibration` characterizes a single achromatic
   `[lum_min_cdm2, lum_max_cdm2]` range (from a photometer sweep of the
   whole display); this pipeline assumes each channel's own gamma curve
   spans that same shared range (per-channel gamma exponents, if measured,
   *are* used) -- the standard channel-independence assumption for a
   computer display, but see "Limitations".
5. The gamma-encoded drive value is **bit-stealing dithered**
   (`vpsych.core.calibration.dither.dither_to_uint8`, Allard & Faubert 2008)
   before being sent to PsychoPy, using `trial_ctx["rng"]` as the sole
   randomness source -- one dithered draw per disc per trial (not
   re-dithered across frames within a single stimulus presentation, a
   simplification; see "Limitations").

**Round-trip validity**: `tests/tests_catalog/test_color_discrimination.py`
checks headlessly that a rendered disc's drive values, pushed back through
the forward gamma model and primary matrix
(`colorspace.inverse_render_to_uv_y`), reproduce the originally requested
`(u', v', Y)` to well within `1e-4` (this is an algebraically exact
round-trip for the parametric gamma model, limited only by floating-point
precision -- measured error is on the order of `1e-14`-`1e-15`, not merely
`1e-4`), and that a chromaticity intentionally placed outside the display's
gamut triangle is correctly flagged `in_gamut=False` and clamped into
`[0, 1]`.

## Interleaving and threshold criterion

`TrivectorProcedure` wraps three `QuestPlusProcedure` instances (one per
axis), each on its own intensity/threshold/slope grid (20 intensity levels,
10 threshold levels, log10-uniform over that axis's own
`[5, axis_max]` range; slope grid `[0.2, 0.35, 0.5, 0.65, 0.8]`; lapse-rate
grid `[0.0, 0.02, 0.04]` -- implementation choices in the same spirit as
`vpsych.tests_catalog._example`'s illustrative grids, not derived from a
specific published CCT slope estimate). `next_stimulus()` draws the axis via
**block randomization** each trial (via the same `rng` object the owning
test's `present()` receives through `trial_ctx["rng"]`, so the axis
sequence is reproducible from the session's logged seed): a shuffled bag of
exactly one copy of each axis, refilled with a fresh shuffle whenever
exhausted, so every consecutive block of 3 trials tests each axis exactly
once (only a session's final, possibly-partial block can be uneven, by at
most 1 trial). With the default `max_trials=180` (a multiple of 3), every
axis gets *exactly* 60 trials, not just 60 on average -- see "Validation"
below for why this replaced an earlier plain `rng.integers(3)` draw (a real,
measured source of per-axis bias, not just a theoretical concern).
`next_stimulus()` returns `{"axis": 0|1|2, "intensity": log10(displacement)}`;
`update()` routes the outcome to the drawn axis's own `QuestPlusProcedure`.

**Criterion**: `QuestPlusProcedure.estimate().value` is `questplus`'s own
native Weibull threshold -- the intensity at the ~63.2%-of-range point in
*its* parameterization (`guess + (1 - guess - lapse) * (1 - e^-1)`) -- which
is **not** the same as `vpsych.core.psychometric`'s own `F(0) = 0.5`
convention, despite both projects calling their respective anchor parameter
"threshold" (see `vpsych.core.procedures.questplus_procedure`'s "Criterion
conversion pitfall" docstring section; an earlier version of this
paragraph incorrectly claimed the two were the same, which was itself an
instance of that exact pitfall). To report the `F(0)=0.5`-equivalent
criterion -- `p_correct = guess + (1 - guess - lapse) * 0.5`, exactly
halfway between the guess rate (0.25) and `1 - lapse_rate` --
`TrivectorProcedure.estimate()` now explicitly inverts each axis's own
fitted `questplus` curve via `QuestPlusProcedure.intensity_at_p_correct`,
rather than assuming (incorrectly) that the raw threshold already sits
there. **This differs from the original CCT's own criterion**, an
11-reversal 1-up/1-down transformed staircase (Levitt 1971) whose reversal
mean targets ~50% of the tested range at equilibrium, not a
fitted-psychometric-function percent-correct point; the two are not the
same statistic, and a threshold from this implementation should not be
assumed numerically interchangeable with a published CCT threshold measured
the original way.

**Combined estimate**: the primary `TestSummary.estimate.value` is the
arithmetic mean of the three axes' `log10(displacement)` thresholds --
equivalently, the *geometric* mean of their linear (`u'v' x 1e-4`)
displacement thresholds, since `10**mean(log10(t_i)) == geomean(t_i)`.
`ci_low`/`ci_high` are the same mean taken over each axis's own credible
bound (a documented approximation, not a fully joint interval). Full
per-axis thresholds, credible intervals, fitted slope, and fitted lapse rate
are reported in `estimate.extra["per_axis"]`.

## Output

- `output_units = "log10_uv_displacement_x1e4"`.
- `TestSummary.estimate.value`: the combined trivector threshold (see
  above), log10 of a `u'v' x 1e-4` displacement.
- `TestSummary.estimate.extra["per_axis"]`: for each of `protan`, `deutan`,
  `tritan`, the fitted threshold and 95% credible interval (both in
  `log10_uv_x1e4` units and back-transformed to linear `uv_x1e4` units),
  fitted slope, fitted lapse rate, and the number of trials that axis
  actually received.
- `TestSummary.fit_params`: the background chromaticity actually used (and
  whether the CCT default had to be replaced by the nearest-in-gamut
  fallback), the configured luminance-noise range, the active color and
  luminance calibration grades, and each axis's computed gamut-limited
  displacement ceiling.

### Classification context (not part of this test's own output)

Regan, Reffin & Mollon (1994) and later normative work (e.g. Ventura,
Silveira, Rodrigues, Gualtieri, Souza, Bonci & Costa, 2003, "Preliminary
norms for the Cambridge Colour Test," in J. D. Mollon, J. Pokorny, & K.
Knoblauch (Eds.), *Normal and Defective Colour Vision* (pp. 327-334), Oxford
University Press) report normal-trichromat vs. color-deficient ranges for
CCT vector lengths on **specific, calibrated CRT hardware**. Those
normative ranges are cited here purely for orientation and are **not
applied by this test as a diagnosis**: `TestSpec.description_participant`
and every quality-flag message in this package deliberately avoid any
diagnostic claim, since a threshold from an sRGB-assumed, uncalibrated, or
even a differently-calibrated-but-measured display is not directly
comparable to those published ranges (different primaries, different
luminance range achieved, different gamut). Any downstream classification
against normative data is out of scope for this implementation and would
need its own explicit, display-matched calibration protocol.

## Parameters (`ColorDiscriminationParams`)

| Field | Default | Meaning |
|---|---|---|
| `max_trials` | 180 | Total QUEST+ budget, summed across all three interleaved axes. |
| `outer_diameter_deg` | 4.3 | Landolt-C outer diameter. |
| `gap_deg` | 1.0 | Landolt-C gap width (arc length at mean radius). |
| `luminance_min_cdm2`/`luminance_max_cdm2` | 8 / 18 | Per-disc luminance-noise range. |
| `min_disc_diameter_deg`/`max_disc_diameter_deg` | 0.15 / 0.45 | Disc diameter range. |
| `stimulus_duration_ms` | 2000 | Fixed stimulus presentation duration. |
| `fixation_ms`/`iti_ms` | 500 / 500 | Fixation and inter-trial-interval durations. |
| `n_luminance_gamut_samples` | 5 | Luminance grid density used for gamut-safety checks. |

## Requirements

- `needs_gamma_calibration=True`, `min_luminance_grade="B"`: a luminance/
  gamma characterization (at least the no-photometer psychophysical grade)
  is required, since the whole rendering pipeline depends on it to place
  discs at the requested absolute luminances and to gamma-linearize drive
  values at all.
- `needs_color_calibration=False`, no `min_color_grade`: a **measured**
  color-primary calibration is deliberately *not* required. Requiring one
  would make this test unrunnable for most participants (most displays are
  never spectroradiometrically characterized). Running with the sRGB-
  assumed (grade C) color default is fully supported -- but
  `summarize()` **always** adds a quality flag making the consequence
  explicit:
  - Grade C (sRGB-assumed): `severity="critical"`, message "Color primaries
    not measured: chromaticities are nominal; results are not
    research-grade and not comparable across displays."
  - Grade A (measured): `severity="info"`, confirming the displacement
    values are colorimetrically calibrated for this specific display.

  The active color and luminance calibration grades are also always logged
  in `stimulus_params` (per trial) and `TestSummary.fit_params` (per run).

## Citations

- Mollon, J. D., & Reffin, J. P. (1989). A computer-controlled colour vision
  test that combines the principles of Chibret and of Stilling. *Journal of
  Physiology*, 414, 20P. (Physiological Society meeting abstract; predates
  DOI assignment.)
- Regan, B. C., Reffin, J. P., & Mollon, J. D. (1994). Luminance noise and
  the rapid determination of discrimination ellipses in colour deficiency.
  *Vision Research*, 34(10), 1279-1299.
  https://doi.org/10.1016/0042-6989(94)90203-8
- Reffin, J. P., Astell, S., & Mollon, J. D. (1991). Trials of a
  computer-controlled colour vision test that preserves the advantages of
  pseudo-isochromatic plates. In B. Drum, J. D. Moreland, & A. Serra (Eds.),
  *Colour Vision Deficiencies X* (pp. 69-76). Kluwer.
  https://doi.org/10.1007/978-94-011-3774-4_9
- Vienot, F., Brettel, H., & Mollon, J. D. (1999). Digital video colourmaps
  for checking the legibility of displays by dichromats. *Color Research &
  Application*, 24(4), 243-252.
  https://doi.org/10.1002/(SICI)1520-6378(199908)24:4%3C243::AID-COL5%3E3.0.CO;2-3
- Watson, A. B. (2017). QUEST+: A general multidimensional Bayesian adaptive
  psychometric method. *Journal of Vision*, 17(3):10.
  https://doi.org/10.1167/17.3.10
- Allard, R., & Faubert, J. (2008). The noisy-bit method for digital
  displays: converting a resolution limitation into a pseudo-resolution.
  *Behavior Research Methods*, 40(3), 735-743 (bit-stealing dithering, used
  by the rendering pipeline; see `vpsych.core.calibration.dither`).
- Ventura, D. F., Silveira, L. C. L., Rodrigues, A. R., Gualtieri, M.,
  Souza, J. M., Bonci, D., & Costa, M. F. (2003). Preliminary norms for the
  Cambridge Colour Test. In J. D. Mollon, J. Pokorny, & K. Knoblauch (Eds.),
  *Normal and Defective Colour Vision* (pp. 327-334). Oxford University
  Press. (Classification context only, not applied by this test -- see
  above.)

## Validation

`tests/tests_catalog/test_color_discrimination.py`'s
`test_trivector_procedure_recovery_bias_slow` (`@pytest.mark.slow`) runs
`TrivectorProcedure` against a `TrivectorObserver` with known per-axis
ground-truth thresholds (2.2/1.7/1.3 log10 `uv_x1e4` units for protan/
deutan/tritan) at the default trial budget (180 total).

**Phase 4 fix and re-measurement.** Two independent problems were found and
fixed together (both measured on the same 30-repetition sweep, same seeds,
for a fair before/after comparison):

1. **Criterion-conversion bug (item 1)**: `TrivectorProcedure.estimate()`
   used to report each axis's raw, native-parameterization QUEST+ threshold
   directly, on the mistaken belief that it was already at
   `vpsych.core.psychometric`'s `F(0)=0.5` point (see
   `vpsych.core.procedures.questplus_procedure`'s "Criterion conversion
   pitfall" docstring section for why that belief was wrong). It now
   explicitly converts via `QuestPlusProcedure.intensity_at_p_correct`.
2. **Unbalanced interleaving (item 3)**: `next_stimulus()` used to draw the
   axis via a plain `rng.integers(3)` each trial, which gives each axis 1/3
   of trials only *in expectation* -- over a finite 180-trial session,
   binomial sampling variance means axes can end up with substantially
   unequal trial counts by chance, starving whichever axis drew short straw.
   `TrivectorProcedure` now draws axes via block randomization (a shuffled
   bag of one copy of each axis, refilled whenever exhausted), guaranteeing
   every complete block of 3 trials tests each axis exactly once -- with
   `max_trials=180` (a multiple of 3), every axis gets *exactly* 60 trials,
   not just 60 on average.

**Before** (original code, both problems present): per-axis bias about
**0.13-0.18 log10 units** (SD about 0.13-0.17), somewhat higher than
`QCSF`'s ~0.04-0.08 at 100+ trials.

**After** (both fixes applied, same 30-repetition sweep): per-axis bias
about **-0.05 to -0.13 log10 units** (protan -0.049, deutan -0.100, tritan
-0.126; SD 0.11-0.22) -- roughly half the magnitude of the original bias.
The reduction is not complete: a residual bias remains for the same reason
`motion_coherence`/`critical_flicker_fusion` have one after their own item-1
fix (see those tests' own "Validation" sections) -- `TrivectorObserver`
generates data from `vpsych.core.psychometric`'s own sigmoid family while
each axis's `QuestPlusProcedure` fits `questplus`'s different native
family, so even a fully correct conversion and perfectly balanced
interleaving inherit some family-shape mismatch. The committed test uses a
looser bound (`|bias| < 0.35`) than either measurement, to avoid flaking on
ordinary sampling variation (single-run error has SD comparable to the bias
itself) while still catching a much larger, genuinely broken bias; a future
increase in the default trial budget would be the natural way to tighten
this further.

`test_simulated_end_to_end_recovery_of_three_distinct_thresholds` drives a
real `run_session()` (in-process, with a fake writer) against three visibly
different per-axis ground-truth thresholds and checks the recovered values
are within a loose smoke-level tolerance (`< 1.0` log10 units, matching this
project's other tests' smoke-level tolerance) and preserve the ground
truth's relative ordering.

On a representative wide-gamut sRGB-primaries calibration (red
(0.64, 0.33), green (0.30, 0.60), blue (0.15, 0.06), D65 white at 120
cd/m^2, luminance-noise range 8-18 cd/m^2), the computed per-axis gamut
ceilings were approximately **protan 2250, deutan 630, tritan 2225** (`u'v'
x 1e-4` units) -- all comfortably above the 5-unit floor, and, for protan
and tritan, above CCT's own nominal ~1100 ceiling; deutan is the most
gamut-restricted of the three on this primary set, consistent with deutan
confusion lines generally running more nearly parallel to a typical
display's green-primary edge of the sRGB gamut triangle.

## Limitations

- **LCD viewing angle**: LCD panels' chromaticity and luminance both shift
  with viewing angle far more than a CRT's (the hardware CCT was originally
  validated on); this test does not model or correct for viewing-angle
  effects, so results on an LCD may be biased by head position/screen
  geometry in a way this pipeline cannot detect.
- **Display metamerism and observer cone-fundamental variation**: the color
  pipeline uses the Stockman & Sharpe (2000) standard cone fundamentals (via
  `vpsych.core.calibration.color.xyz_to_lms_matrix`) and the CIE 1931
  standard observer throughout; an individual participant's actual cone
  fundamentals (and hence which exact chromaticity is confusable for them)
  can differ measurably from the standard observer, especially for
  color-anomalous participants -- this is a fundamental limitation of any
  standard-observer-based color test, not specific to this implementation.
- **Gamut**: every display has a finite gamut; the per-axis ceiling computed
  here is only as good as the calibration it is computed from (an
  sRGB-assumed, unmeasured calibration reports nominal, not actual,
  ceilings -- see "Requirements" above). A threshold estimated at or near
  its axis's ceiling is flagged `threshold_at_range_edge`
  and should be interpreted as "at least this large," not a precise value.
- **Channel independence assumption**: the gamma-linearization step (see
  "Trivector axes and the color pipeline," step 4) assumes each R/G/B
  channel's normalized gamma curve applies independently, sharing one
  achromatic `[lum_min, lum_max]` measured for the whole display -- the
  standard assumption for a computer display, but not exact for a real
  panel with any cross-channel interaction.
- **Fixed-duration presentation, not "until response"**: this
  implementation always shows the stimulus for the full configured
  duration, then collects the response from a blank screen -- unlike CCT's
  literal "2-3 s or until response" (which removes the stimulus as soon as
  either condition is met first). This was chosen for implementation
  simplicity and consistency with this project's other tests (see
  `vpsych.tests_catalog._example`), and is not expected to materially
  change a coarse 4AFC orientation judgment's accuracy, but it does mean
  response time is measured from a slightly later reference point (the
  fixed stimulus-offset flip, not a per-frame response check) than the
  literal CCT protocol would give.
- **Per-trial (not per-frame) dithering**: bit-stealing dithering is applied
  once per disc per trial, not re-dithered on every frame of the (multi-
  frame) stimulus presentation -- a real per-frame temporal dither would
  improve the *effective* sub-pixel chromatic resolution further (see
  `vpsych.core.calibration.dither.effective_contrast_resolution`), at the
  cost of a materially more complex per-frame rendering path. This is a
  deliberate implementation simplification, not expected to matter at this
  test's `>= 5` displacement floor.

## Implementation notes

`TrivectorProcedure` (three interleaved `QuestPlusProcedure` instances) is
implemented entirely inside this test package
(`vpsych.tests_catalog.color_discrimination.procedure`), not in
`vpsych.core.procedures`, because neither a plain `AdaptiveProcedure` nor
the existing `QCSF` `MultiParamProcedure` supports interleaving three
independent sub-procedures -- `docs/WRITING_A_TEST.md` explicitly
anticipates this ("Check whether the single-procedure TrialLoop/test
interface supports interleaving. If not, implement a small composite
procedure INSIDE your package"), and no change to shared `core/`
code was needed or made.

Similarly, `vpsych.runner.__main__`'s `--simulate`/`--simulate-config`
machinery only dispatches two built-in observer *kinds* (`"psychometric"`,
a single ground-truth `PsychometricFunction`, and `"csf"`) -- neither can
express three independent per-axis ground-truths keyed by
`stimulus["axis"]`, and the `kind` dispatch in
`vpsych.runner.__main__.build_simulated_observer` is a closed `if`/`elif`
that cannot be extended from outside that module. This package's own
`TrivectorObserver` (`vpsych.tests_catalog.color_discrimination.observer`)
is therefore **not resolvable via `--simulate-config`'s CLI/JSON path at
all** -- it must be constructed directly and passed to
`run_session(..., simulated_observer=...)`, exactly how this test's own
simulated end-to-end validation test uses it. This is a real limitation of
the runner's current simulated-observer configuration surface for any
future multi-dimensional/multi-axis test, reported here rather than worked
around by changing shared runner code.
