# Vernier acuity (hyperacuity)

Vernier acuity measures the smallest lateral misalignment between two
abutting line segments an observer can detect -- the classic demonstration
of *hyperacuity*: normal observers reliably detect offsets of only a few
arcseconds, an order of magnitude finer than the spacing of foveal cone
photoreceptors that limits ordinary (Landolt C/letter) acuity (Westheimer,
G. (1975). Editorial: Visual acuity and hyperacuity. *Investigative
Ophthalmology*, 14(8), 570-572). This test measures the lateral-offset
detection threshold for the classic two-line Vernier configuration.

## Method

**Adaptive procedure**: QUEST+ (Watson, A. B. (2017). QUEST+: A general
multidimensional Bayesian adaptive psychometric method. *Journal of
Vision*, 17(3):10, https://doi.org/10.1167/17.3.10) on a single scalar
intensity dimension, `log10(offset in arcsec)`.

**Psychometric function**: `questplus`'s own log10-scale Weibull (see
`visual_acuity`'s methods doc for the full derivation of why this is
appropriate whenever the driving intensity is already log10 of the
physical quantity -- true here, since the intensity *is*
`log10(offset_arcsec)`). Guess rate fixed at 0.5 (2AFC: left or right).
Lapse rate is a free parameter on the grid `[0.0, 0.02, 0.04]`.

**Number of alternatives**: 2AFC -- left or right arrow key.

**Threshold criterion**: the offset at 75% correct, inverted against
`questplus`'s own Weibull formula via the shared, tested
`QuestPlusProcedure.intensity_at_p_correct`
(`vpsych.core.procedures.questplus_procedure.questplus_weibull_x_at_p`) --
see `docs/METHODS.md`'s "Criterion conversion pitfall" section for why
this must invert `questplus`'s own formula directly and not
`vpsych.core.psychometric.intensity_at_p_correct`. `visual_acuity` shares
this same implementation (previously each test duplicated a private local
copy; both now use the one promoted into `core/`). The credible interval
is shifted by the same constant offset as the point estimate, holding
slope/lapse fixed (the same approximation used throughout this project's
adaptive procedures; see `docs/METHODS.md`).

## Stimulus

**Configuration**: two vertical line segments, one above the other,
separated by a small vertical gap; the lower segment is offset left or
right of the upper one by the trial's intensity. Default dimensions,
within the ranges classic Vernier studies use (Westheimer, G. (1979). The
spatial sense of the eye. *Investigative Ophthalmology & Visual Science*,
18(9), 893-912; McKee, S. P., & Westheimer, G. (1978). Improvement in
vernier acuity with practice. *Perception & Psychophysics*, 24(3),
258-262, https://doi.org/10.3758/BF03204247):

- Line length: 15 arcmin (configurable range 10-20 arcmin).
- Line width: 1 arcmin.
- Gap between segments: 3 arcmin (configurable range 2-4 arcmin).

**Position jitter**: the whole two-segment stimulus is displaced by a
random amount each trial (default up to +/-4 arcmin, `position_jitter_arcmin`,
logged per trial in `stimulus_params["position_jitter_px"]`), so absolute
screen position can't be used as a response cue. Jitter is **rounded to
whole pixels** before being applied to the `ImageStim`'s on-screen
position -- deliberately, since any *fractional*-pixel jitter would be
resampled by the GPU's own texture interpolation and could corrupt the
analytically-encoded sub-pixel offset described below. The stimulus's
overall canvas size (and therefore its apparent size) is held fixed across
the *entire* intensity domain, not just per trial, so it can't be used as
a size-based cue for how large the current offset is.

**Sub-pixel rendering**: because Vernier thresholds are routinely a small
fraction of a display pixel at normal viewing distances, the offset is
encoded directly in the stimulus texture's edge luminance rather than via
a stimulus object's floating-point on-screen position (relying on a
backend's own bilinear texture sampling to reproduce an exact sub-pixel
edge is not something this project depends on). `texture.py` implements
this with **exact analytic area sampling** (not stochastic supersampling,
unlike `visual_acuity`'s optotype texture, which does not need this level
of precision): for a vertical bar of a given width centered at a
(generally fractional) pixel position, each output column's coverage is
the exact overlap length between the bar's span and that column's
continuous pixel interval, computed with `np.clip`/`np.minimum`/`np.maximum`
on continuous interval endpoints -- correct to floating-point precision for
any sub-pixel center, with zero discretization error. This is unit-tested
headless (`texture_centroid_x_px`, inverting the luminance mixture back to
a centroid position): the recovered centroid matches the requested
sub-pixel position to well under 0.02 px (in practice, to floating-point
precision) across a range of test positions spanning whole- and
fractional-pixel offsets (see `test_vernier_acuity.py`).

**Gamma linearization**: each texture pixel's coverage-weighted **linear**
luminance mixture of the background and foreground (line) levels -- the
luminance a photoreceptor pooling light linearly over that pixel's area
would actually integrate (the same principle behind this project's grade-B
psychophysical gamma calibration; see `docs/CALIBRATION.md`'s
half-luminance bisection method) -- is computed first, then this test
itself converts that to the correct gamma-corrected hardware drive level,
via `render_vertical_line_texture`'s `gamma_model` argument
(`vpsych.core.calibration.gamma.linearize`, using a
`GammaChannelModel` built from the session's real calibration via
`gamma_channel_model_from_calibration`). This is the same self-linearizing
pattern `contrast_sensitivity_function`/`letter_contrast_sensitivity`/
`color_discrimination` use, **not** a window-level gamma ramp -- see
`docs/WRITING_A_TEST.md` section 7's "Phase 4 gamma-ramp decision" for why:
`PsychoPyBackend` intentionally never sets one, so an earlier version of
this test (which assumed a ramp would linearize its raw linear-luminance
texture) was silently wrong, while the other three tests (which already
self-linearized, expecting no ramp) were accidentally right. This is why
this test still requires a real gamma calibration (grade B minimum) -- the
calibration is used here, in Python, rather than via a ramp. This general
approach -- antialiasing quality (not just raw pixel pitch) setting the
achievable Vernier threshold on a fixed-resolution display -- is
consistent with empirical findings in Lloyd, C., Winterbottom, M., Gaska,
J., & Williams, L. (2015). Effects of display pixel pitch and antialiasing
on threshold vernier acuity. *Proceedings of the IMAGE Society Annual
Conference*, Dayton, OH.

**Timing**: 175 ms stimulus presentation by default (configurable 150-200
ms range, in frames), chosen to limit involuntary eye movements while
remaining within the temporal integration window classic Vernier studies
used -- Westheimer, G., & McKee, S. P. (1977). Integration regions for
visual hyperacuity. *Vision Research*, 17(1), 89-93,
https://doi.org/10.1016/0042-6989(77)90209-9, report thresholds improving
with duration up to about 100 ms and then plateauing, so 150-200 ms is
comfortably past that plateau while still short enough to discourage a
voluntary saccade toward the stimulus. A blank response window follows
(default cap 5 s), then fixation (500 ms) and a blank ITI (500 ms).

## Output

`output_units = "arcsec"`. `TestSummary.estimate.value` is the offset at
75% correct, in arcsec, with a credible interval in the same units.
`TestSummary.fit_params` additionally reports `slope`, `lapse_rate`,
`guess_rate`, `target_p_correct` (fixed at 0.75), and
`offset_px_at_threshold` (the threshold converted back to pixels at the
session's display/viewing distance, for interpreting the two quality flags
below).

## Requirements

`needs_gamma_calibration = True`, `min_luminance_grade = "B"` -- see
"Gamma linearization" above for why. No static `min_viewing_distance_cm`
requirement is declared (the useful domain depends on the display's pixel
pitch and viewing distance jointly, as for `visual_acuity`); instead,
`summarize()` raises a **dynamic** `short_viewing_distance` quality flag
if the session's viewing distance is under 100 cm, and a
`near_rendering_limit` flag if the measured threshold's pixel-equivalent
(`offset_px_at_threshold`) falls below 0.1 px -- close to the practical
precision floor even with exact analytic antialiasing, since 8-bit output
quantization and residual gamma-ramp imprecision both become significant
relative to such a small fraction of a pixel.

## Limitations

- The reported offset-at-75%-correct threshold (and its CI) holds
  slope/lapse fixed at their point estimates when shifting from
  `QuestPlusProcedure`'s own native threshold -- an approximation shared
  with `visual_acuity` and every adaptive procedure in this project that
  reports at a non-native criterion.
- Sub-pixel encoding is only as good as the display's actual gamma
  linearization in practice; a grade-B (psychophysical) calibration has no
  absolute luminance scale and a noisier gamma estimate than grade A
  (photometer), so the practically achievable precision with grade B is
  somewhat worse than the texture generator's own (effectively exact)
  encoding precision.
- Validation below (like `visual_acuity`'s) uses a *native* ground-truth
  observer defined directly in `questplus`'s own Weibull parameterization,
  to isolate this test's own domain/grid/criterion-conversion choices from
  any family mismatch with `vpsych.core.observers.PsychometricObserver`.

## Citations

- Westheimer, G. (1975). Editorial: Visual acuity and hyperacuity.
  *Investigative Ophthalmology*, 14(8), 570-572.
- Westheimer, G. (1979). The spatial sense of the eye. *Investigative
  Ophthalmology & Visual Science*, 18(9), 893-912.
- McKee, S. P., & Westheimer, G. (1978). Improvement in vernier acuity
  with practice. *Perception & Psychophysics*, 24(3), 258-262.
  https://doi.org/10.3758/BF03204247
- Westheimer, G., & McKee, S. P. (1977). Integration regions for visual
  hyperacuity. *Vision Research*, 17(1), 89-93.
  https://doi.org/10.1016/0042-6989(77)90209-9
- Lloyd, C., Winterbottom, M., Gaska, J., & Williams, L. (2015). Effects
  of display pixel pitch and antialiasing on threshold vernier acuity.
  *Proceedings of the IMAGE Society Annual Conference*, Dayton, OH.
- Watson, A. B. (2017). QUEST+: A general multidimensional Bayesian
  adaptive psychometric method. *Journal of Vision*, 17(3):10.
  https://doi.org/10.1167/17.3.10

## Validation

`tests/tests_catalog/test_vernier_acuity.py::test_recovery_slow_bias_and_coverage`
(`@pytest.mark.slow`, 20 repeats x 3 true `log10(offset_arcsec)` values,
60 trials each, native-family ground truth with slope=3.0, lapse=0.02,
2AFC) is the automated bound (`|mean bias| < 0.15` log10(arcsec)
units, coverage `>= 0.5`). A larger, 16-repeat sweep at the same trial
count and true offsets (run with `OPENBLAS_NUM_THREADS=1`) gave:

| True log10(offset, arcsec) | Mean bias (log10 arcsec) | SD | 95% CI coverage |
|---|---|---|---|
| 0.5 (~3.2 arcsec) | +0.008 | 0.054 | 1.00 |
| 1.0 (~10 arcsec) | +0.015 | 0.067 | 0.94 |
| 1.5 (~32 arcsec) | +0.004 | 0.079 | 0.69 |

Bias is very small (well under 0.02 log10 units, i.e. under ~5% in
arcsec) and coverage is good at 16 repeats (again with substantial
per-value sampling noise at this repeat count -- the lower coverage at
1.5 log10(arcsec) is consistent with edge effects near the domain's upper
bound, analogous to `visual_acuity`'s lower-edge effect). The unit-tested
sub-pixel texture centroid recovery (see "Sub-pixel rendering" above) is
accurate to floating-point precision, far tighter than the Phase 2A
task's own 0.02 px requirement -- the *procedure-level* bias/coverage
numbers above, not the rendering technique itself, are the binding source
of measurement uncertainty in a real session.
