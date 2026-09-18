# Visual acuity (Landolt C)

Visual acuity measures the finest spatial detail an observer can resolve --
here, the smallest gap orientation in a Landolt C ring the observer can
reliably identify. It is the most familiar of all vision measurements
(the clinical eye chart), and this test reproduces the FrACT (Freiburg
Visual Acuity Test) approach to measuring it automatically and adaptively
on a computer display rather than with a printed chart and a human examiner.

## Method

**Adaptive procedure**: QUEST+ (Watson, A. B. (2017). QUEST+: A general
multidimensional Bayesian adaptive psychometric method. *Journal of
Vision*, 17(3):10, https://doi.org/10.1167/17.3.10), driving a single
scalar intensity dimension, `logMAR = log10(gap size in arcmin)`. A full
Bayesian posterior over `(threshold, slope, lapse_rate)` is maintained on a
fixed parameter grid and updated after every trial; each next optotype
size is chosen to minimize expected posterior entropy.

**Psychometric function**: `questplus`'s own log10-scale Weibull,

```
p(x) = 1 - lapse - (1 - guess - lapse) * exp(-10**(slope * (x - threshold)))
```

with `x = logMAR`. Because `x` is passed already log10-transformed
(`logMAR` *is* `log10(gap_arcmin)`), this is algebraically identical to
the classical linear-domain acuity Weibull `p = 1 - lapse - (1 - guess -
lapse) * exp(-(gap/gap_threshold)**slope)` -- the standard model used
throughout the acuity literature -- while still letting QUEST+ drive trials
adaptively on the numerically convenient log axis (this is precisely
Watson & Pelli's 1983 original motivation for QUEST's log-intensity
design). `slope` here is therefore the classical Weibull "beta" shape
exponent (order ~3 for most detection/acuity tasks); this is a genuinely
different parameterization from `vpsych.core.psychometric`'s own weibull
family (rescaled so its `threshold` sits at the `F(0) = 0.5` point -- see
`docs/METHODS.md`'s "Psychometric function families" section), whose
natural threshold point is instead the classical Weibull's `1 - exp(-1) ~=
63.2%`-of-the-way point. This distinction matters for how this test
reports its threshold (see below); it is documented and derived in full in
`src/vpsych/tests_catalog/visual_acuity/__init__.py`'s module docstring
and `_questplus_weibull_x_at_p`.

Guess rate is fixed by task design: `1/8` for the default 8AFC
(orientation in 45-degree steps), or `1/4` for 4AFC (cardinal directions
only). Lapse rate is a free parameter on the grid `[0.0, 0.02, 0.04]`.

**Number of alternatives**: 8AFC by default (Bach's own FrACT default),
scored via numpad keys 1-9 (excluding 5), mapped to the 8 compass
directions the gap can open toward. A 4AFC mode (cardinal directions only)
additionally accepts the 4 arrow keys.

**Optotype geometry**: ISO 8596:2017 / EN ISO 8596 ("Ophthalmic optics --
Visual acuity testing -- Standard optotype and its presentation")
proportions: outer diameter = 5x the gap width, stroke width = the gap
width, inner diameter = 3x the gap width (`optotype.landolt_c_geometry`).

**Threshold criterion**: FrACT's own criterion (Bach, M. (1996). The
Freiburg Visual Acuity test -- automatic measurement of visual acuity.
*Optometry and Vision Science*, 73(1), 49-53,
https://doi.org/10.1097/00006324-199601000-00008, Methods) reports
threshold at the intensity where p-correct sits exactly midway between the
guess rate and 100% correct:

```
target_p = guess_rate + 0.5 * (1 - guess_rate - lapse_rate)
```

This is inverted against `questplus`'s own Weibull formula (not
`vpsych.core.psychometric.intensity_at_p_correct`, which assumes the
differently-parameterized rescaled family and would give a silently wrong
answer applied to `QuestPlusProcedure`'s own fitted parameters -- see
`_questplus_weibull_x_at_p`), using the posterior's own point estimates of
slope and lapse rate. The reported credible interval is the same
equal-tailed interval `QuestPlusProcedure.estimate()` computes on its own
native (~63.2%-point) threshold, shifted by the same constant offset as the
point estimate -- holding slope/lapse fixed at their point estimates, the
same approximation `ConstantStimuli`/`WeightedStaircase` use to convert a
bootstrap CI to an arbitrary target proportion correct (see
`docs/METHODS.md`'s "Adaptive procedures" section).

## Stimulus

**Rendering**: the Landolt C is rendered as a precomputed antialiased
texture (`optotype.render_landolt_c`), not a `ShapeStim` polygon: an exact
geometric membership test (inside the ring annulus, outside the gap wedge)
is evaluated on a 4x-supersampled grid (16x oversampling in area) and
block-averaged down to the output texture resolution, giving each texel a
continuous coverage fraction rather than a hard edge. The rendered gap
size in pixels (`display.deg_to_px(gap_arcmin / 60)`) is logged per trial
in `stimulus_params["gap_px"]`.

**Contrast**: high-contrast dark optotype on a bright background, Weber
contrast configurable (default -0.99, i.e. approximately -99%). Gamma
calibration is not required for this test -- see Requirements -- so this
contrast is an uncorrected linear approximation, not a photometrically
exact one; adequate given the contrast is deliberately far from any
near-threshold value.

**Domain**: the nominal logMAR domain is -0.5 to 1.3 by default, clipped at
construction time to what the configured display can actually render:

- **Lower bound (sharpest gap)**: bounded by `min_gap_px` (default 1.0
  physical pixel). Below about 1 px, the gap's antialiased edge coverage
  collapses toward a single texel's worth of gradient, so there is no
  longer meaningful sub-pixel *geometric feature* (a gap's presence or
  absence) left to render -- unlike `vernier_acuity`'s hyperacuity
  offsets, which encode a *position*, not a feature's presence, and so
  remain meaningful far below 1 px.
- **Upper bound (coarsest gap)**: bounded by what fits on screen (90% of
  the smaller display dimension).

**Minimum viewing distance for normal acuity**: a smaller physical pixel
subtends a smaller visual angle at a *longer* viewing distance. Using
`vpsych.core.display.DisplayGeometry`'s exact geometry for a representative
1920x1080, 53.13x29.88 cm display (a common ~24" monitor), the viewing
distance needed for a single physical pixel to subtend no more than the
gap size at -0.3 logMAR (0.50 arcmin, i.e. 20/10 Snellen-equivalent acuity)
works out to **~190 cm (~1.9 m)** -- computed by solving
`display.px_to_deg(1.0) * 60 == 10**(-0.3)` for `viewing_distance_cm`.
At a typical desk viewing distance (~57 cm), the same display can only
render gaps down to about **0.22 logMAR** (worked example: at 57 cm this
display's 1-pixel gap already subtends ~1.75 arcmin, i.e. logMAR ~0.24,
close to normal 20/20-ish acuity) -- a normal-or-better observer's true
threshold may be *unmeasurable*, not merely imprecise, at typical
close-range desk distances on a standard monitor. This is exactly why
`summarize()` implements the dynamic floor check below: a static
requirement can't capture this (it depends on both the specific display
and the configured viewing distance), but a per-session check can.

**Timing**: by default the optotype stays visible until response, capped
at 30 s (`max_response_ms`, FrACT-like); a `fixed_stimulus_duration_ms`
option (in frames) can instead show it for a fixed duration and then blank
the screen for the remainder of the response window. Fixation (default
500 ms) precedes the stimulus; a blank ITI (default 500 ms) follows the
response.

## Output

`output_units = "logMAR"`. `TestSummary.estimate.value` is the FrACT-
criterion logMAR threshold described above. `TestSummary.fit_params`
additionally reports:

- `decimal_acuity`: `10 ** (-logMAR)` (e.g. 1.0 = 20/20, 0.5 = 20/40).
- `snellen_20` / `snellen_6`: Snellen-equivalent acuity in both US (`20/x`)
  and metric (`6/x`) form.
- `slope`, `lapse_rate`, `guess_rate`, `target_p_correct`: the fitted
  psychometric parameters and the FrACT criterion actually used.

## Requirements

No calibration is required (`TestRequirements()`, no fields set) -- the
Weber contrast used is deliberately far from threshold, so an uncorrected
linear contrast approximation is adequate, and no absolute luminance scale
is needed. There is no static `min_viewing_distance_cm` requirement either,
since the achievable domain depends on the *combination* of the specific
display's pixel pitch and the configured viewing distance, not either
alone (see above). Instead, `summarize()` implements a **dynamic**
check: it computes the display's pixel-bounded floor
(`optotype.min_renderable_logmar`) at the session's actual geometry, and
raises a `display_resolution_limited` quality flag if the measured
threshold ends up within 0.1 logMAR of that floor, with the message
*"Threshold limited by display resolution: increase viewing distance."*

## Limitations

- The FrACT-criterion conversion (and its CI) holds slope/lapse fixed at
  their point estimates when shifting the credible interval -- an
  approximation, not a fully re-derived interval at the target criterion.
- The Weber contrast approximation is uncorrected for gamma, appropriate
  only because the target contrast is deliberately extreme (not a
  near-threshold measurement).
- Validation below uses a *native* ground-truth observer (defined directly
  in `questplus`'s own Weibull parameterization) to isolate this test's own
  domain/grid/criterion-conversion choices from any family mismatch with
  `vpsych.core.observers.PsychometricObserver` (which uses the
  differently-parameterized `vpsych.core.psychometric` family) -- the
  runner-level end-to-end test in `test_visual_acuity.py` additionally
  exercises the full pipeline through that (family-mismatched, deliberately
  loose-tolerance) observer, consistent with `_example`'s own convention.

## Citations

- Bach, M. (1996). The Freiburg Visual Acuity test -- automatic measurement
  of visual acuity. *Optometry and Vision Science*, 73(1), 49-53.
  https://doi.org/10.1097/00006324-199601000-00008
- Bach, M. (2007). The Freiburg Visual Acuity Test -- variability unchanged
  by post-hoc re-analysis. *Graefe's Archive for Clinical and Experimental
  Ophthalmology*, 245(7), 965-971. https://doi.org/10.1007/s00417-006-0474-4
- ISO 8596:2017. Ophthalmic optics -- Visual acuity testing -- Standard
  optotype and its presentation.
- Watson, A. B. (2017). QUEST+: A general multidimensional Bayesian
  adaptive psychometric method. *Journal of Vision*, 17(3):10.
  https://doi.org/10.1167/17.3.10
- Watson, A. B., & Pelli, D. G. (1983). QUEST: A Bayesian adaptive
  psychometric method. *Perception & Psychophysics*, 33(2), 113-120.
  https://doi.org/10.3758/BF03202828

## Validation

`tests/tests_catalog/test_visual_acuity.py::test_recovery_slow_bias_and_coverage`
(`@pytest.mark.slow`, 20 repeats x 3 true thresholds, 40 trials each, a
native-family ground-truth observer with slope=3.0, lapse=0.02, 8AFC) is
the automated bound (`|mean bias| < 0.25` logMAR, coverage `>= 0.5`,
deliberately loose for a fast CI-adjacent run, not a full scientific
sweep). A larger, 16-repeat sweep at the same trial count and true
thresholds (slope=3.0, lapse=0.02, 8AFC, run with `OPENBLAS_NUM_THREADS=1`)
gave:

| True logMAR (criterion pt.) | Mean bias (logMAR) | SD | 95% CI coverage |
|---|---|---|---|
| -0.1 | -0.022 | 0.066 | 0.69 |
| 0.3 | -0.013 | 0.042 | 0.94 |
| 0.7 | -0.001 | 0.041 | 0.94 |

Bias is small (a few hundredths of a logMAR unit) and consistently
negative-or-zero; this is well within the precision a ~40-trial clinical
acuity measurement is normally used at. Coverage of the equal-tailed
QUEST+ credible interval (after the criterion-shift approximation above)
is good (0.69-0.94 across the three true thresholds at only 16 repeats,
so individual coverage numbers carry substantial sampling noise) but not
perfectly calibrated -- consistent with QUEST+'s own documented coverage
behavior elsewhere in this project (see `docs/METHODS.md`'s "Adaptive
procedures" section). The lower coverage at `true = -0.1` (near the
domain's lower edge, where the psychometric function's rising limb is
partly cut off by the tested range) is consistent with the same
edge-truncation effect documented for QUEST+ generally.
