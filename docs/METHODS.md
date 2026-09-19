# Methods

One section per test in the battery, documenting the psychophysical method
and citing the literature it is based on. Filled in as each test is
implemented (see the plan's Phase 2).

## Adaptive procedures and psychometric fitting

This section documents `vpsych.core.psychometric`, `vpsych.core.observers`,
and the four procedures in `vpsych.core.procedures` (`questplus_procedure`,
`staircase`, `constant_stimuli`, `qcsf`). Every test in the battery drives
its trial-by-trial intensity selection through one of these, and estimates
its threshold with the fitting machinery below.

### Psychometric function families

A psychometric function maps stimulus intensity `x` to probability correct:

```
p(x) = guess + (1 - guess - lapse) * F((x - threshold) / slope)
```

`guess` is the chance/floor rate fixed by task design (e.g. `0.5` for 2AFC,
`1/n` for n-AFC); `lapse` is the upper-asymptote miss rate. `F` is one of
three base sigmoid families (`vpsych.core.psychometric.PsychometricFamily`),
each parameterized so that **`F(0) == 0.5` by construction**:

- `logistic`: the standard logistic CDF, `1 / (1 + exp(-z))`.
- `norm_cdf`: the standard normal CDF, `Phi(z)`.
- `weibull`: `1 - 2**(-2**z)`, a Gumbel-type extreme-value sigmoid -- the
  classic Weibull-based psychometric function used since Watson & Pelli
  (1983, the original QUEST paper) when the intensity axis is already
  log-transformed. Note this is *not* the raw Weibull CDF (whose natural
  "threshold" is the 63.2% point); it is rescaled so `F(0) = 0.5`.

Because every family satisfies `F(0) = 0.5`, **`threshold` is always the
intensity at which the base sigmoid crosses 0.5** -- not necessarily the
task's conventional criterion (e.g. 75% correct in 2AFC), which generally
sits at a different point once `guess`/`lapse` are folded in. To convert to
an intensity at an arbitrary target proportion correct, use
`vpsych.core.psychometric.intensity_at_p_correct(function, p_target)`, which
inverts the same relationship. `WeightedStaircase` and `ConstantStimuli`
both use this helper internally to report thresholds at their configured
target percentage rather than always at the F=0.5 point.

**Intensity scale**: `PsychometricFunction.intensity_scale` is `"log10"` or
`"linear"`, and must match the scale of every intensity value passed to
`fit_mle`/`p_correct`. In practice: contrast, coherence, and other
naturally-multiplicative dimensions are fit on `"log10"` (matching how
`weibull` is normally used); logMAR and other quantities that are already a
log measure are fit on `"linear"`.

### Maximum-likelihood fitting (`fit_mle`)

Fits `threshold`, `slope`, and (optionally) `lapse` to binomial trial data
`(intensities, n_correct, n_total)` aggregated by intensity level, by
minimizing the binomial negative log-likelihood with
`scipy.optimize.minimize` (L-BFGS-B, bounded, with an analytic gradient --
see `_negloglik_grad` -- for speed). `guess` is always fixed by task design,
never estimated.

- **Lapse rate**: if `fix_lapse` is given, it is held fixed (common practice
  when trial counts are too small to estimate it reliably -- use
  catch-trial data instead, e.g. `TestSummary.catch_lapse_rate`). If left
  free, it is bounded to **[0, 0.06]** (Wichmann, F. A., & Hill, N. J.
  (2001). The psychometric function: I. Fitting, sampling, and goodness of
  fit. *Perception & Psychophysics*, 63(8), 1293-1313 -- they recommend a
  small bound rather than a fully free lapse rate, since with realistic
  trial counts it otherwise trades off against slope and is poorly
  identified).
- **Multiple starts**: to avoid local minima, `fit_mle` optimizes from a
  small grid of starting points (3 threshold values spanning the data range,
  2 slope values, and 1-2 lapse values) and keeps the best (highest
  likelihood) result. `bootstrap_ci`/`deviance_gof` instead use a *single*
  start (from the original fit's parameters) for their many resample refits,
  since resampled data stays close to the original MLE -- this is what keeps
  them fast enough for thousands of refits per call.
- **Bounds**: `threshold` is bounded to the tested intensity range extended
  by 2x its span on each side; `slope` to
  `[span * 1e-3, span * 10]` (with a small-span fallback for degenerate
  data).

### Parametric bootstrap confidence intervals (`bootstrap_ci`)

Given a `FitResult` (which retains the design intensities/trial counts it
was fit from), `bootstrap_ci` resamples `n_correct ~ Binomial(n_total,
fitted p(x))` at each design intensity, refits by MLE (single-start, see
above), and takes the empirical percentile interval of the resampled
`threshold` (or `slope`/`lapse`) values. `rng` must be a seeded
`numpy.random.Generator` (see `vpsych.core.rng.make_rng`) for reproducible
CIs; if omitted, an unseeded generator is used and results are not
reproducible. Only the percentile method is implemented; a BCa correction is
a documented possible future extension, not required here.

### Deviance goodness of fit (`deviance_gof`)

Computes the deviance statistic (2x the log-likelihood-ratio between the
fitted model and the saturated model that fits each intensity level's
observed proportion exactly), and a **Monte-Carlo p-value**: `n_mc` datasets
are simulated from the *fitted* model at the original design points, each is
refit, and the p-value is the fraction of simulated deviances (each against
its own refit) at least as large as the observed deviance (Wichmann & Hill,
2001, as above -- they recommend a Monte-Carlo reference distribution over
the nominal chi-squared(df) one, whose asymptotics are often poor at typical
psychophysics trial counts). `df` is still reported (levels minus free
parameters) as the nominal chi-squared reference, informationally.

### QUEST+ (`QuestPlusProcedure`)

Wraps `questplus.QuestPlus` (Watson, A. B. (2017). QUEST+: A general
multidimensional Bayesian adaptive psychometric method. *Journal of Vision*,
17(3):10) for single-parameter (threshold) adaptive estimation: a full
posterior over `(threshold, slope, lapse_rate)` on a user-supplied grid is
maintained and updated after every trial, and each next stimulus is chosen
to minimize expected posterior entropy (`stim_selection_method="min_entropy"`
by default). `questplus`'s built-in psychometric functions are `weibull`,
`csf`, `norm_cdf`, `norm_cdf_2`, and `thurstone_scaling` -- no logistic --
so `function="logistic"` is rejected with a clear error; use
`ConstantStimuli`/`WeightedStaircase` for a logistic fit instead. The guess
rate is fixed by passing it as a singleton `lower_asymptote` parameter-grid
entry. `estimate()` reports the posterior mean (or mode, via
`param_estimation_method`) threshold, with an **equal-tailed credible
interval taken from quantiles of the marginal threshold posterior**
(`_equal_tailed_credible_interval`: the `alpha/2`/`1-alpha/2` quantiles of
the posterior's cumulative distribution over the threshold grid, linearly
interpolated between grid points). This replaced an earlier
normal-approximation interval (`mean +/- z * SD`), which is a poor fit
whenever the posterior is skewed or truncated by the grid's edges -- both
common at realistic trial counts -- and measured coverage below nominal;
the quantile-based interval is valid for any posterior shape.
`state_dict()` is compact: posterior mean/SD per free parameter and the
trial count, not the full posterior array.

### Criterion conversion pitfall

**This bit hard, project-wide, and is worth stating explicitly.**
`estimate().value` for a `QuestPlusProcedure` is **not** on the same
footing as `threshold` for a `WeightedStaircase`/`ConstantStimuli`/
`fit_mle` fit, even though both are called "threshold" and both drive a
`"weibull"` psychometric function. The two are genuinely different
functional forms:

- `vpsych.core.psychometric`'s own `"weibull"` family (see "Psychometric
  function families" above) is `1 - 2**(-2**z)`, rescaled so
  **`F(0) == 0.5` by construction** -- `threshold` is always the 50%-of-
  the-way point between `guess` and `1 - lapse`.
- `questplus`'s own built-in `"weibull"` function (what `QuestPlusProcedure`
  actually fits, via the `questplus` package -- see Watson & Pelli 1983's
  original QUEST parameterization) is
  `p(x) = 1 - lapse - (1 - guess - lapse) * exp(-10**(slope*(x-threshold)))`.
  At `x = threshold`, the exponent is `10**0 = 1`, so
  `p(threshold) = guess + (1 - guess - lapse) * (1 - e^-1)` -- the
  **~63.2%-of-the-way point**, not 50%.

Passing a `QuestPlusProcedure` estimate's `threshold`/`slope`/`lapse` into
`vpsych.core.psychometric.intensity_at_p_correct` -- which assumes the
*other* family -- silently computes the wrong intensity for any
`p_target` other than `questplus`'s own ~63.2% anchor. This was a real,
shipped bug (Phase 4 fix list item 1): `motion_coherence` and
`critical_flicker_fusion` both built a `PsychometricFunction` directly from
a raw `QuestPlusProcedure` estimate and inverted it with
`intensity_at_p_correct`; `color_discrimination` and
`letter_contrast_sensitivity` both *assumed* (without conversion) that the
raw estimate was already at the `F(0)=0.5` point, which it is not. All four
were measurably biased as a result -- see each test's own
`docs/methods/*.md` "Validation" section for the before/after numbers.

**The fix**: `vpsych.core.procedures.questplus_procedure` provides the one
shared, tested inversion of `questplus`'s *own* formula --
`questplus_weibull_x_at_p(threshold, slope, guess, lapse, p_target)` (a
module-level function) and `QuestPlusProcedure.intensity_at_p_correct
(p_target)` (an instance method, reads `estimate()`'s own fitted
slope/lapse) -- documented in that module's own "Criterion conversion
pitfall" docstring section. Every `QuestPlusProcedure`-driven test in this
battery now uses one of these, never
`vpsych.core.psychometric.intensity_at_p_correct`, to report a threshold at
any criterion other than `questplus`'s own native anchor.

**Rule of thumb**: if code has both a `QuestPlusProcedure`
(`vpsych.core.procedures.questplus_procedure`) and a call to
`vpsych.core.psychometric.intensity_at_p_correct` nearby, stop and check
which family the `PsychometricFunction` being passed to it actually came
from. If it was built from a `QuestPlusProcedure.estimate()` result, that
call is almost certainly this bug.

### Weighted up/down and transformed up/down staircases (`WeightedStaircase`)

Implements two classic non-Bayesian adaptive rules, sharing the same
reversal-tracking and estimation machinery:

- **Weighted up/down** (Kaernbach, C. (1991). Simple adaptive testing with
  the weighted up-down method. *Perception & Psychophysics*, 49(3),
  227-229): every trial steps the intensity down (harder) by `step_down`
  after a correct response, or up (easier) by `step_up` after an incorrect
  one. The `step_up / step_down` ratio sets the staircase's target
  proportion correct: for target `p`, `step_down / step_up == p / (1-p)`.
- **Transformed up/down** (Levitt, E. (1971). Transformed up-down methods in
  psychoacoustics. *The Journal of the Acoustical Society of America*,
  49(2), 467-477), selected with `rule="transformed"`: `n_down` consecutive
  correct responses are required before a step down; any single incorrect
  response steps up and resets the consecutive-correct count. This targets
  `p = 0.5 ** (1 / n_down)` (`transformed_rule_target_p`), e.g. ~70.7% for
  2-down/1-up.

**Step-size reduction**: `step_size_reduction_after` halves both step sizes
each time the reversal count passes another multiple of that value (coarse,
then fine).

**Estimate**: `estimate()`'s primary point estimate and CI come from an
**MLE psychometric-function fit to every trial the staircase has seen**
(`psychometric.fit_mle`, on trials aggregated to a rounded-intensity grid
for bootstrap speed -- see `_aggregate_trials`), read off at the
staircase's `target_p_correct` (`intensity_at_p_correct`), with the CI from
`psychometric.bootstrap_ci_at_p_correct` -- a **full-refit** parametric
bootstrap that refits `threshold`/`slope`/`lapse` jointly on every resample
and evaluates `intensity_at_p_correct` against each resample's own fitted
function (not a `threshold`-only bootstrap interval converted while holding
`slope`/`lapse` fixed, which an earlier version used -- see "Phase 4 fix"
below). The classic reversal mean (mean of the intensities at the last
`n_reversals_for_estimate` reversals) is still reported, in
`extra["reversal_mean"]`, but **without a CI**: an earlier Student-t
interval on those reversal values (`mean +/- t_(n-1, 1-alpha/2) * SEM`)
treated them as i.i.d., which they are not (each reversal is shaped by the
run of trials since the previous one), and measured coverage was ~20-30%
for a nominal 95% interval -- a mislabeled CI, not reported any more.

**Phase 4 fix: the CI is a large, real improvement, still short of fully
calibrated.** The MLE-fit point estimate was already correct; its CI
originally held slope/lapse fixed at their point estimates while
converting a `threshold`-only bootstrap interval to the target-percent
intensity (like `ConstantStimuli`'s CI conversion) -- an approximation
that's only good when slope uncertainty is modest, which a staircase's
narrow dynamic range violates (it concentrates trials tightly around
threshold *by design*, unlike `ConstantStimuli`'s deliberately wide-spread
levels, so `slope` is genuinely poorly identified from the data). That
version's slow validation (200 reps x 3 true thresholds) measured coverage
around **0.72-0.77** for a nominal 95%. `estimate()` now uses the
full-refit `bootstrap_ci_at_p_correct` described above -- what this section
previously flagged as a possible future improvement ("would require a
bespoke bootstrap loop refitting per resample") -- and re-measurement (same
200-rep/3-threshold protocol) found coverage improved to roughly
**0.85-0.92**, with bias still comfortably under 0.05 log10 units. This is
a substantial, real improvement, not a full fix to the 0.85-0.99 target
other procedures reach with the same underlying machinery -- the residual
gap is the same genuine "slope poorly identified" cause, not a remaining
bug, and is documented plainly rather than hidden behind a widened test
tolerance. Prefer `QuestPlusProcedure` or `ConstantStimuli` where the
tightest possible CI calibration matters.

### Method of constant stimuli (`ConstantStimuli`)

Presents a fixed set of intensity levels, each repeated `n_reps_per_level`
times, in an order randomized once (via the supplied
`numpy.random.Generator`) at construction. `estimate()` fits a psychometric
function to the resulting per-level proportions correct (`fit_mle`), reads
the threshold off at `target_p_correct` (`intensity_at_p_correct`), and
computes its CI by running `bootstrap_ci` on the fitted `threshold`
parameter and converting each bound the same way (holding slope/lapse fixed
at their point estimates -- an approximation that is exact when
`target_p_correct` equals the F=0.5 point and a good approximation
otherwise, for the typically modest slope uncertainty at constant-stimuli
trial counts). `extra` also carries the deviance goodness-of-fit statistic
when at least 3 levels have data.

### qCSF (`QCSF`)

Implements the quick CSF method (Lesmes, L. A., Lu, Z.-L., Baek, J., &
Albright, T. D. (2010). Bayesian adaptive estimation of the contrast
sensitivity function: The quick CSF method. *Journal of Vision*, 10(3):17):
a Bayesian grid posterior over the 4 parameters of a **truncated
log-parabola CSF model**, jointly estimated from responses over a 2D
stimulus space (spatial frequency x contrast).

**CSF model** (`vpsych.core.procedures.qcsf.log_contrast_sensitivity`),
directly in log10-sensitivity / log2-frequency units:

```
log10 CS_unclipped(f) = peak_gain - 4*log10(2) * ((log2(f) - log2(peak_freq)) / bandwidth)^2
log10 CS(f) = max(log10 CS_unclipped(f), peak_gain - low_freq_truncation)   for f <= peak_freq
            = log10 CS_unclipped(f)                                        for f >  peak_freq
```

`bandwidth` is the full width at half maximum in octaves; `low_freq_truncation`
(log10 units, >= 0) lets the low-frequency "shoulder" plateau below the peak
instead of continuing to fall with the parabola (`0` degenerates to a fully
symmetric log-parabola).

**Psychometric function**: at a given spatial frequency, `p(correct |
f, contrast)` follows the same `F(0)=0.5` Weibull-family sigmoid as
`psychometric.py`, in log10 contrast around the CSF-predicted threshold
contrast `1 / CS(f)`. Its slope and lapse rate are **fixed, not estimated**
(standard qCSF practice): `psychometric_slope` defaults to **0.35** (log10
contrast units) and `lapse_rate` defaults to **0.02**. These are
implementation choices documented here (Lesmes et al. 2010 use a different
parameterization, so their reported slope value does not translate
directly), not values taken verbatim from the source paper.

**Implementation**: rather than the `questplus` package (whose built-in
`func="csf"` is a different, linear-in-frequency threshold model, not this
truncated log-parabola), `QCSF` precomputes the full stimulus x
parameter-grid likelihood table once at construction and reuses it for
one-step-ahead expected-entropy stimulus selection every trial, fully
vectorized with NumPy (a "precomputed-likelihood approach", the speed
strategy this project's Phase 1a scope explicitly allows). The
per-trial entropy calculation is further reduced to matrix-vector products
against that precomputed table (no per-trial `log()` over the full grid) --
see `QCSF.next_stimulus`'s docstring for the closed-form derivation.
`QCSF.default_grids()` returns the recommended grids (10 spatial
frequencies x 10 contrasts = 100 stimuli; 10x10x8x6 = 4800 parameter
combinations), measured well under 100 ms per trial (typically <1 ms) on a
development laptop -- comfortably inside the Phase 1a budget.

**`estimate()`** returns **AULCSF** (area under the log CSF): `trapz(log10
CS(f), x=log10(f))` integrated over `f` in **[1, 18] cpd** (50 log-spaced
points), in `log10(CS) * log10(cpd)` units (not normalized by the
integration range). Because the parameter grid is small, AULCSF's posterior
mean and credible interval are computed *exactly to grid resolution* --
evaluated at every grid point and weighted by the posterior -- rather than
via additional Monte Carlo sampling. `extra` carries the posterior mean/SD
of all 4 CSF parameters and the posterior-mean log10 CS at the standard
frequency set **[1, 1.5, 3, 6, 12, 18] cpd**. `state_dict()` is compact
(mean/SD per parameter and trial count), not the full ~4800-point posterior.

**AULCSF point-estimate bias and recommended trial count**: the plug-in
AULCSF estimate (evaluated at the posterior-mean parameters, not the
posterior mean of AULCSF itself -- see `estimate()`'s docstring for why)
is systematically biased low at modest trial counts, consistent with
Lesmes et al. (2010), who likewise report a small residual bias after
about 100 trials. Measured with `default_grids()` against a true CSF
(peak gain 1.6, peak freq 3 cpd, bandwidth 3 oct, truncation 1.0):

| Trials | Bias (log units) | n (simulated runs) |
|---|---|---|
| 100 | -0.057 (+/- 0.028 SEM) | 80 |
| 150 | -0.032 to -0.054 across seeds | 150-250 |
| 200 | -0.035 to -0.066 across seeds | 250 |
| 300 | **-0.039 (+/- 0.007 SEM)** | 450 (pooled, 3 seeds) |
| 500 | -0.033 (+/- 0.007 SEM) | 450 (pooled, 3 seeds) |

Single-run-to-single-run variability is high (SD ~0.15-0.25 log units)
regardless of trial count, so small-sample sweeps (80-250 runs) at 150-200
trials gave noisy, seed-dependent point estimates that sometimes exceeded
0.05 by chance even though the underlying bias is smaller; a well-powered
pooled sweep (450 runs) at 300 and 500 trials resolved this and confirms
the bias is comfortably under 0.05 *on average* at those trial counts.

Grid resolution was investigated as an alternative cause (finer contrast,
spatial-frequency, or CSF-parameter grids, up to ~18-24 points per
dimension and ~90000 parameter combinations) and did not reliably reduce
bias beyond sampling noise, while increasing `next_stimulus()` time (up to
~48 ms at the finest grid tested, still under the 100 ms budget but with
no benefit) -- so this is not primarily a grid-coarseness effect. **The
recommended default trial count is 300** (`QCSF.RECOMMENDED_MIN_TRIALS`;
not `QCSF`'s own default, since it has none -- a real test implementation
should pass `max_trials=300` or more), where average bias is about -0.04,
comfortably under the |bias| < 0.05 target; 100 trials remains usable but
carries a larger (~-0.06) bias worth disclosing in a test's own
summary/quality flags if a shorter run is required.
`tests/procedures/test_qcsf.py::test_recovery_slow` uses `RECOMMENDED_MIN_TRIALS`
trials but asserts a looser |bias| < 0.08 at its own, smaller n_reps=150 to
avoid flaking on the sampling noise described above -- the 0.05 target is
the honestly-reported average from the larger pooled sweep, not the
automated test's own bound.

### Simulated observers (`vpsych.core.observers`)

`PsychometricObserver` responds according to a known `PsychometricFunction`
(`p_correct(stimulus["intensity"])`); `CSFObserver` responds according to a
known ground-truth CSF (the same `log_contrast_sensitivity` model `QCSF`
fits) and a fixed psychometric slope/lapse. Both draw exactly one
`rng.random()` call to decide correct/incorrect and, only on an incorrect
trial, one `rng.integers()` call to pick uniformly among the other `n_afc -
1` response alternatives (`stimulus["alternatives"]`, defaulting to
`range(n_afc)`; the correct label is `stimulus["correct_alternative"]`,
defaulting to `0`) -- so response sequences are fully reproducible from a
seeded `numpy.random.Generator` and no other source of randomness is used.

### Validation

`tests/core/test_psychometric.py`, `tests/core/test_observers.py`, and
`tests/procedures/test_*.py` validate all of the above against simulated
observers at multiple true thresholds (or, for qCSF, multiple true CSF
parameters): unbiased recovery (|bias| < 0.05 log10 units) and CI coverage
checked against nominal, each with both a fast always-on smoke variant and a
`@pytest.mark.slow` variant using >=200 repetitions (excluded from the
default `pytest` run via `-m "not slow"`; run explicitly with `pytest -m
slow`). Determinism (identical trial sequences from identical seeds) and
JSON-serializability of every `state_dict()`/estimate model are also
checked for every procedure.

## Visual acuity

See `docs/methods/visual_acuity.md` for the full method write-up (FrACT-style
Landolt C acuity via QUEST+, ISO 8596 optotype geometry, display-resolution
clipping, and the "Criterion conversion pitfall" section above's worked
FrACT-criterion application, plus validation bias/coverage numbers).

## Contrast sensitivity function (qCSF)

See `docs/methods/contrast_sensitivity_function.md` for the full method
write-up (the qCSF Bayesian adaptive method, spatial-frequency grid
derivation, gamma linearization/dithering, and validation bias numbers).

## Letter contrast sensitivity

See `docs/methods/letter_contrast_sensitivity.md` for the full method
write-up (Pelli-Robson-style 10AFC Sloan-letter contrast sensitivity via
QUEST+, procedural letter rendering, and the "Output and threshold
criterion" section's worked application of the "Criterion conversion
pitfall" above).

## Color discrimination

See `docs/methods/color_discrimination.md` for the full method write-up
(Cambridge Colour Test-style trivector chromatic discrimination, three
interleaved QUEST+ procedures, block-randomized axis allocation, and the
"Interleaving and threshold criterion" section's worked application of the
"Criterion conversion pitfall" above).

## Motion coherence

See `docs/methods/motion_coherence.md` for the full method write-up
(random-dot kinematogram direction-discrimination threshold, noise-dot
algorithm, `d_max` correspondence limit, and the "Validation" section's
honest before/after bias-coverage numbers from the Phase 4 criterion-
conversion fix).

## Vernier acuity

See `docs/methods/vernier_acuity.md` for the full method write-up (2AFC
lateral-offset hyperacuity via QUEST+, sub-pixel-accurate analytically
area-sampled rendering, and the self-linearizing gamma correction
`vernier_acuity.texture` now uses -- see that module's docstring for the
Phase 4 gamma-ramp decision).

## Critical flicker fusion

See `docs/methods/critical_flicker_fusion.md` for the full method write-up
(frame-sampled flicker-detection threshold in Hz, the display's "usable
ceiling" and display-limited lower-bound reporting, and the "Validation"
section's honest before/after bias-coverage numbers from the Phase 4
criterion-conversion fix).
