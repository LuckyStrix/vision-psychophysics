# Critical flicker fusion (CFF)

Critical flicker fusion is the highest temporal frequency at which a
flickering light is still reliably distinguished from a steady one of the
same mean luminance. It is one of the oldest quantitative measures in visual
psychophysics (Hecht & Shlaer, 1936) and follows the Ferry-Porter law: CFF
increases roughly linearly with log luminance (Porter, 1902). This
implementation is deliberately honest about the limits of a frame-based
display for measuring a genuinely high-frequency phenomenon -- see
"Display limitations" below.

## Method

**Procedure**: QUEST+ (Watson, 2017) drives the *transformed* intensity
`x = -log10(frequency_hz)`, not frequency directly, because CFF performance
*decreases* with frequency while QUEST+ (and every adaptive procedure in
this project) assumes probability-correct is increasing in intensity. Since
`x` is a decreasing function of `f`, and performance is also a decreasing
function of `f`, performance is an *increasing* function of `x` -- exactly
what QUEST+ needs. See `waveform.frequency_to_intensity` /
`intensity_to_frequency` for the transform and its round-trip. The domain
runs from `min_frequency_hz` (default 2 Hz, an easily-detected anchor) to the
display's "usable ceiling" (`refresh_hz / 3`, see below). Psychometric
function: fixed-family Weibull, guess rate 0.5 (2AFC), free lapse rate from
`[0.0, 0.02, 0.04]`. QUEST+'s `threshold_values` grid spans the tested
domain (21 points), but `slope_values` is **not** domain-derived -- it is
the fixed `DEFAULT_SLOPE_VALUES` constant (a generic typical-beta range for
2AFC near-threshold psychometric functions, the same convention
`visual_acuity`/`vernier_acuity` use), independent of `refresh_hz` or
`min_frequency_hz`. See "Validation" below for why a domain-derived slope
grid was a bug, not a design choice.

**Task**: 2AFC. Two identical discs appear left and right of fixation, same
mean luminance; one flickers, one stays steady; the observer presses the
left or right arrow key for the side that flickers. The flickering side is
randomized per trial via `trial_ctx["rng"]`.

**Threshold reporting**: `summarize()` converts the QUEST+ posterior's
F=0.5-crossing threshold (in `x`) to the 75%-correct point via
`intensity_at_p_correct`, then back to Hz via `intensity_to_frequency`.
`TestSummary.estimate.units == "hz"`. **If** that 75%-correct point falls
within `EDGE_TOLERANCE_FRAC` (15%) of the tested domain's hard
(highest-frequency) edge, `summarize()` reports a **lower bound** instead --
`estimate.value`/`ci_low`/`ci_high` are all pinned to `usable_ceiling_hz`,
`estimate.extra["display_limited"] = True`, and a **critical**
`cff_display_limited` quality flag is raised -- never a fabricated point
value beyond what the display could actually test. See "Display-limited
detection" below for why a *tolerance band*, not exact edge equality, is
used.

## Stimulus

| Parameter | Default | Unit | Notes / citation |
|---|---|---|---|
| `disc_diameter_deg` | 2.0 | deg | Each disc's diameter (Hecht & Shlaer 1936; Tyler & Hamer 1990-scale parafoveal test fields). |
| `eccentricity_deg` | 5.0 | deg | Distance from fixation to each disc's center. |
| `modulation_depth` | 1.0 | fraction | Michelson modulation depth of the flickering disc (1.0 = 100%), configurable. |
| `mean_luminance_fraction` | 0.5 | fraction of `[Lmin, Lmax]` | Both discs' mean luminance, converted to cd/m^2 via the active gamma calibration; logged in `stimulus_params["mean_luminance_cdm2"]`. |
| `duration_ms` | 1000.0 | ms | ~1 s presentation. |
| `onset_ramp_ms` | 50.0 | ms | Raised-cosine on/off ramp on the modulation *envelope* -- see below. |
| `waveform_mode` | `"continuous"` | -- | `"continuous"` or `"square_wave"`, see below. |
| `min_frequency_hz` | 2.0 | Hz | Easy end of the QUEST+ domain. |

A fixation cross is shown throughout. Both discs' `fillColor` is set every
frame from a gamma-linearized drive level (`colorSpace="rgb1"`) computed
from the target luminance for that frame, so contrast between the two discs
at flicker minima/maxima is always physically calibrated, not just nominal.

### Onset transients: why the default ramps, not an abrupt step

The task literature's baseline design uses an abrupt onset. We instead
default to a 50 ms raised-cosine ramp
(`waveform.raised_cosine_envelope`, `onset_ramp_ms` is a configurable
parameter -- set it to `0` to get the literal abrupt-onset design) applied
to the *modulation envelope* (not the mean luminance, which never changes at
onset). Rationale: an abrupt luminance step accompanying stimulus onset is
itself a potent, frequency-*independent* transient cue -- especially
troublesome exactly where this task is hardest (high frequencies, where the
steady-state modulation amplitude is already heavily attenuated by frame
sampling; see below), an observer could detect the transient rather than
genuine steady-state flicker, artificially inflating the measured CFF. Tyler
& Hamer (1990) discuss transient-response contamination in flicker
paradigms generally; our ramped-envelope choice follows that concern. This
is a documented, overridable default, not a hidden behavior --
`stimulus_params["onset_ramp_frames"]` is logged every trial.

## Waveform generation and honesty about display limits

A frame-based display cannot present an arbitrary continuous sinusoid: each
frame holds one constant luminance for its whole duration. Given a target
frequency `f`, mean luminance `Lmean`, modulation depth `m`, and the
*measured* refresh rate `R` (`display.refresh_hz`), frame `n`'s luminance is

```
L_n = Lmean * (1 + m * envelope(n) * sin(2*pi*f*n/R + phi))
```

(`waveform.frame_luminance_sequence`), where `envelope` is the raised-cosine
onset/offset ramp above and `phi` is a **random starting phase**, drawn
fresh each trial and logged (`stimulus_params["phase_rad"]`) -- without a
random phase, a fixed relationship between frame boundaries and waveform
zero-crossings could itself become alias-dependent and systematically bias
which frequencies look "cleaner" than others.

**Two modes** (`waveform_mode` param):

- **`"continuous"`** (default): any frequency up to the usable ceiling (see
  below) is allowed; the waveform is generated by the formula above and is
  therefore subject to ordinary sampling error as `f` approaches `R/2`.
- **`"square_wave"`**: only frequencies `f = R / (2k)` for a positive
  integer `k` are allowed (`waveform.valid_square_wave_frequency`,
  `usable_square_wave_frequencies`); each half-cycle is then exactly `k`
  frames long, so the waveform (`waveform.square_wave_frame_luminance_sequence`)
  has **zero sampling error** at its fundamental -- every frame is exactly
  the high or low luminance level, with no partial-frame averaging. This
  trades continuous frequency resolution for an exactly alias-free stimulus,
  useful when the experimenter wants to rule out sampling-error questions
  entirely. Square-wave phase is necessarily quantized to whole frames
  (`phase_frames`, also logged).

### The usable ceiling: why `refresh_hz / 3`, not `refresh_hz / 2`

The temporal Nyquist limit is `refresh_hz / 2` (`DisplayGeometry.max_flicker_hz`),
but at exactly the Nyquist rate a sinusoid is sampled at only 2 points per
cycle -- alternating between two luminance values whose *magnitude* is
entirely determined by starting phase, from full amplitude (phase = 90 deg)
down to **exactly zero** (phase = 0 deg: `sin(pi*n) == 0` for every integer
`n`, so the frame sequence is perfectly flat despite 100% nominal
modulation -- verified directly in
`test_critical_flicker_fusion_waveform.py::test_dft_fundamental_amplitude_catastrophically_attenuated_at_exact_nyquist`).
Three samples per cycle (`refresh_hz / 3`, `USABLE_CEILING_DIVISOR = 3.0`) is
the traditional floor for a frame-sampled waveform to still resemble a
sinusoid at *any* starting phase, and leaves comfortable margin for the
per-trial amplitude check below to catch graceful, not catastrophic,
degradation before it's severe enough to threaten validity.
`CFFParams.waveform_mode` and this ceiling interact: `"square_wave"` mode's
frequency grid is intersected with `[min_frequency_hz, usable_ceiling_hz]`
too (`usable_square_wave_frequencies`), so both modes share the same honest
upper bound.

### Per-trial amplitude honesty check

Every trial, `present()` computes the actually-presented sequence's
fundamental amplitude via a direct (non-FFT-bin-restricted) evaluation of
the discrete-time Fourier transform at exactly `frequency_hz`
(`waveform.dft_fundamental_amplitude` -- a Goertzel-style single-frequency
correlation, valid whether or not `frequency_hz * n_frames / refresh_hz` is
an integer), and logs both the **nominal** amplitude
(`modulation_depth * mean_luminance_cdm2`) and the **measured** one into
`stimulus_params`, plus their ratio
(`amplitude_attenuation_ratio`) and a boolean `amplitude_attenuated`
(`ratio < 0.9`). `summarize()` raises a `warning`-severity
`waveform_amplitude_attenuated` flag if any main-block trial was attenuated.

## Display-limited detection: a tolerance band, not exact edge equality

Naively, one might detect "CFF exceeds this display's ceiling" by checking
whether the *fitted* 75%-correct frequency reaches `usable_ceiling_hz`
exactly. In practice this essentially never happens within a realistic trial
budget: QUEST+'s min-entropy stimulus selection resolves the domain's exact
edge only very slowly for a near-ceiling-everywhere observer, because
adjacent `threshold_values` grid points near the edge predict almost
identical probability-correct (at the edge itself, with the guess rate fixed
at 0.5 and near-zero lapse, `p(x_min) approx 0.75` -- i.e. the domain edge's
model-implied performance is *already* right at the criterion, so
Bayesian evidence for "the true threshold is exactly at, vs. just inside,
the edge" accumulates only as the ratio of two very close probabilities,
which takes very many trials to resolve with confidence). We verified this
numerically (see `tests/tests_catalog/test_critical_flicker_fusion.py` and
its module docstring): a genuinely-always-correct-even-at-ceiling simulated
observer's fitted 75%-point reaches within our 15%-of-domain-span tolerance
band of the hard edge reliably by ~150-200 trials, but essentially never
exactly reaches the edge itself even at 2000+ trials of direct replay.
Given that, `EDGE_TOLERANCE_FRAC = 0.15` (three times the generic
`vpsych.data.quality.check_threshold_at_range_edge`'s default 5%) is used:
if the fitted 75%-point falls within 15% of the domain span from the hard
edge, we report display-limited rather than a point estimate. This
deliberately errs toward flagging a borderline case as display-limited
rather than reporting an overconfident point estimate right at the edge --
the honest choice when the display itself cannot resolve the difference.

## Requirements

- `needs_gamma_calibration=True`, `min_luminance_grade="B"`: mean luminance
  must be gamma-linearized and must match *exactly* between the flickering
  and steady discs, or an uncontrolled luminance difference becomes a cue
  independent of flicker.
- `min_refresh_hz = 240` (**changed from an earlier 120 Hz -- see "Why 240,
  not 120" below**): CFF in young observers under good photopic conditions
  commonly falls in the ~30-60 Hz range (Hecht & Shlaer 1936; higher still
  at high luminance per the Ferry-Porter law), and a frame-based display's
  `"continuous"`-mode usable ceiling is only about a third of its refresh
  rate (see above). 240 Hz gives a usable ceiling of 80 Hz, comfortably
  clear of that range with real margin; a 120 Hz display's ceiling (40 Hz)
  sits *inside*, not above, the realistic range.
- `stimulus_params["eccentricity_deg"]` is logged every trial (CFF varies
  with retinal eccentricity).
- `TestSummary.quality_flags` always includes an `info`-severity
  `lcd_response_time_limitation` flag: **LCD pixel response time attenuates
  high-frequency luminance modulation in a way this software cannot measure
  from software timing alone** -- the frame-luminance-sequence model in this
  module describes what the *frame buffer* asks the panel to show, not what
  the panel's liquid crystals physically achieve in the available time. A
  photodiode check (`tools/timing_check.py`) against the actual panel is
  recommended, especially near a run's usable ceiling.

### Why 240 Hz, not 120 Hz (or a mode-conditional requirement)

An earlier version of this test required only `refresh_hz >= 120`, reasoning
that a 120 Hz display's 40 Hz `"continuous"`-mode ceiling was "comfortably
above typical human CFF." That claim was false -- 40 Hz sits *inside*
Hecht & Shlaer's realistic 30-60 Hz range, not above it -- and this module's
own simulated validation confirms the practical consequence directly: at
120 Hz, even a true CFF as low as 30 Hz (well under the nominal 40 Hz
ceiling) already lands the majority of simulated runs in the
`display_limited` regime (13-15 out of 15 runs across the slopes tested; see
"Validation" below), because the domain-edge tolerance band
(`EDGE_TOLERANCE_FRAC`, 15% of the tested span) reaches much further into
the domain when the ceiling itself is low. 120 Hz was, in practice, mostly
unable to measure a realistic observer's CFF at all -- it would silently
report a lower bound (or a biased point estimate near the edge) for most
runs, not a trustworthy number.

240 Hz (ceiling 80 Hz) fixes this with real margin: the same simulated
battery at 240 Hz shows low display-limited rates (0-3 out of 20-15 runs)
and small bias for true thresholds up to 30 Hz, only rising again as the
true threshold approaches the ceiling itself (as it honestly should).

Two other options were considered and rejected for now:

- **A mode-conditional requirement** (allow 120 Hz *only* in `"square_wave"`
  mode, whose ceiling is `refresh_hz / 2` = 60 Hz, closer to the realistic
  range): `TestRequirements`/`check_requirements`
  (`vpsych.tests_catalog.base`) express a single `min_refresh_hz` per test,
  with no hook for a requirement that depends on the test's own configured
  parameters (`waveform_mode` here). Building that hook would touch shared
  infrastructure used by every test in the catalog, well beyond this fix's
  scope. A simple, uniformly-safe 240 Hz floor was chosen instead of a
  narrower, mode-aware one; a future revision could add the hook and relax
  this for `"square_wave"` mode specifically.
- **Leaving 120 Hz allowed and relying only on the `display_limited` flag**:
  rejected because, per the measurement above, that flag would fire on the
  *majority* of runs at 120 Hz even for realistic (non-extreme) observers --
  technically honest (never a fabricated point value), but practically
  useless as the normal outcome of a 4-minute test. Refusing to run below
  240 Hz is the more useful honest behavior: an explicit refusal up front,
  not a near-certain "display-limited" result at the end.

**Measurable window per refresh rate** (`"continuous"` mode, `min_frequency_hz`
default 2 Hz):

| `refresh_hz` | Usable ceiling | Allowed by `check_requirements`? | Practical window |
|---|---|---|---|
| 120 | 40 Hz | No (`min_refresh_hz=240`) | Effectively unusable for realistic CFF (see above) |
| 240 | 80 Hz | Yes | Accurate roughly up to ~30 Hz for typical-to-steep observers (slope >= ~1.5); increasingly display-limited (correctly) as true CFF approaches ~45+ Hz |
| 360+ | 120+ Hz | Yes | Same accurate window, with more margin before the edge band |

`"square_wave"` mode's ceiling is `refresh_hz / 2` at any refresh (not
gated differently by `check_requirements`, per the policy above), so a
240 Hz display running `waveform_mode="square_wave"` reaches a 120 Hz
ceiling, comfortably past the realistic range, at the cost of only testing
the discrete frequency grid `refresh_hz / (2k)`.

## Output

`output_units = "hz"`. `TestSummary.estimate.value` is either the
75%-correct CFF in Hz, or (if display-limited) the usable ceiling itself as
an explicit lower bound -- always check
`TestSummary.estimate.extra.get("display_limited")` before treating `value`
as a point estimate.

## Validation

Simulated end-to-end recovery
(`tests/tests_catalog/test_critical_flicker_fusion.py::test_simulated_end_to_end_recovery_via_runner`):
80 trials against a `psychometric:threshold=-1.0,slope=1.5,lapse=0.02`
observer (expressed directly in the transformed intensity, per the task
requirement) recovers a threshold within 0.5 log10-Hz of the known
75%-correct target when not display-limited. The display-limited path is
covered separately
(`test_simulated_end_to_end_display_limited_case`, 300 trials against an
observer whose true threshold is far below the tested domain) and on a
hand-built fixture (`test_summarize_display_limited_case_on_fixture`, 200
trials all-correct at the hardest grid point) -- see "Display-limited
detection" above for why these need a larger trial count than the smoke-level
recovery check.

### Phase 4: the slope-grid bug, and its fix

An earlier version of `make_procedure()` derived `slope_values` from the
*width of the tested domain* rather than from any property of human
psychophysics: `linspace(max(span*0.02, 1e-3), span*0.5, 6)`, which at a
240 Hz display gives slopes of roughly **0.03-0.80** -- values far shallower
than a realistic human psychometric-function slope (see
`DEFAULT_SLOPE_VALUES`'s docstring in
`vpsych.tests_catalog.critical_flicker_fusion` for the literature/generic-
beta reasoning behind the replacement). This one bug compounded three ways:

1. **The grid could not represent a realistic observer.** A true slope of
   1.5 or 3.0 (plausible for a near-threshold 2AFC psychometric function,
   comparable to the beta~2-5 typical range used by `visual_acuity`'s and
   `vernier_acuity`'s own grids) was entirely outside the domain-derived
   grid, forcing QUEST+'s fit onto its own shallowest available point
   regardless of the data.
2. **The 75%-criterion conversion divides by the fitted slope**
   (`questplus_weibull_x_at_p`'s `log10(-ln(q)) / slope` term), so an
   artificially shallow fit doesn't just misestimate the slope -- it
   *amplifies* the reported frequency's error by roughly `1/slope`.
3. **That amplified offset pushed the fitted 75%-point toward the domain's
   hard edge**, spuriously triggering the `cff_display_limited` flag even
   for a true threshold nowhere near the display's actual ceiling. Measured
   directly (ad hoc simulation, not itself checked in): with the old grid at
   240 Hz, a true CFF of 20 Hz with a true slope of 1.5 or 3.0 landed
   **100% of simulated runs** in the display-limited regime -- a false
   verdict, not a real ceiling effect.

Compounding this, the test's own slow validation
(`test_summarize_bias_and_coverage_over_many_simulated_runs`) simulated only
`slope_true=0.25` -- a value that happened to fall *inside* the narrow,
accidental old grid -- and `continue`d silently past every display-limited
run without ever reporting the rate. So the published bias/coverage figures
were simultaneously parameter-matched to a value that couldn't expose the
bug, and selection-biased (silently discarding runs rather than reporting
how many were censored).

**The fix** (`DEFAULT_SLOPE_VALUES = linspace(0.5, 6.0, 6)`, module-level,
independent of the tested domain's width -- see its docstring) mirrors
`visual_acuity`/`vernier_acuity`'s own generic-beta grids: no published
source was found giving a CFF-specific Weibull-beta estimate, so (following
those two tests' precedent) this uses the same typical beta~2-5 range
documented for 2AFC near-threshold detection tasks generally, rather than
inventing a CFF-specific citation, and was verified empirically (see below)
to bracket realistic slopes with margin.

### Measured bias / coverage / display-limited rate (current, post-fix)

Re-measured via `test_summarize_bias_and_coverage_over_many_simulated_runs`,
now parametrized over realistic true thresholds (15-45 Hz, not the old
3-5 Hz) and several true slopes spanning the plausible range -- including
1.5 and 3.0, both entirely outside the old grid. Ground truth is defined
directly in `questplus`'s own native Weibull parameterization (matching
`visual_acuity`/`vernier_acuity`'s validation approach, avoiding the
family-mismatch confound a `vpsych.core.psychometric`-family ground truth
would introduce -- see `QuestPlusProcedure`'s "Criterion conversion
pitfall" docstring). N=20 runs/case, 50 trials/run, `refresh_hz=240`
(ceiling 80 Hz), run with `OPENBLAS_NUM_THREADS=1`. **The display-limited
rate is reported for every case, not discarded** -- bias/coverage are
computed only over the non-display-limited runs within each case (a
display-limited run reports a censored lower bound, not a point estimate
comparable in Hz), but the censored fraction itself is always shown:

| true CFF | true slope | mean bias (log10-Hz-equivalent `x`) | CI coverage | display-limited rate |
|---|---|---|---|---|
| 15 Hz | 1.5 | +0.042 | 0.95 | 0/20 (0%) |
| 20 Hz | 1.5 | +0.065 | 1.00 | 0/20 (0%) |
| 30 Hz | 1.5 | +0.044 | 1.00 | 1/20 (5%) |
| 20 Hz | 3.0 | -0.010 | 0.90 | 0/20 (0%) |
| 20 Hz | 0.6 | +0.225 | 0.94 | 3/20 (15%) |

For comparison, the *old* domain-derived grid at the same refresh/true
values (ad hoc, not checked in): true CFF 20 Hz, true slope 1.5 ->
100% display-limited (no comparable point-estimate runs at all); true CFF
20-30 Hz, true slope 0.25 (the old validation's only tested slope) -> mean
bias +0.45 to +0.70, coverage as low as 0.11-0.22 once the grid no longer
happened to bracket the observer's actual slope.

**Where this test is, and is not, accurate**, stated explicitly:

- **Accurate** (bias typically < 0.1 log10-Hz-equivalent units, coverage
  >= 0.9, display-limited rate low): true CFF roughly 15-30 Hz at a 240 Hz
  display, for observers with a typical-to-steep psychometric slope
  (>= ~1.5 in this `x = -log10(f)` parameterization).
- **Degraded but not fabricated**: for a genuinely shallow observer (slope
  ~0.6), bias grows to ~0.2-0.5 log10-Hz-equivalent units and the
  display-limited rate rises -- the 75%-criterion conversion's `1/slope`
  amplification (see above) is a property of the conversion itself, not
  something a wider slope grid can fix, since the grid already brackets
  0.6 comfortably. This is an honest residual limitation, not swept under
  the old validation's selection bias.
- **Correctly refuses a point estimate, rather than reporting a wrong one**,
  as true CFF approaches the display's usable ceiling (the display-limited
  rate rises toward the edge, as intended -- see "Display-limited
  detection" above).
- **Not measurable at all** on a 120 Hz display in `"continuous"` mode
  (rejected outright by `check_requirements`; see "Why 240 Hz, not 120 Hz"
  above) -- its 40 Hz ceiling sits inside, not above, the realistic human
  CFF range, and the same simulated battery at 120 Hz shows the majority of
  runs display-limited even at a true CFF of 30 Hz.

## Citations

- Hecht, S., & Shlaer, S. (1936). Intermittent stimulation by light: V. The
  relation between intensity and critical frequency for different parts of
  the spectrum. *Journal of General Physiology*, 19(6), 965-977.
  https://doi.org/10.1085/jgp.19.6.965
- Porter, T. C. (1902). Contributions to the study of flicker. *Proceedings
  of the Royal Society of London*, 70, 313-329.
  https://doi.org/10.1098/rspl.1902.0031
- Tyler, C. W., & Hamer, R. D. (1990). Analysis of visual modulation
  sensitivity. IV. Validity of the Ferry-Porter law. *Journal of the
  Optical Society of America A*, 7(4), 743-758.
  https://doi.org/10.1364/JOSAA.7.000743
- Watson, A. B. (2017). QUEST+: A general multidimensional Bayesian adaptive
  psychometric method. *Journal of Vision*, 17(3), 10.
  https://doi.org/10.1167/17.3.10
