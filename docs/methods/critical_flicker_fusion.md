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
`[0.0, 0.02, 0.04]`.

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
- `min_refresh_hz = 120`: CFF in young observers under good photopic
  conditions can approach or exceed 60 Hz (Hecht & Shlaer 1936; higher still
  at high luminance per the Ferry-Porter law), and a frame-based display's
  usable ceiling is only about a third of its refresh rate (see above) --
  120 Hz keeps the usable ceiling (40 Hz) comfortably above typical human
  CFF, whereas a 60 Hz display's ceiling (20 Hz) would be display-limited
  for most observers before the flicker even feels fast.
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

## Output

`output_units = "hz"`. `TestSummary.estimate.value` is either the
75%-correct CFF in Hz, or (if display-limited) the usable ceiling itself as
an explicit lower bound -- always check
`TestSummary.estimate.extra.get("display_limited")` before treating `value`
as a point estimate.

## Validation

Simulated end-to-end recovery
(`tests/tests_catalog/test_critical_flicker_fusion.py::test_simulated_end_to_end_recovery_via_runner`):
80 trials against a `psychometric:threshold=-1.0,slope=0.25,lapse=0.02`
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

**Bias/coverage of this test's own `summarize()` conversion**
(`@pytest.mark.slow` `test_summarize_bias_and_coverage_over_many_simulated_runs`)
-- **Phase 4 criterion-conversion pitfall fix and re-measurement.** An
earlier version of `summarize()` built a
`vpsych.core.psychometric.PsychometricFunction` directly from the raw
QUEST+ estimate and converted it via
`vpsych.core.psychometric.intensity_at_p_correct` -- silently the *wrong*
formula for a `QuestPlusProcedure` fit (see
`vpsych.core.procedures.questplus_procedure`'s "Criterion conversion
pitfall" docstring section). That version, measured at N=50 simulated
50-trial runs against a `threshold=-1.0, slope=0.25, lapse=0.02`
(`vpsych.core.psychometric`-family) observer (all runs landed in the
non-display-limited regime): mean bias **+0.22 log10(Hz) units** (SD 0.17),
**80%** empirical coverage of the nominal 95% credible interval.

`summarize()` now uses the correct, shared
`QuestPlusProcedure.intensity_at_p_correct`. Re-measuring with that same
`vpsych.core.psychometric`-family ground truth turned out to be
uninformative here: at `threshold=-1.0` the great majority of runs (43/50 in
one re-measurement) now land in the *display-limited* regime and are
excluded from the bias/coverage statistic entirely (a direct, mechanical
consequence of the fix -- see below), leaving too few comparable runs for a
meaningful before/after bias comparison at that specific true value.

Re-measured properly instead with ground truth defined directly in
`questplus`'s own native parameterization (matching
`visual_acuity`/`vernier_acuity`'s validation approach, which avoids the
family-mismatch confound entirely -- see
`test_summarize_bias_and_coverage_over_many_simulated_runs`, now
parametrized over 3 true values, N=30/value, 50 trials/run): mean bias
**-0.24 to -0.36** log10(Hz)-equivalent `x`-units (`x = -log10(f)`) across
`true_x` in `{-0.7, -0.6, -0.5}` (~5.0, ~4.0, ~3.2 Hz), with CI coverage
0.73-0.87 -- closer to, but still somewhat below, nominal 95%. Two things
explain both numbers honestly:

- **Why the display-limited fraction increased**: `questplus`'s own native
  anchor sits at the ~80%-of-range point for this test's `guess=0.5`
  (`0.5 + (1 - 0.5 - lapse) * (1 - e^-1) ~= 0.80`), *above* the 75%
  conventional criterion this test reports. Converting a fitted curve down
  from ~80% to 75% moves the reported point *toward* the domain's
  high-frequency ceiling (`x` decreases as `f` increases), so the fix
  itself pushes borderline-central true values closer to the
  `display_limited` edge-tolerance band -- a real, expected consequence of
  reporting an honest, correctly-derived criterion rather than an artifact.
- **Why coverage is still below nominal**: the reported CI is the raw
  QUEST+ credible interval shifted by a single additive offset, exact only
  to the extent the fitted slope/lapse posterior means are themselves
  well-estimated; at 50 trials on a threshold+slope+lapse joint posterior
  they are not always precise, so the *shape* (not just location) of the
  true sampling distribution is under-represented by a pure shift of the
  raw interval. A slope-and-lapse-aware re-derivation of the CI (e.g. via
  bootstrap resampling in `x`-space, mirroring `WeightedStaircase`'s
  approach) would likely improve coverage further and is a reasonable
  future improvement; the `@pytest.mark.slow` test itself uses a loose
  coverage floor (`>= 0.5`) so it still catches a gross regression without
  being a tight calibration gate on an already-documented limitation.

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
