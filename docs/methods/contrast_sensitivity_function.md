# Contrast sensitivity function (qCSF)

Measures the full contrast sensitivity function (CSF) -- contrast
sensitivity (1 / threshold contrast) as a function of spatial frequency --
rather than a single threshold. The CSF summarizes an observer's spatial
vision far more completely than acuity alone: two observers can have
identical high-contrast acuity (smallest resolvable letter) but very
different sensitivity to low-contrast, mid-frequency gratings, which is
where many real-world visual complaints (fog, glare, early cataract,
amblyopia, optic neuropathy) actually show up first.

## Method

Bayesian adaptive estimation via the quick CSF (qCSF) method (Lesmes, Lu,
Baek & Albright 2010), which jointly estimates the 4 parameters of a
truncated log-parabola CSF model from responses over a 2D stimulus space
(spatial frequency x contrast) -- see `vpsych.core.procedures.qcsf` and
`docs/METHODS.md`'s "Adaptive procedures" section for the full model and
implementation (precomputed-likelihood one-step-ahead expected-entropy
stimulus selection, not the `questplus` package). Each stimulus is scored
via 2AFC orientation identification (left/right arrow key: "which way is it
tilted?"), the variant Hou et al. (2010) validated for gratings and used
clinically to characterize CSFs in amblyopia; guess rate 0.5. The
underlying psychometric function's slope (0.35 log10-contrast units) and
lapse rate (0.02) are fixed, not estimated, per standard qCSF practice.

## Stimulus

A sinusoidal grating windowed by a **raised-cosine circular envelope**
(`vpsych.tests_catalog._contrast_rendering.raised_cosine_disc_envelope`: a
smooth, standard circular aperture -- full contrast out to half the patch
radius, then a half-cosine ramp to zero at the edge, avoiding the spurious
high-frequency content a hard-edged disc would introduce), tilted +/-45 deg
from vertical by default (`orientation_deg`, configurable), centered on a
mid-gray background at the calibration's mean luminance
(`(lum_min_cdm2 + lum_max_cdm2) / 2`). Spatial phase is randomized
uniformly per trial and logged in `stimulus_params["phase_rad"]`.

**Size**: 4 deg diameter by default (`grating_diameter_deg`). This is not
an arbitrary choice: it is exactly the size that keeps the lowest tested
spatial frequency (0.5 cpd, the bottom of the standard 0.5-16 cpd CSF
range) at 2 cycles across the patch (`grating_diameter_deg * 0.5 == 2`) --
fewer cycles than that and the grating's identity is dominated by
edge/envelope effects rather than the intended spatial frequency. A larger
patch would relax this constraint further but increases stimulus area
(more retina stimulated, closer to the fovea/parafovea boundary) and
presentation-frame computation cost; 4 deg was chosen as a standard,
moderate qCSF/grating aperture size balancing both (see Lesmes et al. 2010;
Hou et al. 2010, who both use small, few-degree Gabor/grating apertures for
this method, though this project does not attempt to reproduce their exact
stated size, as their reports do not commit to one number across studies).

**Contrast**: Michelson contrast in the *luminance domain*
(`vpsych.tests_catalog._contrast_rendering.michelson_drive_from_pattern`):
target luminance is `L_mean * (1 + contrast * pattern)`, converted to a
fraction of the calibration's `[Lmin, Lmax]` range and linearized against
the gamma calibration
(`vpsych.core.calibration.gamma.linearize`) before being handed to
PsychoPy. **Gamma calibration grade B (psychophysical) or better is
required** (`TestRequirements.needs_gamma_calibration=True,
min_luminance_grade="B"`) -- an uncharacterized ("grade C") display cannot
be trusted to reproduce a specified Michelson contrast at all, since the
whole point of linearization is correcting for the display's (typically
far from linear) luminance response. Contrasts below the ~1/255 (~0.39%)
step an 8-bit frame buffer can natively represent are still presented
correctly *in expectation* via noisy-bit dithering (Allard & Faubert 2008;
`vpsych.tests_catalog._contrast_rendering.dither_frame`, wrapping
`vpsych.core.calibration.dither.dither_to_uint8`). **The dither pattern is
refreshed every stimulus frame, not once per trial**: each of the
`stimulus_duration_ms`-worth of frames draws a freshly, independently
dithered array at that frame's (temporally-enveloped) target contrast, so
the *effective* contrast resolution after temporal averaging improves with
`sqrt(n_frames)`
(`vpsych.core.calibration.dither.effective_contrast_resolution`) --
logged nowhere per-trial today (a `TrialRecord`-level field for this would
be a natural Phase 3 addition), but computable post hoc from
`n_frames = stimulus_duration_ms / (1000 / refresh_hz)`. At the default 500
ms / 60 Hz (30 frames), effective resolution is about
`1 / (255 * sqrt(30)) ~= 0.07%` -- well under the ~0.1-0.2% fidelity this
method needs near the sensitivity peak. `tests/tests_catalog/
test_contrast_rendering.py::test_dither_to_uint8_mean_reproduces_low_contrast_target`
is the headless unit test verifying the mean rendered drive level over many
independent dithered samples reproduces a 0.2% contrast target within a
tight (multiple-of-`effective_contrast_resolution`) tolerance.

**Timing** (all converted to frames via `DisplayGeometry.frames_for_ms`,
never timed by the wall clock): a **temporal raised-cosine envelope**
multiplies the per-frame contrast, 500 ms total by default
(`stimulus_duration_ms`) with 100 ms onset/offset ramps
(`ramp_duration_ms`, `vpsych.tests_catalog._contrast_rendering.
raised_cosine_temporal_envelope`) -- i.e. a 300 ms plateau at full contrast
bracketed by two 100 ms half-cosine ramps, avoiding the transient temporal-
frequency content an abrupt on/off would introduce. A fixation cross is
shown for 500 ms before the grating (`fixation_duration_ms`); the response
window is up to 3000 ms after stimulus offset (`response_timeout_ms`);
inter-trial interval is 500 ms (`iti_ms`). All are configurable per-session
`CSFParams` fields.

## Spatial frequency grid and rendering-fidelity limits

`QCSF.default_grids()`'s spatial-frequency grid (0.5-24 cpd, log-spaced) is
intersected, **at each test instance's actual configured display and
viewing distance**, with two limits:

- **Lower limit** (stimulus size): the lowest tested frequency must still
  show >= 2 cycles across the patch, i.e. `2 / grating_diameter_deg` cpd --
  0.5 cpd at the default 4 deg diameter (see above).
- **Upper limit** (rendering fidelity): at most `display.nyquist_cpd / 2`,
  i.e. **at least 4 pixels per cycle** -- twice the bare 2-px/cycle Nyquist
  limit `DisplayGeometry.nyquist_cpd` already represents. This is a
  deliberately conservative rendering-fidelity margin: a sinusoidal grating
  sampled at only 2 px/cycle (the literal Nyquist limit) is visibly
  distorted (aliased, and its effective contrast attenuated by the
  display's pixel/subpixel point-spread function) well before the true
  optical limit; 4 px/cycle is a common, conservative criterion for
  presenting a *recognizable, correctly-contrasted* grating, not just an
  aliasing-free one.

If this intersection is narrower than the intended standard 0.5-16 cpd
range, the actually-tested range is still used (never silently widened past
what the display supports), and `summarize()` raises a
`reduced_csf_frequency_coverage` quality flag reporting exactly what range
was tested.

`TestRequirements.max_required_cpd = 16.0` (checked against the display's
*raw* Nyquist limit, `DisplayGeometry.nyquist_cpd`, by
`vpsych.tests_catalog.base.check_requirements` -- a weaker, generic check
than this test's own internal 4-px/cycle grid-construction criterion above)
and `min_viewing_distance_cm = 100.0`. The 100 cm floor is a conservative
lower bound: on a "typical" 1920x1080, ~53 cm-wide display, 100 cm gives
`nyquist_cpd / 2` of about 15.8 cpd (just under the standard range's top);
the recommended 2-3 m viewing distance (see "Stimulus" language elsewhere
in the codebase and the general qCSF literature's convention of testing
gratings at longer distances specifically to raise pixels-per-degree, not
because the stimulus itself needs a large working distance) gives
`nyquist_cpd / 2` of roughly 31-47 cpd on the same display, comfortably
covering 0.5-16 cpd with margin. Because real displays and setups vary,
`summarize()`'s `reduced_csf_frequency_coverage` flag is the authoritative,
per-session source of truth for what was actually tested, not this
document's illustrative numbers.

## Procedure and trial count

`QCSF` (`vpsych.core.procedures.qcsf`), `max_trials` defaulting to
`QCSF.RECOMMENDED_MIN_TRIALS` (300) -- see `docs/METHODS.md`'s qCSF section
for the pooled-simulation bias numbers this recommendation is based on
(-0.039 +/- 0.007 log units at 300 trials, vs. -0.06 to -0.08 at 100).
`CSFParams.quick=True` overrides this to a fast ~100-trial run (ignoring
`max_trials`) for demos/piloting; a `quick_csf_run` quality flag is always
raised on such a run, disclosing the larger expected bias, so a "quick" CSF
estimate is never silently indistinguishable from a full one downstream.

Catch trials present a low spatial frequency (1.5 cpd by default,
`catch_spatial_frequency_cpd`) at high contrast (0.5 by default,
`catch_contrast`) -- easily visible to any attentive observer with normal
or corrected vision, contributing to `TestSummary.catch_lapse_rate` without
updating the driving `QCSF` posterior (per the shared trial-loop protocol).

## Output

**Primary**: AULCSF (area under the log CSF), `trapz(log10 CS(f), x=log10(f))`
integrated over `f` in `[1, 18]` cpd, with a posterior credible interval
(`QCSF.estimate()`; see `docs/METHODS.md` for why the point estimate is the
plug-in AULCSF at posterior-mean parameters rather than the posterior mean
of AULCSF itself, and that section's bias table). `TestSummary.estimate.units
== "aulcsf_log10cs_log10cpd"`.

`TestSummary.estimate.extra` additionally carries:

- `posterior_mean` / `posterior_sd`: the 4 CSF parameters (peak gain,
  peak frequency, bandwidth, low-frequency truncation) with their posterior
  SDs.
- `log_cs_at_frequencies_cpd`: posterior-mean log10 CS at the 6 standard
  frequencies `[1, 1.5, 3, 6, 12, 18]` cpd.
- `frequency_range_tested_cpd`: `[low, high]`, the actual spatial-frequency
  range this session's display/viewing-distance combination could test
  (see above).
- `cutoff_spatial_frequency_cpd`: the acuity-equivalent cutoff -- the
  frequency at which the plug-in CSF (posterior-mean parameters) crosses
  `CS = 1` (`log10 CS = 0`), found on a dense `[0.1, 200]` cpd log-spaced
  grid. Reported as a convenience cross-reference to acuity, not a
  replacement for the dedicated `visual_acuity` test.

`ContrastSensitivityFunctionTest.csf_curve(trials, n_points=...)` is a
figure-ready helper (for Phase 3 plotting, not part of the primary summary)
returning a `log10 CS(f)` curve with an *approximate* credible band across
`n_points` log-spaced frequencies spanning the range actually tested. It
Monte Carlo samples the 4 CSF parameters as **independent** normals around
their posterior mean/SD (a documented approximation -- it ignores posterior
correlation *between* parameters, unlike the exact joint-grid AULCSF CI
`summarize()` reports) using a fixed local seed, so it is reproducible from
`trials` alone.

## `summarize()` and reanalysis

`summarize()` replays a fresh `QCSF` (same grids the constructing test
instance would build, which are themselves a deterministic function of
`self.params`/`self.display`) over every main-block, non-catch trial's
logged `stimulus_params["spatial_frequency_cpd"]`/`["contrast"]`, in
`trial_index` order, then calls `estimate()` -- no live in-memory procedure
state is trusted, so `vpsych reanalyze` (which reconstructs a fresh test
instance from the session's recorded plan/calibration and re-parses
`_trials.tsv`) reproduces the identical summary. `QCSF.update`/`estimate`
use no randomness; `csf_curve`'s own randomness is a fixed local seed, not
`self.rng`, for the same reason.

## Requirements

- `needs_gamma_calibration=True`, `min_luminance_grade="B"`: absolute
  Michelson contrast cannot be trusted without at least a psychophysical
  gamma estimate (grade B); grade A (photometer) is strictly better but not
  required.
- `min_viewing_distance_cm=100.0`, `max_required_cpd=16.0`: see "Spatial
  frequency grid" above.

## Limitations

- The 4-px/cycle rendering-fidelity margin is a conservative, standard
  engineering choice, not derived from a specific psychophysical study of
  this exact display/stimulus combination; a well-characterized display
  might tolerate closer to the bare 2-px/cycle Nyquist limit in practice.
- Dropped frames are detected from the flip timestamps of the stimulus
  phase (`vpsych.core.timing.presentation_timing`, shared by every test).
  Only the stimulus phase is monitored: a drop during fixation, the
  response wait or the ITI is not counted.
- AULCSF's point estimate is systematically biased low at typical trial
  counts (see `docs/METHODS.md`); 300+ trials keeps this under 0.05 log
  units on average, but single-run variability remains substantial
  (SD ~0.15-0.25 log units) regardless of trial count.
- `csf_curve`'s credible band ignores cross-parameter posterior
  correlation (documented above); it is a visualization aid, not a
  substitute for the exact AULCSF CI.

## Shared-code notes (Phase 4: all three resolved)

- `vpsych.core.calibration.gamma` originally had no built-in constructor
  from a *stored* `GammaCalibration` straight to a `GammaChannelModel`;
  this test (and `letter_contrast_sensitivity`) built one locally
  (`vpsych.tests_catalog._contrast_rendering.gamma_channel_model_from_calibration`).
  **Resolved**: that constructor is now promoted into
  `vpsych.core.calibration.gamma.gamma_channel_model_from_calibration`
  (`_contrast_rendering` re-exports it for backward compatibility), and
  `vernier_acuity` uses it too (see next point).
- `vpsych.core.trial_loop.PsychoPyBackend` does not set a window gamma
  ramp. **Resolved explicitly, not left as a gap**: this is now the
  documented, permanent design (see `docs/WRITING_A_TEST.md` section 7's
  "Phase 4 gamma-ramp decision") -- every gamma-dependent test
  self-linearizes in Python instead, including `vernier_acuity`, which
  previously (incorrectly) assumed a window ramp it never actually got;
  see that test's module docstring for the fix.
- `PsychophysicalTest.make_catch_trial_intensity` was declared `-> float`
  in `tests_catalog/base.py`, though `vpsych.core.trial_loop.TrialLoop`
  already typed the value it assigns from that call as
  `float | dict[str, float]`. **Resolved**: the abstract signature is now
  `-> float | dict[str, float]`, and this test's override no longer needs
  its `# type: ignore[override]`.

## Citations

- Lesmes, L. A., Lu, Z.-L., Baek, J., & Albright, T. D. (2010). Bayesian
  adaptive estimation of the contrast sensitivity function: The quick CSF
  method. *Journal of Vision*, 10(3):17. https://doi.org/10.1167/10.3.17
- Hou, F., Huang, C.-B., Lesmes, L., Feng, L.-X., Tao, L., Zhou, Y.-F., &
  Lu, Z.-L. (2010). qCSF in clinical application: Efficient
  characterization and classification of contrast sensitivity functions in
  amblyopia. *Investigative Ophthalmology & Visual Science*, 51(10),
  5365-5377. https://doi.org/10.1167/iovs.10-5636
- Allard, R., & Faubert, J. (2008). The noisy-bit method for digital
  displays: converting a resolution limitation into a pseudo-resolution.
  *Behavior Research Methods*, 40(3), 735-743.
  https://doi.org/10.3758/BRM.40.3.735
- Watson, A. B. (2017). QUEST+: A general multidimensional Bayesian
  adaptive psychometric method. *Journal of Vision*, 17(3):10 (the
  min-entropy stimulus-selection principle `QCSF` applies to its own 2D
  stimulus space). https://doi.org/10.1167/17.3.10

## Validation

- `tests/procedures/test_qcsf.py` (pre-existing, not part of this task)
  validates the bare `QCSF` procedure's AULCSF recovery bias/CI against
  simulated `CSFObserver`s at scale (450 pooled runs at 300 trials: bias
  -0.039 +/- 0.007 log units) -- see `docs/METHODS.md`'s qCSF section.
- `tests/tests_catalog/test_contrast_sensitivity_function.py::
  test_simulated_end_to_end_recovery_via_runner` is a fast smoke-level
  check (150 trials, one true CSF, tolerance 0.5 log units) that the full
  test/runner pipeline (this test's own frequency-grid construction,
  `summarize()`, catch trials, quality flags) does not introduce bias
  beyond what the bare procedure already has.
- `tests/tests_catalog/test_contrast_sensitivity_function.py::
  test_recovery_slow_over_several_true_parameter_sets` (`@pytest.mark.slow`)
  runs the full pipeline at 3 true CSF parameter sets x 3 reps x 200
  trials each; measured mean bias per parameter set was well within the
  asserted 0.4 log-unit tolerance in local runs (a deliberately loose
  smoke-level bound given only 3 reps per set -- see
  `tests/procedures/test_qcsf.py` for the properly-powered version).
- Dithering accuracy: `tests/tests_catalog/test_contrast_rendering.py::
  test_dither_to_uint8_mean_reproduces_low_contrast_target` confirms the
  mean rendered drive level over 20,000 independent dithered samples
  reproduces a 0.2% contrast target within `5x effective_contrast_resolution`.
- Display smoke test: `tests/tests_catalog/test_contrast_sensitivity_function.py::
  test_display_smoke_two_trials` (`@pytest.mark.display`) opens a real
  PsychoPy window and presents 2 trials with a fake keyboard; passed
  locally (see the implementation report for the run result).
