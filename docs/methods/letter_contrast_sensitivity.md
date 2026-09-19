# Letter contrast sensitivity

Measures contrast sensitivity for letter identification -- the log contrast
at which an observer can no longer reliably identify a fixed-size letter --
in the spirit of the Pelli-Robson chart, the clinical standard for
low-contrast letter acuity. Letter-based contrast sensitivity correlates
better with real-world visual function (reading, mobility, face
recognition) than grating-based CSF measures alone, and is fast enough to
administer routinely.

## Method

10-alternative-forced-choice (10AFC) letter identification: a single Sloan
letter is shown, the participant types the corresponding key, and log
contrast sensitivity is estimated from a QUEST+ run
(`vpsych.core.procedures.questplus_procedure.QuestPlusProcedure`, Watson
2017) on **log10 Weber contrast**, fitting `questplus`'s own native
`"weibull"` psychometric function (Watson & Pelli 1983's original QUEST
parameterization -- see `QuestPlusProcedure`'s "Criterion conversion
pitfall" docstring section for exactly how this differs from
`vpsych.core.psychometric`'s own, differently-parameterized `weibull`
family) with guess rate 0.1 (1/10, matching the 10 letter alternatives) and
a small free lapse-rate grid (`[0.0, 0.02, 0.04]`).

**This is not the Pelli-Robson chart itself** -- see "Difference from the
Pelli-Robson chart" below.

## Optotypes: procedural Sloan letters, not a bundled font

The 10 Sloan letters -- **C D H K N O R S V Z** (Sloan 1959), as
standardized by the NAS-NRC (1980) Committee on Vision report -- are the
alphabet the Pelli-Robson chart itself uses.

**Font licensing was investigated and explicitly avoided**: this repository
is GPL-3.0 and public. Denis Pelli distributes a "Sloan" font for
research use, but no verified OFL (SIL Open Font License) or other clearly
redistribution-permitting license text for that specific font could be
located during this implementation, so it was **not bundled**, per the
task's explicit instruction to bundle a font only when its license clearly
permits redistribution. Instead, following the preferred fallback the task
specifies, letters are rendered **procedurally** from coarse 5x5-grid
stroke definitions
(`vpsych.tests_catalog._contrast_rendering.render_letter_ink_mask` /
`_SLOAN_5X5_GRIDS`): authentic Sloan letterforms are themselves designed on
a 5x5 unit grid with stroke width 1/5 of the letter height (this is the
typographic convention the task description also invokes), but the actual
letterforms are not simple filled blocks. The grids in this codebase are a
**documented, coarse block-letter approximation** of that convention, not
a reproduction of Denis Pelli's or the NAS-NRC committee's actual
letterforms -- legible and distinguishable at a normal viewing size, but
**not clinically equivalent to a real Sloan/Pelli-Robson chart's optotype
fidelity**. Each letter's 5x5 grid is a hand-authored `1`/`0` stroke
pattern (see `_SLOAN_5X5_GRIDS`'s source for all 10); `render_letter_ink_mask`
nearest-neighbor-upsamples each grid cell to a `size_px // 5`-pixel block,
so the rendered stroke width is always exactly `letter_height_deg / 5`, as
intended.

If a verified openly-licensed Sloan font becomes available in the future
(the task's stated preference, ahead of procedural rendering, when a clean
license can be confirmed), swapping the rendering path is localized to
`build_stimuli`/`present`'s letter-drawing call and does not change this
test's procedure, scoring, or output.

## Task and stimulus

Contrast is defined as **Weber contrast** against the mid-gray background:
`contrast = (L_background - L_letter) / L_background`, dark letter on a
mid-gray field at the calibration's mean luminance
(`vpsych.tests_catalog._contrast_rendering.weber_drive_from_ink_mask`).
Background pixels are always held at exactly the mean-luminance fraction
(0.5) regardless of the trial's contrast; only ink pixels darken as
contrast increases. As with `contrast_sensitivity_function`, the target
drive value is linearized against the gamma calibration and noisy-bit
dithered (Allard & Faubert 2008) **every frame the letter is visible** (not
once per trial), so effective contrast resolution improves with the number
of frames the letter is shown for -- see that test's own docs for the
shared dithering-accuracy unit test and the identical
`gamma_channel_model_from_calibration`/`PsychoPyBackend`-gamma-ramp
shared-code notes, which apply here unchanged.

**Letter height**: 2.8 deg by default (`letter_height_deg`), matching the
Pelli-Robson chart's design angle at its intended ~1 m viewing distance
(each Pelli-Robson letter is 4.9 cm tall, subtending ~2.8 deg at 1 m).
Configurable per session (`PlannedTest.viewing_distance_cm` and
`letter_height_deg` jointly determine the on-screen size via
`DisplayGeometry.deg_to_px`).

**Presentation**: unlike a brief-flash detection task, the letter remains
visible (like a printed chart) from stimulus onset until a response key is
pressed or `response_timeout_ms` (5000 ms default) elapses -- `present()`
polls the keyboard every frame inside its drawing loop rather than drawing
a fixed number of frames then waiting separately. A 500 ms fixation cross
precedes the letter; a 500 ms inter-trial interval follows.

**Letter identity** is drawn uniformly at random from the 10 Sloan letters
each trial (`trial_ctx["rng"]`) and logged in `stimulus_params["letter"]`.

## Intensity (contrast) grid

`DEFAULT_INTENSITY_VALUES` spans log10 Weber contrast from -2.4 (~0.4%) to
-0.05 (~89%), 25 log-spaced points. The lower bound is set by what noisy-bit
dithering can reliably resolve at this test's response-window frame counts
(`vpsych.core.calibration.dither.effective_contrast_resolution`) -- at
several hundred milliseconds of frames, effective resolution is comfortably
finer than 0.4%, so the grid's own floor, not rendering precision, is the
binding constraint on how low a contrast this test can meaningfully probe.
The upper bound is just short of a maximal, unmistakably-black letter
(reserved headroom for catch trials, which use 0.9 Weber contrast by
default).

## Procedure and trial count

Default **40 main trials** (`LetterCSParams.max_trials`). This follows the
same "few tens of trials" convention Pelli-Robson-style tests use in
practice (the physical chart itself presents far fewer "trials" -- 8
triplets per line, ~2-3 lines actually read near threshold -- but a
QUEST+-driven digital analogue needs enough trials for the posterior to
concentrate). Simulated-recovery validation
(`tests/tests_catalog/test_letter_contrast_sensitivity.py::
test_recovery_slow_over_several_true_thresholds`) runs 40-trial sessions
against 3 true threshold values (5 reps each) and finds mean bias well
within a loose 0.5 log-unit smoke tolerance; QUEST+'s own, much
larger-scale bias/coverage validation lives in
`tests/procedures/test_questplus_procedure.py` and is not duplicated here
per `docs/WRITING_A_TEST.md`.

Catch trials use a high (0.9 by default, `catch_contrast`) Weber contrast
-- an essentially unmistakable letter for any attentive observer.

## Output and threshold criterion

**Output**: log contrast sensitivity, `log10 CS = -log10(threshold Weber
contrast)`, with a CI (`TestSummary.estimate.units ==
"log10_contrast_sensitivity"`).

**Threshold criterion** (explicitly stated per the task requirement, and
corrected as part of the Phase 4 fix list's item 1 -- see
`vpsych.core.procedures.questplus_procedure`'s "Criterion conversion
pitfall" docstring section): `QuestPlusProcedure.estimate().value` is
`questplus`'s own *native* Weibull threshold -- the ~63.2%-of-range point in
*its* parameterization, which is **not** the same as
`vpsych.core.psychometric`'s own `F(0) = 0.5` convention, contrary to what
an earlier version of this section claimed. This test's documented
criterion is instead the FrACT-style midpoint,
`p_correct = guess + 0.5 * (1 - guess - lapse)` -- equivalent to
`vpsych.core.psychometric`'s `F(0) = 0.5` point -- roughly **55% correct**
with `guess=0.1` and a small lapse rate; `summarize()` now reaches it by
explicitly inverting `questplus`'s own fitted curve via
`QuestPlusProcedure.intensity_at_p_correct`, rather than (as an earlier,
buggy version did) assuming the raw native threshold was already there
(which is actually closer to **67%** correct for this guess rate -- a
real, if modest, numeric difference this bug introduced into every
previously-reported threshold). Either way, this criterion is *not* the
75%-correct point conventionally used for this suite's 2AFC tests. Log CS
is then `-1 *` the (correctly criterion-converted) log10 Weber contrast
threshold, with the CI transformed the same way (`ci_low`/`ci_high` swap
under negation: `log_cs_ci_low = -contrast_ci_high`,
`log_cs_ci_high = -contrast_ci_low`).
`TestSummary.estimate.extra["threshold_criterion"]` states this in plain
text alongside every summary; `extra["threshold_log10_weber_contrast"]`
carries the criterion-converted (pre-negation) contrast-domain threshold,
and `extra["raw_questplus_native_threshold_log10_weber_contrast"]` the raw,
unconverted native threshold, for anyone who wants to re-derive a
different criterion from the logged trials.

## Difference from the Pelli-Robson chart

**This is explicitly not equivalent to a physical Pelli-Robson chart
score**, for several reasons that matter for anyone comparing results
across the two:

1. **Scoring rule**: the physical chart is read letter-triplet by
   letter-triplet, and the conventional clinical scoring rule credits a
   *partial* final triplet (commonly 1 letter = 0.05 log unit) rather than
   an all-or-nothing threshold crossing. This test scores each letter
   trial independently and fits a continuous psychometric function
   instead -- there is no triplet structure or partial-credit rule at all.
2. **Procedure**: the chart presents a fixed, monotonically decreasing
   contrast sequence once; this test is trial-by-trial Bayesian adaptive
   (QUEST+), which is more statistically efficient per trial but samples
   contrast non-monotonically and non-exhaustively.
3. **Letterforms**: procedurally-approximated Sloan letters (see above),
   not the chart's actual optotypes.
4. **Threshold definition**: the `F(0)=0.5` (~55%-correct) criterion above,
   vs. the chart's own scoring convention (approximately, though not
   formally, an 8-alternative-per-triplet-implied criterion around the
   observer's last fully/partially read line).

Both approaches estimate "the same underlying quantity" in only a loose,
qualitative sense; do not treat this test's log CS as numerically
interchangeable with a clinically-recorded Pelli-Robson chart score.

## Confusion scoring

`score()` checks **exact letter identity only** -- there is no partial
credit or similarity weighting (e.g. confusing "O" with "C", or "N" with
"K", scores exactly as wrong as any other mismatch). This deliberately
avoids baking any letter-confusability assumption into the adaptive
procedure or the reported threshold. Every trial's `TrialRecord` already
carries both `correct_response` (the target letter) and `response` (what
was typed) in the standard trial schema, which is sufficient to build a
full 10x10 confusion matrix (response vs. target) from the trials TSV
alone in a later analysis pass; this test does not compute one itself.

## Requirements

- `needs_gamma_calibration=True`, `min_luminance_grade="B"`: Weber contrast
  cannot be trusted without at least a psychophysical gamma estimate,
  exactly as for `contrast_sensitivity_function`.
- No color-calibration or minimum-viewing-distance/spatial-frequency
  requirement: letter strokes are coarse relative to the display's Nyquist
  limit at any reasonable viewing distance (a 2.8 deg-tall letter's
  1/5-height stroke is ~0.56 deg wide, orders of magnitude coarser than
  this suite's grating-based tests), so rendering fidelity is not the
  binding constraint here -- the minimum representable contrast is (see
  "Intensity grid" above).

## Limitations

- Procedural letterform approximation (not real Sloan letterforms) -- see
  above.
- Not equivalent to a physical Pelli-Robson chart score -- see above.
- `n_dropped_frames` is currently always logged as 0 by this test's
  real-display presentation path, matching the same current project-wide
  limitation `contrast_sensitivity_function` documents.
- The default 40-trial budget is a documented convention, not derived from
  a formal power analysis targeting a specific bias/coverage bound (unlike
  qCSF's 300-trial recommendation, which does rest on such a sweep); the
  slow validation test above is the available evidence it behaves
  reasonably at that trial count.

## Citations

- Pelli, D. G., Robson, J. G., & Wilkins, A. J. (1988). The design of a new
  letter chart for measuring contrast sensitivity. *Clinical Vision
  Sciences*, 2(3), 187-199.
- Sloan, L. L. (1959). New test charts for the measurement of visual acuity
  at far and near distances. *American Journal of Ophthalmology*, 48(6),
  807-813. https://doi.org/10.1016/0002-9394(59)90626-9
- National Academy of Sciences - National Research Council, Committee on
  Vision (1980). Recommended standard procedures for the clinical
  measurement and specification of visual acuity. *Advances in
  Ophthalmology*, 41, 103-148.
- Watson, A. B. (2017). QUEST+: A general multidimensional Bayesian
  adaptive psychometric method. *Journal of Vision*, 17(3):10.
  https://doi.org/10.1167/17.3.10
- Allard, R., & Faubert, J. (2008). The noisy-bit method for digital
  displays: converting a resolution limitation into a pseudo-resolution.
  *Behavior Research Methods*, 40(3), 735-743.
  https://doi.org/10.3758/BRM.40.3.735

## Validation

- `tests/tests_catalog/test_letter_contrast_sensitivity.py::
  test_simulated_end_to_end_recovery_via_runner`: fast smoke-level check
  (60 trials, one true threshold, tolerance 1.0 log unit -- deliberately
  loose given only 60 trials and the `F(0)=0.5` criterion's non-standard
  target percentage).
- `tests/tests_catalog/test_letter_contrast_sensitivity.py::
  test_recovery_slow_over_several_true_thresholds` (`@pytest.mark.slow`):
  3 true thresholds x 5 reps x 40 trials; measured mean bias per threshold
  was within the asserted 0.5 log-unit tolerance in local runs.
- Dithering accuracy and rendering math are covered by the shared
  `tests/tests_catalog/test_contrast_rendering.py` (see
  `contrast_sensitivity_function.md`'s Validation section for the same
  dithering-accuracy test, which both tests rely on identically).
- Display smoke test: `tests/tests_catalog/test_letter_contrast_sensitivity.py::
  test_display_smoke_two_trials` (`@pytest.mark.display`) opens a real
  PsychoPy window and presents 2 trials with a fake keyboard; passed
  locally (see the implementation report for the run result).
