# Motion coherence (random-dot kinematogram)

Motion coherence measures the smallest fraction of "signal" dots -- moving
together in one direction within a field of otherwise directionless "noise"
dots -- an observer needs to reliably report the coherent direction. It is
the standard psychophysical probe of global-motion integration in area MT/V5
(Newsome & Paré, 1988; Britten, Shadlen, Newsome, & Movshon, 1992), and is
widely used clinically and developmentally as a marker of dorsal-stream
("magnocellular") visual function.

## Method

**Procedure**: QUEST+ (Watson, 2017) drives `log10(coherence fraction)` over
the domain `[log10(0.01), log10(1.0)]` (1%-100% coherence). Catch trials use
100% coherence, the domain's most-detectable value
(`make_catch_trial_intensity`). The psychometric function is a fixed-family
Weibull (see `docs/METHODS.md`'s "Psychometric function families"), guess
rate 0.5 (2AFC), free lapse rate drawn from `[0.0, 0.02, 0.04]`.

**Task**: 2AFC. A random-dot kinematogram is shown for one interval; the
observer presses the left or right arrow key for the direction the dot field,
as a whole, appeared to move.

**Threshold reporting**: `summarize()` converts the QUEST+ posterior's
F=0.5-crossing threshold (in log10 coherence) to the intensity at 75% correct
via `vpsych.core.psychometric.intensity_at_p_correct`, then reports it as
coherence *percent* (`10**x * 100`), with a 95% credible interval converted
by the same monotonic offset. `TestSummary.estimate.units ==
"coherence_percent"`.

## Stimulus

Every default is documented with its unit and a citation; see
`MotionCoherenceParams` (`src/vpsych/tests_catalog/motion_coherence/__init__.py`)
for the authoritative field list.

| Parameter | Default | Unit | Citation / rationale |
|---|---|---|---|
| `aperture_diameter_deg` | 10.0 | deg | Newsome & Paré (1988)-style circular RDK aperture, centered on fixation. |
| `dot_diameter_arcmin` | 6.0 | arcmin | Newsome & Paré (1988). |
| `dot_density_per_deg2` | 2.0 | dots/deg^2 | Within the density range surveyed by Pilly & Seitz (2009); moderate density avoids dot-overlap crowding while keeping the field visually dense. |
| `speed_deg_per_s` | 5.0 | deg/s | Newsome & Paré (1988)-typical RDK speed. |
| `dot_lifetime_frames` | 10 | frames | Limited dot lifetime (Scase, Braddick, & Raymond, 1996) -- at 60 Hz, ~167 ms -- reduces the ability to track any single dot across the whole trial. |
| `duration_ms` | 500.0 | ms | ~500 ms presentation, converted to whole frames via `DisplayGeometry.frames_for_ms`. |

A central fixation point is shown throughout. Dots are bright
(`colors=(1, 1, 1)`) on the window's background, drawn with a
`psychopy.visual.ElementArrayStim` whose per-frame `xys` come directly from
`vpsych.tests_catalog.motion_coherence.dots.step_dot_field` -- **not**
PsychoPy's own `DotStim` motion logic, so the exact displayed positions are
computable, loggable, and headless-testable.

## Algorithms (design decisions)

All three are fixed implementation choices (not user-tunable parameters),
logged verbatim into every trial's `stimulus_params` so downstream analysis
always knows exactly which scheme produced the data:

- **Noise algorithm — `"random_direction"`** (`dots.NOISE_ALGORITHM`): each
  frame, every non-signal ("noise") dot is assigned a fresh, independent,
  uniformly random direction and moves the *same distance* as a signal dot
  that frame. Scase, Braddick, & Raymond (1996) compared this against
  "random position" (full re-plotting each frame, discontinuous jumps) and
  found "random position" gives the purest measure of global-motion
  sensitivity (least contaminated by local, dot-level motion energy a
  low-level detector could pick up). We chose "random direction" instead
  because (a) it keeps a well-defined, testable per-frame displacement
  magnitude for *every* dot, signal or noise (needed for the
  `dots.py` unit tests' "noise-dot speed distribution" check), and (b) it is
  the scheme used in most of the RDK literature descending from Newsome &
  Paré (1988) and Britten et al. (1992). The trade-off (documented, not
  hidden) is a small amount of local-motion-energy contamination relative to
  "random position"; a future revision could add "random position" as a
  second `noise_algorithm` option if a study specifically needs it.
- **Signal-dot selection — "white noise"** (`dots.SIGNAL_SELECTION =
  "white_noise"`): on every frame, an *exact* count of
  `round(coherence * n_dots)` dots is freshly, independently chosen (via
  `rng.permutation`) to be that frame's signal dots -- not a fixed subset
  held for the whole trial. Reselecting afresh each frame (per Pilly & Seitz,
  2009) prevents an observer from tracking a persistent subset of dots as
  individuated objects across the whole trial, which would inflate apparent
  coherence sensitivity for reasons unrelated to the global-motion percept
  the task intends to measure.
- **Edge handling — random replot, not wrap**
  (`dots.EDGE_HANDLING = "random_replot_in_aperture"`): a dot whose step
  would carry it outside the circular aperture is replotted at a fresh
  uniformly random in-aperture position. Wrapping to "the opposite side
  along the direction of motion" is a common RDK alternative, but on a
  *circular* aperture that phrase is ambiguous (the chord length through the
  aperture varies with the dot's off-axis position), and a naive wrap either
  breaks containment or introduces a position-dependent bias in wrap timing
  that could act as a coherence-independent, direction-correlated density
  cue near the aperture edge -- exactly the class of edge-handling artifact
  Scase et al. (1996) and Pilly & Seitz (2009) warn about. Random replotting
  sacrifices a small amount of local-density continuity but introduces no
  directional cue.
- **Limited dot lifetime**: every dot is independently replotted at a fresh
  random position after `dot_lifetime_frames` frames (regardless of whether
  it also left the aperture this frame), further reducing single-dot
  trackability. Initial dot ages are staggered uniformly across
  `[0, lifetime_frames)` (`dots.init_dot_field`) so replot events are spread
  across frames rather than all dots refreshing in synchrony (itself a
  salient, coherence-independent transient).

## Correspondence limit (Braddick's `d_max`)

Braddick (1974) describes a short-range motion-correspondence process that
fails once successive positions of a moving element are displaced by more
than roughly half the mean inter-element spacing -- beyond that, the visual
system can no longer reliably solve the frame-to-frame correspondence
problem for individual dots (a dot's post-displacement position becomes
roughly as close to a *different* dot's pre-displacement position as to its
own, i.e. spatial aliasing). `dots.check_displacement_dmax` computes the
per-frame displacement (`speed_deg_per_s * dt_s`, using the *measured*
`display.refresh_hz`) and the mean dot spacing
(`1 / sqrt(dot_density_per_deg2)`), and `present()` issues a `UserWarning`
(and logs `dmax_exceeded`/`displacement_per_frame_deg`/`dot_spacing_deg` into
`stimulus_params`) whenever displacement exceeds half that spacing. At the
defaults (5 deg/s, 60 Hz, 2 dots/deg^2), displacement is 0.083 deg per frame
against a spacing of 0.71 deg -- comfortably (8.5x) under the limit.

## Requirements

- `min_refresh_hz = 60`: motion coherence needs the display fast enough that
  per-frame displacement stays well under Braddick's `d_max` at a usable dot
  speed; see the check above.
- No gamma calibration is required (dots are drawn at maximum, undithered
  luminance; the task depends on motion structure, not fine luminance
  steps).
- Quality flag `excess_dropped_frames` (via the shared
  `vpsych.data.quality.check_dropped_frames`, already a warning above a 1%
  dropped-frame fraction) applies directly -- motion perception is
  unusually sensitive to dropped frames (a dropped frame is a discontinuous
  jump exactly like a `d_max` violation), so no test-specific override of
  that shared check was needed; it already covers this test's requirement
  as-is.

## Output

`output_units = "coherence_percent"`. `TestSummary.estimate.value` is the
coherence percentage (0-100) at which the fitted psychometric function
predicts 75% correct (2AFC), with a 95% credible interval in the same units.
`TestSummary.estimate.extra` also carries the raw QUEST+ F=0.5-crossing
threshold (`"threshold_f0.5_log10_coherence"`) and fitted slope/lapse rate
for transparency.

## Validation

Simulated end-to-end recovery (`tests/tests_catalog/test_motion_coherence.py::test_simulated_end_to_end_recovery_via_runner`):
80 simulated trials against a `psychometric:threshold=log10(0.30),slope=0.3,lapse=0.02`
observer recovers a threshold within 0.5 log10-coherence units of the known
75%-correct target -- a loose, smoke-level check (properly-powered
bias/coverage validation for QUEST+ itself lives in
`tests/procedures/test_questplus_procedure.py`, not duplicated per-test per
`docs/WRITING_A_TEST.md`).

**Bias/coverage of this test's own `summarize()` conversion** (F=0.5 ->
75%-correct, log10 -> percent; `@pytest.mark.slow`
`test_summarize_bias_and_coverage_over_many_simulated_runs`): measured at
N=100 simulated 50-trial runs against a
`threshold=-1.0, slope=0.3, lapse=0.02` observer: mean bias **+0.10
log10-coherence units** (SD 0.23), **99%** empirical coverage of the nominal
95% credible interval. The positive bias (estimated threshold reported
somewhat higher/easier than truth) is a small-sample effect of the
grid-discretized QUEST+ posterior combined with the 75%-point conversion,
not a sign error; it is well within the loose smoke-level tolerance the
slow test itself checks (N=50 there, for runtime).

## Citations

- Newsome, W. T., & Paré, E. B. (1988). A selective impairment of motion
  perception following lesions of the middle temporal visual area (MT).
  *Journal of Neuroscience*, 8(6), 2201-2211.
- Britten, K. H., Shadlen, M. N., Newsome, W. T., & Movshon, J. A. (1992).
  The analysis of visual motion: a comparison of neuronal and
  psychophysical performance. *Journal of Neuroscience*, 12(12), 4745-4765.
- Scase, M. O., Braddick, O. J., & Raymond, J. E. (1996). What is noise for
  the motion system? *Vision Research*, 36(18), 2579-2586.
  https://doi.org/10.1016/0042-6989(96)00016-9
- Pilly, P. K., & Seitz, A. R. (2009). What a difference a parameter makes: a
  psychophysical comparison of random dot motion algorithms. *Vision
  Research*, 49(13), 1599-1612. https://doi.org/10.1016/j.visres.2009.03.019
- Braddick, O. (1974). A short-range process in apparent motion. *Vision
  Research*, 14(7), 519-527. https://doi.org/10.1016/0042-6989(74)90041-8
- Watson, A. B. (2017). QUEST+: A general multidimensional Bayesian adaptive
  psychometric method. *Journal of Vision*, 17(3), 10.
  https://doi.org/10.1167/17.3.10
