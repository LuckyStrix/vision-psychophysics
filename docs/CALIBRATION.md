# Calibration

Explains how to calibrate a display for research-grade testing: geometry
measurement, luminance/gamma characterization (photometer vs.
psychophysical grades), color primary characterization, the environment
checklist, and calibration grading and staleness. See
`vpsych/core/calibration/models.py` for the authoritative grading rules and
`vpsych/core/display.py` for the geometry math this document describes.
Implementation of the math described here lives in
`vpsych/core/calibration/{gamma,dither,color,geometry,photometer}.py`.

A calibration is only as good as the procedure that produced it. This
document exists so a participant's data can carry a grade that means
something specific and reproducible, not just "calibrated: yes/no".

## Overview: what a calibration contains

A `Calibration` (`vpsych.core.calibration.models.Calibration`) bundles:

- `geometry`: resolution, physical size, viewing distance, refresh rate
  (`vpsych.core.display.DisplayGeometry`).
- `gamma`: the luminance response curve, grade A/B/C (see below).
- `color`: the display's primaries, grade A/C (see below).
- `environment`: a self-reported checklist completed at calibration time.
- `created_utc`, `software_version`, `notes`.

It is immutable and content-addressed: `Calibration.content_hash()` is the
SHA-256 of its canonical JSON, and it is stored as
`<data_root>/calibration/cal-<hash>.json`. A session references its
calibration by that hash (`SessionInfo.calibration_hash`), so reanalysis
always knows exactly which calibration produced a given session's data, and
two calibrations that happen to be identical (e.g. re-running the geometry
step with the same numbers) collapse to the same file instead of
proliferating near-duplicates.

Calibrations older than a configurable age (default 30 days,
`Calibration.is_stale(max_age_days=30)`) are flagged stale in the UI. A
monitor can drift (backlight aging, driver updates, OS color-management
changes) enough over weeks to matter for a threshold measurement, so a
stale calibration should be re-run before a new session, not just
acknowledged and ignored.

## 1. Geometry

Three numbers are needed: physical screen width/height in cm, pixel
resolution, and viewing distance in cm. Resolution normally comes from the
OS (`vpsych.core.calibration.geometry.query_os_resolution_refresh`, lazily
querying pyglet/psychopy, with a clear error rather than a silent guess if
no display server is available -- enter it by hand in that case). Viewing
distance is measured with a tape measure or chin-rest, to the plane of the
screen, at session time (different tests within a session may use different
distances, e.g. near vs. distance acuity -- `PlannedTest.viewing_distance_cm`
is set per test, not just once per calibration).

Physical screen size is the one number people don't have a ruler handy for,
so a **credit-card matching** procedure is supported
(`vpsych.core.calibration.geometry`): display an on-screen rectangle, resize
it (arrow keys / drag) until its width visually matches a real ID-1 format
card (a standard credit/debit card, or most driver's licenses/national ID
cards) held flat against the screen. ID-1 is fixed by international standard
(ISO/IEC 7810) at **85.60 mm x 53.98 mm**, so the matched pixel width alone
determines the display's pixel density and hence its total physical width
at the known horizontal resolution
(`estimate_display_width_cm`/`px_per_cm_from_card_match`). This is less
precise than a ruler against the actual bezel (parallax in the match, card
wear, screen curvature on some panels) but is good to a percent or two in
practice, which is generally within the geometry error budget compared to
viewing-distance measurement error.

Once geometry is set, `DisplayGeometry` computes px/deg, the display's
Nyquist limit (`nyquist_cpd`, the highest spatial frequency the display can
faithfully render -- two pixels per cycle), and the temporal Nyquist limit
(`max_flicker_hz`, `refresh_hz / 2`). `check_requirements` uses these to
refuse to run a test whose required spatial frequency is at or above the
display's Nyquist limit at the configured viewing distance -- pushing the
viewing distance back (which is free) is usually the fix, not "buy a higher
resolution monitor."

**Refresh rate:** the OS-reported nominal refresh (used in `geometry`) is a
*starting point*, not the number sessions actually run at. Every real
session run re-measures the refresh rate live
(`vpsych.core.timing.measure_refresh`) right after opening the fullscreen
window and aborts (`RunnerExitCode.REFRESH_MISMATCH`) if it differs from the
calibration's `refresh_hz` by more than 1% -- a calibration made at 60 Hz is
not valid evidence for a session that (for whatever OS/driver reason) is
actually running at 59.94 Hz or 75 Hz. See `tools/timing_check.py` to
measure refresh, jitter, and dropped frames standalone before running real
sessions.

## 2. Luminance and gamma

The display's luminance response is modelled as a power law,

    L(v) = Lmin + (Lmax - Lmin) * v**gamma

where `v` is the normalized drive level (0 = black, 1 = max white) and `L`
is luminance in cd/m^2 (`vpsych.core.calibration.gamma`). Two ways to
characterize it, in order of preference:

### Grade A: photometer

A supported photometer or colorimeter (via `psychopy.hardware`, e.g. a
Cambridge Research Systems ColorCAL, or a PR-series spectroradiometer) reads
luminance at a series of drive levels (`measurement_levels`, evenly spaced
including 0 and 1), and `fit_gamma` fits `gamma`/`Lmin`/`Lmax` by ordinary
least squares on the linearized relationship
`ln((L - Lmin) / (Lmax - Lmin)) = gamma * ln(v)`. Channels can be measured
independently (`measure_gamma_per_channel`) if per-channel gamma matters for
a test (e.g. isoluminant color stimuli), or combined (`measure_gamma`,
driving R=G=B together) for a single `gamma_single`. `tools/gamma_measure.py`
drives this end to end with a live photometer. A full per-channel lookup
table (`fit_gamma_lookup`, monotone PCHIP interpolation of the raw measured
points, no parametric power-law assumption) is available when the response
doesn't follow a clean power law closely enough (e.g. near black, where
many panels deviate from a pure gamma curve).

This is the only grade that gives an absolute luminance scale (`Lmin`,
`Lmax` in cd/m^2) and hence is required for anything reporting or targeting
an absolute contrast/luminance value.

### Grade B: psychophysical (no photometer)

When no photometer is available, gamma can still be *estimated* (not
absolute luminance) from the observer's own visual system, using a
half-luminance bisection procedure
(`vpsych.core.calibration.gamma.estimate_gamma_psychophysical`): at each of
several target fractions `p`, show a fine checkerboard/dithered pattern
where a fraction `p` of pixels are driven to level 1 and the rest to level
0, and have the observer adjust a uniform gray patch until it looks equally
bright. Because the retina pools light **linearly** over a fine spatial
pattern (photoreceptors integrate incident luminous energy, not
gamma-encoded pixel values), the checkerboard's perceived luminance is the
linear mixture `Lmin + p * (Lmax - Lmin)`, so the matched drive level
`v_match` satisfies `v_match**gamma = p`, and gamma is recovered (with a
confidence interval, via log-log least squares across levels) without ever
measuring an absolute luminance. This needs no hardware beyond the display
and the observer's own judgment, at the cost of (a) no absolute luminance
scale, and (b) a CI reflecting the noisiness of the perceptual match itself.

### Grade C: none

No characterization; sRGB gamma (2.2-ish) is assumed. Any test declaring
`needs_gamma_calibration` refuses to run under grade C.

## Bit depth and dithering

An 8-bit-per-channel frame buffer can only show 256 distinct levels per
channel -- a contrast step no finer than ~1/255 -- which is coarser than
many threshold measurements need (e.g. contrast sensitivity near the
sensitivity peak, or color discrimination trivectors). `vpsych.core.
calibration.dither` implements the **noisy-bit** (bit-stealing) method
(Allard & Faubert, 2008, "The noisy-bit method for digital displays:
converting a resolution limitation into a pseudo-resolution," *Behavior
Research Methods*, 40(3), 735-743): instead of always rounding a desired
intensity to the nearest representable 8-bit level, round it
*stochastically* so the **expected value** of the displayed level equals
the desired (sub-LSB) intensity exactly. A single dithered frame looks
slightly noisy at the pixel level, but averaged over many independent draws
-- successive frames (temporal dithering) and/or neighbouring pixels
(spatial dithering) -- the display reproduces the target intensity far more
precisely than 8 bits alone would allow.

`effective_contrast_resolution(n_samples, bit_depth)` reports the resulting
effective step size: quantization noise from stochastic rounding averages
down with `sqrt(n_samples)` (central limit theorem), so e.g. averaging over
4 independent samples yields roughly double the effective resolution of a
single 8-bit sample. This effective resolution is what should be logged
alongside any contrast value finer than 1/255, and is part of why "what bit
depth did this test render at" is a meaningful, loggable quantity rather
than always just "8".

## 4. Color

Color primary characterization (`vpsych.core.calibration.color`) has two
grades:

- **Grade A (measured):** primaries (R/G/B chromaticity + luminance) and
  the white point are measured with a spectroradiometer or colorimeter, as
  CIE 1931 `xyY`. From these, `primaries_to_xyz_matrix` builds the exact
  `RGB -> XYZ` transform for *this specific display* (the standard
  normalized-primary-matrix construction: expand each primary to
  unit-`Y` tristimulus values, then scale columns so the primaries'
  weighted sum reproduces the measured white point exactly -- the same
  algorithm underlying `colour-science`'s `normalised_primary_matrix`,
  validated against it in `tests/core/test_calibration_color.py` to better
  than 1e-6 relative error).
- **Grade C (sRGB assumed):** nominal sRGB/BT.709 primaries and D65 white
  are assumed, uncalibrated. Fine for casual/demo use; any test declaring
  `needs_color_calibration` refuses to run under grade C.

From XYZ, the module provides: CIE 1976 `(u', v')` chromaticity
(`xy_to_uv_prime`/`xyz_to_uv_prime`, the perceptually-more-uniform
chromaticity space color-discrimination thresholds are conventionally
reported in, e.g. "u'v' x10^-4"); Stockman-Sharpe **LMS cone space**
(`xyz_to_lms`/`xyz_to_lms_matrix`) via a matrix derived by regressing the
Stockman & Sharpe (2000) 2-degree cone fundamentals against the CIE 1931
2-degree standard observer color-matching functions (both loaded from
`colour-science`'s bundled standard data tables -- the *data* is the cited
external source, the regression that turns it into a fixed 3x3 matrix is
this module's own, standard construction); Weber **cone contrast**
(`cone_contrast`, `(L,M,S) vs background -> (dL/L, dM/M, dS/S)`);
**confusion-line directions** (`confusion_line_direction_uv`) from a given
background `u'v'` toward the protan/deutan/tritan **copunctal points**
(CIE 1931 xy values from Vienot, Brettel & Mollon 1999, Table I, themselves
derived from Smith & Pokorny 1975 cone fundamentals -- the standard
reference values for dichromat confusion-line simulation, used e.g. by the
planned trivector color-discrimination test); and a chromaticity
**gamut check** (`gamut_check`, point-in-triangle against the display's
primaries).

## 5. Environment checklist

Completed (self-reported) before each session, stored on both the
`Calibration` and the per-session `SessionInfo`
(`vpsych.core.calibration.models.EnvironmentChecklist`):

- **Room lighting controlled:** dim, free of glare/reflections on the
  screen. Stray light raises the effective black level and can bias
  low-luminance/low-contrast measurements.
- **Monitor warmed up:** at least 15 minutes powered on before calibrating
  or testing. LCD backlights (LED or CCFL) measurably drift in luminance
  and sometimes color for the first several minutes after power-on;
  calibrating a cold display produces numbers that don't hold for the rest
  of the session.
- **Night light / blue-light filter disabled:** OS-level color temperature
  adjustments (e.g. Linux "Night Light" / `gsd-color`, Windows Night Light,
  macOS Night Shift) silently retint the whole display and invalidate a
  color calibration if left on, or if toggled on partway through a
  session.
- **HDR / adaptive-brightness / auto-contrast disabled:** any OS or panel
  feature that changes the input-to-luminance mapping dynamically (based on
  content, ambient light sensors, or power state) breaks the fixed gamma
  curve a calibration assumes. See docs/SETUP_LINUX.md for how to check and
  disable these on Linux specifically.

`notes` is free text (e.g. an ambient-light meter reading), and must stay
pseudonymous like every other data field -- no names, locations, or other
identifying detail.

## Calibration grading and staleness

Summary of the grading rules (`vpsych.core.calibration.models.
grade_luminance`/`grade_color`, enforced by `vpsych.tests_catalog.base.
check_requirements`):

| Grade | Luminance (`gamma.method`) | Color (`color.method`) |
|---|---|---|
| A | `photometer` | `measured` |
| B | `psychophysical` | *(no grade B for color)* |
| C | `none` | `srgb_assumed` |

A test declares its prerequisites via `TestRequirements`
(`needs_gamma_calibration`, `needs_color_calibration`, `min_luminance_grade`,
`min_color_grade`, `min_viewing_distance_cm`, `min_refresh_hz`,
`max_required_cpd`). `check_requirements` returns a list of human-readable
reasons a test currently cannot run "research-grade" (empty means it can);
the test-catalog UI disables the card and shows those reasons verbatim, and
the runner refuses to start a session with any unmet requirement
(`RunnerExitCode.REQUIREMENTS_UNMET`), listing every reason in the final
status rather than failing silently or partway through.

A calibration's grade and age are shown prominently wherever its data is
shown -- on the home screen's calibration badge, in the session builder, and
written into every trial/summary file a session produces, so a reader of
the data six months later can tell at a glance how much to trust it without
having to cross-reference a separate calibration log.
