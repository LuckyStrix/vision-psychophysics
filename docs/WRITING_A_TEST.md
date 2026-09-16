# Writing a perceptual test

A guide for implementing one of the 7 tests in the v1 battery (visual
acuity, qCSF, letter contrast sensitivity, color discrimination, motion
coherence, vernier acuity, critical flicker fusion) as a `tests_catalog`
plugin. It assumes you have read `CONTRIBUTING.md`'s hard rules (frames not
seconds, no silent data loss, units in every column, headless-testable
`core`/`data`, pseudonymous IDs only) and the plan's "Trial protocol"
section.

Everything here is grounded in the actual code, not aspirational: where the
runner's real behavior differs from what a docstring once promised, this
guide describes the real behavior (and says so). The single best resource
alongside this guide is `src/vpsych/tests_catalog/_example/__init__.py`, a
complete, runnable (but `hidden=True`, so never shown in a UI) example test
this guide walks through section by section. Read the two side by side.

## 1. The plugin interface

Your test is a subclass of `vpsych.tests_catalog.base.PsychophysicalTest`,
registered with `@register_test`, in its own subpackage of
`src/vpsych/tests_catalog/` (e.g. `tests_catalog/visual_acuity/__init__.py`).
It must implement every abstract method:

```python
@register_test
class VisualAcuityTest(PsychophysicalTest):
    spec = TestSpec(...)                     # class attribute, see below
    def __init__(self, params, display, calibration, rng): ...
    def make_procedure(self) -> AdaptiveProcedure | MultiParamProcedure: ...
    def make_catch_trial_intensity(self) -> float: ...
    def build_stimuli(self, win) -> dict[str, Any]: ...
    def present(self, win, intensity_or_stimulus, trial_ctx) -> PresentedTrial: ...
    def response_keys(self) -> list[str]: ...
    def score(self, response, stimulus_params) -> bool: ...
    def instructions(self) -> str: ...
    def summarize(self, trials: pd.DataFrame) -> TestSummary: ...
```

`simulated_response` is *not* abstract -- it has a concrete default (see
section 9) you may override but don't have to implement.

**`spec` is a class attribute**, a `TestSpec` (`tests_catalog/base.py`):
`id` (stable snake_case, becomes `TrialRecord.task_id` and the registry
key), `version` (semver -- bump it when you change the method, so old and
new data never silently mix at reanalysis), `domain`, participant- and
technical-facing descriptions, `output_units`, `estimated_minutes`,
`allowed_eyes`, `requirements` (a `TestRequirements`; see
`check_requirements` in `tests_catalog/base.py` for what each field
checks), `citations`, `params_model`, and `hidden` (default `False`; set
`True` only for a test-only fixture like `_example`, never for a real
test -- see section 12).

**Discovery**: your subpackage registers itself as an import-time side
effect of `@register_test`. Nothing imports it for you implicitly at
`import vpsych` time -- `vpsych.tests_catalog.base.discover_tests()`
(called once by `run_session` and by `reanalyze_session` before their first
`get_test()` lookup) walks and imports every direct submodule of
`tests_catalog`, which is what actually makes your test findable in a
fresh process. You don't need to call it yourself; just make sure your
test lives directly under `tests_catalog/` (a package, `__init__.py` is
enough) and it will be found.

## 2. `trial_ctx`: input and output keys

`present(self, win, intensity_or_stimulus, trial_ctx)`'s `trial_ctx` dict
is defined by `vpsych.core.trial_loop` (see that module's docstring, the
authoritative source -- this section summarizes it). Treat it as **both**
input and output.

**Input** (set by `TrialLoop`/the backend before calling `present`):

| Key | Type | Meaning |
|---|---|---|
| `"timeline"` | `TrialTimeline` | Frame counts for fixation/stimulus/ITI (a generic default from the runner; build your own if your test needs different phase counts). |
| `"rng"` | `numpy.random.Generator` | The *only* source of randomness you may use in `present()` (trial order, jitter, catch-trial correct side, ...) -- never `random`, never an unseeded generator, so a session is exactly reproducible from its logged seed. |
| `"block"` | `"practice"` \| `"main"` | For behavior that differs by block (most tests don't need this). |
| `"is_catch"` | `bool` | Whether this is a suprathreshold catch trial. |
| `"trial_index"` | `int` | 0-based index within this block/run. |
| `"keyboard"` | live `psychopy.hardware.keyboard.Keyboard`, or `None` | Non-`None` only under the real `PsychoPyBackend`. |
| `"simulated_observer"` | a `SimulatedObserver`, or `None` | Non-`None` only under `SimulatedBackend`. When set, skip drawing to `win` (`win` itself is `None` in this case) and use it to get a response -- see section 9. |

**Output** (you must set these before returning, since `PresentedTrial`
itself doesn't carry them but `TrialRecord` needs them):

| Key | Type | Meaning |
|---|---|---|
| `"stimulus_params"` | `dict[str, Any]` | The full stimulus parameterization actually presented this trial (test-specific keys). Written to `TrialRecord.stimulus_params` (JSON-encoded in the TSV). |
| `"correct_response"` | `Any` | The response that would have scored correct. Written to `TrialRecord.correct_response`. |

If you don't set them, the trial loop defaults to `{}`/`None` and the
record is still written, just uninformative -- always set them.

**One more input your test builds itself, not the trial loop**:
`build_stimuli(win)`'s return value is *discarded* by the caller
(`run_session` calls `test.build_stimuli(win)` and never looks at the
result again; `trial_ctx` has no `"stimuli"` key). Store whatever you build
on `self` (e.g. `self._stims = {...}`) for `present()` to reuse across
trials -- see `_example`'s `build_stimuli`/`_draw_grating`.

## 3. `stimulus_params` and `correct_response` conventions

- `stimulus_params` is a plain dict, test-specific, JSON-encoded verbatim
  into the TSV -- put whatever downstream analysis or debugging would want
  (intensity, the correct side/orientation/interval, any jittered
  parameters). Keep it JSON-native (numbers, strings, dicts, lists); avoid
  bare booleans if you also reuse the same value as a `response`/
  `correct_response` (see `vpsych.data.tsv`'s module docstring for that
  specific round-trip caveat).
- `correct_response` should be directly comparable (via `==`) to whatever
  `PresentedTrial.response` your test produces, since `score()` and (by
  default) `simulated_response()` both compare them with `==`.
- `score(response, stimulus_params)` reads `stimulus_params` (not
  `trial_ctx`) -- it's called by the trial loop with exactly what you put
  in `trial_ctx["stimulus_params"]`.

## 4. The `"intensity"` key, and `MultiParamProcedure`

Most tests drive a single scalar intensity through an `AdaptiveProcedure`
(`WeightedStaircase`, `QuestPlusProcedure`, `ConstantStimuli`) --
`next_intensity()` returns a `float`, and that's what's passed to
`present()` as `intensity_or_stimulus` and logged directly as
`TrialRecord.intensity`.

qCSF (`QCSF`, a `MultiParamProcedure`) is different: `next_stimulus()`
returns a `dict[str, float]` (e.g. `{"spatial_frequency_cpd":
..., "contrast": ...}`), a 2D stimulus space that doesn't reduce to one
scalar on its own. Because `TrialRecord.intensity` is always a scalar,
`vpsych.core.trial_loop._scalar_intensity` requires such a dict to include
an `"intensity"` key -- the chosen scalar summary dimension (for qCSF,
`log10(contrast)`) -- and raises `ValueError` if it's missing. If your test
uses `QCSF`, make sure the stimulus dict you build (typically just
returning what `next_stimulus()` gave you, perhaps with your own extra
keys) still has `"intensity"` in it; `QCSF.next_stimulus()` already
includes it for you.

## 5. Frames-only timing

**Never call `time.sleep()` or otherwise time a stimulus by the clock**
(`CONTRIBUTING.md` hard rule 1). Convert every design duration (ms) to a
frame count once, with `display.frames_for_ms(ms)`
(`vpsych.core.display.DisplayGeometry`), and drive presentation by drawing
for exactly that many frames:

```python
for _ in range(timeline.fixation_frames):
    fixation.draw()
    win.flip()
for frame in range(timeline.stimulus_frames):
    stim.draw()
    flip_time = win.flip()
    if frame == 0:
        onset_s = flip_time
```

`win.flip()` returns the flip timestamp; capture the *first* stimulus
frame's as `PresentedTrial.stimulus_onset_s`. Record dropped frames
(`PresentedTrial.n_dropped_frames`) and the raw inter-flip intervals
(`PresentedTrial.frame_intervals_s`, written to `_frames.tsv`) -- see
`vpsych.core.timing` for how the runner itself measures/summarizes these at
the session level; a test only needs to report what it observed for its
own stimulus phase.

Under `SimulatedBackend` there is no real display to measure timing from
-- fabricate plausible values (perfect, undropped frames at the nominal
refresh) rather than trying to time anything:

```python
frame_intervals_s=[1.0 / 60.0] * max(timeline.stimulus_frames, 1)
```

## 6. Lazy `psychopy` imports

`core/` and `data/` (and, by the same logic, every `tests_catalog` test,
since `discover_tests()` imports every test subpackage unconditionally --
including in headless CI) must import and run without a display. Import
`psychopy` (and anything display-dependent) *inside* the method that
actually needs a window, never at module import time:

```python
def build_stimuli(self, win: Any) -> dict[str, Any]:
    if win is None:
        return {}                    # SimulatedBackend: nothing to build
    from psychopy import visual      # lazy -- not at module top
    ...
```

`present()`'s real-display branch may also need a lazy `psychopy` import
(e.g. for `psychopy.core.wait`-style helpers -- though see section 5,
you generally shouldn't need clock-based waits at all). `win is None` is
exactly the signal to branch into the simulated path instead (see section
9); check it, don't assume you're always driving a real window.

## 7. Using calibration

`self.calibration` (a `vpsych.core.calibration.models.Calibration`) and
`self.display` (a `vpsych.core.display.DisplayGeometry`, already updated
with this test's configured viewing distance) are available from
`__init__` onward. See `docs/CALIBRATION.md` for the full model; in
`present()`/`build_stimuli()` you'll typically use:

- **deg <-> px**: `self.display.deg_to_px(size_deg)` /
  `self.display.px_to_deg(size_px)` to size and position stimuli in
  degrees of visual angle, and `self.display.px_per_deg_at_center` for
  spatial-frequency conversions (cycles/deg -> cycles/px for
  `psychopy.visual.GratingStim(sf=...)`).
- **Gamma linearization**: a real test should linearize contrast against
  `self.calibration.gamma` (`vpsych.core.calibration.gamma.linearize`) when
  setting the window's gamma ramp at window-creation time (once per
  session, not per trial -- this happens in `PsychoPyBackend`/window setup,
  not typically inside your test).
  `check_requirements`/`TestSpec.requirements.needs_gamma_calibration` is
  how you *require* a real (non-`"none"`) gamma calibration before your
  test is runnable at all.
- **Bit-stealing dithering**: for contrasts finer than the display's 8-bit
  step (~1/255), dither before drawing:
  `vpsych.core.calibration.dither.dither_to_uint8(intensity_0_to_1, rng)`
  (see `_example._draw_grating` for a worked call). Use `trial_ctx["rng"]`
  for this, per the "rng is the only source of randomness" rule.
- **Color transforms**: a color test uses
  `vpsych.core.calibration.color` against `self.calibration.color`'s
  measured (or sRGB-assumed) primaries; declare
  `requirements.needs_color_calibration`/`min_color_grade` accordingly.

## 8. Catch trial intensity

`make_catch_trial_intensity() -> float` returns a suprathreshold intensity
-- comfortably above threshold, so an attentive observer answers correctly
almost always. The trial loop calls it for the demo trial, every practice
trial, and ~10% of main-block trials (never two catch trials in a row,
never among the first `min_main_trials_before_catch`; see
`TrialLoopConfig`). It does **not** update the driving procedure. A catch
trial's `TrialRecord.procedure_state` is therefore identical to the
preceding trial's (verified by
`tests/core/test_trial_loop.py::test_catch_trial_procedure_state_unchanged_from_preceding_trial`)
-- if you ever see it differ, that's a bug, not expected variation.

A simple, robust choice: the single highest-contrast/most-detectable
value on your intensity grid, e.g. `max(YOUR_INTENSITY_VALUES)` for a
log10-contrast test (see `_example.make_catch_trial_intensity`).

## 9. The `simulated_response` hook

An observer (`vpsych.core.observers`) decides correct/incorrect against
its own ground truth; your test needs an actual response value in *its*
representation. Two supported ways to bridge that (see
`SimulatedObserver`'s docstring for the full contract):

1. Call `observer.respond(stimulus_params, rng)` directly when your
   stimulus dict already carries `"correct_alternative"`/`"alternatives"`
   (the convention `PsychometricObserver`/`CSFObserver` use) and your
   response representation matches what it returns.
2. **The generally-preferred, decoupled path** -- ask only for the
   correct/incorrect decision, then map it yourself:

   ```python
   observer = trial_ctx["simulated_observer"]
   if observer is not None:
       is_correct = observer.decide_correct(stimulus_params, rng)
       response = self.simulated_response(is_correct, stimulus_params, rng)
       return PresentedTrial(response=response, ...)
   ```

   `PsychophysicalTest.simulated_response(correct, stimulus_params, rng)`
   is concrete (not abstract) with a sensible default: it echoes
   `stimulus_params["correct_response"]` when `correct`, otherwise a
   uniformly random *other* value from `response_keys()` (falling back to
   `correct_response` itself if there's no alternative). Override it only
   if your response representation isn't one of `response_keys()` (e.g. a
   continuous judgment rather than a discrete key).

`decide_correct` is implemented by every built-in observer
(`PsychometricObserver`, `CSFObserver`) though it isn't a required part of
the `SimulatedObserver` Protocol; `_AlwaysCorrectObserver` (the runner's
`--simulate always_correct`) implements it too (trivially, always `True`).

The runner's `--simulate` accepts parameterized specs beyond
`always_correct` -- `psychometric:threshold=-1.0,slope=3.5,lapse=0.02` and
`csf:peak_gain=...,peak_freq=...,bandwidth=...,low_freq_truncation=...`
(see `vpsych.runner.__main__.parse_simulated_observer_spec`), plus
`--simulate-config PATH` for a per-task JSON map of specs in a multi-task
session. Use these from a shell or a test to drive your test's own
simulated-recovery validation (see section 11's item 3).

## 10. `summarize()` and quality flags

`summarize(self, trials: pd.DataFrame) -> TestSummary` computes your
test's result from its raw trials. Two rules matter more than the fitting
math itself:

- **Filter `trials` to `block == "main"` and (for the threshold fit)
  `~is_catch`** -- practice trials are never analyzed, and catch trials
  only inform `catch_lapse_rate`, not the threshold. Catch trials are
  still in `trials` (for that lapse-rate computation), and practice trials
  are in `trials` too (mostly for quality-flag purposes downstream) --
  don't just fit everything indiscriminately.
- **Prefer recomputing your procedure's state by replaying trials, not by
  trusting any live in-memory procedure object.** `summarize()` is called
  twice in the life of one test run: once by the runner right after the
  session (with the procedure that actually ran, still in memory), and
  again, independently, by `vpsych reanalyze` (`vpsych.data.reanalyze`) --
  which constructs a *fresh* test instance from nothing but the session's
  recorded plan/calibration and re-parses `_trials.tsv`. For `reanalyze` to
  reproduce your summary exactly (a hard requirement --
  `tests/integration/test_end_to_end.py` checks it, and every real test's
  own tests should too), `summarize()` must be a pure function of
  `trials` alone. The simplest way: construct a fresh procedure
  (`self.make_procedure()`) and replay `procedure.update(intensity,
  correct)` for every non-catch main trial in `trial_index` order, then
  call `procedure.estimate()` -- see `_example.summarize`. This works
  cleanly for `QuestPlusProcedure`/`WeightedStaircase`/`ConstantStimuli`/
  `QCSF`, none of which use any randomness in `update`/`estimate`. If your
  test's fitting genuinely needs randomness (a bootstrap CI), seed a fresh,
  local `numpy.random.Generator` with a fixed seed *inside* `summarize()`
  (not from `self.rng`, which has already been advanced by however many
  draws the live session made) so reanalysis draws the identical sequence.

**One nullable-dtype gotcha**: `vpsych.data.tsv.read_trials_tsv` decodes
`trials["correct"]` to pandas' nullable `"boolean"` dtype (not `"object"`)
specifically so `~trials["correct"]` (a natural way to compute "incorrect")
negates logically and propagates missing values as `pd.NA` -- this was a
real bug found and fixed while building `_example` (an object-dtype column
of bare Python `True`/`False` makes `~` apply Python's *bitwise* invert per
element, silently giving `-2`/`-1`). If you ever see an impossible
negative count/rate downstream of a `~trials[...]` expression, this is the
first thing to check.

**Quality flags**: call `vpsych.data.quality.compute_quality_flags` (or
the individual `check_*` functions it wraps) with whatever inputs your
test has -- at minimum `catch_lapse_rate`/`n_catch`/`dropped_fraction`
(always required), and optionally `threshold`/`range_min`/`range_max`
(pins-at-edge check), `luminance_grade`/`calibration_age_days` (staleness/
grade check -- read off `self.calibration`), `gof_p_value` (if you compute
a goodness-of-fit statistic, e.g. via `vpsych.core.psychometric.deviance_gof`),
and `n_trials`/`min_trials`. Fold the result straight into
`TestSummary.quality_flags`.

## 11. The required tests

Every real test needs, at minimum (see `tests/tests_catalog/test_example.py`
for a concrete instance of every one of these):

1. **Requirements check**: `check_requirements(YourTest.spec.requirements,
   a_display, a_calibration_or_None)` behaves as expected for at least one
   "met" and one "unmet" case, if your test declares any requirements.
2. **Score correctness**: `score()` returns `True`/`False` correctly for
   matching/mismatching/`None` responses.
3. **Simulated end-to-end recovery via the runner**: drive a real
   `run_session()` call (in-process, with a fake or real `Writer`) using
   `--simulate psychometric:...` (or `csf:...` for a `QCSF`-driven test)
   at a *known* true threshold/CSF, and check the recovered
   `TestSummary.estimate.value` is in the right ballpark of the known
   truth. This is a smoke-level check (loose tolerance, few trials) --
   the properly-powered bias/coverage validation belongs in
   `core/procedures`'s own tests (`tests/procedures/test_*.py`), not
   here; don't duplicate that work per-test.
4. **`summarize()` on a fixture**: build a small, fixed `pandas.DataFrame`
   of trials (not a live run) with a known pattern (e.g. 80% correct main
   trials, some catch trials), call `summarize()` twice, and assert the
   two results are identical (`model_dump()` equality) -- this is your
   local, fast check of the "pure function of `trials`" property section
   10 describes, before it's exercised for real by
   `tests/integration/test_end_to_end.py`'s `reanalyze` check. Also assert
   at least one quality-flag case fires when you engineer the fixture to
   trigger it (e.g. a 100% catch-trial miss rate).

## 12. Test-only fixtures: `hidden`

`TestSpec.hidden: bool = False`: set `True` for a test that must be
genuinely registered and runnable end to end (via `get_test`/the runner)
purely to support automated tests, but that a participant should never see
in a real catalog/session-builder UI. `vpsych.tests_catalog.visible_tests()`
is `all_tests()` filtered to non-hidden; a future UI should list from that,
not `all_tests()`. Never set `hidden=True` on one of the 7 real tests.

## 13. The `docs/METHODS.md` section template

`docs/METHODS.md` has a placeholder heading for each of the 7 tests
already (`## Visual acuity`, `## Contrast sensitivity function (qCSF)`,
...). Fill yours in with, at minimum:

```markdown
## <Test name>

<One paragraph: what this measures and why it matters, in plain language.>

**Method**: <adaptive procedure used (cite it), psychometric function
family/parameterization, guess rate, fixed vs. free lapse rate, number of
alternatives (2AFC/4AFC/8AFC/...), any test-specific stimulus model (like
qCSF's truncated log-parabola) with its own citation.>

**Stimulus**: <spatial/temporal parameters: size (deg), spatial frequency
(cpd) if relevant, duration (ms, converted to frames at the configured
refresh), eccentricity/eye condition, contrast/luminance range.>

**Output**: <output_units, what `TestSummary.estimate.value` means
concretely (e.g. "detection threshold, 75%-correct point"), any per-item
report analogous to qCSF's log CS at standard frequencies.>

**Requirements**: <what `TestSpec.requirements` declares and why --
gamma/color calibration grade, minimum viewing distance, Nyquist-limited
spatial frequency, minimum refresh rate.>

**Citations**: <full references, matching `TestSpec.citations`.>

**Validation**: <bias/coverage numbers from your test's own simulated
recovery, once measured -- follow the style of the "Adaptive procedures
and psychometric fitting" section's per-procedure numbers and qCSF's
AULCSF bias table further up this file. Report honestly; if a target
isn't met, say so and explain why (see the qCSF and staircase sections for
two worked examples of exactly that).>
```
