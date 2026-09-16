# Contributing

This project is pre-alpha and under active development. If you are working
on it (including as an automated agent), follow these rules — they exist
because this is scientific instrumentation, not a typical app.

## Hard rules

1. **All stimulus durations are specified in frames, never seconds.**
   Convert milliseconds to a frame count once, with
   `DisplayGeometry.frames_for_ms`, and drive presentation from the frame
   count. Never call `time.sleep()` or otherwise time a stimulus by the
   clock.
2. **No silent data loss.** Every trial is persisted (appended to the trial
   TSV and flushed/fsynced) *before* the next trial begins. Interrupted
   sessions are marked `status: incomplete` and kept — never deleted or
   silently discarded.
3. **Every column or quantity has explicit units**, both in the name
   (`_cm`, `_deg`, `_hz`, `_cdm2`, `_s`, `_ms`, `_px`, `_cpd`) and in the
   sidecar/documentation description. A bare `duration` or `size` field is a
   bug.
4. **Core logic is headless-testable.** Code under `core/` and `data/` must
   import and run without a display, X server, or GPU — CI runs headless.
   `psychopy` (and any other display-dependent import) is imported lazily,
   inside the function that actually needs a window, never at module import
   time in `core/`.
5. **Pseudonymous IDs only.** Participant identifiers are `sub-XXXX` style.
   Never add names, emails, or other direct identifiers to data schemas or
   filenames.

## Workflow

- Python 3.10, managed with `uv`. `uv sync` to install, `uv run pytest` to
  test.
- `uv run ruff check .` and `uv run ruff format --check .` must pass.
- `uv run mypy src/vpsych/core src/vpsych/data` must pass (these packages
  are checked strictly; other packages are checked more leniently while
  interfaces settle).
- Add or update tests for anything you change under `core/` or `data/`.
- Interfaces frozen in Phase 0 (display geometry, calibration models,
  procedure protocol, trial record, test-catalog plugin interface, data
  schemas, runner status contract) should not change shape without updating
  every implementation that depends on them — treat changes to those files
  as an explicit, reviewed decision, not an incidental edit.
