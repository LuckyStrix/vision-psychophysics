# Data format

Describes the on-disk layout of a vpsych data root: `dataset_description.json`,
`participants.tsv`/`.json`, `calibration/cal-<hash>.json`,
`sub-XXXX/ses-.../session.json`, `_trials.tsv` (+ sidecar), `_summary.json`,
`_frames.tsv`, and `MANIFEST.sha256`. See `vpsych/data/schemas.py` and
`vpsych/data/paths.py` for the authoritative schemas and path layout this
document describes.

The layout is inspired by [BIDS-behavioral](https://bids-specification.readthedocs.io/)
and [Psych-DS](https://psych-ds.github.io/): a data root is a self-describing
directory tree that can be validated, browsed, and re-analyzed without a
database, using only the files on disk.

The data root lives **outside this repository** (default `~/vpsych-data`,
overridable with the `VPSYCH_DATA_ROOT` environment variable -- see
`vpsych.data.paths.data_root`). This repo's `.gitignore` excludes it, and no
vpsych data root should ever be committed.

```
<data_root>/
  dataset_description.json
  participants.tsv
  participants.json
  calibration/
    cal-<sha256>.json
    cal-<sha256>.json
  sub-0001/
    ses-20260916T103000/
      session.json
      MANIFEST.sha256
      beh/
        sub-0001_ses-20260916T103000_task-acuity_eye-OD_run-1_trials.tsv
        sub-0001_ses-20260916T103000_task-acuity_eye-OD_run-1_trials.json
        sub-0001_ses-20260916T103000_task-acuity_eye-OD_run-1_summary.json
        sub-0001_ses-20260916T103000_task-acuity_eye-OD_run-1_frames.tsv
    ses-20260920T090000/
      ...
  sub-0002/
    ...
  catalog.sqlite
```

Every path is built and validated by `vpsych.data.paths`; nothing in this
document should be assembled by hand-formatting strings elsewhere in the
codebase.

## Dataset root

`dataset_description.json` (`vpsych.data.schemas.DatasetDescription`,
written by `vpsych.data.dataset.init_dataset`) identifies and versions the
whole data root:

| Field | Type | Description |
|---|---|---|
| `schema_version` | string (semver) | The vpsych data schema version this dataset was written with (see [Schema versioning and migration](#schema-versioning-and-migration)). |
| `software_version` | string | The `vpsych` package version that created/last wrote to this dataset. |
| `git_commit` | string \| null | Git commit hash of the `vpsych` checkout that created/last wrote to this dataset, if known (best-effort `git rev-parse HEAD`; `null` if unavailable, e.g. an installed package with no `.git`). |
| `name` | string | Human-readable dataset name. |
| `created_utc` | datetime | UTC timestamp the dataset root was created. |

`vpsych-data init [--root PATH] [--name NAME]` creates a new data root (or
is a no-op returning the existing description if one is already there;
pass `--force` to overwrite). It also creates an empty `calibration/`
directory and empty `participants.tsv`/`.json`.

`catalog.sqlite` (see [`vpsych.data.catalog`](#catalog-index)) is a
rebuildable SQLite index of the dataset, for the UI. It is **not** a source
of truth -- delete it and run `vpsych-data rebuild-catalog` at any time to
regenerate it from the files above.

## Participants

`participants.tsv` + its BIDS-style sidecar `participants.json` hold every
participant, managed by `vpsych.data.dataset` (`create_participant`,
`update_participant`, `list_participants`, `get_participant`). Both files
are rewritten wholesale on every change (atomic temp-file-then-rename), since
the participant list is small.

**Privacy is enforced at the schema level**: `Participant`
(`vpsych.data.schemas.Participant`) has `extra="forbid"` — constructing one
with an unrecognized field (e.g. `name`, `email`) raises a
`pydantic.ValidationError` immediately. Only pseudonymous data belongs here.

| Column | Units | Description |
|---|---|---|
| `participant_id` | -- | Pseudonymous ID, `sub-NNNN` (four digits). Allocated by `create_participant` as the lowest unused number (fills gaps left by manual edits, not just `max + 1`). |
| `year_of_birth` | year | Birth year only, never a full date of birth. `n/a` if not recorded. |
| `sex` | -- | Self-reported sex, free-text/short-code. `n/a` if not recorded. |
| `refractive_correction` | -- | Refractive correction worn during testing, e.g. `glasses`, `contacts`, `none`. `n/a` if not recorded. |
| `notes` | -- | Free-text notes. Must not contain identifying information (callers are responsible for keeping this pseudonymous). `n/a` if empty. |

Missing values are written as the literal string `n/a`, matching the
convention used in `_trials.tsv` (see [Trials](#trials)).

## Calibration

`calibration/cal-<sha256>.json` (`vpsych.core.calibration.models.Calibration`,
managed by `vpsych.data.dataset.save_calibration` /
`load_calibration` / `list_calibrations` / `latest_calibration`) holds one
complete, **immutable**, content-addressed calibration record:

- `created_utc`: when the calibration was performed.
- `geometry`: display geometry (`DisplayGeometry` -- resolution, physical
  size, viewing distance, refresh rate).
- `gamma`: luminance/gamma characterization and its grade (`A` = photometer
  measured, `B` = psychophysical estimate, `C` = uncalibrated/sRGB gamma
  assumed).
- `color`: color primary characterization and its grade (`A` = measured
  with a spectroradiometer/colorimeter, `C` = nominal sRGB assumed).
- `environment`: the environment checklist completed at calibration time
  (room lighting, monitor warm-up, night-light/HDR disabled).
- `software_version`, `notes`.

The filename hash is `Calibration.content_hash()`: the SHA-256 digest of
the calibration's canonical (sorted-key) JSON serialization. Two
calibrations with identical field values collapse to the same file;
`save_calibration` is idempotent for identical content. Calibration files
are chmod'd `0o444` (read-only) immediately after being written --
calibrations are never edited in place, only superseded by a new file with
a new hash. `load_calibration` re-verifies the hash on every read
(`vpsych.data.dataset.CalibrationIntegrityError` if the file was tampered
with after being written).

`Calibration.is_stale(max_age_days=30)` flags calibrations older than 30
days (configurable); `vpsych.data.validate` surfaces this as a warning for
every stored calibration, and `vpsych.data.quality.check_calibration_quality`
lets any test's `summarize` fold staleness/grade into its own
`quality_flags`.

Sessions reference their calibration only by hash
(`SessionInfo.calibration_hash`) -- never by embedding the calibration
inline -- so reanalysis always knows exactly which calibration produced a
given session's data, and recalibrating never silently changes the
interpretation of old sessions.

## Sessions

`sub-XXXX/ses-YYYYMMDDTHHMMSS/session.json`
(`vpsych.data.schemas.SessionInfo`) is everything needed to interpret and
reproduce one session, written at session start (`status: "running"`) and
updated in place (atomically) when the session ends:

| Field | Description |
|---|---|
| `session_id` | `ses-YYYYMMDDTHHMMSS`. |
| `participant_id` | The participant this session belongs to (must exist in `participants.tsv`). |
| `plan` | The `SessionPlan` that was run: `participant_id`, ordered/randomized list of `PlannedTest`s (`task_id`, `eye`, `params`, `viewing_distance_cm`), `ordering`, the session-level RNG `seed`, and an optional `calibration_hash` (the specific calibration to use; if omitted, the runner uses the most recently created calibration under `calibration/`). `SessionPlan.participant_id` and `SessionInfo.calibration_hash` (the calibration actually resolved and used) are the source of truth the runner reads to build this `SessionInfo` -- see `vpsych.runner.__main__.load_session_plan`. |
| `calibration_hash` | `Calibration.content_hash()` of the calibration used (must exist under `calibration/`). |
| `display` | Display geometry in effect for this session. |
| `os_info`, `python_version`, `psychopy_version`, `gpu_info` | Environment provenance, free-form. |
| `software_version`, `git_commit` | vpsych version/commit that ran this session. |
| `status` | `"running"` \| `"complete"` \| `"incomplete"` \| `"aborted"`. |
| `started_utc`, `ended_utc` | UTC timestamps (`ended_utc` is `null` while `status == "running"`). |
| `environment` | The environment checklist completed before this session. |

`beh/` holds this session's per-test-run behavioral data files (BIDS
`beh/` = "behavioral" datatype), one quadruple of files per
`(task_id, eye, run)`:

```
sub-0001_ses-20260916T103000_task-acuity_eye-OD_run-1_trials.tsv
sub-0001_ses-20260916T103000_task-acuity_eye-OD_run-1_trials.json   # sidecar
sub-0001_ses-20260916T103000_task-acuity_eye-OD_run-1_summary.json
sub-0001_ses-20260916T103000_task-acuity_eye-OD_run-1_frames.tsv
```

## Trials

`..._trials.tsv` holds one row per trial (`vpsych.core.trial.TrialRecord`),
appended by `vpsych.data.writer.SessionWriter.append_trial` -- **flushed
and `fsync`ed to disk before the call returns**, so a trial is durable the
instant `append_trial` returns, regardless of what happens to the process
immediately afterward (see [Integrity and crash safety](#integrity-and-crash-safety)).

Every column has an entry in the sidecar `..._trials.json`
(`vpsych.data.schemas.ColumnSidecar` per column: `description`, `units`,
and `levels` for categorical columns), written once, the first time a given
`(task_id, eye, run)` trials file is created. `vpsych.data.writer.TRIAL_COLUMN_SIDECARS`
is the canonical column documentation, covering every `TrialRecord` field
(tested to stay in sync in `tests/data/test_writer.py`).

Column dictionary (see `vpsych.core.trial.TrialRecord` for the full
docstrings this table summarizes):

| Column | Units | Description |
|---|---|---|
| `participant_id` | -- | Pseudonymous participant ID. |
| `session_id` | -- | Session identifier. |
| `task_id` | -- | The test's `TestSpec.id` (snake_case). |
| `task_version` | -- | The test's `TestSpec.version` (semver) at the time this trial ran. |
| `run` | -- | 1-based run number. |
| `eye` | -- | `OD` (right) \| `OS` (left) \| `OU` (both/binocular). |
| `block` | -- | `practice` (not analyzed) \| `main`. |
| `trial_index` | -- | 0-based index within this block and run. |
| `is_catch` | -- | Whether this was a suprathreshold catch trial (estimates lapse rate). |
| `intensity` | see `intensity_units` | Scalar stimulus intensity presented. |
| `intensity_units` | -- | Units of `intensity`, e.g. `logMAR`, `log10_contrast`. |
| `stimulus_params` | -- | Full stimulus parameterization (test-specific), JSON-encoded. |
| `correct_response` | -- | The response that would have scored correct. |
| `response` | -- | The observer's actual response, or `n/a` if none given (e.g. timeout). |
| `correct` | -- | Whether `response` was scored correct, or `n/a` if unscored. |
| `rt_s` | s | Response time from stimulus onset to the response event, or `n/a`. |
| `stimulus_onset_s` | s | Stimulus-onset flip timestamp, on the runner's monotonic clock. |
| `n_dropped_frames_trial` | -- | Dropped frames detected during this trial's stimulus presentation. |
| `procedure_state` | -- | JSON-encoded snapshot of the driving adaptive procedure's internal state after this trial. |
| `timestamp_utc` | -- | Wall-clock UTC timestamp this trial was recorded, ISO 8601. |
| `rng_seed` | -- | RNG seed in effect for this trial's stochastic choices. |

**Serialization rules** (`TrialRecord.to_tsv_row`, BIDS-style): missing
scalar values are the literal string `n/a`; `dict`/`list` values
(`stimulus_params`, `procedure_state`, and `response`/`correct_response`
when dict/list-valued) are compact sorted-key JSON, not `str()`; `bool`
values are `"True"`/`"False"`; `datetime` values are ISO 8601; everything
else is `str()`.

`vpsych.data.tsv.read_trials_tsv(path) -> pandas.DataFrame` is the inverse:
it knows every column's type and decodes accordingly (`"n/a"` -> `None`,
JSON dict columns parsed, `int`/`float`/`bool`/`datetime` columns typed).
One caveat: `correct_response`/`response` are typed `Any` on `TrialRecord`,
and `to_tsv_row` only special-cases `None`/dict/list/datetime -- a bare
`bool` written into one of those two columns round-trips as the string
`"True"`/`"False"`, not a `bool` (`json.loads` requires lowercase
`true`/`false`). Use a JSON-native representation (a dict, or a string
label) for anything boolean-like in those two columns.

## Summaries

`..._summary.json` (`vpsych.data.schemas.TestSummary`) is the computed
result of one test run, written atomically by
`vpsych.data.writer.SessionWriter.write_summary`:

| Field | Description |
|---|---|
| `task_id`, `task_version` | Which test/version produced this. |
| `eye`, `run` | Which run this summarizes. |
| `estimate` | `ThresholdEstimate`: `value`, `ci_low`, `ci_high`, `ci_level`, `units`, `method`, `extra`. |
| `fit_params` | Fitted psychometric-function (or CSF-model) parameters, flat dict. |
| `gof` | Goodness-of-fit diagnostics (deviance, df, p-value), flat dict. |
| `quality_flags` | List of `QualityFlag` (`code`, `severity`, `message`) -- see [`vpsych.data.quality`](#quality-flags). |
| `n_trials`, `n_catch` | Main-block trial count and catch-trial count. |
| `catch_lapse_rate` | Proportion of catch trials answered incorrectly, in [0, 1]. |
| `frame_stats` | `FrameTimingStats` for this run, flat dict. |
| `analysis_version` | Version of the *analysis* code that produced this summary (distinct from `task_version` -- `vpsych reanalyze` can recompute a summary with newer analysis code against unchanged raw trials). |

### Quality flags

`vpsych.data.quality` provides pure, reusable `check_*` functions every
test's `summarize` calls into (directly or via `compute_quality_flags`):

- `high_catch_lapse_rate`: catch-trial lapse rate > 10% (`warning`) or >
  20% (`critical`).
- `excess_dropped_frames`: dropped-frame fraction > 1% (`warning`).
- `threshold_at_range_edge`: the threshold estimate is within 5% of either
  end of the tested stimulus range (`warning`).
- `low_calibration_grade` / `stale_calibration`: luminance grade worse than
  the test's minimum, or the calibration is older than 30 days (`warning`).
- `poor_gof`: goodness-of-fit p-value < 0.05 (`warning`).
- `too_few_trials`: fewer main-block trials than expected (`warning`).

## Frame timing

`..._frames.tsv` is a single-column TSV (`frame_interval_s`, one row per
measured inter-flip interval in seconds, typically from
`win.frameIntervals`), appended to by
`vpsych.data.writer.SessionWriter.write_frames`. `vpsych.core.timing.summarize_frame_intervals`
turns a sequence of these into a `FrameTimingStats` (mean/SD interval,
dropped-frame count/fraction against the nominal refresh interval) for
`TestSummary.frame_stats`.

## Integrity and crash safety

**No silent data loss** (see `CONTRIBUTING.md`): `SessionWriter.append_trial`
appends one row and calls `flush()` + `os.fsync()` on the file descriptor
before returning. If the process is killed (`SIGKILL`, power loss, a
segfault -- anything that skips Python's own `__exit__`/`finally` handling)
immediately after `append_trial` returns, that row is already durable on
disk. `tests/data/test_crash_safety.py` verifies this directly: a real
subprocess appends N trials through a `SessionWriter` and then calls
`os._exit(1)`, bypassing all cleanup, and the test confirms all N rows are
present and parseable afterward.

**Atomic writes**: every JSON file (`session.json`, `..._summary.json`,
calibration files, `dataset_description.json`, `participants.json`,
`MANIFEST.sha256`) is written to a temporary file in the same directory,
`fsync`ed, renamed into place with `os.replace` (atomic on POSIX), and the
containing directory is then `fsync`ed too -- so a reader (or a crash
partway through) never observes a half-written file, and a rename that
completes is durable. On any exception during this sequence, the temporary
file is removed and the original destination (if any) is left untouched.

**A session that ends abnormally is marked, never deleted**:
`SessionWriter.__exit__` calls `finalize("complete")` if no exception
propagated, or `finalize("incomplete")` if one did. If the process is
killed before `__exit__` can even run (the crash-safety scenario above),
`session.json` is simply left with `status: "running"` and no
`MANIFEST.sha256` -- `vpsych.data.validate` treats this specific
combination as **informational** (`session_running_or_interrupted`), not
an error: it is exactly what an interrupted-but-not-corrupted session looks
like, and every trial appended before the interruption remains valid.

**Finalization and the manifest**: `SessionWriter.finalize(status)`
(`status` one of `"complete"`, `"incomplete"`, `"aborted"` -- never
`"running"`) writes the final `session.json`, then walks the session
directory and writes `MANIFEST.sha256`: one `<sha256 hex digest><two
spaces><path relative to the session dir>` line per file, in the same
format `sha256sum` produces (verifiable with `sha256sum -c MANIFEST.sha256`
from inside the session directory). By default (`readonly_after_finalize=True`,
constructor argument to `SessionWriter`), every file is then `chmod`'d
`0o444` (read-only) -- raw data is never edited after a session ends. Set
`readonly_after_finalize=False` to disable this (used by some test
fixtures that need to clean up a temp directory afterward).

**Refusing to overwrite a finalized session**: `SessionWriter` checks the
on-disk `session.json` on `__enter__` (and again at the start of
`finalize`) and raises `SessionAlreadyFinalizedError` if a
non-`"running"` status is already recorded for that `(participant_id,
session_id)` -- a finalized session's directory is never silently reused.

**Tamper detection**: `vpsych.data.validate.validate_session`/`validate_dataset`
recompute every file's SHA-256 and compare against `MANIFEST.sha256`,
reporting a `checksum_mismatch` error for any file that was modified after
finalization, a `manifest_file_missing` error for any listed file that's
gone, and an `untracked_file` warning for any file present on disk but not
in the manifest.

## Reanalysis

`vpsych.data.reanalyze.reanalyze_session(session_path, write=False)` never
touches raw data. For every `..._trials.tsv` in a session's `beh/`
directory, it:

1. Parses the trials with `read_trials_tsv`.
2. Looks up the registered test class for `task_id` via
   `vpsych.tests_catalog.base.get_test`.
3. Reconstructs a test instance from the session's recorded `display`
   geometry, the calibration loaded by `session.json`'s
   `calibration_hash`, and the matching `PlannedTest.params` from
   `session.json`'s `plan`.
4. Calls `summarize(trials_df)` and compares the result to the stored
   `..._summary.json` (field-by-field).

If `write=True`, the recomputed summary is written **alongside** the
original as a new file, `..._summary-reanalysis-<analysis_version>.json`
(`<analysis_version>` is the freshly computed summary's own
`analysis_version` field) -- the original `..._summary.json` is never
overwritten or deleted. `vpsych-data reanalyze <session_path> [--write]`
is the CLI entry point; it exits `0` only if every run's recomputed summary
matched the one on disk.

## Export

`vpsych.data.export.export_participant`/`export_dataset` (also
`vpsych-data export <out.zip> [--participant-id ID]`) produce a
self-contained zip:

```
export.zip
  raw/                          # the BIDS-like tree verbatim (dataset_description.json,
                                 # participants.tsv/.json, calibration/, sub-XXXX/...)
  summaries_tidy.csv            # one row per TestSummary, flattened
  data_dictionary.md            # every file type's columns/fields, generated from the
                                 # same ColumnSidecar definitions written into _trials.json
  validation_report.json        # vpsych.data.validate.validate_dataset(), run before export
  json_schemas/*.schema.json    # JSON Schema for every model (from vpsych.data.schemas.export_json_schemas,
                                 # plus TrialRecord and Calibration)
```

`summaries_tidy.csv` columns: `participant_id`, `session_id`, `task_id`,
`task_version`, `eye`, `run`, `value`, `ci_low`, `ci_high`, `ci_level`,
`units`, `method`, `n_trials`, `n_catch`, `catch_lapse_rate`,
`quality_flags` (`;`-joined codes), `analysis_version`.

## Catalog index

`catalog.sqlite` (stdlib `sqlite3` only, `vpsych.data.catalog`) indexes
`participants`, `sessions`, and `test_summaries` tables for the UI's
history/longitudinal views. It is always derivable from the files above:
`rebuild_catalog(root)` wipes and rebuilds it from scratch by walking the
data root; `index_session(participant_id, session_id, root)` incrementally
(re-)indexes one session, so the runner/app can keep it current without a
full rebuild after every session. `sessions_for_participant` and
`task_history(participant_id, task_id, eye)` (ordered oldest-first) are the
query helpers the results/history screens use.

## Schema versioning and migration

`vpsych.data.schemas.SCHEMA_VERSION` (currently `1.0.0`) is the version
recorded in every dataset's `dataset_description.json`. When a breaking
change is made to any schema in this document, `SCHEMA_VERSION` is bumped
and a migration is registered in `vpsych.data.migrations`:

```python
@migrations.register_migration("TestSummary", "1.0.0", "1.1.0")
def _add_some_field(obj: dict) -> dict:
    return {**obj, "some_new_field": "default"}
```

`migrations.migrate(obj, schema_name, from_version, to_version)` walks a
chain of registered edges (breadth-first) to migrate a raw (not yet
model-validated) JSON document between arbitrary versions, so a multi-step
upgrade doesn't need every version pair registered explicitly.
`from_version == to_version` is always the identity transform. As of
`SCHEMA_VERSION = "1.0.0"` there is exactly one version and nothing to
migrate from; `tests/data/test_migrations.py` demonstrates the framework
end to end with a mock schema and migration.

## Validation

`vpsych.data.validate.validate_session`/`validate_dataset` (also
`vpsych-data validate [--participant-id ID [--session-id ID]]`) never raise
on data problems -- they return a `ValidationReport` (`.errors`,
`.warnings`, `.ok`) covering:

- Schema validation of every JSON file (`dataset_description.json`,
  `session.json`, `..._summary.json`, `participants.tsv` rows).
- `_trials.tsv` columns matching their `_trials.json` sidecar exactly.
- `MANIFEST.sha256` checksums (see
  [Integrity and crash safety](#integrity-and-crash-safety)).
- `session.json`'s `participant_id` existing in `participants.tsv`.
- `session.json`'s `calibration_hash` existing under `calibration/`.
- Calibration staleness (warning).
- A still-`"running"` session with no manifest yet: reported as `info`
  (`session_running_or_interrupted`), not an error -- see
  [Integrity and crash safety](#integrity-and-crash-safety).

## Privacy

- Participant identifiers are `sub-NNNN` only, everywhere (filenames, every
  JSON/TSV field) -- never a name, email, or other direct identifier.
- `vpsych.data.schemas.Participant` uses `extra="forbid"`: constructing one
  with any field beyond `participant_id`, `year_of_birth`, `sex`,
  `refractive_correction`, `notes` raises immediately, so an accidental
  `name=` or `email=` fails loudly instead of silently ending up in
  `participants.tsv`.
- `year_of_birth` is a birth year only, never a full date of birth.
- Free-text fields (`Participant.notes`, `Calibration.notes`) are the
  caller's responsibility to keep pseudonymous -- nothing in the schema
  can enforce the *content* of a free-text field, so no PII should ever be
  typed into one.
- This repository's own `.gitignore` excludes data roots by default; a
  vpsych data root should never be committed to a public (or private) git
  repository.
