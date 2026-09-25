"""Read a `_trials.tsv` file back into a typed `pandas.DataFrame`.

`read_trials_tsv` is the inverse of `vpsych.core.trial.TrialRecord.to_tsv_row`:
it knows exactly which columns `TrialRecord` writes and how each one was
serialized, and undoes that serialization column-by-column rather than
guessing generically from the file contents.

Round-trip caveat: `correct_response` and `response` are typed `Any` on
`TrialRecord` and `to_tsv_row` only special-cases `None` (-> `"n/a"`),
`dict`/`list` (-> JSON) and `datetime`; every other value (including
`bool`) is written with plain `str()`. On the way back, this module tries
`json.loads` on those two columns so JSON-native values (numbers, `null`,
`dict`, `list`) and plain strings round-trip exactly, but a Python `bool`
written into one of those columns comes back as the string `"True"`/
`"False"`, not a `bool` -- `json.loads` requires lowercase `true`/`false`
and `str(True)` is `"True"`. Tests should not put bare booleans in
`correct_response`/`response`; use a JSON-native representation (e.g. a
dict, or a string label) if a boolean-like response needs to round-trip.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

_INT_COLUMNS = ("run", "trial_index", "n_dropped_frames_trial", "rng_seed")
_BOOL_COLUMNS = ("is_catch",)
_NULLABLE_BOOL_COLUMNS = ("correct",)
_FLOAT_COLUMNS = ("intensity", "stimulus_onset_s")
_NULLABLE_FLOAT_COLUMNS = ("rt_s",)
_JSON_DICT_COLUMNS = ("stimulus_params", "procedure_state")
_JSON_OR_STR_COLUMNS = ("correct_response", "response")
_DATETIME_COLUMNS = ("timestamp_utc",)


def _parse_bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"Cannot parse {value!r} as a bool (expected 'True' or 'False').")


def _parse_json_or_str(value: str) -> Any:
    if value == "n/a":
        return None
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return value


def read_trials_tsv(path: str | Path) -> pd.DataFrame:
    """Read a `TrialRecord`-derived `_trials.tsv` file into a typed DataFrame.

    Every column `TrialRecord.to_tsv_row` writes is decoded back to its
    Python type: `"n/a"` becomes `None`/`pandas.NA`, JSON-encoded dict
    columns (`stimulus_params`, `procedure_state`) are parsed with
    `json.loads`, integer/float/bool/datetime columns are parsed by type,
    and `correct_response`/`response` are JSON-decoded where possible (see
    the module docstring for the one known round-trip caveat). Unknown
    extra columns (not part of `TrialRecord`) are left as raw strings.

    Args:
        path: Path to the `_trials.tsv` file.

    Returns:
        A `pandas.DataFrame` with one row per trial, columns typed as
        described above.
    """
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_values=[])

    for col in _INT_COLUMNS:
        if col in df.columns:
            # Python ints, object dtype: rng_seed is a 128-bit value that overflows int64.
            df[col] = df[col].map(int).astype("object" if col == "rng_seed" else int)

    for col in _BOOL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].map(_parse_bool)

    for col in _NULLABLE_BOOL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].map(lambda v: None if v == "n/a" else _parse_bool(v))
            # pandas' nullable "boolean" extension dtype, not "object": an
            # object-dtype column of plain Python True/False/None applies `~`
            # elementwise as Python's bitwise-invert operator (~True == -2,
            # ~False == -1), not logical negation -- a landmine for any
            # summarize() that writes the natural `~trials["correct"]` to get
            # "incorrect", silently producing nonsense instead of an error
            # (found via tests/integration/test_end_to_end.py). The nullable
            # "boolean" dtype supports `~` (and `.mean()`/`.sum()`) correctly,
            # propagating NA as NA.
            df[col] = df[col].astype("boolean")

    for col in _FLOAT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(float)

    for col in _NULLABLE_FLOAT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].map(lambda v: None if v == "n/a" else float(v))
            df[col] = df[col].astype("object")

    for col in _JSON_DICT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].map(lambda v: {} if v == "n/a" else json.loads(v))

    for col in _JSON_OR_STR_COLUMNS:
        if col in df.columns:
            df[col] = df[col].map(_parse_json_or_str)

    for col in _DATETIME_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True)

    return df
