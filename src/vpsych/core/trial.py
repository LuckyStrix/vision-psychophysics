"""Generic trial state machine types and the canonical per-trial data record.

Every test follows the same trial phase sequence (fixation -> stimulus ->
response [-> ITI]), timed entirely in frames per the project rule that
*all stimulus durations are specified in frames, never seconds*. This module
defines that shared timeline and the single canonical row schema
(`TrialRecord`) that every test writes to its `_trials.tsv` file, so the
data layer, quality-flagging, and reanalysis code work uniformly across
every test in the catalog.
"""

from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Eye = Literal["OD", "OS", "OU"]
Block = Literal["practice", "main"]


class TrialPhase(Enum):
    """Phases of the generic per-trial timeline, in presentation order."""

    FIXATION = "fixation"
    """Fixation cross/point shown before the stimulus, to stabilize gaze."""

    STIMULUS = "stimulus"
    """The test stimulus is on screen."""

    RESPONSE = "response"
    """Awaiting the observer's response (may overlap STIMULUS if the
    stimulus remains visible until response, per-test)."""

    ITI = "iti"
    """Inter-trial interval: blank/neutral screen between trials."""


class TrialTimeline(BaseModel):
    """Frame-count durations for one trial's phases.

    All durations are in whole display refresh frames (never seconds/ms) so
    presentation timing is locked to the display's actual refresh cycle;
    convert a design duration in milliseconds with
    `vpsych.core.display.DisplayGeometry.frames_for_ms` when constructing
    one of these.

    Attributes:
        fixation_frames: Duration of the fixation phase, in frames.
        stimulus_frames: Duration of the stimulus phase, in frames.
        response_timeout_frames: Maximum duration to wait for a response
            after stimulus onset, in frames, or `None` for no timeout
            (waits indefinitely).
        iti_frames: Duration of the inter-trial interval, in frames.
    """

    model_config = ConfigDict(frozen=True)

    fixation_frames: int = Field(ge=0, description="Fixation phase duration, in frames.")
    stimulus_frames: int = Field(ge=1, description="Stimulus phase duration, in frames.")
    response_timeout_frames: int | None = Field(
        default=None, ge=1, description="Max frames to wait for a response, or None for no timeout."
    )
    iti_frames: int = Field(ge=0, description="Inter-trial interval duration, in frames.")


def _json_or_default(value: Any, default: str) -> str:
    """Serialize `value` as compact JSON, or return `default` if `value` is None."""
    if value is None:
        return default
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


class TrialRecord(BaseModel):
    """The canonical per-trial data row, common to every test in the catalog.

    One `TrialRecord` is appended to a session's
    `..._trials.tsv` file per trial (see
    `vpsych.data.writer.SessionWriter.append_trial`); the file's `.json`
    sidecar documents every column's units and levels (see
    `vpsych.data.schemas.ColumnSidecar`). Every field with a physical
    quantity carries its units in the name (`_s`, `_px`, etc.) per project
    convention.

    Attributes:
        participant_id: Pseudonymous participant ID, `sub-XXXX` (never a
            name or other direct identifier).
        session_id: Session identifier, `ses-YYYYMMDDTHHMMSS` (see
            `vpsych.data.schemas.SessionInfo`).
        task_id: The test's `TestSpec.id` (snake_case), e.g. `"acuity"`.
        task_version: The test's `TestSpec.version` (semver) at the time
            this trial was run.
        run: 1-based run number, for tests run more than once per session
            (e.g. per-eye repeats).
        eye: Eye tested: `"OD"` (right), `"OS"` (left), or `"OU"`
            (both/binocular).
        block: `"practice"` (not analyzed) or `"main"`.
        trial_index: 0-based index of this trial within its block and run.
        is_catch: Whether this was a suprathreshold catch trial (used to
            estimate lapse rate, not to drive the adaptive procedure's
            threshold estimate).
        intensity: The scalar intensity presented, in `intensity_units`, for
            single-parameter procedures. For multi-parameter procedures
            (e.g. qCSF) the corresponding scalar summary dimension may be
            used, with the full stimulus in `stimulus_params`; test authors
            document which.
        intensity_units: Units of `intensity`, e.g. `"logMAR"`,
            `"log10_contrast"`.
        stimulus_params: Full stimulus parameterization for this trial
            (test-specific keys and values), e.g. orientation, spatial
            frequency, direction. JSON-encoded in the TSV (see
            `to_tsv_row`).
        correct_response: The response that would have been scored correct
            for this stimulus (test-specific representation, e.g. a
            direction label or orientation index).
        response: The observer's actual response (same representation as
            `correct_response`), or `None` if no response was given (e.g.
            a response timeout).
        correct: Whether `response` was scored correct, or `None` if
            unscored (e.g. no response).
        rt_s: Response time, in seconds, measured from stimulus onset
            (the flip that made the stimulus visible) to the response
            event's hardware timestamp, or `None` if no response was given.
        stimulus_onset_s: Timestamp of the stimulus-onset flip, in seconds,
            on the runner's monotonic clock (not wall-clock; used only to
            compute `rt_s` and relative timing within a run).
        n_dropped_frames_trial: Number of dropped frames detected during
            this trial's stimulus presentation (see
            `vpsych.core.timing.summarize_frame_intervals`).
        procedure_state: JSON-serializable snapshot of the driving adaptive
            procedure's internal state immediately after this trial (see
            `vpsych.core.procedures.base.AdaptiveProcedure.state_dict`).
            JSON-encoded in the TSV (see `to_tsv_row`).
        timestamp_utc: Wall-clock UTC timestamp when this trial was
            recorded.
        rng_seed: The RNG seed in effect for this trial's stochastic
            choices (see `vpsych.core.rng.make_rng`); normally constant
            across a session's trials (the session-level seed) but recorded
            per trial for self-contained rows.
    """

    model_config = ConfigDict(frozen=True)

    participant_id: str = Field(
        pattern=r"^sub-[0-9]{4}$", description="Pseudonymous participant ID."
    )
    session_id: str = Field(description="Session identifier, ses-YYYYMMDDTHHMMSS.")
    task_id: str = Field(description="Test spec id (snake_case).")
    task_version: str = Field(description="Test spec semver at the time this trial was run.")
    run: int = Field(ge=1, description="1-based run number.")
    eye: Eye = Field(description="Eye tested: OD, OS, or OU.")
    block: Block = Field(description="practice (not analyzed) or main.")
    trial_index: int = Field(ge=0, description="0-based trial index within block and run.")
    is_catch: bool = Field(description="Whether this was a suprathreshold catch trial.")
    intensity: float = Field(description="Scalar intensity presented, in intensity_units.")
    intensity_units: str = Field(description="Units of intensity, e.g. logMAR, log10_contrast.")
    stimulus_params: dict[str, Any] = Field(
        default_factory=dict, description="Full stimulus parameterization (test-specific)."
    )
    correct_response: Any = Field(description="The response that would have been scored correct.")
    response: Any = Field(default=None, description="The observer's actual response, or None.")
    correct: bool | None = Field(default=None, description="Whether response was scored correct.")
    rt_s: float | None = Field(
        default=None, ge=0, description="Response time in seconds from stimulus onset."
    )
    stimulus_onset_s: float = Field(
        description="Stimulus-onset flip time, in seconds (monotonic clock)."
    )
    n_dropped_frames_trial: int = Field(
        ge=0, description="Dropped frames detected during this trial."
    )
    procedure_state: dict[str, Any] = Field(
        default_factory=dict, description="Snapshot of procedure internal state after this trial."
    )
    timestamp_utc: datetime = Field(description="Wall-clock UTC timestamp this trial was recorded.")
    rng_seed: int = Field(description="RNG seed in effect for this trial.")

    @field_validator("timestamp_utc")
    @classmethod
    def _require_timezone_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("timestamp_utc must be timezone-aware (UTC)")
        return v

    def to_tsv_row(self) -> dict[str, str]:
        """Serialize this record to a flat dict of strings for one TSV row.

        Serialization rules (BIDS-style):
        - Missing/`None` scalar values become the literal string `"n/a"`.
        - `dict`/`list` values (`stimulus_params`, `procedure_state`, and
          `response`/`correct_response` when they are dict/list-valued) are
          JSON-encoded (compact, sorted keys) rather than stringified with
          `str()`, so they round-trip exactly via `json.loads`.
        - `bool` values are written as the literal strings `"True"`/
          `"False"` (via `str()`), consistent with pandas' default TSV
          round-trip.
        - `datetime` values are written in ISO 8601.
        - All other scalars are written via `str()`.

        Returns:
            A dict mapping column name to its string representation, ready
            to be written as one row by a TSV writer (e.g. `csv.DictWriter`
            with `delimiter="\\t"`).
        """
        row: dict[str, str] = {}
        for name, value in self.model_dump(mode="python").items():
            if value is None:
                row[name] = "n/a"
            elif isinstance(value, dict | list):
                row[name] = _json_or_default(value, "n/a")
            elif isinstance(value, datetime):
                row[name] = value.isoformat()
            else:
                row[name] = str(value)
        return row
