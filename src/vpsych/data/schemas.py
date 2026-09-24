"""Pydantic schemas for every JSON/TSV file vpsych writes to a data root.

These models are the source of truth for the on-disk data format described
in `docs/DATA_FORMAT.md` (BIDS-behavioral / Psych-DS inspired). JSON Schemas
are generated from them via :func:`export_json_schemas` so external tools
can validate a dataset without importing Python. `SCHEMA_VERSION` is bumped
whenever a breaking change is made to any of these models; migrations
between versions live alongside this module in a later phase.

No identifying fields (names, emails, free-text that could contain them)
belong in any of these models -- participants are `sub-XXXX` only.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from vpsych.core.calibration.models import EnvironmentChecklist
from vpsych.core.display import DisplayGeometry
from vpsych.core.procedures.base import ThresholdEstimate

SCHEMA_VERSION = "1.0.0"

Eye = Literal["OD", "OS", "OU"]
Ordering = Literal["fixed", "randomized"]
SessionStatus = Literal["running", "complete", "incomplete", "aborted"]
Severity = Literal["info", "warning", "critical"]


class DatasetDescription(BaseModel):
    """Top-level `dataset_description.json`: identifies and versions a data root.

    Attributes:
        schema_version: Version of the vpsych data schema this dataset was
            written with (see `SCHEMA_VERSION`).
        software_version: `vpsych` package version that created/last wrote
            to this dataset.
        git_commit: Git commit hash of the `vpsych` checkout that
            created/last wrote to this dataset, if known.
        name: Human-readable dataset name.
        created_utc: UTC timestamp the dataset root was created.
    """

    model_config = ConfigDict(frozen=True)

    schema_version: str = Field(description="vpsych data schema version.")
    software_version: str = Field(description="vpsych package version.")
    git_commit: str | None = Field(default=None, description="Git commit hash, if known.")
    name: str = Field(description="Human-readable dataset name.")
    created_utc: datetime = Field(description="UTC timestamp the dataset root was created.")


class Participant(BaseModel):
    """One row of `participants.tsv` / one entry in `participants.json`.

    Pseudonymous only: no names, emails, or other direct identifiers.

    Attributes:
        participant_id: Pseudonymous ID, `sub-XXXX` (four digits).
        year_of_birth: Birth year only (not full date of birth), or `None`
            if not recorded.
        sex: Self-reported sex, free-text/short-code, or `None` if not
            recorded.
        refractive_correction: Refractive correction worn during testing
            (e.g. `"glasses"`, `"contacts"`, `"none"`), or `None` if not
            recorded.
        notes: Free-text notes. Must not contain identifying information;
            callers are responsible for keeping this pseudonymous.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    participant_id: str = Field(
        pattern=r"^sub-[0-9]{4}$", description="Pseudonymous participant ID."
    )
    year_of_birth: int | None = Field(
        default=None, ge=1900, le=2100, description="Birth year only."
    )
    sex: str | None = Field(default=None, description="Self-reported sex.")
    refractive_correction: str | None = Field(
        default=None, description="Refractive correction worn during testing."
    )
    notes: str | None = Field(default=None, description="Free-text notes (must stay pseudonymous).")


class PlannedTest(BaseModel):
    """One test entry within a :class:`SessionPlan`.

    Attributes:
        task_id: The `TestSpec.id` of the test to run.
        eye: Eye condition to test under.
        params: Test-specific parameters, matching the test's
            `TestSpec.params_model` when validated.
        viewing_distance_cm: Viewing distance to use for this test, in
            centimetres (may differ per test within one session, e.g. near
            vs. distance acuity).
    """

    model_config = ConfigDict(frozen=True)

    task_id: str = Field(description="TestSpec.id of the test to run.")
    eye: Eye = Field(description="Eye condition to test under.")
    params: dict[str, Any] = Field(default_factory=dict, description="Test-specific parameters.")
    viewing_distance_cm: float = Field(gt=0, description="Viewing distance for this test, in cm.")


class SessionPlan(BaseModel):
    """An ordered set of tests to run in one session, decided before the run starts.

    Attributes:
        participant_id: Pseudonymous ID, `sub-XXXX`, of the participant this
            plan is for.
        tests: The planned tests, in the order given (subject to
            `ordering`).
        ordering: `"fixed"` (run `tests` in the listed order) or
            `"randomized"` (shuffle order using `seed` at session start).
        seed: RNG seed used to determine trial/test order when
            `ordering == "randomized"`, or to seed any other
            session-level randomization; always logged even when
            `ordering == "fixed"` for reproducibility of within-test
            randomization.
        calibration_hash: `Calibration.content_hash()` of the specific
            calibration to use for this session, or `None` to use the most
            recently created calibration under the data root's
            `calibration/` directory.
    """

    model_config = ConfigDict(frozen=True)

    participant_id: str = Field(
        pattern=r"^sub-[0-9]{4}$", description="Pseudonymous participant ID this plan is for."
    )
    tests: list[PlannedTest] = Field(description="Planned tests, in listed order.")
    ordering: Ordering = Field(description="fixed or randomized.")
    seed: int = Field(description="RNG seed for session-level randomization.")
    calibration_hash: str | None = Field(
        default=None,
        description="Calibration.content_hash() to use, or None for the most recent calibration.",
    )


class QualityFlag(BaseModel):
    """One machine-readable quality concern about a test run's data.

    Attributes:
        code: Short stable identifier, e.g. `"high_catch_lapse_rate"`,
            `"excess_dropped_frames"`, `"threshold_at_range_edge"`,
            `"stale_calibration"`, `"poor_gof"`.
        severity: How serious this concern is.
        message: Human-readable explanation, suitable for display in the
            results UI.
    """

    model_config = ConfigDict(frozen=True)

    code: str = Field(description="Short stable identifier for this flag.")
    severity: Severity = Field(description="info, warning, or critical.")
    message: str = Field(description="Human-readable explanation.")


class TestSummary(BaseModel):
    """Computed summary for one test run: `..._summary.json`.

    Attributes:
        task_id: The `TestSpec.id` this summary is for.
        task_version: The `TestSpec.version` this run used.
        eye: Eye condition tested.
        run: 1-based run number.
        estimate: The final threshold estimate and CI.
        fit_params: Fitted psychometric-function (or CSF-model) parameters,
            as a flat JSON-serializable dict, for tests that fit one.
        gof: Goodness-of-fit diagnostics, as a flat JSON-serializable dict
            (e.g. deviance, df, p_value from
            `vpsych.core.psychometric.deviance_gof`).
        quality_flags: Any quality concerns raised about this run.
        n_trials: Total number of main-block trials.
        n_catch: Total number of catch trials.
        catch_lapse_rate: Proportion of catch trials answered incorrectly,
            in [0, 1].
        frame_stats: Frame-timing summary for this run (see
            `vpsych.core.timing.FrameTimingStats`), as a flat
            JSON-serializable dict.
        analysis_version: Version of the analysis code that produced this
            summary (distinct from `task_version`, since `vpsych reanalyze`
            can recompute a summary with newer analysis code against
            unchanged raw trial data).
    """

    __test__ = False  # not a pytest test class, despite the name

    model_config = ConfigDict(frozen=True)

    task_id: str = Field(description="TestSpec.id this summary is for.")
    task_version: str = Field(description="TestSpec.version this run used.")
    eye: Eye = Field(description="Eye condition tested.")
    run: int = Field(ge=1, description="1-based run number.")
    estimate: ThresholdEstimate = Field(description="Final threshold estimate and CI.")
    fit_params: dict[str, Any] = Field(default_factory=dict, description="Fitted model parameters.")
    gof: dict[str, Any] = Field(default_factory=dict, description="Goodness-of-fit diagnostics.")
    quality_flags: list[QualityFlag] = Field(default_factory=list, description="Quality concerns.")
    n_trials: int = Field(ge=0, description="Total number of main-block trials.")
    n_catch: int = Field(ge=0, description="Total number of catch trials.")
    catch_lapse_rate: float = Field(ge=0, le=1, description="Proportion of catch trials missed.")
    frame_stats: dict[str, Any] = Field(default_factory=dict, description="Frame-timing summary.")
    analysis_version: str = Field(
        description="Version of the analysis code that produced this summary."
    )


class SessionInfo(BaseModel):
    """`session.json`: everything needed to interpret and reproduce one session.

    Attributes:
        session_id: Session identifier, `ses-YYYYMMDDTHHMMSS`.
        participant_id: The participant this session belongs to.
        plan: The session plan that was run.
        calibration_hash: `Calibration.content_hash()` of the calibration
            used for this session.
        display: Display geometry in effect for this session.
        os_info: Free-form OS identification string (e.g.
            `platform.platform()`).
            `platform.platform()`).
        python_version: Python version string that ran this session.
        psychopy_version: PsychoPy version string that ran this session.
        gpu_info: Free-form GPU identification string, if available.
        software_version: `vpsych` package version that ran this session.
        git_commit: Git commit hash of the `vpsych` checkout that ran this
            session, if known.
        status: Current/final status of this session.
        started_utc: UTC timestamp the session started.
        ended_utc: UTC timestamp the session ended, or `None` if still
            running.
        environment: Environment checklist completed for this session.
    """

    model_config = ConfigDict(frozen=True)

    session_id: str = Field(
        pattern=r"^ses-\d{8}T\d{6}$", description="Session identifier, ses-YYYYMMDDTHHMMSS."
    )
    participant_id: str = Field(
        pattern=r"^sub-[0-9]{4}$", description="Participant this session belongs to."
    )
    plan: SessionPlan = Field(description="The session plan that was run.")
    calibration_hash: str = Field(description="Calibration.content_hash() used for this session.")
    display: DisplayGeometry = Field(description="Display geometry in effect for this session.")
    os_info: str = Field(description="Free-form OS identification string.")
    python_version: str = Field(description="Python version string.")
    psychopy_version: str = Field(description="PsychoPy version string.")
    gpu_info: str | None = Field(default=None, description="Free-form GPU identification string.")
    software_version: str = Field(description="vpsych package version.")
    git_commit: str | None = Field(default=None, description="Git commit hash, if known.")
    status: SessionStatus = Field(description="Current/final status of this session.")
    started_utc: datetime = Field(description="UTC timestamp the session started.")
    ended_utc: datetime | None = Field(default=None, description="UTC timestamp the session ended.")
    environment: EnvironmentChecklist = Field(
        description="Environment checklist completed for this session."
    )

    @field_validator("started_utc", "ended_utc")
    @classmethod
    def _require_timezone_aware(cls, v: datetime | None) -> datetime | None:
        if v is not None and v.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware (UTC)")
        return v


class ColumnSidecar(BaseModel):
    """Documentation for one column of a `_trials.tsv` file, BIDS-sidecar style.

    A `_trials.json` sidecar is a mapping of column name to `ColumnSidecar`
    (or a compatible plain dict), describing every column of
    `vpsych.core.trial.TrialRecord` written to the corresponding
    `_trials.tsv`.

    Attributes:
        description: Human-readable description of what this column holds.
        units: Units of this column's values, e.g. `"s"`, `"deg"`,
            `"log10_contrast"`, or `None` for unitless/categorical columns.
        levels: For categorical columns, a mapping from each possible value
            to a human-readable description of that value, or `None` for
            non-categorical columns.
    """

    model_config = ConfigDict(frozen=True)

    description: str = Field(description="Human-readable description of this column.")
    units: str | None = Field(default=None, description="Units of this column's values.")
    levels: dict[str, str] | None = Field(
        default=None, description="For categorical columns, value -> description."
    )


def export_json_schemas(directory: str) -> list[str]:
    """Write JSON Schema files for every top-level model in this module.

    Args:
        directory: Directory to write `<ModelName>.schema.json` files into
            (created if it does not exist).

    Returns:
        The list of file paths written.
    """
    import json
    import os

    models: list[type[BaseModel]] = [
        DatasetDescription,
        Participant,
        PlannedTest,
        SessionPlan,
        QualityFlag,
        TestSummary,
        SessionInfo,
        ColumnSidecar,
    ]
    os.makedirs(directory, exist_ok=True)
    written: list[str] = []
    for model in models:
        path = os.path.join(directory, f"{model.__name__}.schema.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(model.model_json_schema(), f, indent=2, sort_keys=True)
        written.append(path)
    return written
