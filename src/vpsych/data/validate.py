"""Validate a session or a whole dataset: schema, checksums, and cross-references.

`validate_session` and `validate_dataset` never raise on data problems --
they return a structured `ValidationReport` of errors/warnings so callers
(the CLI, the UI's Data screen, tests) can decide what to do. The one thing
this module treats as expected, not corrupt, is a session whose
`session.json` still says `status: "running"` with no `MANIFEST.sha256`:
that is exactly what a session killed mid-run (e.g. `os._exit` in a crash
test) looks like, and `vpsych.data.writer.SessionWriter` guarantees every
trial appended before the crash is durably on disk -- so it is reported as
an informational note, not an error.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from vpsych.data import dataset, paths
from vpsych.data.schemas import DatasetDescription, SessionInfo, TestSummary
from vpsych.data.tsv import read_trials_tsv
from vpsych.data.writer import _sha256_file

Severity = Literal["error", "warning", "info"]


class ValidationIssue(BaseModel):
    """One finding from validation.

    Attributes:
        severity: `"error"` (the dataset/session is invalid or corrupt),
            `"warning"` (valid but worth a human's attention), or `"info"`
            (expected, not a problem -- e.g. a still-running session).
        code: Short stable identifier, e.g. `"checksum_mismatch"`.
        message: Human-readable description.
        path: Path (relative to the thing being validated, or absolute)
            the issue concerns, if applicable.
    """

    model_config = ConfigDict(frozen=True)

    severity: Severity = Field(description="error, warning, or info.")
    code: str = Field(description="Short stable identifier for this issue.")
    message: str = Field(description="Human-readable description.")
    path: str | None = Field(default=None, description="Path this issue concerns, if applicable.")


class ValidationReport(BaseModel):
    """Structured result of a validation run.

    Attributes:
        issues: Every finding, in the order discovered.
    """

    model_config = ConfigDict(frozen=True)

    issues: list[ValidationIssue] = Field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        """Every `severity == "error"` issue."""
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        """Every `severity == "warning"` issue."""
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def ok(self) -> bool:
        """`True` if there are no `error`-severity issues (warnings/info are fine)."""
        return len(self.errors) == 0

    def __add__(self, other: ValidationReport) -> ValidationReport:
        return ValidationReport(issues=[*self.issues, *other.issues])


def _issue(severity: Severity, code: str, message: str, path: str | None = None) -> ValidationIssue:
    return ValidationIssue(severity=severity, code=code, message=message, path=path)


def validate_session(
    participant_id: str, session_id: str, root: Path | None = None
) -> ValidationReport:
    """Validate one session: schema, manifest checksums, sidecar/column agreement, references.

    Args:
        participant_id: The session's participant, `sub-XXXX`.
        session_id: The session ID, `ses-YYYYMMDDTHHMMSS`.
        root: Data root, or `None` to use `vpsych.data.paths.data_root()`.

    Returns:
        A `ValidationReport` covering this session only.
    """
    root = root or paths.data_root()
    issues: list[ValidationIssue] = []
    session_dir = paths.session_dir(participant_id, session_id, root)
    session_json_path = paths.session_json_path(participant_id, session_id, root)
    rel_session = f"{participant_id}/{session_id}"

    if not session_json_path.exists():
        issues.append(
            _issue("error", "missing_session_json", "session.json is missing.", rel_session)
        )
        return ValidationReport(issues=issues)

    session_info: SessionInfo | None = None
    try:
        session_info = SessionInfo.model_validate_json(
            session_json_path.read_text(encoding="utf-8")
        )
    except (ValidationError, json.JSONDecodeError) as exc:
        issues.append(
            _issue(
                "error",
                "invalid_session_json",
                f"session.json failed schema validation: {exc}",
                rel_session,
            )
        )

    manifest_path = paths.session_manifest_path(participant_id, session_id, root)

    if session_info is not None and session_info.status == "running" and not manifest_path.exists():
        issues.append(
            _issue(
                "info",
                "session_running_or_interrupted",
                "Session status is 'running' and no MANIFEST.sha256 exists yet -- either still "
                "in progress, or interrupted before finalize() ran. Trials appended before an "
                "interruption remain valid and durable; this is not corruption.",
                rel_session,
            )
        )
    elif manifest_path.exists():
        manifest_text = manifest_path.read_text(encoding="utf-8")
        listed: dict[str, str] = {}
        for line in manifest_text.splitlines():
            if not line.strip():
                continue
            digest, _, rel_path = line.partition("  ")
            listed[rel_path] = digest
        on_disk = {
            p.relative_to(session_dir).as_posix()
            for p in session_dir.rglob("*")
            if p.is_file() and p != manifest_path
        }
        for rel_path, expected_digest in listed.items():
            fpath = session_dir / rel_path
            if not fpath.exists():
                issues.append(
                    _issue(
                        "error",
                        "manifest_file_missing",
                        f"MANIFEST.sha256 lists {rel_path!r} but it is missing on disk.",
                        f"{rel_session}/{rel_path}",
                    )
                )
                continue
            actual_digest = _sha256_file(fpath)
            if actual_digest != expected_digest:
                issues.append(
                    _issue(
                        "error",
                        "checksum_mismatch",
                        f"{rel_path} does not match its checksum in MANIFEST.sha256 "
                        "(file was modified after the session was finalized).",
                        f"{rel_session}/{rel_path}",
                    )
                )
        for extra_file in on_disk - listed.keys():
            issues.append(
                _issue(
                    "warning",
                    "untracked_file",
                    f"{extra_file} exists on disk but is not listed in MANIFEST.sha256.",
                    f"{rel_session}/{extra_file}",
                )
            )

    beh_dir = paths.beh_dir(participant_id, session_id, root)
    if beh_dir.exists():
        for trials_path in sorted(beh_dir.glob("*_trials.tsv")):
            sidecar_path = trials_path.with_suffix(".json")
            rel_trials = f"{rel_session}/beh/{trials_path.name}"
            try:
                header = trials_path.read_text(encoding="utf-8").splitlines()[0]
                columns = header.split("\t")
            except IndexError:
                issues.append(
                    _issue("error", "empty_trials_tsv", "File has no header row.", rel_trials)
                )
                continue
            if not sidecar_path.exists():
                issues.append(
                    _issue(
                        "error",
                        "missing_trials_sidecar",
                        f"{sidecar_path.name} is missing for {trials_path.name}.",
                        rel_trials,
                    )
                )
            else:
                try:
                    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError as exc:
                    issues.append(
                        _issue(
                            "error",
                            "invalid_sidecar_json",
                            f"Sidecar is not valid JSON: {exc}",
                            rel_trials,
                        )
                    )
                    sidecar = {}
                missing = [c for c in columns if c not in sidecar]
                extra = [c for c in sidecar if c not in columns]
                if missing:
                    issues.append(
                        _issue(
                            "error",
                            "sidecar_column_mismatch",
                            f"Columns {missing} have no entry in the _trials.json sidecar.",
                            rel_trials,
                        )
                    )
                if extra:
                    issues.append(
                        _issue(
                            "warning",
                            "sidecar_extra_column",
                            f"Sidecar documents columns {extra} not present in the TSV.",
                            rel_trials,
                        )
                    )
            try:
                read_trials_tsv(trials_path)
            except Exception as exc:
                issues.append(
                    _issue(
                        "error",
                        "unparsable_trials_tsv",
                        f"Could not parse trials TSV: {exc}",
                        rel_trials,
                    )
                )

        for summary_path in sorted(beh_dir.glob("*_summary*.json")):
            rel_summary = f"{rel_session}/beh/{summary_path.name}"
            try:
                TestSummary.model_validate_json(summary_path.read_text(encoding="utf-8"))
            except (ValidationError, json.JSONDecodeError) as exc:
                issues.append(
                    _issue(
                        "error",
                        "invalid_summary_json",
                        f"Summary failed schema validation: {exc}",
                        rel_summary,
                    )
                )

    if session_info is not None:
        participant = dataset.get_participant(session_info.participant_id, root)
        if participant is None:
            issues.append(
                _issue(
                    "error",
                    "unknown_participant",
                    f"session.json references participant {session_info.participant_id!r}, "
                    "which is not in participants.tsv.",
                    rel_session,
                )
            )
        cal_path = paths.calibration_path(session_info.calibration_hash, root)
        if not cal_path.exists():
            issues.append(
                _issue(
                    "error",
                    "unknown_calibration",
                    f"session.json references calibration hash "
                    f"{session_info.calibration_hash!r}, which has no file in calibration/.",
                    rel_session,
                )
            )

    return ValidationReport(issues=issues)


def validate_dataset(root: Path | None = None) -> ValidationReport:
    """Validate an entire data root: dataset description, participants, calibrations, sessions.

    Args:
        root: Data root, or `None` to use `vpsych.data.paths.data_root()`.

    Returns:
        A `ValidationReport` covering the whole dataset.
    """
    root = root or paths.data_root()
    issues: list[ValidationIssue] = []

    desc_path = paths.dataset_description_path(root)
    if not desc_path.exists():
        issues.append(
            _issue("error", "missing_dataset_description", "dataset_description.json is missing.")
        )
    else:
        try:
            DatasetDescription.model_validate_json(desc_path.read_text(encoding="utf-8"))
        except (ValidationError, json.JSONDecodeError) as exc:
            issues.append(
                _issue("error", "invalid_dataset_description", f"Failed schema validation: {exc}")
            )

    participants_tsv = paths.participants_tsv_path(root)
    if participants_tsv.exists():
        try:
            participants = dataset.list_participants(root)
        except (ValidationError, ValueError) as exc:
            issues.append(_issue("error", "invalid_participants_tsv", f"{exc}"))
            participants = []
        seen: set[str] = set()
        for p in participants:
            if p.participant_id in seen:
                issues.append(
                    _issue(
                        "error",
                        "duplicate_participant_id",
                        f"{p.participant_id} appears more than once in participants.tsv.",
                    )
                )
            seen.add(p.participant_id)

    cal_dir = paths.calibration_dir(root)
    cal_files = sorted(cal_dir.glob("cal-*.json")) if cal_dir.exists() else []
    for cal_path in cal_files:
        content_hash = cal_path.stem.removeprefix("cal-")
        try:
            calibration = dataset.load_calibration(content_hash, root)
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            issues.append(
                _issue("error", "invalid_calibration", f"{cal_path.name}: {exc}", cal_path.name)
            )
            continue
        if calibration.is_stale():
            issues.append(
                _issue(
                    "warning",
                    "stale_calibration",
                    f"{cal_path.name} is older than 30 days.",
                    cal_path.name,
                )
            )

    for participant_dir in sorted(p for p in root.glob("sub-*") if p.is_dir()):
        for session_dir in sorted(s for s in participant_dir.glob("ses-*") if s.is_dir()):
            report = validate_session(participant_dir.name, session_dir.name, root)
            issues.extend(report.issues)

    return ValidationReport(issues=issues)
