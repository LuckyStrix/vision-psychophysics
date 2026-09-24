"""Export one participant, or a whole dataset, to a self-contained zip.

The zip contains the raw BIDS-like tree (or the subset for one
participant), a `summaries_tidy.csv` with one row per test summary, a
`data_dictionary.md` describing every file and column, the JSON Schemas for
every model in play, and a `validation_report.json` -- so the zip is
independently interpretable without needing `vpsych` installed.
"""

from __future__ import annotations

import csv
import io
import json
import tempfile
import zipfile
from pathlib import Path

from vpsych.core.calibration.models import Calibration
from vpsych.core.trial import TrialRecord
from vpsych.data import dataset, paths
from vpsych.data.dataset import PARTICIPANT_COLUMN_SIDECARS
from vpsych.data.schemas import TestSummary, export_json_schemas
from vpsych.data.validate import ValidationReport, validate_dataset
from vpsych.data.writer import TRIAL_COLUMN_SIDECARS

_TIDY_COLUMNS = [
    "participant_id",
    "session_id",
    "task_id",
    "task_version",
    "eye",
    "run",
    "value",
    "ci_low",
    "ci_high",
    "ci_level",
    "units",
    "method",
    "n_trials",
    "n_catch",
    "catch_lapse_rate",
    "quality_flags",
    "analysis_version",
]


def _iter_summaries(root: Path, participant_id: str | None) -> list[tuple[str, str, TestSummary]]:
    """Yield (participant_id, session_id, TestSummary) for every summary under `root`."""
    found: list[tuple[str, str, TestSummary]] = []
    participant_dirs = (
        [root / participant_id]
        if participant_id
        else sorted(p for p in root.glob("sub-*") if p.is_dir())
    )
    for pdir in participant_dirs:
        if not pdir.is_dir():
            continue
        for sdir in sorted(s for s in pdir.glob("ses-*") if s.is_dir()):
            beh_dir = sdir / "beh"
            if not beh_dir.exists():
                continue
            for summary_path in sorted(beh_dir.glob("*_summary.json")):
                try:
                    summary = TestSummary.model_validate_json(
                        summary_path.read_text(encoding="utf-8")
                    )
                except Exception:
                    continue
                found.append((pdir.name, sdir.name, summary))
    return found


def _build_tidy_csv(rows: list[tuple[str, str, TestSummary]]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_TIDY_COLUMNS)
    writer.writeheader()
    for participant_id, session_id, s in rows:
        writer.writerow(
            {
                "participant_id": participant_id,
                "session_id": session_id,
                "task_id": s.task_id,
                "task_version": s.task_version,
                "eye": s.eye,
                "run": s.run,
                "value": s.estimate.value,
                "ci_low": s.estimate.ci_low,
                "ci_high": s.estimate.ci_high,
                "ci_level": s.estimate.ci_level,
                "units": s.estimate.units,
                "method": s.estimate.method,
                "n_trials": s.n_trials,
                "n_catch": s.n_catch,
                "catch_lapse_rate": s.catch_lapse_rate,
                "quality_flags": ";".join(f.code for f in s.quality_flags),
                "analysis_version": s.analysis_version,
            }
        )
    return buf.getvalue()


def _build_data_dictionary() -> str:
    lines: list[str] = ["# vpsych data dictionary", ""]
    lines += [
        "This file documents every column/field written by vpsych. It is generated from the "
        "same column-sidecar definitions written into each dataset's `_trials.json` and "
        "`participants.json` files, plus the pydantic field descriptions used to generate the "
        "JSON Schemas alongside this file.",
        "",
        "## `_trials.tsv` columns (see `docs/DATA_FORMAT.md` for the full file layout)",
        "",
        "| Column | Units | Description |",
        "|---|---|---|",
    ]
    for name, field in TrialRecord.model_fields.items():
        sidecar = TRIAL_COLUMN_SIDECARS.get(name)
        units = (sidecar.units if sidecar else None) or ""
        description = (sidecar.description if sidecar else field.description) or ""
        lines.append(f"| `{name}` | {units} | {description} |")

    lines += [
        "",
        "## `participants.tsv` columns",
        "",
        "| Column | Units | Description |",
        "|---|---|---|",
    ]
    for name, sidecar in PARTICIPANT_COLUMN_SIDECARS.items():
        lines.append(f"| `{name}` | {sidecar.units or ''} | {sidecar.description} |")

    lines += [
        "",
        "## `summaries_tidy.csv` columns",
        "",
        "| Column | Description |",
        "|---|---|",
    ]
    for name in _TIDY_COLUMNS:
        lines.append(f"| `{name}` | See `TestSummary`/`ThresholdEstimate` JSON Schema. |")

    lines += [
        "",
        "## JSON Schemas",
        "",
        "See the `json_schemas/` directory in this export for the full field-level schema of every JSON document type (`SessionInfo`, `TestSummary`, `Participant`, `DatasetDescription`, `Calibration`, `TrialRecord` (its TSV row shape), and `ColumnSidecar`).",
    ]
    return "\n".join(lines) + "\n"


def _write_json_schemas(tmp_dir: Path) -> Path:
    schema_dir = tmp_dir / "json_schemas"
    export_json_schemas(str(schema_dir))
    (schema_dir / "TrialRecord.schema.json").write_text(
        json.dumps(TrialRecord.model_json_schema(), indent=2, sort_keys=True), encoding="utf-8"
    )
    (schema_dir / "Calibration.schema.json").write_text(
        json.dumps(Calibration.model_json_schema(), indent=2, sort_keys=True), encoding="utf-8"
    )
    return schema_dir


def _export(root: Path, out_zip: Path, participant_id: str | None) -> Path:
    root = Path(root)
    out_zip = Path(out_zip)
    report: ValidationReport = validate_dataset(root)

    summaries = _iter_summaries(root, participant_id)
    tidy_csv = _build_tidy_csv(summaries)
    data_dictionary = _build_data_dictionary()

    out_zip.parent.mkdir(parents=True, exist_ok=True)
    out_zip_resolved = out_zip.resolve()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        schema_dir = _write_json_schemas(tmp_dir)

        with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            # Dataset-level files.
            for name in ("dataset_description.json", "participants.tsv", "participants.json"):
                p = root / name
                if p.exists():
                    zf.write(p, arcname=f"raw/{name}")

            # Calibration store (always included in full -- calibrations are
            # small, immutable, and sessions reference them by hash).
            cal_dir = paths.calibration_dir(root)
            if cal_dir.exists():
                for cal_file in sorted(cal_dir.glob("cal-*.json")):
                    zf.write(cal_file, arcname=f"raw/calibration/{cal_file.name}")

            # Raw session tree.
            participant_dirs = (
                [root / participant_id]
                if participant_id
                else sorted(p for p in root.glob("sub-*") if p.is_dir())
            )
            for pdir in participant_dirs:
                if not pdir.is_dir():
                    continue
                for file_path in sorted(pdir.rglob("*")):
                    # Never archive the zip being written, if it lives under the tree.
                    if file_path.is_file() and file_path.resolve() != out_zip_resolved:
                        zf.write(file_path, arcname=f"raw/{file_path.relative_to(root).as_posix()}")

            zf.writestr("summaries_tidy.csv", tidy_csv)
            zf.writestr("data_dictionary.md", data_dictionary)
            zf.writestr("validation_report.json", report.model_dump_json(indent=2))

            for schema_file in sorted(schema_dir.glob("*.schema.json")):
                zf.write(schema_file, arcname=f"json_schemas/{schema_file.name}")

    return out_zip


def export_participant(participant_id: str, out_zip: str | Path, root: Path | None = None) -> Path:
    """Export one participant's data to a zip.

    Args:
        participant_id: The participant to export, `sub-XXXX`.
        out_zip: Destination zip file path.
        root: Data root, or `None` to use `vpsych.data.paths.data_root()`.

    Returns:
        `out_zip`, for chaining.

    Raises:
        FileNotFoundError: If the participant does not exist in the dataset.
    """
    root = root or paths.data_root()
    if dataset.get_participant(participant_id, root) is None:
        raise FileNotFoundError(f"No participant {participant_id!r} in dataset at {root}.")
    return _export(root, Path(out_zip), participant_id)


def export_dataset(out_zip: str | Path, root: Path | None = None) -> Path:
    """Export the whole dataset to a zip.

    Args:
        out_zip: Destination zip file path.
        root: Data root, or `None` to use `vpsych.data.paths.data_root()`.

    Returns:
        `out_zip`, for chaining.
    """
    root = root or paths.data_root()
    return _export(root, Path(out_zip), None)


__all__ = ["export_dataset", "export_participant"]
