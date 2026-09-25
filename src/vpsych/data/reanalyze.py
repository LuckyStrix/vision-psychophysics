"""Reanalysis: recompute a test's summary from its raw trials and compare.

`vpsych reanalyze` never touches raw data (`_trials.tsv`/`_trials.json`/
`_frames.tsv`, or the original `_summary.json`) -- it only ever *adds* a new
file, `..._summary-reanalysis-<analysis_version>.json`, alongside the
original. This lets analysis code improve over time (a better fit, a bug
fix, a new quality check) without ever losing or silently overwriting what
was originally recorded, per the project's no-silent-data-loss rule.
"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from vpsych.core.rng import make_rng
from vpsych.data import dataset
from vpsych.data.schemas import SessionInfo, TestSummary
from vpsych.data.tsv import read_trials_tsv
from vpsych.data.writer import _atomic_write_bytes
from vpsych.tests_catalog.base import discover_tests, get_test

_STEM_RE = re.compile(
    r"task-(?P<task_id>[a-z][a-z0-9_]*)_eye-(?P<eye>OD|OS|OU)_run-(?P<run>\d+)_trials$"
)


class ReanalysisResult(BaseModel):
    """Outcome of reanalyzing one test run within a session.

    Attributes:
        task_id: The test's `TestSpec.id`.
        eye: Eye tested.
        run: 1-based run number.
        old_summary: The previously stored summary, or `None` if none existed.
        new_summary: The freshly recomputed summary.
        matches: Whether `new_summary` is identical to `old_summary` (field
            for field, `analysis_version` included) -- `False` whenever
            `old_summary` is `None`.
        differing_fields: Names of top-level fields that differ between
            `old_summary` and `new_summary`, empty if `matches` or if
            `old_summary` is `None`.
        written_path: Path the new summary was written to, if `write=True`
            was passed to `reanalyze_session`, else `None`.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    task_id: str
    eye: str
    run: int
    old_summary: TestSummary | None
    new_summary: TestSummary
    matches: bool
    differing_fields: list[str] = Field(default_factory=list)
    written_path: Path | None = None


def _parse_trials_stem(stem: str) -> tuple[str, str, int]:
    m = _STEM_RE.search(stem)
    if m is None:
        raise ValueError(f"Could not parse task/eye/run from trials filename stem {stem!r}.")
    return m.group("task_id"), m.group("eye"), int(m.group("run"))


def _diff_fields(old: TestSummary, new: TestSummary) -> list[str]:
    old_dump = old.model_dump(mode="json")
    new_dump = new.model_dump(mode="json")
    return [k for k in old_dump if old_dump.get(k) != new_dump.get(k)]


def reanalyze_session(path: str | Path, write: bool = False) -> list[ReanalysisResult]:
    """Recompute summaries for every test run in a session and compare to what's stored.

    For each `..._trials.tsv` in the session's `beh/` directory: parses the
    trials via `vpsych.data.tsv.read_trials_tsv`, looks up the registered
    test class for its `task_id` via `vpsych.tests_catalog.base.get_test`,
    reconstructs a test instance using the session's recorded display
    geometry, calibration (loaded from the dataset's calibration store by
    `session.json`'s `calibration_hash`), and plan parameters, and calls
    `summarize(df)` on it.

    Args:
        path: Path to the session directory (e.g.
            `<data_root>/sub-0001/ses-20260916T103000`).
        write: If `True`, write each recomputed summary alongside the
            original as `..._summary-reanalysis-<analysis_version>.json`
            (never overwriting the original `..._summary.json`).

    Returns:
        One `ReanalysisResult` per `_trials.tsv` file found, in filename
        order.

    Raises:
        FileNotFoundError: If `path` has no `session.json`.
    """
    session_dir = Path(path)
    session_json_path = session_dir / "session.json"
    if not session_json_path.exists():
        raise FileNotFoundError(f"No session.json found at {session_json_path}.")
    session_info = SessionInfo.model_validate_json(session_json_path.read_text(encoding="utf-8"))

    # <data_root>/sub-XXXX/ses-.../  -> data_root is two levels up.
    root = session_dir.parent.parent
    calibration = dataset.load_calibration(session_info.calibration_hash, root)

    beh_dir = session_dir / "beh"
    results: list[ReanalysisResult] = []
    if not beh_dir.exists():
        return results

    # A fresh process (e.g. the vpsych-data CLI) may not have imported any
    # real test's subpackage yet, in which case get_test() below would find
    # an empty registry -- see discover_tests()'s docstring.
    discover_tests()

    for trials_path in sorted(beh_dir.glob("*_trials.tsv")):
        task_id, eye, run = _parse_trials_stem(trials_path.stem)
        df = read_trials_tsv(trials_path)

        test_cls = get_test(task_id)
        # Runs are numbered per task in plan order (see the runner), so the run-th planned
        # entry for this task is the one that produced this file.
        planned_for_task = [t for t in session_info.plan.tests if t.task_id == task_id]
        planned = planned_for_task[run - 1] if run - 1 < len(planned_for_task) else None
        raw_params = planned.params if planned is not None else {}
        params = test_cls.spec.params_model.model_validate(raw_params)
        rng, _ = make_rng(session_info.plan.seed)
        test_instance = test_cls(
            params=params, display=session_info.display, calibration=calibration, rng=rng
        )
        new_summary = test_instance.summarize(df)

        summary_path = beh_dir / trials_path.name.replace("_trials.tsv", "_summary.json")
        old_summary: TestSummary | None = None
        if summary_path.exists():
            old_summary = TestSummary.model_validate_json(summary_path.read_text(encoding="utf-8"))

        matches = old_summary is not None and old_summary == new_summary
        differing = [] if old_summary is None else _diff_fields(old_summary, new_summary)

        written_path: Path | None = None
        if write:
            written_path = beh_dir / (
                trials_path.name.replace("_trials.tsv", "_summary")
                + f"-reanalysis-{new_summary.analysis_version}.json"
            )
            _atomic_write_bytes(written_path, new_summary.model_dump_json(indent=2).encode("utf-8"))

        results.append(
            ReanalysisResult(
                task_id=task_id,
                eye=eye,
                run=run,
                old_summary=old_summary,
                new_summary=new_summary,
                matches=matches,
                differing_fields=differing,
                written_path=written_path,
            )
        )

    return results
