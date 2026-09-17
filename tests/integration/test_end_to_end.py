"""End-to-end integration test: a full session through the real runner CLI.

Exercises the whole pipeline together, the way a real session actually
runs, rather than each module in isolation: `vpsych.data.dataset` creates a
participant and stores a calibration, the real `vpsych.runner` CLI runs as
a subprocess in `--simulate` mode against `vpsych.tests_catalog._example`
(a small but realistic 2AFC test driven by `QuestPlusProcedure`) with the
real `vpsych.data.writer.SessionWriter`, and then every downstream data-layer
operation (`validate`, `reanalyze`, `export`, `catalog`) runs against what
that subprocess actually wrote to disk.

A second test covers aborting mid-session (via `run_session`'s test-only
`abort_after_trials`, in-process -- see that function's docstring for why
this is preferred over racing an OS signal against a real subprocess) and
checks the interrupted session still retains its trials and validates
cleanly.
"""

from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from vpsych.core.calibration.models import (
    Calibration,
    ColorCalibration,
    EnvironmentChecklist,
    GammaCalibration,
    PrimaryChromaticity,
)
from vpsych.core.display import DisplayGeometry
from vpsych.core.psychometric import intensity_at_p_correct
from vpsych.data import catalog, dataset, export, reanalyze, validate
from vpsych.data.schemas import SessionInfo, TestSummary
from vpsych.data.tsv import read_trials_tsv
from vpsych.runner.__main__ import build_arg_parser, parse_simulated_observer_spec, run_session
from vpsych.runner.status import RunnerExitCode, RunnerStatus

TASK_ID = "example_contrast_2afc"
TRUE_THRESHOLD = -1.0
TRUE_SLOPE = 0.3
TRUE_LAPSE = 0.02
SIM_SPEC = f"psychometric:threshold={TRUE_THRESHOLD},slope={TRUE_SLOPE},lapse={TRUE_LAPSE}"


def _display() -> DisplayGeometry:
    return DisplayGeometry(
        width_px=1920,
        height_px=1080,
        width_cm=53.13,
        height_cm=29.88,
        viewing_distance_cm=57.0,
        refresh_hz=60.0,
    )


def _calibration() -> Calibration:
    return Calibration(
        created_utc=datetime(2026, 9, 1, tzinfo=timezone.utc),
        geometry=_display(),
        gamma=GammaCalibration(
            method="photometer", gamma_single=2.2, lum_min_cdm2=0.3, lum_max_cdm2=120.0
        ),
        color=ColorCalibration(
            method="measured",
            red=PrimaryChromaticity(x=0.64, y=0.33, Y_cdm2=22.0),
            green=PrimaryChromaticity(x=0.30, y=0.60, Y_cdm2=72.0),
            blue=PrimaryChromaticity(x=0.15, y=0.06, Y_cdm2=6.0),
            white=PrimaryChromaticity(x=0.3127, y=0.3290, Y_cdm2=120.0),
        ),
        environment=EnvironmentChecklist(
            room_lighting_controlled=True,
            monitor_warmed_up=True,
            night_light_disabled=True,
            hdr_disabled=True,
        ),
        software_version="0.1.0",
    )


def _write_plan(
    path: Path,
    participant_id: str,
    calibration_hash: str,
    *,
    max_trials: int,
    seed: int,
) -> None:
    plan = {
        "participant_id": participant_id,
        "tests": [
            {
                "task_id": TASK_ID,
                "eye": "OU",
                "params": {"max_trials": max_trials},
                "viewing_distance_cm": 57.0,
            }
        ],
        "ordering": "fixed",
        "seed": seed,
        "calibration_hash": calibration_hash,
    }
    path.write_text(json.dumps(plan), encoding="utf-8")


def _find_session_dir(data_root: Path, participant_id: str) -> Path:
    sessions = sorted((data_root / participant_id).glob("ses-*"))
    assert len(sessions) == 1, f"expected exactly one session directory, got {sessions}"
    return sessions[0]


def _make_writable(session_dir: Path) -> None:
    """Undo SessionWriter's readonly_after_finalize chmod so tmp_path cleanup can remove it."""
    for p in session_dir.rglob("*"):
        if p.is_file():
            p.chmod(0o644)


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    dataset.init_dataset(root, name="end-to-end test dataset")
    return root


def test_end_to_end_happy_path_via_subprocess(tmp_path: Path, data_root: Path) -> None:
    participant = dataset.create_participant(data_root)
    cal = _calibration()
    dataset.save_calibration(cal, data_root)

    plan_path = tmp_path / "plan.json"
    status_path = tmp_path / "status.json"
    _write_plan(plan_path, participant.participant_id, cal.content_hash(), max_trials=50, seed=777)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "vpsych.runner",
            "--session-plan",
            str(plan_path),
            "--status-file",
            str(status_path),
            "--data-root",
            str(data_root),
            "--simulate",
            SIM_SPEC,
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == int(RunnerExitCode.OK), (
        f"exit_code={result.returncode}\nstdout={result.stdout}\nstderr={result.stderr}"
    )

    status = RunnerStatus.read(status_path)
    assert status.state == "finished"

    session_dir = _find_session_dir(data_root, participant.participant_id)
    session_info = SessionInfo.model_validate_json(
        (session_dir / "session.json").read_text(encoding="utf-8")
    )
    assert session_info.status == "complete"
    assert session_info.participant_id == participant.participant_id
    assert session_info.calibration_hash == cal.content_hash()

    # --- Trials TSV: practice + main, ~10% catch, catch trials don't update the procedure.
    trials_path = next((session_dir / "beh").glob(f"*task-{TASK_ID}*_trials.tsv"))
    df = read_trials_tsv(trials_path)
    assert (df["block"] == "practice").sum() == 3  # TrialLoopConfig's default n_practice_trials

    main = df[df["block"] == "main"].sort_values("trial_index").reset_index(drop=True)
    assert len(main) > 0
    n_catch = int(main["is_catch"].sum())
    catch_fraction = n_catch / len(main)
    assert n_catch >= 1
    assert 0.0 < catch_fraction < 0.30  # nominal 10%, loose bound against sampling noise

    for i in range(1, len(main)):
        if bool(main.loc[i, "is_catch"]):
            # See core/trial_loop.py's procedure_state fix: a catch trial must
            # leave the procedure state exactly as the preceding trial left it.
            assert main.loc[i, "procedure_state"] == main.loc[i - 1, "procedure_state"]

    # --- Summary: threshold in the right ballpark of the simulated truth.
    summary_path = next((session_dir / "beh").glob(f"*task-{TASK_ID}*_summary.json"))
    summary = TestSummary.model_validate_json(summary_path.read_text(encoding="utf-8"))
    observer = parse_simulated_observer_spec(SIM_SPEC)
    target = intensity_at_p_correct(observer.true_function, 0.75)
    assert abs(summary.estimate.value - target) < 1.0

    # --- Manifest and read-only files.
    manifest_path = session_dir / "MANIFEST.sha256"
    assert manifest_path.exists()
    assert manifest_path.read_text(encoding="utf-8").strip() != ""
    assert (trials_path.stat().st_mode & 0o777) == 0o444
    assert (summary_path.stat().st_mode & 0o777) == 0o444

    try:
        # --- validate: no errors.
        report = validate.validate_session(participant.participant_id, session_dir.name, data_root)
        assert report.ok, [i.model_dump() for i in report.issues]

        # --- reanalyze: reproduces the summary exactly.
        results = reanalyze.reanalyze_session(session_dir, write=False)
        assert results
        for r in results:
            assert r.matches, (r.task_id, r.eye, r.run, r.differing_fields)

        # --- export: valid zip, contains the raw tree + tidy summaries.
        zip_path = tmp_path / "export.zip"
        export.export_participant(participant.participant_id, zip_path, data_root)
        assert zipfile.is_zipfile(zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            bad = zf.testzip()
            assert bad is None, f"corrupt member: {bad}"
            names = zf.namelist()
            assert "summaries_tidy.csv" in names
            assert "data_dictionary.md" in names
            assert any(
                n.startswith(f"raw/{participant.participant_id}/") and n.endswith("_trials.tsv")
                for n in names
            )

        # --- catalog: indexes the summary.
        catalog.rebuild_catalog(data_root)
        sessions = catalog.sessions_for_participant(participant.participant_id, data_root)
        assert len(sessions) == 1
        assert sessions[0]["status"] == "complete"
        history = catalog.task_history(participant.participant_id, TASK_ID, "OU", data_root)
        assert len(history) == 1
        assert history[0]["value"] == pytest.approx(summary.estimate.value)
    finally:
        _make_writable(session_dir)


def test_abort_mid_session_retains_trials_and_validates(tmp_path: Path, data_root: Path) -> None:
    """abort_after_trials deterministically interrupts the session (see run_session's
    docstring for why this, not a racy SIGTERM on a subprocess, is used here); the
    resulting session must be marked incomplete/aborted, keep every trial recorded
    before the abort, and still pass validation."""
    participant = dataset.create_participant(data_root)
    cal = _calibration()
    dataset.save_calibration(cal, data_root)

    plan_path = tmp_path / "plan_abort.json"
    status_path = tmp_path / "status_abort.json"
    # A large max_trials so the procedure would not finish on its own before
    # the abort fires.
    _write_plan(plan_path, participant.participant_id, cal.content_hash(), max_trials=500, seed=99)

    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--session-plan",
            str(plan_path),
            "--status-file",
            str(status_path),
            "--data-root",
            str(data_root),
            "--simulate",
            SIM_SPEC,
        ]
    )
    exit_code = run_session(args, abort_after_trials=8)
    assert exit_code == RunnerExitCode.ABORTED_BY_USER

    status = RunnerStatus.read(status_path)
    assert status.state == "aborted"

    session_dir = _find_session_dir(data_root, participant.participant_id)
    session_info = SessionInfo.model_validate_json(
        (session_dir / "session.json").read_text(encoding="utf-8")
    )
    assert session_info.status == "aborted"

    trials_path = next((session_dir / "beh").glob(f"*task-{TASK_ID}*_trials.tsv"))
    df = read_trials_tsv(trials_path)
    assert len(df) > 0  # every trial presented before the abort is retained

    try:
        report = validate.validate_session(participant.participant_id, session_dir.name, data_root)
        assert report.ok, [i.model_dump() for i in report.issues]
    finally:
        _make_writable(session_dir)
