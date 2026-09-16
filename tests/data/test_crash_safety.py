"""Crash-safety test: kill a SessionWriter mid-session and verify no data loss.

Spawns a real subprocess (not a thread, not a mock) that appends N trials
through `SessionWriter` and then calls `os._exit(1)` -- bypassing all
Python cleanup (`__exit__`, `atexit`, `finally` blocks) exactly as a SIGKILL
or a segfault would. Per the project's no-silent-data-loss rule, every
trial `append_trial` returned from before the crash must be durably on
disk, and `vpsych.data.validate` must report the resulting session as
"still running / interrupted" (informational), not corrupt.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from vpsych.data import dataset, validate
from vpsych.data.tsv import read_trials_tsv

from .conftest import make_calibration

_CHILD_SCRIPT = Path(__file__).with_name("crash_safety_child.py")


def test_crash_mid_session_preserves_all_appended_trials(tmp_path: Path) -> None:
    root = tmp_path / "data"
    dataset.init_dataset(root, name="crash safety test")
    participant = dataset.create_participant(root)
    calibration = make_calibration()
    dataset.save_calibration(calibration, root)

    session_id = "ses-20260916T120000"
    n_trials = 11

    result = subprocess.run(
        [
            sys.executable,
            str(_CHILD_SCRIPT),
            str(root),
            participant.participant_id,
            session_id,
            str(n_trials),
            calibration.content_hash(),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    # os._exit(1) means the child's exit code is 1, not a Python traceback.
    assert result.returncode == 1, f"stderr: {result.stderr}"

    from vpsych.data import paths

    trials_path = paths.trials_tsv_path(
        participant.participant_id, session_id, "dummy_test", "OD", 1, root
    )
    assert trials_path.exists()
    df = read_trials_tsv(trials_path)
    assert len(df) == n_trials
    assert list(df["trial_index"]) == list(range(n_trials))

    session_json_path = paths.session_json_path(participant.participant_id, session_id, root)
    assert '"status": "running"' in session_json_path.read_text(encoding="utf-8")

    manifest_path = paths.session_manifest_path(participant.participant_id, session_id, root)
    assert not manifest_path.exists()  # finalize() never ran

    report = validate.validate_dataset(root)
    assert report.ok, f"unexpected errors: {report.errors}"
    assert any(i.code == "session_running_or_interrupted" for i in report.issues)
    assert not any(i.severity == "error" for i in report.issues)


def test_crash_mid_session_survives_multiple_kills(tmp_path: Path) -> None:
    """Two independent crashed sessions for the same participant both survive intact."""
    root = tmp_path / "data"
    dataset.init_dataset(root)
    participant = dataset.create_participant(root)
    calibration = make_calibration()
    dataset.save_calibration(calibration, root)

    from vpsych.data import paths

    for i, session_id in enumerate(["ses-20260101T090000", "ses-20260102T090000"]):
        n_trials = 3 + i
        result = subprocess.run(
            [
                sys.executable,
                str(_CHILD_SCRIPT),
                str(root),
                participant.participant_id,
                session_id,
                str(n_trials),
                calibration.content_hash(),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 1, f"stderr: {result.stderr}"
        trials_path = paths.trials_tsv_path(
            participant.participant_id, session_id, "dummy_test", "OD", 1, root
        )
        df = read_trials_tsv(trials_path)
        assert len(df) == n_trials

    report = validate.validate_dataset(root)
    assert report.ok
