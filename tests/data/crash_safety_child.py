"""Child process for test_crash_safety.py: write N trials, then os._exit(1) with no cleanup.

Run as a script (not imported) so pytest never collects it as a test module,
and so `os._exit` in it can't affect the parent test process.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from vpsych.core.calibration.models import EnvironmentChecklist
from vpsych.core.display import DisplayGeometry
from vpsych.core.trial import TrialRecord
from vpsych.data.schemas import PlannedTest, SessionInfo, SessionPlan
from vpsych.data.writer import SessionWriter


def main() -> None:
    root = Path(sys.argv[1])
    participant_id = sys.argv[2]
    session_id = sys.argv[3]
    n = int(sys.argv[4])
    calibration_hash = sys.argv[5]

    display = DisplayGeometry(
        width_px=1920,
        height_px=1080,
        width_cm=53.0,
        height_cm=30.0,
        viewing_distance_cm=57.0,
        refresh_hz=60.0,
    )
    env = EnvironmentChecklist(
        room_lighting_controlled=True,
        monitor_warmed_up=True,
        night_light_disabled=True,
        hdr_disabled=True,
    )
    plan = SessionPlan(
        tests=[PlannedTest(task_id="dummy_test", eye="OD", params={}, viewing_distance_cm=57.0)],
        ordering="fixed",
        seed=1,
    )
    session_info = SessionInfo(
        session_id=session_id,
        participant_id=participant_id,
        plan=plan,
        calibration_hash=calibration_hash,
        display=display,
        os_info="linux-test",
        python_version="3.10.0",
        psychopy_version="n/a",
        software_version="0.1.0",
        status="running",
        started_utc=datetime.now(timezone.utc),
        environment=env,
    )

    writer = SessionWriter(participant_id, session_id, session_info, root)
    writer.__enter__()
    for i in range(n):
        trial = TrialRecord(
            participant_id=participant_id,
            session_id=session_id,
            task_id="dummy_test",
            task_version="1.0.0",
            run=1,
            eye="OD",
            block="main",
            trial_index=i,
            is_catch=False,
            intensity=0.1 * i,
            intensity_units="logMAR",
            stimulus_params={},
            correct_response="left",
            response="left",
            correct=True,
            rt_s=0.3,
            stimulus_onset_s=float(i),
            n_dropped_frames_trial=0,
            procedure_state={},
            timestamp_utc=datetime.now(timezone.utc),
            rng_seed=1,
        )
        writer.append_trial(trial)

    # Simulate a hard crash: no __exit__, no finalize(), nothing -- exactly
    # what happens if the OS kills this process (e.g. SIGKILL, power loss
    # after a segfault). Every trial appended above must already be durable.
    os._exit(1)


if __name__ == "__main__":
    main()
