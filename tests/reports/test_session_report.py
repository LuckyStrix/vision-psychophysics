"""Tests for vpsych.reports.session_report: HTML report rendering."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from vpsych.core.calibration.models import (
    Calibration,
    ColorCalibration,
    EnvironmentChecklist,
    GammaCalibration,
    PrimaryChromaticity,
)
from vpsych.core.display import DisplayGeometry
from vpsych.core.procedures.base import ThresholdEstimate
from vpsych.data.schemas import QualityFlag, SessionInfo, SessionPlan, TestSummary
from vpsych.reports.session_report import render_session_report


def _example_display() -> DisplayGeometry:
    """Create a standard display for testing."""
    return DisplayGeometry(
        width_px=1920,
        height_px=1080,
        width_cm=53.13,
        height_cm=29.88,
        viewing_distance_cm=57.0,
        refresh_hz=60.0,
    )


def _example_calibration() -> Calibration:
    """Create a standard calibration for testing."""
    return Calibration(
        created_utc=datetime(2026, 9, 1, tzinfo=timezone.utc),
        geometry=_example_display(),
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


def test_render_session_report_basic(tmp_path: Path) -> None:
    """Test basic report rendering from a complete session."""
    data_root = tmp_path / "data"
    data_root.mkdir()

    # Write calibration
    cal = _example_calibration()
    cal_dir = data_root / "calibration"
    cal_dir.mkdir()
    cal_hash = cal.content_hash()
    (cal_dir / f"cal-{cal_hash}.json").write_text(cal.model_dump_json(), encoding="utf-8")

    # Create session directory
    session_dir = data_root / "sub-0001" / "ses-20260901T100000"
    session_dir.mkdir(parents=True)

    # Write session.json
    display = _example_display()
    session_info = SessionInfo(
        session_id="ses-20260901T100000",
        participant_id="sub-0001",
        plan=SessionPlan(
            participant_id="sub-0001",
            tests=[],
            ordering="fixed",
            seed=42,
            calibration_hash=cal_hash,
        ),
        calibration_hash=cal_hash,
        display=display,
        os_info="Linux",
        python_version="3.10",
        psychopy_version="2026.2.4",
        gpu_info=None,
        software_version="0.1.0",
        git_commit="abc123",
        status="complete",
        started_utc=datetime(2026, 9, 1, 10, 0, 0, tzinfo=timezone.utc),
        ended_utc=datetime(2026, 9, 1, 10, 30, 0, tzinfo=timezone.utc),
        environment=EnvironmentChecklist(
            room_lighting_controlled=True,
            monitor_warmed_up=True,
            night_light_disabled=True,
            hdr_disabled=True,
        ),
    )
    (session_dir / "session.json").write_text(session_info.model_dump_json(), encoding="utf-8")

    # Render report
    out_path = render_session_report(session_dir)

    # Verify report exists and is readable
    assert out_path.exists()
    content = out_path.read_text(encoding="utf-8")
    assert "Vision Psychophysics Session Report" in content
    assert "sub-0001" in content
    assert "ses-20260901T100000" in content
    assert "not a medical device" in content.lower()


def test_render_session_report_with_custom_output(tmp_path: Path) -> None:
    """Test report rendering with custom output path."""
    data_root = tmp_path / "data"
    data_root.mkdir()

    # Write calibration
    cal = _example_calibration()
    cal_dir = data_root / "calibration"
    cal_dir.mkdir()
    cal_hash = cal.content_hash()
    (cal_dir / f"cal-{cal_hash}.json").write_text(cal.model_dump_json(), encoding="utf-8")

    # Create session
    session_dir = data_root / "sub-0001" / "ses-20260901T100000"
    session_dir.mkdir(parents=True)

    session_info = SessionInfo(
        session_id="ses-20260901T100000",
        participant_id="sub-0001",
        plan=SessionPlan(
            participant_id="sub-0001",
            tests=[],
            ordering="fixed",
            seed=42,
            calibration_hash=cal_hash,
        ),
        calibration_hash=cal_hash,
        display=_example_display(),
        os_info="Linux",
        python_version="3.10",
        psychopy_version="2026.2.4",
        gpu_info=None,
        software_version="0.1.0",
        git_commit="abc123",
        status="complete",
        started_utc=datetime(2026, 9, 1, 10, 0, 0, tzinfo=timezone.utc),
        ended_utc=datetime(2026, 9, 1, 10, 30, 0, tzinfo=timezone.utc),
        environment=EnvironmentChecklist(
            room_lighting_controlled=True,
            monitor_warmed_up=True,
            night_light_disabled=True,
            hdr_disabled=True,
        ),
    )
    (session_dir / "session.json").write_text(session_info.model_dump_json(), encoding="utf-8")

    # Render to custom path
    custom_out = tmp_path / "custom_report.html"
    out_path = render_session_report(session_dir, custom_out)

    assert out_path == custom_out
    assert out_path.exists()
    content = out_path.read_text(encoding="utf-8")
    assert "Vision Psychophysics Session Report" in content


def test_render_session_report_missing_session_json(tmp_path: Path) -> None:
    """Test report rendering fails gracefully when session.json is missing."""
    session_dir = tmp_path / "sub-0001" / "ses-20260901T100000"
    session_dir.mkdir(parents=True)

    with pytest.raises(FileNotFoundError):
        render_session_report(session_dir)


def test_render_session_report_with_missing_calibration(tmp_path: Path) -> None:
    """Test report rendering handles missing calibration gracefully."""
    data_root = tmp_path / "data"
    data_root.mkdir()

    session_dir = data_root / "sub-0001" / "ses-20260901T100000"
    session_dir.mkdir(parents=True)

    # Session references non-existent calibration
    session_info = SessionInfo(
        session_id="ses-20260901T100000",
        participant_id="sub-0001",
        plan=SessionPlan(
            participant_id="sub-0001",
            tests=[],
            ordering="fixed",
            seed=42,
            calibration_hash="nonexistent_hash",
        ),
        calibration_hash="nonexistent_hash",
        display=_example_display(),
        os_info="Linux",
        python_version="3.10",
        psychopy_version="2026.2.4",
        gpu_info=None,
        software_version="0.1.0",
        git_commit="abc123",
        status="complete",
        started_utc=datetime(2026, 9, 1, 10, 0, 0, tzinfo=timezone.utc),
        ended_utc=datetime(2026, 9, 1, 10, 30, 0, tzinfo=timezone.utc),
        environment=EnvironmentChecklist(
            room_lighting_controlled=True,
            monitor_warmed_up=True,
            night_light_disabled=True,
            hdr_disabled=True,
        ),
    )
    (session_dir / "session.json").write_text(session_info.model_dump_json(), encoding="utf-8")

    # Should still render successfully (with "not found" for calibration)
    out_path = render_session_report(session_dir)

    assert out_path.exists()
    content = out_path.read_text(encoding="utf-8")
    assert "not found" in content.lower()


def test_render_session_report_html_is_self_contained(tmp_path: Path) -> None:
    """Test that rendered HTML contains no external HTTP/HTTPS references."""
    data_root = tmp_path / "data"
    data_root.mkdir()

    # Write calibration
    cal = _example_calibration()
    cal_dir = data_root / "calibration"
    cal_dir.mkdir()
    cal_hash = cal.content_hash()
    (cal_dir / f"cal-{cal_hash}.json").write_text(cal.model_dump_json(), encoding="utf-8")

    session_dir = data_root / "sub-0001" / "ses-20260901T100000"
    session_dir.mkdir(parents=True)

    session_info = SessionInfo(
        session_id="ses-20260901T100000",
        participant_id="sub-0001",
        plan=SessionPlan(
            participant_id="sub-0001",
            tests=[],
            ordering="fixed",
            seed=42,
            calibration_hash=cal_hash,
        ),
        calibration_hash=cal_hash,
        display=_example_display(),
        os_info="Linux",
        python_version="3.10",
        psychopy_version="2026.2.4",
        gpu_info=None,
        software_version="0.1.0",
        git_commit="abc123",
        status="complete",
        started_utc=datetime(2026, 9, 1, 10, 0, 0, tzinfo=timezone.utc),
        ended_utc=datetime(2026, 9, 1, 10, 30, 0, tzinfo=timezone.utc),
        environment=EnvironmentChecklist(
            room_lighting_controlled=True,
            monitor_warmed_up=True,
            night_light_disabled=True,
            hdr_disabled=True,
        ),
    )
    (session_dir / "session.json").write_text(session_info.model_dump_json(), encoding="utf-8")

    out_path = render_session_report(session_dir)
    content = out_path.read_text(encoding="utf-8")

    # Check for no external HTTP/HTTPS references (except data URIs)
    assert "http://" not in content or "data:image" in content
    assert "https://" not in content or "data:image" in content
    # Should contain self-contained elements
    assert "data:image/png;base64," in content or "No test results" in content


def test_render_session_report_html_contains_required_sections(tmp_path: Path) -> None:
    """Test that HTML report contains all required sections."""
    data_root = tmp_path / "data"
    data_root.mkdir()

    cal = _example_calibration()
    cal_dir = data_root / "calibration"
    cal_dir.mkdir()
    cal_hash = cal.content_hash()
    (cal_dir / f"cal-{cal_hash}.json").write_text(cal.model_dump_json(), encoding="utf-8")

    session_dir = data_root / "sub-0001" / "ses-20260901T100000"
    session_dir.mkdir(parents=True)

    session_info = SessionInfo(
        session_id="ses-20260901T100000",
        participant_id="sub-0001",
        plan=SessionPlan(
            participant_id="sub-0001",
            tests=[],
            ordering="fixed",
            seed=42,
            calibration_hash=cal_hash,
        ),
        calibration_hash=cal_hash,
        display=_example_display(),
        os_info="Linux",
        python_version="3.10",
        psychopy_version="2026.2.4",
        gpu_info=None,
        software_version="0.1.0",
        git_commit="abc123",
        status="complete",
        started_utc=datetime(2026, 9, 1, 10, 0, 0, tzinfo=timezone.utc),
        ended_utc=datetime(2026, 9, 1, 10, 30, 0, tzinfo=timezone.utc),
        environment=EnvironmentChecklist(
            room_lighting_controlled=True,
            monitor_warmed_up=True,
            night_light_disabled=True,
            hdr_disabled=True,
        ),
    )
    (session_dir / "session.json").write_text(session_info.model_dump_json(), encoding="utf-8")

    out_path = render_session_report(session_dir)
    content = out_path.read_text(encoding="utf-8")

    # Check for required sections
    assert "Session Information" in content
    assert "Calibration Information" in content
    assert "Display Information" in content
    assert "Participant ID" in content
    assert "Session ID" in content
    assert "not a medical device" in content.lower()


def test_render_session_report_with_summary_with_quality_flags(tmp_path: Path) -> None:
    """Test report handles summary with quality flags."""
    data_root = tmp_path / "data"
    data_root.mkdir()

    cal = _example_calibration()
    cal_dir = data_root / "calibration"
    cal_dir.mkdir()
    cal_hash = cal.content_hash()
    (cal_dir / f"cal-{cal_hash}.json").write_text(cal.model_dump_json(), encoding="utf-8")

    session_dir = data_root / "sub-0001" / "ses-20260901T100000"
    session_dir.mkdir(parents=True)
    (session_dir / "beh").mkdir()

    # Write session.json
    session_info = SessionInfo(
        session_id="ses-20260901T100000",
        participant_id="sub-0001",
        plan=SessionPlan(
            participant_id="sub-0001",
            tests=[],
            ordering="fixed",
            seed=42,
            calibration_hash=cal_hash,
        ),
        calibration_hash=cal_hash,
        display=_example_display(),
        os_info="Linux",
        python_version="3.10",
        psychopy_version="2026.2.4",
        gpu_info=None,
        software_version="0.1.0",
        git_commit="abc123",
        status="complete",
        started_utc=datetime(2026, 9, 1, 10, 0, 0, tzinfo=timezone.utc),
        ended_utc=datetime(2026, 9, 1, 10, 30, 0, tzinfo=timezone.utc),
        environment=EnvironmentChecklist(
            room_lighting_controlled=True,
            monitor_warmed_up=True,
            night_light_disabled=True,
            hdr_disabled=True,
        ),
    )
    (session_dir / "session.json").write_text(session_info.model_dump_json(), encoding="utf-8")

    # Write a summary with quality flags
    summary = TestSummary(
        task_id="test_task",
        task_version="0.1.0",
        eye="OD",
        run=1,
        estimate=ThresholdEstimate(
            value=-1.0,
            ci_low=-1.2,
            ci_high=-0.8,
            ci_level=0.95,
            units="logMAR",
            method="quest_plus",
            extra={},
        ),
        fit_params={},
        gof={},
        quality_flags=[
            QualityFlag(
                code="excess_dropped_frames",
                severity="warning",
                message="Dropped frame rate exceeded 1% during this run",
            )
        ],
        n_trials=30,
        n_catch=3,
        catch_lapse_rate=0.0,
        frame_stats={},
        analysis_version="0.1.0",
    )
    summary_path = (
        session_dir
        / "beh"
        / "sub-0001_ses-20260901T100000_task-test_task_eye-OD_run-1_summary.json"
    )
    summary_path.write_text(summary.model_dump_json(), encoding="utf-8")

    # Write corresponding trials file
    trials_df = pd.DataFrame(
        [
            {
                "block": "main",
                "is_catch": False,
                "trial_index": i,
                "intensity": -1.0,
                "correct": i % 2 == 0,
                "eye": "OD",
                "n_dropped_frames_trial": 0,
            }
            for i in range(30)
        ]
    )
    trials_path = summary_path.parent / summary_path.name.replace("_summary.json", "_trials.tsv")
    trials_df.to_csv(trials_path, sep="\t", index=False)

    # Should render without error
    out_path = render_session_report(session_dir)

    assert out_path.exists()
    content = out_path.read_text(encoding="utf-8")
    assert "excess_dropped_frames" in content or "warning" in content.lower()
