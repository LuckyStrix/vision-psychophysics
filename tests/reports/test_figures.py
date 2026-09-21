"""Tests for vpsych.reports.figures: figure generation functions."""

from __future__ import annotations

from datetime import datetime, timezone

import matplotlib
import numpy as np
import pandas as pd

from vpsych.core.calibration.models import (
    Calibration,
    ColorCalibration,
    EnvironmentChecklist,
    GammaCalibration,
    PrimaryChromaticity,
)
from vpsych.core.display import DisplayGeometry
from vpsych.core.procedures.base import ThresholdEstimate
from vpsych.data.schemas import QualityFlag, TestSummary
from vpsych.reports.figures import (
    csf_figure,
    history_figure,
    psychometric_figure,
    quality_badges,
)

# Use Agg backend for headless testing
matplotlib.use("Agg")


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


def _example_summary(task_id: str = "test_task", with_fit: bool = True) -> TestSummary:
    """Create a basic TestSummary for testing."""
    estimate = ThresholdEstimate(
        value=-1.0,
        ci_low=-1.2,
        ci_high=-0.8,
        ci_level=0.95,
        units="logMAR",
        method="quest_plus",
        extra={},
    )
    fit_params = {"threshold": -1.0, "slope": 3.0, "lapse": 0.02, "guess": 0.5} if with_fit else {}
    return TestSummary(
        task_id=task_id,
        task_version="0.1.0",
        eye="OD",
        run=1,
        estimate=estimate,
        fit_params=fit_params,
        gof={},
        quality_flags=[
            QualityFlag(code="test_flag", severity="info", message="This is a test flag")
        ],
        n_trials=50,
        n_catch=5,
        catch_lapse_rate=0.0,
        frame_stats={},
        analysis_version="0.1.0",
    )


def _example_trials(n_main: int = 50, n_catch: int = 5) -> pd.DataFrame:
    """Create a DataFrame of synthetic trials."""
    rng = np.random.default_rng(42)
    rows = []

    # Main block trials at different intensities
    intensities = [-1.2, -1.0, -0.8]
    for intensity in intensities:
        n_at_intensity = n_main // len(intensities)
        for i in range(n_at_intensity):
            correct = bool(rng.random() < (0.5 + 0.3 * (intensity + 1.0)))
            rows.append(
                {
                    "block": "main",
                    "is_catch": False,
                    "trial_index": i,
                    "intensity": intensity,
                    "correct": correct,
                    "eye": "OD",
                    "n_dropped_frames_trial": 0,
                }
            )

    # Catch trials (high contrast, should be mostly correct)
    for i in range(n_catch):
        rows.append(
            {
                "block": "main",
                "is_catch": True,
                "trial_index": n_main + i,
                "intensity": 1.0,
                "correct": True,
                "eye": "OD",
                "n_dropped_frames_trial": 0,
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# Tests for psychometric_figure
# =============================================================================


def test_psychometric_figure_basic() -> None:
    """Test that psychometric_figure returns a Figure with expected axes."""
    summary = _example_summary(with_fit=True)
    trials = _example_trials()

    fig = psychometric_figure(summary, trials)

    assert fig is not None
    assert len(fig.axes) == 1
    ax = fig.axes[0]
    assert "Proportion Correct" in ax.get_ylabel()
    assert "logMAR" in ax.get_xlabel()
    matplotlib.pyplot.close(fig)


def test_psychometric_figure_without_fit() -> None:
    """Test psychometric_figure handles missing fit parameters gracefully."""
    summary = _example_summary(with_fit=False)
    trials = _example_trials()

    fig = psychometric_figure(summary, trials)

    assert fig is not None
    assert len(fig.axes) == 1
    matplotlib.pyplot.close(fig)


def test_psychometric_figure_empty_trials() -> None:
    """Test psychometric_figure handles empty trial dataframe."""
    summary = _example_summary()
    trials = pd.DataFrame()

    fig = psychometric_figure(summary, trials)

    assert fig is not None
    assert len(fig.axes) == 1
    matplotlib.pyplot.close(fig)


def test_psychometric_figure_no_main_block_trials() -> None:
    """Test psychometric_figure when all trials are catch trials."""
    summary = _example_summary()
    # Create a trials frame with only catch trials
    trials = pd.DataFrame(
        [
            {
                "block": "main",
                "is_catch": True,
                "trial_index": i,
                "intensity": 1.0,
                "correct": True,
                "eye": "OD",
                "n_dropped_frames_trial": 0,
            }
            for i in range(5)
        ]
    )

    fig = psychometric_figure(summary, trials)

    assert fig is not None
    matplotlib.pyplot.close(fig)


# =============================================================================
# Tests for csf_figure
# =============================================================================


def test_csf_figure_with_csf_data() -> None:
    """Test CSF figure generation with valid CSF curve data."""
    # Create a summary with CSF-specific extra data
    estimate = ThresholdEstimate(
        value=2.0,
        ci_low=1.9,
        ci_high=2.1,
        ci_level=0.95,
        units="log10_cs",
        method="qcsf",
        extra={
            "spatial_frequency_cpd": [0.5, 1.0, 2.0, 4.0, 8.0],
            "log10_cs_mean": [1.5, 2.0, 2.3, 2.0, 1.2],
            "log10_cs_ci_low": [1.4, 1.9, 2.2, 1.9, 1.0],
            "log10_cs_ci_high": [1.6, 2.1, 2.4, 2.1, 1.4],
            "aulcsf": 1.95,
        },
    )
    summary = TestSummary(
        task_id="contrast_sensitivity_function",
        task_version="0.1.0",
        eye="OD",
        run=1,
        estimate=estimate,
        fit_params={},
        gof={},
        quality_flags=[],
        n_trials=100,
        n_catch=10,
        catch_lapse_rate=0.05,
        frame_stats={},
        analysis_version="0.1.0",
    )

    fig = csf_figure(summary)

    assert fig is not None
    assert len(fig.axes) == 1
    ax = fig.axes[0]
    assert "Spatial Frequency" in ax.get_xlabel()
    assert "Log10 Contrast Sensitivity" in ax.get_ylabel()
    assert "contrast_sensitivity_function" in ax.get_title().lower()
    matplotlib.pyplot.close(fig)


def test_csf_figure_without_csf_data() -> None:
    """Test CSF figure handles missing CSF curve data."""
    summary = _example_summary()

    fig = csf_figure(summary)

    assert fig is not None
    assert len(fig.axes) == 1
    matplotlib.pyplot.close(fig)


# =============================================================================
# Tests for history_figure
# =============================================================================


def test_history_figure_basic() -> None:
    """Test history figure with longitudinal data."""
    history = [
        {
            "session_id": "ses-20260901T100000",
            "run": 1,
            "value": -1.0,
            "ci_low": -1.2,
            "ci_high": -0.8,
            "ci_level": 0.95,
            "units": "logMAR",
            "eye": "OD",
            "started_utc": "2026-09-01T10:00:00+00:00",
        },
        {
            "session_id": "ses-20260902T100000",
            "run": 1,
            "value": -0.95,
            "ci_low": -1.15,
            "ci_high": -0.75,
            "ci_level": 0.95,
            "units": "logMAR",
            "eye": "OD",
            "started_utc": "2026-09-02T10:00:00+00:00",
        },
        {
            "session_id": "ses-20260903T100000",
            "run": 1,
            "value": -0.9,
            "ci_low": -1.1,
            "ci_high": -0.7,
            "ci_level": 0.95,
            "units": "logMAR",
            "eye": "OD",
            "started_utc": "2026-09-03T10:00:00+00:00",
        },
    ]

    fig = history_figure(history, "test_task", "logMAR")

    assert fig is not None
    assert len(fig.axes) == 1
    ax = fig.axes[0]
    assert "Threshold" in ax.get_ylabel()
    assert "logMAR" in ax.get_ylabel()
    matplotlib.pyplot.close(fig)


def test_history_figure_empty() -> None:
    """Test history figure with empty history."""
    fig = history_figure([], "test_task", "logMAR")

    assert fig is not None
    assert len(fig.axes) == 1
    matplotlib.pyplot.close(fig)


def test_history_figure_multiple_eyes() -> None:
    """Test history figure with data from multiple eyes."""
    history = [
        {
            "session_id": "ses-20260901T100000",
            "run": 1,
            "value": -1.0,
            "ci_low": -1.2,
            "ci_high": -0.8,
            "ci_level": 0.95,
            "units": "logMAR",
            "eye": "OD",
            "started_utc": "2026-09-01T10:00:00+00:00",
        },
        {
            "session_id": "ses-20260901T100000",
            "run": 1,
            "value": -1.05,
            "ci_low": -1.25,
            "ci_high": -0.85,
            "ci_level": 0.95,
            "units": "logMAR",
            "eye": "OS",
            "started_utc": "2026-09-01T10:00:00+00:00",
        },
        {
            "session_id": "ses-20260902T100000",
            "run": 1,
            "value": -0.95,
            "ci_low": -1.15,
            "ci_high": -0.75,
            "ci_level": 0.95,
            "units": "logMAR",
            "eye": "OD",
            "started_utc": "2026-09-02T10:00:00+00:00",
        },
        {
            "session_id": "ses-20260902T100000",
            "run": 1,
            "value": -1.0,
            "ci_low": -1.2,
            "ci_high": -0.8,
            "ci_level": 0.95,
            "units": "logMAR",
            "eye": "OS",
            "started_utc": "2026-09-02T10:00:00+00:00",
        },
    ]

    fig = history_figure(history, "test_task", "logMAR")

    assert fig is not None
    assert len(fig.axes) == 1
    matplotlib.pyplot.close(fig)


# =============================================================================
# Tests for quality_badges
# =============================================================================


def test_quality_badges_extraction() -> None:
    """Test that quality_badges extracts flags correctly."""
    summary = _example_summary()
    badges = quality_badges(summary)

    assert len(badges) == 1
    code, severity, message = badges[0]
    assert code == "test_flag"
    assert severity == "info"
    assert message == "This is a test flag"


def test_quality_badges_empty() -> None:
    """Test quality_badges with no flags."""
    summary = TestSummary(
        task_id="test",
        task_version="0.1.0",
        eye="OD",
        run=1,
        estimate=ThresholdEstimate(
            value=1.0,
            ci_low=0.9,
            ci_high=1.1,
            ci_level=0.95,
            units="units",
            method="method",
            extra={},
        ),
        fit_params={},
        gof={},
        quality_flags=[],
        n_trials=0,
        n_catch=0,
        catch_lapse_rate=0.0,
        frame_stats={},
        analysis_version="0.1.0",
    )
    badges = quality_badges(summary)

    assert badges == []


def test_quality_badges_multiple() -> None:
    """Test quality_badges with multiple flags."""
    summary = TestSummary(
        task_id="test",
        task_version="0.1.0",
        eye="OD",
        run=1,
        estimate=ThresholdEstimate(
            value=1.0,
            ci_low=0.9,
            ci_high=1.1,
            ci_level=0.95,
            units="units",
            method="method",
            extra={},
        ),
        fit_params={},
        gof={},
        quality_flags=[
            QualityFlag(code="flag_1", severity="info", message="First flag"),
            QualityFlag(code="flag_2", severity="warning", message="Second flag"),
            QualityFlag(code="flag_3", severity="critical", message="Third flag"),
        ],
        n_trials=0,
        n_catch=0,
        catch_lapse_rate=0.0,
        frame_stats={},
        analysis_version="0.1.0",
    )
    badges = quality_badges(summary)

    assert len(badges) == 3
    assert badges[0][0] == "flag_1"
    assert badges[1][1] == "warning"
    assert badges[2][2] == "Third flag"
