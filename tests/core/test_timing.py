"""Unit tests for vpsych.core.timing: frame-interval summaries and drop detection."""

from __future__ import annotations

import pytest

from vpsych.core.timing import FrameTimingStats, summarize_frame_intervals


def test_empty_intervals() -> None:
    stats = summarize_frame_intervals([], refresh_hz=60.0)
    assert stats == FrameTimingStats(
        n_frames=0, mean_ms=0.0, sd_ms=0.0, n_dropped=0, dropped_fraction=0.0, expected_ms=1000 / 60
    )


def test_perfect_intervals_no_drops() -> None:
    refresh_hz = 60.0
    nominal_s = 1.0 / refresh_hz
    intervals = [nominal_s] * 100
    stats = summarize_frame_intervals(intervals, refresh_hz=refresh_hz)
    assert stats.n_frames == 100
    assert stats.mean_ms == pytest.approx(1000 / 60)
    assert stats.sd_ms == pytest.approx(0.0, abs=1e-9)
    assert stats.n_dropped == 0
    assert stats.dropped_fraction == 0.0
    assert stats.expected_ms == pytest.approx(1000 / 60)


def test_dropped_frame_detected_above_tolerance() -> None:
    refresh_hz = 60.0
    nominal_s = 1.0 / refresh_hz
    # One interval takes 2x nominal -> a dropped frame at default tolerance 1.5.
    intervals = [nominal_s] * 9 + [nominal_s * 2.0]
    stats = summarize_frame_intervals(intervals, refresh_hz=refresh_hz)
    assert stats.n_dropped == 1
    assert stats.dropped_fraction == pytest.approx(0.1)


def test_not_dropped_within_tolerance() -> None:
    refresh_hz = 60.0
    nominal_s = 1.0 / refresh_hz
    # 1.2x nominal is within the default 1.5x tolerance -> not a drop.
    intervals = [nominal_s] * 9 + [nominal_s * 1.2]
    stats = summarize_frame_intervals(intervals, refresh_hz=refresh_hz)
    assert stats.n_dropped == 0


def test_custom_drop_tolerance() -> None:
    refresh_hz = 60.0
    nominal_s = 1.0 / refresh_hz
    intervals = [nominal_s] * 9 + [nominal_s * 1.2]
    stats = summarize_frame_intervals(intervals, refresh_hz=refresh_hz, drop_tolerance=1.1)
    assert stats.n_dropped == 1


def test_mean_and_sd() -> None:
    intervals_s = [0.016, 0.017, 0.015, 0.016]
    stats = summarize_frame_intervals(intervals_s, refresh_hz=60.0)
    intervals_ms = [x * 1000 for x in intervals_s]
    expected_mean = sum(intervals_ms) / len(intervals_ms)
    assert stats.mean_ms == pytest.approx(expected_mean)
    assert stats.sd_ms > 0


def test_single_interval_sd_is_zero() -> None:
    stats = summarize_frame_intervals([0.0167], refresh_hz=60.0)
    assert stats.n_frames == 1
    assert stats.sd_ms == 0.0


def test_invalid_refresh_hz_raises() -> None:
    with pytest.raises(ValueError, match="refresh_hz"):
        summarize_frame_intervals([0.01], refresh_hz=0.0)
    with pytest.raises(ValueError, match="refresh_hz"):
        summarize_frame_intervals([0.01], refresh_hz=-60.0)


def test_invalid_drop_tolerance_raises() -> None:
    with pytest.raises(ValueError, match="drop_tolerance"):
        summarize_frame_intervals([0.01], refresh_hz=60.0, drop_tolerance=0.0)
