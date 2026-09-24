"""Unit tests for vpsych.core.timing: frame-interval summaries and drop detection."""

from __future__ import annotations

import pytest

from vpsych.core.timing import (
    FrameTimingStats,
    measure_refresh,
    presentation_timing,
    refresh_matches,
    summarize_frame_intervals,
)


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


class _FakeWinWithFrameRate:
    def __init__(self, rate: float | None) -> None:
        self._rate = rate
        self.calls: list[dict[str, int]] = []

    def getActualFrameRate(  # noqa: N802 - mimics psychopy.visual.Window's real method name
        self,
        nIdentical: int = 10,  # noqa: N803 - mimics psychopy's real argument name
        nMaxFrames: int = 100,  # noqa: N803 - mimics psychopy's real argument name
        nWarmUpFrames: int = 10,  # noqa: N803 - mimics psychopy's real argument name
    ) -> float | None:
        self.calls.append(
            {"nIdentical": nIdentical, "nMaxFrames": nMaxFrames, "nWarmUpFrames": nWarmUpFrames}
        )
        return self._rate


class _FakeWinWithIntervalsOnly:
    def __init__(self, intervals: list[float]) -> None:
        self.frameIntervals = intervals


def test_measure_refresh_uses_get_actual_frame_rate() -> None:
    win = _FakeWinWithFrameRate(59.94)
    assert measure_refresh(win, n_frames=120) == pytest.approx(59.94)
    assert win.calls[0]["nIdentical"] == 120


def test_measure_refresh_falls_back_to_frame_intervals() -> None:
    win = _FakeWinWithFrameRate(None)
    win.frameIntervals = [1 / 60.0] * 10  # type: ignore[attr-defined]
    assert measure_refresh(win) == pytest.approx(60.0)


def test_measure_refresh_no_get_actual_frame_rate_uses_intervals() -> None:
    win = _FakeWinWithIntervalsOnly([1 / 120.0] * 20)
    assert measure_refresh(win) == pytest.approx(120.0)


def test_measure_refresh_raises_when_unmeasurable() -> None:
    win = _FakeWinWithFrameRate(None)
    with pytest.raises(RuntimeError):
        measure_refresh(win)


def test_measure_refresh_raises_on_empty_intervals() -> None:
    win = _FakeWinWithIntervalsOnly([])
    with pytest.raises(RuntimeError):
        measure_refresh(win)


@pytest.mark.parametrize(
    ("measured", "expected", "tol", "matches"),
    [
        (60.0, 60.0, 0.01, True),
        (60.5, 60.0, 0.01, True),  # 0.83% off, within 1%
        (61.0, 60.0, 0.01, False),  # 1.67% off, outside 1%
        (59.45, 60.0, 0.01, True),  # 0.92% off, within 1%
    ],
)
def test_refresh_matches(measured: float, expected: float, tol: float, matches: bool) -> None:
    assert refresh_matches(measured, expected, tol=tol) is matches


def test_refresh_matches_invalid_expected_raises() -> None:
    with pytest.raises(ValueError, match="expected_hz"):
        refresh_matches(60.0, 0.0)


def test_presentation_timing_clean_flips() -> None:
    flips = [i / 60.0 for i in range(10)]
    intervals, n_dropped = presentation_timing(flips, refresh_hz=60.0)
    assert len(intervals) == 9
    assert n_dropped == 0


def test_presentation_timing_detects_skipped_frame() -> None:
    flips = [0.0, 1 / 60, 3 / 60, 4 / 60]  # one refresh skipped between flips 2 and 3
    intervals, n_dropped = presentation_timing(flips, refresh_hz=60.0)
    assert intervals == pytest.approx([1 / 60, 2 / 60, 1 / 60])
    assert n_dropped == 1


@pytest.mark.parametrize("flips", [[], [0.5]])
def test_presentation_timing_too_few_flips(flips: list[float]) -> None:
    assert presentation_timing(flips, refresh_hz=60.0) == ([], 0)
