"""Frame-interval accounting: summarize measured frame durations and detect drops.

PsychoPy windows can record the wall-clock interval between successive
flips (e.g. ``win.frameIntervals`` when ``win.recordFrameIntervals = True``).
This module turns a raw sequence of such intervals into a summary that flags
dropped frames, independent of PsychoPy itself, so it is unit-testable
headless.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field


class FrameTimingStats(BaseModel):
    """Summary statistics for a sequence of measured inter-flip intervals.

    Attributes:
        n_frames: Number of frame intervals summarized.
        mean_ms: Mean inter-flip interval, in milliseconds.
        sd_ms: Sample standard deviation of the inter-flip intervals, in
            milliseconds (0.0 if fewer than 2 intervals).
        n_dropped: Count of intervals classified as a dropped frame (see
            ``drop_tolerance`` in :func:`summarize_frame_intervals`).
        dropped_fraction: ``n_dropped / n_frames`` (0.0 if ``n_frames`` is 0).
        expected_ms: Nominal inter-flip interval implied by the display's
            refresh rate, i.e. ``1000 / refresh_hz``, in milliseconds.
    """

    model_config = ConfigDict(frozen=True)

    n_frames: int = Field(ge=0, description="Number of frame intervals summarized.")
    mean_ms: float = Field(description="Mean inter-flip interval in milliseconds.")
    sd_ms: float = Field(
        ge=0, description="Sample standard deviation of inter-flip intervals in milliseconds."
    )
    n_dropped: int = Field(ge=0, description="Count of intervals classified as dropped frames.")
    dropped_fraction: float = Field(
        ge=0, le=1, description="n_dropped / n_frames, 0.0 if n_frames is 0."
    )
    expected_ms: float = Field(
        gt=0, description="Nominal inter-flip interval implied by refresh_hz, in milliseconds."
    )


def summarize_frame_intervals(
    intervals_s: Sequence[float],
    refresh_hz: float,
    drop_tolerance: float = 1.5,
) -> FrameTimingStats:
    """Summarize measured frame intervals and count dropped frames.

    A frame is classified as dropped when its measured interval exceeds
    ``drop_tolerance`` times the nominal frame interval implied by
    ``refresh_hz`` (i.e. the flip took long enough that one or more refresh
    cycles were skipped).

    Args:
        intervals_s: Measured inter-flip intervals, in seconds, typically
            from ``win.frameIntervals`` (PsychoPy reports these in seconds).
        refresh_hz: Display refresh rate used to compute the nominal
            (expected) frame interval, in hertz.
        drop_tolerance: Multiplier on the nominal frame interval above which
            a measured interval counts as a dropped frame. Defaults to 1.5,
            i.e. an interval more than 50% longer than nominal.

    Returns:
        A :class:`FrameTimingStats` summarizing the sequence. If
        ``intervals_s`` is empty, ``n_frames``, ``n_dropped`` and
        ``dropped_fraction`` are all 0 and ``mean_ms``/``sd_ms`` are 0.0.

    Raises:
        ValueError: If ``refresh_hz`` is not positive, or ``drop_tolerance``
            is not positive.
    """
    if refresh_hz <= 0:
        raise ValueError(f"refresh_hz must be positive, got {refresh_hz}")
    if drop_tolerance <= 0:
        raise ValueError(f"drop_tolerance must be positive, got {drop_tolerance}")

    expected_ms = 1000.0 / refresh_hz
    n_frames = len(intervals_s)

    if n_frames == 0:
        return FrameTimingStats(
            n_frames=0,
            mean_ms=0.0,
            sd_ms=0.0,
            n_dropped=0,
            dropped_fraction=0.0,
            expected_ms=expected_ms,
        )

    intervals_ms = [x * 1000.0 for x in intervals_s]
    mean_ms = sum(intervals_ms) / n_frames

    if n_frames >= 2:
        variance = sum((x - mean_ms) ** 2 for x in intervals_ms) / (n_frames - 1)
        sd_ms = variance**0.5
    else:
        sd_ms = 0.0

    drop_threshold_ms = expected_ms * drop_tolerance
    n_dropped = sum(1 for x in intervals_ms if x > drop_threshold_ms)
    dropped_fraction = n_dropped / n_frames

    return FrameTimingStats(
        n_frames=n_frames,
        mean_ms=mean_ms,
        sd_ms=sd_ms,
        n_dropped=n_dropped,
        dropped_fraction=dropped_fraction,
        expected_ms=expected_ms,
    )
