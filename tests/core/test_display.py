"""Unit tests for vpsych.core.display: exact cm <-> px <-> deg conversions."""

from __future__ import annotations

import math

import pytest

from vpsych.core.display import DisplayGeometry


@pytest.fixture
def geometry() -> DisplayGeometry:
    # A representative 24" 1920x1080 monitor at a typical lab viewing distance.
    return DisplayGeometry(
        width_px=1920,
        height_px=1080,
        width_cm=53.13,
        height_cm=29.88,
        viewing_distance_cm=57.0,
        refresh_hz=60.0,
    )


def test_px_per_cm(geometry: DisplayGeometry) -> None:
    assert geometry.px_per_cm == pytest.approx(1920 / 53.13)


def test_deg_to_px_matches_exact_trig(geometry: DisplayGeometry) -> None:
    deg = 2.0
    expected_cm = 2.0 * geometry.viewing_distance_cm * math.tan(math.radians(deg) / 2.0)
    expected_px = expected_cm * geometry.px_per_cm
    assert geometry.deg_to_px(deg) == pytest.approx(expected_px)


def test_px_to_deg_matches_exact_trig(geometry: DisplayGeometry) -> None:
    px = 100.0
    x_cm = px / geometry.px_per_cm
    expected_deg = 2.0 * math.degrees(math.atan((x_cm / 2.0) / geometry.viewing_distance_cm))
    assert geometry.px_to_deg(px) == pytest.approx(expected_deg)


def test_deg_px_roundtrip(geometry: DisplayGeometry) -> None:
    for deg in (0.01, 0.5, 1.0, 5.0, 10.0, 30.0):
        px = geometry.deg_to_px(deg)
        assert geometry.px_to_deg(px) == pytest.approx(deg, rel=1e-9)


def test_deg_to_px_not_small_angle(geometry: DisplayGeometry) -> None:
    # At large angles, exact trig and the small-angle approximation
    # (x_cm = d_cm * radians(deg)) diverge measurably; verify we are NOT
    # doing the small-angle approximation.
    deg = 30.0
    exact_px = geometry.deg_to_px(deg)
    small_angle_cm = geometry.viewing_distance_cm * math.radians(deg)
    small_angle_px = small_angle_cm * geometry.px_per_cm
    assert exact_px != pytest.approx(small_angle_px, rel=1e-6)
    assert abs(exact_px - small_angle_px) > 1.0


def test_px_per_deg_at_center(geometry: DisplayGeometry) -> None:
    assert geometry.px_per_deg_at_center == pytest.approx(geometry.deg_to_px(1.0))


def test_nyquist_cpd(geometry: DisplayGeometry) -> None:
    assert geometry.nyquist_cpd == pytest.approx(geometry.px_per_deg_at_center / 2.0)


def test_max_flicker_hz(geometry: DisplayGeometry) -> None:
    assert geometry.max_flicker_hz == pytest.approx(30.0)


@pytest.mark.parametrize(
    ("ms", "refresh_hz", "expected_frames"),
    [
        (0.0, 60.0, 1),  # clamped to minimum of 1 frame
        (16.67, 60.0, 1),
        (33.33, 60.0, 2),
        (100.0, 60.0, 6),
        (8.0, 60.0, 1),  # rounds up from ~0.48 frames, clamped to 1
        (1000.0, 60.0, 60),
    ],
)
def test_frames_for_ms(ms: float, refresh_hz: float, expected_frames: int) -> None:
    geometry = DisplayGeometry(
        width_px=1920,
        height_px=1080,
        width_cm=53.13,
        height_cm=29.88,
        viewing_distance_cm=57.0,
        refresh_hz=refresh_hz,
    )
    assert geometry.frames_for_ms(ms) == expected_frames


def test_frames_for_ms_minimum_is_one(geometry: DisplayGeometry) -> None:
    assert geometry.frames_for_ms(0.0) >= 1
    assert geometry.frames_for_ms(-5.0) >= 1


def test_ms_for_frames(geometry: DisplayGeometry) -> None:
    assert geometry.ms_for_frames(60) == pytest.approx(1000.0)
    assert geometry.ms_for_frames(1) == pytest.approx(1000.0 / 60.0)


def test_frames_ms_roundtrip_is_stable(geometry: DisplayGeometry) -> None:
    for n in (1, 2, 5, 10, 30, 120):
        ms = geometry.ms_for_frames(n)
        assert geometry.frames_for_ms(ms) == n


def test_geometry_is_immutable(geometry: DisplayGeometry) -> None:
    with pytest.raises(Exception):  # noqa: B017 - pydantic frozen model raises
        geometry.width_px = 100  # type: ignore[misc]
