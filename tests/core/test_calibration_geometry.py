"""Unit tests for vpsych.core.calibration.geometry: credit-card width estimation, geometry building."""

from __future__ import annotations

import pytest

from vpsych.core.calibration.geometry import (
    ID1_CARD_HEIGHT_MM,
    ID1_CARD_WIDTH_MM,
    DisplayQueryError,
    build_display_geometry,
    estimate_display_width_cm,
    px_per_cm_from_card_match,
    query_os_resolution_refresh,
)
from vpsych.core.display import DisplayGeometry


def test_id1_card_dimensions() -> None:
    # ISO/IEC 7810 ID-1 format.
    assert ID1_CARD_WIDTH_MM == 85.60
    assert ID1_CARD_HEIGHT_MM == 53.98


def test_px_per_cm_from_card_match() -> None:
    # If 100 px matched the card's 8.56 cm width, density is ~11.68 px/cm.
    density = px_per_cm_from_card_match(100.0)
    assert density == pytest.approx(100.0 / 8.56)


def test_px_per_cm_from_card_match_rejects_non_positive() -> None:
    with pytest.raises(ValueError):
        px_per_cm_from_card_match(0.0)
    with pytest.raises(ValueError):
        px_per_cm_from_card_match(-10.0)


def test_estimate_display_width_cm_self_consistent() -> None:
    # A card matched at exactly 1/20th of a 1920px-wide screen implies the
    # screen is 20 cards wide, i.e. 20 * 8.56 cm.
    card_width_px = 1920.0 / 20.0
    width_cm = estimate_display_width_cm(card_width_px, horizontal_resolution_px=1920)
    assert width_cm == pytest.approx(20 * 8.56)


def test_estimate_display_width_cm_matches_known_monitor() -> None:
    # A representative 24" 1920x1080 monitor (~53.13 cm wide): compute what
    # pixel width a card match would produce, then recover the width from it.
    true_width_cm = 53.13
    px_per_cm = 1920 / true_width_cm
    card_width_px = px_per_cm * 8.56
    recovered = estimate_display_width_cm(card_width_px, 1920)
    assert recovered == pytest.approx(true_width_cm, rel=1e-9)


def test_estimate_display_width_cm_rejects_non_positive() -> None:
    with pytest.raises(ValueError):
        estimate_display_width_cm(0.0, 1920)
    with pytest.raises(ValueError):
        estimate_display_width_cm(100.0, 0)


def test_build_display_geometry_returns_display_geometry() -> None:
    geom = build_display_geometry(
        width_px=1920,
        height_px=1080,
        width_cm=53.13,
        height_cm=29.88,
        viewing_distance_cm=57.0,
        refresh_hz=60.0,
    )
    assert isinstance(geom, DisplayGeometry)
    assert geom.width_px == 1920
    assert geom.refresh_hz == 60.0


def test_query_os_resolution_refresh_wraps_display_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulate a headless environment (no display server) by making pyglet's
    # display lookup fail, regardless of whether *this* test machine happens
    # to have a display available -- the function must wrap such a failure in
    # DisplayQueryError rather than letting it propagate raw or hang.
    import pyglet

    def _boom() -> None:
        raise RuntimeError("no display server available")

    monkeypatch.setattr(pyglet.canvas, "get_display", _boom)
    with pytest.raises(DisplayQueryError):
        query_os_resolution_refresh()


def test_query_os_resolution_refresh_wraps_missing_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    import pyglet

    class _FakeScreen:
        def get_mode(self) -> None:
            return None

    class _FakeDisplay:
        def get_default_screen(self) -> _FakeScreen:
            return _FakeScreen()

    monkeypatch.setattr(pyglet.canvas, "get_display", lambda: _FakeDisplay())
    with pytest.raises(DisplayQueryError, match="display mode"):
        query_os_resolution_refresh()


def test_query_os_resolution_refresh_wraps_missing_refresh_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pyglet

    class _FakeMode:
        width = 1920
        height = 1080
        rate = 0

    class _FakeScreen:
        def get_mode(self) -> _FakeMode:
            return _FakeMode()

    class _FakeDisplay:
        def get_default_screen(self) -> _FakeScreen:
            return _FakeScreen()

    monkeypatch.setattr(pyglet.canvas, "get_display", lambda: _FakeDisplay())
    with pytest.raises(DisplayQueryError, match="refresh rate"):
        query_os_resolution_refresh()


def test_query_os_resolution_refresh_succeeds_with_fake_display(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pyglet

    class _FakeMode:
        width = 2560
        height = 1440
        rate = 144.0

    class _FakeScreen:
        def get_mode(self) -> _FakeMode:
            return _FakeMode()

    class _FakeDisplay:
        def get_default_screen(self) -> _FakeScreen:
            return _FakeScreen()

    monkeypatch.setattr(pyglet.canvas, "get_display", lambda: _FakeDisplay())
    width, height, refresh_hz = query_os_resolution_refresh()
    assert (width, height, refresh_hz) == (2560, 1440, 144.0)
