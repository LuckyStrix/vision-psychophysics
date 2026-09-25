"""Display geometry calibration: credit-card screen-width estimation and OS queries.

Physical screen width can be estimated without a ruler by displaying a
rectangle on screen, resizing it until it visually matches a real ID-1
format card (a credit/debit card, most driver's licenses, etc.) held up to
the screen, and reading off the rectangle's pixel width -- since the card's
physical width is fixed by international standard. Resolution and refresh
rate, when not supplied by the user, come from the OS via a lazy
(psychopy/pyglet) import so this module stays headless-importable.
"""

from __future__ import annotations

from vpsych.core.display import DisplayGeometry

# ISO/IEC 7810 ID-1 format: the standard size for credit/debit cards and
# most national ID/driver's license cards. Width 85.60 mm, height 53.98 mm.
ID1_CARD_WIDTH_MM = 85.60
ID1_CARD_HEIGHT_MM = 53.98


def px_per_cm_from_card_match(card_width_px: float) -> float:
    """Compute pixel density implied by a credit-card-matched on-screen rectangle width.

    Args:
        card_width_px: Pixel width of an on-screen rectangle the observer
            has resized to visually match a real ISO/IEC 7810 ID-1 card
            (85.60 mm wide) held against the screen.

    Returns:
        Pixel density, in pixels per centimetre.

    Raises:
        ValueError: If ``card_width_px`` is not positive.
    """
    if card_width_px <= 0:
        raise ValueError(f"card_width_px must be positive, got {card_width_px}")
    return card_width_px / (ID1_CARD_WIDTH_MM / 10.0)


def estimate_display_width_cm(card_width_px: float, horizontal_resolution_px: int) -> float:
    """Estimate a display's physical width from a credit-card pixel-width match.

    Args:
        card_width_px: Pixel width of an on-screen rectangle the observer
            has resized to visually match a real ISO/IEC 7810 ID-1 card
            (85.60 mm wide) held against the screen.
        horizontal_resolution_px: The display's total horizontal
            resolution, in pixels.

    Returns:
        Estimated physical screen width, in centimetres:
        ``horizontal_resolution_px / (card_width_px / (ID1_CARD_WIDTH_MM / 10))``.

    Raises:
        ValueError: If ``card_width_px`` or ``horizontal_resolution_px`` is
            not positive.
    """
    if card_width_px <= 0:
        raise ValueError(f"card_width_px must be positive, got {card_width_px}")
    if horizontal_resolution_px <= 0:
        raise ValueError(
            f"horizontal_resolution_px must be positive, got {horizontal_resolution_px}"
        )
    return horizontal_resolution_px / px_per_cm_from_card_match(card_width_px)


def build_display_geometry(
    width_px: int,
    height_px: int,
    width_cm: float,
    height_cm: float,
    viewing_distance_cm: float,
    refresh_hz: float,
) -> DisplayGeometry:
    """Construct a :class:`DisplayGeometry` from calibration-wizard inputs.

    Thin, explicit constructor kept alongside the geometry-estimation
    helpers in this module so calibration UI code has one obvious place to
    assemble the final geometry object (validation is enforced by
    ``DisplayGeometry`` itself, a pydantic model).

    Args:
        width_px: Horizontal display resolution, in pixels.
        height_px: Vertical display resolution, in pixels.
        width_cm: Physical screen width, in centimetres (e.g. from
            :func:`estimate_display_width_cm`, or a direct ruler
            measurement).
        height_cm: Physical screen height, in centimetres.
        viewing_distance_cm: Eye-to-screen distance, in centimetres.
        refresh_hz: Display refresh rate, in hertz.

    Returns:
        The constructed :class:`DisplayGeometry`.
    """
    return DisplayGeometry(
        width_px=width_px,
        height_px=height_px,
        width_cm=width_cm,
        height_cm=height_cm,
        viewing_distance_cm=viewing_distance_cm,
        refresh_hz=refresh_hz,
    )


class DisplayQueryError(RuntimeError):
    """Raised when the OS-reported display resolution/refresh rate cannot be determined."""


#: Above this the OS-reported "refresh rate" is almost certainly a driver placeholder.
_MAX_PLAUSIBLE_REFRESH_HZ = 500.0


def query_os_resolution_refresh() -> tuple[int, int, float]:
    """Query the primary display's resolution and refresh rate from the OS.

    Uses a lazily imported ``psychopy`` (via pyglet) to enumerate the
    primary screen, per the project's rule that display-dependent imports
    happen only inside the function that needs them, never at module import
    time -- so this module stays importable (though this specific function
    is not callable) in headless CI.

    Returns:
        A ``(width_px, height_px, refresh_hz)`` tuple for the primary
        display.

    Raises:
        DisplayQueryError: If no display server is available, or the
            resolution/refresh rate could not be determined (e.g. running
            headless, or the reported refresh rate is non-positive/unknown).
    """
    try:
        import pyglet
    except Exception as exc:  # pragma: no cover - exercised only with a real display stack
        raise DisplayQueryError(f"Could not import pyglet to query the display: {exc}") from exc

    try:
        display = pyglet.canvas.get_display()
        screen = display.get_default_screen()
        mode = screen.get_mode()
    except Exception as exc:  # pragma: no cover - exercised only with a real display stack
        raise DisplayQueryError(f"Could not query the OS display mode: {exc}") from exc

    if mode is None or not mode.width or not mode.height:
        raise DisplayQueryError("OS did not report a usable display mode (width/height).")

    refresh_hz = getattr(mode, "rate", None)
    if refresh_hz and refresh_hz > _MAX_PLAUSIBLE_REFRESH_HZ:
        raise DisplayQueryError(
            f"OS reported an implausible refresh rate ({refresh_hz:g} Hz); supply it manually."
        )
    if not refresh_hz or refresh_hz <= 0:
        raise DisplayQueryError(
            "OS did not report a usable (positive) refresh rate; supply it manually."
        )

    return int(mode.width), int(mode.height), float(refresh_hz)
