"""Regression: textures built top-row-first must appear upright on a real PsychoPy window.

PsychoPy/OpenGL treats row 0 of an `ImageStim` array as the *bottom* of the
image, so every test that hands it a top-first array must `np.flipud` it
first (a missed flip mirrored the Landolt C vertically and scored correct
answers as wrong). Needs a real window; run with `pytest -m display`.
"""

from __future__ import annotations

import numpy as np
import pytest

from vpsych.tests_catalog.visual_acuity.optotype import render_landolt_c


def _draw_and_grab(win, stim) -> np.ndarray:  # type: ignore[no-untyped-def]
    stim.draw()
    win.flip()
    stim.draw()
    return np.array(win.getMovieFrame(buffer="back").convert("L"), dtype=float)


@pytest.mark.display
def test_imagestim_row0_is_screen_bottom_so_flipud_is_required() -> None:
    from psychopy import visual

    win = visual.Window((200, 200), units="pix", color=0, fullscr=False, allowGUI=False)
    try:
        tex = -np.ones((64, 64))
        tex[:16, :] = 1.0  # bright band in the array's first (top) rows
        stim = visual.ImageStim(win, image=np.flipud(tex), size=(64, 64), interpolate=False)
        buf = _draw_and_grab(win, stim)
        assert buf[68:84, 90:110].mean() > buf[116:132, 90:110].mean()  # bright band on top
    finally:
        win.close()


@pytest.mark.display
@pytest.mark.parametrize("orientation_deg", [90, 270])
def test_landolt_gap_shows_at_requested_up_down_orientation(orientation_deg: int) -> None:
    from psychopy import visual

    win = visual.Window((300, 300), units="pix", color=0, fullscr=False, allowGUI=False)
    try:
        tex = render_landolt_c(100, gap_px=16.0, orientation_deg=orientation_deg, supersample=2)
        stim = visual.ImageStim(win, image=np.flipud(tex), size=(100, 100), interpolate=False)
        buf = _draw_and_grab(win, stim)
        top = buf[150 - 40 : 150 - 25, 140:160].mean()
        bottom = buf[150 + 25 : 150 + 40, 140:160].mean()
        gap_side_is_brighter = top > bottom if orientation_deg == 90 else bottom > top
        assert gap_side_is_brighter  # ring stroke is dark; the gap side is background
    finally:
        win.close()
