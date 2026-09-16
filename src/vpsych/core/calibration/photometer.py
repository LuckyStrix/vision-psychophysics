"""Photometer-driven gamma measurement (grade A luminance calibration).

Thin orchestration around a photometer device: generate the sequence of
drive levels to measure per channel, ask a photometer object for a
luminance reading at each, and fit the result into a
:class:`vpsych.core.calibration.models.GammaCalibration`. The photometer
itself is accessed only through the small :class:`Photometer` protocol
below, so this module is fully unit-testable with a fake/simulated device
(no real hardware, no psychopy import) -- only
:func:`open_psychopy_photometer` touches ``psychopy.hardware``, and it does
so lazily.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from vpsych.core.calibration.gamma import fit_gamma
from vpsych.core.calibration.models import GammaCalibration, GammaCalibrationPoint

Channel = str  # "r", "g", "b", or "gray" (all channels driven together)


@runtime_checkable
class Photometer(Protocol):
    """Protocol for a device that measures screen luminance at a given drive level.

    Concrete implementations wrap a real PsychoPy-supported photometer (see
    :func:`open_psychopy_photometer`) or, in tests, a fake that returns
    synthetic luminance values from a known ground-truth gamma curve.
    """

    def measure_luminance_cdm2(self, channel: Channel, level: float) -> float:
        """Set ``channel`` to normalized drive ``level`` and return measured luminance.

        Args:
            channel: Which channel to drive: ``"r"``, ``"g"``, ``"b"``, or
                ``"gray"`` (all three channels driven equally, for a
                single-gamma measurement).
            level: Normalized drive level in ``[0, 1]``.

        Returns:
            Measured luminance at that level, in cd/m^2.
        """
        ...


def measurement_levels(n_levels: int) -> list[float]:
    """Generate evenly spaced normalized drive levels to measure, including both endpoints.

    Args:
        n_levels: Number of levels to generate (at least 2, so black and
            white are both included).

    Returns:
        A list of ``n_levels`` values evenly spaced over ``[0, 1]``
        inclusive.

    Raises:
        ValueError: If ``n_levels`` is less than 2.
    """
    if n_levels < 2:
        raise ValueError(f"n_levels must be at least 2, got {n_levels}")
    return list(np.linspace(0.0, 1.0, n_levels))


def measure_gamma(
    photometer: Photometer,
    n_levels: int = 17,
    channel: Channel = "gray",
) -> GammaCalibration:
    """Run a full measurement sequence and fit a photometer-grade gamma calibration.

    Steps the display through :func:`measurement_levels` on ``channel``,
    reads luminance from ``photometer`` at each level, fits the power-law
    gamma model (see :func:`vpsych.core.calibration.gamma.fit_gamma`), and
    packages the result as a grade-A (``method="photometer"``)
    :class:`GammaCalibration`, retaining the raw measured points for
    provenance/re-fitting.

    Args:
        photometer: The (real or fake) photometer to read from.
        n_levels: Number of drive levels to sample (default 17: 0, 1/16,
            2/16, ..., 1).
        channel: Which channel to measure (``"gray"`` measures a single
            combined-channel gamma, used to populate ``gamma_single``).

    Returns:
        A :class:`GammaCalibration` with ``method="photometer"``,
        ``gamma_single`` set from the fit, and ``measured_points``
        populated.
    """
    levels = measurement_levels(n_levels)
    points = [
        GammaCalibrationPoint(
            input_level=level, luminance_cdm2=photometer.measure_luminance_cdm2(channel, level)
        )
        for level in levels
    ]
    fit = fit_gamma(points)
    return GammaCalibration(
        method="photometer",
        gamma_single=fit.gamma,
        lum_min_cdm2=fit.lum_min_cdm2,
        lum_max_cdm2=fit.lum_max_cdm2,
        measured_points=points,
    )


def measure_gamma_per_channel(
    photometer: Photometer,
    n_levels: int = 17,
) -> GammaCalibration:
    """Like :func:`measure_gamma`, but fits red, green, and blue channels independently.

    Args:
        photometer: The (real or fake) photometer to read from.
        n_levels: Number of drive levels to sample per channel.

    Returns:
        A :class:`GammaCalibration` with ``method="photometer"`` and
        ``gamma_r``/``gamma_g``/``gamma_b`` set independently.
        ``lum_min_cdm2``/``lum_max_cdm2`` and ``measured_points`` are taken
        from the red channel's sweep (the channels' black/white points
        should agree closely for a well-behaved display; per-channel
        detail lives in the fitted gammas).
    """
    fits = {}
    points_by_channel = {}
    for channel in ("r", "g", "b"):
        levels = measurement_levels(n_levels)
        points = [
            GammaCalibrationPoint(
                input_level=level,
                luminance_cdm2=photometer.measure_luminance_cdm2(channel, level),
            )
            for level in levels
        ]
        points_by_channel[channel] = points
        fits[channel] = fit_gamma(points)

    return GammaCalibration(
        method="photometer",
        gamma_r=fits["r"].gamma,
        gamma_g=fits["g"].gamma,
        gamma_b=fits["b"].gamma,
        lum_min_cdm2=fits["r"].lum_min_cdm2,
        lum_max_cdm2=fits["r"].lum_max_cdm2,
        measured_points=points_by_channel["r"],
    )


def open_psychopy_photometer(device_name: str, port: str | None = None) -> Any:
    """Open a real PsychoPy-supported photometer device, wrapped to satisfy :class:`Photometer`.

    Lazily imports ``psychopy.hardware`` (per the project's rule that
    display/hardware-dependent imports happen only inside the function that
    needs them), so this module remains importable headless; this function
    itself requires the actual hardware and OS drivers and is not exercised
    by the automated test suite.

    Args:
        device_name: Name of the PsychoPy-supported photometer class (e.g.
            ``"PR655"``, ``"CRS_ColorCAL"``), as accepted by
            ``psychopy.hardware.findPhotometer``.
        port: Serial port the device is connected to, or `None` to let
            PsychoPy auto-detect it.

    Returns:
        A live photometer handle satisfying the :class:`Photometer`
        protocol used by :func:`measure_gamma`.

    Raises:
        RuntimeError: If no matching photometer could be found/opened.
    """
    from psychopy import hardware

    device = hardware.findPhotometer(device=device_name, ports=[port] if port else None)
    if device is None:
        raise RuntimeError(f"Could not find/open a {device_name!r} photometer.")
    return _PsychoPyPhotometerAdapter(device)


class _PsychoPyPhotometerAdapter:
    """Adapts a live ``psychopy.hardware`` photometer object to the :class:`Photometer` protocol.

    Assumes the caller sets the display to ``level`` on ``channel`` before
    each reading is taken via other means (e.g. a PsychoPy window draw);
    this adapter only performs the luminance read via the device's
    ``getLum()`` method, since actually drawing the stimulus needs a live
    ``psychopy.visual.Window`` that this module does not own.
    """

    def __init__(self, device: Any) -> None:
        self._device = device

    def measure_luminance_cdm2(self, channel: Channel, level: float) -> float:
        del channel, level  # display state is driven by the caller, not this adapter
        reading = self._device.getLum()
        return float(reading)
