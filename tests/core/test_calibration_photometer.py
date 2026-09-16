"""Unit tests for vpsych.core.calibration.photometer, using a fake Photometer device."""

from __future__ import annotations

import pytest

from vpsych.core.calibration.photometer import (
    Photometer,
    measure_gamma,
    measure_gamma_per_channel,
    measurement_levels,
)


class _FakePhotometer:
    """Simulates a photometer reading a display with a known gamma per channel."""

    def __init__(
        self, gammas: dict[str, float], lum_min: float = 0.4, lum_max: float = 130.0
    ) -> None:
        self.gammas = gammas
        self.lum_min = lum_min
        self.lum_max = lum_max
        self.calls: list[tuple[str, float]] = []

    def measure_luminance_cdm2(self, channel: str, level: float) -> float:
        self.calls.append((channel, level))
        gamma = self.gammas[channel]
        return self.lum_min + (self.lum_max - self.lum_min) * (level**gamma)


def test_photometer_protocol_satisfied_by_fake() -> None:
    fake = _FakePhotometer({"gray": 2.2})
    assert isinstance(fake, Photometer)


def test_measurement_levels_includes_endpoints() -> None:
    levels = measurement_levels(5)
    assert levels[0] == 0.0
    assert levels[-1] == 1.0
    assert len(levels) == 5


def test_measurement_levels_rejects_too_few() -> None:
    with pytest.raises(ValueError):
        measurement_levels(1)


def test_measure_gamma_recovers_known_gamma() -> None:
    fake = _FakePhotometer({"gray": 2.4}, lum_min=0.5, lum_max=150.0)
    cal = measure_gamma(fake, n_levels=13, channel="gray")
    assert cal.method == "photometer"
    assert cal.gamma_single == pytest.approx(2.4, rel=1e-6)
    assert cal.lum_min_cdm2 == pytest.approx(0.5)
    assert cal.lum_max_cdm2 == pytest.approx(150.0)
    assert cal.measured_points is not None
    assert len(cal.measured_points) == 13
    assert cal.grade == "A"


def test_measure_gamma_calls_photometer_at_each_level() -> None:
    fake = _FakePhotometer({"gray": 2.2})
    measure_gamma(fake, n_levels=9, channel="gray")
    assert len(fake.calls) == 9
    assert all(c[0] == "gray" for c in fake.calls)


def test_measure_gamma_per_channel_recovers_independent_gammas() -> None:
    fake = _FakePhotometer({"r": 2.1, "g": 2.3, "b": 2.5}, lum_min=0.3, lum_max=140.0)
    cal = measure_gamma_per_channel(fake, n_levels=11)
    assert cal.gamma_r == pytest.approx(2.1, rel=1e-6)
    assert cal.gamma_g == pytest.approx(2.3, rel=1e-6)
    assert cal.gamma_b == pytest.approx(2.5, rel=1e-6)
    assert cal.gamma_single is None


def test_open_psychopy_photometer_lazy_import_does_not_crash_module_import() -> None:
    # Importing the module itself must not import psychopy; only calling
    # open_psychopy_photometer() should attempt it (and fail without hardware).
    import vpsych.core.calibration.photometer as photometer_module

    assert hasattr(photometer_module, "open_psychopy_photometer")
