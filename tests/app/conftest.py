"""Shared fixtures for `vpsych.app` tests.

Reuses `tests.data.conftest`'s helpers (a real data root, calibration,
display geometry, a registered dummy test) as plain imports -- not pytest
fixture inheritance across package boundaries, just ordinary Python
imports, since `tests/data/conftest.py` lives in a proper package
(`tests/data/__init__.py` exists).
"""

from __future__ import annotations

import os

import pytest

from tests.data.conftest import (
    DummyParams,
    DummyTest,
    make_calibration,
    make_display,
    registered_dummy_test,
)

__all__ = ["DummyParams", "DummyTest", "make_calibration", "make_display", "registered_dummy_test"]


def _offscreen_available() -> bool:
    """Whether PySide6 can be driven with the `offscreen` Qt platform plugin."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication([])
        del app
        return True
    except Exception:
        return False


requires_offscreen_qt = pytest.mark.skipif(
    not _offscreen_available(), reason="offscreen Qt platform plugin is not available"
)


@pytest.fixture
def qapp():
    """A shared `QApplication` instance for widget-instantiating tests (offscreen platform)."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app
