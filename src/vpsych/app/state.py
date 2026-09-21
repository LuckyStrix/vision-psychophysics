"""Shared application state: data root, current participant, calibration cache.

A single `AppState` (a `QObject` so screens can connect to its signals) is
constructed once in `vpsych.app.main` and handed to every screen. It never
touches the filesystem on its own except through `vpsych.data`/
`vpsych.core.calibration` calls -- no ad-hoc participant-data writes.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Signal

from vpsych.core.calibration.models import Calibration
from vpsych.data import dataset, paths


class AppState(QObject):
    """Mutable app-wide state shared by every screen.

    Signals:
        participant_changed: Emitted with the new `participant_id` (or `""`)
            whenever the selected participant changes.
        calibration_changed: Emitted whenever the cached "latest calibration"
            is refreshed.
    """

    participant_changed = Signal(str)
    calibration_changed = Signal()

    def __init__(self, data_root: Path | None = None) -> None:
        super().__init__()
        self._data_root = data_root or paths.data_root()
        self._participant_id: str | None = None
        self._latest_calibration: Calibration | None = None
        self.ensure_dataset_initialized()
        self.refresh_calibration()

    @property
    def data_root(self) -> Path:
        """The active vpsych data root."""
        return self._data_root

    @property
    def participant_id(self) -> str | None:
        """The currently selected participant's `sub-XXXX` ID, or `None`."""
        return self._participant_id

    def set_participant(self, participant_id: str | None) -> None:
        """Select a participant (or clear the selection with `None`)."""
        if participant_id == self._participant_id:
            return
        self._participant_id = participant_id
        self.participant_changed.emit(participant_id or "")

    def ensure_dataset_initialized(self) -> None:
        """Initialize the data root (idempotent) if it hasn't been already."""
        dataset.init_dataset(self._data_root)

    def refresh_calibration(self) -> None:
        """Reload the most recent calibration from disk and emit `calibration_changed`."""
        self._latest_calibration = dataset.latest_calibration(self._data_root)
        self.calibration_changed.emit()

    @property
    def latest_calibration(self) -> Calibration | None:
        """The most recently created stored calibration, or `None` if uncalibrated."""
        return self._latest_calibration
