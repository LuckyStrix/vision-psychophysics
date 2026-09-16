"""Tests for the shared atomic-write primitive used throughout vpsych.data."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from vpsych.data.writer import _atomic_write_bytes


def test_atomic_write_creates_file(tmp_path: Path) -> None:
    path = tmp_path / "out.json"
    _atomic_write_bytes(path, b'{"a": 1}')
    assert path.read_bytes() == b'{"a": 1}'
    # No leftover temp files.
    assert list(tmp_path.iterdir()) == [path]


def test_atomic_write_leaves_original_untouched_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "out.json"
    path.write_bytes(b"original content")

    def _boom(*args: object, **kwargs: object) -> None:
        raise OSError("simulated failure during rename")

    monkeypatch.setattr(os, "replace", _boom)

    with pytest.raises(OSError, match="simulated failure"):
        _atomic_write_bytes(path, b"new content")

    assert path.read_bytes() == b"original content"
    # The temp file was cleaned up, not left behind.
    leftovers = [p for p in tmp_path.iterdir() if p != path]
    assert leftovers == []


def test_atomic_write_no_partial_file_on_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "out.json"

    class _BoomFile:
        def write(self, data: bytes) -> int:
            raise OSError("simulated write failure")

        def flush(self) -> None:
            pass

        def fileno(self) -> int:
            return 0

        def __enter__(self) -> _BoomFile:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    import vpsych.data.writer as writer_mod

    monkeypatch.setattr(writer_mod.os, "fdopen", lambda fd, mode: _BoomFile())

    with pytest.raises(OSError, match="simulated write failure"):
        _atomic_write_bytes(path, b"data")

    assert not path.exists()
    assert list(tmp_path.iterdir()) == []
