"""Tests for vpsych.data.cli: the `vpsych-data` console script's argument handling."""

from __future__ import annotations

import json
from pathlib import Path

from vpsych.data import cli

from .conftest import build_full_session


def test_init_command(tmp_path: Path, capsys: object) -> None:
    rc = cli.main(["--root", str(tmp_path), "init", "--name", "cli test dataset"])
    assert rc == 0
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    out = json.loads(captured.out)
    assert out["name"] == "cli test dataset"
    assert (tmp_path / "dataset_description.json").exists()


def test_validate_command_clean_dataset(tmp_path: Path, capsys: object) -> None:
    build_full_session(tmp_path)
    rc = cli.main(["--root", str(tmp_path), "validate"])
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    report = json.loads(captured.out)
    assert report["issues"] == [] or all(issue["severity"] != "error" for issue in report["issues"])
    assert rc == 0


def test_validate_command_missing_dataset_returns_nonzero(tmp_path: Path) -> None:
    rc = cli.main(["--root", str(tmp_path), "validate"])
    assert rc == 1


def test_validate_command_specific_session(tmp_path: Path, capsys: object) -> None:
    pid, sid, _ = build_full_session(tmp_path)
    rc = cli.main(
        ["--root", str(tmp_path), "validate", "--participant-id", pid, "--session-id", sid]
    )
    assert rc == 0


def test_validate_command_session_without_participant_id_errors(tmp_path: Path) -> None:
    rc = cli.main(["--root", str(tmp_path), "validate", "--session-id", "ses-20260916T103000"])
    assert rc == 2


def test_rebuild_catalog_command(tmp_path: Path, capsys: object) -> None:
    build_full_session(tmp_path)
    rc = cli.main(["--root", str(tmp_path), "rebuild-catalog"])
    assert rc == 0
    assert (tmp_path / "catalog.sqlite").exists()


def test_export_command(tmp_path: Path, capsys: object) -> None:
    build_full_session(tmp_path)
    out_zip = tmp_path.parent / "cli_export.zip"
    rc = cli.main(["--root", str(tmp_path), "export", str(out_zip)])
    assert rc == 0
    assert out_zip.exists()


def test_reanalyze_command(tmp_path: Path, capsys: object, registered_dummy_test: object) -> None:
    pid, sid, _ = build_full_session(tmp_path)
    session_dir = tmp_path / pid / sid
    rc = cli.main(["--root", str(tmp_path), "reanalyze", str(session_dir)])
    assert rc == 0
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    results = json.loads(captured.out)
    assert len(results) == 1
    assert results[0]["matches"] is True
