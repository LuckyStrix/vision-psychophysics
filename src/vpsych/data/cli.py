"""`vpsych-data`: command-line interface to the data layer.

Subcommands: `init`, `validate`, `reanalyze`, `export`, `rebuild-catalog`,
`import-calsuite`. Every subcommand accepts `--root` to point at a data
root other than the default (`$VPSYCH_DATA_ROOT` or `~/vpsych-data`).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import vpsych
from vpsych.core.calibration import calsuite_import
from vpsych.core.calibration.geometry import build_display_geometry
from vpsych.core.calibration.models import Calibration, EnvironmentChecklist
from vpsych.data import catalog, dataset, export, paths, reanalyze, validate


def build_parser() -> argparse.ArgumentParser:
    """Build the `vpsych-data` argument parser.

    Returns:
        The configured `argparse.ArgumentParser`, with one subparser per
        subcommand (`init`, `validate`, `reanalyze`, `export`,
        `rebuild-catalog`).
    """
    parser = argparse.ArgumentParser(
        prog="vpsych-data", description="Manage a vpsych data root: validate, reanalyze, export."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Data root (default: $VPSYCH_DATA_ROOT or ~/vpsych-data).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="Initialize a data root.")
    p_init.add_argument("--name", default="vpsych dataset", help="Human-readable dataset name.")
    p_init.add_argument(
        "--force", action="store_true", help="Overwrite an existing dataset_description.json."
    )

    p_validate = sub.add_parser("validate", help="Validate a session, or the whole dataset.")
    p_validate.add_argument(
        "--participant-id", default=None, help="Validate only this participant."
    )
    p_validate.add_argument(
        "--session-id",
        default=None,
        help="Validate only this session (requires --participant-id).",
    )

    p_reanalyze = sub.add_parser(
        "reanalyze", help="Recompute summaries for a session from raw trials."
    )
    p_reanalyze.add_argument("session_path", type=Path, help="Path to the session directory.")
    p_reanalyze.add_argument(
        "--write", action="store_true", help="Write recomputed summaries alongside the originals."
    )

    p_export = sub.add_parser("export", help="Export a participant or the whole dataset to a zip.")
    p_export.add_argument("out_zip", type=Path, help="Destination zip file.")
    p_export.add_argument("--participant-id", default=None, help="Export only this participant.")

    sub.add_parser("rebuild-catalog", help="Rebuild catalog.sqlite from the files on disk.")

    p_import = sub.add_parser(
        "import-calsuite",
        help="Build and save a Calibration from calsuite display record(s).",
        description=(
            "Import a calsuite display.nominal and/or display.measurement JSON record "
            "(shared/calibration_suite). calsuite has no notion of pixel resolution, refresh "
            "rate, viewing distance, or the environment checklist -- those must always be "
            "supplied here. Refused or untrustworthy calsuite data (status=refused, "
            "non-measured provenance, synthetic backend) is reported and never imported."
        ),
    )
    p_import.add_argument(
        "--nominal", type=Path, default=None, help="Path to a calsuite display.nominal-*.json record."
    )
    p_import.add_argument(
        "--measurement",
        type=Path,
        default=None,
        help="Path to a calsuite display.measurement-*.json record.",
    )
    p_import.add_argument(
        "--width-cm",
        type=float,
        default=None,
        help="Physical screen width in cm, if no --nominal record supplies one.",
    )
    p_import.add_argument(
        "--height-cm",
        type=float,
        default=None,
        help="Physical screen height in cm, if no --nominal record supplies one.",
    )
    p_import.add_argument("--width-px", type=int, required=True, help="Horizontal resolution in pixels.")
    p_import.add_argument("--height-px", type=int, required=True, help="Vertical resolution in pixels.")
    p_import.add_argument(
        "--viewing-distance-cm", type=float, required=True, help="Eye-to-screen distance in cm."
    )
    p_import.add_argument("--refresh-hz", type=float, required=True, help="Display refresh rate in Hz.")
    p_import.add_argument(
        "--room-lighting-controlled",
        action="store_true",
        help="Confirm: room lighting is dim and free of glare on the screen.",
    )
    p_import.add_argument(
        "--monitor-warmed-up",
        action="store_true",
        help="Confirm: monitor has been powered on for at least 15 minutes.",
    )
    p_import.add_argument(
        "--night-light-disabled",
        action="store_true",
        help="Confirm: OS night-light/blue-light filter is disabled.",
    )
    p_import.add_argument(
        "--hdr-disabled",
        action="store_true",
        help="Confirm: OS HDR/adaptive-brightness/auto-contrast is disabled.",
    )
    p_import.add_argument(
        "--environment-notes", default=None, help="Free-text environment notes (must stay pseudonymous)."
    )
    p_import.add_argument("--notes", default=None, help="Free-text notes for the Calibration record.")
    p_import.add_argument(
        "--force", action="store_true", help="Overwrite an existing calibration file with this content hash."
    )

    return parser


def _cmd_init(args: argparse.Namespace, root: Path) -> int:
    description = dataset.init_dataset(root, name=args.name, force=args.force)
    print(description.model_dump_json(indent=2))
    return 0


def _cmd_validate(args: argparse.Namespace, root: Path) -> int:
    if args.session_id is not None:
        if args.participant_id is None:
            print("--session-id requires --participant-id", file=sys.stderr)
            return 2
        report = validate.validate_session(args.participant_id, args.session_id, root)
    else:
        report = validate.validate_dataset(root)
    print(report.model_dump_json(indent=2))
    return 0 if report.ok else 1


def _cmd_reanalyze(args: argparse.Namespace, root: Path) -> int:
    del root  # reanalyze operates on an explicit session path, not a data root
    results = reanalyze.reanalyze_session(args.session_path, write=args.write)
    summary = [
        {
            "task_id": r.task_id,
            "eye": r.eye,
            "run": r.run,
            "matches": r.matches,
            "differing_fields": r.differing_fields,
            "written_path": str(r.written_path) if r.written_path else None,
        }
        for r in results
    ]
    print(json.dumps(summary, indent=2))
    return 0 if all(r.matches for r in results) else 1


def _cmd_export(args: argparse.Namespace, root: Path) -> int:
    if args.participant_id:
        out = export.export_participant(args.participant_id, args.out_zip, root)
    else:
        out = export.export_dataset(args.out_zip, root)
    print(str(out))
    return 0


def _cmd_rebuild_catalog(args: argparse.Namespace, root: Path) -> int:
    del args
    catalog.rebuild_catalog(root)
    print(f"Rebuilt {paths.catalog_sqlite_path(root)}")
    return 0


def _environment_checklist_warnings(env: EnvironmentChecklist) -> list[str]:
    items = {
        "room_lighting_controlled": "room lighting is not confirmed dim/glare-free",
        "monitor_warmed_up": "monitor warm-up (15+ minutes) is not confirmed",
        "night_light_disabled": "OS night-light/blue-light filter is not confirmed disabled",
        "hdr_disabled": "OS HDR/adaptive-brightness is not confirmed disabled",
    }
    return [msg for attr, msg in items.items() if not getattr(env, attr)]


def _cmd_import_calsuite(args: argparse.Namespace, root: Path) -> int:
    if args.nominal is None and args.measurement is None:
        print("error: at least one of --nominal / --measurement is required.", file=sys.stderr)
        return 2

    try:
        nominal_raw = calsuite_import.load_calsuite_json(args.nominal) if args.nominal else None
        measurement_raw = (
            calsuite_import.load_calsuite_json(args.measurement) if args.measurement else None
        )
    except calsuite_import.CalsuiteImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    result = calsuite_import.import_calsuite_records(nominal=nominal_raw, measurement=measurement_raw)

    if result.rejected:
        print("REJECTED (not imported):", file=sys.stderr)
        for reason in result.rejected:
            print(f"  - {reason}", file=sys.stderr)
    if result.warnings:
        print("WARNINGS:", file=sys.stderr)
        for warning in result.warnings:
            print(f"  - {warning}", file=sys.stderr)

    width_cm = result.width_cm if result.width_cm is not None else args.width_cm
    height_cm = result.height_cm if result.height_cm is not None else args.height_cm

    still_needed = []
    if result.gamma is None:
        still_needed.append(
            "gamma/luminance calibration (no usable display.measurement record -- see REJECTED above)"
        )
    if result.color is None:
        still_needed.append(
            "color calibration (no usable display.measurement record -- see REJECTED above)"
        )
    if width_cm is None:
        still_needed.append("physical screen width (no --nominal record and no --width-cm given)")
    if height_cm is None:
        still_needed.append("physical screen height (no --nominal record and no --height-cm given)")
    if still_needed:
        print("Cannot complete a calibration from calsuite data alone. Still needed:", file=sys.stderr)
        for item in still_needed:
            print(f"  - {item}", file=sys.stderr)
        print(
            "Complete the missing step(s) in the vpsych calibration wizard instead, or "
            "re-run with a display.measurement record / --width-cm / --height-cm as noted above.",
            file=sys.stderr,
        )
        return 1

    environment = EnvironmentChecklist(
        room_lighting_controlled=args.room_lighting_controlled,
        monitor_warmed_up=args.monitor_warmed_up,
        night_light_disabled=args.night_light_disabled,
        hdr_disabled=args.hdr_disabled,
        notes=args.environment_notes,
    )
    for warning in _environment_checklist_warnings(environment):
        print(f"warning: environment checklist -- {warning}", file=sys.stderr)

    geometry = build_display_geometry(
        width_px=args.width_px,
        height_px=args.height_px,
        width_cm=width_cm,
        height_cm=height_cm,
        viewing_distance_cm=args.viewing_distance_cm,
        refresh_hz=args.refresh_hz,
    )
    notes = result.notes
    if args.notes:
        notes = f"{notes}\n{args.notes}" if notes else args.notes
    calibration = Calibration(
        created_utc=result.source_created_utc or datetime.now(timezone.utc),
        geometry=geometry,
        gamma=result.gamma,
        color=result.color,
        environment=environment,
        software_version=vpsych.__version__,
        notes=notes,
    )
    path = dataset.save_calibration(calibration, root=root, force=args.force)
    print(f"Saved: {path}")
    print(
        f"Luminance grade {calibration.luminance_grade} ({calibration.gamma.method}); "
        f"color grade {calibration.color_grade} ({calibration.color.method})."
    )
    print(f"Content hash: {calibration.content_hash()}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for the `vpsych-data` console script.

    Args:
        argv: Arguments to parse, or `None` to use `sys.argv[1:]`.

    Returns:
        Process exit code: `0` on success, `1` if validation/reanalysis
        found problems, `2` for a usage error.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    root = args.root or paths.data_root()

    handlers = {
        "init": _cmd_init,
        "validate": _cmd_validate,
        "reanalyze": _cmd_reanalyze,
        "export": _cmd_export,
        "rebuild-catalog": _cmd_rebuild_catalog,
        "import-calsuite": _cmd_import_calsuite,
    }
    return handlers[args.command](args, root)


if __name__ == "__main__":
    raise SystemExit(main())
