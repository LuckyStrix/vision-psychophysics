"""`vpsych-data`: command-line interface to the data layer.

Subcommands: `init`, `validate`, `reanalyze`, `export`, `rebuild-catalog`.
Every subcommand accepts `--root` to point at a data root other than the
default (`$VPSYCH_DATA_ROOT` or `~/vpsych-data`).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
    }
    return handlers[args.command](args, root)


if __name__ == "__main__":
    raise SystemExit(main())
