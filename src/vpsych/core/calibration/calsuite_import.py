"""Import display calibration data from `calsuite` (`shared/calibration_suite`).

calsuite writes two kinds of display record as JSON: `display.nominal` (an
EDID read -- physical size and *claimed* chromaticity, `provenance="nominal"`)
and `display.measurement` (a real photometer/colorimeter/camera session --
per-channel tone response, black/white luminance, and measured primaries,
`provenance="measured"`). Both carry calsuite's own provenance/status axes
(see calsuite's `docs/design.md` house rule 2 and `CLAUDE.md`): a record can
be honestly `measured` and still `status="refused"` if its own quality
checks failed, and that must be respected here exactly as calsuite's own
`store.require_exportable` respects it -- a refused record is a finding, not
usable data.

calsuite has no notion of pixel resolution, refresh rate, viewing distance,
or the self-reported environment checklist (room lighting, warm-up,
night-light, HDR) -- those are vpsych-specific and always come from the
wizard or explicit CLI flags. `import_calsuite_records` never invents
values for them; it reports them in `CalsuiteImportResult.missing_required`
instead.
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vpsych.core.calibration.models import ColorCalibration, GammaCalibration, PrimaryChromaticity

#: calsuite measurement backends known to produce a real light measurement.
#: "synthetic" (calsuite's own demo/test data generator) is deliberately
#: excluded -- see the check in `_import_measurement`.
_TRUSTED_BACKENDS = ("argyll", "spectro", "camera")

#: Same threshold as `Calibration.is_stale`'s default, used to warn (not
#: reject) about an old calsuite measurement.
_STALE_AGE_DAYS = 30.0


class CalsuiteImportError(ValueError):
    """Raised for a calsuite import that cannot proceed at all (bad file, no usable record)."""


def load_calsuite_json(path: Path) -> dict[str, Any]:
    """Load and parse a calsuite record JSON file.

    Args:
        path: Path to a `display.nominal-*.json` or `display.measurement-*.json`
            file written by `calsuite`.

    Returns:
        The parsed record as a dict.

    Raises:
        CalsuiteImportError: If the file cannot be read or is not valid JSON.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CalsuiteImportError(f"Could not read {path}: {exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CalsuiteImportError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise CalsuiteImportError(f"{path} does not contain a JSON object.")
    return data


def _parse_calsuite_timestamp(value: str) -> datetime:
    """Parse a calsuite `created` timestamp (`YYYYMMDDTHHMMSSZ`) as UTC."""
    return datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class CalsuiteImportResult:
    """Everything `import_calsuite_records` could -- and couldn't -- extract.

    Attributes:
        width_cm: Physical screen width, from a `display.nominal` record's
            EDID-decoded `physical_size_mm`, or `None` if no usable nominal
            record was given.
        height_cm: Physical screen height, same source.
        gamma: A grade-A `GammaCalibration` built from a `display.measurement`
            record's tone-response curve, or `None` if no usable measurement
            record was given (or it was rejected -- see `rejected`).
        color: A grade-A `ColorCalibration` built from the same measurement
            record's measured primaries, or `None` for the same reasons.
        source_created_utc: Timestamp to use for the resulting `Calibration.created_utc`
            -- the measurement record's own `created` time if one was
            imported (since that is when the display was actually
            characterized), else the nominal record's, else `None`.
        notes: An auto-generated provenance note describing what was
            imported from where, suitable for `Calibration.notes`.
        rejected: Human-readable reasons specific data was refused import
            (a refused calsuite record, an untrusted/synthetic backend, a
            record of the wrong kind, malformed fields). Each entry names
            what was *not* imported and why.
        warnings: Non-blocking concerns about data that *was* imported
            (staleness, a camera-backend accuracy caveat, a
            measurement-only import with no matching nominal record, etc.).
        missing_required: Fields/steps that still need to be supplied by
            hand (directly, or via the wizard) before a `Calibration` can be
            built -- calsuite never provides these, or this particular
            import didn't produce them.
    """

    width_cm: float | None = None
    height_cm: float | None = None
    gamma: GammaCalibration | None = None
    color: ColorCalibration | None = None
    source_created_utc: datetime | None = None
    notes: str | None = None
    rejected: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)
    missing_required: tuple[str, ...] = field(default_factory=tuple)


def _import_nominal(
    nominal: dict[str, Any],
    *,
    rejected: list[str],
    notes: list[str],
) -> tuple[float | None, float | None, datetime | None]:
    """Extract physical size (cm) and a timestamp from a `display.nominal` record."""
    record_id = nominal.get("id", "display.nominal")
    kind = nominal.get("kind")
    if kind != "display.nominal":
        rejected.append(
            f"{record_id}: kind={kind!r}, expected 'display.nominal' -- ignored as a nominal record."
        )
        return None, None, None

    size = nominal.get("result", {}).get("physical_size_mm")
    width_cm = height_cm = None
    if not (isinstance(size, (list, tuple)) and len(size) == 2):
        rejected.append(
            f"{record_id}: missing result.physical_size_mm -- screen size not imported."
        )
    else:
        width_cm, height_cm = size[0] / 10.0, size[1] / 10.0

    created = None
    raw_created = nominal.get("created")
    if isinstance(raw_created, str):
        with contextlib.suppress(ValueError):
            created = _parse_calsuite_timestamp(raw_created)

    if width_cm is not None:
        device_id = nominal.get("device", {}).get("id", "?")
        notes.append(
            f"Physical screen size from calsuite {record_id} (device {device_id}, "
            "provenance nominal/EDID)."
        )
    return width_cm, height_cm, created


def _import_measurement(
    measurement: dict[str, Any],
    *,
    rejected: list[str],
    warnings: list[str],
    notes: list[str],
) -> tuple[GammaCalibration | None, ColorCalibration | None, datetime | None]:
    """Extract gamma/color calibrations and a timestamp from a `display.measurement` record."""
    record_id = measurement.get("id", "display.measurement")
    kind = measurement.get("kind")
    if kind != "display.measurement":
        rejected.append(
            f"{record_id}: kind={kind!r}, expected 'display.measurement' -- ignored as a "
            "measurement record."
        )
        return None, None, None

    status = measurement.get("status")
    provenance = measurement.get("provenance")
    backend = (measurement.get("method", {}).get("params") or {}).get("backend")

    if status != "ok":
        refusals = measurement.get("refusals") or []
        reason = "; ".join(str(r) for r in refusals) or "no reason given in the record"
        rejected.append(
            f"{record_id}: status={status!r} (refused by calsuite's own checks) -- gamma/color "
            f"NOT imported. Reason: {reason}"
        )
        return None, None, None
    if provenance not in ("measured", "derived"):
        rejected.append(
            f"{record_id}: provenance={provenance!r}, not 'measured'/'derived' -- gamma/color "
            "NOT imported (this mirrors calsuite's own store.require_exportable rule: only "
            "measured or derived records are trusted for export)."
        )
        return None, None, None
    if backend == "synthetic":
        rejected.append(
            f"{record_id}: backend='synthetic' -- this is calsuite demo/test data (see "
            "`calsuite demo`), not a real measurement of any physical display. gamma/color "
            "NOT imported."
        )
        return None, None, None
    if backend not in _TRUSTED_BACKENDS:
        rejected.append(
            f"{record_id}: unrecognized measurement backend {backend!r} -- gamma/color NOT "
            "imported (refusing rather than guessing at its trustworthiness)."
        )
        return None, None, None

    result = measurement.get("result", {})
    gamma = _build_gamma(result, record_id=record_id, rejected=rejected)
    color = _build_color(result, record_id=record_id, rejected=rejected)

    if backend == "camera" and (gamma is not None or color is not None):
        warnings.append(
            f"{record_id}: measured via calsuite's 'camera' backend. calsuite's own accuracy "
            "estimate for this backend is ΔE00 2-5 (vs ≈ 1 for a colorimeter), and its docs "
            "say it 'is only trustworthy once cross-checked' against the argyll/spectro "
            "backends -- treat this as a coarser grade A than a colorimeter reading."
        )
        cross_checked = (result.get("backend_accuracy") or {}).get("cross_checked_against")
        if not cross_checked:
            warnings.append(
                f"{record_id}: backend_accuracy.cross_checked_against is not set -- this "
                "camera-backend measurement has not been cross-checked against a "
                "colorimeter/spectro reading."
            )

    created = None
    raw_created = measurement.get("created")
    if isinstance(raw_created, str):
        with contextlib.suppress(ValueError):
            created = _parse_calsuite_timestamp(raw_created)

    if gamma is not None or color is not None:
        device_id = measurement.get("device", {}).get("id", "?")
        notes.append(
            f"Gamma/color from calsuite {record_id} (device {device_id}, backend {backend!r}, "
            "provenance measured)."
        )
    return gamma, color, created


def _build_gamma(
    result: dict[str, Any], *, record_id: str, rejected: list[str]
) -> GammaCalibration | None:
    trc = result.get("trc") or {}
    effective_gamma = trc.get("effective_gamma") or {}
    black_contrast = result.get("black_contrast") or {}
    gamma_r = effective_gamma.get("r")
    gamma_g = effective_gamma.get("g")
    gamma_b = effective_gamma.get("b")
    lum_min = black_contrast.get("black_luminance_cdm2")
    lum_max = black_contrast.get("white_luminance_cdm2")
    if None in (gamma_r, gamma_g, gamma_b, lum_min, lum_max):
        rejected.append(
            f"{record_id}: missing result.trc.effective_gamma or result.black_contrast fields "
            "-- gamma NOT imported."
        )
        return None
    try:
        return GammaCalibration(
            method="photometer",
            gamma_r=gamma_r,
            gamma_g=gamma_g,
            gamma_b=gamma_b,
            lum_min_cdm2=lum_min,
            lum_max_cdm2=lum_max,
        )
    except ValueError as exc:
        rejected.append(
            f"{record_id}: gamma fields failed validation ({exc}) -- gamma NOT imported."
        )
        return None


def _build_color(
    result: dict[str, Any], *, record_id: str, rejected: list[str]
) -> ColorCalibration | None:
    chromaticity = (result.get("primaries") or {}).get("measured_chromaticity")
    xyz = result.get("primaries_measured")
    if not chromaticity or not xyz:
        rejected.append(
            f"{record_id}: missing result.primaries.measured_chromaticity or "
            "result.primaries_measured -- color NOT imported."
        )
        return None
    try:
        primaries = {
            channel: PrimaryChromaticity(
                x=chromaticity[key][0], y=chromaticity[key][1], Y_cdm2=xyz[key][1]
            )
            for channel, key in (
                ("red", "r"),
                ("green", "g"),
                ("blue", "b"),
                ("white", "w"),
            )
        }
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        rejected.append(f"{record_id}: malformed primaries data ({exc}) -- color NOT imported.")
        return None
    return ColorCalibration(method="measured", **primaries)


def import_calsuite_records(
    *,
    nominal: dict[str, Any] | None = None,
    measurement: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> CalsuiteImportResult:
    """Extract as much of a `Calibration` as possible from calsuite record(s).

    Args:
        nominal: A parsed `display.nominal` record (from `load_calsuite_json`),
            or `None` if not available.
        measurement: A parsed `display.measurement` record, or `None`.
        now: Reference time for staleness warnings, or `None` for the
            current UTC time.

    Returns:
        A `CalsuiteImportResult` with whatever could be extracted, plus
        `rejected`/`warnings`/`missing_required` explaining the rest.

    Raises:
        CalsuiteImportError: If neither `nominal` nor `measurement` is given.
    """
    if nominal is None and measurement is None:
        raise CalsuiteImportError(
            "At least one of a display.nominal or display.measurement record is required."
        )

    rejected: list[str] = []
    warnings: list[str] = []
    notes: list[str] = []

    width_cm = height_cm = None
    nominal_created: datetime | None = None
    if nominal is not None:
        width_cm, height_cm, nominal_created = _import_nominal(
            nominal, rejected=rejected, notes=notes
        )

    gamma = color = None
    measurement_created: datetime | None = None
    if measurement is not None:
        gamma, color, measurement_created = _import_measurement(
            measurement, rejected=rejected, warnings=warnings, notes=notes
        )

    # The measurement's own timestamp reflects when the display was actually
    # characterized, which matters more for staleness than an EDID read that
    # may predate it by months.
    source_created = measurement_created or nominal_created
    if source_created is not None:
        reference = now if now is not None else datetime.now(timezone.utc)
        age_days = (reference - source_created).total_seconds() / 86400.0
        if age_days > _STALE_AGE_DAYS:
            warnings.append(
                f"Imported calsuite data is {age_days:.0f} days old (created {source_created.isoformat()}), "
                f"past vpsych's own {_STALE_AGE_DAYS:.0f}-day staleness threshold -- consider "
                "re-running calsuite's display measurement before relying on this for a session."
            )

    if measurement is None and nominal is not None and width_cm is not None:
        warnings.append(
            "Only a display.nominal (EDID) record was given: EDID-claimed chromaticity/gamma "
            "are NOT usable as a color/gamma calibration here -- vpsych has no 'nominal, "
            "unmeasured' grade, and a panel's actual primaries commonly differ from its EDID "
            "claim. Only the physical screen size was imported; provide a display.measurement "
            "record too, or complete gamma/color in the wizard."
        )

    missing_required: list[str] = [
        "width_px (calsuite does not record pixel resolution)",
        "height_px (calsuite does not record pixel resolution)",
        "viewing_distance_cm (session-specific; calsuite has no equivalent)",
        "refresh_hz (calsuite does not measure this; vpsych re-measures it live anyway)",
        "environment checklist: room lighting / warm-up / night-light / HDR (self-reported; "
        "calsuite's osstate conditions record HDR/gamma-reset state but are not a substitute)",
    ]
    if width_cm is None:
        missing_required.append("width_cm (no usable display.nominal record)")
    if height_cm is None:
        missing_required.append("height_cm (no usable display.nominal record)")
    if gamma is None:
        missing_required.append("gamma calibration (no usable display.measurement record)")
    if color is None:
        missing_required.append("color calibration (no usable display.measurement record)")

    return CalsuiteImportResult(
        width_cm=width_cm,
        height_cm=height_cm,
        gamma=gamma,
        color=color,
        source_created_utc=source_created,
        notes=" ".join(notes) or None,
        rejected=tuple(rejected),
        warnings=tuple(warnings),
        missing_required=tuple(missing_required),
    )


__all__ = [
    "CalsuiteImportError",
    "CalsuiteImportResult",
    "import_calsuite_records",
    "load_calsuite_json",
]
