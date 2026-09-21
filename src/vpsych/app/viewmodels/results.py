"""Results-screen display logic: threshold formatting and quality-flag styling.

Severity is always conveyed with both a text marker and a word (never color
alone, per the design rules): `severity_marker`/`severity_label` give the
non-color cues a widget pairs with its (secondary) color styling.
"""

from __future__ import annotations

from dataclasses import dataclass

from vpsych.data.schemas import QualityFlag, Severity, TestSummary

_SEVERITY_MARKERS: dict[Severity, str] = {
    "info": "i",
    "warning": "⚠",  # warning sign
    "critical": "✕",  # heavy X
}

_SEVERITY_LABELS: dict[Severity, str] = {
    "info": "Info",
    "warning": "Warning",
    "critical": "Critical",
}

_SEVERITY_ORDER: dict[Severity, int] = {"critical": 0, "warning": 1, "info": 2}


def severity_marker(severity: Severity) -> str:
    """Non-color glyph for a severity, e.g. a warning triangle. Always paired with text."""
    return _SEVERITY_MARKERS[severity]


def severity_label(severity: Severity) -> str:
    """Human-readable word for a severity, e.g. `"Warning"`."""
    return _SEVERITY_LABELS[severity]


def format_threshold(
    estimate_value: float, ci_low: float, ci_high: float, ci_level: float, units: str
) -> str:
    """Format a threshold estimate with its confidence interval and units.

    Args:
        estimate_value: Point estimate.
        ci_low: Lower CI bound.
        ci_high: Upper CI bound.
        ci_level: Nominal CI coverage, e.g. 0.95.
        units: Units string, e.g. `"logMAR"`.

    Returns:
        e.g. `"-0.12 logMAR (95% CI [-0.20, -0.05])"`.
    """
    pct = round(ci_level * 100)
    return f"{estimate_value:.3g} {units} (95% CI [{ci_low:.3g}, {ci_high:.3g}])".replace(
        "95%", f"{pct}%"
    )


@dataclass(frozen=True)
class QualityFlagViewModel:
    """Display-ready form of one `QualityFlag`.

    Attributes:
        code: Machine-readable code.
        severity: Severity level.
        marker: Non-color glyph for the severity.
        label: Human-readable severity word.
        message: The flag's explanation.
    """

    code: str
    severity: Severity
    marker: str
    label: str
    message: str


def build_quality_flag_view_models(flags: list[QualityFlag]) -> list[QualityFlagViewModel]:
    """Build display view models for a summary's quality flags, most severe first.

    Args:
        flags: Quality flags, e.g. `TestSummary.quality_flags`.

    Returns:
        View models sorted critical, then warning, then info (stable within
        each severity, preserving input order).
    """
    indexed = list(enumerate(flags))
    indexed.sort(key=lambda pair: (_SEVERITY_ORDER[pair[1].severity], pair[0]))
    return [
        QualityFlagViewModel(
            code=f.code,
            severity=f.severity,
            marker=severity_marker(f.severity),
            label=severity_label(f.severity),
            message=f.message,
        )
        for _, f in indexed
    ]


@dataclass(frozen=True)
class ResultViewModel:
    """Everything the Results screen needs to render one test summary.

    Attributes:
        task_id: The test's `TestSpec.id`.
        eye: Eye tested.
        run: 1-based run number.
        threshold_text: Formatted threshold + CI (see `format_threshold`).
        n_trials: Main-block trial count.
        n_catch: Catch-trial count.
        catch_lapse_rate_text: Formatted catch lapse rate percentage.
        analysis_version: Analysis version that produced this summary.
        flags: Quality flag view models, most severe first.
    """

    task_id: str
    eye: str
    run: int
    threshold_text: str
    n_trials: int
    n_catch: int
    catch_lapse_rate_text: str
    analysis_version: str
    flags: list[QualityFlagViewModel]


def build_result_view_model(summary: TestSummary) -> ResultViewModel:
    """Build the Results screen's view model for one `TestSummary`.

    Args:
        summary: The test summary to render.

    Returns:
        The constructed `ResultViewModel`.
    """
    e = summary.estimate
    return ResultViewModel(
        task_id=summary.task_id,
        eye=summary.eye,
        run=summary.run,
        threshold_text=format_threshold(e.value, e.ci_low, e.ci_high, e.ci_level, e.units),
        n_trials=summary.n_trials,
        n_catch=summary.n_catch,
        catch_lapse_rate_text=f"{summary.catch_lapse_rate * 100:.1f}%",
        analysis_version=summary.analysis_version,
        flags=build_quality_flag_view_models(summary.quality_flags),
    )
