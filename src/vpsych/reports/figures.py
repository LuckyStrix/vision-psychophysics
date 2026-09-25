"""Figure-generation functions for session and longitudinal reports.

Produces matplotlib figures suitable for HTML embedding, designed to be
colorblind-safe, grayscale-legible, and readable at 800px width.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from vpsych.core.procedures.questplus_procedure import questplus_weibull_x_at_p
from vpsych.data.schemas import TestSummary

# Colorblind-safe palette (Okabe-Ito): suitable for most types of color blindness
# and legible in grayscale.
_PALETTE = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
}

# Map eye identifiers to descriptive labels and distinct palette colors
_EYE_COLORS = {
    "OD": (_PALETTE["blue"], "Right (OD)"),
    "OS": (_PALETTE["orange"], "Left (OS)"),
    "OU": (_PALETTE["black"], "Both (OU)"),
}


def _find_slope_param(fit_params: dict) -> float | None:
    """Find a psychometric-function slope in `fit_params`, tolerant of each real test's
    own key naming (e.g. `visual_acuity`'s `"slope"`, `letter_contrast_sensitivity`'s
    `"slope_log10_contrast"`, `critical_flicker_fusion`'s `"slope_neg_log10_hz"`) rather
    than requiring one exact key name.

    Args:
        fit_params: A `TestSummary.fit_params` dict.

    Returns:
        The first numeric value found under a key starting with `"slope"`
        (case-insensitive), or `None` if there is none.
    """
    for key, value in fit_params.items():
        if key.lower().startswith("slope") and isinstance(value, (int, float)):
            return float(value)
    return None


def _questplus_weibull_curve_params(
    summary: TestSummary,
) -> tuple[float, float, float, float] | None:
    """Recover (native_threshold, slope, guess_rate, lapse_rate) for the fitted curve.

    Every QUEST+-driven test in `tests_catalog` fits `questplus`'s own Weibull, whose
    threshold parameter sits at a different point than the criterion reported in
    `summary.estimate.value` (see `vpsych.core.procedures.questplus_procedure` and
    docs/METHODS.md, "Criterion conversion pitfall"). Reconstructing the curve therefore
    needs the *native* threshold, which those tests record in `estimate.extra` under a
    `"raw_questplus_native_threshold*"` key, plus the fitted slope/guess/lapse.

    Args:
        summary: The summary to read parameters from.

    Returns:
        The four parameters, or `None` when this summary does not record enough to draw
        the curve it actually fitted -- in which case no curve should be drawn at all.
    """
    extra = summary.estimate.extra or {}
    fit_params = summary.fit_params or {}

    native_threshold: float | None = None
    for key, value in extra.items():
        if key.startswith("raw_questplus_native_threshold") and isinstance(value, (int, float)):
            native_threshold = float(value)
            break

    slope = _find_slope_param(fit_params)
    if slope is None:
        slope = _find_slope_param(extra)

    def _lookup(*names: str) -> float | None:
        for source in (fit_params, extra):
            for name in names:
                value = source.get(name)
                if isinstance(value, (int, float)):
                    return float(value)
        return None

    guess = _lookup("guess_rate", "guess")
    lapse = _lookup("lapse_rate", "lapse")

    if native_threshold is None or slope is None or guess is None or lapse is None:
        return None
    return native_threshold, slope, guess, lapse


def psychometric_figure(summary: TestSummary, trials: pd.DataFrame) -> Figure:
    """Plot observed proportion-correct binned by intensity, fitted curve, and threshold.

    Binned points have size proportional to trial count. Practice and catch trials
    are excluded. If no fit params are available, marks the known range with a
    "lower bound" annotation instead.

    Args:
        summary: The TestSummary containing fit params and threshold estimate.
        trials: DataFrame with one row per trial (e.g., from read_trials_tsv).
            Must have columns: `intensity`, `correct`, `block`, `is_catch`.

    Returns:
        A matplotlib Figure with axes labeled with real units from summary.estimate.units.
    """
    matplotlib_use_agg()
    fig = Figure(figsize=(10, 6))
    ax = fig.subplots()

    # Filter to main block, exclude catch trials
    if len(trials) == 0 or "block" not in trials.columns or "is_catch" not in trials.columns:
        main_trials = pd.DataFrame()
    else:
        main_trials = trials[(trials["block"] == "main") & ~trials["is_catch"]].copy()

    if len(main_trials) == 0:
        ax.text(
            0.5,
            0.5,
            "No main-block trials available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_xlabel(f"Intensity ({summary.estimate.units})")
        ax.set_ylabel("Proportion Correct")
        ax.set_title(f"{summary.task_id}: Psychometric Function")
        return fig

    # Bin by intensity and compute proportion correct
    intensity_groups = main_trials.groupby("intensity")
    intensities_list: list[float] = []
    proportions_list: list[float] = []
    trial_counts_list: list[int] = []

    for intensity, group in intensity_groups:
        intensities_list.append(float(intensity))  # type: ignore[arg-type]
        n_trials = len(group)
        n_correct = int(group["correct"].sum())
        proportions_list.append(n_correct / n_trials)
        trial_counts_list.append(n_trials)

    intensities = np.array(intensities_list, dtype=float)
    proportions = np.array(proportions_list, dtype=float)
    trial_counts = np.array(trial_counts_list, dtype=int)

    # Sort by intensity
    sort_idx = np.argsort(intensities)
    intensities = intensities[sort_idx]
    proportions = proportions[sort_idx]
    trial_counts = trial_counts[sort_idx]

    # Plot binned points (size proportional to trial count)
    sizes = np.clip(trial_counts * 10, 20, 200)
    ax.scatter(
        intensities,
        proportions,
        s=sizes,
        alpha=0.6,
        color=_PALETTE["blue"],
        edgecolors="black",
        linewidth=0.5,
    )

    # Plot fitted curve if available
    if summary.fit_params:
        # For simple psychometric fits (threshold-slope-lapse style),
        # draw a smooth curve across the range
        x_range = np.linspace(intensities.min(), intensities.max(), 100)

        # Draw the curve the test ACTUALLY fitted, or none at all.
        #
        # Two earlier versions of this function got this wrong in ways only real data
        # exposed. The first recognized only the literal keys "threshold"/"slope"/"lapse"/
        # "guess", which no real test writes, so no real session ever drew a curve. The
        # second drew a *logistic* through `summary.estimate.value` using whatever
        # "slope*" key it found -- but every QUEST+-driven test here fits questplus's own
        # Weibull, and its slope is that family's shape exponent, not a logistic slope.
        # Feeding one family's parameter to the other draws a curve that matches neither
        # the model nor the data (the same criterion/parameterization confusion documented
        # in docs/METHODS.md, "Criterion conversion pitfall").
        #
        # `summary.estimate.value` is also the *converted* criterion (e.g. the 75%-correct
        # point), not the curve's own threshold parameter, so it cannot anchor the curve
        # either. The tests that record enough to reconstruct their fit store the native
        # threshold in `estimate.extra` under a "raw_questplus_native_threshold*" key.
        curve = _questplus_weibull_curve_params(summary)
        if curve is not None:
            native_threshold, slope, guess, lapse = curve
            y_fit = (
                1.0
                - lapse
                - (1.0 - guess - lapse) * np.exp(-(10.0 ** (slope * (x_range - native_threshold))))
            )
            ax.plot(
                x_range,
                y_fit,
                "-",
                color=_PALETTE["sky_blue"],
                linewidth=2,
                label="Fitted curve (Weibull)",
            )
        else:
            ax.text(
                0.02,
                0.02,
                "Fitted curve not shown:\nmodel parameters not recorded for this test",
                transform=ax.transAxes,
                fontsize=8,
                va="bottom",
                ha="left",
                bbox={"boxstyle": "round", "facecolor": "lightgray", "alpha": 0.6},
            )

        # Mark the threshold -- but only where it can be placed on THIS axis.
        #
        # `summary.estimate.value` is reported in the test's output units, which are often
        # not the trial intensity units this axis is drawn in: vernier_acuity reports
        # arcsec over a log10(arcsec) axis, critical_flicker_fusion reports Hz over a
        # -log10(Hz) axis. Drawing the raw value as a vertical line put the marker in a
        # completely wrong place for those tests, so place it by converting the criterion
        # into intensity units from the fitted curve, and otherwise fall back to the raw
        # value only when the trials confirm the units match.
        trial_units = (
            str(main_trials["intensity_units"].iloc[0])
            if "intensity_units" in main_trials.columns and len(main_trials)
            else None
        )
        target_p = summary.estimate.extra.get("target_p_correct")
        threshold_x: float | None = None
        ci_span: tuple[float, float] | None = None
        if curve is not None and isinstance(target_p, (int, float)):
            native_threshold, slope, guess, lapse = curve
            threshold_x = questplus_weibull_x_at_p(
                native_threshold, slope, guess, lapse, float(target_p)
            )
        elif trial_units is not None and trial_units == summary.estimate.units:
            threshold_x = summary.estimate.value
            ci_span = (summary.estimate.ci_low, summary.estimate.ci_high)

        if threshold_x is not None:
            ax.axvline(threshold_x, color=_PALETTE["orange"], linestyle="--", linewidth=2)
            if ci_span is not None:
                ax.fill_betweenx(
                    [0, 1], ci_span[0], ci_span[1], alpha=0.2, color=_PALETTE["orange"]
                )
            ax.text(
                threshold_x,
                0.95,
                f"Threshold: {summary.estimate.value:.3g} {summary.estimate.units}\n"
                f"[{summary.estimate.ci_low:.3g}, {summary.estimate.ci_high:.3g}]",
                ha="center",
                va="top",
                fontsize=9,
                bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
            )
    else:
        # No fit params: mark tested range as lower bound
        tested_min = intensities.min()
        tested_max = intensities.max()
        ax.fill_betweenx([0, 1], tested_min, tested_max, alpha=0.1, color="#888888")
        ax.text(
            (tested_min + tested_max) / 2,
            0.5,
            "Lower bound\n(no fit available)",
            ha="center",
            va="center",
            fontsize=10,
            bbox={"boxstyle": "round", "facecolor": "lightgray", "alpha": 0.7},
        )

    ax.set_xlabel(f"Intensity ({summary.estimate.units})")
    ax.set_ylabel("Proportion Correct")
    ax.set_title(f"{summary.task_id}: Psychometric Function")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    if "Fit" in [line.get_label() for line in ax.get_lines()]:
        ax.legend()

    return fig


def csf_figure(summary: TestSummary) -> Figure:
    """Plot log contrast sensitivity vs. spatial frequency with credible band.

    Uses the CSF parameters from summary.estimate.extra (set by the qCSF test's
    csf_curve method). Marks the tested frequency range, includes AULCSF in title.

    Args:
        summary: A TestSummary from the contrast_sensitivity_function test,
            with extra containing csf_curve output.

    Returns:
        A matplotlib Figure showing the CSF curve with credible band.
    """
    matplotlib_use_agg()
    fig = Figure(figsize=(10, 6))
    ax = fig.subplots()

    # Extract CSF curve data from extra
    extra = summary.estimate.extra
    # A real `contrast_sensitivity_function` summary nests the curve under "csf_curve"
    # (written by its `summarize`); flat keys are also accepted so a caller can pass a
    # bare `csf_curve()` result straight through.
    curve = extra.get("csf_curve") if extra else None
    if isinstance(curve, dict) and "spatial_frequency_cpd" in curve:
        extra = {**extra, **curve}
    if not extra or "spatial_frequency_cpd" not in extra:
        ax.text(
            0.5,
            0.5,
            "No CSF curve data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_xlabel("Spatial Frequency (cpd)")
        ax.set_ylabel("Log10 Contrast Sensitivity")
        ax.set_title(f"{summary.task_id}: Contrast Sensitivity Function")
        return fig

    freqs = np.array(extra["spatial_frequency_cpd"])
    mean_cs = np.array(extra["log10_cs_mean"])
    ci_low = np.array(extra["log10_cs_ci_low"])
    ci_high = np.array(extra["log10_cs_ci_high"])

    # Plot credible band
    ax.fill_between(freqs, ci_low, ci_high, alpha=0.3, color=_PALETTE["sky_blue"], label="95% CI")

    # Plot mean curve
    ax.plot(freqs, mean_cs, "-", color=_PALETTE["blue"], linewidth=2, label="Mean")

    # Shade the frequency range this display/distance could actually test.
    freq_range = extra.get("frequency_range_tested_cpd")
    if isinstance(freq_range, (list, tuple)) and len(freq_range) == 2:
        ax.axvspan(
            float(freq_range[0]),
            float(freq_range[1]),
            alpha=0.1,
            color=_PALETTE["orange"],
            label="Frequencies tested",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Spatial Frequency (cpd)")
    ax.set_ylabel("Log10 Contrast Sensitivity")

    # Include AULCSF in title if available
    title = f"{summary.task_id}: Contrast Sensitivity Function"
    # A real summary reports AULCSF as the estimate itself; a bare curve dict may carry it
    # in "aulcsf" instead.
    aulcsf = extra.get("aulcsf")
    if aulcsf is None and "aulcsf" in summary.estimate.units:
        aulcsf = summary.estimate.value
    if aulcsf is not None:
        title += f" (AULCSF={float(aulcsf):.2f})"

    ax.set_title(title)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()

    return fig


def history_figure(history: list[dict], task_id: str, units: str) -> Figure:
    """Plot threshold over time with per-session CI error bars, one series per eye.

    Chronological x-axis (oldest to newest). Each point represents one session/run.

    Args:
        history: List of dicts from catalog.task_history(), one per test run,
            ordered oldest-first. Each dict has keys: `session_id`, `run`,
            `value`, `ci_low`, `ci_high`, `ci_level`, `units`, `started_utc`, etc.
        task_id: The test's task ID for the plot title.
        units: Units string for the y-axis label.

    Returns:
        A matplotlib Figure showing threshold history over time.
    """
    matplotlib_use_agg()
    fig = Figure(figsize=(12, 6))
    ax = fig.subplots()

    if not history:
        ax.text(
            0.5,
            0.5,
            "No history available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_xlabel("Date")
        ax.set_ylabel(f"Threshold ({units})")
        ax.set_title(f"{task_id}: Threshold History")
        return fig

    # Convert to DataFrame for easier manipulation
    history_df = pd.DataFrame(history)
    if "started_utc" in history_df.columns:
        history_df["started_utc"] = pd.to_datetime(history_df["started_utc"])

    # Group by eye
    eyes_in_data: list[str] = (
        list(history_df["eye"].unique()) if "eye" in history_df.columns else ["OU"]
    )

    for eye in eyes_in_data:
        if "eye" in history_df.columns:
            eye_data = history_df[history_df["eye"] == eye].sort_values(
                "started_utc", na_position="last"
            )
        else:
            eye_data = history_df.sort_values("started_utc", na_position="last")

        if len(eye_data) == 0:
            continue

        x = np.arange(len(eye_data))
        y = np.array(eye_data["value"].values, dtype=float)
        y_err_low = np.clip(y - np.array(eye_data["ci_low"].values, dtype=float), 0, None)
        y_err_high = np.clip(np.array(eye_data["ci_high"].values, dtype=float) - y, 0, None)

        color, eye_label = _EYE_COLORS.get(eye, (_PALETTE["black"], eye))

        ax.errorbar(
            x,
            y,
            yerr=np.vstack([y_err_low, y_err_high]),
            fmt="o-",
            capsize=5,
            capthick=1,
            label=eye_label,
            color=color,
            markersize=6,
        )

    ax.set_xlabel("Session (chronological)")
    ax.set_ylabel(f"Threshold ({units})")
    ax.set_title(f"{task_id}: Threshold History")
    ax.grid(True, alpha=0.3)
    if len(eyes_in_data) > 1:
        ax.legend()

    return fig


def quality_badges(summary: TestSummary) -> list[tuple[str, str, str]]:
    """Extract quality flags as (code, severity, message) tuples.

    Args:
        summary: The TestSummary to extract quality flags from.

    Returns:
        A list of (code, severity, message) tuples, one per quality flag.
    """
    return [(f.code, f.severity, f.message) for f in summary.quality_flags]


def matplotlib_use_agg() -> None:
    """Ensure matplotlib uses the Agg backend (headless, no X11)."""
    import matplotlib

    matplotlib.use("Agg")
