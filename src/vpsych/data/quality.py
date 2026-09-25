"""Generic quality-flag computation, shared by every test's `summarize`.

Pure functions: given plain numbers (never a live display/calibration
object), each `check_*` function returns a list of
`vpsych.data.schemas.QualityFlag` -- empty if there is nothing to flag.
Every test's `summarize(trials)` is expected to call the relevant subset of
these (or `compute_quality_flags`, which calls all of them) and fold the
results into its `TestSummary.quality_flags`.

Thresholds match `docs/DATA_FORMAT.md` / the project plan's "Quality flags"
section:

- catch-trial lapse rate > 10% is a warning, > 20% is critical.
- dropped-frame fraction > 1% is a warning.
- a threshold estimate within `edge_tolerance_frac` of either end of the
  tested stimulus range is a warning (the adaptive procedure may have been
  unable to bracket the true threshold).
- calibration grade worse than `"A"`, or a stale calibration, is a warning.
- goodness-of-fit p-value below `alpha` (default 0.05) is a warning.
- fewer than `min_trials` main-block trials is a warning.
"""

from __future__ import annotations

from vpsych.data.schemas import QualityFlag


def check_catch_lapse_rate(catch_lapse_rate: float, n_catch: int) -> list[QualityFlag]:
    """Flag a high catch-trial lapse rate.

    Args:
        catch_lapse_rate: Proportion of catch trials answered incorrectly,
            in [0, 1].
        n_catch: Number of catch trials the rate is based on; if 0, no flag
            is raised (the rate is undefined/uninformative).

    Returns:
        A list with one `QualityFlag` if the rate exceeds 10% (warning) or
        20% (critical), otherwise an empty list.
    """
    if n_catch == 0:
        return []
    if catch_lapse_rate > 0.20:
        return [
            QualityFlag(
                code="high_catch_lapse_rate",
                severity="critical",
                message=(
                    f"{catch_lapse_rate:.0%} of {n_catch} catch trials were missed (>20%). "
                    "This threshold estimate is unreliable -- the observer may have been "
                    "inattentive or the response mapping misunderstood."
                ),
            )
        ]
    if catch_lapse_rate > 0.10:
        return [
            QualityFlag(
                code="high_catch_lapse_rate",
                severity="warning",
                message=(
                    f"{catch_lapse_rate:.0%} of {n_catch} catch trials were missed (>10%). "
                    "Interpret this threshold with some caution."
                ),
            )
        ]
    return []


def check_dropped_frames(dropped_fraction: float) -> list[QualityFlag]:
    """Flag an excessive dropped-frame fraction during stimulus presentation.

    Args:
        dropped_fraction: Fraction of frames classified as dropped, in
            [0, 1] (see `vpsych.core.timing.FrameTimingStats.dropped_fraction`).

    Returns:
        A list with one warning `QualityFlag` if `dropped_fraction` exceeds
        1%, otherwise an empty list.
    """
    if dropped_fraction > 0.01:
        return [
            QualityFlag(
                code="excess_dropped_frames",
                severity="warning",
                message=(
                    f"{dropped_fraction:.1%} of frames were dropped during this run (>1%). "
                    "Stimulus timing may have been compromised; check for background load on "
                    "the test machine."
                ),
            )
        ]
    return []


def check_timeouts(
    n_timeouts: int, n_answered: int, max_fraction: float = 0.1
) -> list[QualityFlag]:
    """Flag a run where many main trials timed out with no response.

    Args:
        n_timeouts: Main, non-catch trials with no response (excluded from the estimate).
        n_answered: Main, non-catch trials that were answered.
        max_fraction: Timeout fraction above which to warn.

    Returns:
        One warning flag if timeouts exceed `max_fraction` of attempted trials, else empty.
    """
    total = n_timeouts + n_answered
    if n_timeouts == 0 or n_timeouts / total <= max_fraction:
        return []
    return [
        QualityFlag(
            code="many_timeouts",
            severity="warning",
            message=(
                f"{n_timeouts} of {total} trials timed out with no response and were excluded "
                "from the estimate. The observer may have been inattentive or the response "
                "window too short."
            ),
        )
    ]


def check_threshold_at_range_edge(
    threshold: float,
    range_min: float,
    range_max: float,
    edge_tolerance_frac: float = 0.05,
) -> list[QualityFlag]:
    """Flag a threshold estimate pinned at or near the edge of the tested stimulus range.

    Args:
        threshold: The estimated threshold value.
        range_min: Lowest intensity the procedure was allowed to present.
        range_max: Highest intensity the procedure was allowed to present.
        edge_tolerance_frac: Fraction of the range width, from either edge,
            counted as "at the edge". Defaults to 0.05 (5%).

    Returns:
        A list with one warning `QualityFlag` if `threshold` falls within
        `edge_tolerance_frac` of `range_min` or `range_max`, otherwise an
        empty list. Returns an empty list if `range_max <= range_min`
        (degenerate/misconfigured range).
    """
    width = range_max - range_min
    if width <= 0:
        return []
    tolerance = edge_tolerance_frac * width
    if threshold <= range_min + tolerance or threshold >= range_max - tolerance:
        return [
            QualityFlag(
                code="threshold_at_range_edge",
                severity="warning",
                message=(
                    f"The estimated threshold ({threshold:g}) is at or near the edge of the "
                    f"tested stimulus range [{range_min:g}, {range_max:g}]. The true threshold "
                    "may lie outside the range this test could present."
                ),
            )
        ]
    return []


def check_calibration_quality(
    luminance_grade: str,
    calibration_age_days: float,
    max_age_days: int = 30,
    min_acceptable_grade: str = "A",
) -> list[QualityFlag]:
    """Flag a sub-optimal or stale calibration.

    Args:
        luminance_grade: The active calibration's luminance grade (`"A"`,
            `"B"`, or `"C"`; see `vpsych.core.calibration.models.grade_luminance`).
        calibration_age_days: Age of the active calibration, in days.
        max_age_days: Maximum age before a calibration is considered stale.
            Defaults to 30, matching
            `vpsych.core.calibration.models.Calibration.is_stale`.
        min_acceptable_grade: Best grade considered "not worth flagging";
            grades worse than this (later in `"A" < "B" < "C"`) raise a
            flag. Defaults to `"A"`.

    Returns:
        Zero, one, or two `QualityFlag`s: `"low_calibration_grade"` if
        `luminance_grade` is worse than `min_acceptable_grade`, and/or
        `"stale_calibration"` if `calibration_age_days > max_age_days`.
    """
    rank = {"A": 0, "B": 1, "C": 2}
    flags: list[QualityFlag] = []
    if rank[luminance_grade] > rank[min_acceptable_grade]:
        flags.append(
            QualityFlag(
                code="low_calibration_grade",
                severity="warning",
                message=(
                    f"This session's display calibration is grade {luminance_grade} "
                    f"(better than {min_acceptable_grade} recommended for research-grade data)."
                ),
            )
        )
    if calibration_age_days > max_age_days:
        flags.append(
            QualityFlag(
                code="stale_calibration",
                severity="warning",
                message=(
                    f"This session's calibration was {calibration_age_days:.0f} days old "
                    f"(more than {max_age_days} days). Consider recalibrating."
                ),
            )
        )
    return flags


def check_goodness_of_fit(p_value: float | None, alpha: float = 0.05) -> list[QualityFlag]:
    """Flag a poor psychometric-function/model fit.

    Args:
        p_value: The fit's goodness-of-fit p-value (see
            `vpsych.core.psychometric.deviance_gof`), or `None` if no GOF
            test was performed (no flag is raised in that case).
        alpha: Significance level below which the fit is flagged as poor.
            Defaults to 0.05.

    Returns:
        A list with one warning `QualityFlag` if `p_value < alpha`,
        otherwise an empty list.
    """
    if p_value is None:
        return []
    if p_value < alpha:
        return [
            QualityFlag(
                code="poor_gof",
                severity="warning",
                message=(
                    f"The fitted psychometric function fit these data poorly "
                    f"(goodness-of-fit p = {p_value:.3g} < {alpha:g}). The threshold estimate "
                    "may not be well described by the assumed model."
                ),
            )
        ]
    return []


def check_trial_count(n_trials: int, min_trials: int) -> list[QualityFlag]:
    """Flag too few main-block trials to trust the resulting estimate.

    Args:
        n_trials: Number of main-block trials the estimate is based on.
        min_trials: Minimum number of trials expected for this test/procedure.

    Returns:
        A list with one warning `QualityFlag` if `n_trials < min_trials`,
        otherwise an empty list.
    """
    if n_trials < min_trials:
        return [
            QualityFlag(
                code="too_few_trials",
                severity="warning",
                message=(
                    f"Only {n_trials} main-block trials were completed (expected at least "
                    f"{min_trials}). The threshold estimate may be imprecise."
                ),
            )
        ]
    return []


def compute_quality_flags(
    *,
    catch_lapse_rate: float,
    n_catch: int,
    dropped_fraction: float,
    threshold: float | None = None,
    range_min: float | None = None,
    range_max: float | None = None,
    luminance_grade: str | None = None,
    calibration_age_days: float | None = None,
    gof_p_value: float | None = None,
    n_trials: int | None = None,
    min_trials: int | None = None,
    max_calibration_age_days: int = 30,
    edge_tolerance_frac: float = 0.05,
    gof_alpha: float = 0.05,
    n_timeouts: int = 0,
) -> list[QualityFlag]:
    """Run every applicable quality check and return the combined flag list.

    Every check is optional: pass only the quantities a given test has
    available (e.g. a test with no fixed stimulus range can omit
    `threshold`/`range_min`/`range_max`) and the corresponding check is
    skipped. `catch_lapse_rate`/`n_catch`/`dropped_fraction` are always
    required since every test has them.

    Args:
        catch_lapse_rate: See `check_catch_lapse_rate`.
        n_catch: See `check_catch_lapse_rate`.
        dropped_fraction: See `check_dropped_frames`.
        threshold: See `check_threshold_at_range_edge` (all three of
            `threshold`/`range_min`/`range_max` must be given to run it).
        range_min: See `check_threshold_at_range_edge`.
        range_max: See `check_threshold_at_range_edge`.
        luminance_grade: See `check_calibration_quality` (both it and
            `calibration_age_days` must be given to run it).
        calibration_age_days: See `check_calibration_quality`.
        gof_p_value: See `check_goodness_of_fit`.
        n_trials: See `check_trial_count` (both it and `min_trials` must be
            given to run it).
        min_trials: See `check_trial_count`.
        max_calibration_age_days: Passed through to `check_calibration_quality`.
        edge_tolerance_frac: Passed through to `check_threshold_at_range_edge`.
        gof_alpha: Passed through to `check_goodness_of_fit`.
        n_timeouts: Non-catch main trials with no response, excluded from the estimate
            (see `check_timeouts`; `n_trials` is taken as the answered count).

    Returns:
        The concatenation of every applicable check's flags, in the order
        listed above.
    """
    flags: list[QualityFlag] = []
    flags += check_catch_lapse_rate(catch_lapse_rate, n_catch)
    flags += check_dropped_frames(dropped_fraction)
    if threshold is not None and range_min is not None and range_max is not None:
        flags += check_threshold_at_range_edge(threshold, range_min, range_max, edge_tolerance_frac)
    if luminance_grade is not None and calibration_age_days is not None:
        flags += check_calibration_quality(
            luminance_grade, calibration_age_days, max_calibration_age_days
        )
    flags += check_goodness_of_fit(gof_p_value, gof_alpha)
    if n_trials is not None and min_trials is not None:
        flags += check_trial_count(n_trials, min_trials)
    if n_timeouts:
        flags += check_timeouts(n_timeouts, n_trials or 0)
    return flags
