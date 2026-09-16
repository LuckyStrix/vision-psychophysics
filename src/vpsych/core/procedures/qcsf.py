"""Quick Contrast Sensitivity Function (qCSF) procedure (Lesmes et al. 2010).

Implements :class:`vpsych.core.procedures.base.MultiParamProcedure`. qCSF
jointly estimates the four parameters of a truncated log-parabola model of
the contrast sensitivity function (peak gain, peak spatial frequency,
bandwidth, and low-frequency truncation) using a QUEST+-style Bayesian
adaptive procedure over a 2D stimulus space (spatial frequency x contrast),
selecting each trial's stimulus to maximize expected information gain about
the joint parameter posterior (Lesmes, L. A., Lu, Z.-L., Baek, J., &
Albright, T. D. (2010). Bayesian adaptive estimation of the contrast
sensitivity function: The quick CSF method. Journal of Vision, 10(3):17).

CSF model
---------
Truncated log-parabola, parameterized directly in log10-sensitivity /
log2-frequency units (matching the constructor's documented grid units):

    log10 CS_unclipped(f) = peak_gain - 4*log10(2) * ((log2(f) - log2(peak_freq)) / bandwidth) ** 2
    log10 CS(f) = max(log10 CS_unclipped(f), peak_gain - low_freq_truncation)   for f <= peak_freq
                = log10 CS_unclipped(f)                                        for f >  peak_freq

`bandwidth` is the full width at half maximum in octaves (the unclipped
parabola equals `peak_gain - log10(2)` -- i.e. half the peak sensitivity --
at `|log2(f) - log2(peak_freq)| == bandwidth / 2`). `low_freq_truncation`
(log10 units, >= 0) is how far below the peak the low-frequency "shoulder"
is allowed to plateau instead of continuing to fall off with the parabola;
`low_freq_truncation == 0` degenerates to an untruncated (fully symmetric)
log-parabola. See :func:`log_contrast_sensitivity`.

Psychometric function and implementation approach
---------------------------------------------------
At a given spatial frequency, the probability of a correct response as a
function of presented contrast is modeled as a Weibull-family function (in
the sense of `vpsych.core.psychometric`'s `F(0) == 0.5` convention) of
log10 contrast around the CSF-predicted threshold contrast
(`1 / CS(f)`), i.e.

    p(correct | f, contrast) = guess + (1 - guess - lapse) *
        F( (log10(contrast) - log10(threshold_contrast(f))) / psychometric_slope )

`psychometric_slope` and `lapse_rate` are fixed (not estimated) during the
adaptive run, following common qCSF practice of fixing the psychometric
function's shape and estimating only the CSF's own 4 parameters;
`psychometric_slope` defaults to 0.35 (log10-contrast units) and
`lapse_rate` defaults to 0.02 -- both are implementation choices (Lesmes et
al. 2010 do not use this exact F(0)=0.5 parameterization, so their
reported slope value does not translate directly) and are documented here
and in `docs/METHODS.md` rather than derived from the source paper.

Unlike `QuestPlusProcedure`, this does not use the `questplus` package: its
built-in `func="csf"` option is a different (linear-in-frequency) threshold
model, not the truncated log-parabola this project's plan specifies. Instead
this module precomputes the full stimulus x parameter-grid likelihood table
`p_correct[stim, param]` once at construction (a "precomputed-likelihood
approach", explicitly permitted by the Phase 1a task for speed) and reuses
it, fully vectorized with NumPy, for one-step-ahead expected-entropy
stimulus selection on every trial -- no per-trial re-evaluation of the CSF
formula is needed. See `QCSF.default_grids()` for the grid sizes used to
keep `next_stimulus()` comfortably under the 100 ms/trial budget on a
laptop (measured well under 10 ms for those grids).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vpsych.core.procedures.base import MultiParamProcedure, ThresholdEstimate

#: Default fixed psychometric-function slope (log10 contrast units). See
#: module docstring.
DEFAULT_PSYCHOMETRIC_SLOPE = 0.35

#: Standard spatial frequencies (cpd) at which `estimate()` reports log CS.
STANDARD_FREQUENCIES_CPD = (1.0, 1.5, 3.0, 6.0, 12.0, 18.0)


def log_contrast_sensitivity(
    spatial_frequency_cpd: np.ndarray | float,
    peak_gain_log10: np.ndarray | float,
    peak_freq_cpd: np.ndarray | float,
    bandwidth_octaves: np.ndarray | float,
    low_freq_truncation_log10: np.ndarray | float,
) -> np.ndarray:
    """Truncated log-parabola CSF model (Lesmes, Lu, Baek & Albright 2010).

    Pure function of the model's 4 parameters and spatial frequency; shared
    by `QCSF` (for stimulus-likelihood precomputation and posterior
    summaries) and `vpsych.core.observers.CSFObserver` (so the simulated
    observer used to validate `QCSF` evaluates literally the same formula).
    All arguments broadcast against each other via NumPy, so passing e.g.
    `spatial_frequency_cpd` with shape `(n_freq, 1)` and the parameter
    arguments with shape `(1, n_param)` yields a `(n_freq, n_param)` result.

    Returns:
        log10 contrast sensitivity, `log10(1 / threshold_contrast)`.
    """
    f = np.asarray(spatial_frequency_cpd, dtype=float)
    gain = np.asarray(peak_gain_log10, dtype=float)
    fmax = np.asarray(peak_freq_cpd, dtype=float)
    bw = np.asarray(bandwidth_octaves, dtype=float)
    trunc = np.asarray(low_freq_truncation_log10, dtype=float)

    x = np.log2(f) - np.log2(fmax)
    unclipped = gain - 4.0 * np.log10(2.0) * (x / bw) ** 2
    floor = gain - trunc
    low_side = f <= fmax
    return np.where(low_side, np.maximum(unclipped, floor), unclipped)


def _weighted_mean_sd(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    mean = float(np.sum(weights * values))
    var = float(np.sum(weights * (values - mean) ** 2))
    return mean, float(np.sqrt(max(var, 0.0)))


def _weighted_percentile(
    values: np.ndarray, weights: np.ndarray, quantiles: list[float]
) -> list[float]:
    order = np.argsort(values)
    v_sorted = values[order]
    w_sorted = weights[order]
    cum = np.cumsum(w_sorted)
    cum = cum / cum[-1]
    return [float(np.interp(q, cum, v_sorted)) for q in quantiles]


class QCSF(MultiParamProcedure):
    """qCSF adaptive procedure over the truncated log-parabola CSF model.

    Stimuli are specified as dicts with keys `"spatial_frequency_cpd"`
    (cycles per degree of visual angle), `"contrast"` (Michelson contrast,
    unitless in [0, 1]), and `"intensity"` (log10 of `contrast`) -- the
    latter is the scalar the trial loop logs as the trial's intensity
    (`TrialRecord.intensity`, `intensity_units="log10_contrast"`) even
    though qCSF's own stimulus space is 2D; `update()` ignores it and only
    uses `spatial_frequency_cpd`/`contrast` to locate the presented
    stimulus on the grid.

    Args:
        spatial_frequency_values_cpd: Candidate spatial-frequency levels the
            procedure may select from, in cycles per degree.
        contrast_values: Candidate contrast levels the procedure may select
            from, unitless Michelson contrast in (0, 1].
        peak_gain_values: Candidate grid of peak sensitivity (gain)
            parameter values (log10 sensitivity units).
        peak_freq_values_cpd: Candidate grid of peak spatial-frequency
            parameter values, in cycles per degree.
        bandwidth_values_octaves: Candidate grid of bandwidth parameter
            values (full width at half maximum), in octaves.
        low_freq_truncation_values: Candidate grid of low-frequency
            truncation (log10 sensitivity) parameter values.
        guess_rate: Fixed guess (chance/floor) rate of the psychometric
            function relating contrast to detection probability at a given
            spatial frequency, e.g. `0.5` for 2AFC, `0.25` for 4AFC.
        lapse_rate: Fixed lapse rate assumed during fitting/selection.
        max_trials: Maximum number of trials before `finished` becomes
            `True` regardless of posterior precision, or `None` for no
            fixed cap.
        ci_level: Nominal coverage for summary-estimate confidence
            intervals, e.g. `0.95`.
        psychometric_slope: Fixed slope of the underlying (log10-contrast)
            Weibull-family psychometric function. Not part of the Phase 0
            freeze; added (with the documented default
            `DEFAULT_PSYCHOMETRIC_SLOPE`) because the frozen constructor has
            no field for it. See the module docstring.

    `next_stimulus()` picks the (spatial frequency, contrast) pair
    minimizing one-step-ahead expected posterior entropy over the joint
    4-parameter grid (the same principle QUEST+ uses, applied here to a
    precomputed likelihood table rather than via the `questplus` package;
    see module docstring).
    """

    def __init__(
        self,
        spatial_frequency_values_cpd: list[float],
        contrast_values: list[float],
        peak_gain_values: list[float],
        peak_freq_values_cpd: list[float],
        bandwidth_values_octaves: list[float],
        low_freq_truncation_values: list[float],
        guess_rate: float,
        lapse_rate: float = 0.02,
        max_trials: int | None = None,
        ci_level: float = 0.95,
        psychometric_slope: float = DEFAULT_PSYCHOMETRIC_SLOPE,
    ) -> None:
        for name, values in (
            ("spatial_frequency_values_cpd", spatial_frequency_values_cpd),
            ("contrast_values", contrast_values),
            ("peak_gain_values", peak_gain_values),
            ("peak_freq_values_cpd", peak_freq_values_cpd),
            ("bandwidth_values_octaves", bandwidth_values_octaves),
            ("low_freq_truncation_values", low_freq_truncation_values),
        ):
            if not values:
                raise ValueError(f"{name} must be non-empty")

        self.spatial_frequency_values_cpd = list(spatial_frequency_values_cpd)
        self.contrast_values = list(contrast_values)
        self.peak_gain_values = list(peak_gain_values)
        self.peak_freq_values_cpd = list(peak_freq_values_cpd)
        self.bandwidth_values_octaves = list(bandwidth_values_octaves)
        self.low_freq_truncation_values = list(low_freq_truncation_values)
        self.guess_rate = guess_rate
        self.lapse_rate = lapse_rate
        self.max_trials = max_trials
        self.ci_level = ci_level
        self.psychometric_slope = psychometric_slope
        self._finished: bool = False
        self._n_trials = 0

        freqs = np.asarray(self.spatial_frequency_values_cpd, dtype=float)
        contrasts = np.asarray(self.contrast_values, dtype=float)
        freq_grid, contrast_grid = np.meshgrid(freqs, contrasts, indexing="ij")
        self._stim_freq = freq_grid.ravel()
        self._stim_contrast = contrast_grid.ravel()
        self._n_stim = self._stim_freq.size

        gain = np.asarray(self.peak_gain_values, dtype=float)
        pfreq = np.asarray(self.peak_freq_values_cpd, dtype=float)
        bw = np.asarray(self.bandwidth_values_octaves, dtype=float)
        trunc = np.asarray(self.low_freq_truncation_values, dtype=float)
        gain_grid, pfreq_grid, bw_grid, trunc_grid = np.meshgrid(
            gain, pfreq, bw, trunc, indexing="ij"
        )
        self._param_gain = gain_grid.ravel()
        self._param_freq = pfreq_grid.ravel()
        self._param_bw = bw_grid.ravel()
        self._param_trunc = trunc_grid.ravel()
        self._n_param = self._param_gain.size

        log_cs = self._log_cs_at(
            self._stim_freq, self._param_gain, self._param_freq, self._param_bw, self._param_trunc
        )  # (n_stim, n_param)
        log10_threshold_contrast = -log_cs
        log10_contrast = np.log10(self._stim_contrast)[:, None]
        z = (log10_contrast - log10_threshold_contrast) / self.psychometric_slope
        z = np.clip(z, -50.0, 50.0)
        f = 1.0 - np.exp2(-np.exp2(z))
        p_correct = self.guess_rate + (1.0 - self.guess_rate - self.lapse_rate) * f
        self._p_correct = np.clip(p_correct, 1e-6, 1.0 - 1e-6)  # (n_stim, n_param)
        # Precomputed once (fixed for the procedure's lifetime, since only the
        # posterior -- not the likelihood table -- changes trial to trial):
        # x*log(x) for p and its complement, used by the closed-form
        # expected-entropy reformulation in `next_stimulus` to avoid any
        # full (n_stim, n_param) `log()` call on the hot per-trial path (see
        # that method's docstring for the derivation).
        self._q_correct = 1.0 - self._p_correct
        self._p_log_p = self._p_correct * np.log(self._p_correct)
        self._q_log_q = self._q_correct * np.log(self._q_correct)

        self._posterior = np.full(self._n_param, 1.0 / self._n_param)

    @staticmethod
    def default_grids() -> dict[str, list[float]]:
        """Recommended default grids, chosen to keep `next_stimulus()` well under 100 ms.

        Grid sizes: 10 spatial frequencies x 10 contrasts = 100 stimuli;
        10 x 10 x 8 x 6 = 4800 parameter combinations. The precomputed
        likelihood table is therefore 100 x 4800 floats (~3.8 MB), and
        `next_stimulus()`'s matrix operations over it measure a few
        milliseconds on a laptop (see
        `tests/procedures/test_qcsf.py::test_next_stimulus_is_fast`).

        Returns:
            A dict of the 6 grid keyword arguments `QCSF.__init__` expects,
            ready to be splatted in: `QCSF(**QCSF.default_grids(), guess_rate=0.5)`.
        """
        return {
            "spatial_frequency_values_cpd": [float(v) for v in np.geomspace(0.5, 24.0, 10)],
            "contrast_values": [float(v) for v in np.geomspace(0.005, 0.9, 10)],
            "peak_gain_values": [float(v) for v in np.linspace(0.3, 2.7, 10)],
            "peak_freq_values_cpd": [float(v) for v in np.geomspace(0.3, 12.0, 10)],
            "bandwidth_values_octaves": [float(v) for v in np.linspace(1.0, 6.0, 8)],
            "low_freq_truncation_values": [float(v) for v in np.linspace(0.0, 2.0, 6)],
        }

    @staticmethod
    def _log_cs_at(
        freqs: np.ndarray, gain: np.ndarray, pfreq: np.ndarray, bw: np.ndarray, trunc: np.ndarray
    ) -> np.ndarray:
        """log10 CS at each of `freqs` (n_freq,) for each of `gain`/etc. (n_param,) -> (n_freq, n_param)."""
        return log_contrast_sensitivity(
            spatial_frequency_cpd=freqs[:, None],
            peak_gain_log10=gain[None, :],
            peak_freq_cpd=pfreq[None, :],
            bandwidth_octaves=bw[None, :],
            low_freq_truncation_log10=trunc[None, :],
        )

    @property
    def finished(self) -> bool:
        return self._finished

    def _nearest_stim_index(self, spatial_frequency_cpd: float, contrast: float) -> int:
        d = (self._stim_freq - spatial_frequency_cpd) ** 2 + (self._stim_contrast - contrast) ** 2
        return int(np.argmin(d))

    def next_stimulus(self) -> dict[str, float]:
        """Pick the (frequency, contrast) minimizing one-step-ahead expected posterior entropy.

        For a candidate stimulus with per-param correct-probability row `p`
        (and `q = 1-p`), writing `pk = p . post` for the marginal P(correct)
        under the current posterior `post`, the post-outcome entropies work
        out (by expanding `post_outcome = outcome_likelihood * post / pk` in
        `H = -sum(post_outcome * log(post_outcome))`) to:

            pk_c * H_correct   = -(A_c + B_c) + pk_c * log(pk_c)
            pk_i * H_incorrect = -(A_i + B_i) + pk_i * log(pk_i)

        where `A_c = (p*log(p)) . post`, `B_c = p . (post*log(post))`, and
        `A_i`/`B_i` are the same with `q` in place of `p`. `p*log(p)` and
        `q*log(q)` are constant across trials (precomputed once in
        `__init__` as `_p_log_p`/`_q_log_q`); only `post*log(post)` (a
        length-`n_param` vector) depends on the current trial. So every
        per-trial quantity above is a matrix-vector product against the
        precomputed `(n_stim, n_param)` tables, with no full-matrix `log()`
        call needed on the hot path -- this is what keeps stimulus selection
        comfortably under the 100 ms/trial budget for the default grids
        (measured ~1-2 ms; the brute-force approach of literally normalizing
        and taking `log` of two `(n_stim, n_param)` posteriors per trial
        measured ~40-60 ms for the same grids).
        """
        post = self._posterior  # (n_param,)
        post_safe = np.clip(post, 1e-300, None)
        log_post = np.log(post_safe)
        w = post * log_post  # (n_param,)

        pk_correct = np.clip(self._p_correct @ post, 1e-12, 1.0 - 1e-12)  # (n_stim,)
        pk_incorrect = 1.0 - pk_correct

        term_correct = self._p_log_p @ post + self._p_correct @ w  # (n_stim,)
        term_incorrect = self._q_log_q @ post + self._q_correct @ w  # (n_stim,)

        expected_entropy = (
            -term_correct
            - term_incorrect
            + pk_correct * np.log(pk_correct)
            + pk_incorrect * np.log(pk_incorrect)
        )
        best_idx = int(np.argmin(expected_entropy))
        contrast = float(self._stim_contrast[best_idx])
        return {
            "spatial_frequency_cpd": float(self._stim_freq[best_idx]),
            "contrast": contrast,
            # log10 contrast, duplicated from `contrast` under the key the
            # trial loop logs as the trial's scalar intensity (every
            # procedure's stimulus dict carries an "intensity" field for
            # `TrialRecord.intensity`/`intensity_units="log10_contrast"`,
            # even though qCSF's own stimulus space is 2D).
            "intensity": float(np.log10(contrast)),
        }

    def update(self, stimulus: dict[str, float], correct: bool) -> None:
        """Record a trial's outcome.

        Accepts the full dict `next_stimulus()` returns (including the
        `"intensity"` key) but only `spatial_frequency_cpd` and `contrast`
        are used to locate the presented stimulus on the grid; any other
        keys (e.g. `"intensity"`) are ignored.
        """
        if self._finished:
            return
        idx = self._nearest_stim_index(stimulus["spatial_frequency_cpd"], stimulus["contrast"])
        like = self._p_correct[idx, :] if correct else (1.0 - self._p_correct[idx, :])
        new_post = self._posterior * like
        total = float(new_post.sum())
        if total > 0 and np.isfinite(total):
            self._posterior = new_post / total
        self._n_trials += 1
        if self.max_trials is not None and self._n_trials >= self.max_trials:
            self._finished = True

    def _aulcsf_grid(
        self,
        f_lo: float = 1.0,
        f_hi: float = 18.0,
        n_pts: int = 50,
        gain: np.ndarray | None = None,
        pfreq: np.ndarray | None = None,
        bw: np.ndarray | None = None,
        trunc: np.ndarray | None = None,
    ) -> np.ndarray:
        """AULCSF for each of `gain`/`pfreq`/`bw`/`trunc` (default: every grid point)."""
        if gain is None:
            gain, pfreq, bw, trunc = (
                self._param_gain,
                self._param_freq,
                self._param_bw,
                self._param_trunc,
            )
        assert pfreq is not None and bw is not None and trunc is not None
        log10_f_grid = np.linspace(np.log10(f_lo), np.log10(f_hi), n_pts)
        f_grid = 10.0**log10_f_grid
        log_cs = self._log_cs_at(f_grid, gain, pfreq, bw, trunc)
        return np.asarray(np.trapezoid(log_cs, x=log10_f_grid, axis=0))  # (n_param,) or (1,)

    def estimate(self) -> ThresholdEstimate:
        """Return AULCSF (area under the log CSF) with a posterior credible interval.

        AULCSF is `trapz(log10 CS(f), x=log10(f))` integrated over
        `f` in [1, 18] cpd (50 log-spaced points) -- a standard
        log-log-space area, in `log10(CS) * log10(cpd)` units, *not*
        normalized by the integration range.

        The point estimate is the "plug-in" AULCSF computed *from the
        posterior-mean CSF parameters* (below), not the posterior mean of
        AULCSF evaluated grid-point-by-grid-point. The two differ because
        AULCSF is a concave function of the parameters (e.g. diminishing
        returns from `bandwidth`): by Jensen's inequality,
        `E[AULCSF(params)] <= AULCSF(E[params])` whenever the posterior
        hasn't yet concentrated, and averaging AULCSF itself was measured to
        be systematically biased low by as much as ~0.14 log units at the
        trial counts this project's validation tests use (see
        `tests/procedures/test_qcsf.py`), only closing as the posterior
        concentrates over many more trials. The plug-in estimate at the
        posterior mean avoids this and roughly halved the observed bias in
        the same simulations, which is also how AULCSF is conventionally
        reported (from a point CSF fit, not integrated over fit
        uncertainty). The *credible interval* is still the posterior-
        weighted percentile of per-grid-point AULCSF values (no Monte Carlo
        needed, since the grid itself is small) -- a CI legitimately
        describes spread and isn't subject to the same point-estimate bias.

        `extra` holds the posterior mean/SD of all 4 CSF parameters and the
        posterior-mean log10 CS at `STANDARD_FREQUENCIES_CPD`.
        """
        if self._n_trials == 0:
            raise RuntimeError("QCSF.estimate() called before any trials were recorded")

        gain_mean, gain_sd = _weighted_mean_sd(self._param_gain, self._posterior)
        freq_mean, freq_sd = _weighted_mean_sd(self._param_freq, self._posterior)
        bw_mean, bw_sd = _weighted_mean_sd(self._param_bw, self._posterior)
        trunc_mean, trunc_sd = _weighted_mean_sd(self._param_trunc, self._posterior)

        point_aulcsf = float(
            self._aulcsf_grid(
                gain=np.array([gain_mean]),
                pfreq=np.array([freq_mean]),
                bw=np.array([bw_mean]),
                trunc=np.array([trunc_mean]),
            )[0]
        )
        aulcsf_grid = self._aulcsf_grid()
        alpha = 1.0 - self.ci_level
        ci_low, ci_high = _weighted_percentile(
            aulcsf_grid, self._posterior, [alpha / 2, 1 - alpha / 2]
        )

        std_freqs = np.array(STANDARD_FREQUENCIES_CPD)
        log_cs_grid = self._log_cs_at(
            std_freqs, self._param_gain, self._param_freq, self._param_bw, self._param_trunc
        )  # (6, n_param)
        log_cs_mean = (log_cs_grid * self._posterior[None, :]).sum(axis=1)

        return ThresholdEstimate(
            value=point_aulcsf,
            ci_low=float(ci_low),
            ci_high=float(ci_high),
            ci_level=self.ci_level,
            units="aulcsf_log10cs_log10cpd",
            method="qcsf_aulcsf_at_posterior_mean_params",
            extra={
                "n_trials": self._n_trials,
                "aulcsf_integration_range_cpd": [1.0, 18.0],
                "posterior_mean": {
                    "peak_gain_log10cs": gain_mean,
                    "peak_freq_cpd": freq_mean,
                    "bandwidth_octaves": bw_mean,
                    "low_freq_truncation_log10": trunc_mean,
                },
                "posterior_sd": {
                    "peak_gain_log10cs": gain_sd,
                    "peak_freq_cpd": freq_sd,
                    "bandwidth_octaves": bw_sd,
                    "low_freq_truncation_log10": trunc_sd,
                },
                "log_cs_at_frequencies_cpd": {
                    str(f): float(v)
                    for f, v in zip(STANDARD_FREQUENCIES_CPD, log_cs_mean, strict=True)
                },
            },
        )

    def state_dict(self) -> dict[str, Any]:
        gain_mean, gain_sd = _weighted_mean_sd(self._param_gain, self._posterior)
        freq_mean, freq_sd = _weighted_mean_sd(self._param_freq, self._posterior)
        bw_mean, bw_sd = _weighted_mean_sd(self._param_bw, self._posterior)
        trunc_mean, trunc_sd = _weighted_mean_sd(self._param_trunc, self._posterior)
        return {
            "n_trials": self._n_trials,
            "finished": self._finished,
            "posterior_mean": {
                "peak_gain_log10cs": gain_mean,
                "peak_freq_cpd": freq_mean,
                "bandwidth_octaves": bw_mean,
                "low_freq_truncation_log10": trunc_mean,
            },
            "posterior_sd": {
                "peak_gain_log10cs": gain_sd,
                "peak_freq_cpd": freq_sd,
                "bandwidth_octaves": bw_sd,
                "low_freq_truncation_log10": trunc_sd,
            },
        }
