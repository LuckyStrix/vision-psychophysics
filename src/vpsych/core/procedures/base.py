"""Adaptive-procedure interface shared by all threshold-estimation methods.

Every psychophysical test drives its trial-by-trial intensity selection
through an :class:`AdaptiveProcedure` (single-parameter: QUEST+, weighted
staircase, method of constant stimuli) or a :class:`MultiParamProcedure`
(multi-parameter: qCSF). This lets `tests_catalog` plugins, the trial loop,
and the simulated-observer recovery tests in `tests/` all depend on one
stable contract instead of each procedure's own API.

Concrete procedures live in sibling modules (`questplus_procedure.py`,
`staircase.py`, `constant_stimuli.py`, `qcsf.py`) and are documented there;
this module only defines the shared contract and the result types.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class ThresholdEstimate(BaseModel):
    """A point estimate and confidence interval for a psychophysical threshold.

    Attributes:
        value: The estimated threshold, in ``units``. For log-scaled
            intensity dimensions (e.g. contrast, coherence) this is
            typically reported in the same scale the procedure ran on
            (check ``units`` / the owning test's ``output_units``).
        ci_low: Lower bound of the confidence interval, in ``units``.
        ci_high: Upper bound of the confidence interval, in ``units``.
        ci_level: Nominal coverage of the interval, e.g. ``0.95`` for a 95%
            CI.
        units: Units of ``value``/``ci_low``/``ci_high``, e.g. ``"logMAR"``,
            ``"log10_contrast"``, ``"coherence_fraction"``, ``"arcsec"``,
            ``"hz"``.
        method: Name of the estimation method that produced this estimate,
            e.g. ``"quest_plus_posterior_mean"``, ``"weibull_mle"``,
            ``"reversal_mean"``.
        extra: Method-specific additional detail (e.g. posterior SD, number
            of reversals used), JSON-serializable.
    """

    # NaN/inf must survive a JSON round trip (default serialization turns them into null,
    # which then fails validation and makes the whole summary unreadable).
    model_config = ConfigDict(frozen=True, ser_json_inf_nan="constants")

    value: float = Field(description="Estimated threshold, in `units`.")
    ci_low: float = Field(description="Lower confidence bound, in `units`.")
    ci_high: float = Field(description="Upper confidence bound, in `units`.")
    ci_level: float = Field(gt=0, lt=1, description="Nominal CI coverage, e.g. 0.95.")
    units: str = Field(description="Units of value/ci_low/ci_high.")
    method: str = Field(description="Name of the estimation method used.")
    extra: dict[str, Any] = Field(
        default_factory=dict, description="Method-specific extra detail, JSON-serializable."
    )


@runtime_checkable
class AdaptiveProcedure(Protocol):
    """Protocol for a single-parameter adaptive (or fixed-set) intensity-selection procedure.

    Implementations drive one scalar intensity dimension per trial (e.g.
    logMAR letter size, log contrast, motion coherence) and update their
    internal state from each trial's outcome. The trial loop in
    `core/trial.py` interacts with a procedure only through this interface,
    so QUEST+, a weighted staircase, and the method of constant stimuli are
    interchangeable from its point of view.
    """

    intensity_units: str
    """Units of the intensity values returned by `next_intensity` and passed
    to `update`, e.g. `"logMAR"`, `"log10_contrast"`, `"coherence_fraction"`."""

    finished: bool
    """`True` once this procedure has collected enough trials to stop
    (fixed trial count reached, staircase reversal criterion met, or
    equivalent); the trial loop stops requesting intensities once this is
    `True`."""

    def next_intensity(self) -> float:
        """Return the intensity to present on the next trial, in `intensity_units`.

        Returns:
            The next stimulus intensity to present.
        """
        ...

    def update(self, intensity: float, correct: bool) -> None:
        """Record the outcome of a trial and advance the procedure's internal state.

        Args:
            intensity: The intensity that was actually presented on this
                trial, in `intensity_units` (normally the value most
                recently returned by `next_intensity`).
            correct: Whether the observer's response was scored correct.
        """
        ...

    def estimate(self) -> ThresholdEstimate:
        """Return the procedure's current best threshold estimate and CI.

        May be called before `finished` is `True` to get an interim
        estimate (e.g. for adaptive stopping rules or live progress
        display); the estimate typically becomes more precise as more
        trials accumulate.

        Returns:
            The current threshold estimate.
        """
        ...

    def state_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of internal state, for per-trial logging.

        Called after each `update` so the trial record
        (`vpsych.core.trial.TrialRecord.procedure_state`) captures enough of
        the procedure's internal state (e.g. QUEST+ posterior, staircase
        reversal count and direction) to fully reconstruct its trajectory
        during reanalysis.

        Returns:
            A JSON-serializable dict of procedure-specific state.
        """
        ...


class MultiParamProcedure(ABC):
    """Base class for adaptive procedures over multiple stimulus dimensions (e.g. qCSF).

    Unlike :class:`AdaptiveProcedure`, a multi-parameter procedure selects
    and updates on a *stimulus* -- a dict of named dimensions (e.g.
    ``{"spatial_frequency_cpd": 2.0, "contrast": 0.1}``) -- rather than a
    single scalar intensity, because the underlying model (e.g. the
    truncated-log-parabola CSF in qCSF) has several free parameters jointly
    informed by stimuli that vary along more than one axis.
    """

    @abstractmethod
    def next_stimulus(self) -> dict[str, float]:
        """Return the next stimulus to present, as a dict of named parameter values.

        Returns:
            A mapping from parameter name (e.g. ``"spatial_frequency_cpd"``,
            ``"contrast"``) to its value for the next trial. Keys are
            procedure-specific but must be documented by the concrete
            subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, stimulus: dict[str, float], correct: bool) -> None:
        """Record the outcome of a trial and advance the procedure's internal state.

        Args:
            stimulus: The stimulus parameter dict that was actually
                presented (normally the value most recently returned by
                `next_stimulus`).
            correct: Whether the observer's response was scored correct.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def finished(self) -> bool:
        """`True` once this procedure has collected enough trials to stop."""
        raise NotImplementedError

    @abstractmethod
    def estimate(self) -> ThresholdEstimate:
        """Return the procedure's current best parameter estimate and CI.

        For qCSF this is typically the derived summary threshold (e.g. peak
        sensitivity or AULCSF); full fitted parameters belong in
        `ThresholdEstimate.extra`.

        Returns:
            The current estimate.
        """
        raise NotImplementedError

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of internal state, for per-trial logging.

        Returns:
            A JSON-serializable dict of procedure-specific state.
        """
        raise NotImplementedError
