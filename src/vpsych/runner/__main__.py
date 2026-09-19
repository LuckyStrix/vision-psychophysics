"""Runner subprocess entry point: `python -m vpsych.runner` / the `vpsych-run` console script.

Runs one session end to end in its own process, so a fullscreen PsychoPy
window never has to share a process (or an event loop) with the PySide6
app. Writes trial data via a `Writer` (see below) and a
`vpsych.runner.status.RunnerStatus` snapshot to `--status-file` as it goes,
for the app to poll/watch.

`--simulate OBSERVER` runs the same trial loop headless, driven by a
`vpsych.core.observers.SimulatedObserver` instead of a live PsychoPy window
and keyboard (see `vpsych.core.trial_loop.SimulatedBackend`), so end-to-end
session logic (trial loop, writer, status updates, exit codes) is
exercisable in CI without a display.

## The session-plan file

`--session-plan` points at a JSON file that validates directly as a
`vpsych.data.schemas.SessionPlan`, which carries `participant_id` (the
pseudonymous `sub-XXXX` ID this session belongs to) and an optional
`calibration_hash` (the specific calibration to use; see
`vpsych.data.paths.calibration_path`). If `calibration_hash` is omitted,
the runner picks the most recently created `cal-*.json` file under the data
root's `calibration/` directory (see `_most_recent_calibration`).

## The `Writer` protocol

`vpsych.data.writer.SessionWriter` is a Phase-0 interface stub (its
`__init__` raises `NotImplementedError`) still being implemented
concurrently. Rather than depend on it directly, this module defines
`Writer`, a structural protocol matching the four methods it actually
calls (`append_trial`, `write_summary`, `write_frames`, `finalize`) --
`vpsych.core.trial_loop.TrialWriter` (a narrower protocol covering just the
two methods the trial loop itself needs) is structurally compatible with
any `Writer`. `run_session`'s `writer_factory` parameter lets tests supply
an in-memory fake; the default factory constructs a real `SessionWriter`
(and so will raise `NotImplementedError` until that module is implemented
-- caught like any other exception, producing `status="error"` and
`RunnerExitCode.ERROR`, which is the correct/expected behavior until then).
"""

from __future__ import annotations

import argparse
import json
import platform
import signal
import sys
import traceback
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from vpsych.core.calibration.models import Calibration
from vpsych.core.display import DisplayGeometry
from vpsych.core.observers import SimulatedObserver
from vpsych.core.rng import make_rng
from vpsych.core.timing import measure_refresh, refresh_matches
from vpsych.core.trial import TrialRecord, TrialTimeline
from vpsych.core.trial_loop import (
    PresentationBackend,
    PsychoPyBackend,
    SimulatedBackend,
    TrialLoop,
    TrialLoopResult,
)
from vpsych.data.paths import calibration_dir
from vpsych.data.paths import data_root as default_data_root
from vpsych.data.schemas import SessionInfo, SessionPlan, SessionStatus, TestSummary
from vpsych.runner.status import RunnerExitCode, RunnerStatus

try:  # pragma: no cover - version metadata only
    from importlib.metadata import version as _pkg_version

    _VPSYCH_VERSION = _pkg_version("vpsych")
except Exception:  # pragma: no cover
    _VPSYCH_VERSION = "0.0.0+unknown"


@runtime_checkable
class Writer(Protocol):
    """Structural protocol matching the `SessionWriter` methods this runner calls.

    See the module docstring's "The `Writer` protocol" section.
    """

    def append_trial(self, trial: TrialRecord) -> None: ...

    def write_summary(self, task_id: str, eye: str, run: int, summary: TestSummary) -> None: ...

    def write_frames(self, task_id: str, eye: str, run: int, intervals_s: list[float]) -> None: ...

    def finalize(self, status: SessionStatus) -> None: ...


class _AlwaysCorrectObserver:
    """Trivial built-in simulated observer: answers every trial correctly.

    Requires the presented stimulus dict passed to `respond` to include a
    ``"correct_response"`` key (see the `trial_ctx` contract documented in
    `vpsych.core.trial_loop`). Useful for smoke-testing the runner's
    session lifecycle (status updates, exit codes, writer calls) against a
    minimal test before real, statistically calibrated simulated observers
    (`vpsych.core.observers.PsychometricObserver`) are implemented.

    Also implements `decide_correct` (trivially: always `True`), the
    decoupled response-mapping path documented on
    `vpsych.core.observers.SimulatedObserver` -- unlike `respond`, it needs
    no stimulus keys at all.
    """

    def decide_correct(self, stimulus: dict[str, Any], rng: Any) -> bool:
        del stimulus, rng
        return True

    def respond(self, stimulus: dict[str, Any], rng: Any) -> Any:
        del rng
        if "correct_response" not in stimulus:
            raise ValueError(
                "_AlwaysCorrectObserver requires stimulus['correct_response'] to be set by "
                "the test's present() implementation."
            )
        return stimulus["correct_response"]


_BUILTIN_SIMULATED_OBSERVERS: dict[str, Callable[[], SimulatedObserver]] = {
    "always_correct": _AlwaysCorrectObserver,
}

#: Registry of `build_simulated_observer` "kind" extensions, beyond the two
#: built in (`"psychometric"`, `"csf"`). Was previously a closed `if/elif`
#: in `build_simulated_observer` that only this module could extend --
#: `color_discrimination`'s multi-axis `TrivectorObserver` needs a per-axis
#: threshold spec (`{"threshold_protan": ..., "threshold_deutan": ...,
#: "threshold_tritan": ...}`) that neither of the two built-in kinds can
#: express, and had to bypass `--simulate`/`--simulate-config` entirely,
#: injecting a directly-constructed observer via `run_session`'s
#: `simulated_observer=` kwarg instead (see
#: `vpsych.tests_catalog.color_discrimination.observer`'s module docstring
#: for the full history). `register_simulated_observer_kind` lets a test
#: package register its own kind (as an import-time side effect, e.g. in
#: its own `observer.py`, imported by that package's `__init__.py`) so it
#: becomes resolvable through the normal `--simulate`/`--simulate-config`
#: CLI/JSON path like any built-in kind.
_REGISTERED_SIMULATED_OBSERVER_KINDS: dict[
    str, Callable[[dict[str, float]], SimulatedObserver]
] = {}


def register_simulated_observer_kind(
    kind: str, builder: Callable[[dict[str, float]], SimulatedObserver]
) -> None:
    """Register a `build_simulated_observer` "kind" extension.

    Args:
        kind: The `--simulate kind:...`/`{"kind": ...}` name this builder
            handles, e.g. `"trivector"`. Must not collide with a built-in
            kind (`"psychometric"`, `"csf"`) or an already-registered one.
        builder: Called with the parsed `dict[str, float]` params (see
            `_parse_kv_params`) and must return a constructed
            `SimulatedObserver`, raising `ValueError` for a missing/invalid
            parameter -- the same contract `build_simulated_observer`'s
            built-in kinds follow.

    Raises:
        ValueError: If `kind` is a built-in kind or already registered.
    """
    if kind in ("psychometric", "csf"):
        raise ValueError(f"Simulated observer kind {kind!r} is a built-in kind, cannot register.")
    if kind in _REGISTERED_SIMULATED_OBSERVER_KINDS:
        raise ValueError(f"Simulated observer kind {kind!r} is already registered.")
    _REGISTERED_SIMULATED_OBSERVER_KINDS[kind] = builder


def resolve_simulated_observer(name: str) -> SimulatedObserver:
    """Resolve a bare `--simulate` observer name to a `SimulatedObserver` instance.

    Args:
        name: One of the built-in observer names (currently just
            ``"always_correct"``; see `_BUILTIN_SIMULATED_OBSERVERS`). For a
            parameterized observer spec (``"psychometric:..."``,
            ``"csf:..."``), use `parse_simulated_observer_spec` instead.

    Returns:
        The constructed observer.

    Raises:
        ValueError: If `name` is not a known built-in observer.
    """
    try:
        return _BUILTIN_SIMULATED_OBSERVERS[name]()
    except KeyError:
        raise ValueError(
            f"Unknown simulated observer {name!r}. Known: {sorted(_BUILTIN_SIMULATED_OBSERVERS)}"
        ) from None


def _parse_kv_params(rest: str) -> dict[str, float]:
    """Parse a comma-separated ``key=value`` parameter string into a `dict[str, float]`."""
    params: dict[str, float] = {}
    for pair in rest.split(","):
        pair = pair.strip()
        if not pair:
            continue
        key, sep, value = pair.partition("=")
        if not sep:
            raise ValueError(f"Malformed observer parameter {pair!r}; expected key=value.")
        try:
            params[key.strip()] = float(value)
        except ValueError:
            raise ValueError(f"Observer parameter {pair!r} has a non-numeric value.") from None
    return params


def build_simulated_observer(kind: str, params: dict[str, float]) -> SimulatedObserver:
    """Construct a parameterized `SimulatedObserver` from a `kind` and numeric `params`.

    Args:
        kind: ``"psychometric"`` (-> `vpsych.core.observers.PsychometricObserver`,
            a fixed-family weibull psychometric function on a log10 intensity
            scale) or ``"csf"`` (-> `vpsych.core.observers.CSFObserver`).
        params: Numeric parameters for `kind` (see `parse_simulated_observer_spec`
            for the recognized keys and their defaults).

    Returns:
        The constructed observer.

    Raises:
        ValueError: If `kind` is unknown, or a required parameter is missing.
    """
    from vpsych.core.observers import CSFObserver, PsychometricObserver
    from vpsych.core.procedures.qcsf import DEFAULT_PSYCHOMETRIC_SLOPE
    from vpsych.core.psychometric import PsychometricFunction

    if kind in _REGISTERED_SIMULATED_OBSERVER_KINDS:
        return _REGISTERED_SIMULATED_OBSERVER_KINDS[kind](params)

    if kind == "psychometric":
        missing = [k for k in ("threshold", "slope") if k not in params]
        if missing:
            raise ValueError(
                f"psychometric observer spec is missing required parameter(s) {missing}: "
                f"needs at least threshold=<float>,slope=<float>."
            )
        fn = PsychometricFunction(
            family="weibull",
            threshold=params["threshold"],
            slope=params["slope"],
            guess=params.get("guess", 0.5),
            lapse=params.get("lapse", 0.02),
            intensity_scale="log10",
        )
        return PsychometricObserver(fn, n_afc=int(params.get("n_afc", 2)))

    if kind == "csf":
        required = ("peak_gain", "peak_freq", "bandwidth", "low_freq_truncation")
        missing = [k for k in required if k not in params]
        if missing:
            raise ValueError(f"csf observer spec is missing required parameter(s) {missing}.")
        return CSFObserver(
            peak_gain_log10=params["peak_gain"],
            peak_freq_cpd=params["peak_freq"],
            bandwidth_octaves=params["bandwidth"],
            low_freq_truncation_log10=params["low_freq_truncation"],
            n_afc=int(params.get("n_afc", 2)),
            slope=params.get("slope", DEFAULT_PSYCHOMETRIC_SLOPE),
            lapse_rate=params.get("lapse", 0.02),
        )

    known = ["psychometric", "csf", *sorted(_REGISTERED_SIMULATED_OBSERVER_KINDS)]
    raise ValueError(f"Unknown simulated observer kind {kind!r}. Known: {', '.join(known)}.")


def parse_simulated_observer_spec(spec: str) -> SimulatedObserver:
    """Parse a `--simulate`/`--simulate-config` observer spec string.

    Recognized forms:

    - ``"always_correct"``: the trivial built-in observer (see
      `resolve_simulated_observer`).
    - ``"psychometric:threshold=<f>,slope=<f>[,lapse=<f>][,guess=<f>][,n_afc=<int>]"``:
      a `vpsych.core.observers.PsychometricObserver` with a fixed
      ``family="weibull"``, ``intensity_scale="log10"`` true function.
      ``lapse`` defaults to ``0.02``, ``guess`` to ``0.5``, ``n_afc`` to ``2``.
    - ``"csf:peak_gain=<f>,peak_freq=<f>,bandwidth=<f>,low_freq_truncation=<f>"
      "[,slope=<f>][,lapse=<f>][,n_afc=<int>]"``: a
      `vpsych.core.observers.CSFObserver`. ``slope`` defaults to
      `vpsych.core.procedures.qcsf.DEFAULT_PSYCHOMETRIC_SLOPE`, ``lapse`` to
      ``0.02``, ``n_afc`` to ``2``.

    Args:
        spec: The spec string, e.g.
            ``"psychometric:threshold=-1.0,slope=3.5,lapse=0.02"``.

    Returns:
        The constructed observer.

    Raises:
        ValueError: If `spec` names an unknown kind/observer, is missing a
            required parameter, or has a malformed ``key=value`` pair.
    """
    if ":" not in spec:
        return resolve_simulated_observer(spec)
    kind, _, rest = spec.partition(":")
    return build_simulated_observer(kind.strip(), _parse_kv_params(rest))


def _simulated_observer_from_json(obj: Any) -> SimulatedObserver:
    """Build a `SimulatedObserver` from one `--simulate-config` JSON entry.

    `obj` may be a spec string (parsed via `parse_simulated_observer_spec`)
    or a JSON object ``{"kind": "psychometric", "threshold": -1.0, ...}``
    (numeric keys other than ``"kind"`` are passed to `build_simulated_observer`).
    """
    if isinstance(obj, str):
        return parse_simulated_observer_spec(obj)
    if isinstance(obj, dict):
        kind = obj.get("kind")
        if not kind:
            raise ValueError(f"--simulate-config entry is missing a 'kind' key: {obj!r}")
        params = {k: float(v) for k, v in obj.items() if k != "kind"}
        return build_simulated_observer(str(kind), params)
    raise ValueError(
        f"--simulate-config entries must be a spec string or a JSON object, got {obj!r}"
    )


def load_simulate_config(path: Path) -> dict[str, SimulatedObserver]:
    """Load `--simulate-config PATH`: a per-task JSON map of observer specs.

    Args:
        path: Path to a JSON file whose top level is an object mapping
            `TestSpec.id` (task id) to either a spec string (see
            `parse_simulated_observer_spec`) or a JSON object
            (see `_simulated_observer_from_json`), e.g.::

                {
                  "visual_acuity": "psychometric:threshold=-1.0,slope=3.5,lapse=0.02",
                  "contrast_sensitivity_function": {
                    "kind": "csf", "peak_gain": 1.6, "peak_freq": 3.0,
                    "bandwidth": 3.0, "low_freq_truncation": 1.0
                  }
                }

    Returns:
        A `dict` mapping task id to the constructed observer for that task.

    Raises:
        ValueError: If the file's top level isn't a JSON object, or any
            entry is malformed (see `_simulated_observer_from_json`).
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a JSON object mapping task_id -> observer spec.")
    return {str(task_id): _simulated_observer_from_json(v) for task_id, v in raw.items()}


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the `vpsych-run` / `python -m vpsych.runner` argument parser.

    Returns:
        A configured `argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="vpsych-run",
        description="Run one vpsych session (a subprocess launched by the app, or standalone).",
    )
    parser.add_argument(
        "--session-plan",
        type=Path,
        required=True,
        metavar="PATH",
        help="Path to a session-plan JSON file (see vpsych.data.schemas.SessionPlan).",
    )
    parser.add_argument(
        "--status-file",
        type=Path,
        required=True,
        metavar="PATH",
        help=(
            "Path to write RunnerStatus JSON snapshots to as the session progresses "
            "(see vpsych.runner.status.RunnerStatus.write_atomic)."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        metavar="PATH",
        help="Data root to write session output under (default: VPSYCH_DATA_ROOT env var, or ~/vpsych-data).",
    )
    parser.add_argument(
        "--simulate",
        type=str,
        default=None,
        metavar="OBSERVER",
        help=(
            "Run headless against a simulated observer instead of a live PsychoPy window and "
            "keyboard (e.g. for CI end-to-end tests), used as every task's observer except where "
            "--simulate-config overrides it for a specific task. OBSERVER is a built-in name "
            "('always_correct') or a parameterized spec ('psychometric:threshold=-1.0,"
            "slope=3.5,lapse=0.02' or 'csf:peak_gain=1.6,peak_freq=3.0,bandwidth=3.0,"
            "low_freq_truncation=1.0'; see parse_simulated_observer_spec). Omit both --simulate "
            "and --simulate-config for a normal, real display run."
        ),
    )
    parser.add_argument(
        "--simulate-config",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Path to a JSON file mapping task_id -> observer spec (string or object; see "
            "load_simulate_config) for per-task simulated-observer parameters. Implies "
            "--simulate mode even without --simulate itself; a task not listed here falls back "
            "to --simulate's observer, or is an error if --simulate was not given either."
        ),
    )
    return parser


class _SessionAbortedError(Exception):
    """Raised from a SIGINT/SIGTERM handler to unwind the session cleanly as an abort."""


def _install_signal_handlers() -> Callable[[], None]:
    """Install SIGINT/SIGTERM handlers that raise `_SessionAbortedError`; return a restore function."""
    previous = {
        signal.SIGINT: signal.getsignal(signal.SIGINT),
        signal.SIGTERM: signal.getsignal(signal.SIGTERM),
    }

    def _handler(signum: int, frame: Any) -> None:
        del frame
        raise _SessionAbortedError(f"Received signal {signum}")

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)

    def _restore() -> None:
        for sig, handler in previous.items():
            signal.signal(sig, handler)

    return _restore


def _most_recent_calibration(cal_dir: Path) -> Calibration | None:
    """Pick the most recently created calibration under `cal_dir`, or `None` if there is none."""
    if not cal_dir.exists():
        return None
    candidates: list[Calibration] = []
    for path in sorted(cal_dir.glob("cal-*.json")):
        try:
            candidates.append(Calibration.model_validate_json(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    if not candidates:
        return None
    return max(candidates, key=lambda c: c.created_utc)


def load_session_plan(
    session_plan_path: Path, data_root: Path
) -> tuple[SessionPlan, str, Calibration | None]:
    """Load the session plan, participant ID, and active calibration.

    Args:
        session_plan_path: Path to the `--session-plan` JSON file (must
            validate as `vpsych.data.schemas.SessionPlan`).
        data_root: Data root to resolve `calibration_hash` against.

    Returns:
        A `(plan, participant_id, calibration)` tuple. `calibration` is
        `None` if `plan.calibration_hash` is `None` and no calibration file
        exists under the data root's `calibration/` directory.

    Raises:
        FileNotFoundError: If `plan.calibration_hash` was given but no
            matching calibration file exists.
    """
    raw = json.loads(session_plan_path.read_text(encoding="utf-8"))
    plan = SessionPlan.model_validate(raw)

    cal_dir = calibration_dir(data_root)
    if plan.calibration_hash:
        cal_path = cal_dir / f"cal-{plan.calibration_hash}.json"
        if not cal_path.exists():
            raise FileNotFoundError(
                f"Calibration {plan.calibration_hash!r} not found at {cal_path}."
            )
        calibration: Calibration | None = Calibration.model_validate_json(
            cal_path.read_text(encoding="utf-8")
        )
    else:
        calibration = _most_recent_calibration(cal_dir)

    return plan, plan.participant_id, calibration


def _default_writer_factory(
    participant_id: str, session_id: str, session_info: SessionInfo, root: Path
) -> Writer:
    from vpsych.data.writer import SessionWriter

    return SessionWriter(participant_id, session_id, session_info, root)


def run_session(
    args: argparse.Namespace,
    *,
    writer_factory: Callable[[str, str, SessionInfo, Path], Writer] = _default_writer_factory,
    simulated_observer: SimulatedObserver | None = None,
    abort_after_trials: int | None = None,
) -> RunnerExitCode:
    """Run one full session per `args`, returning the process exit code.

    Split out from `main` so tests can call it directly with an injected
    in-memory `writer_factory` (and, for `--simulate` runs, a pre-resolved
    `simulated_observer`), without needing a real `SessionWriter` or CLI
    invocation.

    Args:
        args: Parsed arguments (see `build_arg_parser`). `args.simulate`
            and/or `args.simulate_config` select simulated-observer mode
            (see `parse_simulated_observer_spec`/`load_simulate_config`);
            when `args.simulate_config` gives a per-task observer for every
            planned task, `args.simulate` may be omitted.
        writer_factory: Constructs the `Writer` used to persist this
            session's data; defaults to the real `SessionWriter` (which
            currently raises `NotImplementedError` until that module is
            implemented -- caught by this function's error handling like
            any other exception).
        simulated_observer: When simulate mode is active, the *default*
            observer to drive `SimulatedBackend` with for any task not
            covered by `args.simulate_config`; if `None`, resolved from
            `args.simulate` via `parse_simulated_observer_spec`.
        abort_after_trials: Only meaningful in `--simulate` mode: forwarded
            to `vpsych.core.trial_loop.SimulatedBackend`'s constructor so
            the session aborts partway through, deterministically, after
            this many presented trials. Test-only support for exercising
            the abort path (an interrupted session's data retained, status
            `"incomplete"`/`"aborted"`) without needing OS signal timing on
            a real subprocess; `None` (the default) never aborts this way.

    Returns:
        A `RunnerExitCode` value.
    """
    status_path: Path = args.status_file
    restore_signals = _install_signal_handlers()
    backend: PresentationBackend | None = None
    writer: Writer | None = None
    final_status: SessionStatus = "incomplete"
    exit_code = RunnerExitCode.ERROR

    def _status(**kwargs: Any) -> None:
        RunnerStatus.create(**kwargs).write_atomic(status_path)

    try:
        _status(state="starting", message="Loading session plan and calibration.")

        data_root = args.data_root or default_data_root()
        plan, participant_id, calibration = load_session_plan(args.session_plan, data_root)

        if calibration is None:
            _status(
                state="error",
                error="No calibration available. Run calibration before starting a session.",
            )
            return RunnerExitCode.REQUIREMENTS_UNMET

        # Resolve tests and check requirements against every planned test before
        # touching a display. Imported lazily to avoid a hard dependency on the
        # tests_catalog registry being populated at module import time.
        from vpsych.tests_catalog.base import check_requirements, discover_tests, get_test

        # This is a fresh process (the runner is always launched as a subprocess):
        # nothing has imported any real test's subpackage yet, so get_test() below
        # would find an empty registry without this -- see discover_tests()'s
        # docstring.
        discover_tests()

        unmet: list[str] = []
        resolved_tests = []
        for planned in plan.tests:
            try:
                test_cls = get_test(planned.task_id)
            except KeyError as exc:
                unmet.append(str(exc))
                continue
            resolved_tests.append((planned, test_cls))
            test_display = calibration.geometry.model_copy(
                update={"viewing_distance_cm": planned.viewing_distance_cm}
            )
            reasons = check_requirements(test_cls.spec.requirements, test_display, calibration)
            unmet.extend(f"{planned.task_id}: {reason}" for reason in reasons)

        if unmet:
            _status(state="error", error="; ".join(unmet))
            return RunnerExitCode.REQUIREMENTS_UNMET

        n_tests = len(resolved_tests)
        _status(state="calibrating_refresh", n_tests=n_tests, message="Opening display.")

        rng, rng_seed = make_rng(plan.seed)

        simulate_config_path: Path | None = getattr(args, "simulate_config", None)
        per_task_observers: dict[str, SimulatedObserver] = (
            load_simulate_config(simulate_config_path) if simulate_config_path else {}
        )
        default_observer = simulated_observer
        if default_observer is None and args.simulate:
            default_observer = parse_simulated_observer_spec(args.simulate)

        if args.simulate or simulate_config_path is not None:
            for planned, _test_cls in resolved_tests:
                if planned.task_id not in per_task_observers and default_observer is None:
                    raise ValueError(
                        f"--simulate-config has no entry for task {planned.task_id!r} and no "
                        "default --simulate observer was given."
                    )
            initial_observer = (
                per_task_observers.get(resolved_tests[0][0].task_id, default_observer)
                if resolved_tests
                else default_observer
            )
            assert initial_observer is not None  # guaranteed by the loop above
            backend = SimulatedBackend(initial_observer, abort_after_trials=abort_after_trials)
            measured_refresh_hz = calibration.geometry.refresh_hz
        else:
            backend = PsychoPyBackend(calibration.geometry)
            measured_refresh_hz = measure_refresh(backend.win)
            if not refresh_matches(measured_refresh_hz, calibration.geometry.refresh_hz, tol=0.01):
                _status(
                    state="error",
                    n_tests=n_tests,
                    error=(
                        f"Measured refresh {measured_refresh_hz:.3f} Hz does not match "
                        f"calibration refresh {calibration.geometry.refresh_hz:.3f} Hz "
                        "within 1%."
                    ),
                )
                return RunnerExitCode.REFRESH_MISMATCH

        # Order tests per the plan.
        order = list(range(n_tests))
        if plan.ordering == "randomized":
            rng.shuffle(order)

        session_id = f"ses-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}"
        session_info = SessionInfo(
            session_id=session_id,
            participant_id=participant_id,
            plan=plan,
            calibration_hash=calibration.content_hash(),
            display=calibration.geometry.model_copy(update={"refresh_hz": measured_refresh_hz}),
            os_info=platform.platform(),
            python_version=platform.python_version(),
            psychopy_version=_psychopy_version(),
            gpu_info=None,
            software_version=_VPSYCH_VERSION,
            git_commit=None,
            status="running",
            started_utc=datetime.now(timezone.utc),
            environment=_placeholder_environment_checklist(),
        )
        writer = writer_factory(participant_id, session_id, session_info, data_root)

        run_numbers: dict[str, int] = {}
        aborted = False

        for idx in order:
            planned, test_cls = resolved_tests[idx]
            run_number = run_numbers.get(planned.task_id, 0) + 1
            run_numbers[planned.task_id] = run_number

            if isinstance(backend, SimulatedBackend):
                task_observer = per_task_observers.get(planned.task_id, default_observer)
                assert task_observer is not None  # guaranteed by the pre-loop check above
                backend.set_observer(task_observer)

            display = calibration.geometry.model_copy(
                update={
                    "viewing_distance_cm": planned.viewing_distance_cm,
                    "refresh_hz": measured_refresh_hz,
                }
            )
            params = test_cls.spec.params_model.model_validate(planned.params)
            test = test_cls(params=params, display=display, calibration=calibration, rng=rng)
            procedure = test.make_procedure()

            if backend.win is not None:
                test.build_stimuli(backend.win)

            timeline = _default_timeline(display)
            loop = TrialLoop(
                test,
                procedure,
                backend,
                writer,
                participant_id=participant_id,
                session_id=session_id,
                eye=planned.eye,
                run_number=run_number,
                timeline=timeline,
                rng=rng,
                rng_seed=rng_seed,
                status_path=status_path,
                current_test_index=idx,
                n_tests=n_tests,
            )
            result: TrialLoopResult = loop.run()

            if result.trial_records:
                trials_df = pd.DataFrame(
                    [r.model_dump(mode="python") for r in result.trial_records]
                )
                summary = test.summarize(trials_df)
                writer.write_summary(test.spec.id, planned.eye, run_number, summary)

            if result.aborted:
                aborted = True
                break

        if aborted:
            final_status = "aborted"
            exit_code = RunnerExitCode.ABORTED_BY_USER
            _status(state="aborted", n_tests=n_tests, message="Session aborted by observer.")
        else:
            final_status = "complete"
            exit_code = RunnerExitCode.OK
            _status(state="finished", n_tests=n_tests, message="Session complete.")

        return exit_code

    except _SessionAbortedError:
        final_status = "aborted"
        exit_code = RunnerExitCode.ABORTED_BY_USER
        _status(state="aborted", error="Session aborted by signal.")
        return exit_code

    except Exception as exc:
        final_status = "incomplete"
        exit_code = RunnerExitCode.ERROR
        _status(state="error", error=f"{exc}\n{traceback.format_exc()}")
        return exit_code

    finally:
        if writer is not None:
            with _SuppressSecondaryErrors():
                writer.finalize(final_status)
        if backend is not None and hasattr(backend, "close"):
            with _SuppressSecondaryErrors():
                backend.close()
        restore_signals()


class _SuppressSecondaryErrors:
    """Context manager that swallows exceptions raised while already unwinding an error path.

    Used only for best-effort cleanup (`writer.finalize`, `backend.close`) in
    `run_session`'s `finally` block, so a cleanup failure never masks the
    original exit code/status already decided.
    """

    def __enter__(self) -> _SuppressSecondaryErrors:
        return self

    def __exit__(self, exc_type: object, exc_value: object, tb: object) -> bool:
        return exc_type is not None


def _psychopy_version() -> str:
    try:
        import psychopy

        return str(getattr(psychopy, "__version__", "unknown"))
    except Exception:  # pragma: no cover
        return "unknown"


def _placeholder_environment_checklist() -> Any:
    from vpsych.core.calibration.models import EnvironmentChecklist

    return EnvironmentChecklist(
        room_lighting_controlled=True,
        monitor_warmed_up=True,
        night_light_disabled=True,
        hdr_disabled=True,
        notes="Environment checklist not yet collected by the calibration wizard (Phase 3).",
    )


def _default_timeline(display: DisplayGeometry) -> TrialTimeline:
    """A generic default trial timeline; individual tests may use their own instead.

    See `vpsych.core.trial_loop`'s module docstring: `trial_ctx["timeline"]`
    is a convenience default, not authoritative -- a test's `present()` is
    free to derive its own frame counts from its own params.
    """
    return TrialTimeline(
        fixation_frames=display.frames_for_ms(500.0),
        stimulus_frames=display.frames_for_ms(200.0),
        response_timeout_frames=None,
        iti_frames=display.frames_for_ms(500.0),
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point for the `vpsych-run` console script.

    Args:
        argv: Command-line arguments (excluding the program name), or
            `None` to use `sys.argv[1:]`.

    Returns:
        A `RunnerExitCode` value (see `vpsych.runner.status.RunnerExitCode`)
        to use as the process exit code.
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return int(run_session(args))


if __name__ == "__main__":
    sys.exit(main())
