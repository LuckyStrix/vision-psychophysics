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

## The session-plan file, `participant_id`, and `calibration_hash`

`vpsych.data.schemas.SessionPlan` (the frozen pydantic model) only carries
`tests`/`ordering`/`seed` -- it has no `participant_id` or
`calibration_hash` field, both of which this runner needs (to build
`SessionInfo` and to select the calibration to check tests against). Since
that schema is frozen and this runner's CLI contract (`--session-plan`,
etc.) is also frozen from Phase 0 with no separate flags for these, this
module treats the `--session-plan` JSON file as a superset of
`SessionPlan`: it reads the raw JSON first, validates the `SessionPlan`
fields out of it with `SessionPlan.model_validate` (which silently ignores
unknown keys, pydantic's default), and separately reads two additional
top-level keys directly from the raw dict:

- `"participant_id"` (required): the pseudonymous `sub-XXXX` ID this
  session belongs to.
- `"calibration_hash"` (optional): the specific calibration to use (see
  `vpsych.data.paths.calibration_path`). If omitted, the runner picks the
  most recently created `cal-*.json` file under the data root's
  `calibration/` directory (see `_most_recent_calibration`).

This is a provisional convention introduced here to close a real gap
between the frozen `SessionPlan` schema and what a runnable session needs;
it should be reconciled with however the PySide6 app (Phase 3) and the data
layer (`vpsych.data`, built concurrently) end up naming/shaping the file
that's actually written to `--session-plan`.

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
    """

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


def resolve_simulated_observer(name: str) -> SimulatedObserver:
    """Resolve a `--simulate` observer name to a `SimulatedObserver` instance.

    Args:
        name: One of the built-in observer names (currently just
            ``"always_correct"``; see `_BUILTIN_SIMULATED_OBSERVERS`).

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
            "Run headless against a named simulated observer instead of a live PsychoPy window "
            "and keyboard (e.g. for CI end-to-end tests). Omit for a normal, real display run."
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

    See the module docstring's "The session-plan file" section for why
    `participant_id`/`calibration_hash` are read from the raw JSON rather
    than the `SessionPlan` model itself.

    Args:
        session_plan_path: Path to the `--session-plan` JSON file.
        data_root: Data root to resolve `calibration_hash` against.

    Returns:
        A `(plan, participant_id, calibration)` tuple. `calibration` is
        `None` if no `calibration_hash` was given and no calibration file
        exists under the data root's `calibration/` directory.

    Raises:
        ValueError: If the file has no `"participant_id"` key.
        FileNotFoundError: If a `calibration_hash` was given but no
            matching calibration file exists.
    """
    raw = json.loads(session_plan_path.read_text(encoding="utf-8"))
    plan = SessionPlan.model_validate(raw)

    participant_id = raw.get("participant_id")
    if not participant_id:
        raise ValueError(
            f"{session_plan_path} is missing a required top-level 'participant_id' field."
        )

    cal_hash = raw.get("calibration_hash")
    cal_dir = calibration_dir(data_root)
    if cal_hash:
        cal_path = cal_dir / f"cal-{cal_hash}.json"
        if not cal_path.exists():
            raise FileNotFoundError(f"Calibration {cal_hash!r} not found at {cal_path}.")
        calibration: Calibration | None = Calibration.model_validate_json(
            cal_path.read_text(encoding="utf-8")
        )
    else:
        calibration = _most_recent_calibration(cal_dir)

    return plan, str(participant_id), calibration


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
) -> RunnerExitCode:
    """Run one full session per `args`, returning the process exit code.

    Split out from `main` so tests can call it directly with an injected
    in-memory `writer_factory` (and, for `--simulate` runs, a pre-resolved
    `simulated_observer`), without needing a real `SessionWriter` or CLI
    invocation.

    Args:
        args: Parsed arguments (see `build_arg_parser`).
        writer_factory: Constructs the `Writer` used to persist this
            session's data; defaults to the real `SessionWriter` (which
            currently raises `NotImplementedError` until that module is
            implemented -- caught by this function's error handling like
            any other exception).
        simulated_observer: When `args.simulate` is set, the observer to
            drive `SimulatedBackend` with; if `None`, resolved from
            `args.simulate` via `resolve_simulated_observer`.

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
        from vpsych.tests_catalog.base import check_requirements, get_test

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

        if args.simulate:
            observer = simulated_observer or resolve_simulated_observer(args.simulate)
            backend = SimulatedBackend(observer)
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
