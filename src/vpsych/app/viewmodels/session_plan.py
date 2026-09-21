"""Session builder state and `SessionPlan` construction.

`SessionBuilderState` is the mutable, non-Qt state the session-builder
screen edits (selected tests, per-test eye/viewing distance/params,
ordering, seed); `build_session_plan` turns it into an immutable
`vpsych.data.schemas.SessionPlan` ready to write to a JSON file the runner
subprocess reads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vpsych.data.schemas import Ordering, PlannedTest, SessionPlan


class SessionBuilderError(ValueError):
    """Raised when a `SessionBuilderState` cannot be turned into a valid `SessionPlan`."""


@dataclass
class PlannedTestState:
    """One planned test entry as edited by the session-builder screen.

    Attributes:
        task_id: The test's `TestSpec.id`.
        eye: Eye condition to test under.
        viewing_distance_cm: Viewing distance for this test, in cm.
        params: Test-specific parameters (raw dict; validated against the
            test's `params_model` only at plan-build time, by
            `SessionPlan`/`PlannedTest` themselves).
    """

    task_id: str
    eye: str = "OU"
    viewing_distance_cm: float = 60.0
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionBuilderState:
    """Mutable session-builder state, edited by the UI before a plan is built.

    Attributes:
        participant_id: The participant this session is for, or `None` if
            not yet chosen.
        tests: Planned tests, in the order they were added (the order used
            when `ordering == "fixed"`).
        ordering: `"fixed"` or `"randomized"`.
        seed: RNG seed, or `None` to draw a fresh one at build time.
        calibration_hash: Specific calibration to pin this plan to, or
            `None` to let the runner use the most recent calibration.
    """

    participant_id: str | None = None
    tests: list[PlannedTestState] = field(default_factory=list)
    ordering: Ordering = "fixed"
    seed: int | None = None
    calibration_hash: str | None = None

    def add_test(self, task_id: str, eye: str = "OU", viewing_distance_cm: float = 60.0) -> None:
        """Append a test to the plan (does not deduplicate; a test can be run more than once)."""
        self.tests.append(
            PlannedTestState(task_id=task_id, eye=eye, viewing_distance_cm=viewing_distance_cm)
        )

    def remove_test(self, index: int) -> None:
        """Remove the planned test at `index`."""
        del self.tests[index]

    def move_test(self, index: int, new_index: int) -> None:
        """Move the planned test at `index` to `new_index`, shifting others accordingly."""
        item = self.tests.pop(index)
        self.tests.insert(new_index, item)


def validate_builder_state(state: SessionBuilderState) -> list[str]:
    """Check a builder state for problems that would prevent building a valid plan.

    Args:
        state: The builder state to validate.

    Returns:
        Human-readable problem descriptions; empty means `build_session_plan`
        should succeed.
    """
    problems: list[str] = []
    if not state.participant_id:
        problems.append("Choose a participant before building a session.")
    if not state.tests:
        problems.append("Add at least one test to the session.")
    for i, t in enumerate(state.tests):
        if t.viewing_distance_cm <= 0:
            problems.append(f"Test {i + 1} ({t.task_id}): viewing distance must be positive.")
        if t.eye not in ("OD", "OS", "OU"):
            problems.append(f"Test {i + 1} ({t.task_id}): eye must be OD, OS, or OU.")
    return problems


def build_session_plan(state: SessionBuilderState, seed: int | None = None) -> SessionPlan:
    """Build an immutable `SessionPlan` from a builder state.

    Args:
        state: The builder state to convert.
        seed: RNG seed to use if `state.seed` is `None`; if both are `None`,
            `SessionPlan.seed` still requires an int, so a fresh seed is
            drawn from `vpsych.core.rng.make_rng`.

    Returns:
        The constructed `SessionPlan`.

    Raises:
        SessionBuilderError: If `validate_builder_state(state)` reports any
            problems.
    """
    problems = validate_builder_state(state)
    if problems:
        raise SessionBuilderError("; ".join(problems))

    resolved_seed = state.seed if state.seed is not None else seed
    if resolved_seed is None:
        from vpsych.core.rng import make_rng

        _, resolved_seed = make_rng(None)

    assert state.participant_id is not None  # guaranteed by validate_builder_state
    return SessionPlan(
        participant_id=state.participant_id,
        tests=[
            PlannedTest(
                task_id=t.task_id,
                eye=t.eye,
                params=t.params,
                viewing_distance_cm=t.viewing_distance_cm,
            )
            for t in state.tests
        ],
        ordering=state.ordering,
        seed=resolved_seed,
        calibration_hash=state.calibration_hash,
    )


def write_session_plan(plan: SessionPlan, path: Path) -> Path:
    """Write a `SessionPlan` to a JSON file for the runner subprocess to read.

    A plain `Path.write_text` (not the data layer's atomic-write helpers):
    a session-plan file is a transient hand-off to a subprocess the app is
    about to launch, not participant data under the data root, so it does
    not need the durability guarantees `vpsych.data.writer` provides for
    recorded results.

    Args:
        plan: The plan to write.
        path: Destination path.

    Returns:
        `path`, for chaining.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    return path


def estimated_total_minutes(state: SessionBuilderState, durations: dict[str, float]) -> float:
    """Sum expected durations for every planned test.

    Args:
        state: The builder state.
        durations: Map of `task_id` -> `TestSpec.estimated_minutes` (from
            the catalog), e.g. `{c.task_id: spec.estimated_minutes for ...}`.

    Returns:
        Total estimated minutes; a planned test with no entry in
        `durations` contributes 0 (never invents a duration).
    """
    return sum(durations.get(t.task_id, 0.0) for t in state.tests)
