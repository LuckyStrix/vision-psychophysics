"""Tests for `vpsych.app.viewmodels.session_plan`."""

from __future__ import annotations

import json

import pytest

from vpsych.app.viewmodels.session_plan import (
    PlannedTestState,
    SessionBuilderError,
    SessionBuilderState,
    build_session_plan,
    estimated_total_minutes,
    validate_builder_state,
    write_session_plan,
)


def test_validate_builder_state_reports_missing_participant_and_tests() -> None:
    state = SessionBuilderState()
    problems = validate_builder_state(state)
    assert any("participant" in p.lower() for p in problems)
    assert any("test" in p.lower() for p in problems)


def test_validate_builder_state_catches_bad_eye_and_distance() -> None:
    state = SessionBuilderState(participant_id="sub-0001")
    state.tests.append(PlannedTestState(task_id="x", eye="ZZ", viewing_distance_cm=-1))
    problems = validate_builder_state(state)
    assert any("viewing distance" in p for p in problems)
    assert any("eye" in p for p in problems)


def test_build_session_plan_success() -> None:
    state = SessionBuilderState(participant_id="sub-0001", ordering="randomized")
    state.add_test("visual_acuity", eye="OD", viewing_distance_cm=100.0)
    state.add_test("contrast_sensitivity_function", eye="OU", viewing_distance_cm=57.0)
    plan = build_session_plan(state, seed=123)
    assert plan.participant_id == "sub-0001"
    assert plan.ordering == "randomized"
    assert plan.seed == 123
    assert [t.task_id for t in plan.tests] == [
        "visual_acuity",
        "contrast_sensitivity_function",
    ]
    assert plan.tests[0].eye == "OD"
    assert plan.tests[0].viewing_distance_cm == 100.0


def test_build_session_plan_draws_fresh_seed_when_none_given() -> None:
    state = SessionBuilderState(participant_id="sub-0001")
    state.add_test("visual_acuity")
    plan = build_session_plan(state)
    assert isinstance(plan.seed, int)


def test_build_session_plan_raises_on_invalid_state() -> None:
    state = SessionBuilderState()
    with pytest.raises(SessionBuilderError):
        build_session_plan(state)


def test_remove_and_move_test() -> None:
    state = SessionBuilderState(participant_id="sub-0001")
    state.add_test("a")
    state.add_test("b")
    state.add_test("c")
    state.move_test(0, 2)
    assert [t.task_id for t in state.tests] == ["b", "c", "a"]
    state.remove_test(1)
    assert [t.task_id for t in state.tests] == ["b", "a"]


def test_estimated_total_minutes_missing_entries_contribute_zero() -> None:
    state = SessionBuilderState(participant_id="sub-0001")
    state.add_test("known")
    state.add_test("unknown")
    total = estimated_total_minutes(state, {"known": 3.5})
    assert total == 3.5


def test_write_session_plan_round_trips(tmp_path) -> None:
    state = SessionBuilderState(participant_id="sub-0001")
    state.add_test("visual_acuity")
    plan = build_session_plan(state, seed=7)
    path = tmp_path / "plans" / "plan.json"
    write_session_plan(plan, path)
    assert path.exists()
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["participant_id"] == "sub-0001"
    assert raw["seed"] == 7
