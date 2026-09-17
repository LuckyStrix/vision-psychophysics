"""Unit tests for vpsych.tests_catalog.base: PsychophysicalTest.simulated_response.

Uses a minimal dummy PsychophysicalTest (no display, no real procedure) to
exercise the default `simulated_response` hook's correct/incorrect ->
response mapping, independent of any real test implementation or the trial
loop.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel

from vpsych.core.procedures.base import AdaptiveProcedure
from vpsych.tests_catalog import base as catalog_base
from vpsych.tests_catalog.base import (
    PresentedTrial,
    PsychophysicalTest,
    TestRequirements,
    TestSpec,
    discover_tests,
    visible_tests,
)


class _Params(BaseModel):
    pass


class _MinimalTest(PsychophysicalTest):
    spec = TestSpec(
        id="minimal_simresp_test",
        name="Minimal simulated_response Test",
        version="0.1.0",
        domain="contrast",
        description_participant="n/a",
        description_technical="Minimal test exercising simulated_response's default.",
        measures="Nothing real.",
        output_units="log10_contrast",
        estimated_minutes=1.0,
        allowed_eyes=["OU"],
        requirements=TestRequirements(),
        citations=[],
        params_model=_Params,
    )

    def __init__(
        self, params: BaseModel, display: Any, calibration: Any, rng: np.random.Generator
    ) -> None:
        self.params = params

    def make_procedure(self) -> AdaptiveProcedure:
        raise NotImplementedError

    def make_catch_trial_intensity(self) -> float:
        return 0.0

    def build_stimuli(self, win: Any) -> dict[str, Any]:
        return {}

    def present(
        self, win: Any, intensity_or_stimulus: Any, trial_ctx: dict[str, Any]
    ) -> PresentedTrial:
        raise NotImplementedError

    def response_keys(self) -> list[str]:
        return ["left", "right"]

    def score(self, response: Any, stimulus_params: dict[str, Any]) -> bool:
        return bool(response == stimulus_params["correct_response"])

    def instructions(self) -> str:
        return ""

    def summarize(self, trials: Any) -> Any:
        raise NotImplementedError


def _test() -> _MinimalTest:
    return _MinimalTest(
        params=_Params(), display=None, calibration=None, rng=np.random.default_rng(0)
    )


def test_simulated_response_correct_echoes_correct_response() -> None:
    t = _test()
    rng = np.random.default_rng(0)
    response = t.simulated_response(True, {"correct_response": "left"}, rng)
    assert response == "left"


def test_simulated_response_incorrect_picks_other_response_key() -> None:
    t = _test()
    rng = np.random.default_rng(0)
    response = t.simulated_response(False, {"correct_response": "left"}, rng)
    assert response == "right"


def test_simulated_response_incorrect_never_equals_correct_response() -> None:
    t = _test()
    rng = np.random.default_rng(1)
    for _ in range(50):
        response = t.simulated_response(False, {"correct_response": "left"}, rng)
        assert response != "left"
        assert response in t.response_keys()


def test_simulated_response_round_trip_scores_as_expected() -> None:
    """The whole point: score(simulated_response(correct, ...), ...) == correct."""
    t = _test()
    rng = np.random.default_rng(2)
    stim = {"correct_response": "left"}
    for correct in (True, False, True, False):
        response = t.simulated_response(correct, stim, rng)
        assert t.score(response, stim) == correct


def test_simulated_response_falls_back_when_no_other_keys() -> None:
    class _SingleKeyTest(_MinimalTest):
        def response_keys(self) -> list[str]:
            return ["only"]

    t = _SingleKeyTest(
        params=_Params(), display=None, calibration=None, rng=np.random.default_rng(0)
    )
    rng = np.random.default_rng(0)
    # No alternative to "only" exists, so even an "incorrect" trial falls back
    # to echoing correct_response (documented last-resort behavior).
    response = t.simulated_response(False, {"correct_response": "only"}, rng)
    assert response == "only"


def test_simulated_response_missing_correct_response_key_returns_none() -> None:
    t = _test()
    rng = np.random.default_rng(0)
    assert t.simulated_response(True, {}, rng) is None


def test_hidden_defaults_false() -> None:
    assert _MinimalTest.spec.hidden is False


def test_visible_tests_excludes_hidden() -> None:
    class _HiddenTest(_MinimalTest):
        spec = _MinimalTest.spec.model_copy(update={"id": "minimal_hidden_test", "hidden": True})

    catalog_base._REGISTRY["minimal_hidden_test"] = _HiddenTest
    try:
        visible_ids = {cls.spec.id for cls in visible_tests()}
        all_ids = {cls.spec.id for cls in catalog_base.all_tests()}
        assert "minimal_hidden_test" in all_ids
        assert "minimal_hidden_test" not in visible_ids
    finally:
        catalog_base._REGISTRY.pop("minimal_hidden_test", None)


def test_discover_tests_is_idempotent_and_returns_all_tests() -> None:
    before = discover_tests()
    after = discover_tests()
    assert before == after
    assert before == catalog_base.all_tests()
