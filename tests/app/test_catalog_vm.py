"""Tests for `vpsych.app.viewmodels.catalog`."""

from __future__ import annotations

from tests.data.conftest import make_calibration, make_display
from vpsych.app.viewmodels.catalog import build_test_card, build_test_cards, group_by_domain
from vpsych.tests_catalog.base import TestRequirements, TestSpec


def test_build_test_card_enabled_when_requirements_met(registered_dummy_test) -> None:
    card = build_test_card(registered_dummy_test, make_display(), make_calibration())
    assert card.task_id == "dummy_test"
    assert card.enabled is True
    assert card.unmet_reasons == []
    assert card.duration_text == "~1 min"


def test_build_test_card_disabled_states_reason_verbatim(registered_dummy_test) -> None:
    original_spec = registered_dummy_test.spec
    try:
        registered_dummy_test.spec = TestSpec(
            id="dummy_test",
            name="Dummy Test",
            version="1.0.0",
            domain="acuity",
            description_participant="n/a",
            description_technical="n/a",
            measures="n/a",
            output_units="logMAR",
            estimated_minutes=2.0,
            allowed_eyes=["OU"],
            requirements=TestRequirements(needs_gamma_calibration=True),
            params_model=original_spec.params_model,
        )
        card = build_test_card(registered_dummy_test, make_display(), None)
        assert card.enabled is False
        assert card.unmet_reasons == ["Needs a gamma calibration (currently uncalibrated)."]
    finally:
        registered_dummy_test.spec = original_spec


def test_build_test_cards_uses_visible_tests_by_default(registered_dummy_test) -> None:
    cards = build_test_cards(make_display(), make_calibration(), tests=[registered_dummy_test])
    assert [c.task_id for c in cards] == ["dummy_test"]


def test_group_by_domain_orders_known_domains_first(registered_dummy_test) -> None:
    cards = build_test_cards(make_display(), make_calibration(), tests=[registered_dummy_test])
    grouped = group_by_domain(cards)
    labels = [label for label, _ in grouped]
    assert labels == ["Acuity"]
    assert grouped[0][1][0].task_id == "dummy_test"


def test_duration_text_formats_sub_minute_and_whole_minute() -> None:
    from vpsych.app.viewmodels.catalog import _duration_text

    assert _duration_text(0.5) == "~30 sec"
    assert _duration_text(3.0) == "~3 min"
    assert _duration_text(2.5) == "~2.5 min"
