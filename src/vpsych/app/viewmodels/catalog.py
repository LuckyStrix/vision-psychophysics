"""Test-catalog card view models: one per visible test, grouped by domain.

Built from `vpsych.tests_catalog.base.visible_tests()` and
`check_requirements`; a card's `enabled` is `False` and `unmet_reasons` is
non-empty whenever the active display/calibration doesn't satisfy the
test's declared `TestRequirements` -- the reasons are shown verbatim, never
paraphrased, per the project's "no invented values" rule.
"""

from __future__ import annotations

from dataclasses import dataclass

from vpsych.core.calibration.models import Calibration
from vpsych.core.display import DisplayGeometry
from vpsych.tests_catalog.base import PsychophysicalTest, check_requirements, visible_tests


@dataclass(frozen=True)
class TestCardViewModel:
    """Everything one test-catalog card needs to render.

    Attributes:
        task_id: The test's `TestSpec.id`.
        name: Display name.
        domain: Perceptual domain (`"acuity"`, `"contrast"`, ...).
        measures: Short description of what this test measures.
        description_participant: Plain-language participant-facing description.
        output_units: Units of the primary threshold estimate.
        duration_text: Human-readable expected duration, e.g. `"~3 min"`.
        allowed_eyes: Eye conditions this test supports.
        citations: Literature citations.
        enabled: Whether this test can currently run "research-grade".
        unmet_reasons: Human-readable reasons it cannot, verbatim from
            `check_requirements`; empty when `enabled`.
    """

    task_id: str
    name: str
    domain: str
    measures: str
    description_participant: str
    output_units: str
    duration_text: str
    allowed_eyes: list[str]
    citations: list[str]
    enabled: bool
    unmet_reasons: list[str]


def _duration_text(estimated_minutes: float) -> str:
    if estimated_minutes < 1:
        seconds = round(estimated_minutes * 60)
        return f"~{seconds} sec"
    if float(estimated_minutes).is_integer():
        return f"~{int(estimated_minutes)} min"
    return f"~{estimated_minutes:g} min"


def build_test_card(
    test_cls: type[PsychophysicalTest],
    display: DisplayGeometry,
    calibration: Calibration | None,
) -> TestCardViewModel:
    """Build one test's card view model.

    Args:
        test_cls: The registered test class (see `visible_tests()`).
        display: Active display geometry to check requirements against.
        calibration: Active calibration, or `None` if uncalibrated.

    Returns:
        The constructed `TestCardViewModel`.
    """
    spec = test_cls.spec
    reasons = check_requirements(spec.requirements, display, calibration)
    return TestCardViewModel(
        task_id=spec.id,
        name=spec.name,
        domain=spec.domain,
        measures=spec.measures,
        description_participant=spec.description_participant,
        output_units=spec.output_units,
        duration_text=_duration_text(spec.estimated_minutes),
        allowed_eyes=list(spec.allowed_eyes),
        citations=list(spec.citations),
        enabled=not reasons,
        unmet_reasons=reasons,
    )


def build_test_cards(
    display: DisplayGeometry,
    calibration: Calibration | None,
    tests: list[type[PsychophysicalTest]] | None = None,
) -> list[TestCardViewModel]:
    """Build card view models for every visible test.

    Args:
        display: Active display geometry.
        calibration: Active calibration, or `None`.
        tests: Test classes to build cards for, or `None` to use
            `visible_tests()`.

    Returns:
        One card per test, in catalog order.
    """
    candidates = tests if tests is not None else visible_tests()
    return [build_test_card(cls, display, calibration) for cls in candidates]


_DOMAIN_ORDER = ["acuity", "contrast", "color", "motion", "temporal", "hyperacuity"]

_DOMAIN_LABELS = {
    "acuity": "Acuity",
    "contrast": "Contrast sensitivity",
    "color": "Color vision",
    "motion": "Motion perception",
    "temporal": "Temporal vision",
    "hyperacuity": "Hyperacuity",
}


def domain_label(domain: str) -> str:
    """Human-readable label for a domain key, falling back to a title-cased version."""
    return _DOMAIN_LABELS.get(domain, domain.replace("_", " ").title())


def group_by_domain(cards: list[TestCardViewModel]) -> list[tuple[str, list[TestCardViewModel]]]:
    """Group cards by domain, in a fixed, stable domain order.

    Args:
        cards: Cards to group, e.g. from `build_test_cards`.

    Returns:
        `(domain_label, cards)` pairs, domains in `_DOMAIN_ORDER` first (any
        unrecognized domain appended after, alphabetically), cards within a
        domain kept in their input order.
    """
    by_domain: dict[str, list[TestCardViewModel]] = {}
    for card in cards:
        by_domain.setdefault(card.domain, []).append(card)

    ordered_keys = [d for d in _DOMAIN_ORDER if d in by_domain]
    ordered_keys += sorted(d for d in by_domain if d not in _DOMAIN_ORDER)
    return [(domain_label(d), by_domain[d]) for d in ordered_keys]
