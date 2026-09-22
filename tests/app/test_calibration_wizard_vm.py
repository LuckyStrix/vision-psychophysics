"""Tests for `vpsych.app.viewmodels.calibration_wizard`."""

from __future__ import annotations

import pytest

from vpsych.app.viewmodels.calibration_wizard import (
    DEFAULT_BISECTION_FRACTIONS,
    BisectionSequenceState,
    CalibrationWizardState,
    ColorStepState,
    GammaStepState,
    GeometryStepState,
    WizardStepError,
    apply_calsuite_import,
    environment_checklist_complete,
    environment_checklist_warnings,
)
from vpsych.core.calibration.calsuite_import import CalsuiteImportResult
from vpsych.core.calibration.geometry import estimate_display_width_cm
from vpsych.core.calibration.models import (
    ColorCalibration,
    EnvironmentChecklist,
    GammaCalibration,
    PrimaryChromaticity,
)


def test_geometry_step_build_requires_all_fields() -> None:
    step = GeometryStepState()
    with pytest.raises(WizardStepError):
        step.build()


def test_geometry_step_resolve_width_cm_from_card_match() -> None:
    step = GeometryStepState(width_px=1920, card_width_px=300.0)
    expected = estimate_display_width_cm(300.0, 1920)
    assert step.resolve_width_cm() == pytest.approx(expected)


def test_geometry_step_prefers_direct_width_cm_when_no_card_match() -> None:
    step = GeometryStepState(width_px=1920, width_cm=53.0)
    assert step.resolve_width_cm() == 53.0


def test_geometry_step_build_success() -> None:
    step = GeometryStepState(
        width_px=1920,
        height_px=1080,
        refresh_hz=60.0,
        width_cm=53.0,
        height_cm=30.0,
        viewing_distance_cm=60.0,
    )
    geometry = step.build()
    assert geometry.width_px == 1920
    assert geometry.refresh_hz == 60.0


def test_geometry_step_locked_width_cm_overrides_card_match() -> None:
    step = GeometryStepState(
        width_px=1920, card_width_px=300.0, width_cm=34.4, width_cm_locked=True
    )
    assert step.resolve_width_cm() == 34.4


def test_geometry_step_unlocked_width_cm_falls_back_to_card_match() -> None:
    step = GeometryStepState(
        width_px=1920, card_width_px=300.0, width_cm=34.4, width_cm_locked=False
    )
    assert step.resolve_width_cm() == pytest.approx(estimate_display_width_cm(300.0, 1920))


def test_gamma_step_none_mode_gives_grade_c() -> None:
    gamma = GammaStepState(mode="none").build()
    assert gamma.method == "none"
    assert gamma.grade == "C"


def test_gamma_step_photometer_mode_needs_two_points() -> None:
    step = GammaStepState(mode="photometer")
    with pytest.raises(WizardStepError):
        step.build()


def test_gamma_step_photometer_mode_fits_from_points() -> None:
    from vpsych.core.calibration.models import GammaCalibrationPoint

    step = GammaStepState(
        mode="photometer",
        photometer_points=[
            GammaCalibrationPoint(input_level=0.0, luminance_cdm2=0.5),
            GammaCalibrationPoint(input_level=0.5, luminance_cdm2=40.0),
            GammaCalibrationPoint(input_level=1.0, luminance_cdm2=150.0),
        ],
    )
    gamma = step.build()
    assert gamma.method == "photometer"
    assert gamma.grade == "A"
    assert gamma.gamma_single is not None


def test_gamma_step_psychophysical_mode_needs_two_matches() -> None:
    step = GammaStepState(mode="psychophysical")
    with pytest.raises(WizardStepError):
        step.build()


def test_gamma_step_psychophysical_mode_fits_from_matches() -> None:
    step = GammaStepState(
        mode="psychophysical", psychophysical_matches=[(0.2, 0.45), (0.5, 0.7), (0.8, 0.89)]
    )
    gamma = step.build()
    assert gamma.method == "psychophysical"
    assert gamma.grade == "B"


def test_gamma_step_external_mode_uses_value_as_is() -> None:
    from vpsych.core.calibration.models import GammaCalibration

    imported = GammaCalibration(
        method="photometer",
        gamma_r=2.2,
        gamma_g=2.19,
        gamma_b=2.21,
        lum_min_cdm2=0.3,
        lum_max_cdm2=400.0,
    )
    step = GammaStepState(mode="external", external=imported)
    assert step.build() is imported


def test_gamma_step_external_mode_without_value_raises() -> None:
    step = GammaStepState(mode="external")
    with pytest.raises(WizardStepError):
        step.build()


def test_color_step_assumed_gives_grade_c() -> None:
    color = ColorStepState(measured=False).build()
    assert color.method == "srgb_assumed"
    assert color.grade == "C"


def test_color_step_measured_requires_all_primaries() -> None:
    step = ColorStepState(measured=True)
    with pytest.raises(WizardStepError):
        step.build()


def test_color_step_measured_success() -> None:
    primary = PrimaryChromaticity(x=0.64, y=0.33, Y_cdm2=30)
    step = ColorStepState(measured=True, red=primary, green=primary, blue=primary, white=primary)
    color = step.build()
    assert color.method == "measured"
    assert color.grade == "A"


def test_environment_checklist_complete_all_checked() -> None:
    checklist = EnvironmentChecklist(
        room_lighting_controlled=True,
        monitor_warmed_up=True,
        night_light_disabled=True,
        hdr_disabled=True,
    )
    assert environment_checklist_complete(checklist) is True
    assert environment_checklist_warnings(checklist) == []


def test_environment_checklist_incomplete_reports_each_unmet_item() -> None:
    checklist = EnvironmentChecklist(
        room_lighting_controlled=False,
        monitor_warmed_up=True,
        night_light_disabled=False,
        hdr_disabled=True,
    )
    assert environment_checklist_complete(checklist) is False
    warnings = environment_checklist_warnings(checklist)
    assert len(warnings) == 2


def test_environment_checklist_none_is_incomplete_with_all_warnings() -> None:
    assert environment_checklist_complete(None) is False
    assert len(environment_checklist_warnings(None)) == 4


def test_build_calibration_requires_environment_checklist() -> None:
    state = CalibrationWizardState(
        geometry=GeometryStepState(
            width_px=1920,
            height_px=1080,
            refresh_hz=60.0,
            width_cm=53.0,
            height_cm=30.0,
            viewing_distance_cm=60.0,
        )
    )
    with pytest.raises(WizardStepError):
        state.build_calibration(software_version="0.1.0")


def test_build_calibration_full_success() -> None:
    state = CalibrationWizardState(
        geometry=GeometryStepState(
            width_px=1920,
            height_px=1080,
            refresh_hz=60.0,
            width_cm=53.0,
            height_cm=30.0,
            viewing_distance_cm=60.0,
        ),
        environment=EnvironmentChecklist(
            room_lighting_controlled=True,
            monitor_warmed_up=True,
            night_light_disabled=True,
            hdr_disabled=True,
        ),
    )
    calibration = state.build_calibration(software_version="0.1.0")
    assert calibration.luminance_grade == "C"
    assert calibration.color_grade == "C"
    assert calibration.software_version == "0.1.0"
    # Content-addressed: building the identical state twice at the same
    # instant gives the same hash.
    calibration2 = state.build_calibration(software_version="0.1.0", now=calibration.created_utc)
    assert calibration.content_hash() == calibration2.content_hash()


# ---------------------------------------------------------------------------
# BisectionSequenceState (psychophysical gamma UI)
# ---------------------------------------------------------------------------


def test_bisection_sequence_starts_at_first_fraction() -> None:
    seq = BisectionSequenceState()
    assert seq.current_fraction == DEFAULT_BISECTION_FRACTIONS[0]
    assert seq.is_complete is False
    assert seq.progress_text == f"Level 1 of {len(DEFAULT_BISECTION_FRACTIONS)}."


def test_bisection_sequence_adjust_clamps_to_open_interval() -> None:
    seq = BisectionSequenceState(current_level=0.5)
    seq.adjust(-10.0)
    assert seq.current_level == pytest.approx(0.005)
    seq.adjust(10.0)
    assert seq.current_level == pytest.approx(0.995)


def test_bisection_sequence_confirm_match_advances_and_resets_level() -> None:
    seq = BisectionSequenceState(fractions=(0.2, 0.8))
    seq.adjust(0.1)  # 0.6
    seq.confirm_match()
    assert seq.matches == [(0.2, pytest.approx(0.6))]
    assert seq.current_level == 0.5
    assert seq.current_fraction == 0.8
    assert seq.is_complete is False

    seq.confirm_match()
    assert seq.is_complete is True
    assert seq.current_fraction is None
    assert len(seq.matches) == 2


def test_bisection_sequence_confirm_match_after_complete_raises() -> None:
    seq = BisectionSequenceState(fractions=(0.5,))
    seq.confirm_match()
    with pytest.raises(WizardStepError):
        seq.confirm_match()


def test_bisection_sequence_result_needs_two_matches() -> None:
    seq = BisectionSequenceState(fractions=(0.5,))
    seq.confirm_match()
    with pytest.raises(WizardStepError):
        seq.result()


def test_bisection_sequence_result_recovers_a_known_gamma() -> None:
    gamma = 2.2
    seq = BisectionSequenceState(fractions=(0.2, 0.4, 0.6, 0.8))
    for p in seq.fractions:
        seq.current_level = p ** (1.0 / gamma)
        seq.confirm_match()
    estimate = seq.result()
    assert estimate.gamma == pytest.approx(gamma, rel=1e-6)
    assert estimate.n_levels == 4


# ---------------------------------------------------------------------------
# apply_calsuite_import
# ---------------------------------------------------------------------------


def _sample_gamma() -> GammaCalibration:
    return GammaCalibration(
        method="photometer",
        gamma_r=2.2,
        gamma_g=2.19,
        gamma_b=2.21,
        lum_min_cdm2=0.3,
        lum_max_cdm2=400.0,
    )


def _sample_color() -> ColorCalibration:
    primary = PrimaryChromaticity(x=0.64, y=0.33, Y_cdm2=90.0)
    return ColorCalibration(
        method="measured", red=primary, green=primary, blue=primary, white=primary
    )


def test_apply_calsuite_import_full_result_sets_every_field() -> None:
    state = CalibrationWizardState()
    result = CalsuiteImportResult(
        width_cm=34.4,
        height_cm=21.5,
        gamma=_sample_gamma(),
        color=_sample_color(),
        notes="imported from calsuite",
    )
    apply_calsuite_import(state, result)

    assert state.geometry.width_cm == 34.4
    assert state.geometry.width_cm_locked is True
    assert state.geometry.height_cm == 21.5
    assert state.gamma.mode == "external"
    assert state.gamma.external is result.gamma
    assert state.color.measured is True
    assert state.color.red is result.color.red
    assert state.notes == "imported from calsuite"


def test_apply_calsuite_import_partial_result_leaves_other_fields_untouched() -> None:
    state = CalibrationWizardState()
    state.geometry.width_px = 1920
    result = CalsuiteImportResult(width_cm=34.4, height_cm=21.5)
    apply_calsuite_import(state, result)

    assert state.geometry.width_cm == 34.4
    assert state.geometry.width_px == 1920  # untouched
    assert state.gamma.mode == "none"  # untouched
    assert state.color.measured is False  # untouched


def test_apply_calsuite_import_appends_to_existing_notes() -> None:
    state = CalibrationWizardState(notes="pre-existing note")
    result = CalsuiteImportResult(gamma=_sample_gamma(), notes="from calsuite")
    apply_calsuite_import(state, result)
    assert state.notes == "pre-existing note\nfrom calsuite"


def test_apply_calsuite_import_then_build_calibration_succeeds() -> None:
    state = CalibrationWizardState(
        geometry=GeometryStepState(
            width_px=1920, height_px=1080, refresh_hz=60.0, viewing_distance_cm=60.0
        ),
        environment=EnvironmentChecklist(
            room_lighting_controlled=True,
            monitor_warmed_up=True,
            night_light_disabled=True,
            hdr_disabled=True,
        ),
    )
    result = CalsuiteImportResult(
        width_cm=34.4, height_cm=21.5, gamma=_sample_gamma(), color=_sample_color()
    )
    apply_calsuite_import(state, result)

    calibration = state.build_calibration(software_version="0.1.0")
    assert calibration.luminance_grade == "A"
    assert calibration.color_grade == "A"
    assert calibration.geometry.width_cm == 34.4
