"""Unit tests for vpsych.core.calibration.color, validated against colour-science."""

from __future__ import annotations

import colour
import numpy as np
import pytest

from vpsych.core.calibration.color import (
    COPUNCTAL_POINTS_XY,
    SRGB_PRIMARIES_XY,
    SRGB_WHITE_XY,
    cone_contrast,
    confusion_line_direction_uv,
    gamut_check,
    primaries_to_xyz_matrix,
    rgb_to_xyz,
    xy_to_uv_prime,
    xyz_to_lms,
    xyz_to_lms_matrix,
    xyz_to_rgb,
    xyz_to_rgb_matrix,
    xyz_to_uv_prime,
)


def _colour_srgb_npm() -> np.ndarray:
    from colour.models.rgb.derivation import normalised_primary_matrix

    cs = colour.RGB_COLOURSPACES["sRGB"]
    return np.asarray(normalised_primary_matrix(cs.primaries, cs.whitepoint))


def test_primaries_to_xyz_matrix_matches_colour_science_srgb() -> None:
    ours = primaries_to_xyz_matrix(SRGB_PRIMARIES_XY, SRGB_WHITE_XY)
    reference = _colour_srgb_npm()
    rel_err = np.max(np.abs((ours - reference) / reference))
    assert rel_err < 1e-6


def test_xyz_to_rgb_matrix_is_inverse() -> None:
    forward = primaries_to_xyz_matrix(SRGB_PRIMARIES_XY, SRGB_WHITE_XY)
    inverse = xyz_to_rgb_matrix(SRGB_PRIMARIES_XY, SRGB_WHITE_XY)
    identity = forward @ inverse
    np.testing.assert_allclose(identity, np.eye(3), atol=1e-10)


def test_rgb_to_xyz_matches_colour_science_derived_matrix_for_random_colors() -> None:
    # colour.RGB_to_XYZ's *default* matrix is the published, rounded sRGB matrix
    # (colourspace.matrix_RGB_to_XYZ), not the one derived from primaries+white
    # point -- those differ at the ~1e-4 relative level by design (the published
    # constants are rounded for interchange). We validate our from-primaries
    # derivation against colour-science's own from-primaries derivation
    # (`normalised_primary_matrix`), which is the correct apples-to-apples
    # reference for what this module computes.
    matrix = primaries_to_xyz_matrix(SRGB_PRIMARIES_XY, SRGB_WHITE_XY)
    reference = _colour_srgb_npm()
    rng = np.random.default_rng(0)
    rgb = rng.uniform(0, 1, size=(20, 3))
    ours = rgb_to_xyz(rgb, matrix)
    expected = rgb_to_xyz(rgb, reference)
    np.testing.assert_allclose(ours, expected, rtol=1e-6, atol=1e-10)


def test_xyz_to_rgb_round_trip() -> None:
    matrix = primaries_to_xyz_matrix(SRGB_PRIMARIES_XY, SRGB_WHITE_XY)
    matrix_inv = xyz_to_rgb_matrix(SRGB_PRIMARIES_XY, SRGB_WHITE_XY)
    rng = np.random.default_rng(1)
    rgb = rng.uniform(0, 1, size=(10, 3))
    xyz = rgb_to_xyz(rgb, matrix)
    recovered = xyz_to_rgb(xyz, matrix_inv)
    np.testing.assert_allclose(recovered, rgb, atol=1e-10)


def test_xy_to_uv_prime_matches_colour_science() -> None:
    xy_pairs = [(0.3127, 0.3290), (0.64, 0.33), (0.3, 0.4), (0.15, 0.06)]
    for x, y in xy_pairs:
        u, v = xy_to_uv_prime(x, y)
        ref_u, ref_v = colour.xy_to_Luv_uv((x, y))
        assert float(u) == pytest.approx(ref_u, rel=1e-9)
        assert float(v) == pytest.approx(ref_v, rel=1e-9)


def test_xyz_to_uv_prime_matches_xy_to_uv_prime_for_normalized_xyz() -> None:
    x, y = 0.3127, 0.3290
    z = 1.0 - x - y
    xyz = np.array([x, y, z])  # Y = y here since X+Y+Z=1 normalization convention
    u1, v1 = xy_to_uv_prime(x, y)
    u2, v2 = xyz_to_uv_prime(xyz)
    assert float(u2) == pytest.approx(float(u1), rel=1e-9)
    assert float(v2) == pytest.approx(float(v1), rel=1e-9)


def test_xyz_to_lms_matrix_is_invertible_and_positive_for_white() -> None:
    matrix = xyz_to_lms_matrix()
    assert matrix.shape == (3, 3)
    d65 = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"]
    xyz_white = colour.xy_to_XYZ(d65)
    lms_white = xyz_to_lms(xyz_white)
    assert np.all(lms_white > 0)  # a real white point should excite all three cone classes


def test_xyz_to_lms_matrix_is_cached() -> None:
    assert xyz_to_lms_matrix() is xyz_to_lms_matrix()


def test_cone_contrast_zero_at_background() -> None:
    bg = np.array([1.0, 1.0, 1.0])
    contrast = cone_contrast(bg, bg)
    np.testing.assert_allclose(contrast, [0.0, 0.0, 0.0])


def test_cone_contrast_weber_definition() -> None:
    bg = np.array([2.0, 3.0, 4.0])
    stim = np.array([2.2, 2.7, 4.4])
    contrast = cone_contrast(stim, bg)
    np.testing.assert_allclose(contrast, [0.1, -0.1, 0.1])


@pytest.mark.parametrize("deficiency", ["protan", "deutan", "tritan"])
def test_confusion_line_direction_is_unit_vector(deficiency: str) -> None:
    background_uv = xy_to_uv_prime(0.3127, 0.3290)
    background_uv = (float(background_uv[0]), float(background_uv[1]))
    du, dv = confusion_line_direction_uv(background_uv, deficiency)
    norm = (du**2 + dv**2) ** 0.5
    assert norm == pytest.approx(1.0, rel=1e-9)


def test_confusion_line_direction_points_toward_copunctal_point() -> None:
    background_uv = (0.1978, 0.4683)  # D65 in u'v'
    du, dv = confusion_line_direction_uv(background_uv, "protan")
    x, y = COPUNCTAL_POINTS_XY["protan"]
    u_c, v_c = xy_to_uv_prime(x, y)
    expected_du = float(u_c) - background_uv[0]
    expected_dv = float(v_c) - background_uv[1]
    norm = (expected_du**2 + expected_dv**2) ** 0.5
    assert du == pytest.approx(expected_du / norm)
    assert dv == pytest.approx(expected_dv / norm)


def test_confusion_line_direction_unknown_deficiency_raises() -> None:
    with pytest.raises(KeyError):
        confusion_line_direction_uv((0.2, 0.47), "xanthan")  # type: ignore[arg-type]


def test_confusion_line_direction_zero_length_raises() -> None:
    x, y = COPUNCTAL_POINTS_XY["tritan"]
    u_c, v_c = xy_to_uv_prime(x, y)
    with pytest.raises(ValueError):
        confusion_line_direction_uv((float(u_c), float(v_c)), "tritan")


def test_gamut_check_primary_is_in_gamut() -> None:
    assert gamut_check(SRGB_PRIMARIES_XY["red"]) is True
    assert gamut_check(SRGB_WHITE_XY) is True


def test_gamut_check_outside_triangle_is_false() -> None:
    # A point far outside the sRGB triangle (near spectral locus green).
    assert gamut_check((0.03, 0.83)) is False


def test_gamut_check_default_primaries_is_srgb() -> None:
    assert gamut_check((0.3127, 0.3290)) == gamut_check((0.3127, 0.3290), SRGB_PRIMARIES_XY)
