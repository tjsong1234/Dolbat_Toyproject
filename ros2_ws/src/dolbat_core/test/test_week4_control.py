"""Geometric checks independent of the recorded path's visual appearance."""
import math

import numpy as np
import pytest

from dolbat_core.week4_control import (
    PurePursuitConfig, bev_cm_to_rear_m, calculate_pure_pursuit,
)


def test_straight_path_target_is_circle_intersection():
    result = calculate_pure_pursuit([[.9, 0], [3.0, 0]])
    assert result.valid and result.status == 'OK'
    assert result.target_m == pytest.approx((1.7, 0), abs=1e-12)
    assert result.steering_deg == pytest.approx(0)
    assert result.segment_fraction == pytest.approx((1.7 - .9) / (3.0 - .9))
    assert result.circle_residual_m < 1e-12


def test_lateral_path_uses_circle_not_fixed_forward_x():
    result = calculate_pure_pursuit([[.4, .8], [3.0, .8]])
    expected_x = math.sqrt(1.7 ** 2 - .8 ** 2)
    assert result.target_m == pytest.approx((expected_x, .8), abs=1e-12)
    expected_curvature = 2 * .8 / 1.7 ** 2
    assert result.curvature_per_m == pytest.approx(expected_curvature)
    assert result.steering_rad == pytest.approx(math.atan(.55 * expected_curvature))
    assert result.steering_deg == pytest.approx(math.degrees(result.steering_rad))
    assert result.heading_error_rad == pytest.approx(math.atan2(.8, expected_x))


def test_mirrored_observations_give_opposite_angles():
    left = calculate_pure_pursuit([[.2, .3], [3., 1.]])
    right = calculate_pure_pursuit([[.2, -.3], [3., -1.]])
    assert left.steering_deg > 0 > right.steering_deg
    assert left.steering_deg == pytest.approx(-right.steering_deg)
    assert left.target_m[0] == pytest.approx(right.target_m[0])
    assert left.target_m[1] == pytest.approx(-right.target_m[1])


def test_multiple_crossings_select_first_in_forward_path_order():
    path = np.array([[.1, 0], [.3, 2], [.6, 0], [.9, 2], [1.2, 0], [2., 0]])
    result = calculate_pure_pursuit(path)
    assert result.valid and result.candidate_count == 5
    assert result.segment_index == 0
    assert .1 < result.target_m[0] < .3
    assert np.linalg.norm(result.target_m) == pytest.approx(1.7)


def test_shared_vertex_is_one_intersection():
    result = calculate_pure_pursuit([[0, 0], [1.7, 0], [3, 0]])
    assert result.candidate_count == 1
    assert result.segment_index == 0 and result.segment_fraction == pytest.approx(1)


def test_tangent_forward_target_is_counted_once():
    result = calculate_pure_pursuit([[1.7, -1.], [1.7, 1.]])
    assert result.valid and result.candidate_count == 1
    assert result.target_m == pytest.approx((1.7, 0))


@pytest.mark.parametrize('path,status', [
    (None, 'NO_PATH'), ([], 'INVALID_PATH'), ([[1., 0]], 'NO_PATH'),
    ([[1., 0], [1., 0]], 'DEGENERATE_PATH'),
    ([[.9, 0], [1., 0]], 'NO_FORWARD_INTERSECTION'),
    ([[2., 0], [3., 0]], 'NO_FORWARD_INTERSECTION'),
    ([[-3., 0], [-2., 0]], 'NO_FORWARD_INTERSECTION'),
    ([[0., 3], [3., 3]], 'NO_FORWARD_INTERSECTION'),
    ([[1., 0], [.5, 0]], 'INVALID_PATH'),
    ([[0., 0], [float('nan'), 0]], 'INVALID_PATH'),
    ([[0., 0], [2., float('inf')]], 'INVALID_PATH'),
    ([[0., 0, 0], [2., 0, 0]], 'INVALID_PATH'),
])
def test_failures_have_no_fabricated_target_or_steering(path, status):
    result = calculate_pure_pursuit(path)
    assert not result.valid and result.status == status
    assert result.target_m is None and result.steering_deg is None
    assert result.curvature_per_m is None and result.segment_index is None


def test_conversion_uses_given_bev_origin_offset_and_does_not_mutate():
    path_cm = np.array([[0., 0.], [100., 30.], [200., -40.]])
    original = path_cm.copy()
    transformed = bev_cm_to_rear_m(path_cm, .73)
    np.testing.assert_allclose(transformed, [[.73, 0], [1.73, .3], [2.73, -.4]])
    np.testing.assert_array_equal(path_cm, original)
    calculate_pure_pursuit(transformed)
    np.testing.assert_array_equal(path_cm, original)


def test_original_image_bottom_centre_maps_to_course_rear_axle_distance():
    original_bottom_cm = np.array([[35.43527963, -6.71888193]])
    offset_x = .9 - original_bottom_cm[0, 0] / 100
    offset_y = -original_bottom_cm[0, 1] / 100
    np.testing.assert_allclose(bev_cm_to_rear_m(original_bottom_cm, offset_x, offset_y), [[.9, 0]], atol=1e-12)
    origin = bev_cm_to_rear_m([[0., 0.]], offset_x, offset_y)
    np.testing.assert_allclose(origin, [[.5456472037, .0671888193]], atol=1e-12)


@pytest.mark.parametrize('name,value', [
    ('lookahead_m', 0), ('lookahead_m', -1), ('lookahead_m', float('nan')),
    ('wheelbase_m', 0), ('wheelbase_m', -1), ('wheelbase_m', float('inf')),
])
def test_invalid_physical_parameters_are_rejected(name, value):
    with pytest.raises(ValueError):
        PurePursuitConfig(**{name: value})


@pytest.mark.parametrize('points,offset', [
    ([[1., float('nan')]], .9), ([[1., 0]], float('inf')), ([[1., 0, 0]], .9),
])
def test_bad_frame_conversion_is_rejected(points, offset):
    with pytest.raises(ValueError):
        bev_cm_to_rear_m(points, offset)


def test_circle_formula_and_alpha_formula_agree_on_oblique_target():
    config = PurePursuitConfig(lookahead_m=1.7, wheelbase_m=.55)
    result = calculate_pure_pursuit([[0., 0.], [3., -1.4]], config)
    assert result.valid
    assert result.curvature_per_m == pytest.approx(2 * math.sin(result.heading_error_rad) / config.lookahead_m)
    assert result.steering_rad == pytest.approx(math.atan(config.wheelbase_m * result.curvature_per_m))
