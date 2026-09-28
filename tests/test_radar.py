"""Tests for the radar measurement model."""

import numpy as np
import pytest

from fusion.angles import wrap_angle
from fusion.sensors.radar import RadarModel, radar_initial_estimate, radar_measure

JACOBIAN_STATES = [
    [100.0, 50.0, 3.0, -2.0],
    [-30.0, 40.0, 0.0, 0.0],
    [5.0, -80.0, 10.0, 10.0],
    [-50.0, 1e-3, 0.0, 0.0],
    [-50.0, 0.0, 1.0, 1.0],  # exactly on the -x axis, where the bearing wraps
]


def numerical_jacobian(model, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Central-difference Jacobian; angle rows are differenced with wrapping."""
    columns = []
    for i in range(len(x)):
        step = np.zeros(len(x))
        step[i] = eps
        diff = model.h(x + step) - model.h(x - step)
        angles = list(model.angle_indices)
        diff[angles] = wrap_angle(diff[angles])
        columns.append(diff / (2.0 * eps))
    return np.column_stack(columns)


def test_zero_noise_matches_geometry():
    states = np.array([[3.0, 4.0, 0.0, 0.0]])
    z = radar_measure(states, range_std=0.0, bearing_std=0.0)
    np.testing.assert_allclose(z[0], [5.0, np.arctan2(4.0, 3.0)])


def test_target_on_y_axis_has_ninety_degree_bearing():
    states = np.array([[0.0, 10.0, 0.0, 0.0]])
    z = radar_measure(states, range_std=0.0, bearing_std=0.0)
    np.testing.assert_allclose(z[0], [10.0, np.pi / 2])


def test_noise_level_matches_parameters():
    states = np.tile([1000.0, 0.0, 0.0, 0.0], (20_000, 1))
    z = radar_measure(states, range_std=5.0, bearing_std=0.01, rng=np.random.default_rng(0))
    range_errors = z[:, 0] - 1000.0
    bearing_errors = z[:, 1] - 0.0
    np.testing.assert_allclose(range_errors.std(), 5.0, rtol=0.05)
    np.testing.assert_allclose(bearing_errors.std(), 0.01, rtol=0.05)


def test_model_h_matches_geometry():
    model = RadarModel(range_std=5.0, bearing_std=0.01)
    np.testing.assert_allclose(
        model.h(np.array([3.0, 4.0, 7.0, 7.0])), [5.0, np.arctan2(4.0, 3.0)]
    )


def test_model_noise_and_angle_index():
    model = RadarModel(range_std=5.0, bearing_std=0.01)
    np.testing.assert_allclose(model.R, np.diag([25.0, 1e-4]))
    assert model.angle_indices == (1,)


@pytest.mark.parametrize("state", JACOBIAN_STATES)
def test_model_jacobian_matches_numerical_derivative(state):
    model = RadarModel(range_std=5.0, bearing_std=0.01)
    x = np.array(state)
    analytic = model.jacobian(x)
    assert analytic.shape == (2, 4)
    np.testing.assert_allclose(analytic, numerical_jacobian(model, x), rtol=1e-5, atol=1e-8)
    np.testing.assert_array_equal(analytic[:, 2:], 0.0)


def test_model_jacobian_at_radar_position_raises():
    model = RadarModel(range_std=5.0, bearing_std=0.01)
    with pytest.raises(ValueError, match="sensor position"):
        model.jacobian(np.array([0.0, 0.0, 1.0, 1.0]))


def test_initial_estimate_position_and_velocity():
    z = np.array([10.0, np.pi / 2])
    x0, p0 = radar_initial_estimate(z, range_std=5.0, bearing_std=0.01, velocity_std=20.0)
    np.testing.assert_allclose(x0, [0.0, 10.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(p0[2:, 2:], 400.0 * np.eye(2))
    np.testing.assert_array_equal(p0[:2, 2:], 0.0)


def test_initial_estimate_covariance_is_elongated_across_line_of_sight():
    # Target on the +x axis: x is the range direction, y is cross-range.
    z = np.array([1000.0, 0.0])
    _, p0 = radar_initial_estimate(z, range_std=5.0, bearing_std=0.01, velocity_std=20.0)
    np.testing.assert_allclose(p0[:2, :2], np.diag([25.0, (1000.0 * 0.01) ** 2]), atol=1e-9)


def test_initial_estimate_covariance_is_symmetric_positive_definite():
    z = np.array([500.0, 0.7])
    _, p0 = radar_initial_estimate(z, range_std=3.0, bearing_std=0.01, velocity_std=20.0)
    np.testing.assert_allclose(p0, p0.T, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(p0) > 0.0)