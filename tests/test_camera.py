"""Tests for the camera measurement model."""

import numpy as np
import pytest

from fusion.angles import wrap_angle
from fusion.sensors.camera import CameraModel, camera_measure

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


def test_output_shape_is_n_by_one():
    states = np.array(
        [
            [3.0, 4.0, 0.0, 0.0],
            [0.0, 10.0, 0.0, 0.0],
            [-5.0, -5.0, 0.0, 0.0],
        ]
    )
    z = camera_measure(states, bearing_std=0.0)
    assert z.shape == (3, 1)


def test_zero_noise_matches_geometry():
    states = np.array([[3.0, 4.0, 0.0, 0.0]])
    z = camera_measure(states, bearing_std=0.0)
    np.testing.assert_allclose(z[0], [np.arctan2(4.0, 3.0)])


def test_near_and_far_targets_same_direction_give_same_bearing():
    states = np.array(
        [
            [3.0, 4.0, 0.0, 0.0],
            [300.0, 400.0, 0.0, 0.0],
        ]
    )
    z = camera_measure(states, bearing_std=0.0)
    np.testing.assert_allclose(z[0], z[1])


def test_model_h_matches_geometry():
    model = CameraModel(bearing_std=0.01)
    np.testing.assert_allclose(model.h(np.array([3.0, 4.0, 7.0, 7.0])), [np.arctan2(4.0, 3.0)])


def test_model_noise_and_angle_index():
    model = CameraModel(bearing_std=0.01)
    np.testing.assert_allclose(model.R, [[1e-4]])
    assert model.angle_indices == (0,)


@pytest.mark.parametrize("state", JACOBIAN_STATES)
def test_model_jacobian_matches_numerical_derivative(state):
    model = CameraModel(bearing_std=0.01)
    x = np.array(state)
    analytic = model.jacobian(x)
    assert analytic.shape == (1, 4)
    np.testing.assert_allclose(analytic, numerical_jacobian(model, x), rtol=1e-5, atol=1e-8)
    np.testing.assert_array_equal(analytic[:, 2:], 0.0)


def test_model_jacobian_at_camera_position_raises():
    model = CameraModel(bearing_std=0.01)
    with pytest.raises(ValueError, match="sensor position"):
        model.jacobian(np.array([0.0, 0.0, 1.0, 1.0]))
