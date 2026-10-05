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


# --- bias and camera position (Phase 8a hooks) ---------------------------------------

LEVER_STATES = np.array(
    [
        [300.0, 400.0, 1.0, 2.0],
        [-900.0, 5.0, 0.0, 0.0],  # near the -x axis: the bearing is close to +-pi
        [-900.0, -5.0, 0.0, 0.0],
        [20.0, -700.0, 3.0, 1.0],
    ]
)
CAMERA_POSITIONS = [(0.0, 0.0), (20.0, 0.0), (0.0, 20.0), (-35.0, 14.0)]


def test_defaults_reproduce_the_camera_at_the_origin_without_bias_bitwise():
    """Fails if the neutral hooks change a single bit of the original measurement."""
    reference = np.arctan2(LEVER_STATES[:, 1], LEVER_STATES[:, 0]) + np.random.default_rng(
        7
    ).normal(0.0, 0.002, size=len(LEVER_STATES))
    z = camera_measure(LEVER_STATES, 0.002, np.random.default_rng(7))
    np.testing.assert_array_equal(z[:, 0], reference)


@pytest.mark.parametrize("bias", [0.0, 0.001, -0.02, 0.5])
def test_bias_shifts_the_bearing_by_exactly_the_bias_up_to_rounding(bias):
    """Fails if the bias is not added, has the wrong sign or is applied to the noise.

    The same seed gives the same noise, so the wrapped difference of the two measurements
    is the bias; also for targets whose bearing sits at the +-pi seam.
    """
    plain = camera_measure(LEVER_STATES, 0.002, np.random.default_rng(3))
    biased = camera_measure(LEVER_STATES, 0.002, np.random.default_rng(3), bias=bias)
    difference = wrap_angle(biased - plain)
    np.testing.assert_allclose(difference, bias, rtol=0.0, atol=8 * np.spacing(np.pi))


def test_bias_is_added_without_wrapping():
    """Fails if the bias is wrapped into (-pi, pi]: a bearing at +pi must be able to exceed it."""
    z = camera_measure(np.array([[-100.0, 1e-9, 0.0, 0.0]]), 0.0, bias=0.1)
    assert z[0, 0] == pytest.approx(np.pi + 0.1)


@pytest.mark.parametrize("position", CAMERA_POSITIONS)
def test_lever_arm_gives_the_bearing_from_the_camera_position(position):
    """Fails if the bearing is still taken from the origin, or the offset has the wrong sign."""
    z = camera_measure(LEVER_STATES, 0.0, position=position)
    expected = np.arctan2(LEVER_STATES[:, 1] - position[1], LEVER_STATES[:, 0] - position[0])
    np.testing.assert_array_equal(z[:, 0], expected)


def test_a_lever_arm_changes_the_bearing_of_a_near_target_more_than_a_far_one():
    """Fails if the offset is applied to the bearing instead of to the target position."""
    near = np.array([[100.0, 0.0, 0.0, 0.0]])
    far = np.array([[2000.0, 0.0, 0.0, 0.0]])
    shift_near = camera_measure(near, 0.0, position=(0.0, 20.0))[0, 0]
    shift_far = camera_measure(far, 0.0, position=(0.0, 20.0))[0, 0]
    assert shift_near == pytest.approx(-np.arctan2(20.0, 100.0))
    assert abs(shift_near) > 10 * abs(shift_far)


def test_position_and_bias_do_not_change_the_random_draw():
    """Fails if a hook consumes or reorders random numbers: isolation of the noise stream."""
    a, b = np.random.default_rng(5), np.random.default_rng(5)
    camera_measure(LEVER_STATES, 0.002, a)
    camera_measure(LEVER_STATES, 0.002, b, position=(10.0, -4.0), bias=0.3)
    assert a.bit_generator.state == b.bit_generator.state


@pytest.mark.parametrize("position", CAMERA_POSITIONS)
def test_model_h_is_relative_to_its_position(position):
    """Fails if the model's expected bearing ignores the position it was built with."""
    model = CameraModel(0.01, position)
    for state in LEVER_STATES:
        expected = np.arctan2(state[1] - position[1], state[0] - position[0])
        np.testing.assert_array_equal(model.h(state), [expected])


@pytest.mark.parametrize("position", CAMERA_POSITIONS[1:])
@pytest.mark.parametrize("state", JACOBIAN_STATES)
def test_model_jacobian_with_a_position_matches_numerical_derivative(state, position):
    """Fails if the Jacobian is not shifted together with h."""
    model = CameraModel(0.01, position)
    x = np.array(state)
    np.testing.assert_allclose(
        model.jacobian(x), numerical_jacobian(model, x), rtol=1e-5, atol=1e-8
    )


@pytest.mark.parametrize("state", JACOBIAN_STATES)
def test_model_at_the_default_position_is_bitwise_the_original_formula(state):
    """Fails if the neutral model changes h or the Jacobian by a single bit."""
    x = np.array(state)
    model = CameraModel(0.01)
    r = float(np.hypot(x[0], x[1]))
    np.testing.assert_array_equal(model.h(x), [np.arctan2(x[1], x[0])])
    np.testing.assert_array_equal(model.jacobian(x), [[-x[1] / r**2, x[0] / r**2, 0.0, 0.0]])


def test_model_jacobian_at_an_offset_camera_position_raises():
    """Fails if the singularity guard still looks at the origin instead of the camera."""
    model = CameraModel(0.01, (30.0, -40.0))
    with pytest.raises(ValueError, match="sensor position"):
        model.jacobian(np.array([30.0, -40.0, 1.0, 1.0]))
    model.jacobian(np.array([0.0, 0.0, 1.0, 1.0]))  # the origin is an ordinary point now
