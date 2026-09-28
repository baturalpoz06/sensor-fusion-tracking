"""Tests for the camera measurement model."""

import numpy as np

from fusion.sensors.camera import camera_measure


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
