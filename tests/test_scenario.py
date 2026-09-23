"""Tests for ground-truth trajectory generation."""

import numpy as np

from fusion.scenario import constant_velocity_trajectory


def test_output_shape():
    initial = np.array([0.0, 0.0, 10.0, 5.0])
    states = constant_velocity_trajectory(initial, dt=0.1, n_steps=50)
    assert states.shape == (51, 4)


def test_zero_noise_is_straight_line():
    initial = np.array([0.0, 0.0, 10.0, 5.0])
    states = constant_velocity_trajectory(initial, dt=0.1, n_steps=100, accel_std=0.0)
    np.testing.assert_allclose(states[-1], [100.0, 50.0, 10.0, 5.0])


def test_same_seed_same_trajectory():
    initial = np.array([0.0, 0.0, 10.0, 5.0])
    a = constant_velocity_trajectory(
        initial, dt=0.1, n_steps=100, accel_std=1.0, rng=np.random.default_rng(42)
    )
    b = constant_velocity_trajectory(
        initial, dt=0.1, n_steps=100, accel_std=1.0, rng=np.random.default_rng(42)
    )
    np.testing.assert_array_equal(a, b)