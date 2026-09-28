"""Tests for the position measurement model."""

import numpy as np

from fusion.sensors.position import position_measure


def test_output_shape():
    states = np.zeros((7, 4))
    z = position_measure(states, pos_std=1.0, rng=np.random.default_rng(0))
    assert z.shape == (7, 2)


def test_zero_noise_returns_true_position():
    states = np.array([[3.0, -4.0, 1.0, 2.0], [10.0, 20.0, -5.0, 0.0]])
    z = position_measure(states, pos_std=0.0)
    np.testing.assert_allclose(z, states[:, :2])


def test_noise_level_matches_parameters():
    states = np.tile([100.0, -50.0, 0.0, 0.0], (20_000, 1))
    z = position_measure(states, pos_std=3.0, rng=np.random.default_rng(0))
    errors = z - states[:, :2]
    np.testing.assert_allclose(errors.std(axis=0), [3.0, 3.0], rtol=0.05)


def test_same_seed_same_measurements():
    states = np.tile([1.0, 2.0, 0.0, 0.0], (10, 1))
    a = position_measure(states, pos_std=1.0, rng=np.random.default_rng(42))
    b = position_measure(states, pos_std=1.0, rng=np.random.default_rng(42))
    np.testing.assert_array_equal(a, b)
