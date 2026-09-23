"""Tests for the radar measurement model."""

import numpy as np

from fusion.sensors.radar import radar_measure


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