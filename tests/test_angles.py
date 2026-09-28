"""Tests for the angle helpers."""

import numpy as np

from fusion.angles import wrap_angle


def test_difference_across_the_branch_cut_is_small():
    # 3.13 and -3.13 rad are only 2*pi - 6.26 apart on the circle.
    diff = wrap_angle(3.13 - (-3.13))
    np.testing.assert_allclose(diff, 6.26 - 2.0 * np.pi)
    np.testing.assert_allclose(diff, -0.0231853, atol=1e-6)


def test_known_values():
    np.testing.assert_allclose(wrap_angle(0.0), 0.0, atol=1e-15)
    np.testing.assert_allclose(wrap_angle(2.0 * np.pi), 0.0, atol=1e-15)
    np.testing.assert_allclose(wrap_angle(1.5 * np.pi), -0.5 * np.pi)
    np.testing.assert_allclose(wrap_angle(-1.5 * np.pi), 0.5 * np.pi)
    np.testing.assert_allclose(wrap_angle(0.3), 0.3)
    np.testing.assert_allclose(wrap_angle(-0.3 + 4.0 * np.pi), -0.3)


def test_works_elementwise_on_arrays_and_stays_in_range():
    angles = np.linspace(-10.0, 10.0, 201)
    wrapped = wrap_angle(angles)
    assert wrapped.shape == angles.shape
    assert np.all(wrapped >= -np.pi) and np.all(wrapped <= np.pi)
    # Wrapping only removes whole turns.
    turns = (angles - wrapped) / (2.0 * np.pi)
    np.testing.assert_allclose(turns, np.round(turns), atol=1e-12)
