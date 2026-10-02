"""Tests for measurement gating."""

import numpy as np
import pytest

from fusion.association.gating import (
    Innovation,
    gate_threshold,
    gated_costs,
    innovation,
    innovation_covariance,
    mahalanobis_squared,
)
from fusion.filters.ekf import ExtendedKalmanFilter
from fusion.sensors.camera import CameraModel
from fusion.sensors.radar import RadarModel

RADAR = RadarModel(range_std=5.0, bearing_std=float(np.deg2rad(2.0)))
CAMERA = CameraModel(bearing_std=float(np.deg2rad(0.1)))
GAMMA_RADAR = gate_threshold(2, 0.99)
GAMMA_CAMERA = gate_threshold(1, 0.99)
MIN_RANGE = 50.0


def make_filter(x, cov=None) -> ExtendedKalmanFilter:
    if cov is None:
        cov = np.diag([25.0, 25.0, 4.0, 4.0])
    return ExtendedKalmanFilter(dt=0.1, accel_std=0.5, x0=np.asarray(x, dtype=float), P0=cov)


def at_bearing(range_: float, bearing: float) -> np.ndarray:
    return np.array([range_ * np.cos(bearing), range_ * np.sin(bearing), 0.0, 0.0])


def test_gate_threshold_matches_chi_square_table():
    assert GAMMA_RADAR == pytest.approx(9.21, abs=0.005)
    assert GAMMA_CAMERA == pytest.approx(6.63, abs=0.005)


@pytest.mark.parametrize(("dof", "probability"), [(0, 0.99), (-1, 0.99), (2, 0.0), (2, 1.0)])
def test_gate_threshold_rejects_invalid_arguments(dof, probability):
    with pytest.raises(ValueError):
        gate_threshold(dof, probability)


def test_innovation_matches_hand_calculation():
    x = np.array([100.0, 0.0, 0.0, 0.0])
    cov = np.diag([4.0, 9.0, 1.0, 1.0])
    z = np.array([103.0, 0.02])
    inn = innovation(x, cov, z, RADAR)

    # H = [[1, 0, 0, 0], [0, 1/100, 0, 0]]: S = diag(4 + Rrr, 9 / 100^2 + Rbb).
    s_range = 4.0 + 5.0**2
    s_bearing = 9.0 / 100.0**2 + np.deg2rad(2.0) ** 2
    np.testing.assert_allclose(inn.residual, [3.0, 0.02])
    np.testing.assert_allclose(inn.covariance, np.diag([s_range, s_bearing]), rtol=1e-12)
    assert mahalanobis_squared(inn) == pytest.approx(3.0**2 / s_range + 0.02**2 / s_bearing)


def test_innovation_covariance_is_symmetric():
    f = make_filter(at_bearing(500.0, 0.7), np.diag([25.0, 400.0, 4.0, 4.0]))
    S = innovation_covariance(f.x, f.P, RADAR)  # noqa: N806
    np.testing.assert_array_equal(S, S.T)


def test_innovation_agrees_with_ekf_update():
    f = make_filter(at_bearing(800.0, 1.1))
    z = np.array([812.0, 1.12])
    inn = innovation(f.x, f.P, z, RADAR)
    H = RADAR.jacobian(f.x)  # noqa: N806
    expected = f.x + f.P @ H.T @ np.linalg.solve(inn.covariance, inn.residual)
    f.update(z, RADAR)
    np.testing.assert_allclose(f.x, expected, rtol=1e-9, atol=1e-9)


def test_innovation_rejects_wrong_shape():
    f = make_filter(at_bearing(500.0, 0.3))
    with pytest.raises(ValueError, match="shape"):
        innovation(f.x, f.P, np.array([500.0]), RADAR)


@pytest.mark.parametrize(
    ("model", "gamma", "make_z"),
    [
        (RADAR, GAMMA_RADAR, lambda b: np.array([[600.0, b]])),
        (CAMERA, GAMMA_CAMERA, lambda b: np.array([[b]])),
    ],
    ids=["radar", "camera"],
)
def test_bearing_residual_is_wrapped_across_the_minus_x_axis(model, gamma, make_z):
    eps = 0.005
    f = make_filter(at_bearing(600.0, np.pi - eps))
    z = make_z(-np.pi + eps)

    costs = gated_costs([f], z, model, gamma, MIN_RANGE)
    assert costs.in_gate[0, 0]

    # Without wrapping the same pair is a ~2 pi jump and far outside the gate.
    S = innovation_covariance(f.x, f.P, model)  # noqa: N806
    raw = (z[0] - model.h(f.x)).copy()
    assert abs(raw[model.angle_indices[0]]) > 6.0
    assert raw @ np.linalg.solve(S, raw) > 100 * gamma
    # The wrapped residual is the short way around: 2 * eps.
    wrapped = innovation(f.x, f.P, z[0], model).residual
    assert abs(wrapped[model.angle_indices[0]]) == pytest.approx(2 * eps, rel=1e-6)


def test_gate_boundary_is_at_the_chi_square_threshold():
    f = make_filter(at_bearing(700.0, 0.4))
    S = innovation_covariance(f.x, f.P, RADAR)  # noqa: N806
    chol = np.linalg.cholesky(S)
    direction = np.array([0.6, 0.8])  # unit vector in whitened space
    h = RADAR.h(f.x)
    inside = h + chol @ direction * np.sqrt(GAMMA_RADAR * (1 - 1e-6))
    outside = h + chol @ direction * np.sqrt(GAMMA_RADAR * (1 + 1e-6))

    costs = gated_costs([f], np.array([inside, outside]), RADAR, GAMMA_RADAR, MIN_RANGE)
    assert costs.in_gate.tolist() == [[True, False]]


def test_cost_is_distance_plus_log_determinant():
    f = make_filter(at_bearing(700.0, 0.4))
    z = np.array([[705.0, 0.41], [690.0, 0.38]])
    costs = gated_costs([f], z, RADAR, GAMMA_RADAR, MIN_RANGE)
    S = innovation_covariance(f.x, f.P, RADAR)  # noqa: N806
    log_det = np.linalg.slogdet(S)[1]
    for j in range(2):
        d2 = mahalanobis_squared(innovation(f.x, f.P, z[j], RADAR))
        assert costs.cost[0, j] == pytest.approx(d2 + log_det)


def test_gated_costs_are_per_track_and_per_measurement():
    near, far = make_filter(at_bearing(500.0, 0.2)), make_filter(at_bearing(1500.0, -1.0))
    z = np.array([RADAR.h(near.x) + [2.0, 0.003], RADAR.h(far.x) + [-3.0, 0.004]])
    costs = gated_costs([near, far], z, RADAR, GAMMA_RADAR, MIN_RANGE)
    assert costs.cost.shape == costs.in_gate.shape == (2, 2)
    assert costs.in_gate.tolist() == [[True, False], [False, True]]


@pytest.mark.parametrize("x", [[0.0, 0.0, 0.0, 0.0], [1e-3, 0.0, 5.0, 0.0], [30.0, 0.0, 0.0, 0.0]])
def test_track_closer_than_min_range_is_out_of_gate_and_unchanged(x):
    f = make_filter(x)
    x_before, p_before = f.x.copy(), f.P.copy()
    z = np.array([[0.0, 0.0], [10.0, 1.0]])
    costs = gated_costs([f], z, RADAR, GAMMA_RADAR, MIN_RANGE)
    assert not costs.in_gate.any()
    assert np.all(np.isinf(costs.cost))
    # min_range = 0 still guards the exact sensor position through MIN_RANGE.
    if x[0] == 0.0:
        assert not gated_costs([f], z, RADAR, GAMMA_RADAR, 0.0).in_gate.any()
    np.testing.assert_array_equal(f.x, x_before)
    np.testing.assert_array_equal(f.P, p_before)


def test_filters_are_not_modified():
    f = make_filter(at_bearing(900.0, 2.0))
    x_before, p_before = f.x.copy(), f.P.copy()
    gated_costs([f], np.array([[905.0, 2.01]]), RADAR, GAMMA_RADAR, MIN_RANGE)
    np.testing.assert_array_equal(f.x, x_before)
    np.testing.assert_array_equal(f.P, p_before)


def test_non_positive_definite_covariance_is_out_of_gate():
    f = make_filter(at_bearing(900.0, 2.0), cov=-1000.0 * np.eye(4))
    costs = gated_costs([f], np.array([RADAR.h(f.x)]), RADAR, GAMMA_RADAR, MIN_RANGE)
    assert not costs.in_gate.any()
    assert mahalanobis_squared(Innovation(np.zeros(2), -np.eye(2))) == float("inf")


def test_nan_measurement_is_never_in_gate():
    f = make_filter(at_bearing(900.0, 2.0))
    z = np.array([[np.nan, 2.0], RADAR.h(f.x)])
    costs = gated_costs([f], z, RADAR, GAMMA_RADAR, MIN_RANGE)
    assert costs.in_gate.tolist() == [[False, True]]


def test_empty_inputs_give_empty_matrices():
    f = make_filter(at_bearing(900.0, 2.0))
    no_tracks = gated_costs([], np.zeros((3, 2)), RADAR, GAMMA_RADAR, MIN_RANGE)
    assert no_tracks.cost.shape == no_tracks.in_gate.shape == (0, 3)
    no_meas = gated_costs([f, f], np.zeros((0, 2)), RADAR, GAMMA_RADAR, MIN_RANGE)
    assert no_meas.cost.shape == no_meas.in_gate.shape == (2, 0)


def test_measurement_shape_is_validated():
    f = make_filter(at_bearing(900.0, 2.0))
    with pytest.raises(ValueError, match="shape"):
        gated_costs([f], np.zeros((3, 1)), RADAR, GAMMA_RADAR, MIN_RANGE)
    with pytest.raises(ValueError, match="shape"):
        gated_costs([f], np.zeros(2), RADAR, GAMMA_RADAR, MIN_RANGE)
