"""Oracle tests of the IMM filter against hand-computed numbers.

Each test docstring names the defect that makes it fail.
"""

import numpy as np
import pytest

import fusion.filters.imm as imm_module
from fusion.filters.base import cv_transition
from fusion.filters.ekf import ExtendedKalmanFilter
from fusion.filters.imm import (
    IMMFilter,
    combine_modes,
    ct_mode,
    ct_transition,
    cv_mode,
    mix_modes,
    uniform_transition,
    update_mode_probabilities,
)
from fusion.maneuvers import _advance
from fusion.sensors.camera import CameraModel
from fusion.sensors.radar import RadarModel

DT = 0.1
RADAR = RadarModel(range_std=5.0, bearing_std=float(np.deg2rad(2.0)))
CAMERA = CameraModel(bearing_std=float(np.deg2rad(0.1)))
X0 = np.array([600.0, 400.0, 3.0, -2.0])
COV0 = np.diag([30.0, 40.0, 4.0, 5.0])


def per_scan_matrix(stay: float, n_modes: int) -> np.ndarray:
    """The scan matrix written out entry by entry, independent of uniform_transition."""
    off = (1.0 - stay) / (n_modes - 1)
    return np.array([[stay if i == j else off for j in range(n_modes)] for i in range(n_modes)])


def measurements(ekf: ExtendedKalmanFilter, rng: np.random.Generator, radar: bool):
    """Radar or camera measurement around the filter's current position."""
    bearing = np.arctan2(ekf.x[1], ekf.x[0])
    if radar:
        return np.array([np.hypot(ekf.x[0], ekf.x[1]) + rng.normal(0, 5.0),
                         bearing + rng.normal(0, 0.03)])
    return np.array([bearing + rng.normal(0, 0.002)])


@pytest.mark.parametrize("n_modes", [2, 3])
def test_the_step_matrix_raised_to_the_scan_length_is_the_scan_matrix(n_modes):
    """Fails if the per-scan stay probability is used per step, or the root is wrong."""
    step = uniform_transition(0.95, n_modes, 10)
    np.testing.assert_allclose(step.sum(axis=1), 1.0, atol=1e-15)
    np.testing.assert_allclose(
        np.linalg.matrix_power(step, 10), per_scan_matrix(0.95, n_modes), atol=1e-12
    )
    assert step[0, 0] > 0.99


def test_the_transition_matrix_rejects_bad_stay_probabilities_and_handles_one_mode():
    """Fails if p <= 1/M is accepted, or one mode divides by zero."""
    for bad in (0.5, 0.3, 1.2):
        with pytest.raises(ValueError):
            uniform_transition(bad, 2, 10)
    with pytest.raises(ValueError):
        uniform_transition(0.9, 0, 10)
    np.testing.assert_array_equal(uniform_transition(0.95, 1, 10), [[1.0]])


def test_mixing_uses_the_transition_matrix_the_right_way_round_with_the_spread_term():
    """Fails if Pi is used transposed (the matrix is asymmetric) or the spread term is dropped.

    Hand numbers: Pi = [[.9, .1], [.3, .7]], mu = [.6, .4] give c = [.66, .34] and weights
    w_00 = 9/11, w_10 = 2/11, w_01 = 3/17, w_11 = 14/17. With mode states at x = 0 and x = 10
    and unit covariances: x0 = [20/11, 140/17], variance of the spread 19800/1331 and 71400/4913.
    """
    pi = np.array([[0.9, 0.1], [0.3, 0.7]])
    mu = np.array([0.6, 0.4])
    x_modes = np.array([[0.0, 0, 0, 0], [10.0, 0, 0, 0]])
    covs = np.array([np.eye(4), np.eye(4)])
    c, x0, mixed = mix_modes(pi, mu, x_modes, covs)
    np.testing.assert_allclose(c, [0.66, 0.34], rtol=1e-14)
    np.testing.assert_allclose(x0[:, 0], [20 / 11, 140 / 17], rtol=1e-14)
    np.testing.assert_allclose(mixed[0, 0, 0], 1 + 19800 / 1331, rtol=1e-13)
    np.testing.assert_allclose(mixed[1, 0, 0], 1 + 71400 / 4913, rtol=1e-13)
    np.testing.assert_allclose(mixed[0, 1, 1], 1.0, rtol=1e-14)


def test_probability_update_matches_hand_numbers():
    """Fails if the likelihood does not multiply the predicted probability: c = [.6, .4] and
    likelihoods [.2, .05] give [.12, .02] / .14."""
    mu, ok = update_mode_probabilities(np.array([0.6, 0.4]), np.log([0.2, 0.05]))
    assert ok
    np.testing.assert_allclose(mu, [6 / 7, 1 / 7], rtol=1e-14)


def test_probability_update_survives_log_likelihoods_that_underflow_in_the_linear_domain():
    """Fails if likelihoods are exponentiated before normalizing (exp(-2000) = 0 -> NaN)."""
    mu, ok = update_mode_probabilities(np.array([0.5, 0.5]), np.array([-2000.0, -2010.0]))
    expected = 1.0 / (1.0 + np.exp(-10.0))
    assert ok and np.isfinite(mu).all()
    np.testing.assert_allclose(mu, [expected, 1.0 - expected], rtol=1e-12)


def test_probability_update_keeps_the_prediction_when_every_likelihood_fails():
    """Fails if all-impossible likelihoods give NaN probabilities instead of the prediction."""
    c = np.array([0.7, 0.3])
    mu, ok = update_mode_probabilities(c, np.array([-np.inf, np.nan]))
    assert not ok
    np.testing.assert_array_equal(mu, c)


def test_a_zero_predicted_probability_gets_zero_weight():
    """Fails if a mode with c = 0 produces NaN (log 0) instead of staying at zero."""
    mu, ok = update_mode_probabilities(np.array([1.0, 0.0]), np.array([-3.0, 0.0]))
    assert ok
    np.testing.assert_array_equal(mu, [1.0, 0.0])


def test_combination_includes_the_spread_of_the_means():
    """Fails if the combined covariance is the plain average of the mode covariances.

    Two equally likely modes at x = 0 and x = 2 with unit covariance: x = 1, P_xx = 1 + 1.
    """
    x, cov, spread = combine_modes(
        np.array([0.5, 0.5]),
        np.array([[0.0, 0, 0, 0], [2.0, 0, 0, 0]]),
        np.array([np.eye(4), np.eye(4)]),
    )
    np.testing.assert_allclose(x, [1.0, 0, 0, 0])
    np.testing.assert_allclose(cov, np.diag([2.0, 1.0, 1.0, 1.0]))
    assert spread == pytest.approx(1.0)


def test_the_combined_covariance_is_positive_semi_definite():
    """Fails if the combination produces an indefinite matrix for spread-out modes."""
    rng = np.random.default_rng(5)
    x_modes = rng.normal(size=(3, 4)) * 10
    covs = np.array([np.diag(rng.uniform(0.1, 3.0, 4)) for _ in range(3)])
    _, cov, _ = combine_modes(np.array([0.2, 0.5, 0.3]), x_modes, covs)
    assert np.linalg.eigvalsh(0.5 * (cov + cov.T)).min() > 0.0


@pytest.mark.parametrize("omega_deg", [-20.0, 5.0, 30.0])
def test_the_turn_matrix_is_the_exact_motion_of_the_simulator(omega_deg):
    """Fails if the CT matrix differs from fusion.maneuvers._advance (sign, sinc forms)."""
    omega = np.deg2rad(omega_deg)
    F = ct_transition(DT, omega)  # noqa: N806
    for k in range(4):
        unit = np.zeros(4)
        unit[k] = 1.0
        column = _advance(unit, np.zeros(2), omega, DT)
        np.testing.assert_allclose(F[:, k], column, atol=1e-15)


def test_a_zero_turn_rate_is_constant_velocity_and_positive_turns_left():
    """Fails if omega -> 0 is not CV, or the turn direction is flipped."""
    np.testing.assert_allclose(ct_transition(DT, 0.0), cv_transition(DT), atol=0.0)
    np.testing.assert_allclose(ct_transition(DT, 1e-9), cv_transition(DT), atol=1e-9)
    heading_east = ct_transition(1.0, np.deg2rad(30.0)) @ np.array([0.0, 0.0, 10.0, 0.0])
    assert heading_east[3] > 0.0  # velocity turned towards +y: counter-clockwise


def test_the_mode_covariance_is_propagated_as_F_P_Ft():  # noqa: N802
    """Fails if the covariance is propagated as F^T P F (the turn matrix is not symmetric).

    omega = 90 deg/s, dt = 1, P = diag(0, 0, 1, 0), no process noise: F e_2 = [2/pi, 2/pi, 0, 1].
    """
    mode = ct_mode(1.0, np.deg2rad(90.0), accel_std=0.0)
    imm = IMMFilter([mode], uniform_transition(0.95, 1, 1), np.zeros(4), np.diag([0.0, 0, 1, 0]))
    imm.predict()
    column = np.array([2 / np.pi, 2 / np.pi, 0.0, 1.0])
    np.testing.assert_allclose(imm.P, np.outer(column, column), atol=1e-15)


def test_a_single_mode_imm_is_the_ekf_bit_for_bit():
    """Fails if one mode goes through mixing / combination or any different operation order."""
    imm = IMMFilter([cv_mode(DT, 0.5)], uniform_transition(0.95, 1, 10), X0, COV0)
    ekf = ExtendedKalmanFilter(dt=DT, accel_std=0.5, x0=X0, P0=COV0)
    rng = np.random.default_rng(1)
    for k in range(120):
        imm.predict()
        ekf.predict()
        if k % 10 == 0:
            z = measurements(ekf, rng, radar=True)
            imm.update(z, RADAR)
            ekf.update(z, RADAR)
        z = measurements(ekf, rng, radar=False)
        imm.update(z, CAMERA)
        ekf.update(z, CAMERA)
        np.testing.assert_array_equal(imm.x, ekf.x)
        np.testing.assert_array_equal(imm.P, ekf.P)
    np.testing.assert_array_equal(imm.mu, [1.0])


@pytest.mark.parametrize("n_modes", [2, 3])
def test_identical_modes_reproduce_the_ekf(n_modes):
    """Fails if the general mixing / combination path distorts the estimate: modes that are all
    the same CV model must give the plain EKF (to rounding) with an unchanged mu."""
    modes = [cv_mode(DT, 0.5, name=f"cv{i}") for i in range(n_modes)]
    imm = IMMFilter(modes, uniform_transition(0.9, n_modes, 10), X0, COV0)
    ekf = ExtendedKalmanFilter(dt=DT, accel_std=0.5, x0=X0, P0=COV0)
    rng = np.random.default_rng(2)
    for k in range(100):
        imm.predict()
        ekf.predict()
        z = measurements(ekf, rng, radar=False)
        imm.update(z, CAMERA)
        ekf.update(z, CAMERA)
        if k % 10 == 0:
            z = measurements(ekf, rng, radar=True)
            imm.update(z, RADAR)
            ekf.update(z, RADAR)
    np.testing.assert_allclose(imm.x, ekf.x, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(imm.P, ekf.P, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(imm.mu, np.full(n_modes, 1.0 / n_modes), atol=1e-10)


def test_coasting_moves_the_probabilities_with_the_step_matrix():
    """Fails if predict does not apply mu <- Pi^T mu (so coasting would freeze or jump mu)."""
    pi = uniform_transition(0.95, 3, 10)
    mu0 = np.array([0.7, 0.2, 0.1])
    modes = [cv_mode(DT, 0.5), cv_mode(DT, 2.0, "hi"), cv_mode(DT, 1.0, "mid")]
    imm = IMMFilter(modes, pi, X0, COV0, mu0=mu0)
    for _ in range(10):
        imm.predict()
    np.testing.assert_allclose(imm.mu, np.linalg.matrix_power(pi.T, 10) @ mu0, atol=1e-13)
    np.testing.assert_allclose(imm.mu, per_scan_matrix(0.95, 3).T @ mu0, atol=1e-12)


def test_a_failed_update_changes_nothing():
    """Fails if an update that raises for one mode has already changed another mode or mu."""
    modes = [cv_mode(DT, 0.5), cv_mode(DT, 2.0, "hi")]
    imm = IMMFilter(modes, uniform_transition(0.95, 2, 10), X0, COV0)
    imm.predict()
    imm._x_modes[1] = np.zeros(4)  # this mode sits on the radar: it cannot be linearized
    imm._combine()
    before = (imm.x_modes, imm.P_modes, imm.mu, imm.x.copy())
    with pytest.raises(ValueError):
        imm.update(np.array([700.0, 0.5]), RADAR)
    np.testing.assert_array_equal(imm.x_modes, before[0])
    np.testing.assert_array_equal(imm.P_modes, before[1])
    np.testing.assert_array_equal(imm.mu, before[2])
    np.testing.assert_array_equal(imm.x, before[3])


def test_unusable_likelihoods_are_counted_and_keep_the_probabilities(monkeypatch):
    """Fails if the all-failed fallback is not taken or not counted in likelihood_failures."""
    monkeypatch.setattr(imm_module, "mode_log_likelihood", lambda residual, cov: float("-inf"))
    modes = [cv_mode(DT, 0.5), cv_mode(DT, 2.0, "hi")]
    imm = IMMFilter(modes, uniform_transition(0.95, 2, 10), X0, COV0)
    imm.predict()
    predicted = imm.mu
    imm.update(np.array([np.arctan2(X0[1], X0[0])]), CAMERA)
    assert imm.likelihood_failures == 1
    np.testing.assert_array_equal(imm.mu, predicted)


def test_invalid_construction_is_rejected():
    """Fails if bad shapes or non-stochastic matrices are accepted."""
    mode = cv_mode(DT, 0.5)
    with pytest.raises(ValueError):
        IMMFilter([], np.ones((0, 0)), X0, COV0)
    with pytest.raises(ValueError):
        IMMFilter([mode], np.array([[0.5]]), X0, COV0)
    with pytest.raises(ValueError):
        IMMFilter([mode], np.ones((1, 1)), X0[:3], COV0)
    with pytest.raises(ValueError):
        IMMFilter([mode, mode], np.eye(2), X0, COV0, mu0=np.array([0.9, 0.9]))
