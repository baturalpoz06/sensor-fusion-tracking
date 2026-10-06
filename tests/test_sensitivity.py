"""Tests of the bias sensitivity V = d x / d b of the EKF and the IMM filter.

Each test docstring names the defect that makes it fail.
"""

import numpy as np
import pytest

import fusion.filters.imm as imm_module
from fusion.filters.ekf import ExtendedKalmanFilter
from fusion.filters.imm import IMMFilter, ct_mode, cv_mode, uniform_transition
from fusion.filters.sensitivity import SensitivityEKF, bias_jacobian, correct_output
from fusion.maneuvers import Turn, maneuvering_trajectory
from fusion.sensors.camera import CameraModel
from fusion.sensors.radar import RadarModel

DT = 0.1
DEG = float(np.pi / 180.0)
RADAR = RadarModel(range_std=5.0, bearing_std=float(np.deg2rad(2.0)))
CAMERA = CameraModel(bearing_std=float(np.deg2rad(0.1)))
START = np.array([-450.0, 1000.0, 15.0, 0.0])
COV0 = np.diag([25.0, 25.0, 1.0, 1.0])
EPS = 1e-6  # rad, finite-difference step of the bias


def path(turn_deg: float = 0.0, steps: int = 80) -> np.ndarray:
    schedule = [Turn(0.0, steps * DT, turn_deg * DEG)] if turn_deg else []
    return maneuvering_trajectory(
        START, DT, steps, 0.0, np.random.default_rng(0), schedule
    ).states


def drive(filt, truth: np.ndarray, bias: float) -> None:
    """Noise-free radar (1 Hz) and camera (10 Hz) data; the camera bearing carries the bias."""
    for k in range(1, len(truth)):
        filt.predict()
        if k % 10 == 0:
            filt.update(RADAR.h(truth[k]), RADAR)
        filt.update(CAMERA.h(truth[k]) + bias, CAMERA)


def test_the_bias_jacobian_is_one_for_the_camera_and_zero_for_the_radar():
    """Fails if the radar is given a bias or the camera is not."""
    np.testing.assert_array_equal(bias_jacobian(CAMERA), [1.0])
    np.testing.assert_array_equal(bias_jacobian(RADAR), [0.0, 0.0])


def test_one_camera_update_gives_the_gain_as_sensitivity():
    """Fails if V is not K * J_b after the first update (hand numbers).

    Target at (1000, 0) with P = diag(100, 100, 1, 1): H = [0, 1/1000, 0, 0], so
    K_y = P_yy H / (P_yy H^2 + R) and V = [0, K_y, 0, 0].
    """
    filt = SensitivityEKF(DT, 0.5, np.array([1000.0, 0.0, 0.0, 0.0]), np.diag([100, 100, 1, 1.0]))
    np.testing.assert_array_equal(filt.V, np.zeros(4))
    filt.update(np.array([0.0]), CAMERA)
    r = CAMERA.R[0, 0]
    k_y = 100.0 * 1e-3 / (100.0 * 1e-6 + r)
    np.testing.assert_allclose(filt.V, [0.0, k_y, 0.0, 0.0], atol=1e-12)


def test_a_radar_update_alone_does_not_create_sensitivity_but_shrinks_it():
    """Fails if J_b is nonzero for the radar, or V is not multiplied by (I - K H)."""
    filt = SensitivityEKF(DT, 0.5, START, COV0)
    filt.update(np.array([0.0]) + CAMERA.h(START), CAMERA)
    before = filt.V.copy()
    filt.update(RADAR.h(START), RADAR)
    assert np.abs(filt.V).max() < np.abs(before).max()
    fresh = SensitivityEKF(DT, 0.5, START, COV0)
    fresh.update(RADAR.h(START), RADAR)
    np.testing.assert_array_equal(fresh.V, np.zeros(4))


def test_the_sensitivity_filter_runs_exactly_as_the_plain_ekf():
    """Fails if tracking V changes x or P by even one bit."""
    truth = path(5.0)
    plain = ExtendedKalmanFilter(dt=DT, accel_std=0.5, x0=START, P0=COV0)
    sens = SensitivityEKF(DT, 0.5, START, COV0)
    drive(plain, truth, 0.0)
    drive(sens, truth, 0.0)
    np.testing.assert_array_equal(plain.x, sens.x)
    np.testing.assert_array_equal(plain.P, sens.P)


def test_the_sensitivity_matches_a_finite_difference_of_the_filter():
    """Fails if V is wrong in scale or sign, or misses K J_b / the (I - K H) factor.

    Filter run with a constant bias eps added to the camera minus the unbiased run, divided
    by eps, must equal V of the unbiased run. The data are noise-free and the filter model
    is exact, so the innovations vanish and the dependence of the gain on the estimate (a
    second-order effect) does not enter.
    """
    truth = path(0.0)
    base = SensitivityEKF(DT, 0.5, START, COV0)
    biased = SensitivityEKF(DT, 0.5, START, COV0)
    drive(base, truth, 0.0)
    drive(biased, truth, EPS)
    finite_difference = (biased.x - base.x) / EPS
    np.testing.assert_allclose(base.V, finite_difference, rtol=1e-4, atol=1e-6)
    assert np.abs(base.V[:2]).max() > 100.0  # about the range in meters per radian


def test_with_a_model_mismatch_the_sensitivity_is_the_first_order_derivative():
    """Fails if V is wrong by more than the neglected second-order term (1% of its size).

    A turning target against a constant-velocity filter has non-zero innovations, so the exact
    derivative also contains d K / d b times the innovation, which V leaves out (first order).
    """
    truth = path(8.0)
    base = SensitivityEKF(DT, 0.5, START, COV0)
    biased = SensitivityEKF(DT, 0.5, START, COV0)
    drive(base, truth, 0.0)
    drive(biased, truth, EPS)
    finite_difference = (biased.x - base.x) / EPS
    assert np.abs(base.V - finite_difference).max() < 0.01 * np.abs(finite_difference).max()


def imm_pair(motion_modes, monkeypatch):
    """Two IMM filters with the mode probabilities held independent of the data."""
    monkeypatch.setattr(
        imm_module, "update_mode_probabilities", lambda predicted, log_likelihood: (predicted, True)
    )
    transition = uniform_transition(0.95, len(motion_modes), 10)
    mu0 = np.array([0.5, 0.3, 0.2][: len(motion_modes)])
    mu0 = mu0 / mu0.sum()
    return [
        IMMFilter(motion_modes, transition, START, COV0, mu0=mu0, sensitivity=True)
        for _ in range(2)
    ]


@pytest.mark.parametrize("n_modes", [1, 2, 3])
def test_the_imm_sensitivity_matches_a_finite_difference_with_the_mode_probabilities_fixed(
    n_modes, monkeypatch
):
    """Fails if a mode's V ignores its own gain, or V is wrong in scale or sign.

    With mu independent of the data (held by construction), the combined V and every V_j must
    equal the finite difference of the combined estimate and of the mode estimates up to the
    first-order error of 1% (the modes do not match the straight target exactly).
    """
    modes = [cv_mode(DT, 0.5), ct_mode(DT, 10.0 * DEG, 1.0), ct_mode(DT, -10.0 * DEG, 1.0)]
    base, biased = imm_pair(modes[:n_modes], monkeypatch)
    truth = path(0.0)
    drive(base, truth, 0.0)
    drive(biased, truth, EPS)
    combined = (biased.x - base.x) / EPS
    assert np.abs(base.V - combined).max() < 0.01 * np.abs(combined).max()
    per_mode = (biased.x_modes - base.x_modes) / EPS
    assert np.abs(base.V_modes - per_mode).max() < 0.01 * np.abs(per_mode).max()


def test_the_sensitivities_are_mixed_with_the_mixing_weights():
    """Fails if V_j is not mixed like the states, or the weights use Pi transposed.

    Asymmetric Pi = [[.9, .1], [.3, .7]], mu = [.6, .4]: weights w_00 = 9/11, w_10 = 2/11,
    w_01 = 3/17, w_11 = 14/17. Mode states and covariances are equal (so the states do not
    interfere); with V_0 = e_x * 1000 and V_1 = 0 the predicted V_j = F (sum_i w_ij V_i).
    """
    mode = cv_mode(DT, 0.0)
    imm = IMMFilter(
        [mode, mode],
        np.array([[0.9, 0.1], [0.3, 0.7]]),
        START,
        COV0,
        mu0=np.array([0.6, 0.4]),
        sensitivity=True,
    )
    imm._V_modes = np.array([[1000.0, 0.0, 10.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    imm.predict()
    start = np.array([1000.0, 0.0, 10.0, 0.0])
    expected = np.array([9 / 11 * start, 3 / 17 * start])
    expected = expected @ mode.F.T
    np.testing.assert_allclose(imm.V_modes, expected, rtol=1e-13)


def test_a_single_mode_imm_has_the_sensitivity_of_the_sensitivity_ekf_bit_for_bit():
    """Fails if one mode takes a different path from SensitivityEKF for V."""
    truth = path(3.0)
    ekf = SensitivityEKF(DT, 0.5, START, COV0)
    imm = IMMFilter(
        [cv_mode(DT, 0.5)], uniform_transition(0.95, 1, 10), START, COV0, sensitivity=True
    )
    drive(ekf, truth, 1e-3)
    drive(imm, truth, 1e-3)
    np.testing.assert_array_equal(imm.V, ekf.V)
    np.testing.assert_array_equal(imm.x, ekf.x)


def test_tracking_the_sensitivity_does_not_change_the_imm_estimate():
    """Fails if sensitivity=True changes x, P or mu of an IMM filter."""
    modes = [cv_mode(DT, 0.5), ct_mode(DT, 10.0 * DEG, 1.0), ct_mode(DT, -10.0 * DEG, 1.0)]
    transition = uniform_transition(0.95, 3, 10)
    with_v = IMMFilter(modes, transition, START, COV0, sensitivity=True)
    without = IMMFilter(modes, transition, START, COV0)
    truth = path(8.0)
    drive(with_v, truth, 1e-3)
    drive(without, truth, 1e-3)
    np.testing.assert_array_equal(with_v.x, without.x)
    np.testing.assert_array_equal(with_v.P, without.P)
    np.testing.assert_array_equal(with_v.mu, without.mu)
    assert without.V is None and without.V_modes is None


def test_a_failed_update_leaves_the_sensitivity_unchanged():
    """Fails if V is modified before an update that raises."""
    modes = [cv_mode(DT, 0.5), cv_mode(DT, 2.0, "hi")]
    imm = IMMFilter(
        modes, uniform_transition(0.95, 2, 10), START, COV0, sensitivity=True
    )
    drive(imm, path(), 0.0)
    imm._x_modes[1] = np.zeros(4)
    before = imm.V_modes
    with pytest.raises(ValueError):
        imm.update(np.array([700.0, 0.5]), RADAR)
    np.testing.assert_array_equal(imm.V_modes, before)


def test_the_output_correction_subtracts_the_scaled_sensitivity_and_adds_the_bias_variance():
    """Fails if the sign is flipped or the covariance gets no V var V^T (hand numbers)."""
    x = np.array([10.0, 20.0, 1.0, 2.0])
    cov = np.eye(4)
    v = np.array([1000.0, -500.0, 0.0, 10.0])
    x_hat, cov_hat = correct_output(x, cov, v, 0.002, 1e-6)
    np.testing.assert_allclose(x_hat, [8.0, 21.0, 1.0, 1.98])
    np.testing.assert_allclose(cov_hat - cov, 1e-6 * np.outer(v, v))
    assert cov_hat[0, 0] == pytest.approx(1.0 + 1e-6 * 1000.0**2)


def test_applying_the_true_bias_removes_the_error_it_causes():
    """Fails if the corrected estimate of a biased run is not (nearly) the unbiased one.

    With the exact bias b, x - V b equals the estimate of the unbiased data to first order.
    """
    truth = path(0.0)
    bias = 0.5 * DEG
    base = SensitivityEKF(DT, 0.5, START, COV0)
    biased = SensitivityEKF(DT, 0.5, START, COV0)
    drive(base, truth, 0.0)
    drive(biased, truth, bias)
    corrected, _ = correct_output(biased.x, biased.P, biased.V, bias, 0.0)
    assert np.hypot(*(biased.x - base.x)[:2]) > 5.0
    assert np.hypot(*(corrected - base.x)[:2]) < 0.2
