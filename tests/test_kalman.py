"""Tests for the linear Kalman filter."""

import numpy as np
import pytest

from fusion.filters.kalman import KalmanFilter
from fusion.scenario import constant_velocity_trajectory
from fusion.sensors.position import position_measure


def run_filter(kf: KalmanFilter, measurements: np.ndarray) -> np.ndarray:
    """Filter a measurement sequence; the first measurement is at the initial time."""
    estimates = np.zeros((len(measurements), 4))
    kf.update(measurements[0])
    estimates[0] = kf.x
    for k in range(1, len(measurements)):
        kf.predict()
        kf.update(measurements[k])
        estimates[k] = kf.x
    return estimates


def test_predict_applies_constant_velocity_motion():
    kf = KalmanFilter(
        dt=0.5, accel_std=1.0, pos_std=1.0, x0=np.array([0.0, 0.0, 2.0, 1.0]), P0=np.eye(4)
    )
    kf.predict()
    np.testing.assert_allclose(kf.x, [1.0, 0.5, 2.0, 1.0])


def test_process_noise_is_symmetric_and_positive_semidefinite():
    kf = KalmanFilter(dt=0.1, accel_std=2.0, pos_std=1.0, x0=np.zeros(4), P0=np.eye(4))
    np.testing.assert_allclose(kf.Q, kf.Q.T)
    assert np.all(np.linalg.eigvalsh(kf.Q) >= -1e-12)
    # Per-axis block: accel_std^2 * [[dt^4/4, dt^3/2], [dt^3/2, dt^2]].
    np.testing.assert_allclose(kf.Q[0, 0], 4.0 * 0.1**4 / 4)
    np.testing.assert_allclose(kf.Q[0, 2], 4.0 * 0.1**3 / 2)
    np.testing.assert_allclose(kf.Q[2, 2], 4.0 * 0.1**2)


def test_invalid_initial_shapes_raise():
    with pytest.raises(ValueError):
        KalmanFilter(dt=0.1, accel_std=1.0, pos_std=1.0, x0=np.zeros(3), P0=np.eye(4))
    with pytest.raises(ValueError):
        KalmanFilter(dt=0.1, accel_std=1.0, pos_std=1.0, x0=np.zeros(4), P0=np.eye(3))


def test_converges_to_truth_with_noise_free_measurements():
    truth = constant_velocity_trajectory(
        np.array([0.0, 0.0, 10.0, 5.0]), dt=0.1, n_steps=200, accel_std=0.0
    )
    measurements = position_measure(truth, pos_std=0.0)

    # Start far from the truth with no velocity knowledge and a wide covariance.
    kf = KalmanFilter(
        dt=0.1,
        accel_std=0.1,
        pos_std=1.0,
        x0=np.array([20.0, -15.0, 0.0, 0.0]),
        P0=np.diag([100.0, 100.0, 100.0, 100.0]),
    )
    estimates = run_filter(kf, measurements)

    np.testing.assert_allclose(estimates[-1, :2], truth[-1, :2], atol=0.05)
    np.testing.assert_allclose(estimates[-1, 2:], truth[-1, 2:], atol=0.1)


def test_filter_position_rmse_beats_raw_measurements():
    dt = 0.1
    accel_std = 0.5
    pos_std = 5.0
    rng = np.random.default_rng(7)
    truth = constant_velocity_trajectory(
        np.array([0.0, 0.0, 15.0, -8.0]), dt=dt, n_steps=300, accel_std=accel_std, rng=rng
    )
    measurements = position_measure(truth, pos_std=pos_std, rng=rng)

    kf = KalmanFilter(
        dt=dt,
        accel_std=accel_std,
        pos_std=pos_std,
        x0=np.array([measurements[0, 0], measurements[0, 1], 0.0, 0.0]),
        P0=np.diag([pos_std**2, pos_std**2, 20.0**2, 20.0**2]),
    )
    estimates = run_filter(kf, measurements)

    def position_rmse(positions: np.ndarray) -> float:
        errors = positions - truth[:, :2]
        return float(np.sqrt(np.mean(np.sum(errors**2, axis=1))))

    filter_rmse = position_rmse(estimates[:, :2])
    raw_rmse = position_rmse(measurements)
    assert filter_rmse < raw_rmse


def test_predict_increases_and_update_decreases_covariance_trace():
    kf = KalmanFilter(
        dt=0.1,
        accel_std=1.0,
        pos_std=2.0,
        x0=np.zeros(4),
        P0=np.diag([10.0, 10.0, 5.0, 5.0]),
    )
    rng = np.random.default_rng(3)
    for _ in range(5):
        trace_before = np.trace(kf.P)
        kf.predict()
        trace_after_predict = np.trace(kf.P)
        kf.update(rng.normal(0.0, 2.0, size=2))
        trace_after_update = np.trace(kf.P)

        assert trace_after_predict > trace_before
        assert trace_after_update < trace_after_predict
