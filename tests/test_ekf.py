"""Tests for the extended Kalman filter."""

import numpy as np
import pytest

from fusion.filters.ekf import ExtendedKalmanFilter
from fusion.filters.kalman import KalmanFilter
from fusion.scenario import constant_velocity_trajectory
from fusion.sensors.camera import CameraModel
from fusion.sensors.radar import RadarModel, radar_initial_estimate, radar_measure

DT = 0.1
RANGE_STD = 5.0
BEARING_STD = 0.01
VELOCITY_STD = 20.0


def run_radar_ekf(
    measurements: np.ndarray, accel_std: float, model: RadarModel
) -> np.ndarray:
    """Filter radar measurements; the first one initializes the filter."""
    x0, p0 = radar_initial_estimate(measurements[0], RANGE_STD, BEARING_STD, VELOCITY_STD)
    ekf = ExtendedKalmanFilter(dt=DT, accel_std=accel_std, x0=x0, P0=p0)
    estimates = np.zeros((len(measurements), 4))
    estimates[0] = ekf.x
    for k in range(1, len(measurements)):
        ekf.predict()
        ekf.update(measurements[k], model)
        estimates[k] = ekf.x
    return estimates


def position_errors(positions: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.linalg.norm(positions - truth[:, :2], axis=1)


def test_predict_matches_linear_kalman_filter():
    x0 = np.array([1.0, 2.0, 3.0, -1.0])
    p0 = np.diag([4.0, 4.0, 1.0, 1.0])
    ekf = ExtendedKalmanFilter(dt=0.5, accel_std=1.5, x0=x0, P0=p0)
    kf = KalmanFilter(dt=0.5, accel_std=1.5, pos_std=1.0, x0=x0, P0=p0)
    ekf.predict()
    kf.predict()
    np.testing.assert_allclose(ekf.x, kf.x)
    np.testing.assert_allclose(ekf.P, kf.P)


def test_invalid_initial_shapes_raise():
    with pytest.raises(ValueError):
        ExtendedKalmanFilter(dt=0.1, accel_std=1.0, x0=np.zeros(3), P0=np.eye(4))
    with pytest.raises(ValueError):
        ExtendedKalmanFilter(dt=0.1, accel_std=1.0, x0=np.zeros(4), P0=np.eye(3))


def test_update_keeps_covariance_symmetric_and_reduces_trace():
    model = RadarModel(RANGE_STD, BEARING_STD)
    ekf = ExtendedKalmanFilter(
        dt=DT,
        accel_std=1.0,
        x0=np.array([100.0, 50.0, 0.0, 0.0]),
        P0=np.diag([25.0, 25.0, 100.0, 100.0]),
    )
    ekf.predict()
    trace_before = np.trace(ekf.P)
    ekf.update(model.h(ekf.x) + [1.0, 0.005], model)
    np.testing.assert_allclose(ekf.P, ekf.P.T, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(ekf.P) > 0.0)
    assert np.trace(ekf.P) < trace_before


def test_update_rejects_wrong_measurement_shape():
    ekf = ExtendedKalmanFilter(dt=DT, accel_std=1.0, x0=np.array([10.0, 10.0, 0, 0]), P0=np.eye(4))
    with pytest.raises(ValueError, match="shape"):
        ekf.update(np.array([1.0]), RadarModel(RANGE_STD, BEARING_STD))
    with pytest.raises(ValueError, match="shape"):
        ekf.update(np.array([1.0, 2.0]), CameraModel(BEARING_STD))


def test_update_at_zero_range_raises_and_leaves_state_unchanged():
    x0 = np.array([0.0, 0.0, 1.0, 1.0])
    p0 = np.diag([10.0, 10.0, 5.0, 5.0])
    ekf = ExtendedKalmanFilter(dt=DT, accel_std=1.0, x0=x0, P0=p0)
    with pytest.raises(ValueError, match="sensor position"):
        ekf.update(np.array([1.0, 0.3]), RadarModel(RANGE_STD, BEARING_STD))
    np.testing.assert_array_equal(ekf.x, x0)
    np.testing.assert_array_equal(ekf.P, p0)


def test_camera_update_only_corrects_along_the_bearing_direction():
    # A bearing measurement carries no range information: with a wide prior,
    # the radial position component must stay essentially where it was.
    ekf = ExtendedKalmanFilter(
        dt=DT,
        accel_std=1.0,
        x0=np.array([100.0, 0.0, 0.0, 0.0]),
        P0=np.diag([100.0, 100.0, 1.0, 1.0]),
    )
    ekf.update(np.array([0.05]), CameraModel(BEARING_STD))
    # Bearing 0.05 rad at ~100 m moves the estimate mostly in +y.
    assert ekf.x[1] > 1.0
    np.testing.assert_allclose(ekf.x[0], 100.0, atol=0.2)


def test_update_wraps_innovation_when_prediction_and_measurement_straddle_the_cut():
    # Predicted bearing is just below +pi, the measured one just above -pi: the
    # two are only ~0.007 rad apart on the circle, but 2*pi apart numerically.
    model = RadarModel(RANGE_STD, BEARING_STD)
    x0 = np.array([-100.0, 0.5, 0.0, 0.0])
    ekf = ExtendedKalmanFilter(
        dt=DT, accel_std=1.0, x0=x0, P0=np.diag([25.0, 25.0, 100.0, 100.0])
    )
    predicted = model.h(ekf.x)
    assert predicted[1] > 3.1
    ekf.update(np.array([predicted[0], -np.pi + 0.002]), model)
    # Without wrapping the correction would be of order 100 m * 2*pi.
    assert np.linalg.norm(ekf.x[:2] - x0[:2]) < 2.0


def test_filter_survives_target_crossing_the_negative_x_axis():
    # The drone flies slowly past the radar on the -x side, so its bearing jumps
    # from +pi to -pi and the noisy measurements straddle the cut for several steps.
    # Unwrapped, that produces ~2*pi innovations and the filter diverges to
    # hundreds of meters (checked for this seed).
    accel_std = 0.1
    rng = np.random.default_rng(1)
    truth = constant_velocity_trajectory(
        np.array([-100.0, 10.0, 0.0, -2.0]), dt=DT, n_steps=100, accel_std=accel_std, rng=rng
    )
    measurements = radar_measure(truth, RANGE_STD, BEARING_STD, rng=rng)

    # The scenario really contains the jump, and the measurements see it.
    assert np.any(np.abs(np.diff(np.arctan2(truth[:, 1], truth[:, 0]))) > np.pi)
    assert np.any(np.abs(np.diff(measurements[:, 1])) > np.pi)

    estimates = run_radar_ekf(measurements, accel_std, RadarModel(RANGE_STD, BEARING_STD))
    errors = position_errors(estimates[:, :2], truth)

    # Observed with wrapping: max error 9.7 m, mean over the second half 1.7 m.
    assert errors.max() < 15.0
    assert errors[len(errors) // 2 :].mean() < 4.0


def test_radar_ekf_position_rmse_beats_raw_radar_measurements():
    accel_std = 0.5
    rng = np.random.default_rng(7)
    truth = constant_velocity_trajectory(
        np.array([400.0, 300.0, -15.0, 8.0]), dt=DT, n_steps=300, accel_std=accel_std, rng=rng
    )
    measurements = radar_measure(truth, RANGE_STD, BEARING_STD, rng=rng)

    r, b = measurements[:, 0], measurements[:, 1]
    raw_xy = np.column_stack([r * np.cos(b), r * np.sin(b)])
    estimates = run_radar_ekf(measurements, accel_std, RadarModel(RANGE_STD, BEARING_STD))

    def rmse(positions: np.ndarray) -> float:
        return float(np.sqrt(np.mean(position_errors(positions, truth) ** 2)))

    ekf_rmse = rmse(estimates[:, :2])
    raw_rmse = rmse(raw_xy)
    # Observed: EKF 2.2 m vs raw 6.7 m (ratio 0.33; 0.27-0.33 over other seeds).
    assert ekf_rmse < 0.5 * raw_rmse
