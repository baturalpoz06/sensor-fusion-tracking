"""Single-target tracking loop feeding radar and camera into one EKF."""

import numpy as np

from fusion.filters.ekf import ExtendedKalmanFilter
from fusion.sensors.base import MeasurementModel
from fusion.sensors.radar import radar_initial_estimate


def run_tracking(
    states: np.ndarray,
    *,
    dt: float,
    radar_every: int,
    radar_z: np.ndarray,
    camera_z: np.ndarray | None = None,
    radar_model: MeasurementModel,
    camera_model: MeasurementModel | None = None,
    accel_std: float,
    velocity_std: float,
    use_camera: bool,
) -> np.ndarray:
    """Run an EKF over radar measurements and, optionally, camera measurements.

    Timing, with k the step index:
        k = 0: initialize from radar_z[0] (no predict, no radar update);
               then a camera update if use_camera.
        k > 0: predict; radar update if k % radar_every == 0; then a camera
               update if use_camera. Radar always comes before camera.

    Measurements are given for every step so that runs with and without the
    camera see identical radar data; rows of radar_z at non-radar steps are
    ignored. Errors raised by the filter (e.g. zero range) propagate unchanged.

    Args:
        states: Shape (n + 1, 4) true states. Only their count is used, to fix
            the number of steps; the filter never reads the true values.
        dt: Time step in seconds.
        radar_every: Radar measures every this many steps (>= 1).
        radar_z: Shape (n + 1, 2) radar measurements [range, bearing].
        camera_z: Shape (n + 1, 1) camera measurements [bearing]. Required
            only if use_camera.
        radar_model: Radar model; its diagonal R also sets the initial covariance.
        camera_model: Camera model. Required only if use_camera.
        accel_std: Process noise acceleration std of the filter in m/s^2.
        velocity_std: Prior std of each velocity component in m/s.
        use_camera: Whether to apply camera updates.

    Returns:
        Shape (n + 1, 4) estimates; row k is the estimate after all updates at step k.

    Raises:
        ValueError: On inconsistent inputs, or from the filter itself.
    """
    n_points = len(states)
    if radar_every < 1:
        raise ValueError(f"radar_every must be >= 1, got {radar_every}")
    if len(radar_z) != n_points:
        raise ValueError(f"radar_z has {len(radar_z)} rows, expected {n_points}")
    if use_camera:
        if camera_z is None or camera_model is None:
            raise ValueError("use_camera requires camera_z and camera_model")
        if len(camera_z) != n_points:
            raise ValueError(f"camera_z has {len(camera_z)} rows, expected {n_points}")

    range_std, bearing_std = np.sqrt(np.diag(radar_model.R))
    x0, p0 = radar_initial_estimate(radar_z[0], range_std, bearing_std, velocity_std)
    ekf = ExtendedKalmanFilter(dt=dt, accel_std=accel_std, x0=x0, P0=p0)

    estimates = np.zeros((n_points, 4))
    for k in range(n_points):
        if k > 0:
            ekf.predict()
            if k % radar_every == 0:
                ekf.update(radar_z[k], radar_model)
        if use_camera:
            ekf.update(camera_z[k], camera_model)
        estimates[k] = ekf.x
    return estimates
