"""Tests for the radar + camera tracking loop."""

import numpy as np
import pytest

from fusion.filters.ekf import ExtendedKalmanFilter
from fusion.scenario import constant_velocity_trajectory
from fusion.sensors.camera import CameraModel, camera_measure
from fusion.sensors.radar import RadarModel, radar_initial_estimate, radar_measure
from fusion.tracking import run_tracking

DT = 0.1
RANGE_STD = 5.0
RADAR_BEARING_STD = np.deg2rad(2.0)
CAMERA_BEARING_STD = np.deg2rad(0.1)
ACCEL_STD = 0.5
VELOCITY_STD = 20.0


class CountingModel:
    """Wraps a measurement model and logs its name on every update (Jacobian call)."""

    def __init__(self, model, name: str, log: list[str]) -> None:
        self.model = model
        self.name = name
        self.log = log
        self.R = model.R
        self.angle_indices = model.angle_indices

    def h(self, x: np.ndarray) -> np.ndarray:
        return self.model.h(x)

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        self.log.append(self.name)
        return self.model.jacobian(x)


def make_data(n_steps: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    truth = constant_velocity_trajectory(
        np.array([-450.0, 300.0, 15.0, 0.0]), dt=DT, n_steps=n_steps, accel_std=ACCEL_STD, rng=rng
    )
    radar_z = radar_measure(truth, RANGE_STD, RADAR_BEARING_STD, rng=rng)
    camera_z = camera_measure(truth, CAMERA_BEARING_STD, rng=rng)
    return truth, radar_z, camera_z


def track(truth, radar_z, camera_z, radar_every, use_camera, radar_model=None, camera_model=None):
    return run_tracking(
        truth,
        dt=DT,
        radar_every=radar_every,
        radar_z=radar_z,
        camera_z=camera_z,
        radar_model=radar_model or RadarModel(RANGE_STD, RADAR_BEARING_STD),
        camera_model=camera_model or CameraModel(CAMERA_BEARING_STD),
        accel_std=ACCEL_STD,
        velocity_std=VELOCITY_STD,
        use_camera=use_camera,
    )


def reference_radar_only(radar_z: np.ndarray, radar_every: int) -> np.ndarray:
    """Independent radar-only loop written out step by step."""
    model = RadarModel(RANGE_STD, RADAR_BEARING_STD)
    x0, p0 = radar_initial_estimate(radar_z[0], RANGE_STD, RADAR_BEARING_STD, VELOCITY_STD)
    ekf = ExtendedKalmanFilter(dt=DT, accel_std=ACCEL_STD, x0=x0, P0=p0)
    estimates = [ekf.x.copy()]
    for k in range(1, len(radar_z)):
        ekf.predict()
        if k % radar_every == 0:
            ekf.update(radar_z[k], model)
        estimates.append(ekf.x.copy())
    return np.array(estimates)


@pytest.mark.parametrize(
    ("n_steps", "radar_every", "use_camera"),
    [(600, 10, True), (25, 10, True), (600, 10, False), (7, 1, True)],
)
def test_update_counts_and_order(n_steps, radar_every, use_camera):
    truth, radar_z, camera_z = make_data(n_steps)
    log: list[str] = []
    radar_model = CountingModel(RadarModel(RANGE_STD, RADAR_BEARING_STD), "radar", log)
    camera_model = CountingModel(CameraModel(CAMERA_BEARING_STD), "camera", log)
    track(truth, radar_z, camera_z, radar_every, use_camera, radar_model, camera_model)

    # Step 0: radar only initializes, camera may update. Then radar before camera.
    expected = ["camera"] if use_camera else []
    for k in range(1, n_steps + 1):
        if k % radar_every == 0:
            expected.append("radar")
        if use_camera:
            expected.append("camera")
    assert log == expected
    assert log.count("radar") == n_steps // radar_every
    assert log.count("camera") == (n_steps + 1 if use_camera else 0)


def test_output_shape_and_initial_estimate():
    truth, radar_z, camera_z = make_data(30)
    estimates = track(truth, radar_z, camera_z, radar_every=10, use_camera=False)
    assert estimates.shape == (31, 4)
    x0, _ = radar_initial_estimate(radar_z[0], RANGE_STD, RADAR_BEARING_STD, VELOCITY_STD)
    np.testing.assert_array_equal(estimates[0], x0)


def test_camera_off_matches_radar_only_exactly():
    truth, radar_z, camera_z = make_data(200)
    reference = reference_radar_only(radar_z, radar_every=10)

    with_camera_data = track(truth, radar_z, camera_z, radar_every=10, use_camera=False)
    np.testing.assert_array_equal(with_camera_data, reference)

    no_camera_at_all = run_tracking(
        truth,
        dt=DT,
        radar_every=10,
        radar_z=radar_z,
        radar_model=RadarModel(RANGE_STD, RADAR_BEARING_STD),
        accel_std=ACCEL_STD,
        velocity_std=VELOCITY_STD,
        use_camera=False,
    )
    np.testing.assert_array_equal(no_camera_at_all, reference)

    # Poisoned camera data would show up immediately if it were ever read.
    nan_camera = track(truth, radar_z, np.full_like(camera_z, np.nan), 10, use_camera=False)
    np.testing.assert_array_equal(nan_camera, reference)


def test_camera_on_changes_the_estimate():
    truth, radar_z, camera_z = make_data(50)
    off = track(truth, radar_z, camera_z, radar_every=10, use_camera=False)
    on = track(truth, radar_z, camera_z, radar_every=10, use_camera=True)
    assert not np.allclose(on, off)


def test_radar_rows_between_radar_steps_are_ignored():
    truth, radar_z, camera_z = make_data(50)
    poisoned = radar_z.copy()
    poisoned[np.arange(len(radar_z)) % 10 != 0] = np.nan
    np.testing.assert_array_equal(
        track(truth, poisoned, camera_z, radar_every=10, use_camera=True),
        track(truth, radar_z, camera_z, radar_every=10, use_camera=True),
    )


@pytest.mark.parametrize("use_camera", [True, False])
def test_filter_errors_are_not_swallowed(use_camera):
    # The first radar measurement puts the target at the sensor, and with zero
    # initial velocity the prediction stays there: the first update must raise.
    truth, radar_z, camera_z = make_data(20)
    radar_z[0] = [0.0, 0.0]
    with pytest.raises(ValueError, match="sensor position"):
        track(truth, radar_z, camera_z, radar_every=10, use_camera=use_camera)


def test_invalid_inputs_raise():
    truth, radar_z, camera_z = make_data(20)
    with pytest.raises(ValueError, match="radar_every"):
        track(truth, radar_z, camera_z, radar_every=0, use_camera=False)
    with pytest.raises(ValueError, match="radar_z"):
        track(truth, radar_z[:-1], camera_z, radar_every=10, use_camera=False)
    with pytest.raises(ValueError, match="camera_z"):
        track(truth, radar_z, camera_z[:-1], radar_every=10, use_camera=True)
    with pytest.raises(ValueError, match="use_camera requires"):
        run_tracking(
            truth,
            dt=DT,
            radar_every=10,
            radar_z=radar_z,
            radar_model=RadarModel(RANGE_STD, RADAR_BEARING_STD),
            accel_std=ACCEL_STD,
            velocity_std=VELOCITY_STD,
            use_camera=True,
        )
