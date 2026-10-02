"""Multi-target scene: trajectories, missed detections and uniform clutter."""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from fusion.angles import wrap_angle
from fusion.scenario import constant_velocity_trajectory
from fusion.sensors.camera import camera_measure
from fusion.sensors.radar import radar_measure

# Independent random streams of one simulation. Each source of randomness has its
# own stream, so changing one setting (e.g. the clutter rate) never shifts the
# draws of another source (trajectories, target noise, detections).
RNG_STREAMS = (
    "trajectory",
    "radar_noise",
    "radar_detection",
    "radar_clutter",
    "radar_shuffle",
    "camera_noise",
    "camera_detection",
    "camera_clutter",
    "camera_shuffle",
)


@dataclass(frozen=True)
class FieldOfView:
    """Region in which targets are detected and clutter is generated.

    The default is a full circle around the sensor between range_min and range_max.
    The bearing sector is centered on bearing_center with the given half width.

    Attributes:
        range_min: Inner radius in meters.
        range_max: Outer radius in meters.
        bearing_center: Center of the bearing sector in radians.
        bearing_half_width: Half width of the sector in radians; pi is the full circle.
    """

    range_min: float = 50.0
    range_max: float = 3000.0
    bearing_center: float = 0.0
    bearing_half_width: float = float(np.pi)

    def __post_init__(self) -> None:
        if not 0.0 <= self.range_min < self.range_max:
            raise ValueError(
                f"need 0 <= range_min < range_max, got {self.range_min} and {self.range_max}"
            )
        if not 0.0 < self.bearing_half_width <= np.pi:
            raise ValueError(
                f"bearing_half_width must be in (0, pi], got {self.bearing_half_width}"
            )

    @property
    def measurement_volume(self) -> float:
        """Area of the field of view in (range, bearing) space, in meter * radian.

        Clutter is uniform in this space, so its density is rate / measurement_volume.
        """
        return (self.range_max - self.range_min) * 2.0 * self.bearing_half_width

    def contains(self, positions: np.ndarray) -> np.ndarray:
        """Whether each position (x, y) lies inside the field of view.

        Args:
            positions: Shape (n, 2) positions in meters.

        Returns:
            Shape (n,) boolean array.
        """
        positions = np.asarray(positions, dtype=float)
        inside = np.hypot(positions[:, 0], positions[:, 1])
        inside = (inside >= self.range_min) & (inside <= self.range_max)
        # For the full circle there is no bearing test: arctan2 returns +pi or -pi
        # on the -x axis depending on the sign of a zero y, and a seam test would
        # wrongly exclude one of the two.
        if self.bearing_half_width < np.pi:
            bearing = np.arctan2(positions[:, 1], positions[:, 0])
            offset = np.abs(wrap_angle(bearing - self.bearing_center))
            inside &= offset <= self.bearing_half_width
        return inside


@dataclass(frozen=True)
class SimulationConfig:
    """Scene and sensor parameters of the multi-target simulation.

    accel_std is used both to simulate the trajectories and as the filter's
    process noise, so the filter model matches the simulated motion. Detection
    probabilities and clutter rates are per scan: the radar scans every
    radar_every steps (including step 0), the camera at every step.

    Attributes:
        dt: Time step in seconds.
        duration: Simulated time in seconds.
        radar_every: Radar scans every this many steps.
        radar_range_std: Radar range noise std in meters.
        radar_bearing_std: Radar bearing noise std in radians.
        camera_bearing_std: Camera bearing noise std in radians.
        accel_std: Random acceleration std in m/s^2.
        radar_pd: Probability that the radar detects a target inside the field of view.
        camera_pd: Probability that the camera detects a target inside the field of view.
        radar_clutter_rate: Mean number of radar clutter points per scan (Poisson).
        camera_clutter_rate: Mean number of camera clutter points per scan (Poisson).
        fov: Field of view shared by both sensors.
    """

    dt: float = 0.1
    duration: float = 60.0
    radar_every: int = 10
    radar_range_std: float = 5.0
    radar_bearing_std: float = float(np.deg2rad(2.0))
    camera_bearing_std: float = float(np.deg2rad(0.1))
    accel_std: float = 0.5
    radar_pd: float = 0.9
    camera_pd: float = 0.9
    radar_clutter_rate: float = 5.0
    camera_clutter_rate: float = 5.0
    fov: FieldOfView = FieldOfView()

    def __post_init__(self) -> None:
        if self.dt <= 0.0 or self.duration <= 0.0:
            raise ValueError(f"dt and duration must be > 0, got {self.dt} and {self.duration}")
        if self.radar_every < 1:
            raise ValueError(f"radar_every must be >= 1, got {self.radar_every}")
        for name in ("radar_range_std", "radar_bearing_std", "camera_bearing_std"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)}")
        if self.accel_std < 0.0:
            raise ValueError(f"accel_std must be >= 0, got {self.accel_std}")
        for name in ("radar_pd", "camera_pd"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {getattr(self, name)}")
        for name in ("radar_clutter_rate", "camera_clutter_rate"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")

    @property
    def n_steps(self) -> int:
        """Number of steps after the initial one."""
        return round(self.duration / self.dt)


class Scan(NamedTuple):
    """Measurements of one sensor at one step.

    Attributes:
        z: Shape (m, d) measurements in random order; (0, d) if there are none.
        origin: Shape (m,) index of the target that produced each measurement,
            -1 for clutter. For evaluation only: the tracker never reads it.
    """

    z: np.ndarray
    origin: np.ndarray


class MttSimulation(NamedTuple):
    """True states and the scans of every step.

    Attributes:
        truth: Shape (n_targets, n + 1, 4) true states.
        radar_scans: n + 1 entries; None at steps without a radar scan.
        camera_scans: n + 1 entries, one per step.
    """

    truth: np.ndarray
    radar_scans: list[Scan | None]
    camera_scans: list[Scan]


def make_mtt_rngs(seed: int) -> dict[str, np.random.Generator]:
    """Derive one independent generator per stream in RNG_STREAMS from one seed."""
    children = np.random.SeedSequence(seed).spawn(len(RNG_STREAMS))
    generators = (np.random.default_rng(child) for child in children)
    return dict(zip(RNG_STREAMS, generators, strict=True))


def uniform_clutter(
    rng: np.random.Generator, rate: float, fov: FieldOfView, with_range: bool
) -> np.ndarray:
    """Draw one scan of clutter: a Poisson number of points, uniform in the field of view.

    Points are uniform in (range, bearing) measurement space; in Cartesian space
    the density therefore falls off as 1 / range.

    Args:
        rng: Random generator.
        rate: Mean number of clutter points (>= 0).
        fov: Field of view.
        with_range: True for radar points (range, bearing), False for camera points (bearing).

    Returns:
        Shape (c, 2) or (c, 1) array; (0, d) if no point was drawn.

    Raises:
        ValueError: If rate is negative.
    """
    if rate < 0.0:
        raise ValueError(f"rate must be >= 0, got {rate}")
    count = int(rng.poisson(rate))
    ranges = rng.uniform(fov.range_min, fov.range_max, size=count)
    bearings = wrap_angle(
        rng.uniform(
            fov.bearing_center - fov.bearing_half_width,
            fov.bearing_center + fov.bearing_half_width,
            size=count,
        )
    )
    columns = [ranges, bearings] if with_range else [bearings]
    return np.column_stack(columns) if count else np.zeros((0, len(columns)))


def _stack(arrays: list[np.ndarray], empty_shape: tuple[int, ...]) -> np.ndarray:
    """Stack arrays along a new first axis; an empty list gives zeros of empty_shape."""
    return np.stack(arrays) if arrays else np.zeros(empty_shape)


def _make_scan(
    k: int,
    target_z: np.ndarray,
    detected: np.ndarray,
    clutter: np.ndarray,
    rng: np.random.Generator,
) -> Scan:
    """Scan of step k: detected targets plus clutter, in random order."""
    idx = np.flatnonzero(detected)
    z = np.concatenate([target_z[idx, k], clutter])
    origin = np.concatenate([idx, np.full(len(clutter), -1)]).astype(int)
    order = rng.permutation(len(origin))
    return Scan(z[order], origin[order])


def simulate_mtt(
    initial_states: np.ndarray,
    config: SimulationConfig,
    rngs: dict[str, np.random.Generator],
) -> MttSimulation:
    """Simulate several targets and both sensors' scans with misses and clutter.

    Target noise is drawn for every step and target, and detection uniforms with
    a fixed shape, before any clutter; so changing a clutter rate or a detection
    probability leaves the trajectories, the target noise and the detection
    uniforms unchanged (a lower Pd only removes detections). Clutter positions
    do change with the rate, because the Poisson count shifts the clutter stream.
    A target outside the field of view is never detected. Each scan is shuffled,
    so the order of measurements does not reveal their origin.

    Args:
        initial_states: Shape (n_targets, 4) starting states [x, y, vx, vy].
        config: Scene and sensor parameters.
        rngs: Generators from make_mtt_rngs.

    Returns:
        MttSimulation with truth (n_targets, n + 1, 4) and the scans of every step.

    Raises:
        ValueError: If initial_states does not have shape (n_targets, 4).
    """
    initial_states = np.asarray(initial_states, dtype=float)
    if initial_states.ndim != 2 or initial_states.shape[1] != 4:
        raise ValueError(f"initial_states must have shape (n, 4), got {initial_states.shape}")
    n_targets, n_points = len(initial_states), config.n_steps + 1

    truth = _stack(
        [
            constant_velocity_trajectory(
                state, config.dt, config.n_steps, config.accel_std, rngs["trajectory"]
            )
            for state in initial_states
        ],
        (0, n_points, 4),
    )
    in_fov = config.fov.contains(truth[:, :, :2].reshape(-1, 2)).reshape(n_targets, n_points)

    radar_z = _stack(
        [
            radar_measure(
                states, config.radar_range_std, config.radar_bearing_std, rng=rngs["radar_noise"]
            )
            for states in truth
        ],
        (0, n_points, 2),
    )
    camera_z = _stack(
        [
            camera_measure(states, config.camera_bearing_std, rng=rngs["camera_noise"])
            for states in truth
        ],
        (0, n_points, 1),
    )
    radar_u = rngs["radar_detection"].random((n_targets, n_points))
    camera_u = rngs["camera_detection"].random((n_targets, n_points))
    radar_detected = in_fov & (radar_u < config.radar_pd)
    camera_detected = in_fov & (camera_u < config.camera_pd)

    radar_scans: list[Scan | None] = []
    camera_scans: list[Scan] = []
    for k in range(n_points):
        if k % config.radar_every == 0:
            clutter = uniform_clutter(
                rngs["radar_clutter"], config.radar_clutter_rate, config.fov, with_range=True
            )
            radar_scans.append(
                _make_scan(k, radar_z, radar_detected[:, k], clutter, rngs["radar_shuffle"])
            )
        else:
            radar_scans.append(None)
        clutter = uniform_clutter(
            rngs["camera_clutter"], config.camera_clutter_rate, config.fov, with_range=False
        )
        camera_scans.append(
            _make_scan(k, camera_z, camera_detected[:, k], clutter, rngs["camera_shuffle"])
        )
    return MttSimulation(truth, radar_scans, camera_scans)
