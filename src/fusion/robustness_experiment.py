"""Robustness experiment: the tracker's beliefs against a world that differs from them.

Truth and belief are separate objects. The simulator generates the real world from the
SimulationConfig parameters and a `TruthDisturbance` (maneuvers, camera bias, time offset,
lever arm); the tracker is configured through a `TrackerBelief` with possibly wrong assumed
parameters (scale factors on its noise levels, an assumed camera position). Seeds are shared by
all rows of a sweep, so a row differs from the neutral row only by its disturbance or belief:
the same truth noise, detections and clutter (common random numbers), which is what the paired
differences of the report rely on.

All defaults are neutral: with them a trial reproduces the Phase 6 trial bit for bit.
"""

import math
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, fields, replace
from typing import NamedTuple

import numpy as np

from fusion.maneuvers import (
    Acceleration,
    RandomManeuvers,
    TruthDisturbance,
    Turn,
)
from fusion.mtt_experiment import MttConfig, SweepResult
from fusion.mtt_simulation import MttSimulation, make_mtt_rngs, simulate_mtt
from fusion.robustness_metrics import FIELDS, RobustnessMetrics, evaluate_robustness
from fusion.sensors.camera import CameraModel
from fusion.sensors.radar import RadarModel
from fusion.tracker.multi_target import TrackerConfig, TrackerRun, run_multi_target_tracking

DEG = float(np.pi / 180.0)
NOISE_KNOBS = ("radar_range", "radar_bearing", "camera_bearing", "process_noise")


@dataclass(frozen=True)
class TrackerBelief:
    """What the tracker assumes, as deviations from the true noise levels and camera position.

    Each scale multiplies the assumed standard deviation of the true one (so the assumed
    variance, or for the process noise the covariance Q, by the square of the scale). With all
    scales 1 and the camera assumed at the radar the tracker is told the truth.

    Attributes:
        radar_range_scale: Scale of the assumed radar range std.
        radar_bearing_scale: Scale of the assumed radar bearing std.
        camera_bearing_scale: Scale of the assumed camera bearing std.
        process_noise_scale: Scale of the assumed random acceleration std (Q grows with its
            square). The prior velocity std of a newborn track is not scaled.
        assumed_camera_position: Camera position (x, y) in meters that the tracker assumes. The
            default, the radar position, is the truth unless a lever arm is simulated; setting
            it to the true position gives the tracker the calibration.
    """

    radar_range_scale: float = 1.0
    radar_bearing_scale: float = 1.0
    camera_bearing_scale: float = 1.0
    process_noise_scale: float = 1.0
    assumed_camera_position: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        for name in (
            "radar_range_scale",
            "radar_bearing_scale",
            "camera_bearing_scale",
            "process_noise_scale",
        ):
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} must be finite and > 0, got {value}")
        position = self.assumed_camera_position
        if len(position) != 2 or not all(math.isfinite(c) for c in position):
            raise ValueError(f"assumed_camera_position must be two finite numbers, got {position}")


@dataclass(frozen=True)
class RobustnessConfig(MttConfig):
    """Configuration of one robustness trial: Phase 6 parameters, a disturbance and a belief.

    The inherited fields are the true scene and sensors (and the tracker's lifecycle and gate
    settings); `belief` says how the tracker's noise levels and camera position deviate from them.

    Attributes:
        truth: Disturbances of the simulated world.
        belief: What the tracker assumes.
        focus_window: Optional (start, end) in seconds for the window scores.
        diagnostic_match_distance: Match distance in meters of the error diagnostics (see
            fusion.robustness_metrics); match_distance is the one of the Phase 6 scores.
    """

    truth: TruthDisturbance = field(default_factory=TruthDisturbance)
    belief: TrackerBelief = field(default_factory=TrackerBelief)
    focus_window: tuple[float, float] | None = None
    diagnostic_match_distance: float = 200.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.diagnostic_match_distance < self.match_distance:
            raise ValueError(
                f"diagnostic_match_distance must be >= match_distance, got "
                f"{self.diagnostic_match_distance} < {self.match_distance}"
            )
        if self.focus_window is not None:
            start, end = self.focus_window
            if not 0.0 <= start < end <= self.duration:
                raise ValueError(f"focus_window must lie in [0, duration], got {self.focus_window}")

    @classmethod
    def from_mtt(cls, config: MttConfig, **overrides) -> "RobustnessConfig":
        """The robustness configuration of a Phase 6 configuration, with fields overridden."""
        values = {f.name: getattr(config, f.name) for f in fields(MttConfig)}
        return cls(**{**values, **overrides})

    def tracker_setup(self) -> tuple[TrackerConfig, RadarModel, CameraModel | None]:
        """The tracker's configuration and measurement models: the only place a belief enters."""
        belief = self.belief
        tracker_config = replace(
            super().tracker_config(), accel_std=self.accel_std * belief.process_noise_scale
        )
        radar_model = RadarModel(
            self.radar_range_std * belief.radar_range_scale,
            self.radar_bearing_std * belief.radar_bearing_scale,
        )
        camera_model = (
            CameraModel(
                self.camera_bearing_std * belief.camera_bearing_scale,
                belief.assumed_camera_position,
            )
            if self.use_camera
            else None
        )
        return tracker_config, radar_model, camera_model


def simulate_and_track(
    initial_states: np.ndarray, config: RobustnessConfig, seed: int
) -> tuple[MttSimulation, TrackerRun]:
    """Simulate the disturbed world of a seed and run the tracker on it with its beliefs."""
    sim = simulate_mtt(initial_states, config, make_mtt_rngs(seed), config.truth)
    tracker_config, radar_model, camera_model = config.tracker_setup()
    return sim, run_multi_target_tracking(sim, tracker_config, radar_model, camera_model)


def run_robustness_trial(
    initial_states: np.ndarray, config: RobustnessConfig, seed: int
) -> RobustnessMetrics:
    """Simulate one disturbed scene, track it with the configured beliefs and score the result.

    Args:
        initial_states: Shape (n_targets, 4) starting states.
        config: Experiment parameters.
        seed: Seed of every random stream of the simulation.

    Returns:
        RobustnessMetrics including run_births_per_scan from the tracker counters.
    """
    sim, run = simulate_and_track(initial_states, config, seed)
    metrics = evaluate_robustness(
        sim.truth,
        run.history,
        run.tracker.camera_log,
        sim.camera_scans,
        dt=config.dt,
        fov=config.fov,
        max_distance=config.match_distance,
        wide_distance=config.diagnostic_match_distance,
        burn_in_steps=config.burn_in_steps,
        focus_window=config.focus_window,
    )
    return metrics._replace(run_births_per_scan=run.tracker.births / run.tracker.radar_scans)


class Row(NamedTuple):
    """One row of a robustness table: a configuration, where it is plotted and its scene.

    Attributes:
        label: Text of the first table column.
        x: Value on the plot axis (the swept quantity).
        config: Configuration of the row.
        group: Name of the plotted series the row belongs to, or None for a reference row.
        states: Shape (n_targets, 4) initial states of this row, or None for the sweep default.
    """

    label: str
    x: float
    config: RobustnessConfig
    group: str | None = None
    states: np.ndarray | None = None


def _run_task(task: tuple[np.ndarray, RobustnessConfig, int]) -> RobustnessMetrics:
    """Run one (initial_states, config, seed) task; module-level to be picklable."""
    return run_robustness_trial(*task)


def robustness_sweep(
    rows: Sequence[Row],
    default_states: np.ndarray,
    parameter: str,
    seeds: Sequence[int],
    workers: int = 1,
) -> SweepResult:
    """Run the same seeds at every row, optionally in worker processes.

    Trials are independent and deterministic per seed, so the result does not depend on the
    number of workers. All rows run in one process pool, whatever scene each has.

    Args:
        rows: The configurations; a row's own initial states replace the default ones.
        default_states: Initial states of the rows that have none of their own.
        parameter: Name of the swept quantity (the plot axis).
        seeds: Seeds of the trials at each row.
        workers: Number of worker processes; 1 runs in this process.

    Returns:
        SweepResult whose metrics are the RobustnessMetrics fields, shape (n_rows, n_seeds).

    Raises:
        ValueError: If rows or seeds are empty, or workers < 1.
    """
    if len(rows) == 0 or len(seeds) == 0:
        raise ValueError("rows and seeds must not be empty")
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    tasks = [
        (default_states if row.states is None else row.states, row.config, seed)
        for row in rows
        for seed in seeds
    ]
    if workers == 1:
        trials = [_run_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            trials = list(pool.map(_run_task, tasks, chunksize=4))
    metrics = {name: np.zeros((len(rows), len(seeds))) for name in FIELDS}
    for index, result in enumerate(trials):
        i, j = divmod(index, len(seeds))
        for name, score in zip(FIELDS, result, strict=True):
            metrics[name][i, j] = score
    return SweepResult(parameter, np.array([row.x for row in rows], dtype=float), metrics)


# --- scenario 1: wrongly known noise (belief side only) ---------------------------------


def noise_scale_rows(
    base: RobustnessConfig, scales: Sequence[float] = (0.25, 0.5, 1.0, 2.0, 4.0)
) -> list[Row]:
    """The neutral row, then one row per noise knob and non-unit scale, one knob at a time.

    Each of the four knobs (radar range, radar bearing, camera bearing, process noise) is scaled
    alone; the tracker's assumed standard deviation is the true one times the scale.
    """
    rows = [Row("neutral", 1.0, base)]
    for knob in NOISE_KNOBS:
        for scale in scales:
            if scale == 1.0:
                continue
            belief = replace(base.belief, **{f"{knob}_scale": float(scale)})
            rows.append(Row(f"{knob} x{scale:g}", float(scale), replace(base, belief=belief), knob))
    return rows


# --- scenario 2: maneuvering targets (truth side) ---------------------------------------


def maneuver_rows(
    base: RobustnessConfig,
    n_targets: int,
    turn_rates_deg: Sequence[float] = (5.0, 10.0, 20.0),
    turn_window: tuple[float, float] = (15.0, 25.0),
    accelerations: Sequence[float] = (1.0, 2.0, 4.0),
    acceleration_window: tuple[float, float] = (15.0, 18.0),
    random_rates_deg: Sequence[float] = (5.0, 10.0, 20.0),
    random_segment: float = 5.0,
    random_start: float = 10.0,
) -> list[Row]:
    """The neutral row, then coordinated turns, sudden accelerations and random turns.

    Turns and accelerations apply to every target at once; the focus window of the rows is the
    maneuver window plus a recovery time, so the window scores show the response.

    Args:
        base: Neutral configuration.
        n_targets: Number of targets of the scene (the maneuvers are given to all of them).
        turn_rates_deg: Turn rates in deg/s (positive turns left) on turn_window.
        turn_window: (start, end) of the turns in seconds.
        accelerations: Accelerations in m/s^2 along the heading on acceleration_window.
        acceleration_window: (start, end) of the accelerations in seconds.
        random_rates_deg: Largest turn rates in deg/s of the random maneuvers.
        random_segment: Segment length of the random maneuvers in seconds.
        random_start: Start of the random maneuvers in seconds.
    """
    recovery = 10.0

    def focus(window: tuple[float, float]) -> tuple[float, float]:
        return window[0], min(window[1] + recovery, base.duration)

    rows = [Row("neutral", 0.0, base)]
    for rate in turn_rates_deg:
        turns = tuple((i, Turn(*turn_window, rate * DEG)) for i in range(n_targets))
        config = replace(
            base, truth=TruthDisturbance(maneuvers=turns), focus_window=focus(turn_window)
        )
        rows.append(Row(f"turn {rate:g} deg/s", float(rate), config, "turn"))
    for accel in accelerations:
        brakes = tuple((i, Acceleration(*acceleration_window, accel)) for i in range(n_targets))
        config = replace(
            base, truth=TruthDisturbance(maneuvers=brakes), focus_window=focus(acceleration_window)
        )
        rows.append(Row(f"acceleration {accel:g} m/s^2", float(accel), config, "acceleration"))
    for rate in random_rates_deg:
        spec = RandomManeuvers(random_start, random_segment, rate * DEG)
        config = replace(base, truth=TruthDisturbance(random_maneuvers=spec))
        rows.append(Row(f"random turns <= {rate:g} deg/s", float(rate), config, "random"))
    return rows


# --- scenario 3: camera bias and time offset (truth side) -------------------------------


def camera_rows(
    base: RobustnessConfig,
    biases_deg: Sequence[float] = (0.1, 0.25, 0.5, 1.0),
    offsets_ms: Sequence[float] = (25.0, 50.0, 100.0, 200.0),
) -> list[Row]:
    """The neutral row, then a constant camera bearing bias and a camera time offset, alone."""
    rows = [Row("neutral", 0.0, base)]
    for bias in biases_deg:
        config = replace(base, truth=TruthDisturbance(camera_bias=bias * DEG))
        rows.append(Row(f"bias {bias:g} deg", float(bias), config, "bias"))
    for offset in offsets_ms:
        config = replace(base, truth=TruthDisturbance(camera_time_offset=offset / 1000.0))
        rows.append(Row(f"offset {offset:g} ms", float(offset), config, "time offset"))
    return rows


# --- scenario 4a: camera lever arm -------------------------------------------------------


def lever_arm_rows(
    base: RobustnessConfig,
    distances: Sequence[float] = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0),
    angles_deg: Sequence[float] = (0.0, 90.0),
    known_angle_deg: float = 0.0,
) -> list[Row]:
    """The neutral row, then the camera at a distance from the radar, with the offset unknown
    to the tracker (at every angle) and known to it (at one angle).

    Args:
        base: Neutral configuration.
        distances: Camera distances from the radar in meters (zero is the neutral row).
        angles_deg: Directions of the camera from the radar, counter-clockwise from +x.
        known_angle_deg: The direction at which the known-calibration rows are added.
    """
    rows = [Row("neutral", 0.0, base)]

    def position(distance: float, angle_deg: float) -> tuple[float, float]:
        return distance * math.cos(angle_deg * DEG), distance * math.sin(angle_deg * DEG)

    for angle in angles_deg:
        for distance in distances:
            truth = TruthDisturbance(camera_position=position(distance, angle))
            rows.append(
                Row(
                    f"unknown {distance:g} m @{angle:g} deg",
                    float(distance),
                    replace(base, truth=truth),
                    f"unknown @{angle:g} deg",
                )
            )
    for distance in distances:
        place = position(distance, known_angle_deg)
        config = replace(
            base,
            truth=TruthDisturbance(camera_position=place),
            belief=replace(base.belief, assumed_camera_position=place),
        )
        rows.append(
            Row(f"known {distance:g} m @{known_angle_deg:g} deg", float(distance), config, "known")
        )
    return rows


# --- scenario 4b: scene geometry ---------------------------------------------------------

DIRECTIONS = ("inbound", "outbound", "tangential", "crossing")
ZONES = {"near": 600.0, "mid": 1500.0, "far": 2500.0}
GEOMETRY_SPEED = 10.0
REFERENCE_TIME = 30.0
CROSSING_MISS = 10.0


@dataclass(frozen=True)
class GeometryLayout:
    """Targets on straight courses defined by where they are at the reference time of 30 s.

    Target i is at range zone_range and bearing first_bearing_deg + i * spacing_deg at 30 s, and
    moves at 10 m/s: radially inwards or outwards ("inbound", "outbound"), or tangentially,
    counter-clockwise ("tangential"). "crossing" builds pairs of targets on the same tangent
    moving in opposite directions, offset by 10 m radially, so that they pass each other with a
    closest approach of 10 m at 30 s; pair j sits at bearing first_bearing_deg + j * spacing_deg.

    Attributes:
        direction: One of DIRECTIONS.
        zone_range: Range of the reference point in meters.
        n_targets: Number of targets (even for "crossing").
        first_bearing_deg: Bearing of the first target (or pair) at the reference time.
        spacing_deg: Bearing step between targets (or pairs).
    """

    direction: str
    zone_range: float
    n_targets: int
    first_bearing_deg: float = 0.0
    spacing_deg: float = 90.0

    def __post_init__(self) -> None:
        if self.direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {DIRECTIONS}, got {self.direction!r}")
        if self.zone_range <= 0.0 or self.n_targets < 1:
            raise ValueError("need zone_range > 0 and n_targets >= 1")
        if self.direction == "crossing" and self.n_targets % 2:
            raise ValueError(
                f"a crossing layout needs an even number of targets, got {self.n_targets}"
            )

    def states(self) -> np.ndarray:
        """Shape (n_targets, 4) initial states [x, y, vx, vy] at time 0."""
        states = np.zeros((self.n_targets, 4))
        for i in range(self.n_targets):
            pair = i // 2 if self.direction == "crossing" else i
            bearing = (self.first_bearing_deg + pair * self.spacing_deg) * DEG
            radial = np.array([np.cos(bearing), np.sin(bearing)])
            tangent = np.array([-radial[1], radial[0]])
            reach = self.zone_range
            if self.direction == "inbound":
                velocity = -GEOMETRY_SPEED * radial
            elif self.direction == "outbound":
                velocity = GEOMETRY_SPEED * radial
            elif self.direction == "tangential":
                velocity = GEOMETRY_SPEED * tangent
            else:
                velocity = GEOMETRY_SPEED * tangent * (1.0 if i % 2 == 0 else -1.0)
                reach += 0.0 if i % 2 == 0 else CROSSING_MISS
            states[i, :2] = reach * radial - velocity * REFERENCE_TIME
            states[i, 2:] = velocity
        return states


def geometry_rows(
    base: RobustnessConfig,
    lever_arm: float = 20.0,
    lever_angle_deg: float = 45.0,
    clutter_rates: Sequence[float] = (2.0, 5.0, 10.0),
    target_counts: Sequence[int] = (1, 2, 4, 8),
) -> list[Row]:
    """Rows of the scene geometry sweeps, each scene neutral and with an unknown lever arm.

    The scenes, run at the clutter rate of `base`:
        direction x zone: four targets (two pairs for "crossing") on a ring of bearings
            0, 90, 180 and 270 degrees (180 degrees for the pairs), for each of the four
            directions in each of the three zones;
        target count: 1, 2, 4 and 8 tangential targets in a 40 degree sector, mid zone;
        clutter: the mid-zone tangential ring scene at the other clutter rates.
    Each scene runs once with the camera at the radar and once with the camera at lever_arm
    meters from it that the tracker does not know about.

    Args:
        base: Neutral configuration; its clutter rates are those of the direction and count
            scenes.
        lever_arm: Distance of the displaced camera in meters.
        lever_angle_deg: Direction of the displaced camera from the radar.
        clutter_rates: Additional clutter rates of the clutter scenes.
        target_counts: Numbers of targets of the density scenes.
    """
    scenes: list[tuple[str, str, GeometryLayout, RobustnessConfig]] = []
    for direction in DIRECTIONS:
        spacing = 180.0 if direction == "crossing" else 90.0
        for zone, zone_range in ZONES.items():
            layout = GeometryLayout(direction, zone_range, 4, 0.0, spacing)
            scenes.append((f"{direction} {zone}", "direction", layout, base))
    for n in target_counts:
        first, spacing = (45.0, 0.0) if n == 1 else (25.0, 40.0 / (n - 1))
        layout = GeometryLayout("tangential", ZONES["mid"], n, first, spacing)
        scenes.append((f"tangential mid, {n} targets in a sector", "count", layout, base))
    ring = GeometryLayout("tangential", ZONES["mid"], 4, 0.0, 90.0)
    for rate in clutter_rates:
        config = replace(base, radar_clutter_rate=rate, camera_clutter_rate=rate)
        scenes.append((f"tangential mid, clutter {rate:g}", "clutter", ring, config))

    position = (
        lever_arm * math.cos(lever_angle_deg * DEG),
        lever_arm * math.sin(lever_angle_deg * DEG),
    )
    rows = []
    for index, (name, group, layout, config) in enumerate(scenes):
        states = layout.states()
        rows.append(Row(f"{name}", float(index), config, group, states))
        shifted = replace(config, truth=TruthDisturbance(camera_position=position))
        rows.append(
            Row(f"{name}, unknown {lever_arm:g} m arm", float(index), shifted, group, states)
        )
    return rows
