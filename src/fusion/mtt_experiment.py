"""Multi-target tracking experiment: scenarios, trials and parameter sweeps."""

from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from typing import NamedTuple

import numpy as np

from fusion.association.gating import gate_threshold, innovation_covariance
from fusion.filters.ekf import ExtendedKalmanFilter
from fusion.mtt_metrics import MttMetrics, evaluate_mtt
from fusion.mtt_simulation import SimulationConfig, make_mtt_rngs, simulate_mtt
from fusion.sensors.camera import CameraModel
from fusion.sensors.radar import RadarModel, radar_initial_estimate
from fusion.tracker.multi_target import TrackerConfig, run_multi_target_tracking
from fusion.tracker.track import LifecycleConfig

# Initial states [x, y, vx, vy] of the multi-target scenarios. All targets stay
# inside the default field of view and more than 250 m from the sensor for 60 s.
SCENARIOS = {
    # Two targets on crossing courses (the stress case for association and ID
    # switches), one crossing the -x axis (bearing wraps through +-pi), one far away.
    "crossing": np.array(
        [
            [-450.0, 1000.0, 15.0, 0.0],
            [450.0, 1010.0, -15.0, 0.0],
            [-1200.0, 400.0, 0.0, -15.0],
            [-1500.0, -2000.0, 10.0, -3.0],
        ]
    ),
    # Three targets far apart on non-crossing courses: the easy case.
    "separated": np.array(
        [
            [-450.0, 600.0, 15.0, 0.0],
            [400.0, -1400.0, 0.0, 12.0],
            [-1800.0, 1800.0, 10.0, -8.0],
        ]
    ),
}

# Sweepable parameters: name -> the MttConfig fields it sets. The unqualified
# names move both sensors together; the others isolate one sensor.
SWEEP_PARAMETERS = {
    "clutter_rate": ("radar_clutter_rate", "camera_clutter_rate"),
    "radar_clutter_rate": ("radar_clutter_rate",),
    "camera_clutter_rate": ("camera_clutter_rate",),
    "pd": ("radar_pd", "camera_pd"),
    "radar_pd": ("radar_pd",),
    "camera_pd": ("camera_pd",),
}


@dataclass(frozen=True)
class MttConfig(SimulationConfig):
    """Simulation, tracker and scoring parameters of one multi-target experiment.

    Adds to SimulationConfig (scene, sensors, clutter, detection probabilities):

    Attributes:
        velocity_std: Tracker prior std of each velocity component of a new track in m/s.
        gate_probability: Probability mass of the chi-square gates.
        lifecycle: M-of-N confirmation and K-miss deletion parameters.
        use_camera: Whether camera scans update the confirmed tracks.
        match_distance: Largest position error in meters of a track-target match in the metrics.
        burn_in: Initial time in seconds excluded from the metrics.
    """

    velocity_std: float = 20.0
    gate_probability: float = 0.99
    lifecycle: LifecycleConfig = field(default_factory=LifecycleConfig)
    use_camera: bool = True
    match_distance: float = 50.0
    burn_in: float = 5.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.velocity_std <= 0.0:
            raise ValueError(f"velocity_std must be > 0, got {self.velocity_std}")
        if not 0.0 < self.gate_probability < 1.0:
            raise ValueError(f"gate_probability must be in (0, 1), got {self.gate_probability}")
        if self.match_distance <= 0.0:
            raise ValueError(f"match_distance must be > 0, got {self.match_distance}")
        if not 0.0 <= self.burn_in < self.duration:
            raise ValueError(f"burn_in must be in [0, duration), got {self.burn_in}")

    @property
    def burn_in_steps(self) -> int:
        """Number of initial steps excluded from the metrics."""
        return round(self.burn_in / self.dt)

    def tracker_config(self) -> TrackerConfig:
        """Tracker parameters; tracks closer than the field of view allows are not used."""
        return TrackerConfig(
            dt=self.dt,
            accel_std=self.accel_std,
            velocity_std=self.velocity_std,
            gate_probability=self.gate_probability,
            min_range=self.fov.range_min,
            lifecycle=self.lifecycle,
            use_camera=self.use_camera,
        )


def tentative_filter(
    config: MttConfig, range_: float, coast_scans: int, bearing: float = 0.0
) -> ExtendedKalmanFilter:
    """Filter of a newborn track at the given position after coasting without updates.

    The track starts as the tracker would start it (radar_initial_estimate with
    the configured noise and velocity prior) and is predicted coast_scans radar
    periods ahead, which is the state its gate is built from at the next scan.

    Args:
        config: Experiment parameters.
        range_: Range of the birth measurement in meters.
        coast_scans: Radar periods since birth or the last update (1 = the first gating).
        bearing: Bearing of the birth measurement in radians.
    """
    x0, p0 = radar_initial_estimate(
        np.array([range_, bearing]),
        config.radar_range_std,
        config.radar_bearing_std,
        config.velocity_std,
    )
    tracker_filter = ExtendedKalmanFilter(dt=config.dt, accel_std=config.accel_std, x0=x0, P0=p0)
    for _ in range(coast_scans * config.radar_every):
        tracker_filter.predict()
    return tracker_filter


def clutter_per_gate(config: MttConfig, range_: float, coast_scans: int) -> float:
    """Expected radar clutter points in a tentative track's gate, per unit clutter rate.

    Clutter is uniform in (range, bearing) with density rate / V, where V is the
    field of view area in that space (FieldOfView.measurement_volume). The gate
    is the ellipse d^2 <= gamma in the innovation space with covariance S, whose
    area is pi * gamma * sqrt(det S). Multiply the result by the clutter rate
    for the expected count. The gate is not clipped at the field of view edges,
    so near the inner and outer radius this slightly overestimates.

    Args:
        config: Experiment parameters (velocity prior, noise, dt, gate probability, field of view).
        range_: Range of the track in meters.
        coast_scans: Radar periods the track has coasted since birth or its last update.
    """
    tracker_filter = tentative_filter(config, range_, coast_scans)
    model = RadarModel(config.radar_range_std, config.radar_bearing_std)
    S = innovation_covariance(tracker_filter.x, tracker_filter.P, model)  # noqa: N806
    gamma = gate_threshold(model.R.shape[0], config.gate_probability)
    area = np.pi * gamma * np.sqrt(np.linalg.det(S))
    return float(area / config.fov.measurement_volume)


def clutter_gate_report(
    config: MttConfig,
    rates: Sequence[float],
    ranges: Sequence[float] = (100.0, 500.0, 1500.0, 3000.0),
    coast_scans: Sequence[int] = (1, 2, 3),
) -> list[str]:
    """Report lines with the expected clutter count in a tentative track's gate.

    For every coasting time and clutter rate: the count at the given ranges and
    its mean over a track range uniform in the field of view (clutter is uniform
    in range too).

    Args:
        config: Experiment parameters.
        rates: Radar clutter rates (points per scan) to report.
        ranges: Track ranges in meters at which the count is listed.
        coast_scans: Coasting times in radar periods.
    """
    gamma = gate_threshold(2, config.gate_probability)
    fov = config.fov
    lines = [
        f"Expected radar clutter points inside a tentative track's gate "
        f"(gate {100 * config.gate_probability:g}%, gamma={gamma:.2f}, "
        f"velocity prior {config.velocity_std:g} m/s, dt={config.dt:g} s, radar every "
        f"{config.radar_every} steps, field of view area {fov.measurement_volume:.0f} m*rad):"
    ]
    mean_ranges = np.linspace(fov.range_min, fov.range_max, 60)
    for coast in coast_scans:
        per_range = [clutter_per_gate(config, r, coast) for r in ranges]
        mean_per_unit = float(np.mean([clutter_per_gate(config, r, coast) for r in mean_ranges]))
        lines.append(f"  coasted {coast} radar scan(s) since the last update:")
        for rate in rates:
            cells = "  ".join(
                f"r={r:>5.0f}: {rate * v:8.4f}" for r, v in zip(ranges, per_range, strict=True)
            )
            lines.append(
                f"    lambda={rate:>5g}  {cells}  mean over r: {rate * mean_per_unit:8.4f}"
            )
    return lines


def run_mtt_trial(initial_states: np.ndarray, config: MttConfig, seed: int) -> MttMetrics:
    """Simulate one scene, track it and score the result.

    Args:
        initial_states: Shape (n_targets, 4) starting states.
        config: Experiment parameters.
        seed: Seed of every random stream of the simulation.

    Returns:
        MttMetrics including births_per_scan from the tracker counters.
    """
    sim = simulate_mtt(initial_states, config, make_mtt_rngs(seed))
    radar_model = RadarModel(config.radar_range_std, config.radar_bearing_std)
    camera_model = CameraModel(config.camera_bearing_std) if config.use_camera else None
    run = run_multi_target_tracking(sim, config.tracker_config(), radar_model, camera_model)
    metrics = evaluate_mtt(
        sim.truth,
        run.history,
        dt=config.dt,
        fov=config.fov,
        max_distance=config.match_distance,
        burn_in_steps=config.burn_in_steps,
    )
    return metrics._replace(births_per_scan=run.tracker.births / run.tracker.radar_scans)


class SweepResult(NamedTuple):
    """Metrics of a parameter sweep.

    Attributes:
        parameter: Name from SWEEP_PARAMETERS.
        values: Shape (n_values,) the swept values.
        metrics: Metric name -> shape (n_values, n_seeds) array of per-seed scores
            (NaN where a metric is undefined for a seed).
    """

    parameter: str
    values: np.ndarray
    metrics: dict[str, np.ndarray]


def _run_trial(task: tuple[np.ndarray, MttConfig, int]) -> MttMetrics:
    """Run one (initial_states, config, seed) task; module-level to be picklable."""
    return run_mtt_trial(*task)


def _run_points(
    initial_states: np.ndarray,
    points: Sequence[MttConfig],
    seeds: Sequence[int],
    workers: int,
) -> dict[str, np.ndarray]:
    """Metrics of every seed at every configuration: name -> (n_points, n_seeds) array."""
    if len(points) == 0 or len(seeds) == 0:
        raise ValueError("values and seeds must not be empty")
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")

    tasks = [(initial_states, point, seed) for point in points for seed in seeds]
    if workers == 1:
        trials = [_run_trial(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            trials = list(pool.map(_run_trial, tasks))

    metrics = {name: np.zeros((len(points), len(seeds))) for name in MttMetrics._fields}
    for index, trial in enumerate(trials):
        i, j = divmod(index, len(seeds))
        for name, score in zip(MttMetrics._fields, trial, strict=True):
            metrics[name][i, j] = score
    return metrics


def sweep(
    initial_states: np.ndarray,
    config: MttConfig,
    parameter: str,
    values: Sequence[float],
    seeds: Sequence[int],
    workers: int = 1,
) -> SweepResult:
    """Run the same seeds at every value of one parameter.

    The seeds are shared by all values, so truth, target noise and detection
    uniforms are identical across the sweep (see simulate_mtt). Trials are
    independent and deterministic per seed, so the result does not depend on workers.
    Everything except the swept fields, including match_distance, is the same at
    every value.

    Args:
        initial_states: Shape (n_targets, 4) starting states.
        config: Base parameters; the swept fields are overridden.
        parameter: A key of SWEEP_PARAMETERS.
        values: Values to assign.
        seeds: Seeds of the trials at each value.
        workers: Number of worker processes; 1 runs in this process.

    Raises:
        ValueError: If the parameter is unknown, values or seeds are empty, or workers < 1.
    """
    if parameter not in SWEEP_PARAMETERS:
        raise ValueError(f"unknown parameter {parameter!r}, choose from {sorted(SWEEP_PARAMETERS)}")
    points = [
        replace(config, **{name: float(value) for name in SWEEP_PARAMETERS[parameter]})
        for value in values
    ]
    metrics = _run_points(initial_states, points, seeds, workers)
    return SweepResult(parameter, np.asarray(values, dtype=float), metrics)


def compare_fusion(
    initial_states: np.ndarray,
    config: MttConfig,
    seeds: Sequence[int],
    workers: int = 1,
) -> SweepResult:
    """Score the radar-only tracker and the radar + camera tracker on the same seeds.

    Both runs see identical simulations (the camera data exists either way); only
    use_camera differs. The result is a two-point SweepResult over the parameter
    "use_camera": value 0 is radar-only, value 1 is fused.

    Args:
        initial_states: Shape (n_targets, 4) starting states.
        config: Parameters of both runs; its use_camera field is overridden.
        seeds: Seeds of the trials.
        workers: Number of worker processes; 1 runs in this process.
    """
    points = [replace(config, use_camera=False), replace(config, use_camera=True)]
    metrics = _run_points(initial_states, points, seeds, workers)
    return SweepResult("use_camera", np.array([0.0, 1.0]), metrics)
