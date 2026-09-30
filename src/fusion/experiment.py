"""Radar vs. radar + camera fusion experiment on a single simulated target."""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from fusion.consistency import nees_by_component
from fusion.scenario import constant_velocity_trajectory
from fusion.sensors.camera import CameraModel, camera_measure
from fusion.sensors.radar import RadarModel, radar_measure, radar_to_cartesian
from fusion.tracking import run_tracking, run_tracking_with_covariance

METHODS = ("raw", "radar_only", "fusion")
FILTER_METHODS = ("radar_only", "fusion")
MASKS = ("all", "radar")

# Initial states [x, y, vx, vy] of side passes that never fly over the radar:
# over 60 s at 15 m/s the closest approach is about the y offset.
SCENARIOS = {
    "near": np.array([-450.0, 300.0, 15.0, 0.0]),
    "far": np.array([-450.0, 2000.0, 15.0, 0.0]),
}


@dataclass(frozen=True)
class FusionConfig:
    """Simulation and filter parameters shared by all methods.

    accel_std is used both to simulate the trajectory and as the filter's
    process noise, so the filter model matches the simulated motion.

    Attributes:
        dt: Time step in seconds.
        duration: Simulated time in seconds.
        radar_every: Radar measures every this many steps; the camera every step.
        radar_range_std: Radar range noise std in meters.
        radar_bearing_std: Radar bearing noise std in radians.
        camera_bearing_std: Camera bearing noise std in radians.
        accel_std: Random acceleration std in m/s^2 (truth and filter).
        velocity_std: Filter prior std of each velocity component in m/s.
        burn_in: Initial time in seconds excluded from the error metrics.
    """

    dt: float = 0.1
    duration: float = 60.0
    radar_every: int = 10
    radar_range_std: float = 5.0
    radar_bearing_std: float = float(np.deg2rad(2.0))
    camera_bearing_std: float = float(np.deg2rad(0.1))
    accel_std: float = 0.5
    velocity_std: float = 20.0
    burn_in: float = 5.0

    @property
    def n_steps(self) -> int:
        """Number of steps after the initial one."""
        return round(self.duration / self.dt)

    @property
    def burn_in_steps(self) -> int:
        """Number of initial steps excluded from the error metrics."""
        return round(self.burn_in / self.dt)


class Simulation(NamedTuple):
    """True states and measurements for every step, each of length n + 1."""

    truth: np.ndarray
    radar_z: np.ndarray
    camera_z: np.ndarray


def make_rngs(seed: int) -> tuple[np.random.Generator, ...]:
    """Derive independent (trajectory, radar, camera) generators from one seed.

    Separate streams mean that changing one sensor's noise (or how many draws
    it makes) never changes the trajectory or the other sensor's noise.
    """
    children = np.random.SeedSequence(seed).spawn(3)
    return tuple(np.random.default_rng(child) for child in children)


def simulate(
    initial_state: np.ndarray,
    config: FusionConfig,
    rngs: tuple[np.random.Generator, ...],
) -> Simulation:
    """Simulate the trajectory and pre-generate both sensors' measurements at every step.

    Args:
        initial_state: Shape (4,) true starting state.
        config: Experiment parameters.
        rngs: (trajectory, radar, camera) generators, e.g. from make_rngs.

    Returns:
        Simulation with truth (n + 1, 4), radar_z (n + 1, 2), camera_z (n + 1, 1).
    """
    trajectory_rng, radar_rng, camera_rng = rngs
    truth = constant_velocity_trajectory(
        np.asarray(initial_state, dtype=float),
        dt=config.dt,
        n_steps=config.n_steps,
        accel_std=config.accel_std,
        rng=trajectory_rng,
    )
    radar_z = radar_measure(
        truth, config.radar_range_std, config.radar_bearing_std, rng=radar_rng
    )
    camera_z = camera_measure(truth, config.camera_bearing_std, rng=camera_rng)
    return Simulation(truth, radar_z, camera_z)


def _tracking_kwargs(sim: Simulation, config: FusionConfig) -> dict:
    """Keyword arguments shared by every tracking run of one simulation."""
    return dict(
        dt=config.dt,
        radar_every=config.radar_every,
        radar_z=sim.radar_z,
        radar_model=RadarModel(config.radar_range_std, config.radar_bearing_std),
        accel_std=config.accel_std,
        velocity_std=config.velocity_std,
    )


def run_methods(sim: Simulation, config: FusionConfig) -> dict[str, np.ndarray]:
    """Position estimates of each method, each of shape (n + 1, 2).

    "raw" is the unfiltered radar conversion; it is only meaningful at radar
    steps, which evaluation_masks selects.
    """
    camera_model = CameraModel(config.camera_bearing_std)
    common = _tracking_kwargs(sim, config)
    radar_only = run_tracking(sim.truth, use_camera=False, **common)
    fusion = run_tracking(
        sim.truth, camera_z=sim.camera_z, camera_model=camera_model, use_camera=True, **common
    )
    return {
        "raw": radar_to_cartesian(sim.radar_z),
        "radar_only": radar_only[:, :2],
        "fusion": fusion[:, :2],
    }


def evaluation_masks(
    n_points: int, radar_every: int, burn_in_steps: int
) -> dict[str, np.ndarray]:
    """Boolean step masks for the error metrics.

    "all": every step k >= burn_in_steps.
    "radar": steps k >= burn_in_steps with k % radar_every == 0 (radar steps).
    """
    k = np.arange(n_points)
    after_burn_in = k >= burn_in_steps
    return {"all": after_burn_in, "radar": after_burn_in & (k % radar_every == 0)}


def position_rmse(positions: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    """Root mean square position error over the selected steps.

    Args:
        positions: Shape (n, 2) estimated positions.
        truth: Shape (n, 4) true states, or (n, 2) true positions.
        mask: Shape (n,) boolean selection of steps; must select at least one.

    Returns:
        sqrt(mean(|p_est - p_true|^2)) over the selected steps, in meters.
    """
    if not np.any(mask):
        raise ValueError("mask selects no steps")
    squared = np.sum((positions[mask] - truth[mask, :2]) ** 2, axis=1)
    return float(np.sqrt(np.mean(squared)))


def run_trial(
    initial_state: np.ndarray,
    config: FusionConfig,
    rngs: tuple[np.random.Generator, ...],
) -> dict[str, dict[str, float]]:
    """Simulate once and score every method on every mask.

    Returns:
        rmse[method][mask] in meters. rmse["raw"]["all"] is NaN, because raw
        radar positions exist only at radar steps.
    """
    sim = simulate(initial_state, config, rngs)
    positions = run_methods(sim, config)
    masks = evaluation_masks(len(sim.truth), config.radar_every, config.burn_in_steps)
    rmse = {}
    for method in METHODS:
        rmse[method] = {}
        for name, mask in masks.items():
            if method == "raw" and name == "all":
                rmse[method][name] = float("nan")
            else:
                rmse[method][name] = position_rmse(positions[method], sim.truth, mask)
    return rmse


def run_nees_trial(
    initial_state: np.ndarray,
    config: FusionConfig,
    rngs: tuple[np.random.Generator, ...],
) -> dict[str, dict[str, np.ndarray]]:
    """Simulate once and compute the NEES of both filters at every step.

    Error (truth - estimate) and covariance of step k come from the same
    run_tracking_with_covariance call, both recorded after all updates at k.

    Returns:
        nees[method][component], each of shape (n + 1,), for the methods in
        FILTER_METHODS and the components of fusion.consistency.COMPONENTS.
    """
    sim = simulate(initial_state, config, rngs)
    common = _tracking_kwargs(sim, config)
    camera_model = CameraModel(config.camera_bearing_std)
    runs = {
        "radar_only": run_tracking_with_covariance(sim.truth, use_camera=False, **common),
        "fusion": run_tracking_with_covariance(
            sim.truth,
            camera_z=sim.camera_z,
            camera_model=camera_model,
            use_camera=True,
            **common,
        ),
    }
    return {
        method: nees_by_component(sim.truth, run.estimates, run.covariances)
        for method, run in runs.items()
    }
