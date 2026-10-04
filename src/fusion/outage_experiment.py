"""Outage experiment: configuration, trials and sweeps with sensor dropouts.

A trial simulates a scene exactly as the Phase 6 experiment does, optionally removes
targets (they cease to exist), silences sensors in their outage windows (masking the
generated scans) and scores the tracker per window around the outage.

Known limitation: an aware tracker's coast clock counts steps since the last measurement update
of any sensor, and the camera update of a track is skipped when its bearing gate overlaps the
gate of another confirmed track (camera ambiguity). Two real targets with close bearings can
therefore coast without any update; in a radar outage longer than max_coast_time both are then
deleted although they exist. Only radar-only outages longer than max_coast_time with close
targets are affected (in 'crossing', targets 0 and 1 pass within 10 m at about 30 s). The
experiments keep the default max_coast_time above their radar-only outages, and the
coast_deletions counter shows it: it must read 0 there.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from fusion.dropout import (
    DropoutSpec,
    MarkovBursts,
    PeriodicFlicker,
    SingleOutage,
    apply_dropout,
    end_targets,
)
from fusion.mtt_experiment import SCENARIOS, MttConfig, SweepResult, _run_points
from fusion.mtt_simulation import make_mtt_rngs, simulate_mtt
from fusion.outage_metrics import OutageMetrics, evaluate_outage
from fusion.sensors.camera import CameraModel
from fusion.sensors.radar import RadarModel
from fusion.tracker.multi_target import TrackerConfig, run_multi_target_tracking

# The vanishing scenario is the separated layout (constant-velocity targets far apart);
# which target ceases to exist, and when, is part of the configuration (OutageConfig.vanish).
VANISHING_TARGET = 0
OUTAGE_SCENARIOS = {
    "crossing": SCENARIOS["crossing"],
    "separated": SCENARIOS["separated"],
    "vanishing": SCENARIOS["separated"],
}
SENSOR_SETS = {
    "radar": ("radar",),
    "camera": ("camera",),
    "blackout": ("radar", "camera"),
}


@dataclass(frozen=True)
class OutageConfig(MttConfig):
    """Experiment parameters of Phase 6 plus an outage and the tracker's reaction to it.

    Attributes:
        dropout: Outage specification (see fusion.dropout), or None for no outage.
        outage_policy: "unaware" or "aware" (TrackerConfig.outage_policy).
        aware_tentatives: "drop" or "freeze" (TrackerConfig.aware_tentatives).
        max_coast_time: Seconds without a measurement update after which an aware tracker
            deletes a track during a radar outage. Known limitation: with close targets a
            radar outage longer than this deletes real tracks, because the camera update
            that would reset the clock is skipped for overlapping bearing gates (see the
            module docstring).
        after_window: Length in seconds of the window after the outage that is scored.
        vanish: (target index, time in seconds) pairs: the target ceases to exist at that
            time (its detections stop; its truth is still simulated but not scored).
    """

    dropout: DropoutSpec | None = None
    outage_policy: str = "unaware"
    aware_tentatives: str = "drop"
    max_coast_time: float = 15.0
    after_window: float = 30.0
    vanish: tuple[tuple[int, float], ...] = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.tracker_config()  # validates the policy settings
        if self.dropout is not None:
            if not isinstance(self.dropout, SingleOutage | PeriodicFlicker | MarkovBursts):
                raise ValueError(f"dropout must be a dropout specification, got {self.dropout!r}")
            if self.dropout.span[1] >= self.duration:
                raise ValueError(
                    f"the outage must end before the run does: span {self.dropout.span}, "
                    f"duration {self.duration}"
                )
        if self.after_window < 0.0:
            raise ValueError(f"after_window must be >= 0, got {self.after_window}")
        for _, time in self.vanish:
            if not 0.0 <= time < self.duration:
                raise ValueError(f"vanish times must be in [0, duration), got {time}")

    def tracker_config(self) -> TrackerConfig:
        """Tracker parameters including the outage policy."""
        return replace(
            super().tracker_config(),
            outage_policy=self.outage_policy,
            aware_tentatives=self.aware_tentatives,
            max_coast_time=self.max_coast_time,
        )


def run_outage_trial(
    initial_states: np.ndarray, config: OutageConfig, seed: int
) -> OutageMetrics:
    """Simulate one scene with its outage, track it and score the result.

    The scene is generated exactly as in run_mtt_trial; vanishing targets and the outage
    are applied afterwards by masking, so truth, noise, detection and clutter draws do not
    depend on them.

    Args:
        initial_states: Shape (n_targets, 4) starting states.
        config: Experiment parameters.
        seed: Seed of every random stream of the simulation.

    Returns:
        OutageMetrics including the tracker counters.
    """
    rngs = make_mtt_rngs(seed)
    sim = simulate_mtt(initial_states, config, rngs)
    vanish = dict(config.vanish)
    alive = None
    if vanish:
        sim, alive = end_targets(sim, vanish, config.dt)
    windows = () if config.dropout is None else config.dropout.windows(config.dt, rngs["dropout"])
    dropped = apply_dropout(sim, windows, config.dt)

    radar_model = RadarModel(config.radar_range_std, config.radar_bearing_std)
    camera_model = CameraModel(config.camera_bearing_std) if config.use_camera else None
    run = run_multi_target_tracking(
        dropped.sim, config.tracker_config(), radar_model, camera_model, dropped.radar_down
    )
    metrics = evaluate_outage(
        sim.truth,
        run.history,
        dt=config.dt,
        fov=config.fov,
        max_distance=config.match_distance,
        burn_in_steps=config.burn_in_steps,
        span=None if config.dropout is None else config.dropout.span,
        after_window=config.after_window,
        radar_down=dropped.radar_down,
        scheduled=np.array([scan is not None for scan in dropped.sim.radar_scans]),
        alive=alive,
        vanish=vanish or None,
    )
    tracker = run.tracker
    return metrics._replace(
        run_births_per_scan=tracker.births / tracker.radar_scans if tracker.radar_scans else np.nan,
        coast_deletions=float(tracker.coast_deletions),
        tentative_drops=float(tracker.tentative_drops),
    )


def _run_outage_task(task: tuple[np.ndarray, OutageConfig, int]) -> OutageMetrics:
    """Run one (initial_states, config, seed) task; module-level to be picklable."""
    return run_outage_trial(*task)


def outage_sweep(
    initial_states: np.ndarray,
    points: Sequence[OutageConfig],
    values: Sequence[float],
    parameter: str,
    seeds: Sequence[int],
    workers: int = 1,
) -> SweepResult:
    """Run the same seeds at every configuration of a list.

    The seeds are shared by all points, so truth, target noise, detections and clutter are
    identical across the sweep and only the outage and the tracker's policy differ. Trials
    are independent and deterministic per seed, so the result does not depend on workers.

    Args:
        initial_states: Shape (n_targets, 4) starting states.
        points: One configuration per swept value.
        values: The value of the swept parameter at each point (the row labels).
        parameter: Name of the swept parameter.
        seeds: Seeds of the trials at each point.
        workers: Number of worker processes; 1 runs in this process.

    Returns:
        SweepResult whose metrics are the OutageMetrics fields, shape (n_points, n_seeds).

    Raises:
        ValueError: If points and values differ in length, or points or seeds are empty,
            or workers < 1.
    """
    if len(points) != len(values):
        raise ValueError(f"need one value per point, got {len(values)} for {len(points)}")
    metrics = _run_points(
        initial_states,
        points,
        seeds,
        workers,
        trial=_run_outage_task,
        fields=OutageMetrics._fields,
    )
    return SweepResult(parameter, np.asarray(values, dtype=float), metrics)


def duration_points(
    base: OutageConfig, sensors: tuple[str, ...], start: float, durations: Sequence[float]
) -> list[OutageConfig]:
    """One configuration per outage duration: the sensors silent from start for that long."""
    return [replace(base, dropout=SingleOutage(sensors, start, d)) for d in durations]


def flicker_points(
    base: OutageConfig,
    sensors: tuple[str, ...],
    start: float,
    span_length: float,
    period: float,
    off_times: Sequence[float],
    phase: float = 0.0,
) -> list[OutageConfig]:
    """One configuration per off time of a periodic flicker over the span."""
    return [
        replace(base, dropout=PeriodicFlicker(sensors, start, span_length, period, off, phase))
        for off in off_times
    ]


def markov_points(
    base: OutageConfig,
    sensors: tuple[str, ...],
    start: float,
    span_length: float,
    off_fractions: Sequence[float],
    mean_burst: float,
) -> list[OutageConfig]:
    """One configuration per off fraction of random bursts over the span."""
    return [
        replace(base, dropout=MarkovBursts(sensors, start, span_length, fraction, mean_burst))
        for fraction in off_fractions
    ]


def coast_points(base: OutageConfig, max_coast_times: Sequence[float]) -> list[OutageConfig]:
    """One configuration per maximum coasting time."""
    return [replace(base, max_coast_time=t) for t in max_coast_times]
