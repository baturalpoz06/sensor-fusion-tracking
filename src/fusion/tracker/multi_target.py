"""Multi-target tracker: gated global association, track birth and lifecycle."""

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from fusion.angles import wrap_angle
from fusion.association.assignment import gated_assignment
from fusion.association.gating import gate_threshold, gated_costs, innovation_covariance
from fusion.filters.ekf import ExtendedKalmanFilter
from fusion.mtt_simulation import MttSimulation
from fusion.sensors.base import MIN_RANGE, MeasurementModel
from fusion.sensors.camera import CameraModel
from fusion.sensors.radar import RadarModel, radar_initial_estimate
from fusion.tracker.track import Lifecycle, LifecycleConfig, Track, TrackStatus, advance


@dataclass(frozen=True)
class TrackerConfig:
    """Parameters of the multi-target tracker.

    Attributes:
        dt: Time step in seconds.
        accel_std: Process noise acceleration std in m/s^2.
        velocity_std: Prior std of each velocity component of a newborn track in m/s.
        gate_probability: Probability mass of the chi-square gates (99% by default).
        min_range: Predicted range in meters below which a track is neither gated
            nor updated, and a measurement below which does not start a track.
        lifecycle: M-of-N confirmation and K-miss deletion parameters.
        use_camera: Whether camera scans update confirmed tracks.
    """

    dt: float
    accel_std: float
    velocity_std: float
    gate_probability: float = 0.99
    min_range: float = 50.0
    lifecycle: LifecycleConfig = field(default_factory=LifecycleConfig)
    use_camera: bool = True

    def __post_init__(self) -> None:
        if self.dt <= 0.0:
            raise ValueError(f"dt must be > 0, got {self.dt}")
        if self.accel_std < 0.0:
            raise ValueError(f"accel_std must be >= 0, got {self.accel_std}")
        if self.velocity_std <= 0.0:
            raise ValueError(f"velocity_std must be > 0, got {self.velocity_std}")
        if not 0.0 < self.gate_probability < 1.0:
            raise ValueError(f"gate_probability must be in (0, 1), got {self.gate_probability}")
        if self.min_range < 0.0:
            raise ValueError(f"min_range must be >= 0, got {self.min_range}")


class TrackSnapshot(NamedTuple):
    """Copy of one live track after a step.

    Attributes:
        track_id: Identifier of the track.
        status: TENTATIVE or CONFIRMED (deleted tracks are not reported).
        x: Shape (4,) state estimate.
        P: Shape (4, 4) state covariance.
    """

    track_id: int
    status: TrackStatus
    x: np.ndarray
    P: np.ndarray  # noqa: N815 - standard notation


class MultiTargetTracker:
    """Tracks an unknown number of targets from radar and camera scans.

    Per step: predict every track; then, if the radar scanned, associate confirmed
    tracks first and tentative tracks with the leftovers (a global assignment
    inside each group), update the lifecycle counters, and start a tentative
    track from every leftover radar measurement; then, if the camera scanned,
    update the confirmed tracks whose bearing gates do not overlap another
    track's gate. The camera is bearing-only, so it never starts a track and
    never counts as a hit.

    Attributes:
        camera_skipped: Camera updates skipped because bearing gates overlapped
            (counted per track and per camera scan that had measurements).
        births: Number of tracks started so far.
        radar_scans: Number of radar scans processed so far.
    """

    def __init__(
        self,
        config: TrackerConfig,
        radar_model: RadarModel,
        camera_model: CameraModel | None,
    ) -> None:
        """Set up an empty tracker.

        Raises:
            ValueError: If the camera is enabled but no camera model is given.
        """
        if config.use_camera and camera_model is None:
            raise ValueError("use_camera requires a camera_model")
        self.config = config
        self._radar_model = radar_model
        self._camera_model = camera_model
        self._radar_gate = gate_threshold(radar_model.R.shape[0], config.gate_probability)
        self._camera_gate = (
            gate_threshold(camera_model.R.shape[0], config.gate_probability)
            if camera_model is not None
            else float("nan")
        )
        self._tracks: list[Track] = []
        self._next_id = 0
        self.camera_skipped = 0
        self.births = 0
        self.radar_scans = 0

    @property
    def tracks(self) -> list[Track]:
        """Live (tentative and confirmed) tracks, in order of birth."""
        return list(self._tracks)

    def add_track(
        self,
        x: np.ndarray,
        P: np.ndarray,  # noqa: N803 - standard notation
        lifecycle: Lifecycle | None = None,
    ) -> Track:
        """Insert a track with the given estimate; used for births and to set up scenarios.

        Args:
            x: Shape (4,) initial state.
            P: Shape (4, 4) initial covariance.
            lifecycle: Initial lifecycle counters; a fresh birth by default.

        Returns:
            The new track, with the next unused id (ids are never reused).
        """
        cfg = self.config
        track = Track(
            self._next_id,
            ExtendedKalmanFilter(dt=cfg.dt, accel_std=cfg.accel_std, x0=x, P0=P),
            lifecycle if lifecycle is not None else Lifecycle.born(cfg.lifecycle),
        )
        self._next_id += 1
        self._tracks.append(track)
        return track

    def step(self, radar_z: np.ndarray | None, camera_z: np.ndarray | None) -> list[TrackSnapshot]:
        """Advance the tracker by one time step.

        Args:
            radar_z: Shape (m, 2) radar measurements [range, bearing] of this
                step, or None if the radar did not scan. An empty (0, 2) array is
                a scan without detections: every track registers a miss.
            camera_z: Shape (m, 1) camera measurements [bearing], or None if the
                camera did not scan. Ignored if use_camera is False.

        Returns:
            Snapshots of all live tracks after the step.

        Raises:
            ValueError: If a scan does not have shape (m, d) for its sensor.
        """
        radar = self._as_scan(radar_z, self._radar_model)
        camera = self._as_scan(camera_z, self._camera_model) if self.config.use_camera else None

        for track in self._tracks:
            track.filter.predict()
        if radar is not None:
            self._radar_scan(radar)
        if camera is not None:
            self._camera_scan(camera)
        return [
            TrackSnapshot(t.track_id, t.status, t.filter.x.copy(), t.filter.P.copy())
            for t in self._tracks
        ]

    @staticmethod
    def _as_scan(z: np.ndarray | None, model: MeasurementModel | None) -> np.ndarray | None:
        if z is None:
            return None
        m = model.R.shape[0]
        z = np.asarray(z, dtype=float)
        if z.size == 0 and z.ndim < 2:
            z = z.reshape(0, m)
        if z.ndim != 2 or z.shape[1] != m:
            raise ValueError(f"a scan must have shape (n, {m}), got {z.shape}")
        return z

    def _associate(
        self,
        tracks: list[Track],
        z: np.ndarray,
        free: np.ndarray,
        model: MeasurementModel,
        gate: float,
    ) -> set[int]:
        """Assign tracks to the still free measurements, update them, return hit track ids.

        Measurements taken by a track are cleared in `free`.
        """
        columns = np.flatnonzero(free)
        if not tracks or columns.size == 0:
            return set()
        costs = gated_costs(
            [t.filter for t in tracks], z[columns], model, gate, self.config.min_range
        )
        assignment = gated_assignment(costs.cost, costs.in_gate)
        hits = set()
        for row, col in assignment.pairs:
            tracks[row].filter.update(z[columns[col]], model)
            free[columns[col]] = False
            hits.add(tracks[row].track_id)
        return hits

    def _radar_scan(self, z: np.ndarray) -> None:
        self.radar_scans += 1
        free = np.ones(len(z), dtype=bool)
        existing = list(self._tracks)
        hits: set[int] = set()
        for status in (TrackStatus.CONFIRMED, TrackStatus.TENTATIVE):
            group = [t for t in existing if t.status is status]
            hits |= self._associate(group, z, free, self._radar_model, self._radar_gate)

        for track in existing:
            track.lifecycle = advance(
                track.lifecycle, track.track_id in hits, self.config.lifecycle
            )
        self._tracks = [t for t in self._tracks if t.status is not TrackStatus.DELETED]

        range_std, bearing_std = np.sqrt(np.diag(self._radar_model.R))
        for j in np.flatnonzero(free):
            measurement = z[j]
            if not np.isfinite(measurement).all() or measurement[0] < self.config.min_range:
                continue
            x0, p0 = radar_initial_estimate(
                measurement, range_std, bearing_std, self.config.velocity_std
            )
            self.add_track(x0, p0)
            self.births += 1

    def _ambiguous_camera_tracks(self, tracks: list[Track]) -> set[int]:
        """Indices of tracks whose bearing gate overlaps the gate of another track.

        Tracks i and j overlap if |wrap(theta_i - theta_j)| < sqrt(gamma) * (s_i + s_j),
        with s the std of each track's predicted bearing innovation.
        """
        bearings = np.array([np.arctan2(t.filter.x[1], t.filter.x[0]) for t in tracks])
        sigma = np.array(
            [
                np.sqrt(innovation_covariance(t.filter.x, t.filter.P, self._camera_model)[0, 0])
                for t in tracks
            ]
        )
        half_width = np.sqrt(self._camera_gate) * sigma
        gap = np.abs(wrap_angle(bearings[:, None] - bearings[None, :]))
        overlap = gap < half_width[:, None] + half_width[None, :]
        np.fill_diagonal(overlap, False)
        return {int(i) for i in np.flatnonzero(overlap.any(axis=1))}

    def _camera_scan(self, z: np.ndarray) -> None:
        guard = max(self.config.min_range, MIN_RANGE)
        confirmed = [
            t
            for t in self._tracks
            if t.status is TrackStatus.CONFIRMED and np.hypot(*t.filter.x[:2]) >= guard
        ]
        if not confirmed or len(z) == 0:
            return
        ambiguous = self._ambiguous_camera_tracks(confirmed)
        self.camera_skipped += len(ambiguous)
        usable = [t for i, t in enumerate(confirmed) if i not in ambiguous]
        self._associate(
            usable, z, np.ones(len(z), dtype=bool), self._camera_model, self._camera_gate
        )


class TrackerRun(NamedTuple):
    """Result of running the tracker over a simulation.

    Attributes:
        history: history[k] holds the snapshots of all live tracks after step k.
        tracker: The tracker after the last step, carrying the diagnostic counters.
    """

    history: list[list[TrackSnapshot]]
    tracker: MultiTargetTracker


def run_multi_target_tracking(
    sim: MttSimulation,
    config: TrackerConfig,
    radar_model: RadarModel,
    camera_model: CameraModel | None,
) -> TrackerRun:
    """Run the tracker over every step of a simulation.

    Args:
        sim: Simulated truth and scans; the tracker reads only the measurements.
        config: Tracker parameters.
        radar_model: Radar measurement model.
        camera_model: Camera measurement model; may be None if the camera is unused.

    Returns:
        TrackerRun with the per-step snapshots and the final tracker.
    """
    tracker = MultiTargetTracker(config, radar_model, camera_model)
    history = []
    for radar_scan, camera_scan in zip(sim.radar_scans, sim.camera_scans, strict=True):
        history.append(
            tracker.step(
                None if radar_scan is None else radar_scan.z,
                camera_scan.z,
            )
        )
    return TrackerRun(history, tracker)
