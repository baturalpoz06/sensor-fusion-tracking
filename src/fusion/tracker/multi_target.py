"""Multi-target tracker: gated global association, track birth and lifecycle."""

import math
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from fusion.angles import wrap_angle
from fusion.association.assignment import gated_assignment
from fusion.association.gating import gate_threshold, gated_costs, innovation_covariance
from fusion.dropout import STEP_TOLERANCE
from fusion.filters.ekf import ExtendedKalmanFilter
from fusion.filters.imm import IMMFilter, MotionConfig
from fusion.filters.sensitivity import SensitivityEKF, correct_output
from fusion.mtt_simulation import MttSimulation
from fusion.sensors.base import MIN_RANGE, MeasurementModel
from fusion.sensors.camera import CameraModel
from fusion.sensors.radar import RadarModel, radar_initial_estimate
from fusion.tracker.camera_bias import BiasLogEntry, CameraBiasConfig, CameraBiasEstimator
from fusion.tracker.track import Lifecycle, LifecycleConfig, Track, TrackStatus, advance

OUTAGE_POLICIES = ("unaware", "aware")
TENTATIVE_POLICIES = ("drop", "freeze")


@dataclass(frozen=True)
class TrackerConfig:
    """Parameters of the multi-target tracker.

    The last three fields only matter for a radar outage that the tracker is told about
    (step(..., radar_down=True)) and only if the policy is "aware"; with the defaults and
    no outage the tracker behaves exactly as without them.

    Attributes:
        dt: Time step in seconds.
        accel_std: Process noise acceleration std in m/s^2.
        velocity_std: Prior std of each velocity component of a newborn track in m/s.
        gate_probability: Probability mass of the chi-square gates (99% by default).
        min_range: Predicted range in meters below which a track is neither gated
            nor updated, and a measurement below which does not start a track.
        lifecycle: M-of-N confirmation and K-miss deletion parameters.
        use_camera: Whether camera scans update confirmed tracks.
        outage_policy: "unaware" treats a radar outage like any other missing data: the
            scan it is given (an empty one at a scheduled scan step) counts as a miss for
            every track. "aware" knows the radar is down: the lifecycle counters are
            frozen, no scan is processed, no track is born and the tracks coast.
        aware_tentatives: What an aware tracker does with tentative tracks during a radar
            outage: "drop" deletes them, "freeze" freezes their counters like those of
            confirmed tracks.
        max_coast_time: Seconds without any measurement update after which an aware
            tracker deletes a track during a radar outage (finite, at least dt). Tracks
            without a camera update go a whole radar period between radar updates even
            when nothing is wrong, so keep it above that period. Known limitation: the
            camera update that resets a track's clock is skipped when the bearing gates of
            two confirmed tracks overlap, so two close real targets can both be deleted
            in a radar outage longer than this.
        motion: IMM motion model of every track, or None for the plain constant-velocity
            EKF. Gating, association cost and the camera ambiguity rule then use the
            combined (mixed) estimate of each track.
        camera_bias: Global camera bearing bias estimate, or None for none. The tracks run
            on the raw camera data exactly as without it; each carries the sensitivity of its
            estimate to a constant camera bias, and the reported snapshots are corrected by
            the shared estimate (x - V b, P + V var V^T). Needs use_camera.
        record_diagnostics: Whether the tracker keeps its per-step logs (camera_log,
            mode_log, bias_log). Nothing inside the tracker reads them; turning them off must
            not change a track.
    """

    dt: float
    accel_std: float
    velocity_std: float
    gate_probability: float = 0.99
    min_range: float = 50.0
    lifecycle: LifecycleConfig = field(default_factory=LifecycleConfig)
    use_camera: bool = True
    outage_policy: str = "unaware"
    aware_tentatives: str = "drop"
    max_coast_time: float = 15.0
    motion: MotionConfig | None = None
    camera_bias: CameraBiasConfig | None = None
    record_diagnostics: bool = True

    @property
    def max_coast_steps(self) -> int:
        """max_coast_time in steps: a track is deleted after more coasting steps than this."""
        return math.floor(self.max_coast_time / self.dt + STEP_TOLERANCE)

    def __post_init__(self) -> None:
        if self.outage_policy not in OUTAGE_POLICIES:
            raise ValueError(
                f"outage_policy must be one of {OUTAGE_POLICIES}, got {self.outage_policy!r}"
            )
        if self.aware_tentatives not in TENTATIVE_POLICIES:
            raise ValueError(
                f"aware_tentatives must be one of {TENTATIVE_POLICIES}, "
                f"got {self.aware_tentatives!r}"
            )
        if self.camera_bias is not None and not self.use_camera:
            raise ValueError("camera_bias requires use_camera")
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
        if not math.isfinite(self.max_coast_time) or self.max_coast_time < self.dt:
            raise ValueError(
                f"max_coast_time must be finite and >= dt, got {self.max_coast_time} (dt {self.dt})"
            )


class CameraStepLog(NamedTuple):
    """What the camera update of one step did; a diagnostic, never read by the tracker.

    All three fields are empty at a step without a camera scan, without measurements in it, or
    without a confirmed track the camera could use.

    Attributes:
        usable: Ids of the confirmed tracks that were offered the scan (not too close to the
            camera, bearing gate not overlapping the gate of another confirmed track).
        skipped: Ids of the confirmed tracks left out because their bearing gates overlap.
        assigned: (track id, row of the camera scan) pairs of the updates that were made.
    """

    usable: tuple[int, ...] = ()
    skipped: tuple[int, ...] = ()
    assigned: tuple[tuple[int, int], ...] = ()


class ModeLogEntry(NamedTuple):
    """Mode probabilities of one IMM track after a step; a diagnostic, never read.

    Attributes:
        track_id: Identifier of the track.
        mu: Mode probabilities.
        spread_trace: Trace of the spread-of-the-means part of the combined covariance.
    """

    track_id: int
    mu: tuple[float, ...]
    spread_trace: float


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
    update the confirmed tracks whose bearing gates do not overlap the gate of
    another confirmed track (tentative tracks are not updated by the camera,
    so they neither receive nor block camera updates). The camera is bearing-only,
    so it never starts a track and never counts as a hit.

    With the "aware" outage policy and a step flagged radar_down, the radar is not
    processed at all: the tracks only coast (predict, plus camera updates), the lifecycle
    counters stay frozen and nothing is born. Afterwards tentative tracks are deleted
    (aware_tentatives="drop") and every track that has gone more than max_coast_time
    without any measurement update is deleted. These deletions happen only on such steps.

    Attributes:
        camera_skipped: Camera updates skipped because bearing gates overlapped
            (counted per track and per camera scan that had measurements).
        births: Number of tracks started so far.
        radar_scans: Number of radar scans processed so far (none during an aware outage).
        coast_deletions: Tracks deleted for coasting longer than max_coast_time.
        tentative_drops: Tentative tracks deleted at the start of an aware outage step.
        camera_log: One CameraStepLog per call of step, in order (diagnostic only; empty
            if record_diagnostics is off).
        mode_log: One tuple of ModeLogEntry (the IMM tracks) per call of step, in order
            (diagnostic only; empty without a motion model or without diagnostics).
        bias_log: One BiasLogEntry per call of step, in order (diagnostic only; empty without
            camera_bias or without diagnostics).
        bias_pairs_excluded: Radar-camera pairs left out because the radar measurement lay in
            the gate of more than one confirmed track.
        camera_offered: Camera scans offered to confirmed tracks so far (track-scan pairs).
        camera_accepted: Of those, the updates made (a measurement passed the gate).
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
        self.coast_deletions = 0
        self.tentative_drops = 0
        self.camera_log: list[CameraStepLog] = []
        self.mode_log: list[tuple[ModeLogEntry, ...]] = []
        self.bias_log: list[BiasLogEntry] = []
        self.bias_pairs_excluded = 0
        self._bias = (
            CameraBiasEstimator(
                config.camera_bias,
                float(np.sqrt(radar_model.R[1, 1])),
                float(np.sqrt(camera_model.R[0, 0])),
            )
            if config.camera_bias is not None
            else None
        )
        # Radar rows taken by the confirmed tracks of the current step, and rows lying in the
        # gate of several of them (working state of the bias pairs, reset every step).
        self._radar_rows: dict[int, int] = {}
        self._radar_shared: set[int] = set()
        self.camera_offered = 0
        self.camera_accepted = 0

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
        """Insert a track directly. For tests and scenario set-up only.

        The tracker itself never calls this method: it starts tracks only from
        unassigned radar measurements, through a private path with the same effect.

        Args:
            x: Shape (4,) initial state.
            P: Shape (4, 4) initial covariance.
            lifecycle: Initial lifecycle counters; a fresh birth by default.

        Returns:
            The new track, with the next unused id (ids are never reused).
        """
        return self._add_track(x, P, lifecycle)

    def _add_track(
        self,
        x: np.ndarray,
        P: np.ndarray,  # noqa: N803 - standard notation
        lifecycle: Lifecycle | None = None,
    ) -> Track:
        cfg = self.config
        with_bias = cfg.camera_bias is not None
        if cfg.motion is not None:
            track_filter = cfg.motion.build(cfg.dt, cfg.accel_std, x, P, sensitivity=with_bias)
        elif with_bias:
            track_filter = SensitivityEKF(dt=cfg.dt, accel_std=cfg.accel_std, x0=x, P0=P)
        else:
            track_filter = ExtendedKalmanFilter(dt=cfg.dt, accel_std=cfg.accel_std, x0=x, P0=P)
        track = Track(
            self._next_id,
            track_filter,
            lifecycle if lifecycle is not None else Lifecycle.born(cfg.lifecycle),
        )
        self._next_id += 1
        self._tracks.append(track)
        return track

    def step(
        self,
        radar_z: np.ndarray | None,
        camera_z: np.ndarray | None,
        *,
        radar_down: bool = False,
    ) -> list[TrackSnapshot]:
        """Advance the tracker by one time step.

        Args:
            radar_z: Shape (m, 2) radar measurements [range, bearing] of this
                step, or None if the radar did not scan. An empty (0, 2) array is
                a scan without detections: every track registers a miss.
            camera_z: Shape (m, 1) camera measurements [bearing], or None if the
                camera did not scan. Ignored if use_camera is False.
            radar_down: Whether the radar is known to be down at this step (scheduled
                scan step or not). Only the "aware" outage policy reads it; an "unaware"
                tracker processes radar_z as given.

        Returns:
            Snapshots of all live tracks after the step.

        Raises:
            ValueError: If a scan does not have shape (m, d) for its sensor, or if an
                aware tracker is told the radar is down but is given radar measurements.
        """
        radar = self._as_scan(radar_z, self._radar_model)
        camera = self._as_scan(camera_z, self._camera_model) if self.config.use_camera else None
        aware_outage = radar_down and self.config.outage_policy == "aware"
        if aware_outage and radar is not None and len(radar) > 0:
            raise ValueError("the radar is down, but radar measurements were given")

        self._radar_rows = {}
        self._radar_shared = set()
        if self._bias is not None:
            self._bias.predict(self.config.dt)
        for track in self._tracks:
            track.filter.predict()
            track.coast_steps += 1
        if radar is not None and not aware_outage:
            self._radar_scan(radar)
        entry = self._camera_scan(camera) if camera is not None else CameraStepLog()
        if self.config.record_diagnostics:
            self.camera_log.append(entry)
        self.camera_offered += len(entry.usable)
        self.camera_accepted += len(entry.assigned)
        if self._bias is not None:
            self._update_bias(radar, camera, entry)
        if aware_outage:
            self._end_of_outage_step()
        if self.config.record_diagnostics and self.config.motion is not None:
            self.mode_log.append(
                tuple(
                    ModeLogEntry(
                        t.track_id, tuple(float(m) for m in t.filter.mu), t.filter.spread_trace
                    )
                    for t in self._tracks
                    if isinstance(t.filter, IMMFilter)
                )
            )
        if self._bias is not None and self.config.record_diagnostics:
            self.bias_log.append(self._bias.log_entry())
        return [self._snapshot(t) for t in self._tracks]

    def _snapshot(self, track: Track) -> TrackSnapshot:
        """Copy of a track's estimate, corrected by the bias estimate if there is one."""
        x, cov = track.filter.x.copy(), track.filter.P.copy()
        if self._bias is not None:
            bias, variance = self._bias.estimate
            if bias != 0.0 or variance != 0.0:
                x, cov = correct_output(x, cov, track.filter.V, bias, variance)
        return TrackSnapshot(track.track_id, track.status, x, cov)

    def _update_bias(
        self, radar: np.ndarray | None, camera: np.ndarray | None, entry: CameraStepLog
    ) -> None:
        """Feed the bias estimate the radar-camera bearing pairs of this step.

        A pair is a track that was confirmed when this step's radar scan was associated and
        took a radar measurement and a camera measurement now; pairs whose radar measurement
        lies in the gate of several confirmed tracks are left out (the match may be wrong).
        """
        deltas = []
        if radar is not None and camera is not None and self._radar_rows:
            camera_rows = dict(entry.assigned)
            for track_id in sorted(self._radar_rows):
                row = self._radar_rows[track_id]
                if track_id not in camera_rows:
                    continue
                if row in self._radar_shared:
                    self.bias_pairs_excluded += 1
                    continue
                deltas.append(float(camera[camera_rows[track_id], 0] - radar[row, 1]))
        self._bias.update(deltas)

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
    ) -> tuple[set[int], list[tuple[int, int]], set[int]]:
        """Assign tracks to the still free measurements, update them; return the outcome.

        Measurements taken by a track are cleared in `free`.

        Returns:
            The ids of the tracks that were updated, the (track id, row of z) pairs, and the
            rows of z that lie in the gate of more than one of the tracks.
        """
        columns = np.flatnonzero(free)
        if not tracks or columns.size == 0:
            return set(), [], set()
        costs = gated_costs(
            [t.filter for t in tracks], z[columns], model, gate, self.config.min_range
        )
        shared = {int(columns[c]) for c in np.flatnonzero(costs.in_gate.sum(axis=0) > 1)}
        assignment = gated_assignment(costs.cost, costs.in_gate)
        hits = set()
        pairs = []
        for row, col in assignment.pairs:
            tracks[row].filter.update(z[columns[col]], model)
            tracks[row].coast_steps = 0
            free[columns[col]] = False
            hits.add(tracks[row].track_id)
            pairs.append((tracks[row].track_id, int(columns[col])))
        return hits, pairs, shared

    def _end_of_outage_step(self) -> None:
        """Delete what an aware tracker gives up on during a radar outage step.

        Runs after the camera update of the step, so a camera update at this step has
        already reset the coast clock of its track. Tentative tracks go first (when they
        are dropped), then every track with more than max_coast_steps steps without any
        measurement update.
        """
        drop_tentative = self.config.aware_tentatives == "drop"
        limit = self.config.max_coast_steps
        kept = []
        for track in self._tracks:
            if drop_tentative and track.status is TrackStatus.TENTATIVE:
                self.tentative_drops += 1
            elif track.coast_steps > limit:
                self.coast_deletions += 1
            else:
                kept.append(track)
        self._tracks = kept

    def _radar_scan(self, z: np.ndarray) -> None:
        self.radar_scans += 1
        free = np.ones(len(z), dtype=bool)
        existing = list(self._tracks)
        hits: set[int] = set()
        for status in (TrackStatus.CONFIRMED, TrackStatus.TENTATIVE):
            group = [t for t in existing if t.status is status]
            group_hits, group_pairs, shared = self._associate(
                group, z, free, self._radar_model, self._radar_gate
            )
            hits |= group_hits
            if status is TrackStatus.CONFIRMED:
                self._radar_rows = dict(group_pairs)
                self._radar_shared = shared

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
            self._add_track(x0, p0)
            self.births += 1

    def _ambiguous_camera_tracks(self, tracks: list[Track]) -> set[int]:
        """Indices of tracks whose bearing gate overlaps the gate of another one in the list.

        Tracks i and j overlap if |wrap(theta_i - theta_j)| < sqrt(gamma) * (s_i + s_j),
        with s the std of each track's predicted bearing innovation.
        """
        bearings = np.array([self._camera_model.h(t.filter.x)[0] for t in tracks])
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

    def _camera_scan(self, z: np.ndarray) -> CameraStepLog:
        guard = max(self.config.min_range, MIN_RANGE)
        # The guard is on the distance from the camera the model assumes: close to it the
        # bearing Jacobian explodes. At the default position this is the range from the radar.
        cx, cy = self._camera_model.position
        confirmed = [
            t
            for t in self._tracks
            if t.status is TrackStatus.CONFIRMED
            and np.hypot(t.filter.x[0] - cx, t.filter.x[1] - cy) >= guard
        ]
        if not confirmed or len(z) == 0:
            return CameraStepLog()
        ambiguous = self._ambiguous_camera_tracks(confirmed)
        self.camera_skipped += len(ambiguous)
        usable = [t for i, t in enumerate(confirmed) if i not in ambiguous]
        _, pairs, _ = self._associate(
            usable, z, np.ones(len(z), dtype=bool), self._camera_model, self._camera_gate
        )
        return CameraStepLog(
            tuple(t.track_id for t in usable),
            tuple(t.track_id for i, t in enumerate(confirmed) if i in ambiguous),
            tuple(pairs),
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
    radar_down: np.ndarray | None = None,
) -> TrackerRun:
    """Run the tracker over every step of a simulation.

    Args:
        sim: Simulated truth and scans; the tracker reads only the measurements.
        config: Tracker parameters.
        radar_model: Radar measurement model.
        camera_model: Camera measurement model; may be None if the camera is unused.
        radar_down: Shape (n_steps,) boolean, True at the steps in which the radar is known
            to be down (see MultiTargetTracker.step); None means no outage.

    Returns:
        TrackerRun with the per-step snapshots and the final tracker.

    Raises:
        ValueError: If radar_down does not have one entry per step.
    """
    n_steps = len(sim.radar_scans)
    if radar_down is None:
        radar_down = np.zeros(n_steps, dtype=bool)
    elif np.shape(radar_down) != (n_steps,):
        raise ValueError(f"radar_down must have shape ({n_steps},), got {np.shape(radar_down)}")
    tracker = MultiTargetTracker(config, radar_model, camera_model)
    history = []
    for radar_scan, camera_scan, down in zip(
        sim.radar_scans, sim.camera_scans, radar_down, strict=True
    ):
        history.append(
            tracker.step(
                None if radar_scan is None else radar_scan.z,
                camera_scan.z,
                radar_down=bool(down),
            )
        )
    return TrackerRun(history, tracker)
