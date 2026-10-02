"""Tests for the multi-target tracker."""

from dataclasses import replace

import numpy as np
import pytest

from fusion.association.gating import gate_threshold, gated_costs
from fusion.experiment import SCENARIOS as SINGLE_TARGET_SCENARIOS
from fusion.experiment import FusionConfig, make_rngs, simulate
from fusion.filters.ekf import ExtendedKalmanFilter
from fusion.mtt_experiment import SCENARIOS
from fusion.mtt_simulation import SimulationConfig, make_mtt_rngs, simulate_mtt
from fusion.sensors.camera import CameraModel
from fusion.sensors.radar import RadarModel, radar_initial_estimate
from fusion.tracker.multi_target import (
    MultiTargetTracker,
    TrackerConfig,
    run_multi_target_tracking,
)
from fusion.tracker.track import Lifecycle, LifecycleConfig, TrackStatus
from fusion.tracking import run_tracking_with_covariance

DT = 0.1
ACCEL_STD = 0.5
VELOCITY_STD = 20.0
RADAR = RadarModel(range_std=5.0, bearing_std=float(np.deg2rad(2.0)))
CAMERA = CameraModel(bearing_std=float(np.deg2rad(0.1)))
TENTATIVE, CONFIRMED = TrackStatus.TENTATIVE, TrackStatus.CONFIRMED
CONFIRMED_STATE = Lifecycle(CONFIRMED, 3, 3, 0)
EMPTY_RADAR = np.zeros((0, 2))


def make_config(**overrides) -> TrackerConfig:
    return TrackerConfig(dt=DT, accel_std=ACCEL_STD, velocity_std=VELOCITY_STD, **overrides)


def make_tracker(use_camera=False, **overrides) -> MultiTargetTracker:
    config = make_config(use_camera=use_camera, **overrides)
    return MultiTargetTracker(config, RADAR, CAMERA if use_camera else None)


def at_bearing(range_: float, bearing: float) -> np.ndarray:
    return np.array([range_ * np.cos(bearing), range_ * np.sin(bearing), 0.0, 0.0])


def tight(sigma=5.0) -> np.ndarray:
    return np.diag([sigma**2, sigma**2, 1.0, 1.0])


# --- configuration -------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dt": 0.0},
        {"accel_std": -1.0},
        {"velocity_std": 0.0},
        {"gate_probability": 1.0},
        {"gate_probability": 0.0},
        {"min_range": -1.0},
    ],
)
def test_tracker_config_validation(kwargs):
    base = {"dt": DT, "accel_std": ACCEL_STD, "velocity_std": VELOCITY_STD}
    with pytest.raises(ValueError):
        TrackerConfig(**{**base, **kwargs})


def test_camera_use_requires_a_camera_model():
    with pytest.raises(ValueError, match="camera_model"):
        MultiTargetTracker(make_config(use_camera=True), RADAR, None)


def test_scan_shapes_are_validated_before_any_state_changes():
    tracker = make_tracker()
    tracker.add_track(at_bearing(1000.0, 0.3), tight())
    p_before = tracker.tracks[0].filter.P.copy()
    with pytest.raises(ValueError, match="shape"):
        tracker.step(np.zeros((2, 3)), None)
    np.testing.assert_array_equal(tracker.tracks[0].filter.P, p_before)
    tracker.step(np.array([]), None)  # an empty 1-D array is accepted as no detections
    assert tracker.radar_scans == 1


# --- birth ---------------------------------------------------------------------------


def test_unassigned_radar_measurements_start_tentative_tracks():
    tracker = make_tracker()
    z = np.array([[1000.0, 0.3], [2000.0, -1.0]])
    snapshots = tracker.step(z, None)
    assert [s.track_id for s in snapshots] == [0, 1]
    assert all(s.status is TENTATIVE for s in snapshots)
    range_std, bearing_std = np.sqrt(np.diag(RADAR.R))
    for snapshot, measurement in zip(snapshots, z, strict=True):
        x0, p0 = radar_initial_estimate(measurement, range_std, bearing_std, VELOCITY_STD)
        np.testing.assert_array_equal(snapshot.x, x0)
        np.testing.assert_array_equal(snapshot.P, p0)
    assert tracker.births == 2


def test_measurements_that_cannot_start_a_track_do_not():
    tracker = make_tracker(min_range=50.0)
    z = np.array([[49.0, 0.1], [-10.0, 0.1], [np.nan, 0.1], [100.0, np.nan], [50.0, 0.1]])
    snapshots = tracker.step(z, None)
    assert len(snapshots) == 1
    assert np.hypot(*snapshots[0].x[:2]) == pytest.approx(50.0)


def test_camera_measurements_never_start_a_track():
    tracker = make_tracker(use_camera=True)
    assert tracker.step(None, np.array([[0.3], [1.0]])) == []
    assert tracker.step(EMPTY_RADAR, np.array([[0.3]])) == []
    assert tracker.births == 0


def test_a_measurement_outside_every_gate_starts_a_new_track_and_the_old_one_misses():
    tracker = make_tracker()
    track = tracker.add_track(at_bearing(1000.0, 0.3), tight(), CONFIRMED_STATE)
    far = np.array([[1000.0, 2.0]])
    snapshots = tracker.step(far, None)
    assert len(snapshots) == 2
    assert track.lifecycle == Lifecycle(CONFIRMED, 3, 4, 1)  # not updated: a miss
    assert snapshots[0].status is CONFIRMED and snapshots[1].status is TENTATIVE
    np.testing.assert_array_equal(snapshots[0].x, at_bearing(1000.0, 0.3))  # v = 0: prediction


# --- lifecycle inside the tracker ----------------------------------------------------


def test_an_empty_radar_scan_is_a_miss_and_the_third_tentative_miss_deletes():
    tracker = make_tracker()
    tracker.step(np.array([[1000.0, 0.3]]), None)
    states = []
    for _ in range(3):
        tracker.step(EMPTY_RADAR, None)
        states.append(tracker.tracks[0].lifecycle if tracker.tracks else None)
    assert states[0] == Lifecycle(TENTATIVE, 1, 2, 1)
    assert states[1] == Lifecycle(TENTATIVE, 1, 3, 2)
    assert states[2] is None  # third miss of 3-of-5: deleted and dropped
    assert tracker.tracks == []


def test_no_scan_changes_neither_counters_nor_births():
    tracker = make_tracker()
    tracker.step(np.array([[1000.0, 0.3]]), None)
    before = tracker.tracks[0].lifecycle
    trace_before = np.trace(tracker.tracks[0].filter.P)
    for _ in range(9):
        tracker.step(None, None)
    assert tracker.tracks[0].lifecycle == before
    assert np.trace(tracker.tracks[0].filter.P) > trace_before  # still predicted
    assert tracker.radar_scans == 1


def test_a_stationary_target_is_confirmed_on_its_third_hit():
    tracker = make_tracker()
    z = np.array([[1000.0, 0.3]])
    statuses = [tracker.step(z, None)[0].status for _ in range(4)]
    assert statuses == [TENTATIVE, TENTATIVE, CONFIRMED, CONFIRMED]
    assert tracker.births == 1


def test_confirmed_track_survives_k_missed_scans_and_is_dropped_after_k_plus_one():
    lifecycle = LifecycleConfig(max_misses=2)
    tracker = make_tracker(lifecycle=lifecycle)
    tracker.add_track(at_bearing(1000.0, 0.3), tight(), CONFIRMED_STATE)
    alive = [len(tracker.step(EMPTY_RADAR, None)) for _ in range(3)]
    assert alive == [1, 1, 0]


def test_track_ids_are_unique_increasing_and_never_reused():
    tracker = make_tracker()
    seen = []
    tracker.step(np.array([[1000.0, 0.3]]), None)
    seen.append(tracker.tracks[0].track_id)
    for _ in range(3):
        tracker.step(EMPTY_RADAR, None)  # the tentative track dies
    assert tracker.tracks == []
    tracker.step(np.array([[1000.0, 0.3], [1500.0, 1.0]]), None)
    seen += [t.track_id for t in tracker.tracks]
    assert seen == [0, 1, 2]
    assert tracker.births == 3


def test_tracks_property_returns_a_copy():
    tracker = make_tracker()
    tracker.step(np.array([[1000.0, 0.3]]), None)
    tracker.tracks.clear()
    assert len(tracker.tracks) == 1


# --- association order ---------------------------------------------------------------


def test_confirmed_track_wins_a_contested_measurement_over_a_cheaper_tentative_track():
    tracker = make_tracker()
    confirmed = tracker.add_track(at_bearing(1000.0, 0.0), tight(5.0), CONFIRMED_STATE)
    wide = np.diag([400.0, 400.0, 1.0, 1.0])
    tentative = tracker.add_track(at_bearing(1030.0, 0.0), wide)
    z = np.array([[1018.0, 0.0]])

    # Precondition: both tracks gate the measurement and the tentative one is cheaper,
    # so a single joint assignment would hand it to the tentative track.
    predicted = []
    for track in (confirmed, tentative):
        reference = ExtendedKalmanFilter(DT, ACCEL_STD, track.filter.x, track.filter.P)
        reference.predict()
        predicted.append(reference)
    costs = gated_costs(predicted, z, RADAR, gate_threshold(2, 0.99), 50.0)
    assert costs.in_gate.all()
    assert costs.cost[1, 0] < costs.cost[0, 0]

    x_before = confirmed.filter.x.copy()
    tracker.step(z, None)
    assert not np.array_equal(confirmed.filter.x, x_before)  # confirmed track took it
    assert confirmed.lifecycle.hits == 4
    assert tentative.lifecycle == Lifecycle(TENTATIVE, 1, 2, 1)  # tentative track missed
    assert tracker.births == 0  # and the measurement did not also start a track


def test_tentative_tracks_take_only_the_measurements_left_by_confirmed_ones():
    tracker = make_tracker()
    confirmed = tracker.add_track(at_bearing(1000.0, 0.5), tight(), CONFIRMED_STATE)
    tentative = tracker.add_track(at_bearing(2000.0, -1.0), tight())
    z = np.array([[2002.0, -1.0], [1001.0, 0.5]])
    tracker.step(z, None)
    assert confirmed.lifecycle.hits == 4 and tentative.lifecycle.hits == 2
    assert tracker.births == 0


# --- camera --------------------------------------------------------------------------


def test_camera_updates_confirmed_tracks_without_touching_the_lifecycle():
    tracker = make_tracker(use_camera=True)
    track = tracker.add_track(at_bearing(1000.0, 0.3), tight(), CONFIRMED_STATE)
    tracker.step(None, np.array([[0.3 + 0.003]]))
    assert track.filter.x[1] != 1000.0 * np.sin(0.3)  # moved by the bearing update
    assert track.lifecycle == CONFIRMED_STATE  # counters unchanged by a camera hit
    tracker.step(None, np.zeros((0, 1)))
    assert track.lifecycle == CONFIRMED_STATE  # and by a camera miss


def test_camera_never_updates_tentative_tracks():
    tracker = make_tracker(use_camera=True)
    track = tracker.add_track(at_bearing(1000.0, 0.3), tight())
    reference = ExtendedKalmanFilter(DT, ACCEL_STD, track.filter.x, track.filter.P)
    reference.predict()
    tracker.step(None, np.array([[0.3 + 0.002]]))
    np.testing.assert_array_equal(track.filter.x, reference.x)
    np.testing.assert_array_equal(track.filter.P, reference.P)


def test_camera_is_ignored_when_disabled():
    tracker = make_tracker(use_camera=False)
    track = tracker.add_track(at_bearing(1000.0, 0.3), tight(), CONFIRMED_STATE)
    tracker.step(None, np.array([[0.31]]))
    np.testing.assert_array_equal(track.filter.x, at_bearing(1000.0, 0.3))


def test_camera_update_is_skipped_for_tracks_with_overlapping_bearing_gates():
    tracker = make_tracker(use_camera=True)
    a = tracker.add_track(at_bearing(1000.0, 0.0), tight(), CONFIRMED_STATE)
    b = tracker.add_track(at_bearing(1500.0, 0.0033), tight(), CONFIRMED_STATE)
    z = np.array([[0.002]])  # inside the bearing gate of both tracks
    xa, xb = a.filter.x.copy(), b.filter.x.copy()
    tracker.step(None, z)
    np.testing.assert_array_equal(a.filter.x, xa)
    np.testing.assert_array_equal(b.filter.x, xb)
    assert tracker.camera_skipped == 2


def test_camera_updates_tracks_with_well_separated_bearings():
    tracker = make_tracker(use_camera=True)
    a = tracker.add_track(at_bearing(1000.0, 0.0), tight(), CONFIRMED_STATE)
    b = tracker.add_track(at_bearing(1000.0, 1.0), tight(), CONFIRMED_STATE)
    xa, xb = a.filter.x.copy(), b.filter.x.copy()
    tracker.step(None, np.array([[0.002]]))
    assert not np.array_equal(a.filter.x, xa)  # the track whose gate holds the measurement
    np.testing.assert_array_equal(b.filter.x, xb)
    assert tracker.camera_skipped == 0


def test_camera_ambiguity_is_detected_across_the_minus_x_axis():
    delta = 0.005
    tracker = make_tracker(use_camera=True)
    a = tracker.add_track(at_bearing(1000.0, np.pi - delta), tight(), CONFIRMED_STATE)
    b = tracker.add_track(at_bearing(1200.0, -np.pi + delta), tight(), CONFIRMED_STATE)
    # Unwrapped, the bearings are almost 2 pi apart and would look unrelated.
    assert (
        abs(np.arctan2(a.filter.x[1], a.filter.x[0]) - np.arctan2(b.filter.x[1], b.filter.x[0])) > 6
    )
    xa, xb = a.filter.x.copy(), b.filter.x.copy()
    tracker.step(None, np.array([[np.pi]]))
    np.testing.assert_array_equal(a.filter.x, xa)
    np.testing.assert_array_equal(b.filter.x, xb)
    assert tracker.camera_skipped == 2


def test_skipped_updates_are_only_counted_for_camera_scans_with_measurements():
    tracker = make_tracker(use_camera=True)
    tracker.add_track(at_bearing(1000.0, 0.0), tight(), CONFIRMED_STATE)
    tracker.add_track(at_bearing(1500.0, 0.0033), tight(), CONFIRMED_STATE)
    tracker.step(None, np.zeros((0, 1)))
    assert tracker.camera_skipped == 0


# --- Phase 4 regression --------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_single_target_without_clutter_reproduces_the_phase_4_radar_only_filter_bitwise(seed):
    config = FusionConfig()
    sim = simulate(SINGLE_TARGET_SCENARIOS["near"], config, make_rngs(seed))
    reference = run_tracking_with_covariance(
        sim.truth,
        dt=config.dt,
        radar_every=config.radar_every,
        radar_z=sim.radar_z,
        radar_model=RadarModel(config.radar_range_std, config.radar_bearing_std),
        accel_std=config.accel_std,
        velocity_std=config.velocity_std,
        use_camera=False,
    )
    tracker = MultiTargetTracker(
        TrackerConfig(
            dt=config.dt,
            accel_std=config.accel_std,
            velocity_std=config.velocity_std,
            gate_probability=1.0 - 1e-12,  # the gate must not reject any measurement
            use_camera=False,
        ),
        RadarModel(config.radar_range_std, config.radar_bearing_std),
        None,
    )
    for k in range(len(sim.truth)):
        z = sim.radar_z[k : k + 1] if k % config.radar_every == 0 else None
        snapshots = tracker.step(z, None)
        assert len(snapshots) == 1
        np.testing.assert_array_equal(snapshots[0].x, reference.estimates[k])
        np.testing.assert_array_equal(snapshots[0].P, reference.covariances[k])
    assert tracker.births == 1


# --- end to end ----------------------------------------------------------------------


def test_separated_targets_without_clutter_are_tracked_with_stable_ids():
    scene = SimulationConfig(
        radar_pd=1.0, camera_pd=1.0, radar_clutter_rate=0.0, camera_clutter_rate=0.0
    )
    sim = simulate_mtt(SCENARIOS["separated"], scene, make_mtt_rngs(0))
    run = run_multi_target_tracking(
        sim,
        make_config(use_camera=True),
        RadarModel(scene.radar_range_std, scene.radar_bearing_std),
        CameraModel(scene.camera_bearing_std),
    )
    assert len(run.history) == scene.n_steps + 1
    final = run.history[-1]
    assert len(final) == 3 and all(s.status is CONFIRMED for s in final)
    assert {s.track_id for s in final} == {s.track_id for s in run.history[300]} == {0, 1, 2}
    for truth in sim.truth[:, -1, :2]:
        assert min(np.hypot(*(s.x[:2] - truth)) for s in final) < 60.0
    assert run.tracker.births == 3
    assert run.tracker.radar_scans == scene.n_steps // scene.radar_every + 1


def test_run_tracking_does_not_read_the_scan_origins():
    scene = replace(SimulationConfig(), radar_clutter_rate=3.0)
    sim = simulate_mtt(SCENARIOS["separated"], scene, make_mtt_rngs(4))
    scrambled = sim._replace(
        radar_scans=[
            None if s is None else s._replace(origin=np.full_like(s.origin, 7))
            for s in sim.radar_scans
        ],
        camera_scans=[s._replace(origin=np.full_like(s.origin, 7)) for s in sim.camera_scans],
    )
    models = (RadarModel(scene.radar_range_std, scene.radar_bearing_std), CameraModel(0.002))
    a = run_multi_target_tracking(sim, make_config(), *models)
    b = run_multi_target_tracking(scrambled, make_config(), *models)
    for sa, sb in zip(a.history, b.history, strict=True):
        assert [s.track_id for s in sa] == [s.track_id for s in sb]
        for ta, tb in zip(sa, sb, strict=True):
            np.testing.assert_array_equal(ta.x, tb.x)


def test_births_do_not_go_through_the_public_add_track_hook(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("add_track is for tests and scenario set-up only")

    monkeypatch.setattr(MultiTargetTracker, "add_track", forbidden)
    tracker = make_tracker()
    tracker.step(np.array([[1000.0, 0.3], [2000.0, -1.0]]), None)
    assert [t.track_id for t in tracker.tracks] == [0, 1]


def test_add_track_documents_that_it_is_only_a_test_hook():
    assert "tests and scenario set-up only" in MultiTargetTracker.add_track.__doc__
