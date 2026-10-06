"""Tests of the tracker with the global camera-bias estimate.

Each test docstring names the defect that makes it fail.
"""

from dataclasses import replace

import numpy as np
import pytest

from fusion.filters.imm import ModeSpec, MotionConfig
from fusion.maneuvers import TruthDisturbance, Turn
from fusion.mtt_experiment import SCENARIOS, MttConfig
from fusion.mtt_metrics import match_tracks
from fusion.mtt_simulation import make_mtt_rngs, simulate_mtt
from fusion.robustness_metrics import matched_errors
from fusion.sensors.camera import CameraModel
from fusion.sensors.radar import RadarModel
from fusion.tracker.camera_bias import CameraBiasConfig
from fusion.tracker.multi_target import (
    MultiTargetTracker,
    TrackerConfig,
    run_multi_target_tracking,
)
from fusion.tracker.track import Lifecycle, TrackStatus
from tests.test_imm_tracker import history_digest

DT = 0.1
DEG = float(np.pi / 180.0)
RADAR = RadarModel(range_std=5.0, bearing_std=float(np.deg2rad(2.0)))
CAMERA = CameraModel(bearing_std=float(np.deg2rad(0.1)))
PAIR_VAR = RADAR.R[1, 1] + CAMERA.R[0, 0]
SLAB = 1.0 * DEG
ALWAYS = CameraBiasConfig(mode="always", process_noise=0.0)
CONFIRMED = Lifecycle(TrackStatus.CONFIRMED, 3, 3, 0)


def unit_tracker(config: CameraBiasConfig = ALWAYS) -> MultiTargetTracker:
    base = TrackerConfig(dt=DT, accel_std=0.5, velocity_std=20.0, camera_bias=config)
    return MultiTargetTracker(base, RADAR, CAMERA)


def at(range_: float, bearing: float) -> np.ndarray:
    return np.array([range_ * np.cos(bearing), range_ * np.sin(bearing), 0.0, 0.0])


def tight() -> np.ndarray:
    return np.diag([100.0, 100.0, 1.0, 1.0])


def run(layout, seed, *, bias_deg=0.0, turn_deg=0.0, camera_bias=None, motion=None, duration=30.0,
        clutter=0.0, **config_overrides):
    """Simulate (with a true bias / turn) and track with the given estimate settings."""
    config = MttConfig(duration=duration, radar_clutter_rate=clutter, camera_clutter_rate=clutter)
    states = SCENARIOS[layout]
    turns = tuple((i, Turn(10.0, 20.0, turn_deg * DEG)) for i in range(len(states)))
    disturbance = TruthDisturbance(
        camera_bias=bias_deg * DEG, maneuvers=turns if turn_deg else ()
    )
    sim = simulate_mtt(states, config, make_mtt_rngs(seed), disturbance)
    tracker_config = replace(
        config.tracker_config(), camera_bias=camera_bias, motion=motion, **config_overrides
    )
    radar = RadarModel(config.radar_range_std, config.radar_bearing_std)
    camera = CameraModel(config.camera_bearing_std)
    return sim, run_multi_target_tracking(sim, tracker_config, radar, camera), config


def test_a_pair_is_a_confirmed_track_with_a_radar_and_a_camera_measurement_now():
    """Fails if a pair uses the wrong difference (camera minus radar bearing) or the wrong
    tracks: two confirmed tracks, camera bearings 0.01 rad above the radar ones."""
    tracker = unit_tracker()
    for bearing in (0.0, 0.5):
        tracker.add_track(at(1000.0, bearing), tight(), CONFIRMED)
    radar = np.array([[1000.0, 0.0], [1000.0, 0.5]])
    camera = np.array([[0.01], [0.51]])
    tracker.step(radar, camera)
    estimator = tracker._bias
    assert estimator.used == 2 and estimator.rejected == 0
    b, var = 0.0, SLAB**2
    for _ in range(2):
        gain = var / (var + PAIR_VAR)
        b, var = b + gain * (0.01 - b), (1.0 - gain) * var
    assert estimator._b1 == pytest.approx(b, rel=1e-6)


def test_a_track_confirmed_by_this_very_scan_forms_no_pair():
    """Fails if a tentative track that the radar scan confirms is paired (it was not in the
    confirmed group, and its radar row came from the tentative association)."""
    for lifecycle, expected in (
        (Lifecycle(TrackStatus.TENTATIVE, 2, 2, 0), 0),
        (CONFIRMED, 1),
    ):
        tracker = unit_tracker()
        track = tracker.add_track(at(1000.0, 0.3), tight(), lifecycle)
        tracker.step(np.array([[1000.0, 0.3]]), np.array([[0.31]]))
        assert track.status is TrackStatus.CONFIRMED
        assert tracker.camera_log[-1].assigned  # the camera did update the track in both cases
        assert tracker._bias.used == expected


def test_a_radar_measurement_in_the_gate_of_two_confirmed_tracks_forms_no_pair():
    """Fails if close crossing tracks are paired: the radar measurement could belong to either
    of them, so its bearing difference is not evidence about the bias."""
    tracker = unit_tracker()
    # Tight covariances: the camera gates are far apart (so both tracks are offered the camera
    # scan) while the 2 deg radar gates, 100 m wide, both contain the measurement.
    sharp = np.diag([4.0, 4.0, 1.0, 1.0])
    tracker.add_track(np.array([1000.0, 0.0, 0.0, 0.0]), sharp, CONFIRMED)
    tracker.add_track(np.array([1000.0, 30.0, 0.0, 0.0]), sharp, CONFIRMED)
    radar = np.array([[np.hypot(1000.0, 15.0), np.arctan2(15.0, 1000.0)]])
    camera = np.array([[0.0], [np.arctan2(30.0, 1000.0)]])
    tracker.step(radar, camera)
    assert tracker.bias_pairs_excluded == 1
    assert tracker._bias.used == 0


def test_no_pair_without_both_sensors_in_the_same_step():
    """Fails if a pair is formed from a radar-only or a camera-only step."""
    tracker = unit_tracker()
    tracker.add_track(at(1000.0, 0.3), tight(), CONFIRMED)
    tracker.step(np.array([[1000.0, 0.3]]), None)
    tracker.step(None, np.array([[0.31]]))
    tracker.step(np.array([[1000.0, 0.3]]), np.zeros((0, 1)))
    assert tracker._bias.used == 0 and tracker._bias.rejected == 0


def test_the_tracks_run_on_the_raw_camera_data_whatever_the_estimate():
    """Fails if the estimate leaks into association or the filters: with an oracle bias of
    0.5 deg the internal estimates, ids and statuses are those of the tracker without it."""
    sim, plain, config = run("separated", 2000, bias_deg=0.5)
    _, oracle, _ = run(
        "separated", 2000, bias_deg=0.5, camera_bias=CameraBiasConfig(
            mode="oracle", oracle_bias=0.5 * DEG)
    )
    for a, b in zip(plain.history, oracle.history, strict=True):
        assert [(s.track_id, s.status) for s in a] == [(s.track_id, s.status) for s in b]
    for ta, tb in zip(plain.tracker.tracks, oracle.tracker.tracks, strict=True):
        np.testing.assert_array_equal(ta.filter.x, tb.filter.x)
        np.testing.assert_array_equal(ta.filter.P, tb.filter.P)
    assert plain.tracker.births == oracle.tracker.births


def test_a_zero_oracle_bias_reproduces_the_default_tracker_bitwise():
    """Fails if the correction is applied when there is nothing to correct, or the sensitivity
    filter differs from the EKF: reported states and covariances must be those of the default."""
    for layout in ("separated", "crossing"):
        _, plain, _ = run(layout, 2001, clutter=5.0)
        _, zero, _ = run(layout, 2001, clutter=5.0, camera_bias=CameraBiasConfig(mode="oracle"))
        assert history_digest(plain) == history_digest(zero)


def test_the_oracle_bias_removes_the_cross_range_error_of_a_one_degree_bias():
    """Fails if V has the wrong scale or sign, or the correction is not x - V b.

    With the true 1 deg bias applied, the mean cross-range error of the matched tracks is near
    zero while without it it is about +range * 1 deg (counter-clockwise positive).
    """
    wide = 200.0
    sim, plain, config = run("separated", 2000, bias_deg=1.0)
    _, fixed, _ = run(
        "separated", 2000, bias_deg=1.0,
        camera_bias=CameraBiasConfig(mode="oracle", oracle_bias=1.0 * DEG),
    )
    means = []
    for result in (plain, fixed):
        ids = match_tracks(sim.truth, result.history, wide).ids
        errors = matched_errors(sim.truth, result.history, ids, config.burn_in_steps)
        means.append(float(np.mean(errors.cross)))
    assert means[0] > 8.0
    assert abs(means[1]) < 2.0


def test_the_oracle_correction_works_with_an_imm_motion_model_too():
    """Fails if the IMM sensitivity (mixed per-mode V) is wrong or not wired into snapshots."""
    wide = 200.0
    motion = MotionConfig((ModeSpec(), ModeSpec(accel_std=3.0)))
    sim, plain, config = run("separated", 2000, bias_deg=1.0, motion=motion)
    _, fixed, _ = run(
        "separated", 2000, bias_deg=1.0, motion=motion,
        camera_bias=CameraBiasConfig(mode="oracle", oracle_bias=1.0 * DEG),
    )
    means = []
    for result in (plain, fixed):
        ids = match_tracks(sim.truth, result.history, wide).ids
        errors = matched_errors(sim.truth, result.history, ids, config.burn_in_steps)
        means.append(float(np.mean(errors.cross)))
    assert means[0] > 8.0 and abs(means[1]) < 2.5


def test_a_real_bias_is_learned_from_the_pairs():
    """Fails if the tracker feeds the estimate wrongly (sign, wrap, pairs): after 40 s of a
    1 deg bias the applied estimate must be well above zero and nearer to the truth."""
    _, result, _ = run(
        "separated", 2000, bias_deg=1.0, duration=40.0,
        camera_bias=CameraBiasConfig(mode="always"),
    )
    final = result.tracker.bias_log[-1]
    assert abs(final.b_app - 1.0 * DEG) < 0.5 * DEG
    assert final.used > 30


def test_a_turn_is_not_read_as_a_bias():
    """Fails if the estimate is built from track residuals instead of radar-camera pairs.

    Targets turning at 20 deg/s with no bias: a residual-based estimate would follow the
    lagging tracks by degrees; the pair estimate stays within its noise (here 0.25 deg after
    20 s, a loose bound far below what lag would produce).
    """
    for seed in (2000, 2001, 2002):
        _, result, _ = run(
            "separated", seed, turn_deg=20.0, duration=40.0,
            camera_bias=CameraBiasConfig(mode="spike_slab"),
        )
        late = [e.b_app for e in result.tracker.bias_log[200:]]
        assert max(abs(b) for b in late) < 0.25 * DEG, seed


def test_switching_the_diagnostics_off_does_not_change_the_reported_tracks():
    """Fails if anything reads the bias log, or the log is not kept one entry per step."""
    config = MttConfig(duration=20.0, radar_clutter_rate=5.0, camera_clutter_rate=5.0)
    sim = simulate_mtt(
        SCENARIOS["crossing"], config, make_mtt_rngs(2000),
        TruthDisturbance(camera_bias=0.5 * DEG),
    )
    radar = RadarModel(config.radar_range_std, config.radar_bearing_std)
    camera = CameraModel(config.camera_bearing_std)
    base = replace(config.tracker_config(), camera_bias=CameraBiasConfig())
    on = run_multi_target_tracking(sim, base, radar, camera)
    off = run_multi_target_tracking(sim, replace(base, record_diagnostics=False), radar, camera)
    assert history_digest(on) == history_digest(off)
    assert len(on.tracker.bias_log) == len(sim.radar_scans) and off.tracker.bias_log == []


def test_the_bias_estimate_needs_the_camera():
    """Fails if a bias estimate is accepted for a radar-only tracker."""
    with pytest.raises(ValueError):
        TrackerConfig(
            dt=DT, accel_std=0.5, velocity_std=20.0, use_camera=False, camera_bias=ALWAYS
        )


def test_an_outage_step_without_a_radar_scan_makes_no_pairs_and_keeps_the_estimate():
    """Fails if steps without a radar scan (nine of ten) change the estimate except for drift."""
    tracker = unit_tracker()
    tracker.add_track(at(1000.0, 0.3), tight(), CONFIRMED)
    tracker.step(np.array([[1000.0, 0.3]]), np.array([[0.31]]))
    used = tracker._bias.used
    before = tracker._bias.estimate
    for _ in range(9):
        tracker.step(None, np.array([[0.31]]))
    assert tracker._bias.used == used and tracker._bias.estimate == before
