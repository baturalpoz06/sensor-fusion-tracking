"""Tests of the tracker with an IMM motion model.

Each test docstring names the defect that makes it fail.
"""

import hashlib
from dataclasses import replace

import numpy as np
import pytest

import fusion.tracker.multi_target as tracker_module
from fusion.dropout import DropoutWindow, apply_dropout
from fusion.filters.imm import IMMFilter, ModeSpec, MotionConfig, combine_modes
from fusion.maneuvers import Acceleration, Turn, maneuvering_trajectory
from fusion.mtt_experiment import SCENARIOS, MttConfig
from fusion.mtt_simulation import make_mtt_rngs, simulate_mtt
from fusion.sensors.camera import CameraModel
from fusion.sensors.radar import RadarModel
from fusion.tracker.multi_target import (
    MultiTargetTracker,
    TrackerConfig,
    TrackerRun,
    run_multi_target_tracking,
)
from fusion.tracker.track import Lifecycle, TrackStatus

DT = 0.1
DEG = float(np.pi / 180.0)
SEEDS = (2000, 2001)
SINGLE = MotionConfig((ModeSpec(),))
IMM_A = MotionConfig((ModeSpec(), ModeSpec(accel_std=3.0)))
IMM_B = MotionConfig(
    (ModeSpec(), ModeSpec("ct", 1.0, 10.0 * DEG), ModeSpec("ct", 1.0, -10.0 * DEG))
)
RADAR = RadarModel(range_std=5.0, bearing_std=float(np.deg2rad(2.0)))
CAMERA = CameraModel(bearing_std=float(np.deg2rad(0.1)))


def history_digest(run: TrackerRun) -> str:
    """SHA-256 over ids, statuses, states and covariances of every step (as the goldens do)."""
    digest = hashlib.sha256()
    for step in run.history:
        digest.update(f"step:{len(step)}".encode())
        for snapshot in step:
            digest.update(f"{snapshot.track_id}:{snapshot.status.value}".encode())
            digest.update(np.ascontiguousarray(snapshot.x).tobytes())
            digest.update(np.ascontiguousarray(snapshot.P).tobytes())
    return digest.hexdigest()


def models(config: MttConfig):
    radar = RadarModel(config.radar_range_std, config.radar_bearing_std)
    camera = CameraModel(config.camera_bearing_std)
    return radar, camera


def run_pair(
    layout: str, clutter: float, seed: int, *, motion, windows=(), policy="unaware", **overrides
):
    """Default tracker and tracker with `motion` on the same simulation."""
    config = MttConfig(
        duration=30.0, burn_in=5.0, radar_clutter_rate=clutter, camera_clutter_rate=clutter
    )
    sim = simulate_mtt(SCENARIOS[layout], config, make_mtt_rngs(seed))
    radar_down = None
    if windows:
        dropped = apply_dropout(sim, list(windows), config.dt)
        sim, radar_down = dropped.sim, dropped.radar_down
    base = replace(config.tracker_config(), outage_policy=policy, **overrides)
    radar, camera = models(config)
    plain = run_multi_target_tracking(sim, base, radar, camera, radar_down)
    other = run_multi_target_tracking(sim, replace(base, motion=motion), radar, camera, radar_down)
    return plain, other


@pytest.mark.parametrize("layout", ["crossing", "separated"])
@pytest.mark.parametrize("clutter", [0.0, 5.0])
@pytest.mark.parametrize("seed", SEEDS)
def test_a_single_mode_imm_tracker_reproduces_the_default_tracker_bitwise(layout, clutter, seed):
    """Fails if the IMM path changes any track, covariance or id relative to the plain EKF."""
    plain, imm = run_pair(layout, clutter, seed, motion=SINGLE)
    assert history_digest(plain) == history_digest(imm)
    assert plain.tracker.births == imm.tracker.births


@pytest.mark.parametrize("policy", ["unaware", "aware"])
def test_a_single_mode_imm_tracker_reproduces_the_default_through_an_outage(policy):
    """Fails if coasting, freezing or the outage deletions differ with a motion model."""
    windows = [DropoutWindow("radar", 10.0, 18.0)]
    plain, imm = run_pair("separated", 5.0, 2000, motion=SINGLE, windows=windows, policy=policy)
    assert history_digest(plain) == history_digest(imm)
    assert plain.tracker.coast_deletions == imm.tracker.coast_deletions


def test_gating_and_association_receive_the_combined_estimate():
    """Fails if the gate or the cost use one mode (e.g. the first) instead of the mix.

    A target accelerating along its heading drives the two modes apart; at the next radar scan
    the filter handed to gated_costs must carry the moment-matched combination of the modes.
    """
    scene = MttConfig(duration=12.0, radar_clutter_rate=0.0, camera_clutter_rate=0.0)
    start = np.array([-450.0, 1000.0, 15.0, 0.0])
    path = maneuvering_trajectory(
        start, DT, 120, 0.0, np.random.default_rng(0), [Acceleration(1.0, 12.0, 4.0)]
    ).states
    config = replace(scene.tracker_config(), motion=IMM_A)
    tracker = MultiTargetTracker(config, RADAR, CAMERA)
    track = tracker.add_track(
        path[0], np.diag([25.0, 25.0, 1.0, 1.0]), Lifecycle(TrackStatus.CONFIRMED, 3, 3, 0)
    )
    seen = []
    original = tracker_module.gated_costs

    def spy(filters, *args, **kwargs):
        for f in filters:
            seen.append((f.x.copy(), f.P.copy(), f.mu, f.x_modes, f.P_modes))
        return original(filters, *args, **kwargs)

    tracker_module.gated_costs = spy
    try:
        for k in range(1, 100):
            radar_z = RADAR.h(path[k])[None, :] if k % 10 == 0 else None
            tracker.step(radar_z, CAMERA.h(path[k])[None, :])
    finally:
        tracker_module.gated_costs = original
    assert isinstance(track.filter, IMMFilter) and seen
    differs = False
    for x, cov, mu, x_modes, p_modes in seen:
        expected_x, expected_cov, _ = combine_modes(mu, x_modes, p_modes)
        np.testing.assert_array_equal(x, expected_x)
        np.testing.assert_array_equal(cov, expected_cov)
        differs |= bool(np.abs(x - x_modes[0]).max() > 1e-6)
    assert differs, "the modes never separated: the test would not tell mix from mode 0"


def test_unaware_and_aware_blackouts_delete_and_coast_as_without_a_motion_model():
    """Fails if an IMM track coasts or dies differently from an EKF track in an outage."""
    windows = [DropoutWindow("radar", 10.0, 24.0), DropoutWindow("camera", 10.0, 24.0)]
    for policy in ("unaware", "aware"):
        plain, imm = run_pair(
            "separated", 0.0, 2000, motion=IMM_A, windows=windows, policy=policy, max_coast_time=5.0
        )

        def alive(run, k):
            return [(s.track_id, s.status) for s in run.history[k]]

        for k in range(95, 245):
            assert alive(plain, k) == alive(imm, k), (policy, k)
        assert plain.tracker.coast_deletions == imm.tracker.coast_deletions
    assert plain.tracker.coast_deletions > 0 or policy == "unaware"


def test_mode_probabilities_propagate_while_coasting():
    """Fails if mu is frozen or reset during an outage instead of relaxing through the matrix."""
    windows = [DropoutWindow("radar", 10.0, 17.0), DropoutWindow("camera", 10.0, 17.0)]
    _, imm = run_pair("separated", 0.0, 2000, motion=IMM_A, windows=windows, policy="aware")
    before = {e.track_id: e.mu for e in imm.tracker.mode_log[99]}
    after = {e.track_id: e.mu for e in imm.tracker.mode_log[169]}
    common = [i for i in before if i in after and abs(before[i][0] - 0.5) > 0.05]
    assert common
    for i in common:
        assert abs(after[i][0] - 0.5) < abs(before[i][0] - 0.5)
        assert after[i] != before[i]


def truth_run(path: np.ndarray, motion: MotionConfig, steps: int) -> IMMFilter:
    """Filter started at the true state and fed noise-free radar (1 Hz) and camera (10 Hz)."""
    imm = motion.build(DT, 0.5, path[0], np.diag([25.0, 25.0, 1.0, 1.0]))
    for k in range(1, steps + 1):
        imm.predict()
        if k % 10 == 0:
            imm.update(RADAR.h(path[k]), RADAR)
        imm.update(CAMERA.h(path[k]), CAMERA)
    return imm


def test_a_sustained_left_turn_moves_the_probability_to_the_left_turn_mode():
    """Fails if the likelihood does not discriminate the modes, or the turn sign is flipped."""
    path = maneuvering_trajectory(
        np.array([-450.0, 600.0, 15.0, 0.0]),
        DT,
        80,
        0.0,
        np.random.default_rng(0),
        [Turn(0.0, 8.0, 10.0 * DEG)],
    ).states
    mu = truth_run(path, IMM_B, 60).mu
    assert mu[1] > 0.5 and mu[1] > mu[0] and mu[1] > mu[2]
    right = maneuvering_trajectory(
        np.array([-450.0, 600.0, 15.0, 0.0]),
        DT,
        80,
        0.0,
        np.random.default_rng(0),
        [Turn(0.0, 8.0, -10.0 * DEG)],
    ).states
    assert truth_run(right, IMM_B, 60).mu[2] > 0.5


def test_a_sudden_acceleration_moves_the_probability_to_the_high_noise_mode():
    """Fails if the high-noise CV mode never takes over when the target accelerates."""
    path = maneuvering_trajectory(
        np.array([-450.0, 1000.0, 15.0, 0.0]),
        DT,
        120,
        0.0,
        np.random.default_rng(0),
        [Acceleration(1.0, 7.0, 4.0)],
    ).states
    assert truth_run(path, IMM_A, 70).mu[1] > 0.5


def test_a_constant_velocity_target_keeps_the_low_noise_mode_ahead():
    """Fails if the low-noise mode loses a target that does not maneuver."""
    path = maneuvering_trajectory(
        np.array([-450.0, 1000.0, 15.0, 0.0]), DT, 120, 0.0, np.random.default_rng(0)
    ).states
    assert truth_run(path, IMM_A, 100).mu[0] > 0.5


def test_switching_the_diagnostics_off_does_not_change_a_track():
    """Fails if anything in the tracker reads the camera or mode logs."""
    config = MttConfig(duration=20.0, radar_clutter_rate=5.0, camera_clutter_rate=5.0)
    sim = simulate_mtt(SCENARIOS["crossing"], config, make_mtt_rngs(2000))
    radar, camera = models(config)
    base = replace(config.tracker_config(), motion=IMM_B)
    on = run_multi_target_tracking(sim, base, radar, camera)
    off = run_multi_target_tracking(sim, replace(base, record_diagnostics=False), radar, camera)
    assert history_digest(on) == history_digest(off)
    assert len(on.tracker.mode_log) == len(sim.radar_scans) and on.tracker.camera_log
    assert off.tracker.mode_log == [] and off.tracker.camera_log == []
    assert off.tracker.camera_offered == on.tracker.camera_offered


def test_the_imm_tracker_follows_a_turning_target_it_would_lose_with_one_model():
    """Fails if IMM-B is no better than the CV tracker at a hard turn (end to end smoke test)."""
    from fusion.maneuvers import TruthDisturbance
    from fusion.mtt_metrics import match_tracks

    config = MttConfig(duration=45.0, radar_clutter_rate=0.0, camera_clutter_rate=0.0)
    states = SCENARIOS["separated"]
    turns = tuple((i, Turn(15.0, 25.0, 10.0 * DEG)) for i in range(len(states)))
    sim = simulate_mtt(states, config, make_mtt_rngs(2000), TruthDisturbance(maneuvers=turns))
    radar, camera = models(config)
    base = config.tracker_config()
    missed = {}
    for name, motion in (("cv", None), ("imm", IMM_B)):
        run = run_multi_target_tracking(sim, replace(base, motion=motion), radar, camera)
        ids = match_tracks(sim.truth, run.history, 50.0).ids
        missed[name] = float(np.mean(ids[:, 150:] < 0))
    assert missed["imm"] < missed["cv"]


def test_motion_configuration_is_validated_and_hashable():
    """Fails if bad modes or stay probabilities are accepted, or configs cannot be pickled."""
    import pickle

    with pytest.raises(ValueError):
        ModeSpec("turn")
    with pytest.raises(ValueError):
        ModeSpec("cv", omega=0.1)
    with pytest.raises(ValueError):
        ModeSpec("cv", accel_std=-1.0)
    with pytest.raises(ValueError):
        MotionConfig(())
    with pytest.raises(ValueError):
        MotionConfig((ModeSpec(), ModeSpec()), stay_per_scan=0.4)
    assert pickle.loads(pickle.dumps(IMM_B)) == IMM_B
    assert hash(IMM_B) == hash(pickle.loads(pickle.dumps(IMM_B)))
    assert TrackerConfig(dt=DT, accel_std=0.5, velocity_std=20.0).motion is None
