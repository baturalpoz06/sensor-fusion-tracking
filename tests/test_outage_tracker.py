"""Tests for the tracker's radar outage handling: the unaware and the aware policy.

Each test docstring names the condition that makes it fail without the feature.
"""

import copy
from dataclasses import replace

import numpy as np
import pytest

from fusion.dropout import DropoutWindow, apply_dropout, down_steps
from fusion.mtt_experiment import SCENARIOS
from fusion.mtt_simulation import SimulationConfig, make_mtt_rngs, simulate_mtt
from fusion.sensors.camera import CameraModel
from fusion.sensors.radar import RadarModel
from fusion.tracker.multi_target import (
    MultiTargetTracker,
    TrackerConfig,
    run_multi_target_tracking,
)
from fusion.tracker.track import Lifecycle, LifecycleConfig, TrackStatus, advance

DT = 0.1
RANGE, BEARING = 1000.0, 0.3
RADAR = RadarModel(range_std=5.0, bearing_std=float(np.deg2rad(2.0)))
CAMERA = CameraModel(bearing_std=float(np.deg2rad(0.1)))
TENTATIVE, CONFIRMED, DELETED = TrackStatus.TENTATIVE, TrackStatus.CONFIRMED, TrackStatus.DELETED
EMPTY_RADAR = np.zeros((0, 2))
EMPTY_CAMERA = np.zeros((0, 1))
LIFECYCLE = LifecycleConfig(confirm_hits=3, confirm_window=5, max_misses=2)


def make_config(**overrides) -> TrackerConfig:
    return TrackerConfig(dt=DT, accel_std=0.5, velocity_std=20.0, **overrides)


def make_tracker(**overrides) -> MultiTargetTracker:
    return MultiTargetTracker(make_config(**overrides), RADAR, CAMERA)


def stationary(range_: float = RANGE, bearing: float = BEARING) -> np.ndarray:
    return np.array([range_ * np.cos(bearing), range_ * np.sin(bearing), 0.0, 0.0])


def tight() -> np.ndarray:
    return np.diag([25.0, 25.0, 1.0, 1.0])


# --- configuration -------------------------------------------------------------------


def test_outage_defaults_leave_the_tracker_unaware_and_coasting_for_fifteen_seconds():
    """Fails if a default silently enables the aware policy for existing callers."""
    config = make_config()
    assert config.outage_policy == "unaware" and config.aware_tentatives == "drop"
    assert config.max_coast_time == 15.0 and config.max_coast_steps == 150


@pytest.mark.parametrize(
    "kwargs",
    [
        {"outage_policy": "clairvoyant"},
        {"aware_tentatives": "keep"},
        {"max_coast_time": 0.0},
        {"max_coast_time": -1.0},
        {"max_coast_time": DT / 2},
        {"max_coast_time": float("inf")},
        {"max_coast_time": float("nan")},
    ],
)
def test_invalid_outage_settings_are_rejected(kwargs):
    """Fails if a typo in a policy name, a coast time below one step, or inf / nan is accepted."""
    with pytest.raises(ValueError):
        make_config(**kwargs)


@pytest.mark.parametrize(
    ("seconds", "steps"), [(0.1, 1), (0.3, 3), (0.25, 2), (15.0, 150), (0.7, 7), (2.9, 29)]
)
def test_max_coast_time_converts_to_whole_steps_without_float_errors(seconds, steps):
    """Fails if the conversion is floor(seconds / dt) (0.3 / 0.1 = 2.9999999999999996 -> 2)."""
    assert make_config(max_coast_time=seconds).max_coast_steps == steps


# --- aware and unaware tracker on one track ------------------------------------------


def test_an_aware_tracker_rejects_radar_data_while_the_radar_is_down():
    """Fails if an aware tracker silently uses measurements of a sensor it knows is down."""
    aware = make_tracker(outage_policy="aware")
    aware.add_track(stationary(), tight(), Lifecycle(CONFIRMED, 3, 3, 0))
    before = aware.tracks[0].filter.P.copy()
    with pytest.raises(ValueError, match="radar is down"):
        aware.step(np.array([[RANGE, BEARING]]), None, radar_down=True)
    np.testing.assert_array_equal(aware.tracks[0].filter.P, before)  # nothing was predicted
    assert aware.tracks[0].coast_steps == 0

    unaware = make_tracker()
    unaware.add_track(stationary(), tight(), Lifecycle(CONFIRMED, 3, 3, 0))
    unaware.step(np.array([[RANGE, BEARING]]), None, radar_down=True)  # the flag is ignored
    assert unaware.tracks[0].lifecycle.hits == 4


def test_an_outage_is_a_run_of_misses_for_the_unaware_tracker_and_frozen_for_the_aware_one():
    """Fails if the policy is ignored: both trackers would then count the same misses."""
    state = Lifecycle(CONFIRMED, 3, 3, 0)
    unaware = make_tracker(lifecycle=LIFECYCLE)
    aware = make_tracker(lifecycle=LIFECYCLE, outage_policy="aware")
    for tracker in (unaware, aware):
        tracker.add_track(stationary(), tight(), state)
    trace_before = np.trace(aware.tracks[0].filter.P)
    for _ in range(2):
        unaware.step(EMPTY_RADAR, None, radar_down=True)
        aware.step(EMPTY_RADAR, None, radar_down=True)
    assert unaware.tracks[0].lifecycle == Lifecycle(CONFIRMED, 3, 5, 2)
    assert unaware.radar_scans == 2
    assert aware.tracks[0].lifecycle == state and aware.radar_scans == 0
    assert np.trace(aware.tracks[0].filter.P) > trace_before  # still coasting
    unaware.step(EMPTY_RADAR, None, radar_down=True)
    aware.step(EMPTY_RADAR, None, radar_down=True)
    assert unaware.tracks == [] and len(aware.tracks) == 1  # K = 2: the third miss deletes


def test_the_camera_update_resets_the_coast_clock_of_its_track_only():
    """Fails if the clock counts radar hits only: the camera-updated track would be deleted."""
    tracker = make_tracker(outage_policy="aware", max_coast_time=1.0)
    seen = tracker.add_track(stationary(RANGE, 0.3), tight(), Lifecycle(CONFIRMED, 3, 3, 0))
    unseen = tracker.add_track(stationary(RANGE, 1.5), tight(), Lifecycle(CONFIRMED, 3, 3, 0))
    camera_hit = np.array([[0.3]])  # inside the gate of the first track only
    survivors = []
    for _ in range(12):
        tracker.step(None, camera_hit, radar_down=True)
        survivors.append(len(tracker.tracks))
    assert survivors == [2] * 10 + [1, 1]  # the unseen track dies on its 11th step (> 10)
    assert tracker.tracks == [seen] and seen.coast_steps == 0
    assert unseen not in tracker.tracks and tracker.coast_deletions == 1


def test_coasting_deletes_only_during_a_known_radar_outage():
    """Fails if the coast limit is also enforced at ordinary steps without a scan."""
    tracker = make_tracker(
        outage_policy="aware", max_coast_time=0.5, use_camera=False
    )
    track = tracker.add_track(stationary(), tight(), Lifecycle(CONFIRMED, 3, 3, 0))
    for _ in range(9):
        tracker.step(None, None)  # no scheduled scan, radar up: no lifecycle change at all
    assert tracker.tracks == [track] and track.coast_steps == 9 > tracker.config.max_coast_steps
    tracker.step(None, None, radar_down=True)  # the first outage step enforces the limit
    assert tracker.tracks == [] and tracker.coast_deletions == 1


def test_an_empty_camera_scan_and_no_camera_scan_give_identical_tracks():
    """Fails if an empty camera scan changes a track, which would make the camera policy matter."""
    scene = SimulationConfig(duration=20.0, radar_clutter_rate=3.0, camera_clutter_rate=3.0)
    sim = simulate_mtt(SCENARIOS["crossing"], scene, make_mtt_rngs(2))
    window = DropoutWindow("camera", 5.0, 12.0)
    silent = down_steps([window], "camera", DT, len(sim.camera_scans))
    assert silent.any() and any(len(sim.camera_scans[k].z) for k in np.flatnonzero(silent))

    histories = []
    for blank in (EMPTY_CAMERA, None):
        tracker = make_tracker()
        history = []
        for k, (radar, camera) in enumerate(zip(sim.radar_scans, sim.camera_scans, strict=True)):
            history.append(
                tracker.step(
                    None if radar is None else radar.z, blank if silent[k] else camera.z
                )
            )
        histories.append((history, tracker.camera_skipped))
    assert_same_history(histories[0][0], histories[1][0])
    assert histories[0][1] == histories[1][1]


# --- the oracle ----------------------------------------------------------------------

# Events of one step with a single stationary target at (RANGE, BEARING), radar scheduled
# every step. The tracker must agree with a plain reference after every event of every
# sequence. H: radar scan with a detection. M: radar scan without one. N: no scheduled
# scan, radar up. D: scheduled scan lost to an outage (an empty scan is delivered, the radar
# is flagged down). E: outage at a step without a scheduled scan (no data, flagged down).
# C: outage step in which the camera sees the target.
EVENTS = "HMNDEC"
DEPTH = 4
INITIAL_STATES = {
    "tentative_just_born": Lifecycle(TENTATIVE, 1, 1, 0),
    "tentative_two_misses": Lifecycle(TENTATIVE, 1, 3, 2),
    "confirmed_fresh": Lifecycle(CONFIRMED, 3, 3, 0),
    "confirmed_at_k_misses": Lifecycle(CONFIRMED, 3, 5, 2),
}
# Starting states from which a tentative track exists within DEPTH events (a confirmed track
# with K = 2 needs three misses and a birth before a tentative one exists, which is too deep).
REACHES_TENTATIVE = ("tentative_just_born", "tentative_two_misses", "confirmed_at_k_misses")
POLICIES = [("unaware", "drop", 100)] + [
    ("aware", tentatives, steps) for tentatives in ("drop", "freeze") for steps in (1, 3, 100)
]


def step_arguments(event: str):
    """(radar_z, camera_z, radar_down) the runner would pass for an event."""
    hit = np.array([[RANGE, BEARING]])
    return {
        "H": (hit, None, False),
        "M": (EMPTY_RADAR, None, False),
        "N": (None, None, False),
        "D": (EMPTY_RADAR, None, True),
        "E": (None, None, True),
        "C": (None, np.array([[BEARING]]), True),
    }[event]


class Reference:
    """The specified behavior of one track, written without any tracker code."""

    def __init__(self, config: TrackerConfig, lifecycle: Lifecycle) -> None:
        self.config = config
        self.lifecycle: Lifecycle | None = lifecycle
        self.coast = 0
        self.births = self.scans = self.coast_deletions = self.tentative_drops = 0

    def clone(self) -> "Reference":
        return copy.copy(self)

    def apply(self, event: str) -> None:
        config = self.config
        aware = config.outage_policy == "aware"
        if self.lifecycle is not None:
            self.coast += 1  # every step predicts without an update
        if event in "HM" or (event == "D" and not aware):  # a radar scan is processed
            self.scans += 1
            hit = event == "H"
            if self.lifecycle is not None:
                self.lifecycle = advance(self.lifecycle, hit, config.lifecycle)
                self.coast = 0 if hit else self.coast
                if self.lifecycle.status is DELETED:
                    self.lifecycle = None
            elif hit:  # an unexplained detection starts a new tentative track
                self.lifecycle = Lifecycle.born(config.lifecycle)
                self.coast = 0
                self.births += 1
        if event == "C" and self.lifecycle is not None and self.lifecycle.status is CONFIRMED:
            self.coast = 0  # the camera updates confirmed tracks only
        if aware and event in "DEC" and self.lifecycle is not None:
            if config.aware_tentatives == "drop" and self.lifecycle.status is TENTATIVE:
                self.lifecycle = None
                self.tentative_drops += 1
            elif self.coast > config.max_coast_steps:
                self.lifecycle = None
                self.coast_deletions += 1


def explore(tracker, reference, depth, trail, seen, events=EVENTS):
    """Check every event sequence up to the depth; returns the number of sequences checked."""
    checked = 0
    for event in events:
        branch, expected = copy.deepcopy(tracker), reference.clone()
        radar_z, camera_z, down = step_arguments(event)
        branch.step(radar_z, camera_z, radar_down=down)
        expected.apply(event)
        where = trail + event

        assert len(branch.tracks) == (expected.lifecycle is not None), where
        if expected.lifecycle is not None:
            assert branch.tracks[0].lifecycle == expected.lifecycle, where
            assert branch.tracks[0].coast_steps == expected.coast, where
        counters = (branch.births, branch.radar_scans)
        assert counters == (expected.births, expected.scans), where
        assert branch.coast_deletions == expected.coast_deletions, where
        assert branch.tentative_drops == expected.tentative_drops, where
        seen["coast_deletions"] = max(seen["coast_deletions"], expected.coast_deletions)
        seen["tentative_drops"] = max(seen["tentative_drops"], expected.tentative_drops)
        seen["births"] = max(seen["births"], expected.births)

        checked += 1
        if depth > 1:
            checked += explore(branch, expected, depth - 1, where, seen, events)
    return checked


@pytest.mark.parametrize("initial", INITIAL_STATES)
@pytest.mark.parametrize(("policy", "tentatives", "coast_steps"), POLICIES, ids=str)
def test_every_event_sequence_matches_the_reference(initial, policy, tentatives, coast_steps):
    """Exhaustive oracle for the freeze, drop and coast rules.

    Fails on any error in: freezing the counters, deleting tentative tracks, the coast
    clock (increment, reset by a radar hit or a camera update of a confirmed track, limit
    and its off-by-one), enforcing the limit only at outage steps, or births after a death.
    """
    config = make_config(
        lifecycle=LIFECYCLE,
        outage_policy=policy,
        aware_tentatives=tentatives,
        max_coast_time=coast_steps * DT,
    )
    assert config.max_coast_steps == coast_steps
    tracker = MultiTargetTracker(config, RADAR, CAMERA)
    tracker.add_track(stationary(), tight(), INITIAL_STATES[initial])
    seen = {"coast_deletions": 0, "tentative_drops": 0, "births": 0}

    checked = explore(tracker, Reference(config, INITIAL_STATES[initial]), DEPTH, "", seen)

    assert checked == sum(len(EVENTS) ** n for n in range(1, DEPTH + 1))
    # The sequences must reach the rules under test, or agreement would prove nothing.
    confirmed = INITIAL_STATES[initial].status is CONFIRMED
    if policy == "aware" and coast_steps < 100 and confirmed:
        assert seen["coast_deletions"] > 0
    if policy == "aware" and tentatives == "drop" and initial in REACHES_TENTATIVE:
        assert seen["tentative_drops"] > 0
    if policy == "unaware" or tentatives == "freeze":
        assert seen["tentative_drops"] == 0
    assert seen["births"] > 0


LONG_EVENTS = "HMD"
LONG_DEPTH = 6
LONG_POLICIES = [("unaware", "drop", 100), ("aware", "drop", 3), ("aware", "freeze", 3)]


@pytest.mark.parametrize("initial", INITIAL_STATES)
@pytest.mark.parametrize(("policy", "tentatives", "coast_steps"), LONG_POLICIES, ids=str)
def test_long_outage_sequences_match_the_reference(initial, policy, tentatives, coast_steps):
    """Oracle over sequences of six hits, misses and lost scans.

    Fails if a rule only breaks late: counters drifting during a long outage, a coast limit
    that is reached after several lost scans in a row, or a rebirth after a long gap.
    """
    config = make_config(
        lifecycle=LIFECYCLE,
        outage_policy=policy,
        aware_tentatives=tentatives,
        max_coast_time=coast_steps * DT,
    )
    tracker = MultiTargetTracker(config, RADAR, CAMERA)
    tracker.add_track(stationary(), tight(), INITIAL_STATES[initial])
    seen = {"coast_deletions": 0, "tentative_drops": 0, "births": 0}
    reference = Reference(config, INITIAL_STATES[initial])

    checked = explore(tracker, reference, LONG_DEPTH, "", seen, LONG_EVENTS)

    assert checked == sum(len(LONG_EVENTS) ** n for n in range(1, LONG_DEPTH + 1))
    if policy == "aware" and INITIAL_STATES[initial].status is CONFIRMED:
        assert seen["coast_deletions"] > 0 or seen["tentative_drops"] > 0  # the rules were reached


# --- runner --------------------------------------------------------------------------


def assert_same_history(a, b):
    assert len(a) == len(b)
    for step_a, step_b in zip(a, b, strict=True):
        assert [(s.track_id, s.status) for s in step_a] == [(s.track_id, s.status) for s in step_b]
        for snap_a, snap_b in zip(step_a, step_b, strict=True):
            np.testing.assert_array_equal(snap_a.x, snap_b.x)
            np.testing.assert_array_equal(snap_a.P, snap_b.P)


@pytest.mark.parametrize("tentatives", ["drop", "freeze"])
def test_an_aware_tracker_without_outages_is_identical_to_the_tracker_without_the_feature(
    tentatives,
):
    """Fails if the aware path (drops, coast deletion, counter freezing) acts outside outages."""
    scene = SimulationConfig(duration=30.0, radar_clutter_rate=5.0, camera_clutter_rate=5.0)
    sim = simulate_mtt(SCENARIOS["crossing"], scene, make_mtt_rngs(1))
    models = (RADAR, CAMERA)
    config = make_config(max_coast_time=DT)  # the smallest limit: any leak would delete tracks
    plain = MultiTargetTracker(config, *models)
    manual = [
        plain.step(None if r is None else r.z, c.z)  # the call without the new argument
        for r, c in zip(sim.radar_scans, sim.camera_scans, strict=True)
    ]
    aware = replace(config, outage_policy="aware", aware_tentatives=tentatives)
    for flags in (None, np.zeros(len(sim.radar_scans), dtype=bool)):
        run = run_multi_target_tracking(sim, aware, *models, radar_down=flags)
        assert_same_history(manual, run.history)
    # Preconditions: the scene has tentative tracks and many births, so a leak would show.
    assert any(s.status is TENTATIVE for step in manual for s in step)
    assert plain.births > 10


def test_the_runner_checks_the_length_of_the_outage_flags():
    """Fails if flags of the wrong length are silently truncated by zip."""
    sim = simulate_mtt(SCENARIOS["separated"], SimulationConfig(duration=2.0), make_mtt_rngs(0))
    with pytest.raises(ValueError, match="radar_down"):
        run_multi_target_tracking(sim, make_config(), RADAR, CAMERA, radar_down=np.zeros(5, bool))


def test_in_a_clean_scene_an_aware_tracker_keeps_its_tracks_through_an_eight_second_outage():
    """Fails if the policy has no effect: the unaware tracker loses every track and re-births."""
    scene = SimulationConfig(
        duration=40.0,
        radar_pd=1.0,
        camera_pd=1.0,
        radar_clutter_rate=0.0,
        camera_clutter_rate=0.0,
    )
    full = simulate_mtt(SCENARIOS["separated"], scene, make_mtt_rngs(0))
    dropped = apply_dropout(full, [DropoutWindow("radar", 20.0, 28.0)], scene.dt)
    results = {}
    for policy in ("unaware", "aware"):
        config = make_config(outage_policy=policy)
        results[policy] = run_multi_target_tracking(
            dropped.sim, config, RADAR, CAMERA, radar_down=dropped.radar_down
        )
    unaware, aware = results["unaware"], results["aware"]
    outage_end = 280

    ids = lambda run, k: {s.track_id for s in run.history[k]}  # noqa: E731
    assert ids(aware, outage_end - 1) == ids(aware, -1) == {0, 1, 2}
    assert aware.tracker.coast_deletions == 0
    # No track is born during the outage (a rare measurement outside the 99% gate may start
    # a short-lived tentative track at other times, as in Phase 6).
    during = {s.track_id for k in range(200, outage_end) for s in aware.history[k]}
    assert during == {0, 1, 2}
    assert all(s.status is CONFIRMED for s in aware.history[-1])
    for snapshot in aware.history[outage_end - 1]:
        assert snapshot.status is CONFIRMED
        truth = full.truth[:, outage_end - 1, :2]
        assert min(np.hypot(*(snapshot.x[:2] - t)) for t in truth) < 60.0

    assert ids(unaware, outage_end - 1) == set()  # K = 5: all deleted by the sixth lost scan
    assert unaware.tracker.births > 3 and ids(unaware, -1).isdisjoint({0, 1, 2})
