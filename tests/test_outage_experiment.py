"""Tests for the outage experiment: configuration, trials, sweeps and point builders.

Each test docstring names the condition that makes it fail without the feature.
"""

from dataclasses import fields, replace

import numpy as np
import pytest

from fusion.dropout import MarkovBursts, PeriodicFlicker, SingleOutage, apply_dropout
from fusion.mtt_experiment import MttConfig, run_mtt_trial
from fusion.mtt_simulation import make_mtt_rngs, simulate_mtt
from fusion.outage_experiment import (
    OUTAGE_SCENARIOS,
    SENSOR_SETS,
    VANISHING_TARGET,
    OutageConfig,
    coast_points,
    duration_points,
    flicker_points,
    markov_points,
    outage_sweep,
    run_outage_trial,
)
from fusion.outage_metrics import (
    OutageMetrics,
    outage_windows,
    window_births_per_scan,
)
from fusion.sensors.camera import CameraModel
from fusion.sensors.radar import RadarModel
from fusion.tracker.multi_target import run_multi_target_tracking
from fusion.tracker.track import TrackStatus

RADAR = SENSOR_SETS["radar"]
SEPARATED = OUTAGE_SCENARIOS["separated"]
CROSSING = OUTAGE_SCENARIOS["crossing"]
# A clean, fully detected scene: every derived value in the tests below is exact.
SURE = OutageConfig(
    radar_pd=1.0, camera_pd=1.0, radar_clutter_rate=0.0, camera_clutter_rate=0.0, duration=40.0
)


def as_outage_config(config: MttConfig, **overrides) -> OutageConfig:
    values = {f.name: getattr(config, f.name) for f in fields(MttConfig)}
    return OutageConfig(**{**values, **overrides})


# --- configuration -------------------------------------------------------------------


def test_defaults_describe_no_outage_and_an_unaware_tracker():
    """Fails if a default silently adds an outage or the aware policy."""
    config = OutageConfig()
    assert config.dropout is None and config.vanish == ()
    assert config.outage_policy == "unaware" and config.aware_tentatives == "drop"
    assert config.max_coast_time == 15.0 and config.after_window == 30.0
    assert config.tracker_config().outage_policy == "unaware"


def test_tracker_config_carries_the_policy_settings():
    """Fails if the experiment's policy never reaches the tracker."""
    config = OutageConfig(outage_policy="aware", aware_tentatives="freeze", max_coast_time=7.0)
    tracker = config.tracker_config()
    assert (tracker.outage_policy, tracker.aware_tentatives, tracker.max_coast_time) == (
        "aware",
        "freeze",
        7.0,
    )
    assert tracker.min_range == config.fov.range_min  # the Phase 6 fields are still there


@pytest.mark.parametrize(
    "kwargs",
    [
        {"outage_policy": "clairvoyant"},
        {"aware_tentatives": "keep"},
        {"max_coast_time": 0.01},
        {"after_window": -1.0},
        {"dropout": "radar for 4 s"},
        {"dropout": SingleOutage(RADAR, 20.0, 40.0)},  # ends at the run end: no after window
        {"vanish": ((0, 60.0),)},
        {"vanish": ((0, -1.0),)},
        {"burn_in": 60.0},  # inherited validation
    ],
)
def test_invalid_configurations_are_rejected_when_they_are_built(kwargs):
    """Fails if a bad setting is only noticed in a worker process in the middle of a sweep."""
    with pytest.raises(ValueError):
        OutageConfig(**kwargs)


# --- Phase 6 equivalence -------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"dropout": SingleOutage(RADAR, 8.0, 0.0)},
        {"outage_policy": "aware", "dropout": SingleOutage(RADAR, 8.0, 0.0)},
        {"outage_policy": "aware", "aware_tentatives": "freeze", "max_coast_time": 0.5},
    ],
    ids=["no outage", "zero duration", "aware, zero duration", "aware, no outage"],
)
def test_a_run_without_an_outage_reproduces_the_phase_6_trial_bitwise(overrides):
    """Fails if the outage pipeline (masking, flags, evaluation) alters a run without an outage."""
    base = MttConfig(duration=20.0, burn_in=2.0, radar_clutter_rate=3.0, camera_clutter_rate=3.0)
    reference = run_mtt_trial(CROSSING, base, seed=1)
    result = run_outage_trial(CROSSING, as_outage_config(base, **overrides), seed=1)
    assert reference.births_per_scan > 0.5  # precondition: clutter births exist
    for name in reference._fields:
        np.testing.assert_equal(getattr(result, f"run_{name}"), getattr(reference, name))
    assert result.coast_deletions == 0 and result.tentative_drops == 0


def test_a_trial_is_deterministic_per_seed_and_the_dropout_stream_depends_on_it():
    """Fails if the random bursts do not follow the seed (identical or global randomness)."""
    spec = MarkovBursts(RADAR, 8.0, 10.0, 0.3, 2.0)
    config = OutageConfig(duration=30.0, burn_in=2.0, dropout=spec)
    a = run_outage_trial(SEPARATED, config, 1)
    b = run_outage_trial(SEPARATED, config, 1)
    c = run_outage_trial(SEPARATED, config, 2)
    assert isinstance(a, OutageMetrics)
    np.testing.assert_equal(tuple(a), tuple(b))
    assert tuple(a) != tuple(c)


# --- the outage itself does not leak into the past -----------------------------------


def test_everything_before_the_outage_is_identical_whatever_the_outage_or_the_policy():
    """Fails if masking or the dropout stream changes anything before the outage start.

    Clutter rate 5 keeps false tracks and births in the scene, so the equality is not
    trivially zero. The outage starts at 12 s.
    """
    base = OutageConfig(duration=30.0, burn_in=2.0)  # clutter 5 and Pd 0.9 (the defaults)
    variants = [
        replace(base, dropout=SingleOutage(RADAR, 12.0, 8.0)),
        replace(base, dropout=SingleOutage(RADAR, 12.0, 8.0), outage_policy="aware"),
        replace(
            base,
            dropout=SingleOutage(SENSOR_SETS["blackout"], 12.0, 4.0),
            outage_policy="aware",
            aware_tentatives="freeze",
        ),
        replace(base, dropout=MarkovBursts(RADAR, 12.0, 12.0, 0.3, 2.0)),
        replace(base, dropout=PeriodicFlicker(RADAR, 12.0, 12.0, 4.0, 1.0)),
    ]
    results = [run_outage_trial(CROSSING, config, seed=3) for config in variants]
    clean = run_outage_trial(CROSSING, replace(base, dropout=SingleOutage(RADAR, 12.0, 0.0)), 3)
    names = [n for n in OutageMetrics._fields if n.startswith("before_") or n == "nees_before"]
    assert clean.before_births_per_scan > 1.0 and clean.before_false_track_rate >= 0.0
    for result in results:
        for name in names:
            np.testing.assert_equal(getattr(result, name), getattr(clean, name))


# --- what the policies do ------------------------------------------------------------


def test_an_aware_tracker_keeps_the_identities_through_an_outage_that_the_unaware_one_loses():
    """Fails if the policy has no effect on the scores: both would give the same identity_kept.

    Three well separated targets, no clutter, a radar outage of 8 s: the unaware tracker
    deletes every track after six lost scans and re-births them afterwards.
    """
    config = replace(SURE, dropout=SingleOutage(RADAR, 20.0, 8.0))
    unaware = run_outage_trial(SEPARATED, config, 0)
    aware = run_outage_trial(SEPARATED, replace(config, outage_policy="aware"), 0)
    assert unaware.identity_kept == 0.0 and aware.identity_kept == 1.0
    assert aware.reacquired_fraction == 1.0 and aware.reacquisition_time == 0.0
    assert unaware.reacquired_fraction == 1.0 and unaware.reacquisition_time > 0.0
    assert unaware.after_id_switches == 3.0 and aware.after_id_switches == 0.0
    assert aware.during_missed_rate == 0.0 < unaware.during_missed_rate
    assert aware.coast_deletions == 0
    assert 2.0 < aware.nees_outage < 6.0  # consistent: the ideal value is the dimension, 4


def ghost_lifetime(max_coast_time: float, policy: str = "aware") -> float:
    config = replace(
        SURE,
        duration=60.0,
        dropout=SingleOutage(RADAR, 20.0, 24.0),
        vanish=((VANISHING_TARGET, 22.0),),
        outage_policy=policy,
        max_coast_time=max_coast_time,
    )
    return run_outage_trial(OUTAGE_SCENARIOS["vanishing"], config, 0).ghost_lifetime


def test_ghost_lifetime_matches_the_value_derived_from_the_coast_clock():
    """Fails if the coast limit is off by a step, ignores the camera, or is checked outside outages.

    The target ceases to exist at step 220, inside a radar outage that starts at step 200
    and ends at step 440; the camera sees it up to and including step 219. The track is
    then updated for the last time at step 219 and is deleted at the first outage step
    whose coasting exceeds the limit: 219 + 50 + 1 steps for 5 s, so it lives from step 220
    to 270, 5.0 s. With the unaware tracker the lost scans are misses: confirmed with no
    misses at step 190, deleted at the sixth lost scan (step 250), 3.0 s. With a limit
    longer than the outage nothing deletes it until the first radar scans after the outage
    miss it six times (steps 440 .. 490): 27.0 s.
    """
    assert ghost_lifetime(5.0) == pytest.approx(5.0)
    assert ghost_lifetime(10.0) == pytest.approx(10.0)
    assert ghost_lifetime(15.0) == pytest.approx(15.0)
    assert ghost_lifetime(30.0) == pytest.approx(27.0)
    assert ghost_lifetime(15.0, policy="unaware") == pytest.approx(3.0)


@pytest.mark.parametrize("duration", [4.0, 16.0])
def test_the_coast_limit_deletes_exactly_when_it_is_shorter_than_the_blackout(duration):
    """Fails if max_coast_time has no effect, or deletes during outages shorter than the limit.

    In a total blackout nothing updates the tracks, so each of the three targets' tracks is
    deleted once if and only if the limit is shorter than the blackout.
    """
    config = OutageConfig(
        duration=60.0,
        radar_clutter_rate=0.0,
        camera_clutter_rate=0.0,
        outage_policy="aware",
        dropout=SingleOutage(SENSOR_SETS["blackout"], 20.0, duration),
    )
    deletions = {
        limit: run_outage_trial(SEPARATED, replace(config, max_coast_time=limit), 0).coast_deletions
        for limit in (5.0, 10.0, 30.0)
    }
    expected = {limit: 3.0 if limit < duration else 0.0 for limit in deletions}
    assert deletions == expected


# --- sweeps and point builders -------------------------------------------------------


def test_sweep_rows_follow_the_points_and_workers_do_not_change_the_result():
    """Fails if rows are mixed up, or a parallel run differs from the serial one."""
    base = OutageConfig(
        duration=30.0, burn_in=2.0, radar_clutter_rate=0.0, camera_clutter_rate=0.0
    )
    points = duration_points(base, RADAR, 12.0, [0.0, 6.0])
    serial = outage_sweep(SEPARATED, points, [0.0, 6.0], "radar outage [s]", [0, 1])
    parallel = outage_sweep(SEPARATED, points, [0.0, 6.0], "radar outage [s]", [0, 1], workers=2)
    assert serial.parameter == "radar outage [s]"
    np.testing.assert_array_equal(serial.values, [0.0, 6.0])
    assert set(serial.metrics) == set(OutageMetrics._fields)
    assert all(m.shape == (2, 2) for m in serial.metrics.values())
    for name in OutageMetrics._fields:
        np.testing.assert_array_equal(serial.metrics[name], parallel.metrics[name])
    direct = run_outage_trial(SEPARATED, points[1], 1)
    np.testing.assert_equal([serial.metrics[n][1, 1] for n in OutageMetrics._fields], list(direct))
    missed = serial.metrics["during_missed_rate"]
    assert np.isnan(missed[0]).all()  # a zero-length outage has no during window
    assert np.isfinite(missed[1]).all() and (missed[1] > 0.0).all()


def test_outage_sweep_rejects_bad_arguments():
    """Fails if a mismatch of points and values, or empty inputs, are accepted."""
    base = OutageConfig(duration=30.0, burn_in=2.0)
    points = duration_points(base, RADAR, 12.0, [2.0, 4.0])
    with pytest.raises(ValueError, match="one value per point"):
        outage_sweep(SEPARATED, points, [2.0], "x", [0])
    with pytest.raises(ValueError, match="empty"):
        outage_sweep(SEPARATED, points, [2.0, 4.0], "x", [])
    with pytest.raises(ValueError, match="workers"):
        outage_sweep(SEPARATED, points, [2.0, 4.0], "x", [0], workers=0)


def test_point_builders_produce_the_documented_configurations():
    """Fails if a builder swaps arguments (sensor, start, span, off time) or drops a setting."""
    base = OutageConfig(outage_policy="aware", max_coast_time=9.0)
    durations = duration_points(base, SENSOR_SETS["blackout"], 20.0, [0.0, 4.0])
    assert [p.dropout for p in durations] == [
        SingleOutage(("radar", "camera"), 20.0, 0.0),
        SingleOutage(("radar", "camera"), 20.0, 4.0),
    ]
    flicker = flicker_points(base, RADAR, 20.0, 30.0, 10.0, [1.0, 2.0], phase=0.5)
    assert [p.dropout for p in flicker] == [
        PeriodicFlicker(RADAR, 20.0, 30.0, 10.0, 1.0, 0.5),
        PeriodicFlicker(RADAR, 20.0, 30.0, 10.0, 2.0, 0.5),
    ]
    markov = markov_points(base, RADAR, 20.0, 30.0, [0.05, 0.3], mean_burst=2.0)
    assert [p.dropout for p in markov] == [
        MarkovBursts(RADAR, 20.0, 30.0, 0.05, 2.0),
        MarkovBursts(RADAR, 20.0, 30.0, 0.3, 2.0),
    ]
    coast = coast_points(base, [5.0, 30.0])
    assert [p.max_coast_time for p in coast] == [5.0, 30.0]
    assert all(p.outage_policy == "aware" for p in durations + flicker + markov + coast)


def test_births_of_the_windows_add_up_to_the_births_of_the_tracker():
    """Fails if a window double counts or drops births (windows covering the whole run).

    With no burn-in and an after window reaching the end of the run, the three windows
    partition the steps, so births per scan times scans per window summed over the windows
    is the tracker's birth count. Clutter rate 5 makes births frequent.
    """
    config = OutageConfig(duration=30.0, burn_in=0.0, after_window=100.0, outage_policy="aware")
    span = (12.0, 18.0)
    sim = simulate_mtt(CROSSING, config, make_mtt_rngs(2))
    dropped = apply_dropout(sim, SingleOutage(RADAR, *span[:1], 6.0).windows(config.dt), config.dt)
    run = run_multi_target_tracking(
        dropped.sim,
        config.tracker_config(),
        RadarModel(config.radar_range_std, config.radar_bearing_std),
        CameraModel(config.camera_bearing_std),
        dropped.radar_down,
    )
    first_seen = {}
    for k, step in enumerate(run.history):
        for snapshot in step:
            first_seen.setdefault(snapshot.track_id, k)
    scheduled = np.array([s is not None for s in dropped.sim.radar_scans])
    windows = outage_windows(len(run.history), config.dt, 0, span, config.after_window)
    total = 0.0
    for mask in windows.masks().values():
        total += window_births_per_scan(first_seen, scheduled, mask) * np.sum(scheduled & mask)
    assert run.tracker.births > 10  # precondition
    assert total == pytest.approx(run.tracker.births)
    assert any(s.status is TrackStatus.TENTATIVE for step in run.history for s in step)


def test_the_dropout_windows_come_from_the_dropout_stream_of_the_seed(monkeypatch):
    """Fails if the windows are drawn from a fixed or foreign generator instead of the seed's
    "dropout" stream (every seed would then get the same burst pattern)."""
    seen = []
    original = MarkovBursts.windows

    def spy(self, dt, rng=None):
        seen.append(rng.bit_generator.state)
        return original(self, dt, rng)

    monkeypatch.setattr(MarkovBursts, "windows", spy)
    config = OutageConfig(
        duration=30.0, burn_in=2.0, dropout=MarkovBursts(RADAR, 8.0, 10.0, 0.3, 2.0)
    )
    for seed in (5, 6):
        run_outage_trial(SEPARATED, config, seed)
    assert seen == [make_mtt_rngs(seed)["dropout"].bit_generator.state for seed in (5, 6)]


def test_births_per_scan_of_the_run_use_the_scheduled_scans_for_every_policy():
    """Fails if the denominator is the number of scans processed: an aware tracker skips the
    scans of an outage, which would inflate its rate against an unaware tracker."""
    scheduled = int(SURE.duration / (SURE.radar_every * SURE.dt)) + 1
    for policy in ("unaware", "aware"):
        config = replace(SURE, outage_policy=policy, dropout=SingleOutage(RADAR, 20.0, 8.0))
        births = run_outage_trial(SEPARATED, config, 0).run_births_per_scan * scheduled
        assert births == pytest.approx(round(births), abs=1e-9) and births >= 3
