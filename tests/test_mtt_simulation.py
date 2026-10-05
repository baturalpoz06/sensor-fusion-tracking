"""Tests for the multi-target simulation: field of view, clutter, detections, determinism."""

from dataclasses import replace

import numpy as np
import pytest

from fusion.angles import wrap_angle
from fusion.maneuvers import Acceleration, RandomManeuvers, TruthDisturbance, Turn
from fusion.mtt_experiment import SCENARIOS
from fusion.mtt_simulation import (
    RNG_STREAMS,
    FieldOfView,
    SimulationConfig,
    make_mtt_rngs,
    simulate_mtt,
    uniform_clutter,
)
from fusion.sensors.camera import camera_measure
from fusion.sensors.radar import radar_measure, radar_to_cartesian
from tests.test_phase7_regression import PHASE7_STREAMS

TWO_TARGETS = np.array([[-450.0, 600.0, 15.0, 0.0], [400.0, -1400.0, 0.0, 12.0]])
CLEAN = SimulationConfig(
    radar_pd=1.0, camera_pd=1.0, radar_clutter_rate=0.0, camera_clutter_rate=0.0
)


def radar_scans(sim):
    return [s for s in sim.radar_scans if s is not None]


def target_rows(scan):
    """Target measurements of a scan ordered by origin index (clutter dropped)."""
    keep = scan.origin >= 0
    order = np.argsort(scan.origin[keep])
    return scan.origin[keep][order], scan.z[keep][order]


# --- field of view -------------------------------------------------------------------


def test_full_circle_contains_the_minus_x_axis_for_either_zero_sign():
    fov = FieldOfView()
    inside = fov.contains(np.array([[-100.0, 0.0], [-100.0, -0.0], [100.0, 0.0], [0.0, -100.0]]))
    assert inside.all()
    # arctan2 really does return opposite signs here, which is the trap being avoided.
    assert np.arctan2(0.0, -100.0) != np.arctan2(-0.0, -100.0)


def test_range_limits():
    fov = FieldOfView(range_min=50.0, range_max=3000.0)
    points = np.array([[10.0, 0.0], [49.9, 0.0], [50.0, 0.0], [3000.0, 0.0], [3000.1, 0.0]])
    assert fov.contains(points).tolist() == [False, False, True, True, False]


def test_partial_sector_is_centered_and_wraps_through_pi():
    sector = FieldOfView(bearing_center=0.0, bearing_half_width=np.pi / 4)
    angles = np.deg2rad([0.0, 40.0, 50.0, -40.0, -50.0, 180.0])
    points = 500.0 * np.column_stack([np.cos(angles), np.sin(angles)])
    assert sector.contains(points).tolist() == [True, True, False, True, False, False]

    behind = FieldOfView(bearing_center=np.pi, bearing_half_width=0.3)
    angles = np.array([np.pi - 0.1, -np.pi + 0.1, np.pi - 0.5, 0.0])
    points = 500.0 * np.column_stack([np.cos(angles), np.sin(angles)])
    assert behind.contains(points).tolist() == [True, True, False, False]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"range_min": -1.0},
        {"range_min": 100.0, "range_max": 100.0},
        {"bearing_half_width": 0.0},
        {"bearing_half_width": 4.0},
    ],
)
def test_field_of_view_validation(kwargs):
    with pytest.raises(ValueError):
        FieldOfView(**kwargs)


def test_measurement_volume():
    assert FieldOfView(0.0, 100.0).measurement_volume == pytest.approx(100.0 * 2 * np.pi)
    sector = FieldOfView(0.0, 100.0, bearing_half_width=0.5)
    assert sector.measurement_volume == pytest.approx(100.0)


# --- configuration -------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dt": 0.0},
        {"duration": -1.0},
        {"radar_every": 0},
        {"radar_range_std": 0.0},
        {"accel_std": -0.1},
        {"radar_pd": 1.1},
        {"camera_pd": -0.1},
        {"radar_clutter_rate": -1.0},
        {"camera_clutter_rate": -1.0},
    ],
)
def test_simulation_config_validation(kwargs):
    with pytest.raises(ValueError):
        SimulationConfig(**kwargs)


def test_n_steps_follows_duration_and_dt():
    assert SimulationConfig(dt=0.1, duration=60.0).n_steps == 600


# --- random streams ------------------------------------------------------------------


def test_rng_streams_are_reproducible_and_independent():
    first, second = make_mtt_rngs(7), make_mtt_rngs(7)
    assert tuple(first) == RNG_STREAMS
    draws = {name: first[name].random(4) for name in RNG_STREAMS}
    for name in RNG_STREAMS:
        np.testing.assert_array_equal(draws[name], second[name].random(4))
    assert len({tuple(d) for d in draws.values()}) == len(RNG_STREAMS)
    assert not np.array_equal(make_mtt_rngs(8)["trajectory"].random(4), draws["trajectory"])


# --- clutter -------------------------------------------------------------------------


def test_uniform_clutter_count_is_poisson_and_points_are_in_the_field_of_view():
    rng, fov = np.random.default_rng(0), FieldOfView(200.0, 1500.0, 1.0, 0.5)
    batches = [uniform_clutter(rng, 6.0, fov, with_range=True) for _ in range(800)]
    counts = np.array([len(b) for b in batches])
    assert counts.mean() == pytest.approx(6.0, abs=0.35)
    assert counts.var() == pytest.approx(6.0, rel=0.2)
    points = np.concatenate(batches)
    assert points[:, 0].min() >= 200.0 and points[:, 0].max() <= 1500.0
    assert points[:, 1].min() >= 0.5 and points[:, 1].max() <= 1.5
    # Uniform in range: the mean is near the middle of the interval.
    assert points[:, 0].mean() == pytest.approx(850.0, rel=0.03)


def test_clutter_shapes_and_zero_rate():
    rng, fov = np.random.default_rng(0), FieldOfView()
    assert uniform_clutter(rng, 0.0, fov, with_range=True).shape == (0, 2)
    assert uniform_clutter(rng, 0.0, fov, with_range=False).shape == (0, 1)
    assert uniform_clutter(rng, 50.0, fov, with_range=False).shape[1] == 1
    with pytest.raises(ValueError, match="rate"):
        uniform_clutter(rng, -1.0, fov, with_range=True)


def test_full_circle_clutter_covers_both_sides_of_the_seam():
    rng = np.random.default_rng(1)
    bearings = uniform_clutter(rng, 4000.0, FieldOfView(), with_range=False)[:, 0]
    assert bearings.min() < -3.0 and bearings.max() > 3.0
    assert bearings.min() >= -np.pi and bearings.max() <= np.pi


# --- simulate_mtt --------------------------------------------------------------------


def test_same_seed_gives_identical_scans_and_different_seed_does_not():
    config = SimulationConfig()
    a = simulate_mtt(TWO_TARGETS, config, make_mtt_rngs(3))
    b = simulate_mtt(TWO_TARGETS, config, make_mtt_rngs(3))
    c = simulate_mtt(TWO_TARGETS, config, make_mtt_rngs(4))
    np.testing.assert_array_equal(a.truth, b.truth)
    for sa, sb in zip(a.radar_scans, b.radar_scans, strict=True):
        assert (sa is None) == (sb is None)
        if sa is not None:
            np.testing.assert_array_equal(sa.z, sb.z)
            np.testing.assert_array_equal(sa.origin, sb.origin)
    for sa, sb in zip(a.camera_scans, b.camera_scans, strict=True):
        np.testing.assert_array_equal(sa.z, sb.z)
    assert not np.array_equal(a.truth, c.truth)
    assert not np.array_equal(a.camera_scans[5].z, c.camera_scans[5].z)


def test_output_structure():
    config = SimulationConfig(duration=10.0)
    sim = simulate_mtt(TWO_TARGETS, config, make_mtt_rngs(0))
    assert sim.truth.shape == (2, 101, 4)
    assert len(sim.radar_scans) == len(sim.camera_scans) == 101
    assert [k for k, s in enumerate(sim.radar_scans) if s is not None] == list(range(0, 101, 10))
    assert all(s.z.shape[1] == 1 for s in sim.camera_scans)
    assert all(s.z.shape[1] == 2 for s in radar_scans(sim))


def test_initial_states_shape_is_validated():
    with pytest.raises(ValueError, match="shape"):
        simulate_mtt(np.zeros(4), SimulationConfig(), make_mtt_rngs(0))
    with pytest.raises(ValueError, match="shape"):
        simulate_mtt(np.zeros((2, 3)), SimulationConfig(), make_mtt_rngs(0))


def test_perfect_detection_without_clutter_is_the_target_noise_of_radar_measure():
    sim = simulate_mtt(TWO_TARGETS, CLEAN, make_mtt_rngs(5))
    noise_rng = make_mtt_rngs(5)["radar_noise"]
    expected = [
        radar_measure(states, CLEAN.radar_range_std, CLEAN.radar_bearing_std, rng=noise_rng)
        for states in sim.truth
    ]
    camera_rng = make_mtt_rngs(5)["camera_noise"]
    expected_camera = [
        camera_measure(states, CLEAN.camera_bearing_std, rng=camera_rng) for states in sim.truth
    ]
    for k, scan in enumerate(sim.radar_scans):
        if scan is None:
            continue
        origin, z = target_rows(scan)
        assert origin.tolist() == [0, 1]
        np.testing.assert_array_equal(z, [expected[0][k], expected[1][k]])
    for k, scan in enumerate(sim.camera_scans):
        origin, z = target_rows(scan)
        assert origin.tolist() == [0, 1]
        np.testing.assert_array_equal(z, [expected_camera[0][k], expected_camera[1][k]])


def test_zero_pd_gives_clutter_only_and_empty_scans_keep_their_shape():
    config = replace(CLEAN, radar_pd=0.0, camera_pd=0.0, radar_clutter_rate=4.0)
    sim = simulate_mtt(TWO_TARGETS, config, make_mtt_rngs(0))
    assert all((s.origin == -1).all() for s in radar_scans(sim))
    assert any(len(s.origin) for s in radar_scans(sim))
    for scan in sim.camera_scans:
        assert scan.z.shape == (0, 1) and scan.origin.shape == (0,)
    empty_radar = radar_scans(
        simulate_mtt(TWO_TARGETS, replace(config, radar_clutter_rate=0.0), make_mtt_rngs(0))
    )
    assert all(s.z.shape == (0, 2) for s in empty_radar)


def test_zero_clutter_rate_gives_no_clutter():
    sim = simulate_mtt(TWO_TARGETS, CLEAN, make_mtt_rngs(0))
    assert all((s.origin >= 0).all() for s in radar_scans(sim))
    assert all((s.origin >= 0).all() for s in sim.camera_scans)


def test_mean_clutter_count_matches_the_rate_and_lies_in_the_field_of_view():
    config = replace(
        CLEAN, radar_pd=0.0, radar_every=1, radar_clutter_rate=5.0, camera_clutter_rate=3.0
    )
    sim = simulate_mtt(TWO_TARGETS, config, make_mtt_rngs(11))
    radar_counts = np.array([len(s.origin) for s in radar_scans(sim)])
    camera_counts = np.array([len(s.origin) for s in sim.camera_scans])
    assert radar_counts.mean() == pytest.approx(5.0, abs=0.4)
    assert len(radar_counts) == 601
    # The camera has Pd = 1 for both targets, so subtract them.
    assert (camera_counts - 2).mean() == pytest.approx(3.0, abs=0.4)
    z = np.concatenate([s.z for s in radar_scans(sim)])
    assert z[:, 0].min() >= config.fov.range_min and z[:, 0].max() <= config.fov.range_max


def test_detection_frequency_matches_pd():
    config = replace(CLEAN, radar_pd=0.7, radar_every=1)
    sim = simulate_mtt(SCENARIOS["separated"], config, make_mtt_rngs(2))
    detections = sum(len(s.origin) for s in radar_scans(sim))
    assert detections / (3 * 601) == pytest.approx(0.7, abs=0.04)


def test_changing_clutter_rate_leaves_truth_and_target_detections_unchanged():
    base = simulate_mtt(TWO_TARGETS, replace(CLEAN, radar_pd=0.8, radar_every=1), make_mtt_rngs(9))
    noisy_config = replace(
        CLEAN, radar_pd=0.8, radar_every=1, radar_clutter_rate=12.0, camera_clutter_rate=9.0
    )
    noisy = simulate_mtt(TWO_TARGETS, noisy_config, make_mtt_rngs(9))
    np.testing.assert_array_equal(base.truth, noisy.truth)
    for a, b in zip(base.radar_scans, noisy.radar_scans, strict=True):
        oa, za = target_rows(a)
        ob, zb = target_rows(b)
        np.testing.assert_array_equal(oa, ob)
        np.testing.assert_array_equal(za, zb)
    n_noisy = sum(len(s.origin) for s in noisy.radar_scans)
    assert n_noisy > sum(len(s.origin) for s in base.radar_scans)


def test_changing_radar_clutter_rate_does_not_change_the_camera_scans():
    config = replace(CLEAN, camera_clutter_rate=4.0)
    a = simulate_mtt(TWO_TARGETS, config, make_mtt_rngs(1))
    b = simulate_mtt(TWO_TARGETS, replace(config, radar_clutter_rate=15.0), make_mtt_rngs(1))
    for sa, sb in zip(a.camera_scans, b.camera_scans, strict=True):
        np.testing.assert_array_equal(sa.z, sb.z)
        np.testing.assert_array_equal(sa.origin, sb.origin)


def test_lower_pd_only_removes_detections():
    high = simulate_mtt(TWO_TARGETS, replace(CLEAN, radar_pd=0.9, radar_every=1), make_mtt_rngs(6))
    low = simulate_mtt(TWO_TARGETS, replace(CLEAN, radar_pd=0.5, radar_every=1), make_mtt_rngs(6))
    np.testing.assert_array_equal(high.truth, low.truth)
    n_high = n_low = 0
    for a, b in zip(high.radar_scans, low.radar_scans, strict=True):
        oa, za = target_rows(a)
        ob, zb = target_rows(b)
        assert set(ob) <= set(oa)
        for origin, z in zip(ob, zb, strict=True):
            np.testing.assert_array_equal(z, za[list(oa).index(origin)])
        n_high, n_low = n_high + len(oa), n_low + len(ob)
    assert n_low < n_high


def test_scans_are_shuffled_so_clutter_is_not_always_last():
    config = replace(CLEAN, radar_every=1, radar_clutter_rate=4.0)
    sim = simulate_mtt(TWO_TARGETS, config, make_mtt_rngs(0))
    clutter_before_target = sum(
        1
        for s in radar_scans(sim)
        if (s.origin >= 0).any()
        and (s.origin == -1).any()
        and np.flatnonzero(s.origin == -1).min() < np.flatnonzero(s.origin >= 0).max()
    )
    assert clutter_before_target > 100


def test_a_target_outside_the_field_of_view_is_never_detected():
    far = np.array([[5000.0, 0.0, 0.0, 0.0], [-20.0, 0.0, 0.0, 0.0]])
    sim = simulate_mtt(far, replace(CLEAN, accel_std=0.0), make_mtt_rngs(0))
    assert all(len(s.origin) == 0 for s in radar_scans(sim))
    assert all(len(s.origin) == 0 for s in sim.camera_scans)


def test_a_target_leaving_the_field_of_view_stops_being_detected():
    config = replace(CLEAN, accel_std=0.0, radar_every=1, duration=20.0)
    sim = simulate_mtt(np.array([[2900.0, 0.0, 20.0, 0.0]]), config, make_mtt_rngs(0))
    detected = np.array([len(s.origin) for s in radar_scans(sim)])
    r = sim.truth[0, :, 0]
    np.testing.assert_array_equal(detected, (r <= 3000.0).astype(int))


def test_no_targets_gives_clutter_only_scans():
    config = replace(CLEAN, radar_clutter_rate=5.0, duration=5.0)
    sim = simulate_mtt(np.zeros((0, 4)), config, make_mtt_rngs(0))
    assert sim.truth.shape == (0, 51, 4)
    assert any(len(s.origin) for s in radar_scans(sim))
    assert all((s.origin == -1).all() for s in radar_scans(sim))


# --- scenarios -----------------------------------------------------------------------


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_scenario_targets_stay_in_the_field_of_view_and_away_from_the_sensor(name):
    config = SimulationConfig()
    for seed in range(10):
        truth = simulate_mtt(SCENARIOS[name], config, make_mtt_rngs(seed)).truth
        ranges = np.hypot(truth[:, :, 0], truth[:, :, 1])
        assert ranges.min() > 250.0, (name, seed)
        assert config.fov.contains(truth[:, :, :2].reshape(-1, 2)).all()


def test_crossing_scenario_has_a_target_that_crosses_the_minus_x_axis():
    config = SimulationConfig()
    for seed in range(10):
        truth = simulate_mtt(SCENARIOS["crossing"], config, make_mtt_rngs(seed)).truth
        wrapper = truth[2]
        assert (wrapper[:, 0] < 0).all()
        assert wrapper[0, 1] > 0 > wrapper[-1, 1]


# --- truth disturbances (Phase 8a) -----------------------------------------------------

SEPARATED = SCENARIOS["separated"]
SCENE30 = SimulationConfig(duration=30.0, radar_clutter_rate=5.0, camera_clutter_rate=5.0)
DEG = np.pi / 180.0
DISTURBANCES = {
    "bias": TruthDisturbance(camera_bias=0.5 * DEG),
    "time offset": TruthDisturbance(camera_time_offset=0.05),
    "lever arm": TruthDisturbance(camera_position=(20.0, 0.0)),
    "turn": TruthDisturbance(maneuvers=tuple((i, Turn(15.0, 25.0, 10 * DEG)) for i in range(3))),
    "acceleration": TruthDisturbance(
        maneuvers=tuple((i, Acceleration(15.0, 18.0, 2.0)) for i in range(3))
    ),
    "random": TruthDisturbance(random_maneuvers=RandomManeuvers(10.0, 5.0, 10 * DEG)),
}
CAMERA_ONLY = ("bias", "time offset", "lever arm")


def scan_arrays(scans):
    return [None if s is None else (s.z, s.origin) for s in scans]


def assert_scans_equal(a, b):
    assert len(a) == len(b)
    for x, y in zip(scan_arrays(a), scan_arrays(b), strict=True):
        assert (x is None) == (y is None)
        if x is not None:
            np.testing.assert_array_equal(x[0], y[0])
            np.testing.assert_array_equal(x[1], y[1])


def clutter_rows(scans):
    return [s.z[s.origin < 0] for s in scans if s is not None]


def sim_pair(disturbance, seed=3, config=SCENE30, states=SEPARATED):
    base = simulate_mtt(states, config, make_mtt_rngs(seed))
    rngs = make_mtt_rngs(seed)
    return base, simulate_mtt(states, config, rngs, disturbance), rngs


def test_random_streams_append_the_maneuver_stream_at_the_end():
    """Fails if the maneuver stream is inserted before an existing one (all draws would shift)."""
    assert RNG_STREAMS[-1] == "maneuver" and len(RNG_STREAMS) == 11
    assert RNG_STREAMS[:10] == PHASE7_STREAMS


def test_a_neutral_disturbance_is_the_same_as_none_bitwise():
    """Fails if passing the default disturbance changes any result."""
    a = simulate_mtt(SEPARATED, SCENE30, make_mtt_rngs(2))
    b = simulate_mtt(SEPARATED, SCENE30, make_mtt_rngs(2), TruthDisturbance())
    np.testing.assert_array_equal(a.truth, b.truth)
    assert_scans_equal(a.radar_scans, b.radar_scans)
    assert_scans_equal(a.camera_scans, b.camera_scans)


def test_the_camera_bias_reaches_the_camera_scans_and_nothing_else():
    """Fails if simulate_mtt drops the bias, applies it to the radar, or gives it the wrong sign."""
    bias = 0.01
    base = simulate_mtt(TWO_TARGETS, CLEAN, make_mtt_rngs(1))
    biased = simulate_mtt(TWO_TARGETS, CLEAN, make_mtt_rngs(1), TruthDisturbance(camera_bias=bias))
    np.testing.assert_array_equal(base.truth, biased.truth)
    assert_scans_equal(base.radar_scans, biased.radar_scans)
    for a, b in zip(base.camera_scans, biased.camera_scans, strict=True):
        _, za = target_rows(a)
        _, zb = target_rows(b)
        np.testing.assert_allclose(wrap_angle(zb - za), bias, rtol=0.0, atol=8 * np.spacing(np.pi))


def test_the_camera_position_reaches_the_camera_scans_and_nothing_else():
    """Fails if simulate_mtt drops the camera position or moves the radar with it."""
    position = (30.0, -20.0)
    base = simulate_mtt(TWO_TARGETS, CLEAN, make_mtt_rngs(1))
    moved = simulate_mtt(
        TWO_TARGETS, CLEAN, make_mtt_rngs(1), TruthDisturbance(camera_position=position)
    )
    assert_scans_equal(base.radar_scans, moved.radar_scans)
    for k, (a, b) in enumerate(zip(base.camera_scans, moved.camera_scans, strict=True)):
        _, za = target_rows(a)
        _, zb = target_rows(b)
        states = base.truth[:, k]
        shift = np.arctan2(states[:, 1] - position[1], states[:, 0] - position[0]) - np.arctan2(
            states[:, 1], states[:, 0]
        )
        np.testing.assert_allclose(wrap_angle(zb[:, 0] - za[:, 0]), wrap_angle(shift), atol=1e-12)


def test_the_camera_time_offset_makes_the_camera_see_the_past():
    """Fails if the offset is dropped, applied to the radar, or looks into the future.

    With the noise held equal by the seed, the camera measurement differs from the undelayed
    one by the bearing of the truth 0.1 s earlier minus the bearing of the truth now.
    """
    delay = 0.1  # one whole step: the delayed truth is the previous grid state
    base = simulate_mtt(TWO_TARGETS, CLEAN, make_mtt_rngs(1))
    late = simulate_mtt(
        TWO_TARGETS, CLEAN, make_mtt_rngs(1), TruthDisturbance(camera_time_offset=delay)
    )
    np.testing.assert_array_equal(base.truth, late.truth)
    assert_scans_equal(base.radar_scans, late.radar_scans)
    for k in range(2, 30):
        _, za = target_rows(base.camera_scans[k])
        _, zb = target_rows(late.camera_scans[k])
        past, now = base.truth[:, k - 1], base.truth[:, k]
        shift = np.arctan2(past[:, 1], past[:, 0]) - np.arctan2(now[:, 1], now[:, 0])
        np.testing.assert_allclose(wrap_angle(zb[:, 0] - za[:, 0]), wrap_angle(shift), atol=1e-12)
        future = base.truth[:, k + 1]
        wrong = np.arctan2(future[:, 1], future[:, 0]) - np.arctan2(now[:, 1], now[:, 0])
        assert not np.allclose(wrap_angle(zb[:, 0] - za[:, 0]), wrap_angle(wrong), atol=1e-6)


def test_a_maneuver_changes_only_its_target_and_the_truth_the_sensors_see():
    """Fails if a schedule hits the wrong target, or the scans are not made from the moved truth."""
    turn = TruthDisturbance(maneuvers=((0, Turn(5.0, 10.0, 20 * DEG)),))
    base = simulate_mtt(TWO_TARGETS, CLEAN, make_mtt_rngs(1), None)
    turned = simulate_mtt(TWO_TARGETS, CLEAN, make_mtt_rngs(1), turn)
    np.testing.assert_array_equal(turned.truth[1], base.truth[1])
    np.testing.assert_array_equal(turned.truth[0, :51], base.truth[0, :51])
    gap = np.hypot(*(turned.truth[0, 200, :2] - base.truth[0, 200, :2]))
    assert gap > 200.0  # precondition: the turn moved the target far from its straight path
    # The radar scan at that step is built from the turned truth, not from the straight one.
    _, z = target_rows(turned.radar_scans[200])
    measured = radar_to_cartesian(z[:1])[0]
    assert np.hypot(*(measured - turned.truth[0, 200, :2])) < 0.5 * gap
    assert np.hypot(*(measured - base.truth[0, 200, :2])) > 0.5 * gap


def test_a_maneuver_for_an_unknown_target_is_refused():
    """Fails if a schedule for a target that does not exist is silently ignored."""
    disturbance = TruthDisturbance(maneuvers=((5, Turn(1.0, 2.0, 0.1)),))
    with pytest.raises(ValueError, match="unknown target"):
        simulate_mtt(TWO_TARGETS, CLEAN, make_mtt_rngs(0), disturbance)


@pytest.mark.parametrize("name", list(DISTURBANCES))
def test_a_disturbance_leaves_every_random_stream_but_maneuver_untouched(name):
    """Fails if enabling a disturbance consumes or shifts the clutter, detection, noise,
    shuffle, trajectory or dropout streams (Phase 8a RNG isolation).

    The precondition that no target crosses the field of view is checked first: only then are
    the detected counts, and with them the shuffle draws, the same.
    """
    base, disturbed, rngs = sim_pair(DISTURBANCES[name])
    reference = make_mtt_rngs(3)
    simulate_mtt(SEPARATED, SCENE30, reference)
    fov_base = SCENE30.fov.contains(base.truth[:, :, :2].reshape(-1, 2))
    fov_disturbed = SCENE30.fov.contains(disturbed.truth[:, :, :2].reshape(-1, 2))
    np.testing.assert_array_equal(fov_base, fov_disturbed)
    for stream in RNG_STREAMS:
        if stream != "maneuver":
            assert rngs[stream].bit_generator.state == reference[stream].bit_generator.state, stream
    for a, b in zip(
        clutter_rows(base.radar_scans) + clutter_rows(base.camera_scans),
        clutter_rows(disturbed.radar_scans) + clutter_rows(disturbed.camera_scans),
        strict=True,
    ):
        np.testing.assert_array_equal(a, b)
    detected = [s.origin for s in base.camera_scans], [s.origin for s in disturbed.camera_scans]
    for a, b in zip(*detected, strict=True):
        np.testing.assert_array_equal(a, b)  # same targets detected, in the same order


@pytest.mark.parametrize("name", CAMERA_ONLY)
def test_camera_disturbances_do_not_change_truth_or_radar_scans_bitwise(name):
    """Fails if a camera disturbance leaks into the truth or the radar data."""
    base, disturbed, _ = sim_pair(DISTURBANCES[name])
    np.testing.assert_array_equal(base.truth, disturbed.truth)
    assert_scans_equal(base.radar_scans, disturbed.radar_scans)


def test_the_maneuver_stream_is_read_only_for_random_maneuvers():
    """Fails if a disturbance without random maneuvers consumes the maneuver stream."""
    reference = make_mtt_rngs(3)["maneuver"].bit_generator.state
    for name, disturbance in DISTURBANCES.items():
        _, _, rngs = sim_pair(disturbance)
        changed = rngs["maneuver"].bit_generator.state != reference
        assert changed == (name == "random"), name


def test_a_disturbed_simulation_is_deterministic_per_seed():
    """Fails if a disturbance, random maneuvers included, does not follow the seed."""
    d = DISTURBANCES["random"]
    a = simulate_mtt(SEPARATED, SCENE30, make_mtt_rngs(4), d)
    b = simulate_mtt(SEPARATED, SCENE30, make_mtt_rngs(4), d)
    c = simulate_mtt(SEPARATED, SCENE30, make_mtt_rngs(5), d)
    np.testing.assert_array_equal(a.truth, b.truth)
    assert_scans_equal(a.camera_scans, b.camera_scans)
    assert not np.array_equal(a.truth, c.truth)
