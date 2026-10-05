"""Tests for the robustness experiment: truth and belief, trials, sweeps, rows and layouts.

Each test docstring names the condition that makes it fail without the feature.
"""

from dataclasses import replace

import numpy as np
import pytest

from fusion.filters.ekf import ExtendedKalmanFilter
from fusion.maneuvers import Acceleration, RandomManeuvers, TruthDisturbance, Turn
from fusion.mtt_experiment import SCENARIOS, MttConfig, SweepResult, run_mtt_trial
from fusion.mtt_simulation import SimulationConfig, make_mtt_rngs, simulate_mtt
from fusion.robustness_experiment import (
    DIRECTIONS,
    GEOMETRY_SPEED,
    NOISE_KNOBS,
    REFERENCE_TIME,
    ZONES,
    GeometryLayout,
    RobustnessConfig,
    Row,
    TrackerBelief,
    camera_rows,
    geometry_rows,
    lever_arm_rows,
    maneuver_rows,
    noise_scale_rows,
    robustness_sweep,
    run_robustness_trial,
    simulate_and_track,
)
from fusion.robustness_metrics import FIELDS, RobustnessMetrics
from fusion.sensors.radar import radar_initial_estimate
from fusion.tracker.multi_target import MultiTargetTracker, run_multi_target_tracking

DEG = np.pi / 180.0
SEPARATED = SCENARIOS["separated"]
CROSSING = SCENARIOS["crossing"]
SHORT = RobustnessConfig(
    duration=20.0, burn_in=2.0, radar_clutter_rate=3.0, camera_clutter_rate=3.0
)
CLEAN = RobustnessConfig(
    duration=20.0,
    burn_in=2.0,
    radar_pd=1.0,
    camera_pd=1.0,
    radar_clutter_rate=0.0,
    camera_clutter_rate=0.0,
)


# --- belief and configuration ----------------------------------------------------------


def test_defaults_are_neutral():
    """Fails if a default silently adds a disturbance, a scaled noise or a displaced camera."""
    config = RobustnessConfig()
    assert config.truth == TruthDisturbance() and config.belief == TrackerBelief()
    belief = config.belief
    scales = (
        belief.radar_range_scale,
        belief.radar_bearing_scale,
        belief.camera_bearing_scale,
        belief.process_noise_scale,
    )
    assert scales == (1.0, 1.0, 1.0, 1.0) and belief.assumed_camera_position == (0.0, 0.0)
    assert config.focus_window is None and config.diagnostic_match_distance == 200.0


def test_the_neutral_tracker_setup_is_the_phase_6_tracker_bitwise():
    """Fails if the neutral belief changes a noise level or the tracker parameters by one bit."""
    config = RobustnessConfig()
    tracker_config, radar, camera = config.tracker_setup()
    plain = MttConfig()
    assert tracker_config == plain.tracker_config()
    np.testing.assert_array_equal(
        radar.R, np.diag([plain.radar_range_std**2, plain.radar_bearing_std**2])
    )
    np.testing.assert_array_equal(camera.R, [[plain.camera_bearing_std**2]])
    assert camera.position == (0.0, 0.0)
    assert RobustnessConfig(use_camera=False).tracker_setup()[2] is None


def test_scales_reach_the_tracker_models_each_knob_separately():
    """Fails if a scale is dropped, applied to the wrong knob, or applied to the variance.

    The four scales are all different, so swapping two of them is caught.
    """
    belief = TrackerBelief(
        radar_range_scale=0.5,
        radar_bearing_scale=2.0,
        camera_bearing_scale=0.25,
        process_noise_scale=4.0,
    )
    config = RobustnessConfig(belief=belief)
    tracker_config, radar, camera = config.tracker_setup()
    np.testing.assert_allclose(
        radar.R,
        np.diag([(0.5 * config.radar_range_std) ** 2, (2.0 * config.radar_bearing_std) ** 2]),
    )
    np.testing.assert_allclose(camera.R, [[(0.25 * config.camera_bearing_std) ** 2]])
    assert tracker_config.accel_std == pytest.approx(4.0 * config.accel_std)
    assert tracker_config.velocity_std == config.velocity_std  # the prior velocity is not scaled


def test_a_live_track_uses_the_scaled_process_noise_and_birth_covariance():
    """Fails if the scales stop at the config: the track's Q and birth P0 must follow them."""
    belief = TrackerBelief(radar_range_scale=2.0, radar_bearing_scale=0.5, process_noise_scale=3.0)
    config = RobustnessConfig(belief=belief)
    tracker = MultiTargetTracker(*config.tracker_setup())
    tracker.step(np.array([[1000.0, 0.3]]), None)
    track = tracker.tracks[0]
    expected = ExtendedKalmanFilter(config.dt, 3.0 * config.accel_std, np.zeros(4), np.eye(4))
    np.testing.assert_allclose(track.filter.Q, expected.Q, rtol=1e-12)
    x0, p0 = radar_initial_estimate(
        np.array([1000.0, 0.3]),
        2.0 * config.radar_range_std,
        0.5 * config.radar_bearing_std,
        config.velocity_std,
    )
    np.testing.assert_allclose(track.filter.x, x0, rtol=1e-12)
    np.testing.assert_allclose(track.filter.P, p0, rtol=1e-12)


def test_the_assumed_camera_position_reaches_the_camera_model_known_and_unknown():
    """Fails if the known and unknown calibration are swapped or the position never arrives."""
    place = (20.0, 5.0)
    unknown = RobustnessConfig(truth=TruthDisturbance(camera_position=place))
    known = replace(unknown, belief=TrackerBelief(assumed_camera_position=place))
    assert unknown.tracker_setup()[2].position == (0.0, 0.0)
    assert known.tracker_setup()[2].position == place


def test_explicitly_set_neutral_values_equal_the_defaults():
    """Fails if a neutral value spelled out differs from the default through a hidden path."""
    explicit = RobustnessConfig(
        truth=TruthDisturbance(
            camera_position=(0.0, 0.0), camera_bias=0.0, camera_time_offset=0.0, maneuvers=()
        ),
        belief=TrackerBelief(1.0, 1.0, 1.0, 1.0, (0.0, 0.0)),
    )
    assert explicit == RobustnessConfig()


@pytest.mark.parametrize(
    "build",
    [
        lambda: TrackerBelief(radar_range_scale=0.0),
        lambda: TrackerBelief(radar_bearing_scale=-1.0),
        lambda: TrackerBelief(camera_bearing_scale=float("nan")),
        lambda: TrackerBelief(process_noise_scale=float("inf")),
        lambda: TrackerBelief(assumed_camera_position=(1.0,)),
        lambda: RobustnessConfig(diagnostic_match_distance=20.0),
        lambda: RobustnessConfig(focus_window=(30.0, 20.0)),
        lambda: RobustnessConfig(focus_window=(10.0, 80.0)),
        lambda: RobustnessConfig(burn_in=100.0),  # inherited validation
    ],
)
def test_invalid_beliefs_and_configurations_are_rejected_when_they_are_built(build):
    """Fails if a bad setting is only noticed in a worker process in the middle of a sweep."""
    with pytest.raises(ValueError):
        build()


def test_from_mtt_copies_every_phase_6_field_and_applies_overrides():
    """Fails if a Phase 6 setting is lost when a configuration is converted."""
    base = MttConfig(duration=30.0, radar_pd=0.7, radar_clutter_rate=2.0, use_camera=False)
    config = RobustnessConfig.from_mtt(base, burn_in=3.0)
    assert (config.duration, config.radar_pd, config.radar_clutter_rate) == (30.0, 0.7, 2.0)
    assert config.use_camera is False and config.burn_in == 3.0


# --- trials ----------------------------------------------------------------------------


@pytest.mark.parametrize("scenario", ["crossing", "separated"])
def test_a_neutral_trial_reproduces_the_phase_6_trial_bitwise(scenario):
    """Fails if the robustness pipeline alters a run without a disturbance or a wrong belief."""
    base = MttConfig(duration=20.0, burn_in=2.0, radar_clutter_rate=3.0, camera_clutter_rate=3.0)
    reference = run_mtt_trial(SCENARIOS[scenario], base, seed=1)
    result = run_robustness_trial(SCENARIOS[scenario], RobustnessConfig.from_mtt(base), seed=1)
    assert reference.births_per_scan > 0.5  # precondition: clutter births exist
    for name in reference._fields:
        np.testing.assert_equal(getattr(result, f"run_{name}"), getattr(reference, name))


def test_a_trial_is_deterministic_per_seed():
    """Fails if a trial depends on anything but its seed (random maneuvers included)."""
    turn = TruthDisturbance(random_maneuvers=RandomManeuvers(5.0, 5.0, 10 * DEG))
    config = replace(SHORT, truth=turn)
    a = run_robustness_trial(SEPARATED, config, 1)
    b = run_robustness_trial(SEPARATED, config, 1)
    c = run_robustness_trial(SEPARATED, config, 2)
    assert isinstance(a, RobustnessMetrics)
    np.testing.assert_equal(tuple(a), tuple(b))
    assert tuple(a) != tuple(c)


def test_the_diagnostics_do_not_change_the_tracker():
    """Fails if running with the camera log changes any track: the log is read by nobody.

    The tracks of a trial are compared with those of a tracker run by hand on the same scans
    (which has the same log), step by step, bit for bit.
    """
    sim, run = simulate_and_track(CROSSING, SHORT, 3)
    again = run_multi_target_tracking(sim, *SHORT.tracker_setup())
    for a, b in zip(run.history, again.history, strict=True):
        assert [(s.track_id, s.status) for s in a] == [(s.track_id, s.status) for s in b]
        for sa, sb in zip(a, b, strict=True):
            np.testing.assert_array_equal(sa.x, sb.x)
            np.testing.assert_array_equal(sa.P, sb.P)


def test_a_camera_bias_shows_up_as_a_positive_cross_range_error_and_neutral_does_not():
    """Fails if the truth disturbance never reaches the trial, or the cross-range sign is lost.

    A clean scene with a 1 degree bias: the fused tracks are pulled counter-clockwise, which
    is a positive cross-range error; without the bias the mean cross error stays small.
    """
    neutral = run_robustness_trial(SEPARATED, CLEAN, 0)
    biased = run_robustness_trial(
        SEPARATED, replace(CLEAN, truth=TruthDisturbance(camera_bias=1.0 * DEG)), 0
    )
    assert biased.cross_mean > 10.0 > abs(neutral.cross_mean)
    assert biased.run_position_rmse > neutral.run_position_rmse


def test_a_wrong_noise_belief_reaches_the_scores_but_not_the_data():
    """Fails if a belief changes the simulated world, or has no effect on the tracker."""
    narrow = replace(
        CLEAN, belief=TrackerBelief(camera_bearing_scale=0.25, radar_bearing_scale=0.25)
    )
    sim_a, run_a = simulate_and_track(SEPARATED, CLEAN, 0)
    sim_b, run_b = simulate_and_track(SEPARATED, narrow, 0)
    np.testing.assert_array_equal(sim_a.truth, sim_b.truth)
    for a, b in zip(sim_a.camera_scans, sim_b.camera_scans, strict=True):
        np.testing.assert_array_equal(a.z, b.z)
    last_a, last_b = run_a.history[-1], run_b.history[-1]
    assert [s.P[0, 0] for s in last_a] != pytest.approx([s.P[0, 0] for s in last_b])


# --- sweeps ----------------------------------------------------------------------------


def test_a_sweep_has_one_row_per_configuration_and_the_fields_of_the_metrics():
    """Fails if rows or seeds are mixed up in the result array, or a field is missing."""
    rows = [
        Row("neutral", 0.0, SHORT),
        Row("bias", 1.0, replace(SHORT, belief=TrackerBelief(camera_bearing_scale=2.0))),
    ]
    result = robustness_sweep(rows, SEPARATED, "scale", [0, 1, 2], workers=1)
    assert isinstance(result, SweepResult) and result.parameter == "scale"
    np.testing.assert_array_equal(result.values, [0.0, 1.0])
    assert set(result.metrics) == set(FIELDS)
    assert all(values.shape == (2, 3) for values in result.metrics.values())
    direct = run_robustness_trial(SEPARATED, rows[1].config, 2)
    np.testing.assert_equal([result.metrics[name][1, 2] for name in FIELDS], list(direct))


def test_a_row_may_bring_its_own_scene_and_the_worker_count_does_not_change_results():
    """Fails if per-row initial states are ignored, or parallel runs differ from serial ones."""
    other = SEPARATED[:2]
    rows = [Row("a", 0.0, SHORT), Row("b", 1.0, SHORT, None, other)]
    serial = robustness_sweep(rows, SEPARATED, "scene", [0, 1], workers=1)
    parallel = robustness_sweep(rows, SEPARATED, "scene", [0, 1], workers=2)
    for name in FIELDS:
        np.testing.assert_array_equal(serial.metrics[name], parallel.metrics[name])
    direct = run_robustness_trial(other, SHORT, 1)
    assert serial.metrics["run_births_per_scan"][1, 1] == direct.run_births_per_scan


@pytest.mark.parametrize("args", [([], [0]), ([Row("a", 0.0, SHORT)], [])])
def test_a_sweep_without_rows_or_seeds_is_refused(args):
    """Fails if an empty sweep silently returns an empty result."""
    with pytest.raises(ValueError, match="must not be empty"):
        robustness_sweep(args[0], SEPARATED, "x", args[1])
    with pytest.raises(ValueError, match="workers"):
        robustness_sweep([Row("a", 0.0, SHORT)], SEPARATED, "x", [0], workers=0)


# --- rows ------------------------------------------------------------------------------


def labels(rows):
    return [row.label for row in rows]


def test_noise_rows_scale_one_knob_at_a_time_around_one_neutral_row():
    """Fails if a knob is scaled together with another, or the neutral row is duplicated."""
    rows = noise_scale_rows(SHORT)
    assert len(rows) == 17 and rows[0].label == "neutral" and rows[0].config == SHORT
    assert len(set(labels(rows))) == 17
    for row in rows[1:]:
        belief = row.config.belief
        changed = {
            name: getattr(belief, name)
            for name in (
                "radar_range_scale",
                "radar_bearing_scale",
                "camera_bearing_scale",
                "process_noise_scale",
            )
            if getattr(belief, name) != 1.0
        }
        assert list(changed.values()) == [row.x] and len(changed) == 1
        assert row.config.truth == TruthDisturbance()
    assert sorted({row.group for row in rows[1:]}) == sorted(NOISE_KNOBS)


def test_maneuver_rows_give_every_target_the_maneuver_in_the_right_units():
    """Fails if turn rates are not converted to rad/s, a target is left out, or windows differ."""
    rows = maneuver_rows(SHORT, n_targets=3)
    assert len(rows) == 10 and rows[0].config == replace(SHORT, focus_window=(15.0, 20.0))
    turn = rows[1]
    assert turn.group == "turn" and turn.x == 5.0
    maneuvers = turn.config.truth.maneuvers
    assert [target for target, _ in maneuvers] == [0, 1, 2]
    assert all(m == Turn(15.0, 25.0, 5.0 * DEG) for _, m in maneuvers)
    assert turn.config.focus_window == (15.0, 20.0)  # clipped at the 20 s run
    brake = next(row for row in rows if row.group == "acceleration" and row.x == 4.0)
    assert all(m == Acceleration(15.0, 18.0, 4.0) for _, m in brake.config.truth.maneuvers)
    random_row = next(row for row in rows if row.group == "random" and row.x == 20.0)
    assert random_row.config.truth.random_maneuvers == RandomManeuvers(10.0, 5.0, 20.0 * DEG)
    assert random_row.config.truth.maneuvers == ()


def test_camera_rows_convert_degrees_and_milliseconds_and_disturb_one_thing_at_a_time():
    """Fails if bias or offset is in the wrong unit, or the two disturbances are combined."""
    rows = camera_rows(SHORT)
    assert len(rows) == 9
    bias = next(row for row in rows if row.group == "bias" and row.x == 0.5)
    assert bias.config.truth == TruthDisturbance(camera_bias=0.5 * DEG)
    offset = next(row for row in rows if row.group == "time offset" and row.x == 50.0)
    assert offset.config.truth == TruthDisturbance(camera_time_offset=0.05)
    assert all(row.config.belief == TrackerBelief() for row in rows)


def test_lever_arm_rows_cover_unknown_at_two_angles_and_known_at_one():
    """Fails if the known rows do not tell the tracker the position, or unknown rows do."""
    rows = lever_arm_rows(SHORT)
    assert len(rows) == 19 and len(set(labels(rows))) == 19
    unknown = [row for row in rows if row.group and row.group.startswith("unknown")]
    known = [row for row in rows if row.group == "known"]
    assert len(unknown) == 12 and len(known) == 6
    for row in unknown:
        assert row.config.belief.assumed_camera_position == (0.0, 0.0)
        assert row.config.truth.camera_position != (0.0, 0.0)
    for row in known:
        assert row.config.belief.assumed_camera_position == row.config.truth.camera_position
    ninety = next(row for row in unknown if row.group == "unknown @90 deg" and row.x == 20.0)
    assert ninety.config.truth.camera_position == pytest.approx((0.0, 20.0), abs=1e-12)


def test_geometry_rows_run_every_scene_neutral_and_with_an_unknown_arm():
    """Fails if a scene lacks its pair, the arm is known to the tracker, or scenes are shared."""
    rows = geometry_rows(SHORT)
    assert len(rows) == 38 and len(set(labels(rows))) == 38
    for neutral, arm in zip(rows[::2], rows[1::2], strict=True):
        assert arm.label.startswith(neutral.label) and "unknown" in arm.label
        assert arm.config.truth.camera_position != (0.0, 0.0)
        assert arm.config.belief.assumed_camera_position == (0.0, 0.0)
        assert neutral.config.truth == TruthDisturbance()
        np.testing.assert_array_equal(neutral.states, arm.states)
    clutter = [row for row in rows if row.group == "clutter" and "unknown" not in row.label]
    assert [row.config.radar_clutter_rate for row in clutter] == [2.0, 5.0, 10.0]
    counts = [len(row.states) for row in rows if row.group == "count"][::2]
    assert counts == [1, 2, 4, 8]


# --- geometry layouts ------------------------------------------------------------------


def trajectories(states, times=None):
    times = np.linspace(0.0, 60.0, 601) if times is None else times
    return [s[None, :2] + times[:, None] * s[None, 2:] for s in states]


def all_layouts():
    base = RobustnessConfig()
    return [(row.label, row.states) for row in geometry_rows(base)[::2]]


def test_every_planned_layout_stays_inside_the_field_of_view_with_margin_for_the_random_walk():
    """Fails if a layout leaves the field of view or comes within reach of its limits.

    The random acceleration moves a target by about 42 m (1 sigma) over 60 s; three sigma of it
    is kept from both range limits, at the straight-line extremes computed here.
    """
    fov = SimulationConfig().fov
    margin = 3.0 * 42.0
    for label, states in all_layouts():
        ranges = np.concatenate([np.hypot(*p.T) for p in trajectories(states)])
        assert ranges.min() >= fov.range_min + margin, label
        assert ranges.max() <= fov.range_max - margin, label


def test_a_tangential_target_passes_its_reference_point_at_30_seconds():
    """Fails if the reference time or the tangent direction of a layout is wrong."""
    layout = GeometryLayout("tangential", 1500.0, 2, first_bearing_deg=30.0, spacing_deg=60.0)
    for i, state in enumerate(layout.states()):
        at_reference = state[:2] + REFERENCE_TIME * state[2:]
        bearing = (30.0 + 60.0 * i) * DEG
        np.testing.assert_allclose(
            at_reference, 1500.0 * np.array([np.cos(bearing), np.sin(bearing)]), atol=1e-9
        )
        assert np.hypot(*state[2:]) == pytest.approx(GEOMETRY_SPEED)
        radial = np.array([np.cos(bearing), np.sin(bearing)])
        assert abs(np.dot(state[2:], radial)) < 1e-12  # purely tangential


@pytest.mark.parametrize("direction, sign", [("inbound", -1.0), ("outbound", 1.0)])
def test_radial_targets_move_along_the_line_of_sight(direction, sign):
    """Fails if inbound and outbound are swapped or the motion is not radial."""
    state = GeometryLayout(direction, 1500.0, 1, 45.0).states()[0]
    radial = np.array([np.cos(45 * DEG), np.sin(45 * DEG)])
    assert np.dot(state[2:], radial) == pytest.approx(sign * GEOMETRY_SPEED)
    assert np.hypot(*state[:2]) == pytest.approx(1500.0 - sign * GEOMETRY_SPEED * REFERENCE_TIME)


def test_crossing_pairs_have_a_closest_approach_of_ten_meters_at_30_seconds():
    """Fails if a crossing pair misses by another distance, or at another time."""
    layout = GeometryLayout("crossing", 1500.0, 4, 0.0, 180.0)
    states = layout.states()
    times = np.linspace(0.0, 60.0, 6001)
    paths = trajectories(states, times)
    for a, b in ((0, 1), (2, 3)):
        gap = np.hypot(*(paths[a] - paths[b]).T)
        assert gap.min() == pytest.approx(10.0, abs=1e-6)
        assert times[gap.argmin()] == pytest.approx(30.0, abs=0.02)
    # the other pair is on the opposite side of the radar, far away
    assert np.hypot(*(paths[0] - paths[2]).T).min() > 2000.0


def test_a_ring_layout_crosses_the_seam_at_pi():
    """Fails if no planned layout exercises the bearing wrap through +-pi."""
    layout = GeometryLayout("tangential", 1500.0, 4, 0.0, 90.0)
    paths = trajectories(layout.states())
    bearings = np.arctan2(paths[2][:, 1], paths[2][:, 0])
    assert (bearings > 3.0).any() and (bearings < -3.0).any()


def test_non_crossing_layouts_keep_their_targets_apart():
    """Fails if a layout that is meant to be easy has an accidental close encounter."""
    for label, states in all_layouts():
        if "crossing" in label:
            continue
        paths = trajectories(states)
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                assert np.hypot(*(paths[i] - paths[j]).T).min() > 100.0, (label, i, j)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"direction": "diagonal", "zone_range": 1000.0, "n_targets": 1},
        {"direction": "inbound", "zone_range": 0.0, "n_targets": 1},
        {"direction": "inbound", "zone_range": 1000.0, "n_targets": 0},
        {"direction": "crossing", "zone_range": 1000.0, "n_targets": 3},
    ],
)
def test_invalid_layouts_are_rejected(kwargs):
    """Fails if an undefined layout (unknown direction, odd crossing) is accepted."""
    with pytest.raises(ValueError):
        GeometryLayout(**kwargs)


def test_zones_and_directions_are_the_documented_ones():
    """Fails if the documented near / mid / far ranges or the four directions change silently."""
    assert ZONES == {"near": 600.0, "mid": 1500.0, "far": 2500.0}
    assert DIRECTIONS == ("inbound", "outbound", "tangential", "crossing")


def test_simulate_and_track_uses_the_disturbance_and_the_belief_of_its_config():
    """Fails if the helper drops the truth disturbance or the belief on the way."""
    config = replace(
        CLEAN,
        truth=TruthDisturbance(camera_position=(30.0, 0.0)),
        belief=TrackerBelief(assumed_camera_position=(30.0, 0.0)),
    )
    sim, run = simulate_and_track(SEPARATED, config, 0)
    expected = simulate_mtt(SEPARATED, config, make_mtt_rngs(0), config.truth)
    for a, b in zip(sim.camera_scans, expected.camera_scans, strict=True):
        np.testing.assert_array_equal(a.z, b.z)
    assert run.tracker._camera_model.position == (30.0, 0.0)


def test_the_neutral_maneuver_row_has_the_turn_focus_window_and_is_otherwise_unchanged():
    """Fails if the neutral row lacks the window of the turn rows (window scores would not be
    comparable), gains a disturbance or belief, or the window changes any non-window score.

    The window is evaluation only: the other scores of the row equal those of the same
    configuration without a window, bit for bit.
    """
    rows = maneuver_rows(SHORT, n_targets=3)
    neutral, turn = rows[0], rows[1]
    assert neutral.config.focus_window == turn.config.focus_window == (15.0, 20.0)
    assert replace(neutral.config, focus_window=None) == SHORT
    with_window = run_robustness_trial(SEPARATED, neutral.config, 1)
    without = run_robustness_trial(SEPARATED, SHORT, 1)
    for name in FIELDS:
        if name.startswith("window_"):
            assert np.isfinite(getattr(with_window, name)) or name == "window_ghost_rate", name
            assert np.isnan(getattr(without, name)), name
        else:
            np.testing.assert_equal(getattr(with_window, name), getattr(without, name))
