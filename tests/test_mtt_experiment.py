"""Tests for the multi-target experiment: configuration, trials, clutter-in-gate, sweeps."""

from dataclasses import replace

import numpy as np
import pytest

from fusion.association.gating import gate_threshold, gated_costs
from fusion.mtt_experiment import (
    SCENARIOS,
    SWEEP_PARAMETERS,
    MttConfig,
    clutter_gate_report,
    clutter_per_gate,
    run_mtt_trial,
    sweep,
    tentative_filter,
)
from fusion.mtt_metrics import MttMetrics
from fusion.mtt_simulation import FieldOfView, make_mtt_rngs, simulate_mtt
from fusion.sensors.radar import RadarModel, radar_to_cartesian
from fusion.tracker.track import LifecycleConfig

EASY = MttConfig(radar_pd=1.0, camera_pd=1.0, radar_clutter_rate=0.0, camera_clutter_rate=0.0)
SHORT = MttConfig(duration=20.0, burn_in=2.0)


# --- configuration -------------------------------------------------------------------


def test_defaults_and_derived_values():
    config = MttConfig()
    assert config.n_steps == 600 and config.burn_in_steps == 50
    assert config.lifecycle == LifecycleConfig(3, 5, 5)
    assert config.gate_probability == 0.99 and config.use_camera


@pytest.mark.parametrize(
    "kwargs",
    [
        {"velocity_std": 0.0},
        {"gate_probability": 1.0},
        {"match_distance": 0.0},
        {"burn_in": -1.0},
        {"burn_in": 60.0},
        {"radar_pd": 2.0},  # inherited validation
    ],
)
def test_config_validation(kwargs):
    with pytest.raises(ValueError):
        MttConfig(**kwargs)


def test_tracker_config_carries_the_experiment_parameters():
    config = MttConfig(
        velocity_std=15.0,
        gate_probability=0.95,
        lifecycle=LifecycleConfig(2, 4, 3),
        use_camera=False,
    )
    tracker = config.tracker_config()
    assert (tracker.dt, tracker.accel_std, tracker.velocity_std) == (0.1, 0.5, 15.0)
    assert tracker.gate_probability == 0.95 and tracker.min_range == config.fov.range_min
    assert tracker.lifecycle == LifecycleConfig(2, 4, 3) and not tracker.use_camera


# --- trials --------------------------------------------------------------------------


def test_trial_is_deterministic_per_seed_and_differs_between_seeds():
    config = replace(SHORT, radar_clutter_rate=3.0)
    a = run_mtt_trial(SCENARIOS["separated"], config, seed=1)
    b = run_mtt_trial(SCENARIOS["separated"], config, seed=1)
    c = run_mtt_trial(SCENARIOS["separated"], config, seed=2)
    assert isinstance(a, MttMetrics)
    np.testing.assert_equal(tuple(a), tuple(b))
    assert tuple(a) != tuple(c)


def test_easy_case_is_tracked_without_ghosts_misses_or_switches():
    config = EASY
    metrics = run_mtt_trial(SCENARIOS["separated"], config, seed=0)
    assert metrics.ghost_rate == 0.0
    assert metrics.id_switches == 0
    assert metrics.confirmed_fraction == 1.0
    assert metrics.missed_rate < 0.05
    n_scans = config.n_steps // config.radar_every + 1
    assert metrics.births_per_scan == pytest.approx(3 / n_scans)

    # The filter beats the unfiltered radar position of the same run.
    sim = simulate_mtt(SCENARIOS["separated"], config, make_mtt_rngs(0))
    errors = []
    for k, scan in enumerate(sim.radar_scans):
        if scan is not None and k >= config.burn_in_steps:
            positions = radar_to_cartesian(scan.z)
            errors += [
                np.sum((positions[i] - sim.truth[origin, k, :2]) ** 2)
                for i, origin in enumerate(scan.origin)
            ]
    assert metrics.position_rmse < np.sqrt(np.mean(errors))


def test_trial_without_camera_is_less_accurate_but_still_tracks_every_target():
    # Radar-only cross-range error is about 90 m at 2.5 km, so the default 50 m match
    # distance would count a correct but imprecise track as a ghost; widen it here.
    radar_only = replace(EASY, use_camera=False, match_distance=200.0)
    metrics = run_mtt_trial(SCENARIOS["separated"], radar_only, seed=0)
    with_camera = run_mtt_trial(SCENARIOS["separated"], replace(EASY, match_distance=200.0), seed=0)
    assert metrics.ghost_rate == 0.0 and metrics.confirmed_fraction == 1.0
    assert metrics.position_rmse > 2.0 * with_camera.position_rmse


def test_births_per_scan_reflects_clutter_births():
    quiet = run_mtt_trial(SCENARIOS["separated"], replace(SHORT, radar_clutter_rate=0.0), 0)
    noisy = run_mtt_trial(SCENARIOS["separated"], replace(SHORT, radar_clutter_rate=20.0), 0)
    assert noisy.births_per_scan > quiet.births_per_scan + 5.0


# --- clutter in the gate -------------------------------------------------------------


def monte_carlo_fraction(config, tracker_filter, n_samples, seed):
    """Fraction of uniform (range, bearing) points that fall inside the track's gate."""
    rng = np.random.default_rng(seed)
    fov = config.fov
    z = np.column_stack(
        [
            rng.uniform(fov.range_min, fov.range_max, n_samples),
            rng.uniform(-np.pi, np.pi, n_samples),
        ]
    )
    model = RadarModel(config.radar_range_std, config.radar_bearing_std)
    gamma = gate_threshold(2, config.gate_probability)
    costs = gated_costs([tracker_filter], z, model, gamma, config.fov.range_min)
    return float(costs.in_gate.mean())


@pytest.mark.parametrize("bearing", [0.3, np.pi - 0.01, -np.pi + 0.01])
def test_clutter_per_gate_matches_a_monte_carlo_count_also_across_the_seam(bearing):
    config = MttConfig()
    tracker_filter = tentative_filter(config, 500.0, 2, bearing=bearing)
    expected = clutter_per_gate(config, 500.0, 2)
    measured = monte_carlo_fraction(config, tracker_filter, 400_000, seed=3)
    assert measured == pytest.approx(expected, rel=0.1)


def test_clutter_per_gate_grows_with_coasting_and_with_the_velocity_prior():
    config = MttConfig()
    values = [clutter_per_gate(config, 1500.0, n) for n in (1, 2, 3)]
    assert values[0] < values[1] < values[2]
    wide = replace(config, velocity_std=40.0)
    assert clutter_per_gate(wide, 1500.0, 1) > clutter_per_gate(config, 1500.0, 1)


def test_clutter_per_gate_scales_with_the_gate_threshold_and_inversely_with_the_area():
    config = MttConfig()
    base = clutter_per_gate(config, 1000.0, 1)
    other = replace(config, gate_probability=0.95)
    ratio = gate_threshold(2, 0.95) / gate_threshold(2, 0.99)
    assert clutter_per_gate(other, 1000.0, 1) == pytest.approx(base * ratio)
    half = replace(
        config, fov=FieldOfView(range_min=50.0, range_max=3000.0, bearing_half_width=np.pi / 2)
    )
    assert clutter_per_gate(half, 1000.0, 1) == pytest.approx(2 * base)


def test_tentative_filter_coasts_the_covariance_forward():
    config = MttConfig()
    fresh = tentative_filter(config, 800.0, 0)
    coasted = tentative_filter(config, 800.0, 2)
    assert np.trace(coasted.P) > np.trace(fresh.P)
    assert np.hypot(*fresh.x[:2]) == pytest.approx(800.0)


def test_clutter_gate_report_lists_every_rate_and_coasting_time():
    config = MttConfig()
    rates = [1.0, 5.0]
    lines = clutter_gate_report(config, rates, ranges=(500.0, 1500.0), coast_scans=(1, 2))
    assert len(lines) == 1 + 2 * (1 + len(rates))
    assert "gamma=9.21" in lines[0] and "99%" in lines[0]
    expected = 5.0 * clutter_per_gate(config, 1500.0, 2)
    assert any(f"{expected:8.4f}" in line and "lambda=    5" in line for line in lines)
    first = [line for line in lines if "lambda=    1" in line][0]
    assert f"{clutter_per_gate(config, 500.0, 1):8.4f}" in first


# --- sweeps --------------------------------------------------------------------------


def test_sweep_has_one_row_per_value_and_one_column_per_seed():
    result = sweep(SCENARIOS["separated"], SHORT, "clutter_rate", [0.0, 5.0], [0, 1, 2])
    assert result.parameter == "clutter_rate"
    np.testing.assert_array_equal(result.values, [0.0, 5.0])
    assert set(result.metrics) == set(MttMetrics._fields)
    assert all(m.shape == (2, 3) for m in result.metrics.values())


def test_sweep_is_reproducible():
    args = (SCENARIOS["separated"], SHORT, "pd", [0.5, 1.0], [0, 1])
    a, b = sweep(*args), sweep(*args)
    for name in MttMetrics._fields:
        np.testing.assert_equal(a.metrics[name], b.metrics[name])


def test_sweep_gives_the_same_result_with_worker_processes():
    args = (SCENARIOS["separated"], SHORT, "radar_clutter_rate", [0.0, 4.0], [0, 1])
    serial, parallel = sweep(*args), sweep(*args, workers=2)
    for name in MttMetrics._fields:
        np.testing.assert_equal(serial.metrics[name], parallel.metrics[name])


def test_sweeping_pd_applies_to_both_sensors_and_changes_the_misses():
    result = sweep(SCENARIOS["separated"], SHORT, "pd", [0.0, 1.0], [0, 1])
    assert (result.metrics["missed_rate"][0] == 1.0).all()
    assert (result.metrics["missed_rate"][1] < 0.2).all()


def test_sweeping_one_sensor_leaves_the_other_untouched():
    config = replace(SHORT, radar_pd=1.0, camera_pd=1.0, radar_clutter_rate=0.0)
    camera_only = sweep(SCENARIOS["separated"], config, "camera_pd", [0.0, 1.0], [0])
    radar_only = sweep(SCENARIOS["separated"], config, "radar_pd", [0.0, 1.0], [0])
    # Without camera detections the radar still tracks the targets.
    assert camera_only.metrics["confirmed_fraction"][0, 0] == 1.0
    # Without radar detections nothing is ever born.
    assert radar_only.metrics["confirmed_fraction"][0, 0] == 0.0


def test_sweep_rejects_bad_arguments():
    with pytest.raises(ValueError, match="unknown parameter"):
        sweep(SCENARIOS["separated"], SHORT, "nope", [1.0], [0])
    with pytest.raises(ValueError, match="empty"):
        sweep(SCENARIOS["separated"], SHORT, "pd", [], [0])
    with pytest.raises(ValueError, match="empty"):
        sweep(SCENARIOS["separated"], SHORT, "pd", [1.0], [])
    with pytest.raises(ValueError, match="workers"):
        sweep(SCENARIOS["separated"], SHORT, "pd", [1.0], [0], workers=0)
    with pytest.raises(ValueError):
        sweep(SCENARIOS["separated"], SHORT, "pd", [1.5], [0])


def test_sweep_parameters_name_real_config_fields():
    fields = MttConfig.__dataclass_fields__
    assert all(name in fields for names in SWEEP_PARAMETERS.values() for name in names)
