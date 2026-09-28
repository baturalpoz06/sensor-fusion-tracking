"""Tests for the radar vs. fusion experiment helpers."""

import numpy as np
import pytest

from fusion import experiment
from fusion.experiment import (
    SCENARIOS,
    FusionConfig,
    evaluation_masks,
    make_rngs,
    position_rmse,
    run_methods,
    run_trial,
    simulate,
)

# Upper bounds on mean fusion RMSE / mean radar-only RMSE over seeds 0-4.
# Observed (all steps / radar steps): near 2.60/6.96 m = 0.37, 2.46/6.47 m = 0.38;
# far 2.89/30.41 m = 0.095, 2.72/28.47 m = 0.096. Per-seed ratios reach 0.57 (near).
RATIO_BOUND = {"near": 0.6, "far": 0.2}


def test_config_defaults():
    config = FusionConfig()
    assert config.n_steps == 600
    assert config.burn_in_steps == 50
    np.testing.assert_allclose(config.radar_bearing_std, np.deg2rad(2.0))
    np.testing.assert_allclose(config.camera_bearing_std, np.deg2rad(0.1))


def test_make_rngs_is_reproducible_and_streams_are_distinct():
    first = [rng.normal(size=5) for rng in make_rngs(3)]
    second = [rng.normal(size=5) for rng in make_rngs(3)]
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)
    assert not np.allclose(first[0], first[1])
    assert not np.allclose(first[1], first[2])
    assert not np.allclose(first[0], first[2])


def test_simulate_shapes():
    config = FusionConfig(duration=2.0)
    sim = simulate(SCENARIOS["near"], config, make_rngs(0))
    assert sim.truth.shape == (21, 4)
    assert sim.radar_z.shape == (21, 2)
    assert sim.camera_z.shape == (21, 1)


def test_evaluation_masks():
    masks = evaluation_masks(n_points=25, radar_every=10, burn_in_steps=5)
    np.testing.assert_array_equal(np.flatnonzero(masks["all"]), np.arange(5, 25))
    np.testing.assert_array_equal(np.flatnonzero(masks["radar"]), [10, 20])
    # Without burn-in the initializing radar step 0 counts as a radar step.
    masks = evaluation_masks(n_points=25, radar_every=10, burn_in_steps=0)
    np.testing.assert_array_equal(np.flatnonzero(masks["radar"]), [0, 10, 20])


def test_position_rmse_hand_computed():
    positions = np.array([[0.0, 0.0], [3.0, 4.0], [100.0, 100.0]])
    truth = np.zeros((3, 4))
    mask = np.array([True, True, False])
    # Errors 0 and 5 m on the selected steps: sqrt((0 + 25) / 2).
    assert position_rmse(positions, truth, mask) == pytest.approx(np.sqrt(12.5))
    with pytest.raises(ValueError, match="no steps"):
        position_rmse(positions, truth, np.zeros(3, dtype=bool))


def test_changing_camera_seed_leaves_radar_only_unchanged():
    config = FusionConfig(duration=20.0)
    rngs_a = make_rngs(5)
    rngs_b = (*make_rngs(5)[:2], np.random.default_rng(999))
    sim_a = simulate(SCENARIOS["near"], config, rngs_a)
    sim_b = simulate(SCENARIOS["near"], config, rngs_b)

    np.testing.assert_array_equal(sim_a.truth, sim_b.truth)
    np.testing.assert_array_equal(sim_a.radar_z, sim_b.radar_z)
    assert not np.allclose(sim_a.camera_z, sim_b.camera_z)

    est_a = run_methods(sim_a, config)
    est_b = run_methods(sim_b, config)
    np.testing.assert_array_equal(est_a["radar_only"], est_b["radar_only"])
    np.testing.assert_array_equal(est_a["raw"], est_b["raw"])
    assert not np.allclose(est_a["fusion"], est_b["fusion"])


@pytest.mark.parametrize("scenario", ["near", "far"])
def test_fusion_beats_radar_only_on_average(scenario):
    config = FusionConfig()
    results = [run_trial(SCENARIOS[scenario], config, make_rngs(seed)) for seed in range(5)]
    for mask in ("all", "radar"):
        fusion = np.mean([r["fusion"][mask] for r in results])
        radar_only = np.mean([r["radar_only"][mask] for r in results])
        assert fusion < RATIO_BOUND[scenario] * radar_only


def test_filter_uses_the_simulation_accel_std(monkeypatch):
    seen = []
    real_trajectory = experiment.constant_velocity_trajectory
    real_tracking = experiment.run_tracking

    def spy_trajectory(*args, **kwargs):
        seen.append(("trajectory", kwargs["accel_std"]))
        return real_trajectory(*args, **kwargs)

    def spy_tracking(*args, **kwargs):
        seen.append(("filter", kwargs["accel_std"]))
        return real_tracking(*args, **kwargs)

    monkeypatch.setattr(experiment, "constant_velocity_trajectory", spy_trajectory)
    monkeypatch.setattr(experiment, "run_tracking", spy_tracking)

    config = FusionConfig(duration=10.0, accel_std=0.73)
    run_trial(SCENARIOS["near"], config, make_rngs(0))
    assert [name for name, _ in seen] == ["trajectory", "filter", "filter"]
    assert all(value == 0.73 for _, value in seen)


def test_run_trial_structure():
    rmse = run_trial(SCENARIOS["near"], FusionConfig(duration=10.0), make_rngs(0))
    assert set(rmse) == {"raw", "radar_only", "fusion"}
    assert np.isnan(rmse["raw"]["all"])
    assert rmse["raw"]["radar"] > 0.0
    for method in ("radar_only", "fusion"):
        for mask in ("all", "radar"):
            assert np.isfinite(rmse[method][mask]) and rmse[method][mask] > 0.0
