"""Tests of the Phase 8b experiment: arms, rows, the multi-arm trial and the sweep.

Each test docstring names the defect that makes it fail.
"""

import hashlib
from dataclasses import replace

import numpy as np
import pytest

import fusion.improvement_experiment as experiment
from fusion.improvement_experiment import (
    BASELINE,
    EVALUATION_SEEDS,
    PILOT_SEEDS,
    TEST_SEED_START,
    TIMING_SEEDS,
    TUNING_SEEDS,
    UNFROZEN_DEFAULTS,
    Arm,
    bias_block_rows,
    evaluation_blocks,
    hard_block_rows,
    held_out_rows,
    improvement_fields,
    improvement_sweep,
    maneuver_block_rows,
    named_arms,
    oracle_arm,
    run_improvement_trial,
    tuning_blocks,
    tuning_candidates,
)
from fusion.mtt_experiment import SCENARIOS
from fusion.mtt_simulation import RNG_STREAMS
from fusion.robustness_experiment import RobustnessConfig, run_robustness_trial
from fusion.robustness_metrics import FIELDS
from fusion.tracker.multi_target import TrackerConfig

DEG = float(np.pi / 180.0)
SHORT = RobustnessConfig(
    duration=20.0, burn_in=2.0, radar_clutter_rate=0.0, camera_clutter_rate=0.0
)
STATES = SCENARIOS["separated"]
ARMS = named_arms(UNFROZEN_DEFAULTS)


def short_base(clutter: float) -> RobustnessConfig:
    return replace(SHORT, radar_clutter_rate=clutter, camera_clutter_rate=clutter)


def long_base(clutter: float) -> RobustnessConfig:
    return RobustnessConfig(radar_clutter_rate=clutter, camera_clutter_rate=clutter)


def test_the_evaluation_seeds_are_disjoint_from_every_other_seed_set():
    """Fails if a seed used for tuning, the pilot, timing or tests is also an evaluation seed."""
    evaluation = set(EVALUATION_SEEDS)
    assert evaluation == set(range(50))
    for other in (TUNING_SEEDS, PILOT_SEEDS, TIMING_SEEDS):
        assert not evaluation & set(other)
    assert min(TUNING_SEEDS) == 1000 and max(TUNING_SEEDS) == 1019
    assert set(PILOT_SEEDS) <= set(TUNING_SEEDS) and len(PILOT_SEEDS) == 5
    assert TEST_SEED_START > max(TUNING_SEEDS)


def test_the_random_streams_of_the_simulation_are_unchanged_and_nothing_is_appended():
    """Fails if a stream was inserted or added: the features draw no random numbers."""
    assert RNG_STREAMS == (
        "trajectory",
        "radar_noise",
        "radar_detection",
        "radar_clutter",
        "radar_shuffle",
        "camera_noise",
        "camera_detection",
        "camera_clutter",
        "camera_shuffle",
        "dropout",
        "maneuver",
    )


def test_the_eval_config_has_the_frozen_high_q_arm_and_two_descriptive_controls():
    """Fails if a control is missing from the maneuver evaluation, appears elsewhere, or changes
    a frozen arm: the frozen high-Q arm keeps the frozen noise, the controls are 3 and 5."""
    from fusion.improvement_experiment import CONTROL_ARMS, FROZEN, evaluation_blocks, named_arms

    assert CONTROL_ARMS == ("EKF high-Q", "EKF high-Q 3", "EKF high-Q 5")
    parameters = FROZEN or UNFROZEN_DEFAULTS
    arms = named_arms(parameters)
    assert arms["EKF high-Q"].accel_std == parameters.ekf_high_accel_std
    assert arms["EKF high-Q 3"].accel_std == 3.0 and arms["EKF high-Q 5"].accel_std == 5.0
    assert arms["IMM-A"].modes[1].accel_std == parameters.imm_a_high_accel_std
    scenes = {"separated": STATES}
    for group, expected in (("maneuver", True), ("bias", False), ("hard", False)):
        block = evaluation_blocks(group, long_base, scenes, (0.0,), parameters)[0]
        names = {a.name for row in block.rows for a in row.arms}
        assert ("EKF high-Q 3" in names and "EKF high-Q 5" in names) is expected, group
    # The controls reuse two settings of the tuning grid and leave the tuning run untouched:
    # it still has its 21 arms per tuned row, with the same candidates as before.
    grid = {a.name: a.accel_std for a in tuning_candidates()["EKF high-Q"]}
    expected_grid = {1.0: "1", 2.0: "2", 3.0: "3", 5.0: "5"}
    assert grid == {f"EKF high-Q {n}": q for q, n in expected_grid.items()}
    block = tuning_blocks(long_base, {"separated": STATES}, (0.0,))[0]
    tuned = [r for r in block.rows if r.group == "maneuver"]
    assert all(len(r.arms) == 1 + 4 + 3 + 9 + 4 for r in tuned)


def test_the_frozen_parameters_are_grid_values_with_well_formed_block_labels():
    """Fails if the frozen constants are not values the tuning could have chosen, or a winnable
    label does not name a real layout, clutter rate and row."""
    from fusion.improvement_experiment import (
        BIAS_PRIOR_GRID,
        EKF_HIGH_GRID,
        FROZEN,
        IMM_A_GRID,
        IMM_B_ACCEL_GRID,
        IMM_B_OMEGA_GRID_DEG,
    )

    if FROZEN is None:
        pytest.skip("nothing is frozen yet")
    assert FROZEN.ekf_high_accel_std in EKF_HIGH_GRID
    assert FROZEN.imm_a_high_accel_std in IMM_A_GRID
    assert FROZEN.imm_b_omega_deg in IMM_B_OMEGA_GRID_DEG
    assert FROZEN.imm_b_accel_std in IMM_B_ACCEL_GRID
    assert FROZEN.bias_prior in BIAS_PRIOR_GRID
    assert FROZEN.commit and len(FROZEN.commit) == 40 and "dirty" not in FROZEN.commit
    rows = {r.label for r in maneuver_block_rows(long_base(0.0), STATES, [Arm(BASELINE)])}
    rows |= {r.label for r in held_out_rows(long_base(0.0), STATES, [Arm(BASELINE)])}
    assert len(set(FROZEN.winnable)) == len(FROZEN.winnable)
    for label in FROZEN.winnable:
        layout, clutter, row = label.split(" | ")
        assert layout in SCENARIOS and float(clutter) in (0.0, 5.0), label
        assert row in rows and not row.endswith("neutral"), label


def test_the_default_arm_leaves_the_tracker_configuration_unchanged():
    """Fails if the baseline arm alters any tracker setting."""
    base = TrackerConfig(dt=0.1, accel_std=0.5, velocity_std=20.0)
    assert Arm(BASELINE).tracker_config(base, 10) == base


def test_arms_set_their_own_fields_and_the_scan_length_of_the_scene():
    """Fails if an arm drops its setting or the IMM uses a scan length other than radar_every."""
    base = TrackerConfig(dt=0.1, accel_std=0.5, velocity_std=20.0)
    high = ARMS["EKF high-Q"].tracker_config(base, 10)
    assert high.accel_std == UNFROZEN_DEFAULTS.ekf_high_accel_std and high.motion is None
    imm = ARMS["IMM-B+bias"].tracker_config(base, 5)
    assert imm.motion.steps_per_scan == 5 and len(imm.motion.modes) == 3
    assert imm.motion.stay_per_scan == 0.95 and imm.camera_bias is not None
    omegas = [m.omega for m in imm.motion.modes]
    assert omegas == [0.0, 10.0 * DEG, -10.0 * DEG]
    assert ARMS["IMM-A"].tracker_config(base, 10).motion.modes[1].accel_std == 3.0


def test_the_oracle_arm_carries_the_true_bias():
    """Fails if the oracle arm does not apply exactly the bias of its row."""
    arm = oracle_arm(0.25 * DEG)
    assert arm.camera_bias.mode == "oracle" and arm.camera_bias.oracle_bias == 0.25 * DEG


def test_the_baseline_arm_of_a_trial_is_the_phase_8a_trial_bit_for_bit():
    """Fails if the multi-arm trial scores the baseline differently from run_robustness_trial."""
    config = replace(SHORT, focus_window=(5.0, 20.0))
    scores = run_improvement_trial(STATES, config, [ARMS[BASELINE], ARMS["IMM-A"]], 2000)
    reference = run_robustness_trial(STATES, config, 2000)
    for name, value in zip(FIELDS, reference, strict=True):
        np.testing.assert_equal(scores[BASELINE][name], value, err_msg=name)


def test_every_arm_has_every_field_and_irrelevant_ones_are_nan():
    """Fails if a score is missing, or an arm reports scores of a feature it does not use."""
    config = replace(SHORT, focus_window=(5.0, 20.0))
    arms = [ARMS[BASELINE], ARMS["IMM-B"], ARMS["EKF+bias"]]
    scores = run_improvement_trial(STATES, config, arms, 2000)
    fields = improvement_fields(config)
    assert len(fields) == len(set(fields))
    for arm in scores.values():
        assert tuple(arm) == fields
    assert np.isnan(scores[BASELINE]["mode_nonref_window"])
    assert np.isnan(scores[BASELINE]["bias_final_error_deg"])
    assert np.isfinite(scores["IMM-B"]["mode_nonref_window"])
    assert np.isfinite(scores["IMM-B"]["mode_nonref_outside"])
    assert np.isnan(scores["IMM-B"]["bias_final_error_deg"])
    assert np.isfinite(scores["EKF+bias"]["bias_final_error_deg"])
    assert scores["EKF+bias"]["bias_converged_025"] in (0.0, 1.0)
    assert all(np.isfinite(a["runtime_s"]) and a["runtime_s"] > 0 for a in scores.values())


def test_the_common_support_rmse_of_the_baseline_is_its_own_rmse():
    """Fails if the reference arm is compared with something other than itself."""
    config = replace(SHORT, focus_window=(5.0, 20.0))
    scores = run_improvement_trial(STATES, config, [ARMS[BASELINE], ARMS["IMM-A"]], 2000)
    base = scores[BASELINE]
    assert base["cs_run_rmse"] == base["cs_run_rmse_ref"]
    assert base["cs_run_rmse"] == pytest.approx(base["run_position_rmse"], rel=1e-12)
    assert base["cs_window_rmse"] == pytest.approx(base["window_position_rmse"], rel=1e-12)
    other = scores["IMM-A"]
    assert other["cs_run_rmse_ref"] == pytest.approx(base["cs_run_rmse"], rel=0.5)


def test_one_simulation_serves_every_arm_and_no_arm_changes_it(monkeypatch):
    """Fails if the scene is simulated once per arm (arms would not share data), or if a
    tracker run modifies the truth or the scans that the next arm reads."""
    created = []
    real = experiment.simulate_mtt

    def spy(*args, **kwargs):
        sim = real(*args, **kwargs)
        created.append((sim, digest(sim)))
        return sim

    def digest(sim) -> str:
        h = hashlib.sha256(np.ascontiguousarray(sim.truth).tobytes())
        for scan in (*[s for s in sim.radar_scans if s is not None], *sim.camera_scans):
            h.update(np.ascontiguousarray(scan.z).tobytes())
        return h.hexdigest()

    monkeypatch.setattr(experiment, "simulate_mtt", spy)
    arms = [ARMS[BASELINE], ARMS["IMM-B+bias"], ARMS["EKF high-Q"]]
    state = np.random.get_state()[1].copy()
    run_improvement_trial(STATES, SHORT, arms, 2001)
    assert len(created) == 1
    sim, before = created[0]
    assert digest(sim) == before
    assert (np.random.get_state()[1] == state).all()  # the global generator is untouched


def test_a_trial_is_deterministic_per_seed():
    """Fails if two runs of the same seed differ (runtime aside)."""
    arms = [ARMS[BASELINE], ARMS["IMM-A+bias"]]
    a = run_improvement_trial(STATES, SHORT, arms, 2002)
    b = run_improvement_trial(STATES, SHORT, arms, 2002)
    for name in a:
        for field in a[name]:
            if field != "runtime_s":
                np.testing.assert_equal(a[name][field], b[name][field], err_msg=field)


def test_the_arms_of_a_trial_must_be_distinct_and_start_with_the_baseline():
    """Fails if a duplicate name or a missing baseline is accepted."""
    with pytest.raises(ValueError):
        run_improvement_trial(STATES, SHORT, [ARMS[BASELINE], ARMS[BASELINE]], 0)
    with pytest.raises(ValueError):
        run_improvement_trial(STATES, SHORT, [ARMS["IMM-A"], ARMS[BASELINE]], 0)
    with pytest.raises(ValueError):
        run_improvement_trial(STATES, SHORT, [], 0)


def test_maneuver_rows_have_a_window_from_the_onset_to_the_end_in_every_row():
    """Fails if a row (the random ones, the neutral one) has no window or ends it early."""
    rows = maneuver_block_rows(long_base(0.0), STATES, [Arm(BASELINE)])
    assert len(rows) == 10 and rows[0].label == "neutral"
    for row in rows:
        start = 10.0 if row.label.startswith("random") else 15.0
        assert row.config.focus_window == (start, 60.0), row.label
    assert {r.group for r in rows} == {"maneuver"}


def test_held_out_rows_are_new_maneuvers_with_their_own_neutral_row():
    """Fails if a held-out row repeats a tuned maneuver, or has no neutral reference."""
    rows = held_out_rows(long_base(0.0), STATES, [Arm(BASELINE)])
    assert [r.label for r in rows] == [
        "held-out neutral",
        "held-out turn 7 deg/s",
        "held-out turn 15 deg/s",
        "held-out acceleration 3 m/s^2",
    ]
    assert all(r.config.focus_window == (30.0, 60.0) for r in rows)
    assert rows[0].config.truth.maneuvers == ()
    turn = rows[1].config.truth.maneuvers[0][1]
    assert (turn.start, turn.end, turn.turn_rate) == (30.0, 40.0, 7.0 * DEG)
    assert held_out_rows(short_base(0.0), STATES, [Arm(BASELINE)]) == []


def test_bias_rows_carry_the_bias_and_an_oracle_arm_that_knows_it():
    """Fails if a row's oracle arm applies another bias than the world has."""
    rows = bias_block_rows(long_base(0.0), STATES, [Arm(BASELINE), Arm("EKF+bias")])
    assert [r.label for r in rows] == [
        "bias 0 deg", "bias 0.1 deg", "bias 0.25 deg", "bias 0.5 deg", "bias 1 deg"
    ]  # fmt: skip
    for row in rows:
        assert row.arms[-1].name == "EKF+oracle"
        assert row.arms[-1].camera_bias.oracle_bias == row.config.truth.camera_bias
        assert row.config.focus_window is None
    assert rows[3].config.truth.camera_bias == pytest.approx(0.5 * DEG)


def test_hard_rows_combine_a_bias_of_half_a_degree_with_each_maneuver():
    """Fails if a hard row lacks the bias or its maneuver."""
    rows = hard_block_rows(long_base(0.0), STATES, [Arm(BASELINE)])
    assert len(rows) == 4
    for row in rows:
        assert row.config.truth.camera_bias == pytest.approx(0.5 * DEG)
    assert rows[0].config.truth.maneuvers == ()
    assert len(rows[1].config.truth.maneuvers) == 3
    assert rows[3].config.truth.random_maneuvers is not None


def test_evaluation_blocks_use_the_planned_arms_per_group():
    """Fails if a group runs other arms than planned, or the block grid is wrong."""
    scenes = {"separated": STATES, "crossing": SCENARIOS["crossing"]}
    expected = {
        "maneuver": (
            BASELINE, "EKF high-Q", "EKF high-Q 3", "EKF high-Q 5", "IMM-A", "IMM-B", "EKF+bias"
        ),  # fmt: skip
        "bias": (BASELINE, "EKF+bias", "EKF+always", "EKF+oracle"),
        "hard": (BASELINE, "EKF+bias", "IMM-A", "IMM-B", "IMM-A+bias", "IMM-B+bias"),
    }
    for group, names in expected.items():
        blocks = evaluation_blocks(group, long_base, scenes, (0.0, 5.0), UNFROZEN_DEFAULTS)
        assert len(blocks) == 4
        for block in blocks:
            for row in block.rows:
                assert tuple(a.name for a in row.arms) == names
    maneuver = evaluation_blocks("maneuver", long_base, scenes, (0.0,), UNFROZEN_DEFAULTS)
    assert len(maneuver[0].rows) == 14  # 10 tuned rows and 4 held-out rows
    with pytest.raises(ValueError):
        evaluation_blocks("nothing", long_base, scenes, (0.0,), UNFROZEN_DEFAULTS)


def test_the_tuning_run_covers_every_candidate_and_only_the_baseline_on_held_out_rows():
    """Fails if a candidate is missing from the tuned rows, or held-out rows run candidates."""
    candidates = tuning_candidates()
    assert [len(v) for v in candidates.values()] == [4, 3, 9, 4]
    block = tuning_blocks(long_base, {"separated": STATES}, (0.0,))[0]
    tuned = [r for r in block.rows if r.group == "maneuver"]
    held = [r for r in block.rows if r.group == "held-out"]
    bias = [r for r in block.rows if r.group == "bias"]
    assert len(tuned) == 10 and all(len(r.arms) == 1 + 4 + 3 + 9 + 4 for r in tuned)
    assert all(tuple(a.name for a in r.arms) == (BASELINE,) for r in held)
    assert len(bias) == 5 and all(len(r.arms) == 5 for r in bias)
    names = [a.name for a in tuned[0].arms]
    assert len(set(names)) == len(names)


def test_the_sweep_has_one_result_per_row_and_does_not_depend_on_the_worker_count():
    """Fails if rows or arms are mixed up, or results depend on the number of processes."""
    config = replace(SHORT, duration=12.0, burn_in=2.0, focus_window=(5.0, 12.0))
    arms = (ARMS[BASELINE], ARMS["IMM-A"])
    rows = [
        experiment.ImprovementRow("one", config, arms, STATES, "maneuver"),
        experiment.ImprovementRow("two", replace(config, accel_std=1.0), arms, STATES, "maneuver"),
    ]
    one = improvement_sweep(rows, (2000, 2001), workers=1)
    two = improvement_sweep(rows, (2000, 2001), workers=2)
    assert [r.label for r in one] == ["one", "two"]
    assert one[0].arm_names == (BASELINE, "IMM-A") and one[0].seeds == (2000, 2001)
    fields = improvement_fields(config)
    assert tuple(one[0].metrics) == fields
    assert one[0].metrics["run_missed_rate"].shape == (2, 2)
    for a, b in zip(one, two, strict=True):
        for name in fields:
            if name != "runtime_s":
                np.testing.assert_array_equal(a.metrics[name], b.metrics[name], err_msg=name)
    assert not np.array_equal(
        one[0].metrics["run_position_rmse"], one[1].metrics["run_position_rmse"]
    )


def test_an_empty_or_invalid_sweep_is_refused():
    """Fails if empty rows, empty seeds or zero workers are accepted."""
    row = experiment.ImprovementRow("x", SHORT, (ARMS[BASELINE],), STATES, "maneuver")
    for args in (([], (0,), 1), ([row], (), 1), ([row], (0,), 0)):
        with pytest.raises(ValueError):
            improvement_sweep(*args)
