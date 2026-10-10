"""Phase 8b experiment: tracker variants (arms) compared on identical simulated data.

An arm is a tracker setting (a larger process noise, an IMM motion model, a camera-bias
estimate); the simulated world and the tracker's measurement beliefs are those of the robustness
experiment (fusion.robustness_experiment). One trial simulates a scene once and runs every arm on
it, so arms see bit-identical data (common random numbers) and can be scored on the target-steps
they share. The baseline arm "EKF" is the Phase 8a tracker; with the default configuration its
scores are those of run_robustness_trial bit for bit.

Seed sets are disjoint by construction: tuning seeds choose parameters, the pilot and timing
seeds only check that things run, tests use seeds from 2000 on, and the evaluation seeds are
used once, after the parameters are frozen.
"""

import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from typing import NamedTuple

import numpy as np

from fusion.filters.imm import ModeSpec, MotionConfig
from fusion.improvement_metrics import (
    bias_errors,
    bias_summary,
    bin_edges,
    binned_rates,
    common_support_rmse,
    convergence_time,
    drag_mask,
    mode_probabilities,
    mode_summary,
    window_mask,
)
from fusion.maneuvers import Acceleration, RandomManeuvers, TruthDisturbance, Turn
from fusion.mtt_metrics import match_tracks
from fusion.mtt_simulation import make_mtt_rngs, simulate_mtt
from fusion.robustness_experiment import RobustnessConfig, maneuver_rows
from fusion.robustness_metrics import FIELDS, evaluate_robustness
from fusion.tracker.camera_bias import CameraBiasConfig
from fusion.tracker.multi_target import TrackerConfig, run_multi_target_tracking

DEG = float(np.pi / 180.0)
BASELINE = "EKF"
STAY_PER_SCAN = 0.95
BIN_WIDTH = 5.0

TUNING_SEEDS = tuple(range(1000, 1020))
PILOT_SEEDS = tuple(range(1000, 1005))
TIMING_SEEDS = PILOT_SEEDS
EVALUATION_SEEDS = tuple(range(50))
TEST_SEED_START = 2000

ONSET = 15.0  # start of the focus window of the turns and accelerations of the tuned rows
RANDOM_ONSET = 10.0
HELD_OUT_START = 30.0
BIAS_GRID_DEG = (0.1, 0.25, 0.5, 1.0)
HARD_BIAS_DEG = 0.5
CONVERGENCE_THRESHOLDS_DEG = (0.25, 0.1)

# Tuning grids (see the plan): candidates are named after their parameters.
EKF_HIGH_GRID = (1.0, 2.0, 3.0, 5.0)
IMM_A_GRID = (2.0, 3.0, 5.0)
IMM_B_OMEGA_GRID_DEG = (7.5, 10.0, 15.0)
IMM_B_ACCEL_GRID = (0.5, 1.0, 2.0)
BIAS_PRIOR_GRID = (1.0, 0.5, 0.1, 0.01)  # 1.0 stands for "always apply"

# Descriptive sensitivity controls next to the frozen EKF high-Q arm: single EKFs with a larger
# process noise. They are not tuned, not new criteria and do not change the frozen parameters;
# they only let a claim "IMM beats the high-Q EKF" be checked against stronger controls.
EKF_CONTROL_Q = (3.0, 5.0)
CONTROL_ARMS = ("EKF high-Q", "EKF high-Q 3", "EKF high-Q 5")


@dataclass(frozen=True)
class Parameters:
    """The tuned values of the evaluation arms, frozen before the evaluation run.

    Attributes:
        ekf_high_accel_std: Process noise std in m/s^2 of the EKF high-Q control arm.
        imm_a_high_accel_std: Process noise std of the high mode of IMM-A.
        imm_b_omega_deg: Turn rate in deg/s of the two turn modes of IMM-B.
        imm_b_accel_std: Process noise std of the turn modes of IMM-B.
        bias_prior: Prior probability of H1 of the spike-and-slab bias estimate.
        winnable: Labels ("layout | clutter | row") of the maneuver row-blocks in which the
            baseline degrades by more than the practical threshold on the tuning seeds (the
            denominator of the primary criterion).
        commit: Hash of the repository HEAD at which the tuning ran (recorded by the tune stage);
            the evaluation requires it to be an ancestor of HEAD. The commit that stored these
            values in FROZEN is a later one.
    """

    ekf_high_accel_std: float
    imm_a_high_accel_std: float
    imm_b_omega_deg: float
    imm_b_accel_std: float
    bias_prior: float
    winnable: tuple[str, ...] = ()
    commit: str = ""


# Placeholders for runs that must work before anything is tuned (tests, the pilot). The
# evaluation refuses to run with them: it needs FROZEN.
UNFROZEN_DEFAULTS = Parameters(2.0, 3.0, 10.0, 1.0, 0.5)
# Frozen from the tuning run (seeds 1000-1019, results/improvement_tune.txt) at the commit named
# below, before any evaluation seed was used. All four choices are the fallback of the
# pre-registered rules (no candidate met the neutral margins on 20 seeds): the least violating
# candidate of each family was taken. 7.5 is the grid value (the tuning text printed it as
# 7.499999999999999 after a degrees-radians round trip).
FROZEN: Parameters | None = Parameters(
    ekf_high_accel_std=1.0,
    imm_a_high_accel_std=2.0,
    imm_b_omega_deg=7.5,
    imm_b_accel_std=2.0,
    bias_prior=0.01,
    winnable=(
        'crossing | 0 | turn 10 deg/s',
        'crossing | 0 | turn 20 deg/s',
        'crossing | 0 | acceleration 2 m/s^2',
        'crossing | 0 | acceleration 4 m/s^2',
        'crossing | 0 | random turns <= 10 deg/s',
        'crossing | 0 | random turns <= 20 deg/s',
        'crossing | 0 | held-out turn 7 deg/s',
        'crossing | 0 | held-out turn 15 deg/s',
        'crossing | 0 | held-out acceleration 3 m/s^2',
        'crossing | 5 | turn 5 deg/s',
        'crossing | 5 | turn 10 deg/s',
        'crossing | 5 | turn 20 deg/s',
        'crossing | 5 | acceleration 1 m/s^2',
        'crossing | 5 | acceleration 2 m/s^2',
        'crossing | 5 | acceleration 4 m/s^2',
        'crossing | 5 | random turns <= 10 deg/s',
        'crossing | 5 | random turns <= 20 deg/s',
        'crossing | 5 | held-out turn 7 deg/s',
        'crossing | 5 | held-out turn 15 deg/s',
        'crossing | 5 | held-out acceleration 3 m/s^2',
        'separated | 0 | turn 5 deg/s',
        'separated | 0 | turn 10 deg/s',
        'separated | 0 | turn 20 deg/s',
        'separated | 0 | acceleration 2 m/s^2',
        'separated | 0 | acceleration 4 m/s^2',
        'separated | 0 | random turns <= 5 deg/s',
        'separated | 0 | random turns <= 10 deg/s',
        'separated | 0 | random turns <= 20 deg/s',
        'separated | 0 | held-out turn 7 deg/s',
        'separated | 0 | held-out turn 15 deg/s',
        'separated | 0 | held-out acceleration 3 m/s^2',
        'separated | 5 | turn 5 deg/s',
        'separated | 5 | turn 10 deg/s',
        'separated | 5 | turn 20 deg/s',
        'separated | 5 | acceleration 2 m/s^2',
        'separated | 5 | acceleration 4 m/s^2',
        'separated | 5 | random turns <= 5 deg/s',
        'separated | 5 | random turns <= 10 deg/s',
        'separated | 5 | random turns <= 20 deg/s',
        'separated | 5 | held-out turn 7 deg/s',
        'separated | 5 | held-out turn 15 deg/s',
        'separated | 5 | held-out acceleration 3 m/s^2',
    ),
    commit='3354afccc3fc0a15f5c2a881610184be8e2d32f0',
)


@dataclass(frozen=True)
class Arm:
    """One tracker setting; None fields leave the Phase 8a tracker unchanged.

    Attributes:
        name: Label of the arm.
        accel_std: Replaces the tracker's process noise std (the EKF high-Q arm).
        modes: IMM mode recipes (the motion model); the stay probability is STAY_PER_SCAN per
            radar scan.
        camera_bias: Global camera bias estimate settings.
    """

    name: str
    accel_std: float | None = None
    modes: tuple[ModeSpec, ...] | None = None
    camera_bias: CameraBiasConfig | None = None

    def tracker_config(self, base: TrackerConfig, steps_per_scan: int) -> TrackerConfig:
        """The tracker configuration of this arm, built from the baseline one."""
        changes: dict = {}
        if self.accel_std is not None:
            changes["accel_std"] = self.accel_std
        if self.modes is not None:
            changes["motion"] = MotionConfig(self.modes, STAY_PER_SCAN, steps_per_scan)
        if self.camera_bias is not None:
            changes["camera_bias"] = self.camera_bias
        return replace(base, **changes)


def imm_a_modes(high_accel_std: float) -> tuple[ModeSpec, ...]:
    """IMM-A: the tracker's own constant-velocity mode and one with a larger process noise."""
    return (ModeSpec(), ModeSpec(accel_std=high_accel_std))


def imm_b_modes(omega_deg: float, accel_std: float) -> tuple[ModeSpec, ...]:
    """IMM-B: constant velocity and coordinated turns at plus and minus omega."""
    omega = omega_deg * DEG
    return (ModeSpec(), ModeSpec("ct", accel_std, omega), ModeSpec("ct", accel_std, -omega))


def named_arms(parameters: Parameters) -> dict[str, Arm]:
    """The arms of the evaluation, by name."""
    bias = CameraBiasConfig(mode="spike_slab", prior_h1=parameters.bias_prior)
    imm_a = imm_a_modes(parameters.imm_a_high_accel_std)
    imm_b = imm_b_modes(parameters.imm_b_omega_deg, parameters.imm_b_accel_std)
    return {
        BASELINE: Arm(BASELINE),
        "EKF high-Q": Arm("EKF high-Q", accel_std=parameters.ekf_high_accel_std),
        "EKF high-Q 3": Arm("EKF high-Q 3", accel_std=EKF_CONTROL_Q[0]),
        "EKF high-Q 5": Arm("EKF high-Q 5", accel_std=EKF_CONTROL_Q[1]),
        "IMM-A": Arm("IMM-A", modes=imm_a),
        "IMM-B": Arm("IMM-B", modes=imm_b),
        "EKF+bias": Arm("EKF+bias", camera_bias=bias),
        "EKF+always": Arm("EKF+always", camera_bias=CameraBiasConfig(mode="always")),
        "IMM-A+bias": Arm("IMM-A+bias", modes=imm_a, camera_bias=bias),
        "IMM-B+bias": Arm("IMM-B+bias", modes=imm_b, camera_bias=bias),
    }


def oracle_arm(true_bias: float) -> Arm:
    """EKF that applies the true bias (radians) with zero variance: the achievable bound."""
    return Arm(
        "EKF+oracle", camera_bias=CameraBiasConfig(mode="oracle", oracle_bias=true_bias)
    )


def prior_name(prior: float) -> str:
    """Arm name of a bias prior candidate."""
    return "bias always" if prior >= 1.0 else f"bias p={prior:g}"


def tuning_candidates(small: bool = False) -> dict[str, list[Arm]]:
    """The candidate arms of each tuned family, named after their parameters.

    Args:
        small: Use the first two values of each grid (one turn rate for IMM-B and two priors),
            for quick checks that the machinery runs; never for the real tuning.
    """
    ekf_high, imm_a = EKF_HIGH_GRID, IMM_A_GRID
    omegas, accels, priors = IMM_B_OMEGA_GRID_DEG, IMM_B_ACCEL_GRID, BIAS_PRIOR_GRID
    if small:
        ekf_high, imm_a, omegas, accels, priors = (
            ekf_high[:2], imm_a[:2], omegas[:1], accels[:2], priors[:2]
        )  # fmt: skip
    return {
        "EKF high-Q": [Arm(f"EKF high-Q {a:g}", accel_std=a) for a in ekf_high],
        "IMM-A": [Arm(f"IMM-A {a:g}", modes=imm_a_modes(a)) for a in imm_a],
        "IMM-B": [
            Arm(f"IMM-B {w:g}/{a:g}", modes=imm_b_modes(w, a)) for w in omegas for a in accels
        ],
        "bias": [
            Arm(
                prior_name(p),
                camera_bias=(
                    CameraBiasConfig(mode="always")
                    if p >= 1.0
                    else CameraBiasConfig(mode="spike_slab", prior_h1=p)
                ),
            )
            for p in priors
        ],
    }


# --- rows --------------------------------------------------------------------------------


class ImprovementRow(NamedTuple):
    """One scene of a table: its configuration, the arms run on it and its initial states.

    Attributes:
        label: Text of the row.
        config: Truth disturbance, beliefs and focus window of the row.
        arms: The arms run on every seed of the row; the baseline comes first.
        states: Shape (n_targets, 4) initial states of the scene.
        group: "maneuver", "held-out", "bias" or "hard".
    """

    label: str
    config: RobustnessConfig
    arms: tuple[Arm, ...]
    states: np.ndarray
    group: str


class Block(NamedTuple):
    """The rows of one table set: one layout at one clutter rate."""

    title: str
    layout: str
    clutter: float
    rows: list[ImprovementRow]


def _with_window(config: RobustnessConfig, start: float) -> RobustnessConfig:
    return replace(config, focus_window=(start, config.duration))


def maneuver_block_rows(
    base: RobustnessConfig, states: np.ndarray, arms: Sequence[Arm]
) -> list[ImprovementRow]:
    """The ten tuned maneuver rows of Phase 8a; every row's window runs from its onset to the end.

    The window covers the response and the drag tail after the maneuver (random turns start at
    10 s, the others at 15 s).
    """
    rows = []
    for row in maneuver_rows(base, len(states)):
        start = RANDOM_ONSET if row.group == "random" else ONSET
        rows.append(
            ImprovementRow(
                row.label, _with_window(row.config, start), tuple(arms), states, "maneuver"
            )
        )
    return rows


def held_out_rows(
    base: RobustnessConfig, states: np.ndarray, arms: Sequence[Arm]
) -> list[ImprovementRow]:
    """Maneuvers that no parameter was tuned on: turns of 7 and 15 deg/s and 3 m/s^2.

    They start at 30 s (windows from 30 s to the end) and come with their own neutral row, whose
    window is the same. Returns no rows if the run is too short to hold them.
    """
    if base.duration <= HELD_OUT_START + 3.0:
        return []
    n = len(states)
    window = (HELD_OUT_START, base.duration)
    rows = [("held-out neutral", base.truth)]
    for rate in (7.0, 15.0):
        turns = tuple((i, Turn(HELD_OUT_START, 40.0, rate * DEG)) for i in range(n))
        rows.append((f"held-out turn {rate:g} deg/s", TruthDisturbance(maneuvers=turns)))
    brakes = tuple((i, Acceleration(HELD_OUT_START, 33.0, 3.0)) for i in range(n))
    rows.append(("held-out acceleration 3 m/s^2", TruthDisturbance(maneuvers=brakes)))
    return [
        ImprovementRow(
            label, replace(base, truth=truth, focus_window=window), tuple(arms), states, "held-out"
        )
        for label, truth in rows
    ]


def bias_block_rows(
    base: RobustnessConfig,
    states: np.ndarray,
    arms: Sequence[Arm],
    with_oracle: bool = True,
) -> list[ImprovementRow]:
    """Rows without maneuvers and with a constant camera bias of 0 and BIAS_GRID_DEG degrees."""
    rows = []
    for bias in (0.0, *BIAS_GRID_DEG):
        config = replace(base, truth=TruthDisturbance(camera_bias=bias * DEG))
        row_arms = (*arms, oracle_arm(bias * DEG)) if with_oracle else tuple(arms)
        rows.append(ImprovementRow(f"bias {bias:g} deg", config, row_arms, states, "bias"))
    return rows


def hard_block_rows(
    base: RobustnessConfig, states: np.ndarray, arms: Sequence[Arm]
) -> list[ImprovementRow]:
    """A bias of 0.5 deg alone and together with a turn, an acceleration and random turns."""
    n = len(states)
    bias = HARD_BIAS_DEG * DEG
    turns = tuple((i, Turn(15.0, 25.0, 10.0 * DEG)) for i in range(n))
    brakes = tuple((i, Acceleration(15.0, 18.0, 2.0)) for i in range(n))
    random = RandomManeuvers(RANDOM_ONSET, 5.0, 10.0 * DEG)
    specs = [
        ("bias 0.5 deg", TruthDisturbance(camera_bias=bias), None),
        ("turn 10 deg/s + bias", TruthDisturbance(camera_bias=bias, maneuvers=turns), ONSET),
        (
            "acceleration 2 m/s^2 + bias",
            TruthDisturbance(camera_bias=bias, maneuvers=brakes),
            ONSET,
        ),
        (
            "random turns <= 10 deg/s + bias",
            TruthDisturbance(camera_bias=bias, random_maneuvers=random),
            RANDOM_ONSET,
        ),
    ]
    rows = []
    for label, truth, start in specs:
        config = replace(base, truth=truth)
        if start is not None:
            config = _with_window(config, start)
        rows.append(ImprovementRow(label, config, tuple(arms), states, "hard"))
    return rows


def evaluation_blocks(
    group: str,
    base_for_clutter: Callable[[float], RobustnessConfig],
    scenes: dict[str, np.ndarray],
    clutter_rates: Sequence[float],
    parameters: Parameters,
) -> list[Block]:
    """The table sets of an evaluation group: "maneuver" (tuned and held-out rows), "bias", "hard".

    Args:
        group: One of the three groups.
        base_for_clutter: Neutral configuration at a clutter rate.
        scenes: Layout name -> initial states.
        clutter_rates: Clutter rates of the blocks.
        parameters: Frozen (or default) parameters of the arms.
    """
    arms = named_arms(parameters)
    if group == "maneuver":
        names = (BASELINE, *CONTROL_ARMS, "IMM-A", "IMM-B", "EKF+bias")
    elif group == "bias":
        names = (BASELINE, "EKF+bias", "EKF+always")
    elif group == "hard":
        names = (BASELINE, "EKF+bias", "IMM-A", "IMM-B", "IMM-A+bias", "IMM-B+bias")
    else:
        raise ValueError(f"group must be maneuver, bias or hard, got {group!r}")
    chosen = [arms[name] for name in names]
    blocks = []
    for layout, states in scenes.items():
        for clutter in clutter_rates:
            base = base_for_clutter(clutter)
            if group == "maneuver":
                rows = maneuver_block_rows(base, states, chosen)
                rows += held_out_rows(base, states, chosen)
            elif group == "bias":
                rows = bias_block_rows(base, states, chosen)
            else:
                rows = hard_block_rows(base, states, chosen)
            blocks.append(Block(f"{group}, {layout}, clutter {clutter:g}", layout, clutter, rows))
    return blocks


def tuning_blocks(
    base_for_clutter: Callable[[float], RobustnessConfig],
    scenes: dict[str, np.ndarray],
    clutter_rates: Sequence[float],
    small: bool = False,
) -> list[Block]:
    """The table sets of the tuning run: every candidate on the tuned rows, the baseline alone on
    the held-out rows (only to find the winnable blocks), the bias candidates on the bias rows.
    """
    candidates = tuning_candidates(small)
    everything = [Arm(BASELINE)] + [arm for family in candidates.values() for arm in family]
    bias_only = [Arm(BASELINE), *candidates["bias"]]
    blocks = []
    for layout, states in scenes.items():
        for clutter in clutter_rates:
            base = base_for_clutter(clutter)
            rows = maneuver_block_rows(base, states, everything)
            rows += held_out_rows(base, states, [Arm(BASELINE)])
            rows += bias_block_rows(base, states, bias_only, with_oracle=False)
            blocks.append(Block(f"tuning, {layout}, clutter {clutter:g}", layout, clutter, rows))
    return blocks


# --- trial -------------------------------------------------------------------------------


def n_bins(config: RobustnessConfig) -> int:
    """Number of time bins of a configuration (5 s bins from the end of the burn-in)."""
    return len(bin_edges(config.n_steps + 1, config.dt, config.burn_in, BIN_WIDTH))


def improvement_fields(config: RobustnessConfig) -> tuple[str, ...]:
    """Names of every score of an arm in a trial of this configuration."""
    bins = n_bins(config)
    return (
        *FIELDS,
        *(f"missed_bin_{b:02d}" for b in range(bins)),
        *(f"drag_bin_{b:02d}" for b in range(bins)),
        "window_drag_rate",
        "cs_run_rmse",
        "cs_run_rmse_ref",
        "cs_window_rmse",
        "cs_window_rmse_ref",
        "runtime_s",
        "mode_nonref_window",
        "mode_nonref_outside",
        "mode_max_mu",
        "bias_run_mean_abs_error_deg",
        "bias_window_max_abs_estimate_deg",
        "bias_final_error_deg",
        "bias_converged_025",
        "bias_converged_010",
        "bias_conv_025_s",
        "bias_conv_010_s",
    )


def _score_arm(sim, run, config: RobustnessConfig, runtime: float):
    """Scores of one arm's run and the squared errors of its narrow matching."""
    n_targets, n_steps = sim.truth.shape[:2]
    dt = config.dt
    metrics = evaluate_robustness(
        sim.truth,
        run.history,
        run.tracker.camera_log,
        sim.camera_scans,
        dt=dt,
        fov=config.fov,
        max_distance=config.match_distance,
        wide_distance=config.diagnostic_match_distance,
        burn_in_steps=config.burn_in_steps,
        focus_window=config.focus_window,
    )
    values = metrics._asdict()
    values["run_births_per_scan"] = run.tracker.births / run.tracker.radar_scans

    narrow = match_tracks(sim.truth, run.history, config.match_distance)
    wide = match_tracks(sim.truth, run.history, config.diagnostic_match_distance)
    in_view = config.fov.contains(sim.truth[:, :, :2].reshape(-1, 2)).reshape(n_targets, n_steps)
    bins = bin_edges(n_steps, dt, config.burn_in, BIN_WIDTH)
    missed, drag = binned_rates(narrow.ids, wide.ids, in_view, bins)
    for b in range(len(bins)):
        values[f"missed_bin_{b:02d}"] = float(missed[b])
        values[f"drag_bin_{b:02d}"] = float(drag[b])
    after = np.arange(n_steps) >= config.burn_in_steps
    if config.focus_window is not None:
        window = window_mask(n_steps, config.focus_window, dt)
    else:
        window = np.zeros(n_steps, dtype=bool)
    in_window = in_view & window[None, :]
    if in_window.any():
        values["window_drag_rate"] = float(
            drag_mask(narrow.ids, wide.ids, in_view)[in_window].mean()
        )
    values["runtime_s"] = runtime

    motion = run.tracker.config.motion
    if motion is not None and run.tracker.mode_log:
        mu = mode_probabilities(run.tracker.mode_log, wide.ids, len(motion.modes))
        if config.focus_window is not None:
            values["mode_nonref_window"], _ = mode_summary(mu, window & after)
            values["mode_nonref_outside"], _ = mode_summary(mu, after & ~window)
        else:  # no window: every step after the burn-in counts as outside
            values["mode_nonref_outside"], _ = mode_summary(mu, after)
        _, values["mode_max_mu"] = mode_summary(mu, after)
    if run.tracker.config.camera_bias is not None and run.tracker.bias_log:
        truth_bias = config.truth.camera_bias
        errors = bias_errors(run.tracker.bias_log, truth_bias)
        summary = bias_summary(
            errors, run.tracker.bias_log, window if window.any() else after, config.burn_in_steps
        )
        values["bias_run_mean_abs_error_deg"] = summary["run_mean_abs_error"] / DEG
        values["bias_window_max_abs_estimate_deg"] = summary["window_max_abs_estimate"] / DEG
        values["bias_final_error_deg"] = summary["final_error"] / DEG
        for threshold, key in zip(CONVERGENCE_THRESHOLDS_DEG, ("025", "010"), strict=True):
            seconds = convergence_time(errors, dt, threshold * DEG)
            values[f"bias_conv_{key}_s"] = seconds
            values[f"bias_converged_{key}"] = float(np.isfinite(seconds))
    return values, narrow.squared_error


def run_improvement_trial(
    initial_states: np.ndarray, config: RobustnessConfig, arms: Sequence[Arm], seed: int
) -> dict[str, dict[str, float]]:
    """Simulate one scene once and score every arm on it.

    The first arm is the reference of the common-support scores; it must be the baseline.
    Runtime is the wall-clock time of the tracker run alone, measured in this process.

    Args:
        initial_states: Shape (n_targets, 4) starting states.
        config: Scene, disturbance, beliefs and focus window.
        arms: The arms; names must be distinct.
        seed: Seed of every random stream of the simulation.

    Returns:
        Arm name -> score name -> value, with every name of improvement_fields (NaN where
        a score does not apply to the arm).
    """
    names = [arm.name for arm in arms]
    if len(set(names)) != len(names) or not names:
        raise ValueError(f"arm names must be distinct and not empty, got {names}")
    if names[0] != BASELINE:
        raise ValueError(f"the first arm must be {BASELINE!r}, got {names[0]!r}")
    sim = simulate_mtt(initial_states, config, make_mtt_rngs(seed), config.truth)
    base, radar_model, camera_model = config.tracker_setup()
    fields = improvement_fields(config)
    scores: dict[str, dict[str, float]] = {}
    squared: dict[str, np.ndarray] = {}
    for arm in arms:
        tracker_config = arm.tracker_config(base, config.radar_every)
        started = time.perf_counter()
        run = run_multi_target_tracking(sim, tracker_config, radar_model, camera_model)
        runtime = time.perf_counter() - started
        values, squared[arm.name] = _score_arm(sim, run, config, runtime)
        scores[arm.name] = {name: values.get(name, float("nan")) for name in fields}

    n_steps = sim.truth.shape[1]
    after = np.arange(n_steps) >= config.burn_in_steps
    window = (
        window_mask(n_steps, config.focus_window, config.dt)
        if config.focus_window is not None
        else None
    )
    reference = squared[BASELINE]
    for name in names:
        (own, ref), _ = common_support_rmse([squared[name], reference], after)
        scores[name]["cs_run_rmse"], scores[name]["cs_run_rmse_ref"] = own, ref
        if window is not None:
            (own, ref), _ = common_support_rmse([squared[name], reference], window & after)
            scores[name]["cs_window_rmse"], scores[name]["cs_window_rmse_ref"] = own, ref
    return scores


# --- sweep -------------------------------------------------------------------------------


class RowResult(NamedTuple):
    """Scores of every arm of one row on every seed.

    Attributes:
        label: Row label.
        group: Row group.
        arm_names: Names of the arms, baseline first.
        seeds: The seeds.
        metrics: Score name -> shape (n_arms, n_seeds) array (NaN where undefined).
    """

    label: str
    group: str
    arm_names: tuple[str, ...]
    seeds: tuple[int, ...]
    metrics: dict[str, np.ndarray]


def _run_task(task) -> dict[str, dict[str, float]]:
    """One (states, config, arms, seed) task; module-level to be picklable."""
    return run_improvement_trial(*task)


def improvement_sweep(
    rows: Sequence[ImprovementRow], seeds: Sequence[int], workers: int = 1
) -> list[RowResult]:
    """Run every row on the same seeds, optionally in worker processes.

    Trials are independent and deterministic per seed, so the scores (runtime aside) do not
    depend on the number of workers.

    Raises:
        ValueError: If rows or seeds are empty, or workers < 1.
    """
    if len(rows) == 0 or len(seeds) == 0:
        raise ValueError("rows and seeds must not be empty")
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    tasks = [(row.states, row.config, row.arms, seed) for row in rows for seed in seeds]
    if workers == 1:
        trials = [_run_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            trials = list(pool.map(_run_task, tasks, chunksize=2))
    results = []
    for i, row in enumerate(rows):
        names = tuple(arm.name for arm in row.arms)
        fields = improvement_fields(row.config)
        metrics = {f: np.full((len(names), len(seeds)), np.nan) for f in fields}
        for j in range(len(seeds)):
            trial = trials[i * len(seeds) + j]
            for a, name in enumerate(names):
                for f in fields:
                    metrics[f][a, j] = trial[name][f]
        results.append(RowResult(row.label, row.group, names, tuple(seeds), metrics))
    return results


class BlockResult(NamedTuple):
    """The results of the rows of one block.

    Attributes:
        title: Title of the block.
        layout: Layout (scene) name.
        clutter: Clutter rate of the block.
        rows: One RowResult per row of the block.
    """

    title: str
    layout: str
    clutter: float
    rows: list[RowResult]


def run_blocks(
    blocks: Sequence[Block], seeds: Sequence[int], workers: int = 1
) -> list[BlockResult]:
    """Run every row of every block in one process pool and return the results by block."""
    flat = [row for block in blocks for row in block.rows]
    results = improvement_sweep(flat, seeds, workers)
    out, start = [], 0
    for block in blocks:
        stop = start + len(block.rows)
        out.append(BlockResult(block.title, block.layout, block.clutter, results[start:stop]))
        start = stop
    return out
