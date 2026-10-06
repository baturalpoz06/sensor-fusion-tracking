"""Tests of the tuning selection and the success criteria on synthetic paired results.

Each test docstring names the defect that makes it fail. The data are built so that the right
verdict is known in advance (a pass, a fail, a worsening, an underpowered comparison).
"""

import numpy as np
import pytest

from fusion.improvement_criteria import (
    criterion_a1,
    criterion_a2,
    criterion_b1,
    criterion_b2,
    criterion_b3,
    criterion_hdrag,
    equivalence,
    judge,
    non_inferior,
    power_lines,
    practical_threshold,
    projected_half_width,
    select_bias_prior,
    select_family,
    violation,
    winnable_blocks,
)
from fusion.improvement_experiment import BASELINE, BlockResult, Parameters, RowResult
from fusion.mtt_metrics import PairedDifference

N = 50
BINS = 11
READ = (
    "window_missed_rate run_ghost_rate run_id_switches window_position_rmse cs_window_rmse "
    "cs_window_rmse_ref run_position_rmse run_missed_rate cross_rms bias_run_mean_abs_error_deg "
    "runtime_s cs_run_rmse cs_run_rmse_ref"
).split() + [f"missed_bin_{b:02d}" for b in range(BINS)]


def make_row(label: str, group: str, arms: dict[str, dict], seed: int = 0) -> RowResult:
    """A row result: every arm has the same seed noise plus its own shifts.

    `arms` maps an arm name to {metric: shift}. A shift is added to a per-metric common seed
    noise, so arms differ only by their shift plus a small paired jitter.
    """
    rng = np.random.default_rng(seed)
    names = tuple(arms)
    metrics = {}
    for metric in READ:
        common = rng.normal(0.0, 0.01, N)
        rows = []
        for name in names:
            shift = arms[name].get(metric, 0.0)
            rows.append(common + shift + rng.normal(0.0, 0.0005, N))
        metrics[metric] = np.array(rows)
    return RowResult(label, group, names, tuple(range(N)), metrics)


def block(rows: list[RowResult], layout="separated", clutter=0.0) -> BlockResult:
    return BlockResult(f"{layout} {clutter:g}", layout, clutter, rows)


def maneuver_block(arm_effect: dict, degradation: float = 0.1, n_rows: int = 3, seed=0):
    """A neutral row and n_rows maneuver rows where the baseline loses `degradation` missed."""
    neutral = make_row("neutral", "maneuver", {BASELINE: {}, **{a: {} for a in arm_effect}}, seed)
    rows = [neutral]
    for i in range(n_rows):
        arms = {BASELINE: {"window_missed_rate": degradation}}
        for arm, shifts in arm_effect.items():
            shift = shifts(i)
            arms[arm] = {
                **shift,
                "window_missed_rate": degradation + shift.get("window_missed_rate", 0.0),
            }
        rows.append(make_row(f"turn {i}", "maneuver", arms, seed + 1 + i))
    return block(rows)


def diff(mean: float, half: float) -> PairedDifference:
    return PairedDifference(mean, mean - half, mean + half, mean, N, 0)


def test_the_practical_threshold_is_a_floor_or_a_quarter_of_the_degradation():
    """Fails if a floor or the 25% rule is wrong, or NaN degradation breaks the floor."""
    assert practical_threshold("missed", 0.0) == 0.01
    assert practical_threshold("missed", 0.2) == pytest.approx(0.05)
    assert practical_threshold("rmse", 3.0) == 1.0
    assert practical_threshold("rmse", 8.0) == pytest.approx(2.0)
    assert practical_threshold("rmse", -5.0) == 1.0
    assert practical_threshold("ghost", 9.0) == 0.02 and practical_threshold("id", 9.0) == 0.5
    assert practical_threshold("missed", float("nan")) == 0.01


def test_judge_needs_an_interval_that_excludes_zero_and_a_large_enough_mean():
    """Fails if a significant but tiny effect counts, or a large but uncertain one does."""
    assert judge(diff(-0.05, 0.01), 0.02) == "improves"
    assert judge(diff(-0.01, 0.005), 0.02) == "neither"  # excludes zero, but too small
    assert judge(diff(-0.05, 0.08), 0.02) == "neither"  # large, but the interval includes zero
    assert judge(diff(0.05, 0.01), 0.02) == "worsens"
    assert judge(PairedDifference(0.0, np.nan, np.nan, 0.0, 1, 0), 0.1) == "undefined"


def test_equivalence_and_non_inferiority_tell_pass_from_underpowered_from_fail():
    """Fails if a wide interval is called a pass, or a shifted one is called underpowered."""
    assert equivalence(diff(0.0, 0.1), 0.3) == "PASS"
    assert equivalence(diff(0.05, 0.4), 0.3) == "UNDERPOWERED"
    assert equivalence(diff(0.5, 0.1), 0.3) == "FAIL"
    assert non_inferior(diff(-1.0, 0.2), 0.3) == "PASS"  # far better than the baseline
    assert non_inferior(diff(0.1, 0.4), 0.3) == "UNDERPOWERED"
    assert non_inferior(diff(0.6, 0.1), 0.3) == "FAIL"
    assert violation(diff(0.0, 0.1), 0.3, True) == 0.0
    assert violation(diff(0.2, 0.2), 0.3, True) == pytest.approx(0.1 / 0.3)
    assert violation(diff(-0.4, 0.2), 0.3, True) == pytest.approx(0.3 / 0.3)
    assert violation(diff(-0.4, 0.2), 0.3, False) == 0.0


def test_winnable_blocks_are_those_where_the_baseline_loses_more_than_the_floor():
    """Fails if a block without a real degradation enters the denominator or a real one does not."""
    neutral = make_row("neutral", "maneuver", {BASELINE: {}}, 1)
    big = make_row("turn big", "maneuver", {BASELINE: {"window_missed_rate": 0.05}}, 2)
    small = make_row("turn small", "maneuver", {BASELINE: {"window_missed_rate": 0.004}}, 3)
    held_neutral = make_row("held-out neutral", "held-out", {BASELINE: {}}, 4)
    held = make_row("held-out turn 7 deg/s", "held-out", {BASELINE: {"window_missed_rate": 0.1}}, 5)
    labels = winnable_blocks([block([neutral, big, small, held_neutral, held], clutter=5.0)])
    assert labels == ("separated | 5 | turn big", "separated | 5 | held-out turn 7 deg/s")


def improving(shift: float):
    return lambda i: {"window_missed_rate": -shift}


def test_a1_passes_when_most_winnable_blocks_improve_and_none_worsens():
    """Fails if a clear improvement in every winnable block is not a pass."""
    b = maneuver_block({"IMM-A": lambda i: {"window_missed_rate": -0.06}})
    winnable = tuple(f"separated | 0 | turn {i}" for i in range(3))
    result = criterion_a1([b], "IMM-A", winnable, "maneuver")
    assert result.verdict == "PASS", result.lines


def test_a1_fails_if_too_few_winnable_blocks_improve():
    """Fails if one improvement in three winnable blocks (need two) passes."""
    b = maneuver_block(
        {"IMM-A": lambda i: {"window_missed_rate": -0.06 if i == 0 else 0.0}}
    )
    winnable = tuple(f"separated | 0 | turn {i}" for i in range(3))
    assert criterion_a1([b], "IMM-A", winnable, "maneuver").verdict == "FAIL"


@pytest.mark.parametrize(
    ("metric", "shift"),
    [("run_ghost_rate", 0.1), ("run_id_switches", 2.0), ("window_missed_rate", 0.1)],
)
def test_a1_fails_if_any_row_block_worsens_a_primary_score(metric, shift):
    """Fails if a worsening of ghosts, ID switches or misses in one block is overlooked."""
    def effects(i):
        out = {"window_missed_rate": -0.06}
        if i == 2:
            out[metric] = out.get(metric, 0.0) + shift if metric != "window_missed_rate" else shift
        return out

    b = maneuver_block({"IMM-A": effects})
    winnable = tuple(f"separated | 0 | turn {i}" for i in range(3))
    assert criterion_a1([b], "IMM-A", winnable, "maneuver").verdict == "FAIL"


def test_a1_is_not_applicable_without_winnable_blocks():
    """Fails if an empty denominator gives a verdict."""
    b = maneuver_block({"IMM-A": lambda i: {"window_missed_rate": -0.06}})
    assert criterion_a1([b], "IMM-A", (), "maneuver").verdict == "N/A"


def test_a1_lists_a_worse_common_support_rmse_without_failing():
    """Fails if the secondary score decides the verdict, or is not reported."""
    b = maneuver_block(
        {"IMM-A": lambda i: {"window_missed_rate": -0.06, "cs_window_rmse": 3.0}}
    )
    winnable = tuple(f"separated | 0 | turn {i}" for i in range(3))
    result = criterion_a1([b], "IMM-A", winnable, "maneuver")
    assert result.verdict == "PASS"
    assert any("cs window rmse worsens in 3" in line for line in result.lines)


def neutral_block(shifts: dict, label="neutral", group="maneuver", seed=7) -> BlockResult:
    arms = {BASELINE: {}, "IMM-A": shifts}
    return block([make_row(label, group, arms, seed)])


def test_a2_passes_fails_and_flags_underpowered_neutral_comparisons():
    """Fails if a harmful arm passes, or a wide interval is read as a fail."""
    assert criterion_a2([neutral_block({})], "IMM-A").verdict == "PASS"
    assert criterion_a2([neutral_block({"run_position_rmse": 1.0})], "IMM-A").verdict == "FAIL"
    assert criterion_a2([neutral_block({"run_ghost_rate": 0.05})], "IMM-A").verdict == "FAIL"
    # Noise wider than the margin: the mean is inside, the interval cannot show it.
    wide = make_row("neutral", "maneuver", {BASELINE: {}, "IMM-A": {}}, 3)
    wide.metrics["run_id_switches"][1] += np.tile([4.0, -4.0], N // 2)  # mean 0, SD 4
    result = criterion_a2([block([wide])], "IMM-A")
    assert result.verdict == "UNDERPOWERED", result.lines


def drag_row(effect: float, seed: int = 0) -> RowResult:
    arms = {BASELINE: {}, "EKF high-Q": {}, "IMM-A": {}}
    for b in (4, 5, 6, 7):
        arms[BASELINE][f"missed_bin_{b:02d}"] = 0.2
        arms["EKF high-Q"][f"missed_bin_{b:02d}"] = 0.2 + effect
        arms["IMM-A"][f"missed_bin_{b:02d}"] = 0.2 + effect
    return make_row("turn 5 deg/s", "maneuver", arms, seed)


@pytest.mark.parametrize(
    ("effect", "expected"), [(-0.15, "SUPPORTED"), (0.0, "REFUTED"), (-0.05, "PARTIAL")]
)
def test_h_drag_reads_the_high_q_ekf_reduction_of_the_tail(effect, expected):
    """Fails if a 75% reduction is not supported, no effect is not refuted, or a 25% reduction
    (significant but short of half) is called either."""
    blocks = [block([drag_row(effect, 1)], clutter=0.0), block([drag_row(effect, 2)], clutter=5.0)]
    result = criterion_hdrag(blocks, ("EKF high-Q", "IMM-A"))
    assert result.verdict == expected, result.lines
    assert any("seeds 10+" in line for line in result.lines)


def test_h_drag_with_fewer_than_ten_seeds_has_no_late_seed_numbers_and_does_not_warn():
    """Fails if the subset of the seeds from 10 on, empty in a 5-seed pilot, raises a warning or
    changes the verdict."""
    import warnings

    row = drag_row(-0.15, 1)
    short = RowResult(
        row.label, row.group, row.arm_names, tuple(range(5)),
        {name: values[:, :5] for name, values in row.metrics.items()},
    )  # fmt: skip
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = criterion_hdrag([block([short])], ("EKF high-Q",))
    assert result.verdict in ("SUPPORTED", "PARTIAL", "REFUTED")
    assert any("seeds 10+" in line and "nan" in line for line in result.lines)


def test_h_drag_is_not_applicable_without_the_separated_turn_row():
    """Fails if a crossing-only run gives a verdict."""
    crossing = block([drag_row(0.0)], layout="crossing")
    assert criterion_hdrag([crossing], ("EKF high-Q",)).verdict == "N/A"


def bias_block(effect_05: float, effect_0: float = 0.0, seed: int = 0) -> BlockResult:
    rows = []
    for label, base_cost, effect in (
        ("bias 0 deg", 0.0, effect_0),
        ("bias 0.5 deg", 10.0, -effect_05),
        ("bias 1 deg", 24.0, -effect_05),
    ):
        arms = {
            BASELINE: {"run_position_rmse": base_cost, "cross_rms": base_cost},
            "EKF+bias": {
                "run_position_rmse": base_cost + effect,
                "cross_rms": base_cost + effect,
            },
        }
        rows.append(make_row(label, "bias", arms, seed + len(rows)))
    return block(rows)


def test_b1_requires_an_improvement_at_half_a_degree_and_one_degree():
    """Fails if a large gain is not a pass, or no gain is."""
    # At 1 deg the baseline loses 24 m, so the practical threshold is 6 m: a gain of 8 m passes.
    assert criterion_b1([bias_block(effect_05=8.0)]).verdict == "PASS"
    assert criterion_b1([bias_block(effect_05=0.0)]).verdict == "FAIL"


def test_b2_wants_the_whole_interval_inside_the_margins_and_reports_zero_inclusion():
    """Fails if a harmful neutral arm passes, or the report omits whether zero is inside."""
    ok = criterion_b2([bias_block(6.0, effect_0=0.0)])
    assert ok.verdict == "PASS" and all("interval includes 0" in line for line in ok.lines)
    assert criterion_b2([bias_block(6.0, effect_0=2.0)]).verdict == "FAIL"


def test_b3_catches_an_estimate_that_reads_a_maneuver_as_bias():
    """Fails if a maneuver-driven bias estimate of 0.5 deg is accepted."""
    def run_rows(drift: float) -> BlockResult:
        neutral = make_row("neutral", "maneuver", {BASELINE: {}, "EKF+bias": {}}, 1)
        rows = [neutral]
        for i in range(2):
            arms = {
                BASELINE: {},
                "EKF+bias": {"bias_run_mean_abs_error_deg": drift},
            }
            rows.append(make_row(f"turn {i}", "maneuver", arms, 2 + i))
        return block(rows)

    assert criterion_b3([run_rows(0.0)]).verdict == "PASS"
    assert criterion_b3([run_rows(0.5)]).verdict == "FAIL"


def family_blocks(effects: dict[str, dict]) -> list[BlockResult]:
    """Tuning results: a neutral row and one maneuver row per layout; effects per candidate."""
    out = []
    for seed, layout in enumerate(("separated", "crossing")):
        names = {BASELINE: {"window_missed_rate": 0.1, "cs_window_rmse": 5.0}}
        names.update(effects)
        neutral_arms = {
            n: {k: v for k, v in e.items() if k.startswith("run_")} for n, e in names.items()
        }
        neutral_arms[BASELINE] = {}
        neutral = make_row("neutral", "maneuver", neutral_arms, 10 + seed)
        row = make_row("turn 10 deg/s", "maneuver", names, 20 + seed)
        out.append(block([neutral, row], layout=layout))
    return out


def test_family_selection_excludes_the_infeasible_and_breaks_ties_by_rmse():
    """Fails if the best-missed candidate is taken although it harms the neutral world, or a
    near-tie is not broken by the common-support RMSE."""
    effects = {
        "A": {"window_missed_rate": 0.05, "cs_window_rmse": 4.0},
        "B": {"window_missed_rate": 0.0495, "cs_window_rmse": 3.0},
        "C": {"window_missed_rate": 0.01, "cs_window_rmse": 2.0, "run_position_rmse": 1.0},
    }
    selection = select_family(family_blocks(effects), ["A", "B", "C"])
    assert selection.name == "B" and not selection.fallback
    assert len(selection.lines) == 3


def test_family_selection_falls_back_to_the_least_violating_candidate():
    """Fails if no feasible candidate gives an error instead of a flagged least-bad choice."""
    effects = {
        "A": {"window_missed_rate": 0.05, "run_position_rmse": 2.0},
        "B": {"window_missed_rate": 0.05, "run_position_rmse": 1.0},
    }
    selection = select_family(family_blocks(effects), ["A", "B"])
    assert selection.fallback and selection.name == "B"


def prior_blocks(neutral_cost: dict[str, float], gain: dict[str, float], drift: dict[str, float]):
    arms = lambda extra: {BASELINE: {}, **extra}  # noqa: E731
    maneuver = []
    bias = []
    for seed in range(2):
        rows = [make_row("neutral", "maneuver", arms({p: {} for p in gain}), seed)]
        rows.append(
            make_row(
                "turn 10 deg/s",
                "maneuver",
                arms({p: {"bias_run_mean_abs_error_deg": drift[p]} for p in gain}),
                5 + seed,
            )
        )
        maneuver.append(block(rows))
        costs = {"bias 0 deg": neutral_cost, "bias 0.5 deg": {p: -g for p, g in gain.items()}}
        costs["bias 1 deg"] = costs["bias 0.5 deg"]
        brow = [
            make_row(
                label, "bias", arms({p: {"run_position_rmse": c[p]} for p in gain}), 9 + k
            )
            for k, (label, c) in enumerate(costs.items())
        ]
        bias.append(block(brow))
    return maneuver, bias


def test_prior_selection_takes_the_largest_gain_among_the_feasible_priors():
    """Fails if an infeasible prior wins on gain, or the gain does not decide among the rest."""
    maneuver, bias = prior_blocks(
        neutral_cost={"p1": 1.5, "p.5": 0.0, "p.1": 0.0},
        gain={"p1": 9.0, "p.5": 6.0, "p.1": 4.0},
        drift={"p1": 0.0, "p.5": 0.0, "p.1": 0.0},
    )
    selection = select_bias_prior(maneuver, bias, ["p1", "p.5", "p.1"])
    assert selection.name == "p.5" and not selection.fallback


def test_prior_selection_rejects_a_prior_that_reads_maneuvers_as_bias():
    """Fails if B3 plays no role in the selection of the prior."""
    maneuver, bias = prior_blocks(
        neutral_cost={"p.5": 0.0, "p.1": 0.0},
        gain={"p.5": 9.0, "p.1": 4.0},
        drift={"p.5": 0.5, "p.1": 0.0},
    )
    assert select_bias_prior(maneuver, bias, ["p.5", "p.1"]).name == "p.1"


def test_the_power_lines_flag_margins_narrower_than_the_projected_interval():
    """Fails if a noisy neutral difference is not flagged as unable to meet its margin."""
    noisy = make_row("neutral", "maneuver", {BASELINE: {}, "IMM-A": {}}, 1)
    noisy.metrics["run_id_switches"][1] += np.random.default_rng(1).normal(0, 4.0, N)
    lines = power_lines([block([noisy])], ["IMM-A"], 50)
    assert len(lines) == 1 and "run_id_switches" in lines[0] and "WIDER" in lines[0]
    assert projected_half_width(1.0, 50) == pytest.approx(2.0096 / np.sqrt(50), rel=1e-3)


def test_parameters_are_hashable_values():
    """Fails if the frozen parameters cannot be compared or sent to a process pool."""
    a = Parameters(2.0, 3.0, 10.0, 1.0, 0.5, ("x",), "abc")
    assert a == Parameters(2.0, 3.0, 10.0, 1.0, 0.5, ("x",), "abc") and hash(a) == hash(a)
