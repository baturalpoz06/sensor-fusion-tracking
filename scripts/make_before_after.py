"""Descriptive before / after tables of the Phase 8b evaluation (Phase 8c).

Usage:
    python scripts/make_before_after.py [--pickle results/improvement_results.pkl]
        [--out results/before_after_8c.txt]

Reads the saved evaluation results (seeds 0-49, written by run_improvement_experiment.py --stage
eval) and writes three tables. It runs nothing and changes no criterion, parameter or result.
Every cell is a paired difference per seed against the plain EKF on the same data (or, for an
absolute column, the arm's own value): the mean over seeds with the half width of the 95% t
interval. The tables carry no verdicts.

    benefit      window missed rate on the maneuver and held-out rows, averaged over the winnable
                 row-blocks of each group, for the EKF, the three EKF high-Q controls and both IMMs;
    side effect  the neutral row (no maneuver) in every block, (arm - EKF) of run RMSE, missed
                 rate, ghost rate and ID switches, with the neutral margin of criterion A2 beside
                 each column;
    bias         EKF against EKF+bias at b = 0, 0.5 and 1 deg: run RMSE and cross-range RMS.
"""

import argparse
import pickle
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from fusion.improvement_criteria import MARGINS, arm_index, block_label, neutral_label
from fusion.improvement_experiment import BASELINE, FROZEN, BlockResult, RowResult
from fusion.mtt_experiment import SweepResult
from fusion.outage_report import format_table

ROOT = Path(__file__).resolve().parents[1]
CONTROLS = ("EKF high-Q", "EKF high-Q 3", "EKF high-Q 5")
ARMS = (*CONTROLS, "IMM-A", "IMM-B")
GROUPS = (("maneuver", "maneuver"), ("held-out", "held-out"))
BIAS_ROWS = ("bias 0 deg", "bias 0.5 deg", "bias 1 deg")
LABEL_WIDTH = 40
CONFIDENCE = 0.95
LEGEND = (
    "d = arm - EKF, per seed on the same simulated data (paired); cells are the mean over seeds",
    "  +- the half width of the 95% t interval. Descriptive only: no verdicts.",
)


def display_name(arm: str, high_q: float) -> str:
    """Name of an arm in the tables; the frozen high-Q arm carries its process noise."""
    return f"EKF high-Q {high_q:g}" if arm == "EKF high-Q" else arm


def find_row(block: BlockResult, label: str) -> RowResult:
    """The row of a block with this label."""
    for row in block.rows:
        if row.label == label:
            return row
    raise SystemExit(f"{block.title} has no row {label!r}")


def check_arms(row: RowResult, arms: Sequence[str]) -> None:
    missing = [arm for arm in arms if arm not in row.arm_names]
    if missing:
        raise SystemExit(f"row {row.label!r} of the results lacks the arms {missing}")


def series(row: RowResult, metric: str, arm: str) -> np.ndarray:
    """Shape (n_seeds,) scores of one arm on one row."""
    return row.metrics[metric][arm_index(row, arm)]


def sweep(metrics: dict[str, np.ndarray]) -> SweepResult:
    """The per-seed arrays (one row per table row) as the sweep result format_table reads."""
    n = len(next(iter(metrics.values())))
    return SweepResult("row", np.arange(n, dtype=float), metrics)


def table(
    title: str,
    columns: dict[str, tuple[str, int]],
    metrics: dict[str, np.ndarray],
    labels: Sequence[str],
    first_column: str,
    notes: Sequence[str] = (),
) -> list[str]:
    """One titled table: format_table over per-seed arrays, one row per label."""
    valid = tuple(list(columns)[:2])
    body = format_table(
        sweep(metrics), columns, {}, valid, labels, first_column, None, CONFIDENCE, LABEL_WIDTH
    )
    return [title, *notes, *body]


def benefit_table(
    blocks: Sequence[BlockResult], winnable: Sequence[str], high_q: float
) -> list[str]:
    """Window missed rate, averaged over the winnable row-blocks of each group."""
    arms = (BASELINE, *ARMS)
    metrics: dict[str, np.ndarray] = {}
    columns: dict[str, tuple[str, int]] = {}
    counts = {}
    for group, name in GROUPS:
        per_arm: dict[str, list[np.ndarray]] = {arm: [] for arm in arms}
        for block in blocks:
            for row in block.rows:
                if (
                    row.group != group
                    or row.label == neutral_label(row)
                    or block_label(block, row) not in winnable
                ):
                    continue
                check_arms(row, arms)
                for arm in arms:
                    per_arm[arm].append(series(row, "window_missed_rate", arm))
        counts[group] = len(per_arm[BASELINE])
        if not counts[group]:
            raise SystemExit(f"no winnable row-block of the group {group!r} in the results")
        mean = {arm: np.nanmean(np.array(values), axis=0) for arm, values in per_arm.items()}
        metrics[f"{group}_abs"] = np.array([mean[arm] for arm in arms])
        metrics[f"{group}_diff"] = np.array([mean[arm] - mean[BASELINE] for arm in arms])
        n = counts[group]
        columns[f"{group}_abs"] = (f"missed, {name} ({n})", 3)
        columns[f"{group}_diff"] = (f"d missed, {name}", 3)
    labels = [display_name(arm, high_q) for arm in arms]
    notes = [
        "window missed rate: share of in-view target steps without a matched confirmed track in",
        "  the focus window, averaged per seed over the winnable row-blocks of the group (the",
        f"  number in brackets: {counts['maneuver']} maneuver, {counts['held-out']} held-out;",
        "  both layouts, both clutter rates)",
        *LEGEND,
    ]
    return table("TABLE A: benefit on the maneuver rows", columns, metrics, labels, "arm", notes)


def side_effect_table(blocks: Sequence[BlockResult], high_q: float) -> list[str]:
    """(arm - EKF) on the neutral row of every block, with the A2 margins in the headers."""
    labels: list[str] = []
    stacks: dict[str, list[np.ndarray]] = {metric: [] for metric in MARGINS}
    for block in blocks:
        row = find_row(block, "neutral")
        check_arms(row, (BASELINE, *ARMS))
        for arm in ARMS:
            where = f"{block.layout}, clutter {block.clutter:g}"
            labels.append(f"{where} | {display_name(arm, high_q)}")
            for metric in MARGINS:
                stacks[metric].append(series(row, metric, arm) - series(row, metric, BASELINE))
    names = {
        "run_position_rmse": "d rmse [m]",
        "run_missed_rate": "d missed",
        "run_ghost_rate": "d ghost",
        "run_id_switches": "d id sw",
    }
    columns = {m: (f"{names[m]} (A2 {MARGINS[m]:g})", 3) for m in MARGINS}
    metrics = {m: np.array(values) for m, values in stacks.items()}
    notes = [
        "neutral row (no maneuver, no bias), whole run. The number after A2 in each header is the",
        "  neutral margin of criterion A2 (rmse in m, ghost per step, id sw per run).",
        *LEGEND,
    ]
    title = "TABLE B: side effect on the neutral row"
    return table(title, columns, metrics, labels, "block | arm", notes)


def bias_table(blocks: Sequence[BlockResult]) -> list[str]:
    """EKF against EKF+bias at b = 0, 0.5 and 1 deg: run RMSE and cross-range RMS."""
    labels: list[str] = []
    stacks: dict[str, list[np.ndarray]] = {
        key: [] for key in ("ekf_rmse", "bias_rmse", "d_rmse", "ekf_cross", "bias_cross", "d_cross")
    }
    for block in blocks:
        for name in BIAS_ROWS:
            row = find_row(block, name)
            check_arms(row, (BASELINE, "EKF+bias"))
            labels.append(f"{block.layout}, clutter {block.clutter:g} | {name}")
            for short, metric in (("rmse", "run_position_rmse"), ("cross", "cross_rms")):
                before, after = series(row, metric, BASELINE), series(row, metric, "EKF+bias")
                stacks[f"ekf_{short}"].append(before)
                stacks[f"bias_{short}"].append(after)
                stacks[f"d_{short}"].append(after - before)
    columns = {
        "ekf_rmse": ("EKF rmse [m]", 3),
        "bias_rmse": ("EKF+bias rmse [m]", 3),
        "d_rmse": ("d rmse [m]", 3),
        "ekf_cross": ("EKF cross [m]", 3),
        "bias_cross": ("EKF+bias cross [m]", 3),
        "d_cross": ("d cross [m]", 3),
    }
    metrics = {key: np.array(values) for key, values in stacks.items()}
    notes = [
        "constant camera bias of 0, 0.5 and 1 deg, no maneuver; d = EKF+bias - EKF; rmse is the",
        "  run position error, cross is the cross-range rms (200 m match distance).",
        *LEGEND,
    ]
    return table("TABLE C: camera bias estimate", columns, metrics, labels, "block | bias", notes)


def build_report(
    results: dict[str, list[BlockResult]], winnable: Sequence[str], high_q: float
) -> list[str]:
    """The three tables as text lines.

    Raises:
        SystemExit: If the results lack a group, a row or an arm that a table needs.
    """
    for group in ("maneuver", "bias"):
        if group not in results or not results[group]:
            raise SystemExit(f"the results have no {group!r} group")
    seeds = len(next(iter(results["maneuver"][0].rows[0].metrics.values()))[0])
    header = [
        f"BEFORE / AFTER (Phase 8c): {seeds} seeds, from the saved evaluation results; no new run.",
        "",
    ]
    return [
        *header,
        *benefit_table(results["maneuver"], winnable, high_q),
        "",
        *side_effect_table(results["maneuver"], high_q),
        "",
        *bias_table(results["bias"]),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pickle", type=Path, default=ROOT / "results" / "improvement_results.pkl")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "before_after_8c.txt")
    args = parser.parse_args()
    if FROZEN is None:
        sys.exit("improvement_experiment.FROZEN is not set: there are no frozen parameters")
    with args.pickle.open("rb") as handle:
        results = pickle.load(handle)
    lines = build_report(results, FROZEN.winnable, FROZEN.ekf_high_accel_std)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
