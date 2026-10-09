"""The three figures of the README, drawn from the saved Phase 8b evaluation (no new run).

Usage:
    python scripts/make_readme_figures.py [--pickle results/improvement_results.pkl]
        [--out docs/figures]

Reads the evaluation results (seeds 0-49, written by run_improvement_experiment.py --stage eval)
and writes three PNGs. Every number is computed from the per-seed arrays of the pickle; the
script prints the plotted means next to the tables of results/before_after_8c.txt they must equal.

    benefit_cost.png  x: neutral-row RMSE cost against the plain EKF [m], mean over the maneuver
                      blocks; y: reduction of the window missed rate on the winnable maneuver
                      row-blocks (EKF minus arm). One point per arm, 95% t intervals over seeds of
                      the per-seed paired values, dashed line at the neutral RMSE margin of
                      criterion A2.
    camera_bias.png   Run RMSE of the EKF and of the EKF with the camera-bias estimate at a camera
                      bias of 0, 0.5 and 1 deg (clutter 0, both layouts), with 95% t intervals.
    drag_tail.png     Missed rate in 5 s bins on the separated layout, clutter 0, 5 deg/s turn, for
                      the EKF, the EKF with the frozen higher process noise and IMM-B, with 95% t
                      bands over seeds.

All scores are at the 50 m match distance. Colors are fixed per arm across the figures.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from fusion.improvement_criteria import MARGINS, arm_index, block_label, neutral_label
from fusion.improvement_experiment import (
    BASELINE,
    FROZEN,
    ONSET,
    BlockResult,
    RowResult,
)
from fusion.mtt_metrics import SeedInterval, seed_confidence_interval

ROOT = Path(__file__).resolve().parents[1]
HERO_ARMS = ("EKF high-Q", "EKF high-Q 3", "EKF high-Q 5", "IMM-A", "IMM-B")
BIAS_ARMS = (BASELINE, "EKF+bias")
BIAS_ROWS = (("bias 0 deg", 0.0), ("bias 0.5 deg", 0.5), ("bias 1 deg", 1.0))
BIAS_LAYOUTS = ("separated", "crossing")
DRAG_ARMS = (BASELINE, "EKF high-Q", "IMM-B")
DRAG_LAYOUT = "separated"
DRAG_ROW = "turn 5 deg/s"
BIN_WIDTH = 5.0  # [s], the width of the missed_bin_NN scores
BIN_START = 5.0  # [s], the burn-in
DRAG_BINS = 11  # bins [5, 10) .. [55, 60); the last bin of the pickle is a single step at 60 s
TURN_END = 25.0  # [s]; the 5 deg/s turn runs from ONSET to here (robustness_experiment defaults)
CONFIDENCE = 0.95
FILES = ("benefit_cost.png", "camera_bias.png", "drag_tail.png")

# One color per arm in all figures: the first six slots of the reference categorical palette in
# their fixed order, and a neutral for the baseline.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"
ARM_COLORS = {
    BASELINE: INK_MUTED,
    "IMM-A": "#2a78d6",
    "IMM-B": "#eb6834",
    "EKF high-Q": "#1baf7a",
    "EKF high-Q 3": "#eda100",
    "EKF high-Q 5": "#e87ba4",
    "EKF+bias": "#008300",
}


def display_name(arm: str, high_q: float) -> str:
    """Name of an arm in the figures; the frozen high-Q arm carries its process noise."""
    return f"EKF high-Q {high_q:g}" if arm == "EKF high-Q" else arm


def find_block(blocks: Sequence[BlockResult], layout: str, clutter: float) -> BlockResult:
    for block in blocks:
        if block.layout == layout and block.clutter == clutter:
            return block
    raise SystemExit(f"no block {layout!r} with clutter {clutter:g} in the results")


def find_row(block: BlockResult, label: str) -> RowResult:
    for row in block.rows:
        if row.label == label:
            return row
    raise SystemExit(f"{block.title} has no row {label!r}")


def series(row: RowResult, metric: str, arm: str) -> np.ndarray:
    """Shape (n_seeds,) scores of one arm on one row."""
    if arm not in row.arm_names:
        raise SystemExit(f"row {row.label!r} of the results lacks the arm {arm!r}")
    if metric not in row.metrics:
        raise SystemExit(f"row {row.label!r} of the results lacks the score {metric!r}")
    return row.metrics[metric][arm_index(row, arm)]


def group(results: dict[str, list[BlockResult]], name: str) -> list[BlockResult]:
    if not results.get(name):
        raise SystemExit(f"the results have no {name!r} group")
    return results[name]


def interval(values: np.ndarray) -> SeedInterval:
    return seed_confidence_interval(np.asarray(values, dtype=float), CONFIDENCE)


def hero_points(
    results: dict[str, list[BlockResult]],
    winnable: Sequence[str],
    arms: Sequence[str] = HERO_ARMS,
) -> dict[str, tuple[SeedInterval, SeedInterval]]:
    """Per arm, the intervals of (neutral-row RMSE cost, window missed-rate reduction).

    Both are per-seed paired values against the plain EKF on the same data. The cost is the mean
    over the maneuver blocks of the difference of run_position_rmse on the neutral row. The
    reduction is minus the difference of the window missed rate, each arm averaged per seed over
    the winnable maneuver row-blocks first (as Table A of before_after_8c.txt does).

    Raises:
        SystemExit: If the results lack a group, a row, an arm or any winnable row-block.
    """
    blocks = group(results, "maneuver")
    points = {}
    for arm in arms:
        costs = []
        for block in blocks:
            neutral = find_row(block, "neutral")
            costs.append(
                series(neutral, "run_position_rmse", arm)
                - series(neutral, "run_position_rmse", BASELINE)
            )
        rates: dict[str, list[np.ndarray]] = {arm: [], BASELINE: []}
        for block in blocks:
            for row in block.rows:
                if (
                    row.group != "maneuver"
                    or row.label == neutral_label(row)
                    or block_label(block, row) not in winnable
                ):
                    continue
                for name in rates:
                    rates[name].append(series(row, "window_missed_rate", name))
        if not rates[BASELINE]:
            raise SystemExit("no winnable maneuver row-block in the results")
        mean = {name: np.nanmean(np.array(values), axis=0) for name, values in rates.items()}
        points[arm] = (interval(np.mean(costs, axis=0)), interval(mean[BASELINE] - mean[arm]))
    return points


def bias_points(
    results: dict[str, list[BlockResult]],
) -> dict[str, dict[str, list[SeedInterval]]]:
    """Layout -> arm -> interval of the run RMSE at each bias of BIAS_ROWS (clutter 0)."""
    blocks = group(results, "bias")
    points: dict[str, dict[str, list[SeedInterval]]] = {}
    for layout in BIAS_LAYOUTS:
        block = find_block(blocks, layout, 0.0)
        rows = [find_row(block, label) for label, _ in BIAS_ROWS]
        points[layout] = {
            arm: [interval(series(row, "run_position_rmse", arm)) for row in rows]
            for arm in BIAS_ARMS
        }
    return points


def drag_series(
    results: dict[str, list[BlockResult]],
) -> tuple[np.ndarray, dict[str, list[SeedInterval]]]:
    """Bin start times [s] and, per arm, the interval of the missed rate in each 5 s bin."""
    block = find_block(group(results, "maneuver"), DRAG_LAYOUT, 0.0)
    row = find_row(block, DRAG_ROW)
    starts = BIN_START + BIN_WIDTH * np.arange(DRAG_BINS)
    out = {
        arm: [interval(series(row, f"missed_bin_{i:02d}", arm)) for i in range(DRAG_BINS)]
        for arm in DRAG_ARMS
    }
    return starts, out


def seeds_text(results: dict[str, list[BlockResult]]) -> str:
    seeds = next(iter(group(results, "maneuver")[0].rows)).seeds
    return f"seeds {min(seeds)}-{max(seeds)}"


def errors(intervals: Sequence[SeedInterval]) -> np.ndarray:
    """Shape (2, n) lower and upper distances of the intervals from their means."""
    means = np.array([i.mean for i in intervals])
    low = np.array([i.lower for i in intervals])
    high = np.array([i.upper for i in intervals])
    return np.array([np.maximum(means - low, 0.0), np.maximum(high - means, 0.0)])


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelsize=8.5, length=0)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def new_figure(width: float, height: float, ncols: int = 1, **kwargs):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, ncols, figsize=(width, height), facecolor=SURFACE, **kwargs)
    return fig, np.atleast_1d(axes)


def finish(fig, path: Path, title: str, footnote: str) -> Path:
    fig.suptitle(title, x=0.02, ha="left", fontsize=10.5, color=INK, fontweight="bold")
    fig.text(0.02, 0.012, footnote, ha="left", va="bottom", fontsize=7, color=INK_MUTED)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path


# Label offsets of the hero points: (dx, dy) in points and horizontal alignment.
HERO_LABELS = {
    "EKF high-Q": (-9, -4, "right"),
    "EKF high-Q 3": (0, 11, "center"),
    "EKF high-Q 5": (0, 11, "center"),
    "IMM-A": (0, 11, "center"),
    "IMM-B": (10, 3, "left"),
}


def plot_hero(
    points: dict[str, tuple[SeedInterval, SeedInterval]], path: Path, high_q: float, seeds: str
) -> Path:
    fig, (ax,) = new_figure(7.4, 4.7)
    style(ax)
    margin = MARGINS["run_position_rmse"]
    ax.axvline(margin, color=INK_MUTED, linestyle="--", linewidth=1.2, zorder=1)
    ax.annotate(
        f"neutral RMSE margin\nof criterion A2: {margin:g} m",
        xy=(margin, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(5, -4),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=8,
        color=INK_MUTED,
    )
    ax.plot([0], [0], marker="o", markersize=9, markerfacecolor=SURFACE, markeredgewidth=2,
            markeredgecolor=ARM_COLORS[BASELINE], zorder=3)  # fmt: skip
    ax.annotate("EKF (reference)", xy=(0, 0), xytext=(9, -4), textcoords="offset points",
                ha="left", va="top", fontsize=8.5, color=INK)  # fmt: skip
    for arm, (cost, gain) in points.items():
        color = ARM_COLORS[arm]
        ax.errorbar(
            [cost.mean], [gain.mean],
            xerr=errors([cost]), yerr=errors([gain]),
            fmt="o", markersize=9, color=color, markeredgecolor=SURFACE, markeredgewidth=1.5,
            elinewidth=1.6, capsize=3, zorder=3,
        )  # fmt: skip
        dx, dy, align = HERO_LABELS[arm]
        ax.annotate(display_name(arm, high_q), xy=(cost.mean, gain.mean), xytext=(dx, dy),
                    textcoords="offset points", ha=align, va="bottom" if dy > 0 else "top",
                    fontsize=9, color=INK)  # fmt: skip
    ax.set_xlim(-0.08, 1.6)
    ax.set_ylim(-0.006, 0.1)
    ax.set_xlabel("Neutral-row RMSE cost against EKF [m]  (mean of the maneuver blocks)")
    ax.set_ylabel("Window missed-rate reduction  (EKF minus arm)")
    fig.subplots_adjust(left=0.115, right=0.97, top=0.89, bottom=0.2)
    return finish(
        fig, path, f"Window missed-rate reduction against neutral-row RMSE cost ({seeds})",
        f"Source: results/improvement_results.pkl, {seeds}, match distance 50 m. Bars: 95% t "
        "interval over seeds, paired vs EKF.",
    )  # fmt: skip


def plot_bias(
    points: dict[str, dict[str, list[SeedInterval]]], path: Path, seeds: str
) -> Path:
    fig, axes = new_figure(7.4, 4.2, ncols=2, sharey=True)
    x = np.arange(len(BIAS_ROWS))
    for ax, layout in zip(axes, BIAS_LAYOUTS, strict=True):
        style(ax)
        ekf, corrected = points[layout][BASELINE], points[layout]["EKF+bias"]
        level = ekf[0].mean
        ax.axhline(level, color=INK_MUTED, linestyle="--", linewidth=1.2, zorder=1)
        ax.annotate(f"EKF without bias: {level:.2f} m", xy=(x[-1], level), xytext=(-4, 4),
                    textcoords="offset points", ha="right", va="bottom", fontsize=8,
                    color=INK_MUTED)  # fmt: skip
        for arm, values in ((BASELINE, ekf), ("EKF+bias", corrected)):
            ax.errorbar(
                x, [v.mean for v in values], yerr=errors(values),
                color=ARM_COLORS[arm], marker="o", markersize=8, markeredgecolor=SURFACE,
                markeredgewidth=1.5, linewidth=2, elinewidth=1.6, capsize=3, zorder=3,
            )  # fmt: skip
            ax.annotate(arm, xy=(x[-1], values[-1].mean), xytext=(10, 0),
                        textcoords="offset points", ha="left", va="center", fontsize=9,
                        color=INK)  # fmt: skip
        gap = corrected[-1].mean - level
        ax.annotate(
            "", xy=(x[-1] + 0.12, level), xytext=(x[-1] + 0.12, corrected[-1].mean),
            arrowprops={"arrowstyle": "<->", "color": INK_MUTED, "linewidth": 1.2},
        )  # fmt: skip
        ax.annotate(f"{gap:.1f} m above\nthe no-bias level", xy=(x[-1] + 0.12, level + gap / 2),
                    xytext=(-6, 0), textcoords="offset points", ha="right", va="center",
                    fontsize=8, color=INK_MUTED)  # fmt: skip
        ax.set_xticks(x, [f"{value:g}" for _, value in BIAS_ROWS])
        ax.set_xlim(-0.25, x[-1] + 1.0)
        ax.set_xlabel("Camera bearing bias [deg]")
        ax.set_title(f"{layout} layout, clutter 0", loc="left", fontsize=9.5, color=INK)
    axes[0].set_ylabel("Run position RMSE [m]")
    fig.subplots_adjust(left=0.09, right=0.98, top=0.84, bottom=0.2, wspace=0.08)
    return finish(
        fig, path, "Run RMSE of the EKF with and without the camera-bias estimate",
        f"Source: results/improvement_results.pkl, {seeds}, match distance 50 m. Bars: 95% t "
        "interval over seeds.",
    )  # fmt: skip


def plot_drag(
    starts: np.ndarray, data: dict[str, list[SeedInterval]], path: Path, high_q: float, seeds: str
) -> Path:
    fig, (ax,) = new_figure(7.4, 4.4)
    style(ax)
    ax.axvspan(ONSET, TURN_END, color=GRID, alpha=0.8, linewidth=0, zorder=0)
    ax.annotate("5 deg/s turn", xy=((ONSET + TURN_END) / 2, 1.0),
                xycoords=("data", "axes fraction"), xytext=(0, -5), textcoords="offset points",
                ha="center", va="top", fontsize=8, color=INK_MUTED)  # fmt: skip
    centers = starts + BIN_WIDTH / 2
    for arm, values in data.items():
        color = ARM_COLORS[arm]
        mean = np.array([v.mean for v in values])
        ax.fill_between(centers, [v.lower for v in values], [v.upper for v in values],
                        color=color, alpha=0.18, linewidth=0, zorder=2)  # fmt: skip
        ax.plot(centers, mean, color=color, linewidth=2, marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3,
                label=display_name(arm, high_q))  # fmt: skip
    ax.set_ylim(bottom=-0.012)  # room for the markers of rates of zero
    ax.set_yticks([tick for tick in ax.get_yticks() if tick >= 0])
    ax.set_xlim(starts[0], starts[-1] + BIN_WIDTH)
    ax.set_xlabel("Time [s]  (5 s bins, plotted at the bin center)")
    ax.set_ylabel("Missed rate  (share of in-view target steps)")
    legend = ax.legend(loc="upper right", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(INK)
    fig.subplots_adjust(left=0.115, right=0.97, top=0.89, bottom=0.2)
    return finish(
        fig, path, "Missed rate over time after a 5 deg/s turn (separated layout, clutter 0)",
        f"Source: results/improvement_results.pkl, {seeds}, match distance 50 m. Bands: 95% t "
        "interval; last 1-step bin omitted.",
    )  # fmt: skip


def make_figures(
    results: dict[str, list[BlockResult]], winnable: Sequence[str], out_dir: Path, high_q: float
) -> list[Path]:
    """Draw the three figures into out_dir and return their paths.

    Raises:
        SystemExit: If the results lack a group, row, arm or score a figure needs.
    """
    seeds = seeds_text(results)
    starts, drag = drag_series(results)
    return [
        plot_hero(hero_points(results, winnable), out_dir / FILES[0], high_q, seeds),
        plot_bias(bias_points(results), out_dir / FILES[1], seeds),
        plot_drag(starts, drag, out_dir / FILES[2], high_q, seeds),
    ]


def check_lines(results: dict[str, list[BlockResult]], winnable: Sequence[str]) -> list[str]:
    """The plotted means, to compare with before_after_8c.txt (Tables A, B and C)."""
    lines = ["plotted means (compare with results/before_after_8c.txt):"]
    for arm, (cost, gain) in hero_points(results, winnable).items():
        lines.append(
            f"  {arm:<13} neutral RMSE cost {cost.mean:.4g} m (Table B, mean of the blocks), "
            f"missed reduction {gain.mean:.4g} (Table A, minus 'd missed, maneuver')"
        )
    for layout, arms in bias_points(results).items():
        for arm, values in arms.items():
            cells = ", ".join(f"{v.mean:.3g}" for v in values)
            lines.append(f"  {layout}, {arm:<8} RMSE at bias 0 / 0.5 / 1 deg: {cells} (Table C)")
    return lines


def main() -> None:
    import pickle

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pickle", type=Path, default=ROOT / "results" / "improvement_results.pkl")
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "figures")
    args = parser.parse_args()
    if FROZEN is None:
        sys.exit("improvement_experiment.FROZEN is not set: there are no frozen parameters")
    with args.pickle.open("rb") as handle:
        results = pickle.load(handle)
    for line in check_lines(results, FROZEN.winnable):
        print(line)
    for path in make_figures(results, FROZEN.winnable, args.out, FROZEN.ekf_high_accel_std):
        print(f"wrote {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
