"""Tests of scripts/make_readme_figures.py on a small synthetic result (no evaluation is run).

The synthetic scores are the same seed noise (multiples of 1/1024) plus an offset per arm and
metric, so every per-seed paired difference is exactly constant and a figure value can be computed
by hand. Each test docstring names the defect that makes it fail.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from fusion.improvement_experiment import BlockResult, RowResult

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "make_readme_figures.py"
SEEDS = 6
NOISE = np.arange(SEEDS) / 1024.0
ARMS = ("EKF", "EKF high-Q", "EKF high-Q 3", "EKF high-Q 5", "IMM-A", "IMM-B", "EKF+bias")
N_BINS = 12
METRICS = (
    "run_position_rmse", "window_missed_rate", *(f"missed_bin_{i:02d}" for i in range(N_BINS))
)  # fmt: skip
WINNABLE = ("separated | 0 | turn 5 deg/s", "crossing | 0 | turn 5 deg/s")
HIGH_Q = 1.0


def load_script():
    spec = importlib.util.spec_from_file_location("make_readme_figures", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


figures = load_script()


def make_row(label, group, offsets, arms=ARMS) -> RowResult:
    """Arm a has, for each metric, the common seed noise plus offsets[a][metric]."""
    metrics = {
        metric: np.array([NOISE + offsets.get(arm, {}).get(metric, 0.0) for arm in arms])
        for metric in METRICS
    }
    return RowResult(label, group, tuple(arms), tuple(range(SEEDS)), metrics)


def per_arm(metric: str, **values: float) -> dict:
    """Offsets of one metric by arm; keys are q1, q3, q5, a, b, ekf, bias."""
    names = {"ekf": "EKF", "q1": "EKF high-Q", "q3": "EKF high-Q 3", "q5": "EKF high-Q 5",
             "a": "IMM-A", "b": "IMM-B", "bias": "EKF+bias"}  # fmt: skip
    return {names[key]: {metric: value} for key, value in values.items()}


def merge(*parts: dict) -> dict:
    out: dict = {}
    for part in parts:
        for arm, metrics in part.items():
            out.setdefault(arm, {}).update(metrics)
    return out


def maneuver_block(layout: str, scale: float, arms=ARMS) -> BlockResult:
    """A neutral row with RMSE costs (scale times a base), a winnable turn row, a row that is
    not winnable, and bins in which only the baseline misses (growing with the bin)."""
    neutral = merge(
        per_arm("run_position_rmse", q1=0.25 * scale, q3=1.0 * scale, q5=2.0 * scale,
                a=0.125 * scale, b=0.5 * scale),
        per_arm("window_missed_rate", ekf=7.0, q1=7.0, q3=7.0, q5=7.0, a=0.0, b=0.0),
    )  # fmt: skip
    turn = per_arm("window_missed_rate", ekf=0.5, q1=0.4375, q3=0.25, q5=0.125, a=0.375, b=0.0)
    not_winnable = per_arm("window_missed_rate", ekf=0.0, q1=4.0, q3=4.0, q5=4.0, a=4.0, b=4.0)
    bins = merge(*(per_arm(f"missed_bin_{i:02d}", ekf=i / 16.0) for i in range(N_BINS)))
    rows = [
        make_row("neutral", "maneuver", neutral, arms),
        make_row("turn 5 deg/s", "maneuver", merge(turn, bins), arms),
        make_row("turn 99 deg/s", "maneuver", not_winnable, arms),
    ]
    return BlockResult(f"maneuver, {layout}, clutter 0", layout, 0.0, rows)


def bias_block(layout: str) -> BlockResult:
    arms = ("EKF", "EKF+bias")
    rows = []
    for label, before, after in (("bias 0 deg", 2.0, 2.0), ("bias 0.5 deg", 14.0, 12.0),
                                 ("bias 1 deg", 27.0, 16.0)):  # fmt: skip
        offsets = merge(
            per_arm("run_position_rmse", ekf=before), per_arm("run_position_rmse", bias=after)
        )
        rows.append(make_row(label, "bias", offsets, arms))
    return BlockResult(f"bias, {layout}, clutter 0", layout, 0.0, rows)


@pytest.fixture(scope="module")
def results() -> dict:
    return {
        "maneuver": [maneuver_block("separated", 1.0), maneuver_block("crossing", 2.0)],
        "bias": [bias_block("separated"), bias_block("crossing")],
    }


def test_hero_cost_is_the_block_mean_of_the_neutral_rmse_difference(results):
    """Fails if the cost is taken from one block, from a maneuver row, or not against the EKF."""
    points = figures.hero_points(results, WINNABLE)
    expected = {"EKF high-Q": 0.375, "EKF high-Q 3": 1.5, "EKF high-Q 5": 3.0,
                "IMM-A": 0.1875, "IMM-B": 0.75}  # fmt: skip
    for arm, cost in expected.items():
        interval = points[arm][0]
        assert interval.mean == cost
        assert interval.lower == interval.upper == cost  # paired: the seed noise cancels


def test_hero_gain_is_the_missed_rate_reduction_over_the_winnable_rows_only(results):
    """Fails if the neutral or the not-winnable row leaks into the average, or if the sign of
    the reduction is flipped (a positive value must mean fewer misses than the EKF)."""
    points = figures.hero_points(results, WINNABLE)
    expected = {"EKF high-Q": 0.0625, "EKF high-Q 3": 0.25, "EKF high-Q 5": 0.375,
                "IMM-A": 0.125, "IMM-B": 0.5}  # fmt: skip
    for arm, gain in expected.items():
        assert points[arm][1].mean == gain


def test_hero_refuses_missing_arms_rows_and_winnable_blocks(results):
    """Fails if a figure is drawn from results that lack what it plots."""
    arms = tuple(arm for arm in ARMS if arm != "IMM-B")
    without_imm = {"maneuver": [maneuver_block("separated", 1.0, arms)]}
    with pytest.raises(SystemExit):
        figures.hero_points(without_imm, WINNABLE)
    with pytest.raises(SystemExit):
        figures.hero_points(results, ("separated | 0 | no such row",))
    with pytest.raises(SystemExit):
        figures.hero_points({"bias": results["bias"]}, WINNABLE)


def test_bias_points_follow_the_rows_and_layouts_in_order(results):
    """Fails if the rows are out of order, an arm is mixed up or a layout is dropped."""
    points = figures.bias_points(results)
    assert list(points) == ["separated", "crossing"]
    mean_noise = NOISE.mean()
    for layout in points:
        ekf = [i.mean for i in points[layout]["EKF"]]
        corrected = [i.mean for i in points[layout]["EKF+bias"]]
        assert ekf == pytest.approx(np.array([2.0, 14.0, 27.0]) + mean_noise)
        assert corrected == pytest.approx(np.array([2.0, 12.0, 16.0]) + mean_noise)


def test_drag_series_has_eleven_five_second_bins_and_leaves_the_last_one_out(results):
    """Fails if the one-step last bin is plotted, the bins start anywhere but the burn-in, or the
    arms or the row are not the ones of the figure."""
    starts, series = figures.drag_series(results)
    assert starts.tolist() == [5.0 + 5.0 * i for i in range(11)]
    assert list(series) == ["EKF", "EKF high-Q", "IMM-B"]
    ekf = [i.mean for i in series["EKF"]]
    assert ekf == pytest.approx(np.arange(11) / 16.0 + NOISE.mean())
    assert [i.mean for i in series["IMM-B"]] == pytest.approx([NOISE.mean()] * 11)


def test_make_figures_writes_three_png_files(results, tmp_path):
    """Fails if a figure is missing, empty or not a PNG."""
    paths = figures.make_figures(results, WINNABLE, tmp_path / "figs", HIGH_Q)
    assert [p.name for p in paths] == ["benefit_cost.png", "camera_bias.png", "drag_tail.png"]
    for path in paths:
        data = path.read_bytes()
        assert data[:8] == b"\x89PNG\r\n\x1a\n"
        assert len(data) > 5_000


def test_the_check_lines_list_every_hero_arm_and_every_bias_layout(results):
    """Fails if the printed cross-check misses an arm or a layout."""
    text = "\n".join(figures.check_lines(results, WINNABLE))
    for arm in figures.HERO_ARMS:
        assert arm in text
    for layout in figures.BIAS_LAYOUTS:
        assert f"{layout}, EKF " in text and f"{layout}, EKF+bias" in text


def test_values_lines_list_the_reference_and_every_hero_arm_with_hand_computed_cells(results):
    """Fails if a plotted value is missing from the text, shows another statistic than the
    plotted one, or the block and row counts of the header are wrong."""
    lines = figures.values_lines(results, WINNABLE, HIGH_Q)
    text = "\n".join(lines)
    assert "mean of 2 maneuver blocks" in text and "mean of 2 winnable maneuver" in text
    by_name = {line[:16].strip(): line[16:].split("  ") for line in lines[8:]}
    cells = {name: [c.strip() for c in parts if c.strip()] for name, parts in by_name.items()}
    assert cells["EKF (reference)"] == ["0 +- 0", "0 +- 0"]
    assert cells["EKF high-Q 1"] == ["0.375 +- 0", "0.0625 +- 0"]
    assert cells["EKF high-Q 3"] == ["1.5 +- 0", "0.25 +- 0"]
    assert cells["IMM-A"] == ["0.188 +- 0", "0.125 +- 0"]
    assert cells["IMM-B"] == ["0.75 +- 0", "0.5 +- 0"]


def test_write_values_puts_the_provenance_header_first(results, tmp_path):
    """Fails if the values file lacks the command, git HEAD or date lines, or if they do not come
    before the table."""
    path = figures.write_values(tmp_path / "out" / "values.txt", results, WINNABLE, HIGH_Q)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("# command: python ")
    assert lines[1].startswith("# git HEAD: ")
    assert lines[2].startswith("# date (UTC): ")
    assert lines[3] == ""
    assert lines[4].startswith("HERO FIGURE VALUES")
