"""Tests of scripts/make_before_after.py on a small synthetic result (no evaluation is run).

Each test docstring names the defect that makes it fail. The synthetic scores are multiples of
powers of two, so every paired difference is exactly constant over seeds and prints as
"value +- 0".
"""

import importlib.util
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from fusion.improvement_experiment import BlockResult, RowResult
from fusion.outage_report import format_table

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "make_before_after.py"
SEEDS = 6
ARMS = (
    "EKF", "EKF high-Q", "EKF high-Q 3", "EKF high-Q 5", "IMM-A", "IMM-B", "EKF+bias"
)  # fmt: skip
METRICS = (
    "window_missed_rate", "run_position_rmse", "run_missed_rate", "run_ghost_rate",
    "run_id_switches", "cross_rms",
)  # fmt: skip
WINNABLE = ("separated | 0 | turn 5 deg/s", "separated | 0 | turn 10 deg/s",
            "separated | 0 | held-out turn 7 deg/s")  # fmt: skip


def load_script():
    spec = importlib.util.spec_from_file_location("make_before_after", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_row(label: str, group: str, offsets: dict, arms=ARMS) -> RowResult:
    """A row where arm a has, for each metric, the common seed noise plus offsets[a][metric]."""
    noise = np.arange(SEEDS) / 1024.0
    metrics = {}
    for metric in METRICS:
        metrics[metric] = np.array(
            [noise + offsets.get(arm, {}).get(metric, 0.0) for arm in arms]
        )
    return RowResult(label, group, tuple(arms), tuple(range(SEEDS)), metrics)


def missed(**by_arm: float) -> dict:
    """Window missed offsets by arm; the baseline sits at 0.5 (keys are arm names with _ for
    spaces and - and a leading q for the controls)."""
    names = {"ekf": "EKF", "q1": "EKF high-Q", "q3": "EKF high-Q 3", "q5": "EKF high-Q 5",
             "a": "IMM-A", "b": "IMM-B"}  # fmt: skip
    return {names[k]: {"window_missed_rate": v} for k, v in by_arm.items()}


def maneuver_block(arms=ARMS) -> BlockResult:
    """Rows: neutral (poisoned missed rate), two winnable maneuver rows, one that is not
    winnable (poisoned), the held-out neutral and one winnable held-out row."""
    poisoned = missed(ekf=7.0, q1=7.0, q3=7.0, q5=7.0, a=0.0, b=0.0)
    neutral_offsets = {
        **poisoned,
        "IMM-A": {**poisoned["IMM-A"], "run_position_rmse": 0.125},
        "EKF high-Q 5": {**poisoned["EKF high-Q 5"], "run_position_rmse": 1.0,
                         "run_id_switches": 0.5},
    }  # fmt: skip
    first = missed(ekf=0.5, q1=0.4375, q3=0.375, q5=0.25, a=0.125, b=0.0)
    second = {**first, "IMM-A": {"window_missed_rate": 0.375}}  # d = -0.125 instead of -0.375
    outside = missed(ekf=0.0, q1=4.0, q3=4.0, q5=4.0, a=4.0, b=4.0)
    held = missed(ekf=0.5, q1=0.625, q3=0.4375, q5=0.375, a=0.25, b=0.125)
    rows = [
        make_row("neutral", "maneuver", neutral_offsets, arms),
        make_row("turn 5 deg/s", "maneuver", first, arms),
        make_row("turn 10 deg/s", "maneuver", second, arms),
        make_row("turn 99 deg/s", "maneuver", outside, arms),
        make_row("held-out neutral", "held-out", poisoned, arms),
        make_row("held-out turn 7 deg/s", "held-out", held, arms),
    ]
    return BlockResult("maneuver, separated, clutter 0", "separated", 0.0, rows)


def bias_block() -> BlockResult:
    arms = ("EKF", "EKF+bias")
    rows = []
    for label, before, after in (("bias 0 deg", 2.0, 2.0), ("bias 0.5 deg", 14.0, 12.0),
                                 ("bias 1 deg", 27.0, 16.0)):  # fmt: skip
        offsets = {
            "EKF": {"run_position_rmse": before, "cross_rms": before},
            "EKF+bias": {"run_position_rmse": after, "cross_rms": after},
        }
        rows.append(make_row(label, "bias", offsets, arms))
    return BlockResult("bias, separated, clutter 0", "separated", 0.0, rows)


@pytest.fixture(scope="module")
def report() -> str:
    module = load_script()
    results = {"maneuver": [maneuver_block()], "bias": [bias_block()]}
    return "\n".join(module.build_report(results, WINNABLE, 1.0))


def line_of(report: str, label: str) -> str:
    """The table line whose label column (the first 40 characters) is exactly this label."""
    lines = [line for line in report.splitlines() if line[:40].strip() == label]
    assert len(lines) == 1, (label, lines)
    return lines[0]


def test_the_report_has_the_three_tables_and_no_verdicts(report):
    """Fails if a table is missing, or a pass / fail label slips into a descriptive report."""
    for title in ("TABLE A: benefit", "TABLE B: side effect", "TABLE C: camera bias"):
        assert title in report
    assert "no new run" in report and "Descriptive only: no verdicts" in report
    for word in ("PASS", "FAIL", "UNDERPOWERED"):
        assert word not in report


def test_the_script_formats_with_the_existing_table_helper():
    """Fails if the script formats its tables itself instead of the public format_table."""
    assert load_script().format_table is format_table


def test_the_benefit_table_is_the_paired_difference_averaged_over_winnable_blocks(report):
    """Fails if a non-winnable or neutral row enters the mean, the difference is not paired
    against the EKF, or the held-out and maneuver groups are mixed.

    Maneuver d (mean of the two winnable rows): Q1 -0.0625, Q3 -0.125, Q5 -0.25, IMM-A
    (-0.375 and -0.125) -0.25, IMM-B -0.5. Held-out d: Q1 +0.125, Q3 -0.0625, Q5 -0.125,
    IMM-A -0.25, IMM-B -0.375.
    """
    expected = {
        "EKF": ("0 +- 0", "0 +- 0"),
        "EKF high-Q 1": ("-0.0625 +- 0", "0.125 +- 0"),
        "EKF high-Q 3": ("-0.125 +- 0", "-0.0625 +- 0"),
        "EKF high-Q 5": ("-0.25 +- 0", "-0.125 +- 0"),
        "IMM-A": ("-0.25 +- 0", "-0.25 +- 0"),
        "IMM-B": ("-0.5 +- 0", "-0.375 +- 0"),
    }
    table = report.split("TABLE B")[0]
    for label, (maneuver, held_out) in expected.items():
        line = line_of(table, label)
        assert maneuver in line and held_out in line, (label, line)
    assert "missed, maneuver (2)" in table and "missed, held-out (1)" in table


def test_the_side_effect_table_shows_the_neutral_difference_with_the_a2_margins(report):
    """Fails if the neutral row is not the one shown, the margins are missing or wrong, or a
    difference is attached to the wrong arm."""
    table = report.split("TABLE B")[1].split("TABLE C")[0]
    for margin in ("d rmse [m] (A2 0.3)", "d missed (A2 0.005)", "d ghost (A2 0.01)",
                   "d id sw (A2 0.2)"):  # fmt: skip
        assert margin in table
    q5 = line_of(table, "separated, clutter 0 | EKF high-Q 5")
    assert "1 +- 0" in q5 and "0.5 +- 0" in q5  # rmse +1.0, id switches +0.5
    imm_a = line_of(table, "separated, clutter 0 | IMM-A")
    assert "0.125 +- 0" in imm_a
    q1 = line_of(table, "separated, clutter 0 | EKF high-Q 1")
    assert q1.count("0 +- 0") >= 4
    shown = [x for x in table.splitlines() if x[:40].strip().startswith("separated, clutter 0 |")]
    assert len(shown) == 5


def test_the_bias_table_compares_ekf_and_ekf_bias_at_each_bias(report):
    """Fails if a bias row is missing, or the difference is not EKF+bias minus EKF."""
    table = report.split("TABLE C")[1]
    assert "0 +- 0" in line_of(table, "separated, clutter 0 | bias 0 deg")
    half = line_of(table, "separated, clutter 0 | bias 0.5 deg")
    one = line_of(table, "separated, clutter 0 | bias 1 deg")
    assert "-2 +- 0" in half and "-11 +- 0" in one
    assert one.count("-11 +- 0") == 2  # rmse and cross-range
    assert "d = EKF+bias - EKF" in table


def test_a_missing_control_arm_stops_the_script_with_a_message():
    """Fails if results without the control arms produce a table instead of stopping."""
    module = load_script()
    arms = tuple(a for a in ARMS if a != "EKF high-Q 5")
    results = {"maneuver": [maneuver_block(arms)], "bias": [bias_block()]}
    with pytest.raises(SystemExit, match="lacks the arms"):
        module.build_report(results, WINNABLE, 1.0)


def test_results_without_winnable_rows_or_groups_stop_the_script():
    """Fails if an empty winnable set or a missing group gives empty tables."""
    module = load_script()
    results = {"maneuver": [maneuver_block()], "bias": [bias_block()]}
    with pytest.raises(SystemExit, match="no winnable"):
        module.build_report(results, (), 1.0)
    with pytest.raises(SystemExit, match="no 'bias' group"):
        module.build_report({"maneuver": results["maneuver"]}, WINNABLE, 1.0)


def test_the_script_writes_the_file_from_a_pickle(tmp_path):
    """Fails if the command line does not read the pickle and write all three tables.

    The synthetic rows carry labels that are in the frozen winnable list, so the real frozen
    parameters select them.
    """
    source = tmp_path / "results.pkl"
    results = {"maneuver": [maneuver_block()], "bias": [bias_block()], "hard": []}
    source.write_bytes(pickle.dumps(results))
    out = tmp_path / "sub" / "before_after.txt"
    done = subprocess.run(
        [sys.executable, str(SCRIPT), "--pickle", str(source), "--out", str(out)],
        capture_output=True, text=True, check=False,
    )  # fmt: skip
    assert done.returncode == 0, done.stderr
    text = out.read_text(encoding="utf-8")
    for title in ("TABLE A", "TABLE B", "TABLE C"):
        assert title in text
    assert "EKF high-Q 1" in text  # the frozen high-Q process noise is 1 m/s^2
