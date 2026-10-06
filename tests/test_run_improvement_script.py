"""Tests of the improvement experiment script and its report (tiny runs).

Each test docstring names the condition that makes it fail. These runs only check that the
pieces fit together; they say nothing about the tracker. No test starts the real evaluation:
the refusal rules are tested through their function and through argument errors only.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from fusion.improvement_experiment import (
    BASELINE,
    UNFROZEN_DEFAULTS,
    Block,
    ImprovementRow,
    Parameters,
    named_arms,
    run_blocks,
)
from fusion.improvement_report import flatten_block, paired_table, parameters_text
from fusion.mtt_experiment import SCENARIOS
from fusion.robustness_experiment import RobustnessConfig

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_improvement_experiment.py"
TINY = [
    "--duration", "20", "--burn-in", "2", "--layouts", "separated", "--clutter-rates", "0",
    "--workers", "2",
]  # fmt: skip


def run_script(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), *args], capture_output=True, text=True, check=False
    )
    if check:
        assert result.returncode == 0, result.stderr
    return result


def load_script():
    spec = importlib.util.spec_from_file_location("run_improvement_experiment", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def pilot(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("pilot")
    run_script("--stage", "eval", "--pilot", "--seeds", "2", *TINY, "--out", str(out))
    return out / "pilot"


def test_the_pilot_writes_every_table_the_headline_and_the_verdicts(pilot):
    """Fails if a group, the headline or the criteria file is not written, or a banner is lost."""
    for name in (
        "improvement_maneuver.txt",
        "improvement_bias.txt",
        "improvement_hard.txt",
        "summary_8b_headline.txt",
        "criteria_8b.txt",
    ):
        text = (pilot / name).read_text(encoding="utf-8")
        assert "PILOT: a does-it-run check" in text, name
        assert "UNFROZEN" in text, name
    maneuver = (pilot / "improvement_maneuver.txt").read_text(encoding="utf-8")
    for heading in (
        "scores (Phase 6)",
        "error and camera diagnostics",
        "focus window scores",
        "paired differences, arm - EKF",
        "missed and drag rate per time bin",
        "IMM mode probabilities",
    ):
        assert heading in maneuver, heading
    assert "bias estimate" in (pilot / "improvement_bias.txt").read_text(encoding="utf-8")


def test_the_pilot_headline_lists_every_arm_of_every_row_and_only_numbers(pilot):
    """Fails if an arm or row is missing from the headline, or interpretation words slip in."""
    text = (pilot / "summary_8b_headline.txt").read_text(encoding="utf-8")
    lines = text.splitlines()
    for label in (
        "neutral | EKF [abs]",
        "turn 10 deg/s | IMM-B",
        "random turns <= 20 deg/s | EKF+bias",
        "bias 1 deg | EKF+oracle",
        "bias 0.5 deg | EKF+always",
        "turn 10 deg/s + bias | IMM-A+bias",
        "random turns <= 10 deg/s + bias | IMM-B+bias",
    ):
        assert any(line.strip().startswith(label) for line in lines), label
    for word in ("better", "worse", "improves the", "because", "therefore", "suggests"):
        assert word not in text.lower().replace("improves", "")


def test_the_criteria_file_has_a_verdict_per_pre_registered_criterion(pilot):
    """Fails if a criterion is missing from the mechanical verdicts."""
    text = (pilot / "criteria_8b.txt").read_text(encoding="utf-8")
    for name in (
        "A1 IMM-A (maneuver rows)",
        "A1 IMM-B (held-out rows)",
        "A2 IMM-A (neutral non-inferiority)",
        "A2 IMM-B (neutral non-inferiority)",
        "H-drag",
        "B1 EKF+bias",
        "B2 EKF+bias neutral equivalence",
        "B3 EKF+bias maneuvers are not bias",
        "Hard cases: IMM+bias",
        "Runtime per trial, maneuver (reported)",
    ):
        assert f"== {name}" in text, name
    assert text.rstrip().splitlines()[-1].startswith("verdicts: ")


def test_the_tuning_stage_applies_the_selection_and_prints_the_frozen_expression(tmp_path):
    """Fails if the tuning run does not select per family, or the frozen expression is lost."""
    out = run_script(
        "--stage", "tune", "--small-grid", "--seeds", "2", *TINY, "--out", str(tmp_path)
    )
    text = (tmp_path / "improvement_tune.txt").read_text(encoding="utf-8")
    for family in ("EKF high-Q", "IMM-A", "IMM-B", "bias prior"):
        assert f"== {family}: selected" in text
    assert "== winnable maneuver row-blocks" in text
    assert "projected 95% half widths" in text
    assert "bias process noise sensitivity" in text
    assert "FROZEN = Parameters(" in text and "FROZEN = Parameters(" in out.stdout
    assert (tmp_path / "improvement_tune.pkl").exists()


def test_the_timing_stage_reports_every_arm_relative_to_the_baseline(tmp_path):
    """Fails if an arm is left out of the timing table."""
    run_script("--stage", "timing", "--seeds", "1", *TINY, "--out", str(tmp_path))
    text = (tmp_path / "improvement_timing.txt").read_text(encoding="utf-8")
    for arm in named_arms(UNFROZEN_DEFAULTS):
        assert f"neutral | {arm}" in text and f"turn 10 deg/s | {arm}" in text


@pytest.mark.parametrize(
    "extra", [["--seeds", "3"], ["--duration", "20"], ["--layouts", "separated"],
              ["--clutter-rates", "0"], ["--burn-in", "2"]]
)  # fmt: skip
def test_the_real_evaluation_takes_no_overrides(extra):
    """Fails if the evaluation can be started with fewer seeds, other layouts or a shorter run.

    The argument error comes before anything is run, so this never starts an evaluation.
    """
    result = run_script("--stage", "eval", *extra, check=False)
    assert result.returncode == 2 and "takes no" in result.stderr


def test_the_evaluation_refuses_without_frozen_parameters_a_commit_or_a_clean_tree(monkeypatch):
    """Fails if the evaluation could run with placeholders, an unknown commit or a dirty tree."""
    module = load_script()
    assert "FROZEN is not set" in module.refusal_reason(None)
    frozen = Parameters(2.0, 3.0, 10.0, 1.0, 0.5)
    assert "no commit" in module.refusal_reason(frozen)
    nowhere = Parameters(2.0, 3.0, 10.0, 1.0, 0.5, commit="0" * 40)
    assert "not an ancestor" in module.refusal_reason(nowhere)
    head = module.head_commit()
    assert head
    ours = Parameters(2.0, 3.0, 10.0, 1.0, 0.5, commit=head)

    real = module.git

    def dirty(*arguments):
        if arguments[0] == "status":
            return type("R", (), {"returncode": 0, "stdout": " M src/fusion/x.py\n"})()
        return real(*arguments)

    monkeypatch.setattr(module, "git", dirty)
    assert "uncommitted" in module.refusal_reason(ours)

    def clean(*arguments):
        if arguments[0] == "status":
            return type("R", (), {"returncode": 0, "stdout": ""})()
        return real(*arguments)

    monkeypatch.setattr(module, "git", clean)
    assert module.refusal_reason(ours) is None


def test_the_frozen_expression_round_trips():
    """Fails if the text printed for improvement_experiment.FROZEN does not rebuild the
    parameters (including the winnable labels and the commit)."""
    parameters = Parameters(
        3.0, 2.0, 7.5, 0.5, 0.1, ("separated | 0 | turn 5 deg/s", "crossing | 5 | x"), "abc123"
    )
    assert eval(parameters_text(parameters), {"Parameters": Parameters}) == parameters
    empty = Parameters(3.0, 2.0, 7.5, 0.5, 0.1)
    assert eval(parameters_text(empty), {"Parameters": Parameters}) == empty


def test_the_paired_table_lists_every_arm_but_the_baseline_of_each_row():
    """Fails if a baseline pair is listed as a comparison with itself, or an arm is dropped."""
    config = RobustnessConfig(
        duration=12.0, burn_in=2.0, radar_clutter_rate=0.0, camera_clutter_rate=0.0,
        focus_window=(5.0, 12.0),
    )  # fmt: skip
    arms = tuple(named_arms(UNFROZEN_DEFAULTS)[n] for n in (BASELINE, "IMM-A", "EKF+bias"))
    row = ImprovementRow("row", config, arms, SCENARIOS["separated"], "maneuver")
    block = run_blocks([Block("b", "separated", 0.0, [row])], (2000, 2001))[0]
    result, labels, references = flatten_block(block)
    assert labels == ["row | EKF", "row | IMM-A", "row | EKF+bias"] and references == [0, 0, 0]
    assert result.metrics["run_position_rmse"].shape == (3, 2)
    lines = paired_table(block)
    body = [line for line in lines if "|" in line and "row | arm" not in line]
    assert [line.split("|")[1].split()[0] for line in body] == ["IMM-A", "EKF+bias"]
