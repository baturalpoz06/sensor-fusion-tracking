"""Smoke tests of the outage experiment script (tiny runs).

Each test docstring names the condition that makes it fail without the feature.
"""

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_outage_experiment.py"
TINY = [
    "--seeds", "2", "--duration", "30", "--burn-in", "2", "--outage-start", "10",
    "--after-window", "10", "--workers", "1", "--no-plots",
    "--durations", "0", "4", "--flicker-span", "10", "--flicker-off", "1", "2",
    "--burst-off", "0.1", "0.3", "--coast-times", "5", "30", "--coast-durations", "4", "12",
    "--vanish-outage", "12", "--realistic-durations", "4",
]  # fmt: skip


def run_script(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), *args], capture_output=True, text=True, check=False
    )
    if check:
        assert result.returncode == 0, result.stderr
    return result


@pytest.fixture(scope="module")
def all_experiments() -> str:
    return run_script(*TINY, "--experiments", "a", "b", "c", "d", "e", "f", "g").stdout


def test_every_experiment_prints_its_tables_for_every_window(all_experiments):
    """Fails if an experiment is skipped, or a window or the outage table is not printed."""
    out = all_experiments
    for header in (
        "experiment a: radar outage",
        "experiment b: camera outage",
        "experiment c: total blackout",
        "experiment d-periodic: periodic radar flicker, period 10 s",
        "experiment d-burst: random radar bursts, mean burst 2 s",
        "experiment e: aware tracker, max coasting time against the blackout duration",
        "experiment f: target 0 ceases to exist 2 s into a 12 s radar outage",
        "experiment g: clutter rate 5 per scan and sensor",
    ):
        assert header in out
    lines = out.splitlines()
    for heading in ("window: before", "window: during", "window: after", "outage scores"):
        assert lines.count(heading) == 8, heading  # one of each per experiment
    assert "scenario 'vanishing'" in out and "scenario 'crossing'" in out


def test_rows_are_labelled_with_the_policy_and_the_swept_value(all_experiments):
    """Fails if rows lose their policy or value, or an experiment runs the wrong grid."""
    lines = [line.strip() for line in all_experiments.splitlines()]
    for label in (
        "unaware 4 s",
        "aware 4 s",
        "unaware 2 s off",
        "aware 30% off",
        "coast 5 s, 12 s",
        "aware, coast 30 s",
        "radar 4 s, aware freeze",
        "blackout 4 s, unaware",
    ):
        assert any(line.startswith(label) for line in lines), label
    camera = all_experiments.split("experiment b:")[1].split("experiment c:")[0]
    assert "aware 4 s" not in camera.replace("unaware 4 s", "")  # camera outages: unaware only


def test_the_legend_and_the_settings_caption_are_printed(all_experiments):
    """Fails if the table cannot be read without the source: the legend or settings are missing."""
    assert "windows (half-open, in steps)" in all_experiments
    assert "ideal = 4" in all_experiments
    assert "after window 10 s" in all_experiments and "max coast 15 s" in all_experiments


def test_plots_are_written_per_experiment_and_policy(tmp_path):
    """Fails if no figure is saved, or the two policies of an experiment share one figure."""
    pytest.importorskip("matplotlib")
    args = [a for a in TINY if a != "--no-plots"]
    out = run_script(*args, "--experiments", "a", "--out", str(tmp_path)).stdout
    for policy in ("unaware", "aware"):
        image = tmp_path / f"outage_crossing_a_{policy}.png"
        assert image.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        assert f"saved {image}" in out


def test_an_outage_that_runs_to_the_end_of_the_run_is_refused():
    """Fails if a configuration without an after window is silently run."""
    result = run_script(*TINY, "--experiments", "a", "--durations", "25", check=False)
    assert result.returncode != 0 and "must end before the run does" in result.stderr
