"""Smoke tests of the robustness experiment script (tiny runs).

Each test docstring names the condition that makes it fail without the feature. These runs only
check that the pieces fit together; they say nothing about the tracker.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_robustness_experiment.py"
TINY = [
    "--seeds", "2", "--duration", "20", "--burn-in", "2", "--workers", "1", "--no-plots",
    "--layouts", "separated", "--clutter-rates", "0",
]  # fmt: skip
GRIDS = {
    "noise": ["--noise-scales", "0.5", "1", "2"],
    "maneuver": ["--turn-rates", "10", "--accelerations", "2", "--random-rates", "10"],
    "camera": ["--biases", "0.5", "--offsets-ms", "50"],
    "lever-arm": ["--lever-distances", "5", "20", "--lever-angles", "0"],
    "geometry": ["--geometry-counts", "1", "2", "--geometry-clutter", "2"],
}


def run_script(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), *args], capture_output=True, text=True, check=False
    )
    if check:
        assert result.returncode == 0, result.stderr
    return result


def without_timing(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not re.fullmatch(r"\(\d+ s\)", line))


@pytest.fixture(scope="module")
def outputs(tmp_path_factory) -> dict[str, str]:
    out = tmp_path_factory.mktemp("robustness")
    return {
        scenario: run_script(*TINY, *grids, "--scenario", scenario, "--out", str(out)).stdout
        for scenario, grids in GRIDS.items()
    }


def test_every_scenario_prints_its_rows_and_all_its_tables(outputs):
    """Fails if a scenario is skipped, a row is mislabelled, or a table is not printed."""
    expected_rows = {
        "noise": ("neutral", "radar_range x0.5", "process_noise x2", "camera_bearing x2"),
        "maneuver": ("turn 10 deg/s", "acceleration 2 m/s^2", "random turns <= 10 deg/s"),
        "camera": ("bias 0.5 deg", "offset 50 ms"),
        "lever-arm": ("unknown 5 m @0 deg", "unknown 20 m @0 deg", "known 20 m @0 deg"),
        "geometry": ("inbound near", "crossing far, unknown 20 m arm", "tangential mid, clutter 2"),
    }
    for scenario, out in outputs.items():
        assert f"=== scenario {scenario}:" in out
        lines = [line.strip() for line in out.splitlines()]
        for label in expected_rows[scenario]:
            assert any(line.startswith(label) for line in lines), (scenario, label)
        for heading in (
            "scores (Phase 6)",
            "error and camera diagnostics",
            "paired differences, row - neutral",
        ):
            assert lines.count(heading) == 1, (scenario, heading)


def test_the_row_counts_and_trial_counts_are_announced(outputs):
    """Fails if the size of a run is not stated before it starts."""
    assert "1 table set(s), 9 rows, 2 seeds each = 18 trials" in outputs["noise"]
    assert "1 table set(s), 4 rows, 2 seeds each = 8 trials" in outputs["maneuver"]
    assert "1 table set(s), 5 rows, 2 seeds each = 10 trials" in outputs["lever-arm"]
    assert "1 table set(s), 30 rows, 2 seeds each = 60 trials" in outputs["geometry"]


def test_only_the_maneuver_scenario_prints_window_scores_and_the_crossing_caveat(outputs, tmp_path):
    """Fails if window scores appear without a focus window, or a crossing maneuver run has no
    caveat about the lost crossing."""
    assert "focus window scores" in outputs["maneuver"]
    for scenario in ("noise", "camera", "lever-arm", "geometry"):
        assert "focus window scores" not in outputs[scenario]
    out = run_script(
        *TINY, *GRIDS["maneuver"], "--scenario", "maneuver", "--layouts", "crossing", "separated",
        "--out", str(tmp_path),
    ).stdout  # fmt: skip
    assert out.count("targets 0 and 1 no longer cross") == 1
    assert "scenario crossing:" in out and "scenario separated:" in out


def test_the_legend_and_the_settings_caption_are_printed(outputs):
    """Fails if the tables cannot be read without the source: legend or settings are missing."""
    out = outputs["noise"]
    assert "rmse is taken over matched" in out and "not adjusted for the many" in out
    assert "match distance 50 m" in out and "diagnostics within 200 m" in out


def test_the_tables_are_also_written_to_a_text_file(tmp_path):
    """Fails if the results of a scenario are only on the screen."""
    stdout = run_script(*TINY, *GRIDS["camera"], "--scenario", "camera", "--out", str(tmp_path))
    text = (tmp_path / "robustness_camera.txt").read_text(encoding="utf-8")
    assert "=== scenario camera:" in text and "bias 0.5 deg" in text
    assert text.strip() in stdout.stdout


def test_a_pilot_is_marked_runs_five_seeds_and_writes_into_its_own_folder(tmp_path):
    """Fails if a pilot can be mistaken for the full run: banner, seed count and folder."""
    seeds_at = TINY.index("--seeds")
    args = TINY[:seeds_at] + TINY[seeds_at + 2 :]
    out = run_script(
        *args, *GRIDS["camera"], "--scenario", "camera", "--pilot", "--out", str(tmp_path)
    )
    assert out.stdout.count("PILOT: a does-it-run check") == 3  # start, scenario, end
    assert "3 rows, 5 seeds each = 15 trials" in out.stdout
    assert (tmp_path / "pilot" / "robustness_camera.txt").exists()
    assert not (tmp_path / "robustness_camera.txt").exists()


def test_plots_are_written_per_group_with_the_neutral_row_as_anchor(tmp_path):
    """Fails if no figure is saved, or a group is plotted without its neutral reference."""
    pytest.importorskip("matplotlib")
    args = [a for a in TINY if a != "--no-plots"]
    out = run_script(*args, *GRIDS["noise"], "--scenario", "noise", "--out", str(tmp_path)).stdout
    image = tmp_path / "robustness_noise_noise_separated_clutter_0_radar_range.png"
    assert image.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert f"saved {image}" in out
    assert len(list(tmp_path.glob("robustness_noise_*.png"))) == 4  # one per noise knob


def test_a_run_is_deterministic_and_does_not_depend_on_the_worker_count(tmp_path):
    """Fails if the output depends on the process pool or differs between identical runs."""
    args = [*TINY, *GRIDS["camera"], "--scenario", "camera", "--out", str(tmp_path)]
    first = without_timing(run_script(*args).stdout)
    assert first == without_timing(run_script(*args).stdout)
    parallel = [a if a != "1" else "2" for a in args]  # --workers 1 -> 2
    assert "--workers" in parallel and first == without_timing(run_script(*parallel).stdout)


def test_a_maneuver_window_beyond_the_run_is_refused(tmp_path):
    """Fails if a configuration whose focus window cannot be scored is silently run."""
    args = [a if a != "20" else "10" for a in TINY]  # duration 20 -> 10
    result = run_script(
        *args, *GRIDS["maneuver"], "--scenario", "maneuver", "--out", str(tmp_path), check=False
    )
    assert result.returncode != 0 and "focus_window" in result.stderr
