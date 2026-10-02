"""Smoke tests of the multi-target experiment script (a tiny run of each mode)."""

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_mtt_experiment.py"
TINY = ["--seeds", "2", "--duration", "8", "--burn-in", "2", "--workers", "1"]


def run_script(*args: str) -> str:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), *args], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_clutter_sweep_prints_the_report_the_table_and_the_shared_settings():
    out = run_script(*TINY, "--sweeps", "clutter", "--clutter-rates", "0", "5", "--no-plots")
    assert "Expected radar clutter points inside a tentative track's gate" in out
    assert "=== sweep of clutter_rate over [0.0, 5.0] (2 seeds each) ===" in out
    assert "match distance 50 m (same at every row)" in out
    for column in ("rmse [m]", "ghost/step", "false/step", "missed", "id sw", "med rmse [m]"):
        assert column in out


def test_the_default_clutter_grid_depends_on_the_swept_sensor():
    radar = run_script(*TINY, "--sweeps", "clutter", "--sweep-sensors", "radar", "--no-plots")
    camera = run_script(*TINY, "--sweeps", "clutter", "--sweep-sensors", "camera", "--no-plots")
    assert "[0.0, 3.0, 5.0, 7.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0]" in radar
    assert "radar_clutter_rate" in radar
    assert "[0.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0]" in camera
    assert "camera_clutter_rate" in camera


def test_compare_mode_prints_both_trackers_with_the_given_match_distance():
    out = run_script(*TINY, "--sweeps", "compare", "--match-distance", "150", "--no-plots")
    assert "radar-only vs fused tracker" in out
    assert "match distance 150 m (same at every row)" in out
    rows = [
        line.split()[0]
        for line in out.splitlines()
        if line.strip().startswith(("radar-only", "fused"))
    ]
    assert rows == ["radar-only", "fused"]


def test_plots_are_written_to_the_output_directory(tmp_path):
    pytest.importorskip("matplotlib")
    out = run_script(
        *TINY, "--sweeps", "pd", "--pds", "0.8", "1.0", "--out", str(tmp_path / "plots")
    )
    image = tmp_path / "plots" / "mtt_crossing_pd.png"
    assert image.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert "saved" in out
