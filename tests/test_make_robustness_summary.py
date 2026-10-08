"""Tests of scripts/make_robustness_summary.py.

The parser is checked on hand-built rows; the whole script on the tables of one tiny robustness
run (2 seeds, 20 s), whose grids contain every row the prediction checks read. These runs say
nothing about the tracker. Each test docstring names the defect that makes it fail.
"""

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

from fusion.mtt_report import COLUMN_WIDTH, MEDIAN_WIDTH, VALID_WIDTH

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "make_robustness_summary.py"
EXPERIMENT = ROOT / "scripts" / "run_robustness_experiment.py"
TINY = [
    "--scenario", "all", "--seeds", "2", "--duration", "20", "--burn-in", "2", "--workers", "2",
    "--no-plots", "--layouts", "separated", "--clutter-rates", "0", "5",
    "--noise-scales", "0.25", "1", "4",
    "--turn-rates", "5", "20", "--accelerations", "2", "--random-rates", "10",
    "--biases", "0.5", "--offsets-ms", "50",
    "--lever-distances", "20", "--lever-angles", "0",
    "--geometry-counts", "1", "--geometry-clutter", "2",
]  # fmt: skip
HEADER_LINES = 4  # the provenance header, including its blank line


def load_script():
    spec = importlib.util.spec_from_file_location("make_robustness_summary", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


summary = load_script()


def row(label: str, cells: list[str], meds: list[str], med_width: int, valid: str) -> str:
    """A table row composed the way fusion.outage_report.format_table composes it."""
    return (
        f"{label:>40}"
        + "".join(f"{c:>{COLUMN_WIDTH}}" for c in cells)
        + "".join(f"{m:>{med_width}}" for m in meds)
        + f"{valid:>{VALID_WIDTH}}"
    )


def test_parse_row_cuts_label_cells_and_medians_at_the_column_widths():
    """Fails if a cell boundary is off by one (cells would be merged or truncated)."""
    cells = ["2.86 +- 0.11", "+0.0386 +- 0.012", "n/a"]
    meds = ["2.78", "0"]
    line = row("radar_range x0.25", cells, meds, MEDIAN_WIDTH, "50/50")
    assert summary.parse_row(line, 3, 2, MEDIAN_WIDTH) == ("radar_range x0.25", cells, meds)


def test_parse_row_keeps_a_label_longer_than_the_first_column():
    """Fails if the label is cut at a fixed width instead of taken as all text before the cells."""
    label = "tangential mid, 8 targets in a sector, unknown 20 m arm"
    line = row(label, ["1 +- 0"], ["1"], 18, "0/0/0/0/0")
    assert summary.parse_row(line, 1, 1, 18)[0] == label


def test_parse_tables_reads_blocks_and_stops_a_table_at_the_blank_line():
    """Fails if rows of the next block or the 'saved' lines are taken as table rows."""
    cols = len(summary.TABLES["scores (Phase 6)"][1])
    meds = len(summary.TABLES["scores (Phase 6)"][2])
    body = row("neutral", ["1 +- 0"] * cols, ["1"] * meds, MEDIAN_WIDTH, "2/2")
    text = "\n".join(
        [
            "--- noise, separated, clutter 0 ---",
            "",
            "scores (Phase 6)",
            "caption line",
            "header line",
            "-" * 40,
            body,
            "saved results/x.png",
            "--- noise, separated, clutter 5 ---",
            "scores (Phase 6)",
            "-" * 40,
            body,
            "",
        ]
    )
    blocks = summary.parse_tables(text)
    assert list(blocks) == ["noise, separated, clutter 0", "noise, separated, clutter 5"]
    for tables in blocks.values():
        assert [label for label, _, _ in tables["scores"]] == ["neutral"]


def test_short_labels_abbreviate_geometry_rows_and_refuse_overlong_ones():
    """Fails if an abbreviation is lost or an overlong label silently breaks the alignment."""
    label = "tangential mid, clutter 2, unknown 20 m arm"
    assert summary.short_label(label) == "tan mid clutter 2 +arm"
    with pytest.raises(SystemExit):
        summary.short_label("x" * 40)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory) -> dict:
    """Tables of a tiny robustness run, and the summary written from them."""
    results = tmp_path_factory.mktemp("robustness")
    run = subprocess.run(
        [sys.executable, str(EXPERIMENT), *TINY, "--out", str(results)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    out = results / "summary.txt"
    made = subprocess.run(
        [sys.executable, str(SCRIPT), "--results", str(results), "--out", str(out)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert made.returncode == 0, made.stderr
    lines = out.read_text(encoding="utf-8").splitlines()
    return {"results": results, "lines": lines, "stdout": made.stdout}


def test_the_file_starts_with_the_provenance_header(tiny):
    """Fails if the command, commit or date line is missing from the top of the file."""
    lines = tiny["lines"]
    assert lines[0].startswith("# command: python ") and "make_robustness_summary.py" in lines[0]
    assert lines[1].startswith("# git HEAD: ")
    assert lines[2].startswith("# date (UTC): ")
    assert lines[3] == ""
    assert lines[HEADER_LINES].startswith(f"HEADLINE NUMBERS from {tiny['results'].name}/")
    assert "(2 seeds, 60 s, burn-in 5 s)" in lines[HEADER_LINES]


def test_every_block_is_listed_with_its_neutral_row_as_absolute_values(tiny):
    """Fails if a block of the input is dropped or its reference row is shown as a difference."""
    lines = tiny["lines"]
    blocks = [line[3:] for line in lines if line.startswith("== ")]
    for scenario in ("noise", "maneuver", "camera", "lever-arm"):
        for clutter in ("0", "5"):
            assert f"{scenario}, separated, clutter {clutter}" in blocks
    assert "geometry, base clutter 0" in blocks
    neutral = [line for line in lines if line.startswith("neutral [abs]")]
    assert len(neutral) == 8


def test_difference_cells_are_copied_from_the_paired_table(tiny):
    """Fails if a d cell is taken from another column, another row or another block."""
    text = (tiny["results"] / "robustness_noise.txt").read_text(encoding="utf-8")
    paired = summary.index(summary.parse_tables(text)["noise, separated, clutter 5"]["paired"])
    lines = tiny["lines"]
    start = lines.index("== noise, separated, clutter 5")
    line = next(x for x in lines[start:] if x.startswith("camera_bearing x0.25 "))
    expected = [paired["camera_bearing x0.25"][0][c] for c in summary.PAIRED_METRICS]
    # a cell is "mean +- half" (or "n/a"); a long one overflows its 15 characters
    cells = re.findall(r"\S+ \+- \S+|n/a", line[len("camera_bearing x0.25") :])
    assert cells[:5] == expected


def test_prediction_checks_are_present_with_the_labelled_focus_window(tiny):
    """Fails if a prediction check is dropped or its focus window label is not the option's."""
    lines = tiny["lines"]
    assert "PREDICTION CHECKS (plain extractions; mean +- 95% half-width, med = median)" in lines
    assert any(x.startswith("   radar_range     k=0.25  ghost ") for x in lines)
    assert any(x.startswith("   lambda=5 omega=20  whole run ") for x in lines)
    assert any("focus window [15, 35) s" in x for x in lines)
    assert sum(x.startswith("   crossing ") for x in lines) == 3


def test_row_counts_are_reported_on_stdout(tiny):
    """Fails if the per-scenario row count, the guard against a truncated table, is not printed."""
    assert "rows per scenario (all blocks): noise 18, maneuver 10," in tiny["stdout"]


def test_a_missing_table_file_is_refused(tmp_path):
    """Fails if the script writes a partial summary when an input file is missing."""
    made = subprocess.run(
        [sys.executable, str(SCRIPT), "--results", str(tmp_path), "--out", str(tmp_path / "s.txt")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert made.returncode != 0
    assert "missing input" in made.stderr
    assert not (tmp_path / "s.txt").exists()
