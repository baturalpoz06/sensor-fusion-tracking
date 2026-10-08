"""Headline numbers of the robustness measurement (Phase 8a), extracted from its text tables.

Usage:
    python scripts/make_robustness_summary.py [--results results]
        [--out results/summary_headline.txt] [--duration 60] [--burn-in 5] [--focus-window 15 35]

Reads results/robustness_{noise,maneuver,camera,lever-arm,geometry}.txt (written by
run_robustness_experiment.py) and writes one compact table: for every block and row, the paired
differences row - neutral of rmse, missed rate, cross-range rms, NEES and the camera true accept
rate, with the row's own ghost rate and ID switches; the neutral row shows its own values
("[abs]"). Three plain extractions ("prediction checks") follow. It runs nothing and computes
nothing: every cell is copied from the tables, which are parsed by the column widths of
fusion.mtt_report and fusion.robustness_report. The file starts with a provenance header.

--duration, --burn-in and --focus-window only label the header (the tables do not state them);
the number of seeds is read from the tables.
"""

import argparse
import re
import sys
from pathlib import Path

from fusion.mtt_report import COLUMN_WIDTH, MEDIAN_WIDTH, VALID_WIDTH
from fusion.provenance import provenance_lines
from fusion.robustness_report import (
    DIAGNOSTIC_COLUMNS,
    DIAGNOSTIC_MEDIANS,
    PAIRED_MEDIAN_WIDTH,
    PAIRED_METRICS,
    SCORE_COLUMNS,
    SCORE_MEDIANS,
    WINDOW_COLUMNS,
    WINDOW_MEDIANS,
)

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ("noise", "maneuver", "camera", "lever-arm", "geometry")
# Heading line of a table -> (key, mean columns, median columns, width of a median column).
TABLES = {
    "scores (Phase 6)": ("scores", list(SCORE_COLUMNS), list(SCORE_MEDIANS), MEDIAN_WIDTH),
    "error and camera diagnostics": (
        "diag",
        list(DIAGNOSTIC_COLUMNS),
        list(DIAGNOSTIC_MEDIANS),
        MEDIAN_WIDTH,
    ),
    "focus window scores": ("window", list(WINDOW_COLUMNS), list(WINDOW_MEDIANS), MEDIAN_WIDTH),
    "paired differences, row - neutral": (
        "paired",
        list(PAIRED_METRICS),
        list(PAIRED_METRICS),
        PAIRED_MEDIAN_WIDTH,
    ),
}
SEEDS_LINE = re.compile(r"=== scenario (\S+): .*, (\d+) seeds each = \d+ trials ===")
# Short geometry labels, so that every label fits the first column.
SHORT = (
    (", unknown 20 m arm", " +arm"),
    ("tangential mid, ", "tan mid "),
    (" targets in a sector", " tgt/sector"),
    (", clutter ", " lam "),
)
LABEL_WIDTH = 39


def short_label(label: str) -> str:
    """The row label with the geometry abbreviations, at most LABEL_WIDTH characters."""
    for old, new in SHORT:
        label = label.replace(old, new)
    if len(label) > LABEL_WIDTH:
        raise SystemExit(f"row label too long for the summary: {label!r}")
    return label


def parse_row(
    line: str, ncols: int, nmed: int, med_width: int
) -> tuple[str, list[str], list[str]]:
    """Label, mean cells and median cells of one table row, cut at the fixed column widths.

    A row is the label, ncols cells of COLUMN_WIDTH, nmed cells of med_width and the valid-seeds
    (or dropped) cell of VALID_WIDTH; the label is whatever precedes them.
    """
    tail = COLUMN_WIDTH * ncols + med_width * nmed + VALID_WIDTH
    if len(line) < tail:
        raise SystemExit(f"table row shorter than its columns: {line!r}")
    label = line[: len(line) - tail].strip()
    body = line[len(line) - tail :]
    cells = [body[i * COLUMN_WIDTH : (i + 1) * COLUMN_WIDTH].strip() for i in range(ncols)]
    start = COLUMN_WIDTH * ncols
    meds = [body[start + j * med_width : start + (j + 1) * med_width].strip() for j in range(nmed)]
    return label, cells, meds


def parse_tables(text: str) -> dict[str, dict[str, list]]:
    """Block title -> table key -> ordered list of (label, {column: cell}, {median: cell})."""
    blocks: dict[str, dict[str, list]] = {}
    lines = text.splitlines()
    block = None
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("--- ") and line.endswith(" ---"):
            block = line[4:-4]
            blocks[block] = {}
        elif block is not None and line in TABLES:
            key, cols, meds, med_width = TABLES[line]
            i += 1
            while i < len(lines) and not set(lines[i]) <= {"-"}:  # caption and header lines
                i += 1
            i += 1
            rows = []
            while i < len(lines) and lines[i].strip() and not lines[i].startswith("saved "):
                label, cells, med_cells = parse_row(lines[i], len(cols), len(meds), med_width)
                means = dict(zip(cols, cells, strict=True))
                rows.append((label, means, dict(zip(meds, med_cells, strict=True))))
                i += 1
            blocks[block][key] = rows
            continue
        i += 1
    return blocks


def seeds_of(text: str, scenario: str) -> int:
    """The number of seeds stated in the first line of a robustness table file."""
    match = SEEDS_LINE.search(text)
    if match is None or match.group(1) != scenario:
        raise SystemExit(f"robustness_{scenario}.txt has no '=== scenario {scenario}: ...' line")
    return int(match.group(2))


def index(rows: list) -> dict[str, tuple[dict, dict]]:
    return {label: (cells, meds) for label, cells, meds in rows}


def need(mapping: dict, key: str, where: str):
    """mapping[key], or a clear exit if the tables lack it."""
    if key not in mapping:
        raise SystemExit(f"{where}: no {key!r} in the tables")
    return mapping[key]


def headline_lines(
    data: dict, seeds: int, duration: float, burn_in: float, source: str
) -> list[str]:
    """The header and one table block per (scenario, block) of the input."""
    out = [
        f"HEADLINE NUMBERS from {source}/robustness_*.txt ({seeds} seeds, {duration:g} s, "
        f"burn-in {burn_in:g} s).",
        "No interpretation. Cells: mean +- 95% t half-width over seeds. d = row - neutral,",
        "per seed (paired); [abs] = the neutral row's own value. ghost = ghost tracks/step;",
        "id sw = ID switches per run; cam true = camera_true_accept_rate.",
    ]
    heading = (
        f"{'row':<39} {'d rmse [m]':>15} {'d missed':>15} {'d cross [m]':>15} {'d NEES':>15} "
        f"{'d cam true':>15} {'ghost':>15} {'id sw':>12}"
    )
    out += ["", heading, "-" * len(heading)]
    paired_cols = list(PAIRED_METRICS)

    def row_line(label: str, five: list[str], ghost: str, idsw: str) -> str:
        cells = " ".join(f"{c:>15}" for c in five)
        return f"{short_label(label):<39} {cells} {ghost:>15} {idsw:>12}"

    for scenario in SCENARIOS:
        for title, tables in data[scenario].items():
            for key in ("scores", "diag", "paired"):
                need(tables, key, title)
            scores, diag = index(tables["scores"]), index(tables["diag"])
            paired = index(tables["paired"])
            out.append(f"== {title}")
            for label, (score_cells, _) in scores.items():
                diag_cells = need(diag, label, f"{title}, diagnostics")[0]
                ghost, idsw = score_cells["run_ghost_rate"], score_cells["run_id_switches"]
                if label in paired:
                    five = [paired[label][0][c] for c in paired_cols]
                    out.append(row_line(label, five, ghost, idsw))
                else:  # a reference row (neutral): its own values
                    absolute = [
                        score_cells["run_position_rmse"],
                        score_cells["run_missed_rate"],
                        diag_cells["cross_rms"],
                        diag_cells["nees_mean"],
                        diag_cells["camera_true_accept_rate"],
                    ]
                    out.append(row_line(label + " [abs]", absolute, ghost, idsw))
    return out


def prediction_lines(data: dict, focus_window: tuple[float, float]) -> list[str]:
    """The three plain extractions after the table."""
    out = ["", "PREDICTION CHECKS (plain extractions; mean +- 95% half-width, med = median)", ""]
    out.append("1) noise sweep, clutter 5, separated: ghost tracks per step")
    block = need(data["noise"], "noise, separated, clutter 5", "noise")
    scores = index(need(block, "scores", "noise, separated, clutter 5"))
    for knob in ("radar_range", "camera_bearing", "process_noise"):
        for k in ("0.25", "4"):
            cells, meds = need(scores, f"{knob} x{k}", "noise, separated, clutter 5")
            out.append(
                f"   {knob:<15} k={k:<5} ghost {cells['run_ghost_rate']:<16} "
                f"med {meds['run_ghost_rate']}"
            )
    out.append("")
    start, end = focus_window
    out.append(
        "2) maneuver sweep, separated: missed rate, whole run and focus window "
        f"[{start:g}, {end:g}) s"
    )
    for clutter in ("0", "5"):
        title = f"maneuver, separated, clutter {clutter}"
        block = need(data["maneuver"], title, "maneuver")
        scores = index(need(block, "scores", title))
        window = index(need(block, "window", title))
        for omega in ("5", "20"):
            label = f"turn {omega} deg/s"
            s_cells, s_meds = need(scores, label, title)
            w_cells, _ = need(window, label, f"{title}, window")
            out.append(
                f"   lambda={clutter} omega={omega:<3} whole run {s_cells['run_missed_rate']:<16} "
                f"med {s_meds['run_missed_rate']:<8} window {w_cells['window_missed_rate']}"
            )
    out.append("")
    out.append(
        "3) geometry sweep, neutral vs unknown 20 m arm (lever angle 45 deg), base clutter 0"
    )
    out.append(
        f"   {'scene':<28} {'cam true neutral':>18} {'cam true arm':>18} {'cross rms neutral':>19}"
        f" {'cross rms arm':>16}"
    )
    title = "geometry, base clutter 0"
    diag = index(need(need(data["geometry"], title, "geometry"), "diag", title))
    for direction in ("inbound", "outbound", "tangential", "crossing"):
        for zone in ("near", "mid", "far"):
            name = f"{direction} {zone}"
            neutral = need(diag, name, title)[0]
            arm = need(diag, f"{name}, unknown 20 m arm", title)[0]
            out.append(
                f"   {name:<28} {neutral['camera_true_accept_rate']:>18} "
                f"{arm['camera_true_accept_rate']:>18} {neutral['cross_rms']:>19} "
                f"{arm['cross_rms']:>16}"
            )
    return out


def build_summary(
    results: Path, duration: float, burn_in: float, focus_window: tuple[float, float]
) -> tuple[list[str], dict[str, int]]:
    """The summary lines and the number of rows per scenario (summed over its blocks)."""
    data, seeds = {}, {}
    for scenario in SCENARIOS:
        path = results / f"robustness_{scenario}.txt"
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
        text = path.read_text(encoding="utf-8")
        seeds[scenario] = seeds_of(text, scenario)
        data[scenario] = parse_tables(text)
        if not data[scenario]:
            raise SystemExit(f"{path} contains no table block")
    if len(set(seeds.values())) != 1:
        raise SystemExit(f"the tables were run with different seed counts: {seeds}")
    rows = {
        scenario: sum(len(tables.get("scores", ())) for tables in data[scenario].values())
        for scenario in SCENARIOS
    }
    lines = headline_lines(data, seeds["noise"], duration, burn_in, results.name)
    return lines + prediction_lines(data, focus_window), rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "summary_headline.txt")
    parser.add_argument("--duration", type=float, default=60.0, help="[s], header label only")
    parser.add_argument("--burn-in", type=float, default=5.0, help="[s], header label only")
    parser.add_argument(
        "--focus-window", type=float, nargs=2, default=(15.0, 35.0), help="[s], label only"
    )
    args = parser.parse_args()
    lines, rows = build_summary(args.results, args.duration, args.burn_in, tuple(args.focus_window))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join([*provenance_lines(sys.argv), *lines]) + "\n"
    args.out.write_text(text, encoding="utf-8")
    counts = ", ".join(f"{scenario} {n}" for scenario, n in rows.items())
    print(f"wrote {args.out}; rows per scenario (all blocks): {counts}")


if __name__ == "__main__":
    main()
