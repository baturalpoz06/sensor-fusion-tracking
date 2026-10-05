"""Robustness measurement (Phase 8a): how the tracker degrades when its beliefs are wrong.

Usage:
    python scripts/run_robustness_experiment.py [--scenario all] [--seeds 50]
        [--layouts crossing separated] [--clutter-rates 0 5] [--workers 4] [--out results] ...
    python scripts/run_robustness_experiment.py --pilot --workers 4

Scenarios (--scenario):
    noise      the tracker's assumed noise levels scaled by k (radar range, radar bearing, camera
               bearing, process noise), one knob at a time; the simulated world is unchanged
    maneuver   all targets turn (coordinated turn), accelerate, or turn at random, while the
               tracker keeps its constant-velocity model
    camera     a constant camera bearing bias, and a camera time offset (the camera data lag)
    lever-arm  the camera sits away from the radar; the tracker assumes it at the radar
               (unknown) or knows the true position (known)
    geometry   scene geometry: approach direction x range zone, target count, clutter rate, each
               neutral and with an unknown lever arm
    all        every scenario above

Scenarios other than geometry run on every layout in --layouts at every clutter rate in
--clutter-rates. Every sweep includes its neutral row. All rows of a sweep run the same seeds, so
truth, noise, detections and clutter are identical across rows (common random numbers); the
paired tables compare each row with the neutral row seed by seed.

--pilot runs 5 seeds into <out>/pilot with a banner: it only checks that everything runs and
supports no conclusion. Tables go to stdout and to <out>/robustness_<scenario>.txt.
"""

import argparse
import re
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np

from fusion.mtt_experiment import SCENARIOS, SweepResult
from fusion.outage_report import select_rows
from fusion.robustness_experiment import (
    RobustnessConfig,
    Row,
    camera_rows,
    geometry_rows,
    lever_arm_rows,
    maneuver_rows,
    noise_scale_rows,
    robustness_sweep,
)
from fusion.robustness_report import (
    CROSSING_MANEUVER_CAVEAT,
    LEGEND,
    PILOT_BANNER,
    format_diagnostic_table,
    format_paired_table,
    format_score_table,
    format_window_table,
    has_window_scores,
    robustness_caption,
)
from fusion.tracker.track import LifecycleConfig

SCENARIO_KEYS = ("noise", "maneuver", "camera", "lever-arm", "geometry")
PILOT_SEEDS = 5
# Scores drawn in the plots (one panel each); the tables list everything.
PLOT_METRICS = (
    "run_position_rmse",
    "run_missed_rate",
    "cross_rms",
    "along_rms",
    "nees_mean",
    "camera_accept_rate",
    "camera_true_accept_rate",
    "match_fraction_wide",
)
WINDOW_PLOT_METRICS = ("window_position_rmse", "window_cross_rms")
# Axis label of the plotted series: group -> text (the group of a row names what it sweeps).
AXIS_LABELS = {
    "radar_range": "assumed radar range std, scale k",
    "radar_bearing": "assumed radar bearing std, scale k",
    "camera_bearing": "assumed camera bearing std, scale k",
    "process_noise": "assumed process noise std, scale k",
    "turn": "turn rate [deg/s]",
    "acceleration": "acceleration [m/s^2]",
    "random": "largest random turn rate [deg/s]",
    "bias": "camera bearing bias [deg]",
    "time offset": "camera time offset [ms]",
    "known": "camera distance from the radar [m], position known to the tracker",
}


class Block(NamedTuple):
    """The rows of one table set: one scene (layout) at one clutter rate."""

    title: str
    layout: str | None
    rows: list[Row]


def parse_args() -> argparse.Namespace:
    defaults = RobustnessConfig()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenario", choices=(*SCENARIO_KEYS, "all"), default="all")
    parser.add_argument("--seeds", type=int, default=None, help="number of seeds (default 50)")
    parser.add_argument(
        "--pilot", action="store_true", help=f"{PILOT_SEEDS} seeds into <out>/pilot, no conclusions"
    )
    parser.add_argument(
        "--layouts", nargs="+", choices=sorted(SCENARIOS), default=sorted(SCENARIOS)
    )
    parser.add_argument("--clutter-rates", type=float, nargs="+", default=(0.0, 5.0))
    parser.add_argument("--workers", type=int, default=1, help="worker processes")
    parser.add_argument("--out", type=Path, default=Path("results"), help="output directory")
    parser.add_argument("--no-plots", action="store_true", help="print the tables only")

    grids = parser.add_argument_group("grids")
    grids.add_argument("--noise-scales", type=float, nargs="+", default=(0.25, 0.5, 1.0, 2.0, 4.0))
    grids.add_argument(
        "--turn-rates", type=float, nargs="+", default=(5.0, 10.0, 20.0), help="deg/s"
    )
    grids.add_argument("--turn-window", type=float, nargs=2, default=(15.0, 25.0), help="[s]")
    grids.add_argument("--accelerations", type=float, nargs="+", default=(1.0, 2.0, 4.0))
    grids.add_argument("--acceleration-window", type=float, nargs=2, default=(15.0, 18.0))
    grids.add_argument("--random-rates", type=float, nargs="+", default=(5.0, 10.0, 20.0))
    grids.add_argument("--random-start", type=float, default=10.0, help="[s]")
    grids.add_argument("--random-segment", type=float, default=5.0, help="[s]")
    grids.add_argument("--biases", type=float, nargs="+", default=(0.1, 0.25, 0.5, 1.0), help="deg")
    grids.add_argument("--offsets-ms", type=float, nargs="+", default=(25.0, 50.0, 100.0, 200.0))
    grids.add_argument(
        "--lever-distances", type=float, nargs="+", default=(1.0, 2.0, 5.0, 10.0, 20.0, 50.0)
    )
    grids.add_argument("--lever-angles", type=float, nargs="+", default=(0.0, 90.0), help="deg")
    grids.add_argument("--known-angle", type=float, default=0.0, help="deg")
    grids.add_argument("--geometry-lever-arm", type=float, default=20.0, help="[m]")
    grids.add_argument("--geometry-lever-angle", type=float, default=45.0, help="deg")
    grids.add_argument("--geometry-clutter", type=float, nargs="+", default=(2.0, 5.0, 10.0))
    grids.add_argument("--geometry-counts", type=int, nargs="+", default=(1, 2, 4, 8))
    grids.add_argument("--geometry-base-clutter", type=float, default=0.0)

    tracker = parser.add_argument_group("tracker and scoring")
    tracker.add_argument(
        "--confirm-hits", type=int, default=defaults.lifecycle.confirm_hits, help="M"
    )
    tracker.add_argument(
        "--confirm-window", type=int, default=defaults.lifecycle.confirm_window, help="N"
    )
    tracker.add_argument("--max-misses", type=int, default=defaults.lifecycle.max_misses, help="K")
    tracker.add_argument("--velocity-std", type=float, default=defaults.velocity_std, help="[m/s]")
    tracker.add_argument("--gate-probability", type=float, default=defaults.gate_probability)
    tracker.add_argument(
        "--match-distance", type=float, default=defaults.match_distance, help="[m]"
    )
    tracker.add_argument(
        "--diagnostic-match-distance",
        type=float,
        default=defaults.diagnostic_match_distance,
        help="[m]",
    )

    scene = parser.add_argument_group("scene")
    scene.add_argument("--radar-pd", type=float, default=defaults.radar_pd)
    scene.add_argument("--camera-pd", type=float, default=defaults.camera_pd)
    scene.add_argument("--duration", type=float, default=defaults.duration, help="[s]")
    scene.add_argument("--dt", type=float, default=defaults.dt, help="time step [s]")
    scene.add_argument("--radar-every", type=int, default=defaults.radar_every, help="steps")
    scene.add_argument("--range-std", type=float, default=defaults.radar_range_std, help="[m]")
    scene.add_argument(
        "--radar-bearing-std-deg",
        type=float,
        default=float(np.rad2deg(defaults.radar_bearing_std)),
        help="[deg]",
    )
    scene.add_argument(
        "--camera-bearing-std-deg",
        type=float,
        default=float(np.rad2deg(defaults.camera_bearing_std)),
        help="[deg]",
    )
    scene.add_argument("--accel-std", type=float, default=defaults.accel_std, help="[m/s^2]")
    scene.add_argument("--burn-in", type=float, default=defaults.burn_in, help="[s]")
    args = parser.parse_args()
    if args.seeds is None:
        args.seeds = PILOT_SEEDS if args.pilot else 50
    if args.seeds < 1:
        parser.error("--seeds must be >= 1")
    return args


def build_config(args: argparse.Namespace, clutter: float) -> RobustnessConfig:
    """The neutral configuration (no disturbance, a truthful belief) at one clutter rate."""
    return RobustnessConfig(
        dt=args.dt,
        duration=args.duration,
        radar_every=args.radar_every,
        radar_range_std=args.range_std,
        radar_bearing_std=float(np.deg2rad(args.radar_bearing_std_deg)),
        camera_bearing_std=float(np.deg2rad(args.camera_bearing_std_deg)),
        accel_std=args.accel_std,
        radar_pd=args.radar_pd,
        camera_pd=args.camera_pd,
        radar_clutter_rate=clutter,
        camera_clutter_rate=clutter,
        velocity_std=args.velocity_std,
        gate_probability=args.gate_probability,
        lifecycle=LifecycleConfig(args.confirm_hits, args.confirm_window, args.max_misses),
        match_distance=args.match_distance,
        burn_in=args.burn_in,
        diagnostic_match_distance=args.diagnostic_match_distance,
    )


def scenario_rows(scenario: str, base: RobustnessConfig, n_targets: int, args) -> list[Row]:
    """The rows of one scenario for one base configuration."""
    if scenario == "noise":
        return noise_scale_rows(base, args.noise_scales)
    if scenario == "maneuver":
        return maneuver_rows(
            base,
            n_targets,
            turn_rates_deg=args.turn_rates,
            turn_window=tuple(args.turn_window),
            accelerations=args.accelerations,
            acceleration_window=tuple(args.acceleration_window),
            random_rates_deg=args.random_rates,
            random_segment=args.random_segment,
            random_start=args.random_start,
        )
    if scenario == "camera":
        return camera_rows(base, args.biases, args.offsets_ms)
    return lever_arm_rows(base, args.lever_distances, args.lever_angles, args.known_angle)


def build_blocks(scenario: str, args: argparse.Namespace) -> list[Block]:
    """The table sets of a scenario: one per layout and clutter rate (one for geometry)."""
    if scenario == "geometry":
        base = build_config(args, args.geometry_base_clutter)
        rows = geometry_rows(
            base,
            args.geometry_lever_arm,
            args.geometry_lever_angle,
            args.geometry_clutter,
            args.geometry_counts,
        )
        return [Block(f"geometry, base clutter {args.geometry_base_clutter:g}", None, rows)]
    blocks = []
    for layout in args.layouts:
        states = SCENARIOS[layout]
        for clutter in args.clutter_rates:
            rows = scenario_rows(scenario, build_config(args, clutter), len(states), args)
            rows = [row._replace(states=states) for row in rows]
            blocks.append(Block(f"{scenario}, {layout}, clutter {clutter:g}", layout, rows))
    return blocks


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def save_plots(
    scenario: str, block: Block, result: SweepResult, args: argparse.Namespace, out_dir: Path, emit
) -> None:
    """One figure per plotted group of the block, each with the neutral row as its anchor."""
    from fusion.mtt_plots import plot_sweep

    names = list(PLOT_METRICS)
    if scenario == "maneuver":
        names += list(WINDOW_PLOT_METRICS)
    chosen = {name: result.metrics[name] for name in names}
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(block.rows):
        if row.group is not None:
            groups.setdefault(row.group, []).append(index)
    for group, indices in groups.items():
        picked = sorted([0, *indices], key=lambda i: result.values[i])
        if len(picked) < 3:
            continue
        label = AXIS_LABELS.get(group, f"{group}: camera distance from the radar [m]")
        part = select_rows(SweepResult(label, result.values, chosen), picked)
        name = f"robustness_{scenario}_{slug(block.title)}_{slug(group)}.png"
        title = f"{block.title}, {group}: {args.seeds} seeds, 95% interval"
        path = plot_sweep(part, out_dir / name, title=title, log_x=scenario == "noise")
        emit(f"saved {path}")


def run_scenario(scenario: str, args: argparse.Namespace, out_dir: Path) -> None:
    """Run one scenario: all its blocks in one process pool, then print and save the tables."""
    lines: list[str] = []

    def emit(line: str = "") -> None:
        print(line)
        lines.append(line)

    blocks = build_blocks(scenario, args)
    rows = [row for block in blocks for row in block.rows]
    seeds = range(args.seeds)
    emit()
    emit(
        f"=== scenario {scenario}: {len(blocks)} table set(s), {len(rows)} rows, "
        f"{len(seeds)} seeds each = {len(rows) * len(seeds)} trials ==="
    )
    if args.pilot:
        emit(PILOT_BANNER)
    started = time.time()
    result = robustness_sweep(rows, SCENARIOS["separated"], scenario, seeds, args.workers)
    emit(f"({time.time() - started:.0f} s)")

    start = 0
    for block in blocks:
        indices = list(range(start, start + len(block.rows)))
        start += len(block.rows)
        part = select_rows(result, indices)
        labels = [row.label for row in block.rows]
        caption = robustness_caption(block.rows[0].config, block.layout)
        if scenario == "maneuver" and block.layout == "crossing":
            caption += f"; {CROSSING_MANEUVER_CAVEAT}"
        # Rows of the geometry scenario come in pairs (neutral, unknown arm); the others
        # are all compared with their first row, the neutral one.
        references = [i - i % 2 for i in range(len(labels))] if scenario == "geometry" else 0
        emit()
        emit(f"--- {block.title} ---")
        tables = [
            ("scores (Phase 6)", format_score_table(part, labels, caption=caption)),
            ("error and camera diagnostics", format_diagnostic_table(part, labels)),
        ]
        if has_window_scores(part):
            tables.append(("focus window scores", format_window_table(part, labels)))
        tables.append(
            ("paired differences, row - neutral", format_paired_table(part, references, labels))
        )
        for name, table in tables:
            emit()
            emit(name)
            for line in table:
                emit(line)
        if not args.no_plots and scenario != "geometry":
            save_plots(scenario, block, part, args, out_dir, emit)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"robustness_{scenario}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = args.out / "pilot" if args.pilot else args.out
    scenarios = SCENARIO_KEYS if args.scenario == "all" else (args.scenario,)
    rates = ", ".join(f"{c:g}" for c in args.clutter_rates)
    print(
        f"robustness measurement: scenarios {', '.join(scenarios)}; layouts "
        f"{', '.join(args.layouts)}; clutter rates {rates}; "
        f"{args.seeds} seeds, {args.duration:g} s, dt={args.dt:g} s, burn-in {args.burn_in:g} s"
    )
    if args.pilot:
        print(PILOT_BANNER)
    for line in LEGEND:
        print(line)
    for scenario in scenarios:
        run_scenario(scenario, args, out_dir)
    if args.pilot:
        print()
        print(PILOT_BANNER)


if __name__ == "__main__":
    main()
