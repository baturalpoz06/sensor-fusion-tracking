"""Sensor outage experiments: what a radar, camera or total outage does to the tracker.

Usage:
    python scripts/run_outage_experiment.py [--seeds 50] [--scenario crossing]
        [--experiments a b c d e f g] [--workers 4] [--out results] ...

Experiments (clean baseline: no clutter, Pd 0.9, outage from --outage-start):
    a  radar outage of growing duration, unaware vs aware tracker
    b  camera outage of growing duration (the policy cannot matter: the camera never
       touches the lifecycle)
    c  total blackout (radar and camera), unaware vs aware
    d  radar flicker: periodic (--flicker-period, off times) and random bursts (off
       fractions, --mean-burst), unaware vs aware
    e  aware tracker: max coasting time against the blackout duration
    f  a target ceases to exist during a radar outage: ghost lifetime against max
       coasting time (scenario "vanishing", whatever --scenario says)
    g  realistic run with clutter (--realistic-rate) at radar and total outages,
       unaware, aware dropping tentative tracks and aware freezing them

Known limitation: with close targets (the crossing at about 30 s in scenario "crossing") a
radar outage longer than --max-coast-time deletes real tracks in the aware tracker, because
the camera update that resets the coast clock is skipped for overlapping bearing gates. No
planned experiment triggers it (a-d and g keep the default 15 s above their radar outages);
the "coast deletion check" line after each outage table verifies it from the coast del counter.
Identity results are cleanest in scenario "separated"; the crossing tables carry a caveat.
Second effect of the same kind: camera clutter in the widened bearing gate of a coasting track
resets its coast clock, so with camera clutter a ghost outlives max coast time (experiment f
runs without clutter).

Every experiment runs the same seeds at every row, so truth, noise, detections and clutter
are identical and only the outage and the tracker's policy differ. Tables show the mean
over seeds with a 95% Student t interval, and medians of the heavy-tailed scores.
"""

import argparse
import re
import time
from dataclasses import replace
from pathlib import Path
from typing import NamedTuple

import numpy as np

from fusion.dropout import SingleOutage
from fusion.mtt_experiment import SweepResult
from fusion.mtt_simulation import FieldOfView
from fusion.outage_experiment import (
    OUTAGE_SCENARIOS,
    SENSOR_SETS,
    VANISHING_TARGET,
    OutageConfig,
    coast_points,
    duration_points,
    flicker_points,
    markov_points,
    outage_sweep,
)
from fusion.outage_metrics import WINDOWS
from fusion.outage_report import (
    format_coast_check,
    format_outage_table,
    format_window_table,
    outage_caption,
    select_rows,
)
from fusion.tracker.track import LifecycleConfig

EXPERIMENT_KEYS = ("a", "b", "c", "d", "e", "f", "g")
# Scores drawn in the plots (one panel each); the tables list everything.
PLOT_METRICS = (
    "during_missed_rate",
    "after_missed_rate",
    "during_position_rmse",
    "after_position_rmse",
    "after_ghost_rate",
    "after_id_switches",
    "reacquisition_time_censored",
    "reacquired_fraction",
    "identity_kept",
    "nees_outage",
    "ghost_lifetime",
    "coast_deletions",
)
LEGEND = (
    "windows (half-open, in steps): before = after the burn-in up to the outage start;",
    "  during = the outage span; after = from the outage end for --after-window seconds (clipped",
    "  at the end of the run). window scores are the Phase 6 scores restricted to the window;",
    "  id sw counts changes from the last id before the window; false/step counts confirmed",
    "  tracks never matched up to the end of the window; births/scan = tracks first reported in",
    "  the window per scheduled radar scan of it (lost scans included).",
    "outage scores: reacq = time to the first match after the outage of targets that were",
    "  tracked just before it, over those matched again; reacquired = share matched again;",
    "  reacq cens = mean over all of them with the window length for those not matched",
    "  (a lower bound); id kept = share whose first track after the outage has the id from",
    "  before; NEES = normalized error squared of the track that followed a target into the",
    "  outage, on its radar-down steps (ideal = 4, the state dimension; read with the NEES",
    "  samples and the before value); ghost life = seconds a vanished target's track survives;",
    "  coast del / tent drops = tracks deleted for coasting too long / tentative tracks dropped.",
    "  known limitation: with close targets a radar outage longer than max coast deletes real",
    "  tracks (camera ambiguity keeps the coast clock running); the 'coast deletion check' line",
    "  below each table verifies that no planned row triggers it. a second, similar effect:",
    "  camera clutter in the widened gate of a coasting track resets its clock, so with camera",
    "  clutter a ghost outlives max coast (experiment f runs clutter-free).",
    "rmse is taken over matched (target, step) pairs only: a row that loses its tracks (high",
    "  missed) can show a lower rmse; read it together with missed.",
    "after len = seconds of the after window actually scored; it is clipped at the end of the",
    "  run, so long outages have a shorter window and a higher after missed rate without the",
    "  tracker being worse. for flicker and bursts the during window is the whole span and id",
    "  kept / reacq compare the state at the span start with the state after the span; seeds",
    "  without any burst count as kept, which dilutes low off fractions.",
    "ghost life can be shorter than max coast when the outage ends first (the K-miss rule",
    "  deletes the ghost then). experiment f always uses the scenario 'vanishing' (its tables",
    "  are the same in the crossing and the separated run).",
    "a cell 'x +- 0' means no variation across the valid seeds, not certainty. in experiment g",
    "  ghost and false rates are dominated by a few seed events: compare drop and freeze with",
    "  their intervals. NEES means are heavy-tailed (a track that was off its target at the",
    "  outage start can give a huge value in one seed): read them with med NEES and NEES",
    "  outliers. camera outages (b) leave reacq, id kept and NEES neutral by design.",
    "cells are mean +- half-width of the 95% t interval across seeds over the seeds where the",
    "  score is defined; 'med' columns are medians; valid seeds = seeds with a defined",
    "  rmse / missed (window tables) or reacq / NEES (outage table).",
)


class Row(NamedTuple):
    """One row of an experiment table: a configuration and where it is plotted."""

    label: str
    config: OutageConfig
    x: float
    group: str | None


class Experiment(NamedTuple):
    """A set of rows run on the same seeds in one scenario."""

    key: str
    title: str
    scenario: str
    x_label: str
    rows: list[Row]


def parse_args() -> argparse.Namespace:
    defaults = OutageConfig()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=50, help="number of seeds (0..N-1)")
    parser.add_argument("--scenario", choices=["crossing", "separated"], default="crossing")
    parser.add_argument(
        "--experiments", nargs="+", choices=EXPERIMENT_KEYS, default=EXPERIMENT_KEYS
    )
    parser.add_argument("--workers", type=int, default=1, help="worker processes")
    parser.add_argument("--out", type=Path, default=Path("results"), help="plot directory")
    parser.add_argument("--no-plots", action="store_true", help="print the tables only")

    outage = parser.add_argument_group("outage")
    outage.add_argument("--outage-start", type=float, default=20.0, help="[s]")
    outage.add_argument("--after-window", type=float, default=defaults.after_window, help="[s]")
    outage.add_argument(
        "--durations", type=float, nargs="+", default=(0, 2, 4, 6, 8, 12), help="a b c [s]"
    )
    outage.add_argument("--flicker-span", type=float, default=30.0, help="d [s]")
    outage.add_argument("--flicker-period", type=float, default=10.0, help="d [s]")
    outage.add_argument("--flicker-off", type=float, nargs="+", default=(1, 2, 3, 4), help="d [s]")
    outage.add_argument("--flicker-phase", type=float, default=0.0, help="d [s]")
    outage.add_argument(
        "--burst-off", type=float, nargs="+", default=(0.05, 0.1, 0.2, 0.3), help="d"
    )
    outage.add_argument("--mean-burst", type=float, default=2.0, help="d [s]")
    outage.add_argument(
        "--coast-times", type=float, nargs="+", default=(5, 10, 15, 30), help="e f [s]"
    )
    outage.add_argument(
        "--coast-durations", type=float, nargs="+", default=(4, 8, 12, 16, 24, 32), help="e [s]"
    )
    outage.add_argument("--vanish-outage", type=float, default=24.0, help="f radar outage [s]")
    outage.add_argument(
        "--vanish-offset",
        type=float,
        default=2.0,
        help="f: vanishes this long after the outage start [s]",
    )
    outage.add_argument("--realistic-rate", type=float, default=5.0, help="g clutter rate")
    outage.add_argument(
        "--realistic-durations", type=float, nargs="+", default=(4, 8), help="g [s]"
    )

    tracker = parser.add_argument_group("tracker")
    tracker.add_argument(
        "--max-coast-time", type=float, default=defaults.max_coast_time, help="[s]"
    )
    tracker.add_argument(
        "--aware-tentatives", choices=["drop", "freeze"], default=defaults.aware_tentatives
    )
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
    tracker.add_argument("--no-camera", action="store_true", help="radar-only tracker")

    scene = parser.add_argument_group("scene")
    scene.add_argument(
        "--clutter-rate", type=float, default=0.0, help="baseline clutter, both sensors"
    )
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
    scene.add_argument("--range-min", type=float, default=defaults.fov.range_min, help="[m]")
    scene.add_argument("--range-max", type=float, default=defaults.fov.range_max, help="[m]")
    scene.add_argument("--burn-in", type=float, default=defaults.burn_in, help="[s]")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> OutageConfig:
    """The clean baseline configuration (no outage yet) from the command line."""
    return OutageConfig(
        dt=args.dt,
        duration=args.duration,
        radar_every=args.radar_every,
        radar_range_std=args.range_std,
        radar_bearing_std=float(np.deg2rad(args.radar_bearing_std_deg)),
        camera_bearing_std=float(np.deg2rad(args.camera_bearing_std_deg)),
        accel_std=args.accel_std,
        fov=FieldOfView(range_min=args.range_min, range_max=args.range_max),
        radar_pd=args.radar_pd,
        camera_pd=args.camera_pd,
        radar_clutter_rate=args.clutter_rate,
        camera_clutter_rate=args.clutter_rate,
        velocity_std=args.velocity_std,
        gate_probability=args.gate_probability,
        lifecycle=LifecycleConfig(args.confirm_hits, args.confirm_window, args.max_misses),
        use_camera=not args.no_camera,
        match_distance=args.match_distance,
        burn_in=args.burn_in,
        after_window=args.after_window,
        max_coast_time=args.max_coast_time,
        aware_tentatives=args.aware_tentatives,
    )


def policy_rows(
    base: OutageConfig, points: list[OutageConfig], xs, policies, unit: str, scale: float = 1.0
) -> list[Row]:
    """Rows for every policy at every point; each policy is one plotted group."""
    return [
        Row(
            f"{policy} {x * scale:g}{unit}",
            replace(point, outage_policy=policy),
            float(x),
            policy,
        )
        for policy in policies
        for point, x in zip(points, xs, strict=True)
    ]


def build_experiments(args: argparse.Namespace, base: OutageConfig) -> list[Experiment]:
    """The selected experiments with all their rows."""
    start = args.outage_start
    policies = ("unaware", "aware")
    scenario = args.scenario
    experiments = []

    def points_for(sensors):
        return duration_points(base, SENSOR_SETS[sensors], start, args.durations)

    if "a" in args.experiments:
        rows = policy_rows(base, points_for("radar"), args.durations, policies, " s")
        experiments.append(Experiment("a", "radar outage", scenario, "outage [s]", rows))
    if "b" in args.experiments:
        rows = policy_rows(base, points_for("camera"), args.durations, ("unaware",), " s")
        experiments.append(Experiment("b", "camera outage", scenario, "outage [s]", rows))
    if "c" in args.experiments:
        rows = policy_rows(base, points_for("blackout"), args.durations, policies, " s")
        experiments.append(Experiment("c", "total blackout", scenario, "outage [s]", rows))
    if "d" in args.experiments:
        radar = SENSOR_SETS["radar"]
        periodic = flicker_points(
            base,
            radar,
            start,
            args.flicker_span,
            args.flicker_period,
            args.flicker_off,
            args.flicker_phase,
        )
        title = f"periodic radar flicker, period {args.flicker_period:g} s"
        rows = policy_rows(base, periodic, args.flicker_off, policies, " s off")
        experiments.append(Experiment("d-periodic", title, scenario, "off time [s]", rows))
        bursts = markov_points(
            base, radar, start, args.flicker_span, args.burst_off, args.mean_burst
        )
        title = f"random radar bursts, mean burst {args.mean_burst:g} s"
        rows = policy_rows(base, bursts, args.burst_off, policies, "% off", scale=100.0)
        experiments.append(Experiment("d-burst", title, scenario, "off fraction", rows))
    if "e" in args.experiments:
        aware = replace(base, outage_policy="aware")
        rows = []
        for coast in args.coast_times:
            points = duration_points(
                replace(aware, max_coast_time=coast),
                SENSOR_SETS["blackout"],
                start,
                args.coast_durations,
            )
            rows += [
                Row(
                    f"coast {coast:g} s, blackout {d:g} s", point, float(d), f"coast {coast:g} s"
                )
                for point, d in zip(points, args.coast_durations, strict=True)
            ]
        title = "aware tracker, max coasting time against the blackout duration"
        experiments.append(Experiment("e", title, scenario, "blackout [s]", rows))
    if "f" in args.experiments:
        vanishing = replace(
            base,
            dropout=SingleOutage(SENSOR_SETS["radar"], start, args.vanish_outage),
            vanish=((VANISHING_TARGET, start + args.vanish_offset),),
        )
        rows = [Row("unaware", replace(vanishing, outage_policy="unaware"), 0.0, None)]
        aware = replace(vanishing, outage_policy="aware")
        rows += [
            Row(f"aware, coast {p.max_coast_time:g} s", p, p.max_coast_time, "aware")
            for p in coast_points(aware, args.coast_times)
        ]
        title = (
            f"target {VANISHING_TARGET} ceases to exist {args.vanish_offset:g} s into a "
            f"{args.vanish_outage:g} s radar outage"
        )
        experiments.append(Experiment("f", title, "vanishing", "max coast [s]", rows))
    if "g" in args.experiments:
        noisy = replace(
            base, radar_clutter_rate=args.realistic_rate, camera_clutter_rate=args.realistic_rate
        )
        variants = {
            "unaware": {"outage_policy": "unaware"},
            "aware drop": {"outage_policy": "aware", "aware_tentatives": "drop"},
            "aware freeze": {"outage_policy": "aware", "aware_tentatives": "freeze"},
        }
        rows = []
        for sensors in ("radar", "blackout"):
            points = duration_points(noisy, SENSOR_SETS[sensors], start, args.realistic_durations)
            for name, settings in variants.items():
                rows += [
                    Row(
                        f"{sensors} {d:g} s, {name}",
                        replace(point, **settings),
                        float(d),
                        f"{sensors}, {name}",
                    )
                    for point, d in zip(points, args.realistic_durations, strict=True)
                ]
        title = f"clutter rate {args.realistic_rate:g} per scan and sensor"
        experiments.append(Experiment("g", title, scenario, "outage [s]", rows))
    return experiments


def run_experiment(experiment: Experiment, args: argparse.Namespace, seeds: range) -> SweepResult:
    """Run one experiment on the shared seeds and print its tables."""
    print()
    print(
        f"=== experiment {experiment.key}: {experiment.title} (scenario '{experiment.scenario}', "
        f"{len(seeds)} seeds each, {len(experiment.rows)} rows) ==="
    )
    start = time.time()
    result = outage_sweep(
        OUTAGE_SCENARIOS[experiment.scenario],
        [row.config for row in experiment.rows],
        [row.x for row in experiment.rows],
        experiment.x_label,
        seeds,
        args.workers,
    )
    print(f"({time.time() - start:.0f} s)")
    labels = [row.label for row in experiment.rows]
    caption = outage_caption(experiment.rows[0].config, experiment.scenario)
    for window in WINDOWS:
        print()
        print(f"window: {window}")
        for line in format_window_table(
            result, window, labels, caption=caption if window == "before" else None
        ):
            print(line)
    print()
    print("outage scores")
    for line in format_outage_table(result, labels):
        print(line)
    for line in format_coast_check([row.config for row in experiment.rows], result, labels):
        print(line)
    return result


def save_plots(experiment: Experiment, result: SweepResult, args: argparse.Namespace) -> None:
    """One figure per plotted group of the experiment (rows with the same group)."""
    from fusion.mtt_plots import plot_sweep

    chosen = {name: result.metrics[name] for name in PLOT_METRICS}
    groups = {}
    for index, row in enumerate(experiment.rows):
        if row.group is not None:
            groups.setdefault(row.group, []).append(index)
    for group, rows in groups.items():
        if len(rows) < 2:
            continue
        part = select_rows(SweepResult(experiment.x_label, result.values, chosen), rows)
        slug = re.sub(r"[^a-z0-9]+", "_", f"{experiment.key}_{group}".lower()).strip("_")
        path = args.out / f"outage_{experiment.scenario}_{slug}.png"
        title = f"{experiment.title}, {group}: scenario '{experiment.scenario}', 95% interval"
        print(f"saved {plot_sweep(part, path, title=title)}")


def main() -> None:
    args = parse_args()
    base = build_config(args)
    seeds = range(args.seeds)
    lifecycle = base.lifecycle
    print(
        f"scenario '{args.scenario}', {len(seeds)} seeds, {base.duration:g} s, dt={base.dt:g} s, "
        f"radar every {base.radar_every} steps, "
        f"camera {'every step' if base.use_camera else 'off'}, "
        f"burn-in {base.burn_in:g} s, outage from {args.outage_start:g} s"
    )
    print(
        f"lifecycle: {lifecycle.confirm_hits}-of-{lifecycle.confirm_window} confirmation, "
        f"deleted after more than {lifecycle.max_misses} consecutive misses; "
        f"gate {100 * base.gate_probability:g}%; match distance {base.match_distance:g} m"
    )
    for line in LEGEND:
        print(line)

    for experiment in build_experiments(args, base):
        result = run_experiment(experiment, args, seeds)
        if not args.no_plots:
            save_plots(experiment, result, args)


if __name__ == "__main__":
    main()
