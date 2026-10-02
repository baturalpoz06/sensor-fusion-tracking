"""Multi-target tracking with clutter and missed detections: clutter-rate and Pd sweeps.

Usage:
    python scripts/run_mtt_experiment.py [--seeds 50] [--scenario crossing]
        [--sweeps clutter pd] [--sweep-sensors both|radar|camera] [--workers 4]
        [--out results] ...

Each sweep runs the same seeds at every value and reports the mean over seeds with
a 95% Student t interval and the median of the heavy-tailed metrics; the plots show
the interval as a band. "--sweeps compare" scores the radar-only and the fused tracker
on the same seeds instead.
"""

import argparse
import time
from pathlib import Path

from fusion.mtt_experiment import (
    SCENARIOS,
    MttConfig,
    SweepResult,
    clutter_gate_report,
    compare_fusion,
    sweep,
)
from fusion.mtt_report import format_sweep_table
from fusion.tracker.track import LifecycleConfig

# Radar clutter creates ghost tracks: with M-of-N confirmation the ghost rate stays at zero
# up to a few points per scan and then rises steeply, so the grid is dense from 10 to 30.
# Camera clutter never starts tracks and only disturbs the updates of confirmed tracks; the
# effect of the camera clutter rate becomes clear only in the tens of points per scan.
DEFAULT_RADAR_CLUTTER_RATES = (0.0, 3.0, 5.0, 7.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0)
DEFAULT_CAMERA_CLUTTER_RATES = (0.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0)
DEFAULT_PDS = (0.6, 0.7, 0.8, 0.9, 1.0)
SENSOR_PREFIX = {"both": "", "radar": "radar_", "camera": "camera_"}
VALUE_LABELS = {"clutter": "clutter rate [pts/scan]", "pd": "detection probability"}


def parse_args() -> argparse.Namespace:
    defaults = MttConfig()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=50, help="number of seeds (0..N-1)")
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="crossing")
    parser.add_argument(
        "--sweeps", nargs="+", choices=["clutter", "pd", "compare"], default=["clutter", "pd"]
    )
    parser.add_argument(
        "--sweep-sensors",
        choices=sorted(SENSOR_PREFIX),
        default="both",
        help="which sensor's clutter rate / Pd is swept; the other keeps its base value",
    )
    parser.add_argument(
        "--clutter-rates",
        type=float,
        nargs="+",
        default=None,
        help="clutter rates to sweep; default depends on --sweep-sensors (camera: its own grid)",
    )
    parser.add_argument("--pds", type=float, nargs="+", default=DEFAULT_PDS)
    parser.add_argument("--radar-pd", type=float, default=defaults.radar_pd, help="base Pd")
    parser.add_argument("--camera-pd", type=float, default=defaults.camera_pd, help="base Pd")
    parser.add_argument(
        "--radar-clutter-rate", type=float, default=defaults.radar_clutter_rate, help="base rate"
    )
    parser.add_argument(
        "--camera-clutter-rate", type=float, default=defaults.camera_clutter_rate, help="base rate"
    )
    parser.add_argument(
        "--confirm-hits", type=int, default=defaults.lifecycle.confirm_hits, help="M"
    )
    parser.add_argument(
        "--confirm-window", type=int, default=defaults.lifecycle.confirm_window, help="N"
    )
    parser.add_argument("--max-misses", type=int, default=defaults.lifecycle.max_misses, help="K")
    parser.add_argument("--velocity-std", type=float, default=defaults.velocity_std, help="[m/s]")
    parser.add_argument("--gate-probability", type=float, default=defaults.gate_probability)
    parser.add_argument("--match-distance", type=float, default=defaults.match_distance, help="[m]")
    parser.add_argument("--duration", type=float, default=defaults.duration, help="[s]")
    parser.add_argument("--burn-in", type=float, default=defaults.burn_in, help="[s]")
    parser.add_argument("--no-camera", action="store_true", help="radar-only tracker")
    parser.add_argument("--workers", type=int, default=1, help="worker processes")
    parser.add_argument("--out", type=Path, default=Path("results"), help="plot directory")
    parser.add_argument("--no-plots", action="store_true", help="print the tables only")
    return parser.parse_args()


def settings_caption(config: MttConfig) -> str:
    """Settings shared by every row of a table, printed above it."""
    lifecycle = config.lifecycle
    return (
        f"match distance {config.match_distance:g} m (same at every row), "
        f"{lifecycle.confirm_hits}-of-{lifecycle.confirm_window} confirmation, "
        f"K={lifecycle.max_misses}, gate {100 * config.gate_probability:g}%, "
        f"base radar Pd {config.radar_pd:g} / clutter {config.radar_clutter_rate:g}, "
        f"camera Pd {config.camera_pd:g} / clutter {config.camera_clutter_rate:g}"
    )


def run_sweep(kind: str, config: MttConfig, args: argparse.Namespace, seeds: range) -> SweepResult:
    parameter = SENSOR_PREFIX[args.sweep_sensors] + {"clutter": "clutter_rate", "pd": "pd"}[kind]
    if kind == "pd":
        values = args.pds
    elif args.clutter_rates is not None:
        values = args.clutter_rates
    elif args.sweep_sensors == "camera":
        values = DEFAULT_CAMERA_CLUTTER_RATES
    else:
        values = DEFAULT_RADAR_CLUTTER_RATES
    print()
    print(f"=== sweep of {parameter} over {list(values)} ({len(seeds)} seeds each) ===")
    start = time.time()
    result = sweep(SCENARIOS[args.scenario], config, parameter, values, seeds, args.workers)
    print(f"({time.time() - start:.0f} s)")
    for line in format_sweep_table(
        result, value_label=VALUE_LABELS[kind], caption=settings_caption(config)
    ):
        print(line)
    return result


def run_compare(config: MttConfig, args: argparse.Namespace, seeds: range) -> SweepResult:
    print()
    print(f"=== radar-only vs fused tracker ({len(seeds)} seeds, identical simulations) ===")
    start = time.time()
    result = compare_fusion(SCENARIOS[args.scenario], config, seeds, args.workers)
    print(f"({time.time() - start:.0f} s)")
    for line in format_sweep_table(
        result,
        value_label="tracker",
        caption=settings_caption(config),
        row_labels=["radar-only", "fused"],
    ):
        print(line)
    return result


def main() -> None:
    args = parse_args()
    config = MttConfig(
        duration=args.duration,
        radar_pd=args.radar_pd,
        camera_pd=args.camera_pd,
        radar_clutter_rate=args.radar_clutter_rate,
        camera_clutter_rate=args.camera_clutter_rate,
        velocity_std=args.velocity_std,
        gate_probability=args.gate_probability,
        lifecycle=LifecycleConfig(args.confirm_hits, args.confirm_window, args.max_misses),
        use_camera=not args.no_camera,
        match_distance=args.match_distance,
        burn_in=args.burn_in,
    )
    seeds = range(args.seeds)
    lifecycle = config.lifecycle
    print(
        f"scenario '{args.scenario}' ({len(SCENARIOS[args.scenario])} targets), "
        f"{len(seeds)} seeds, "
        f"{config.duration:g} s, dt={config.dt:g} s, radar every {config.radar_every} steps, "
        f"camera {'every step' if config.use_camera else 'off'}, burn-in {config.burn_in:g} s"
    )
    print(
        f"lifecycle: {lifecycle.confirm_hits}-of-{lifecycle.confirm_window} confirmation, "
        f"deleted after more than {lifecycle.max_misses} consecutive misses; "
        f"gate {100 * config.gate_probability:g}%; match distance {config.match_distance:g} m"
    )
    print(
        "metrics (after burn-in): rmse = position error of matched confirmed tracks; ghost/step = "
        "confirmed tracks without a\ntarget per step; missed = share of (target, step) pairs in "
        "view without a confirmed track; delay = mean time to the\nfirst match; confirmed = share "
        "of targets ever matched; id sw = identity changes per run; births/scan = tracks\nstarted "
        "per radar scan (diagnostic). Cells are mean +- half-width of the 95% t interval across "
        "seeds;\nthe 'med' columns are medians across seeds (rmse, missed and id switches are "
        "heavy-tailed); valid seeds = seeds with a defined rmse / delay."
    )

    results = {}
    if "clutter" in args.sweeps:
        # Ghosts come from radar clutter; a camera sweep keeps the radar rate at its base value.
        swept_radar = args.sweep_sensors in ("both", "radar")
        rates = [r for r in (args.clutter_rates or DEFAULT_RADAR_CLUTTER_RATES) if r > 0.0]
        print()
        for line in clutter_gate_report(
            config, rates if swept_radar else [config.radar_clutter_rate]
        ):
            print(line)
        results["clutter"] = run_sweep("clutter", config, args, seeds)
    if "pd" in args.sweeps:
        results["pd"] = run_sweep("pd", config, args, seeds)
    if "compare" in args.sweeps:
        run_compare(config, args, seeds)

    if not args.no_plots:
        from fusion.mtt_plots import plot_sweep

        print()
        for result in results.values():
            name = f"mtt_{args.scenario}_{result.parameter}.png"
            title = f"scenario '{args.scenario}', {len(seeds)} seeds, 95% interval across seeds"
            print(f"saved {plot_sweep(result, args.out / name, title=title)}")


if __name__ == "__main__":
    main()
