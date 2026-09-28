"""Compare raw radar, radar-only EKF and radar + camera fusion EKF over many seeds.

Usage:
    python scripts/run_fusion_experiment.py [--seeds 50] [--burn-in 5.0] ...
"""

import argparse

import numpy as np

from fusion.experiment import (
    MASKS,
    METHODS,
    SCENARIOS,
    FusionConfig,
    evaluation_masks,
    make_rngs,
    run_trial,
    simulate,
)

METHOD_LABELS = {"raw": "raw radar", "radar_only": "radar-only EKF", "fusion": "fusion EKF"}


def parse_args() -> argparse.Namespace:
    defaults = FusionConfig()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=50, help="number of seeds (0..N-1)")
    parser.add_argument("--dt", type=float, default=defaults.dt, help="time step [s]")
    parser.add_argument("--duration", type=float, default=defaults.duration, help="[s]")
    parser.add_argument("--radar-every", type=int, default=defaults.radar_every)
    parser.add_argument("--range-std", type=float, default=defaults.radar_range_std, help="[m]")
    parser.add_argument("--radar-bearing-std-deg", type=float, default=2.0, help="[deg]")
    parser.add_argument("--camera-bearing-std-deg", type=float, default=0.1, help="[deg]")
    parser.add_argument("--accel-std", type=float, default=defaults.accel_std, help="[m/s^2]")
    parser.add_argument("--velocity-std", type=float, default=defaults.velocity_std, help="[m/s]")
    parser.add_argument("--burn-in", type=float, default=defaults.burn_in, help="[s]")
    return parser.parse_args()


def format_stat(values: np.ndarray) -> str:
    if np.all(np.isnan(values)):
        return "n/a"
    return f"{values.mean():7.2f} ± {values.std(ddof=1):5.2f}"


def main() -> None:
    args = parse_args()
    config = FusionConfig(
        dt=args.dt,
        duration=args.duration,
        radar_every=args.radar_every,
        radar_range_std=args.range_std,
        radar_bearing_std=float(np.deg2rad(args.radar_bearing_std_deg)),
        camera_bearing_std=float(np.deg2rad(args.camera_bearing_std_deg)),
        accel_std=args.accel_std,
        velocity_std=args.velocity_std,
        burn_in=args.burn_in,
    )
    seeds = range(args.seeds)
    masks = evaluation_masks(config.n_steps + 1, config.radar_every, config.burn_in_steps)
    n_all = int(masks["all"].sum())
    n_radar = int(masks["radar"].sum())

    print(
        f"{len(seeds)} seeds, dt={config.dt} s, {config.duration} s, radar every "
        f"{config.radar_every} steps, camera every step, burn-in {config.burn_in} s"
    )
    print(
        f"RMSE over 'all steps' = {n_all} steps, 'radar steps' = {n_radar} steps "
        f"(k >= {config.burn_in_steps}); estimates are taken after all updates at step k"
    )
    print()
    header = f"{'scenario':<10}{'method':<18}{'RMSE all steps [m]':>22}{'RMSE radar steps [m]':>24}"
    print(header)
    print("-" * len(header))

    for name, initial_state in SCENARIOS.items():
        results = [run_trial(initial_state, config, make_rngs(seed)) for seed in seeds]
        rmse = {
            method: {mask: np.array([r[method][mask] for r in results]) for mask in MASKS}
            for method in METHODS
        }
        # Re-simulating with the same seeds reproduces the same trajectories.
        ranges = np.concatenate(
            [np.hypot(*simulate(initial_state, config, make_rngs(s)).truth[:, :2].T) for s in seeds]
        )

        for method in METHODS:
            print(
                f"{name:<10}{METHOD_LABELS[method]:<18}"
                f"{format_stat(rmse[method]['all']):>22}{format_stat(rmse[method]['radar']):>24}"
            )
        wins = {
            mask: int(np.sum(rmse["fusion"][mask] < rmse["radar_only"][mask])) for mask in MASKS
        }
        win_all = f"{wins['all']}/{len(seeds)}"
        win_radar = f"{wins['radar']}/{len(seeds)}"
        print(f"{'':<10}{'fusion win rate':<18}{win_all:>22}{win_radar:>24}")
        print(f"{'':<10}{'true range [m]':<18}{f'{ranges.min():.0f} .. {ranges.max():.0f}':>22}")
        print()


if __name__ == "__main__":
    main()
