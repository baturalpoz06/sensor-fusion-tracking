"""NEES consistency of the radar-only EKF and the radar + camera fusion EKF.

Usage:
    python scripts/run_consistency_experiment.py [--seeds 50] [--burn-in 5.0]
"""

import argparse

import numpy as np

from fusion.consistency import COMPONENTS, summarize_nees
from fusion.experiment import (
    FILTER_METHODS,
    SCENARIOS,
    FusionConfig,
    evaluation_masks,
    make_rngs,
    run_nees_trial,
)

METHOD_LABELS = {"radar_only": "radar-only EKF", "fusion": "fusion EKF"}
MASK_LABELS = {"all": "ALL STEPS", "radar": "RADAR STEPS ONLY"}
DIMS = {name: len(range(*part.indices(4))) for name, part in COMPONENTS.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=50, help="number of seeds (0..N-1)")
    parser.add_argument(
        "--burn-in", type=float, default=FusionConfig().burn_in, help="excluded initial time [s]"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = FusionConfig(burn_in=args.burn_in)
    n_seeds = args.seeds
    masks = evaluation_masks(config.n_steps + 1, config.radar_every, config.burn_in_steps)

    print(
        f"{n_seeds} seeds, dt={config.dt} s, {config.duration} s, radar every "
        f"{config.radar_every} steps, camera every step, burn-in {config.burn_in} s "
        f"(k >= {config.burn_in_steps})"
    )
    print(
        "NEES = e^T P^-1 e with e = truth - estimate, P = posterior covariance after all "
        "updates at step k."
    )
    print(
        f"At every step NEES is averaged over the {n_seeds} independent seeds; the 95% band "
        f"is chi2(N*d) quantiles / N."
    )
    print(
        "Steps are correlated in time, so the step-wise seed-averaged NEES is judged against "
        "the band step by step;\nthe overall mean is descriptive only and is NOT tested "
        "against the band."
    )
    print("For a consistent filter about 5% of the steps fall outside a 95% band by chance.")

    # nees[scenario][method][component] -> (n + 1,) NEES averaged over seeds.
    averaged = {}
    for name, initial_state in SCENARIOS.items():
        trials = [run_nees_trial(initial_state, config, make_rngs(seed)) for seed in range(n_seeds)]
        averaged[name] = {
            method: {
                comp: np.mean([t[method][comp] for t in trials], axis=0) for comp in COMPONENTS
            }
            for method in FILTER_METHODS
        }

    header = (
        f"{'scenario':<9}{'method':<16}{'part':<10}{'mean NEES':>10}{'expected':>10}"
        f"{'95% band':>17}{'inside %':>10}{'above %':>9}{'below %':>9}"
    )
    for mask_name, mask in masks.items():
        print()
        print(f"=== {MASK_LABELS[mask_name]} ({int(mask.sum())} steps) ===")
        print(header)
        print("-" * len(header))
        for scenario, by_method in averaged.items():
            for method in FILTER_METHODS:
                for comp in COMPONENTS:
                    summary = summarize_nees(
                        by_method[method][comp], DIMS[comp], n_seeds, mask
                    )
                    band = f"[{summary.lower:.2f}, {summary.upper:.2f}]"
                    print(
                        f"{scenario:<9}{METHOD_LABELS[method]:<16}{comp:<10}"
                        f"{summary.mean:>10.2f}{DIMS[comp]:>10d}{band:>17}"
                        f"{100 * summary.inside:>10.1f}{100 * summary.above:>9.1f}"
                        f"{100 * summary.below:>9.1f}"
                    )


if __name__ == "__main__":
    main()
