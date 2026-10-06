"""Diagnostic of the maneuver sweep (separated layout, lambda 0): where and when targets are missed.

Usage:
    python scripts/diag_maneuver.py [--seeds 10] [--out results/diag_maneuver.txt]
        [--prepend FILE]

Runs the rows neutral, turn 5 and 20 deg/s, acceleration 2 and 4 m/s^2 on a few seeds and prints
numbers only: (a) the share of target-steps outside the field of view, by range; (b) the share of
in-view target-steps without a matched confirmed track in time bins (at the Phase 6 match
distance and at the wide diagnostic distance); (c) confirmed tracks and track births per target.
The text of --prepend, if given, is written first.
"""

import argparse
from pathlib import Path

import numpy as np

from fusion.mtt_experiment import SCENARIOS
from fusion.mtt_metrics import match_tracks
from fusion.robustness_experiment import RobustnessConfig, maneuver_rows, simulate_and_track

ROWS = ("neutral", "turn 5 deg/s", "turn 20 deg/s", "acceleration 2 m/s^2", "acceleration 4 m/s^2")
BIN = 5.0


def diagnose(label: str, config: RobustnessConfig, states: np.ndarray, seeds: range) -> list[str]:
    """Lines of the three sections for one row, pooled over the seeds."""
    n_targets = len(states)
    n_bins = int(np.ceil(config.duration / BIN))
    outside_far = outside_near = total = 0
    lowest, highest = np.inf, 0.0
    missed = {"50": np.zeros(n_bins), "200": np.zeros(n_bins)}
    counted = np.zeros(n_bins)
    confirmed, births, distinct = [], [], []
    for seed in seeds:
        sim, run = simulate_and_track(states, config, seed)
        truth = sim.truth
        n_steps = truth.shape[1]
        distance = np.hypot(truth[:, :, 0], truth[:, :, 1])
        outside_far += int((distance > config.fov.range_max).sum())
        outside_near += int((distance < config.fov.range_min).sum())
        total += distance.size
        lowest, highest = min(lowest, float(distance.min())), max(highest, float(distance.max()))
        in_view = config.fov.contains(truth[:, :, :2].reshape(-1, 2)).reshape(n_targets, n_steps)
        bins = np.minimum((np.arange(n_steps) * config.dt / BIN).astype(int), n_bins - 1)
        narrow = match_tracks(truth, run.history, config.match_distance)
        wide = match_tracks(truth, run.history, config.diagnostic_match_distance)
        for key, match in (("50", narrow), ("200", wide)):
            unmatched = in_view & (match.ids < 0)
            for b in range(n_bins):
                missed[key][b] += unmatched[:, bins == b].sum()
        for b in range(n_bins):
            counted[b] += in_view[:, bins == b].sum()
        after = np.arange(n_steps) >= config.burn_in_steps
        confirmed.append(float(narrow.n_confirmed[after].mean()) / n_targets)
        births.append(run.tracker.births / n_targets)
        distinct.append(len(set(narrow.ids[narrow.ids >= 0].tolist())) / n_targets)
    edges = [f"{b * BIN:g}-{min((b + 1) * BIN, config.duration):g}" for b in range(n_bins)]
    lines = [f"== {label}"]
    lines.append(
        f"(a) target-steps outside the field of view: range > {config.fov.range_max:g} m "
        f"{outside_far / total:.6f}, range < {config.fov.range_min:g} m "
        f"{outside_near / total:.6f} ({outside_far} + {outside_near} of {total}); "
        f"range min {lowest:.0f} m, max {highest:.0f} m"
    )
    lines.append(
        "(b) share of in-view target-steps without a matched confirmed track, per time bin [s]"
    )
    lines.append("    bin         " + " ".join(f"{e:>7}" for e in edges))
    for key, name in (("50", "50 m match "), ("200", "200 m match")):
        shares = missed[key] / np.maximum(counted, 1)
        lines.append(f"    {name} " + " ".join(f"{s:7.4f}" for s in shares))
    lines.append(
        "(c) per target, mean over seeds: "
        f"confirmed tracks (after burn-in) {np.mean(confirmed):.4f}, "
        f"track births {np.mean(births):.3f}, "
        f"distinct track ids matched at 50 m {np.mean(distinct):.3f}"
    )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("results/diag_maneuver.txt"))
    parser.add_argument("--prepend", type=Path, default=None)
    args = parser.parse_args()
    states = SCENARIOS["separated"]
    base = RobustnessConfig(radar_clutter_rate=0.0, camera_clutter_rate=0.0)
    rows = {row.label: row for row in maneuver_rows(base, len(states))}
    seeds = range(args.seeds)
    lines = [] if args.prepend is None else args.prepend.read_text(encoding="utf-8").splitlines()
    lines += [
        "",
        f"TARGETED RUN: separated layout, lambda 0, {args.seeds} seeds (0..{args.seeds - 1}), "
        f"{base.duration:g} s, burn-in {base.burn_in:g} s, match {base.match_distance:g} m / "
        f"{base.diagnostic_match_distance:g} m",
        "",
    ]
    for label in ROWS:
        lines += diagnose(label, rows[label].config, states, seeds)
        lines.append("")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.out, len(lines))


if __name__ == "__main__":
    main()
