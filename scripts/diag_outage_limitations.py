"""Measure the two known limitations of the aware outage policy (Phase 7); descriptive only.

Usage:
    python scripts/diag_outage_limitations.py [--seeds 50] [--workers 8]
        [--out results/outage_limitations.txt]

L1, close targets coast without updates. The aware tracker deletes a track during a radar outage
once it has gone max_coast_time without any measurement update. The camera update that would
reset the clock is skipped while the bearing gates of two confirmed tracks overlap, so two close
real targets can both be deleted. Scene 'crossing' (targets 0 and 1 cross near 30 s) and, as a
negative control, 'separated'; no clutter; a radar outage from 20 s to 40 s; max coast 2, 5, 10
and 15 s, and a control row whose max coast (25 s) exceeds the outage, where no deletion is
planned. Without clutter every coast deletion is a real track; the counter does not say which
target it followed. For the mechanism, a run of the control row also reports, from the tracker's
camera log, the longest stretch of outage steps a confirmed track went without any update (the
highest value its coast clock reached) and the longest run of camera steps it was skipped as
ambiguous. The closest approach of targets 0 and 1 in the simulated truth is reported alongside.

L2, camera clutter resets the coast clock of a ghost. Scene 'vanishing' (the separated layout):
target 0 ceases to exist at 22 s, inside a radar outage from 20 s to 44 s; aware tracker with max
coast 5 and 10 s; clutter (radar, camera) of (0, 0), (5, 0), (0, 5) and (5, 5) points per scan.
ghost life = seconds the track of the vanished target survives.

Every row of a scene runs the same seeds; the outage, the tracker policy and the clutter are
applied to identically drawn scenes (clutter has its own random streams), so differences are
paired per seed. Cells are the mean over seeds +- the half width of the 95% Student t interval;
a "d" column is that interval of the per-seed difference against the row named in the table
(the same interval as fusion.mtt_metrics.paired_difference). The file starts with a provenance
header. No tracker code is changed or read back; nothing here is a verdict.
"""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from fusion.dropout import SingleOutage, apply_dropout
from fusion.mtt_experiment import SweepResult
from fusion.mtt_metrics import seed_confidence_interval
from fusion.mtt_report import format_cell
from fusion.mtt_simulation import make_mtt_rngs, simulate_mtt
from fusion.outage_experiment import OUTAGE_SCENARIOS, VANISHING_TARGET, OutageConfig, outage_sweep
from fusion.outage_report import format_table
from fusion.provenance import provenance_lines
from fusion.sensors.camera import CameraModel
from fusion.sensors.radar import RadarModel
from fusion.tracker.multi_target import run_multi_target_tracking

ROOT = Path(__file__).resolve().parents[1]
CONFIDENCE = 0.95
LABEL_WIDTH = 30
OUTAGE_START = 20.0
# L1: a radar outage covering the crossing of targets 0 and 1 (about 30 s).
L1_DURATION = 20.0
L1_COASTS = (2.0, 5.0, 10.0, 15.0)
L1_CONTROL = 25.0
L1_SCENES = ("crossing", "separated")
CLOSE_PAIR = (0, 1)
# L2: the setup of outage experiment f, with clutter.
L2_DURATION = 24.0
L2_VANISH_TIME = 22.0
L2_COASTS = (5.0, 10.0)
L2_CLUTTER = ((0.0, 0.0), (5.0, 0.0), (0.0, 5.0), (5.0, 5.0))
# The values stated in the docstrings of fusion.outage_experiment and fusion.outage_report.
STATED_GHOST_LIFE = {5.0: 7.0, 10.0: 20.9}


def base_config() -> OutageConfig:
    """Clean baseline of the outage experiments: no clutter, aware tracker dropping tentatives."""
    return replace(
        OutageConfig(),
        radar_clutter_rate=0.0,
        camera_clutter_rate=0.0,
        outage_policy="aware",
        aware_tentatives="drop",
    )


def l1_rows() -> list[tuple[str, OutageConfig]]:
    """(label, configuration) of L1; the last row is the control (max coast above the outage)."""
    if L1_CONTROL < L1_DURATION or max(L1_COASTS) >= L1_DURATION:
        raise ValueError("L1 needs every max coast below the outage and the control above it")
    outage = SingleOutage(("radar",), OUTAGE_START, L1_DURATION)
    base = replace(base_config(), dropout=outage)
    rows = [(f"max coast {c:g} s", replace(base, max_coast_time=c)) for c in L1_COASTS]
    rows.append((f"max coast {L1_CONTROL:g} s (control)", replace(base, max_coast_time=L1_CONTROL)))
    return rows


def l2_rows() -> list[tuple[str, float, tuple[float, float], OutageConfig]]:
    """(label, max coast, (radar, camera) clutter, configuration) of L2."""
    if not OUTAGE_START <= L2_VANISH_TIME < OUTAGE_START + L2_DURATION:
        raise ValueError("L2 needs the target to vanish inside the outage")
    outage = SingleOutage(("radar",), OUTAGE_START, L2_DURATION)
    base = replace(base_config(), dropout=outage, vanish=((VANISHING_TARGET, L2_VANISH_TIME),))
    rows = []
    for coast in L2_COASTS:
        for radar, camera in L2_CLUTTER:
            label = f"coast {coast:g} s, clutter {radar:g}/{camera:g}"
            config = replace(
                base, max_coast_time=coast, radar_clutter_rate=radar, camera_clutter_rate=camera
            )
            rows.append((label, coast, (radar, camera), config))
    return rows


def differences(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Per-seed differences of every row against one reference row; NaN where either is NaN."""
    return np.asarray(values, dtype=float) - np.asarray(reference, dtype=float)[np.newaxis, :]


def table(
    metrics: dict[str, np.ndarray], columns: dict, medians: dict, labels: list[str]
) -> list[str]:
    """format_table over per-seed arrays, one row per label."""
    result = SweepResult("row", np.arange(len(labels), dtype=float), metrics)
    valid = tuple(list(columns)[:2])
    return format_table(
        result, columns, medians, valid, labels, "row", None, CONFIDENCE, LABEL_WIDTH
    )


def l1_table(scene: str, labels: list[str], result: SweepResult) -> list[str]:
    """L1 scores of one scene and their differences against the control row (the last)."""
    m = result.metrics
    deletions = m["coast_deletions"]
    metrics = {
        "deletions": deletions,
        "two_or_more": np.where(np.isfinite(deletions), (deletions >= 2).astype(float), np.nan),
        "during_missed": m["during_missed_rate"],
        "after_missed": m["after_missed_rate"],
        "kept": m["identity_kept"],
        "d_during_missed": differences(m["during_missed_rate"], m["during_missed_rate"][-1]),
        "d_after_missed": differences(m["after_missed_rate"], m["after_missed_rate"][-1]),
        "d_kept": differences(m["identity_kept"], m["identity_kept"][-1]),
    }
    columns = {
        "deletions": ("coast del", 3),
        "two_or_more": ("share >= 2 del", 3),
        "during_missed": ("during missed", 3),
        "after_missed": ("after missed", 3),
        "kept": ("id kept", 3),
        "d_during_missed": ("d during missed", 3),
        "d_after_missed": ("d after missed", 3),
        "d_kept": ("d id kept", 3),
    }
    medians = {"deletions": "med coast del"}
    title = (
        f"L1, scene '{scene}': radar outage {OUTAGE_START:g}-{OUTAGE_START + L1_DURATION:g} s, "
        "no clutter, aware (drop); d = row - control, per seed"
    )
    return [title, *table(metrics, columns, medians, labels)]


def longest_stretches(camera_log: list, radar_down: np.ndarray) -> tuple[int, int]:
    """Longest run of radar-down steps without a camera update, and of steps skipped as ambiguous.

    Both are the maximum over the confirmed tracks the camera step saw (offered or skipped). In a
    radar outage a track is updated only by the camera, so the first number is the largest value
    its coast clock reached; it is deleted once that exceeds max coast.
    """
    no_update: dict[int, int] = {}
    skipped: dict[int, int] = {}
    best_no_update = best_skipped = 0
    for entry, down in zip(camera_log, radar_down, strict=True):
        if not down:
            no_update.clear()
            skipped.clear()
            continue
        updated = {track for track, _ in entry.assigned}
        for track in set(entry.usable) | set(entry.skipped) | set(no_update):
            no_update[track] = 0 if track in updated else no_update.get(track, 0) + 1
            best_no_update = max(best_no_update, no_update[track])
        for track in skipped:
            if track not in entry.skipped:
                skipped[track] = 0
        for track in entry.skipped:
            skipped[track] = skipped.get(track, 0) + 1
            best_skipped = max(best_skipped, skipped[track])
    return best_no_update, best_skipped


def coast_stretches(scene: str, config: OutageConfig, seeds: list[int]) -> tuple[np.ndarray, ...]:
    """Per seed, longest_stretches of one tracker run, in seconds."""
    states = OUTAGE_SCENARIOS[scene]
    no_update, skipped = [], []
    for seed in seeds:
        rngs = make_mtt_rngs(seed)
        sim = simulate_mtt(states, config, rngs)
        dropped = apply_dropout(sim, config.dropout.windows(config.dt, rngs["dropout"]), config.dt)
        radar_model = RadarModel(config.radar_range_std, config.radar_bearing_std)
        camera_model = CameraModel(config.camera_bearing_std)
        run = run_multi_target_tracking(
            dropped.sim, config.tracker_config(), radar_model, camera_model, dropped.radar_down
        )
        a, b = longest_stretches(run.tracker.camera_log, dropped.radar_down)
        no_update.append(a * config.dt)
        skipped.append(b * config.dt)
    return np.array(no_update), np.array(skipped)


def closest_approach(seeds: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Per seed, the minimum distance [m] of the close pair in 'crossing' and its time [s]."""
    config = base_config()
    states = OUTAGE_SCENARIOS["crossing"]
    distance, time = [], []
    for seed in seeds:
        truth = simulate_mtt(states, config, make_mtt_rngs(seed)).truth
        a, b = CLOSE_PAIR
        separation = np.hypot(*(truth[a, :, :2] - truth[b, :, :2]).T)
        step = int(np.argmin(separation))
        distance.append(float(separation[step]))
        time.append(step * config.dt)
    return np.array(distance), np.array(time)


def l2_table(rows: list, result: SweepResult) -> list[str]:
    """Ghost life per row and its difference against the clutter-free row of the same coast."""
    m = result.metrics
    life = m["ghost_lifetime"]
    reference = {coast: i for i, (_, coast, clutter, _) in enumerate(rows) if clutter == (0.0, 0.0)}
    d_life = np.array([life[i] - life[reference[coast]] for i, (_, coast, _, _) in enumerate(rows)])
    metrics = {
        "life": life,
        "d_life": d_life,
        "censored": m["ghost_censored"],
        "deletions": m["coast_deletions"],
    }
    columns = {
        "life": ("ghost life [s]", 3),
        "d_life": ("d ghost life [s]", 3),
        "censored": ("ghost censored", 3),
        "deletions": ("coast del", 3),
    }
    medians = {"life": "med life [s]"}
    title = (
        f"L2, scene 'vanishing': target {VANISHING_TARGET} vanishes at {L2_VANISH_TIME:g} s inside "
        f"a radar outage {OUTAGE_START:g}-{OUTAGE_START + L2_DURATION:g} s, aware (drop); "
        "clutter = radar/camera points per scan; d = row - clutter 0/0 at the same max coast"
    )
    return [title, *table(metrics, columns, medians, [label for label, *_ in rows])]


def stated_check(rows: list, result: SweepResult) -> list[str]:
    """Measured mean ghost life, at one decimal, next to the value the docstrings state."""
    lines = ["docstring check (stated: ghost life 7.0 s and 20.9 s at max coast 5 and 10 s, camera"]
    lines.append("  clutter 5 per scan); measured mean at one decimal:")
    life = result.metrics["ghost_lifetime"]
    for i, (label, coast, clutter, _) in enumerate(rows):
        if clutter[1] == 0.0:
            continue
        mean = seed_confidence_interval(life[i], CONFIDENCE).mean
        stated = STATED_GHOST_LIFE[coast]
        verdict = "same" if round(mean, 1) == stated else "differs"
        lines.append(
            f"  {label:<26} measured {mean:.3f} s -> {mean:.1f} s, stated {stated:.1f} s: {verdict}"
        )
    return lines


def build_report(seeds: list[int], workers: int) -> list[str]:
    """Run both measurements and return the report lines."""
    lines = [
        f"OUTAGE LIMITATIONS (Phase 7): {len(seeds)} seeds ({seeds[0]}..{seeds[-1]}); "
        "descriptive, no verdicts.",
        "cells: mean over seeds +- half width of the 95% t interval; 'share >= 2 del' = share of",
        "  seeds with at least two coast deletions; id kept and missed as in the outage tables.",
        "",
    ]
    labels = [label for label, _ in l1_rows()]
    configs = [config for _, config in l1_rows()]
    for scene in L1_SCENES:
        result = outage_sweep(
            OUTAGE_SCENARIOS[scene], configs, list(range(len(configs))), "row", seeds, workers
        )
        lines += l1_table(scene, labels, result)
        no_update, skipped = coast_stretches(scene, configs[-1], seeds)
        lines.append(
            f"  control run, longest stretch without any update of a confirmed track during the "
            f"outage: {format_cell(no_update, 3, CONFIDENCE)} s (max over seeds "
            f"{no_update.max():.3g} s); longest run of camera steps skipped as ambiguous: "
            f"{format_cell(skipped, 3, CONFIDENCE)} s (max {skipped.max():.3g} s)"
        )
        lines.append("")
    distance, time = closest_approach(seeds)
    lines.append(
        f"closest approach of targets {CLOSE_PAIR[0]} and {CLOSE_PAIR[1]} in 'crossing' (truth): "
        f"distance {format_cell(distance, 3, CONFIDENCE)} m (max {distance.max():.3g}), "
        f"time {format_cell(time, 3, CONFIDENCE)} s"
    )
    lines.append("")
    rows = l2_rows()
    configs = [config for *_, config in rows]
    result = outage_sweep(
        OUTAGE_SCENARIOS["vanishing"], configs, list(range(len(configs))), "row", seeds, workers
    )
    lines += [*l2_table(rows, result), "", *stated_check(rows, result)]
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=50, help="number of seeds (0..N-1)")
    parser.add_argument("--workers", type=int, default=1, help="worker processes")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "outage_limitations.txt")
    args = parser.parse_args()
    if args.seeds < 2:
        sys.exit("--seeds must be at least 2 (a t interval needs two seeds)")
    lines = build_report(list(range(args.seeds)), args.workers)
    text = "\n".join([*provenance_lines(sys.argv), *lines]) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print("\n".join(lines))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
