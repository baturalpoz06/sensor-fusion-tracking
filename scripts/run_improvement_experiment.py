"""Phase 8b: the IMM motion model and the camera-bias estimate against the Phase 8a tracker.

Usage:
    python scripts/run_improvement_experiment.py --stage tune --workers 12
    python scripts/run_improvement_experiment.py --stage timing
    python scripts/run_improvement_experiment.py --stage eval --pilot --workers 12
    python scripts/run_improvement_experiment.py --stage eval --workers 12

Stages:
    tune    every candidate parameter of the arms on the tuning seeds (1000-1019), the tuned
            maneuver rows and the bias rows only; applies the pre-registered selection rules and
            prints the frozen-parameter expression for improvement_experiment.FROZEN. Nothing
            here uses the evaluation seeds.
    timing  tracker runtime of every arm, single process, seeds 1000-1004.
    eval    the evaluation: seeds 0-49, both layouts, clutter 0 and 5, the groups maneuver, bias
            and hard; writes the tables, results/summary_8b_headline.txt and the mechanical
            verdicts results/criteria_8b.txt. It refuses to run unless FROZEN is set, was frozen
            at an ancestor of HEAD and the working tree has no uncommitted tracked changes.
            With --pilot it runs 5 tuning seeds into <out>/pilot with placeholder parameters (a
            does-it-run check, no conclusions) and has no such conditions.

All arms of a row see the same simulated data (common random numbers); paired differences compare
each arm with the baseline "EKF" seed by seed. No file this script writes interprets a result.
"""

import argparse
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

from fusion import improvement_experiment as experiment
from fusion.improvement_criteria import (
    evaluate_all,
    power_lines,
    select_bias_prior,
    select_family,
    winnable_blocks,
)
from fusion.improvement_experiment import (
    BASELINE,
    BIN_WIDTH,
    DEG,
    EVALUATION_SEEDS,
    PILOT_SEEDS,
    TIMING_SEEDS,
    TUNING_SEEDS,
    UNFROZEN_DEFAULTS,
    Arm,
    BlockResult,
    Parameters,
    evaluation_blocks,
    maneuver_block_rows,
    named_arms,
    run_blocks,
    tuning_blocks,
    tuning_candidates,
)
from fusion.improvement_metrics import bin_edges
from fusion.improvement_report import (
    PILOT_BANNER,
    UNFROZEN_BANNER,
    criteria_lines,
    format_block,
    headline_lines,
    parameters_text,
    timing_lines,
    tuning_lines,
)
from fusion.mtt_experiment import SCENARIOS
from fusion.robustness_experiment import RobustnessConfig
from fusion.tracker.camera_bias import CameraBiasConfig

GROUPS = ("maneuver", "bias", "hard")
ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    defaults = RobustnessConfig()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", choices=("tune", "timing", "eval"), required=True)
    parser.add_argument(
        "--pilot", action="store_true", help="with --stage eval: 5 tuning seeds, no conclusions"
    )
    parser.add_argument(
        "--seeds", type=int, default=None, help="tune / timing / pilot: number of seeds"
    )
    parser.add_argument(
        "--layouts", nargs="+", choices=sorted(SCENARIOS), default=sorted(SCENARIOS)
    )
    parser.add_argument("--clutter-rates", type=float, nargs="+", default=(0.0, 5.0))
    parser.add_argument("--duration", type=float, default=defaults.duration, help="[s]")
    parser.add_argument("--burn-in", type=float, default=defaults.burn_in, help="[s]")
    parser.add_argument("--workers", type=int, default=1, help="worker processes")
    parser.add_argument("--out", type=Path, default=Path("results"), help="output directory")
    parser.add_argument(
        "--small-grid", action="store_true", help="tune: a few candidates (machinery check only)"
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.stage == "eval" and not args.pilot:
        fixed = (
            args.seeds is None
            and sorted(args.layouts) == sorted(SCENARIOS)
            and tuple(args.clutter_rates) == (0.0, 5.0)
            and args.duration == defaults.duration
            and args.burn_in == defaults.burn_in
        )
        if not fixed:
            parser.error("the evaluation takes no --seeds, --layouts, --clutter-rates, "
                         "--duration or --burn-in (use --pilot to try other settings)")
    return args


def base_config(args: argparse.Namespace):
    def make(clutter: float) -> RobustnessConfig:
        return RobustnessConfig(
            duration=args.duration,
            burn_in=args.burn_in,
            radar_clutter_rate=clutter,
            camera_clutter_rate=clutter,
        )

    return make


def scenes(args: argparse.Namespace) -> dict[str, np.ndarray]:
    return {layout: SCENARIOS[layout] for layout in args.layouts}


def git(*arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *arguments], cwd=ROOT, capture_output=True, text=True, check=False
    )


def head_commit() -> str:
    result = git("rev-parse", "HEAD")
    return result.stdout.strip() if result.returncode == 0 else ""


def refusal_reason(parameters: Parameters | None) -> str | None:
    """Why the evaluation must not run now, or None if it may."""
    if parameters is None:
        return "improvement_experiment.FROZEN is not set: run --stage tune and freeze first"
    if not parameters.commit:
        return "the frozen parameters carry no commit hash"
    head = head_commit()
    if not head:
        return "cannot read the git commit (not a repository?)"
    ancestor = git("merge-base", "--is-ancestor", parameters.commit, "HEAD")
    if ancestor.returncode != 0:
        return f"the frozen commit {parameters.commit[:10]} is not an ancestor of HEAD"
    status = git("status", "--porcelain", "--untracked-files=no")
    if status.stdout.strip():
        return "the working tree has uncommitted changes to tracked files"
    return None


def write(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path}")


def save(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(obj, handle)


def edges_seconds(args: argparse.Namespace) -> list[tuple[float, float]]:
    n_steps = round(args.duration / 0.1) + 1
    bins = bin_edges(n_steps, 0.1, args.burn_in, BIN_WIDTH)
    return [(a * 0.1, min(b, n_steps - 1) * 0.1) for a, b in bins]


def subset(block: BlockResult, groups: tuple[str, ...]) -> BlockResult:
    return block._replace(rows=[r for r in block.rows if r.group in groups])


def arm_parameters(arm: Arm) -> dict:
    """The tuned values of a selected candidate arm."""
    if arm.accel_std is not None:
        return {"ekf_high_accel_std": arm.accel_std}
    if arm.modes is not None and len(arm.modes) == 2:
        return {"imm_a_high_accel_std": arm.modes[1].accel_std}
    if arm.modes is not None:
        return {
            "imm_b_omega_deg": arm.modes[1].omega / DEG,
            "imm_b_accel_std": arm.modes[1].accel_std,
        }
    prior = 1.0 if arm.camera_bias.mode == "always" else arm.camera_bias.prior_h1
    return {"bias_prior": prior}


def stage_tune(args: argparse.Namespace) -> None:
    seeds = TUNING_SEEDS if args.seeds is None else TUNING_SEEDS[: args.seeds]
    blocks = tuning_blocks(base_config(args), scenes(args), args.clutter_rates, args.small_grid)
    n_rows = sum(len(b.rows) for b in blocks)
    print(f"tuning: {len(blocks)} blocks, {n_rows} rows, {len(seeds)} seeds (1000-...)")
    results = run_blocks(blocks, seeds, args.workers)
    save(args.out / "improvement_tune.pkl", results)

    maneuver = [subset(b, ("maneuver",)) for b in results]
    bias = [subset(b, ("bias",)) for b in results]
    candidates = tuning_candidates(args.small_grid)
    by_name = {arm.name: arm for family in candidates.values() for arm in family}
    selections, chosen = {}, {}
    for family in ("EKF high-Q", "IMM-A", "IMM-B"):
        names = [arm.name for arm in candidates[family]]
        selections[family] = select_family(maneuver, names)
        chosen.update(arm_parameters(by_name[selections[family].name]))
    names = [arm.name for arm in candidates["bias"]]
    selections["bias prior"] = select_bias_prior(maneuver, bias, names)
    chosen.update(arm_parameters(by_name[selections["bias prior"].name]))
    winnable = winnable_blocks([subset(b, ("maneuver", "held-out")) for b in results])
    commit = head_commit()
    dirty = git("status", "--porcelain", "--untracked-files=no").stdout.strip()
    parameters = Parameters(
        **chosen, winnable=winnable, commit=commit + ("+dirty" if dirty else "")
    )
    power = power_lines(maneuver, [a.name for fam in ("EKF high-Q", "IMM-A", "IMM-B")
                                   for a in candidates[fam]])
    sensitivity = bias_sensitivity(args, parameters.bias_prior, seeds)
    write(
        args.out / "improvement_tune.txt",
        tuning_lines(selections, parameters, winnable, power, sensitivity),
    )
    print("FROZEN = " + parameters_text(parameters))


def bias_sensitivity(args: argparse.Namespace, prior: float, seeds) -> list[str]:
    """Mean run RMSE of the bias estimate for three process noise levels (descriptive)."""
    base = CameraBiasConfig().process_noise
    arms = [Arm(BASELINE)]
    for factor in (0.0, 1.0, 100.0):
        config = (
            CameraBiasConfig(mode="always", process_noise=base * factor)
            if prior >= 1.0
            else CameraBiasConfig(prior_h1=prior, process_noise=base * factor)
        )
        arms.append(Arm(f"bias q x{factor:g}", camera_bias=config))
    blocks = []
    for layout, states in scenes(args).items():
        for clutter in args.clutter_rates:
            config = base_config(args)(clutter)
            rows = experiment.bias_block_rows(config, states, arms, with_oracle=False)
            blocks.append(experiment.Block(f"{layout} {clutter:g}", layout, clutter, rows))
    lines = []
    for block in run_blocks(blocks, seeds, args.workers):
        for row in block.rows:
            cells = ", ".join(
                f"{arm} {np.nanmean(row.metrics['run_position_rmse'][a]):.3f} m"
                for a, arm in enumerate(row.arm_names)
            )
            lines.append(f"{block.title} | {row.label}: mean run rmse {cells}")
    return lines


def stage_timing(args: argparse.Namespace) -> None:
    arms = list(named_arms(experiment.FROZEN or UNFROZEN_DEFAULTS).values())
    config = base_config(args)(5.0)
    states = SCENARIOS["separated"]
    wanted = ("neutral", "turn 10 deg/s")
    rows = [r for r in maneuver_block_rows(config, states, arms) if r.label in wanted]
    print(f"timing: {len(rows)} rows, {len(arms)} arms, one process")
    seeds = TIMING_SEEDS if args.seeds is None else TIMING_SEEDS[: args.seeds]
    results = experiment.improvement_sweep(rows, seeds, workers=1)
    lines = ["TIMING RUN (single process)", "", *timing_lines(results)]
    write(args.out / "improvement_timing.txt", lines)


def stage_eval(args: argparse.Namespace) -> None:
    if args.pilot:
        parameters = experiment.FROZEN or UNFROZEN_DEFAULTS
        seeds = TUNING_SEEDS[: args.seeds or len(PILOT_SEEDS)]
        out = args.out / "pilot"
        banners = [PILOT_BANNER] + ([UNFROZEN_BANNER] if experiment.FROZEN is None else [])
    else:
        reason = refusal_reason(experiment.FROZEN)
        if reason is not None:
            print(f"refusing to run the evaluation: {reason}", file=sys.stderr)
            sys.exit(2)
        parameters, seeds, out, banners = experiment.FROZEN, EVALUATION_SEEDS, args.out, []
    edges = edges_seconds(args)
    results: dict[str, list[BlockResult]] = {}
    for group in GROUPS:
        blocks = evaluation_blocks(
            group, base_config(args), scenes(args), args.clutter_rates, parameters
        )
        rows = sum(len(b.rows) for b in blocks)
        print(f"{group}: {len(blocks)} blocks, {rows} rows, {len(seeds)} seeds")
        results[group] = run_blocks(blocks, seeds, args.workers)
        lines = [*banners, f"seeds {seeds[0]}-{seeds[-1]}, {len(seeds)} in all", ""]
        for block in results[group]:
            lines += format_block(block, edges) + [""]
        write(out / f"improvement_{group}.txt", lines)
    save(out / "improvement_results.pkl", results)
    write(out / "summary_8b_headline.txt", headline_lines(results, banners))
    header = [
        *banners,
        f"commit {head_commit()}; seeds {seeds[0]}-{seeds[-1]} ({len(seeds)}); parameters:",
        parameters_text(parameters),
    ]
    criteria = evaluate_all(results["maneuver"], results["bias"], results["hard"], parameters)
    write(out / "criteria_8b.txt", criteria_lines(criteria, header))


def main() -> None:
    args = parse_args()
    if args.stage == "tune":
        stage_tune(args)
    elif args.stage == "timing":
        stage_timing(args)
    else:
        stage_eval(args)


if __name__ == "__main__":
    main()
