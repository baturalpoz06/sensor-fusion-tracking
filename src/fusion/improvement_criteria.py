"""Tuning selection and the pre-registered success criteria of Phase 8b, as code.

Everything here is fixed before the evaluation run and works on the arrays of the sweep, so the
verdicts are mechanical. A comparison is always a per-seed paired difference "arm - baseline" on
the same row and the same seeds (see fusion.mtt_metrics.paired_difference); lower is better for
every score used here.

"Improves" / "worsens": the 95% interval of the paired difference excludes zero in that
direction AND the mean difference is at least the practical threshold. Thresholds (the larger of
a fixed floor and, for the error and miss scores, a quarter of what the baseline loses against
the neutral row):
    missed rate  max(0.01, 25% of the baseline's window degradation)
    rmse, cross  max(1.0 m, 25% of the baseline's degradation)
    ghost rate   0.02 per step           id switches  0.5 per run
Neutral margins (equivalence of the whole interval for the bias estimate, one-sided upper bound
for the IMM arms): rmse 0.3 m, missed 0.005, ghost 0.01 per step, id switches 0.2 per run.
"""

import math
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
from scipy.stats import t as student_t

from fusion.improvement_experiment import (
    BASELINE,
    BlockResult,
    Parameters,
    RowResult,
)
from fusion.mtt_metrics import PairedDifference, paired_difference

MARGINS = {
    "run_position_rmse": 0.3,
    "run_missed_rate": 0.005,
    "run_ghost_rate": 0.01,
    "run_id_switches": 0.2,
}
# kind -> (floor, fraction of the baseline degradation or None)
THRESHOLDS = {
    "missed": (0.01, 0.25),
    "rmse": (1.0, 0.25),
    "ghost": (0.02, None),
    "id": (0.5, None),
}
B3_BIAS_BOUND_DEG = 0.05
MISSED_RATIO_NEEDED = 2.0 / 3.0
DRAG_BINS = (4, 5, 6, 7)  # the 5 s bins of [25, 45) s when bins start at the 5 s burn-in
DRAG_REDUCTION_NEEDED = 0.5
LATE_SEEDS_FROM = 10  # H-drag is also reported on the seeds that did not generate it


class Criterion(NamedTuple):
    """One verdict with the numbers behind it.

    Attributes:
        name: Identifier and short description.
        verdict: PASS, FAIL, UNDERPOWERED, N/A, SUPPORTED, REFUTED, PARTIAL or REPORTED.
        lines: The numbers, one block per line.
    """

    name: str
    verdict: str
    lines: list[str]


# --- paired differences ------------------------------------------------------------------


def arm_index(row: RowResult, arm: str) -> int:
    """Position of an arm in a row result."""
    return row.arm_names.index(arm)


def paired(row: RowResult, metric: str, arm: str, reference: str = BASELINE) -> PairedDifference:
    """Per-seed difference arm - reference of one score on one row."""
    values = row.metrics[metric]
    return paired_difference(values[arm_index(row, arm)], values[arm_index(row, reference)])


def paired_cs(row: RowResult, arm: str, kind: str) -> PairedDifference:
    """Per-seed difference of the common-support RMSE ('run' or 'window'): arm - baseline."""
    index = arm_index(row, arm)
    return paired_difference(
        row.metrics[f"cs_{kind}_rmse"][index], row.metrics[f"cs_{kind}_rmse_ref"][index]
    )


def across_rows(row: RowResult, neutral: RowResult, metric: str, arm: str) -> PairedDifference:
    """Per-seed difference row - neutral row of one score of one arm (same seeds)."""
    return paired_difference(
        row.metrics[metric][arm_index(row, arm)],
        neutral.metrics[metric][arm_index(neutral, arm)],
    )


def practical_threshold(kind: str, degradation: float = 0.0) -> float:
    """The smallest difference that counts: a floor, or a share of the baseline's degradation."""
    floor, fraction = THRESHOLDS[kind]
    if fraction is None or not math.isfinite(degradation):
        return floor
    return max(floor, fraction * max(degradation, 0.0))


def judge(difference: PairedDifference, threshold: float) -> str:
    """"improves", "worsens", "neither" or "undefined" (fewer than two valid seeds)."""
    if difference.n_valid < 2:
        return "undefined"
    if difference.upper < 0.0 and difference.mean <= -threshold:
        return "improves"
    if difference.lower > 0.0 and difference.mean >= threshold:
        return "worsens"
    return "neither"


def _half_width(difference: PairedDifference) -> float:
    return (difference.upper - difference.lower) / 2.0


def equivalence(difference: PairedDifference, margin: float) -> str:
    """PASS if the whole interval lies within +-margin; UNDERPOWERED if the mean does but the
    interval is wider than the margin (it cannot show equivalence); otherwise FAIL."""
    if difference.n_valid < 2:
        return "UNDERPOWERED"
    if difference.lower >= -margin and difference.upper <= margin:
        return "PASS"
    if abs(difference.mean) <= margin and _half_width(difference) > margin:
        return "UNDERPOWERED"
    return "FAIL"


def non_inferior(difference: PairedDifference, margin: float) -> str:
    """PASS if the upper interval bound is at most the margin; UNDERPOWERED if only the mean is
    and the interval is wider than the margin; otherwise FAIL."""
    if difference.n_valid < 2:
        return "UNDERPOWERED"
    if difference.upper <= margin:
        return "PASS"
    if difference.mean <= margin and _half_width(difference) > margin:
        return "UNDERPOWERED"
    return "FAIL"


def violation(difference: PairedDifference, margin: float, two_sided: bool) -> float:
    """How far the interval leaves the margin, in margins (0 if it stays inside)."""
    if difference.n_valid < 2:
        return float("inf")
    excess = max(0.0, difference.upper - margin)
    if two_sided:
        excess = max(excess, -margin - difference.lower)
    return excess / margin


def projected_half_width(sd: float, n_seeds: int, confidence: float = 0.95) -> float:
    """Half width of the t interval of a paired mean with the given SD at n_seeds seeds."""
    return float(student_t.ppf(0.5 + confidence / 2.0, df=n_seeds - 1)) * sd / math.sqrt(n_seeds)


def power_lines(
    blocks: Sequence[BlockResult], arms: Sequence[str], n_seeds: int = 50
) -> list[str]:
    """Projected interval half widths at n_seeds seeds against the neutral margins.

    Uses the paired SD of the arms on the neutral rows of the given (tuning) results. A margin
    narrower than the half width cannot be shown by the evaluation.
    """
    lines = []
    for block in blocks:
        row = _row(block, "neutral")
        for arm in arms:
            if arm not in row.arm_names:
                continue
            cells = []
            for metric, margin in MARGINS.items():
                diff = paired(row, metric, arm)
                if diff.n_valid < 2:
                    cells.append(f"{metric} n/a")
                    continue
                values = row.metrics[metric]
                gap = values[arm_index(row, arm)] - values[arm_index(row, BASELINE)]
                sd = float(np.nanstd(gap, ddof=1))
                half = projected_half_width(sd, n_seeds)
                flag = " WIDER" if half > margin else ""
                cells.append(f"{metric} {half:.3g} vs {margin:g}{flag}")
            lines.append(f"{block.title} | {arm}: " + "; ".join(cells))
    return lines


# --- block bookkeeping -------------------------------------------------------------------


def _row(block: BlockResult, label: str) -> RowResult:
    for row in block.rows:
        if row.label == label:
            return row
    raise KeyError(f"{block.title} has no row {label!r}")


def neutral_label(row: RowResult) -> str:
    """Label of the neutral row that a maneuver row is compared with."""
    return "held-out neutral" if row.group == "held-out" else "neutral"


def block_label(block: BlockResult, row: RowResult) -> str:
    """Identifier of a row-block: "layout | clutter | row"."""
    return f"{block.layout} | {block.clutter:g} | {row.label}"


def _maneuver_rows(block: BlockResult, group: str) -> list[RowResult]:
    return [r for r in block.rows if r.group == group and r.label != neutral_label(r)]


def winnable_blocks(blocks: Sequence[BlockResult]) -> tuple[str, ...]:
    """Maneuver row-blocks in which the baseline's window missed rate degrades by more than the
    practical threshold (that is, more than its floor) against the neutral row.

    Computed from the baseline alone, on the tuning seeds. A block in which the baseline loses
    almost nothing cannot be improved by the threshold and is left out of the denominator of the
    primary criterion.
    """
    labels = []
    for block in blocks:
        for group in ("maneuver", "held-out"):
            for row in _maneuver_rows(block, group):
                neutral = _row(block, neutral_label(row))
                degradation = across_rows(row, neutral, "window_missed_rate", BASELINE).mean
                if degradation > practical_threshold("missed", degradation):
                    labels.append(block_label(block, row))
    return tuple(labels)


# --- tuning selection --------------------------------------------------------------------


class Selection(NamedTuple):
    """Outcome of a selection among candidates.

    Attributes:
        name: The selected candidate.
        fallback: True if no candidate met the feasibility rule and the least violating one
            was taken.
        lines: One line per candidate with the numbers behind the choice.
    """

    name: str
    fallback: bool
    lines: list[str]


def _neutral_violation(blocks: Sequence[BlockResult], arm: str, two_sided: bool) -> float:
    worst = 0.0
    for block in blocks:
        row = _row(block, "neutral")
        for metric, margin in MARGINS.items():
            worst = max(worst, violation(paired(row, metric, arm), margin, two_sided))
    return worst


def select_family(blocks: Sequence[BlockResult], candidates: Sequence[str]) -> Selection:
    """Choose the candidate of an IMM / high-Q family on the tuning results.

    Feasible = on the neutral rows of every block the upper 95% bound of the paired difference to
    the baseline is within the neutral margins. Among the feasible ones the lowest mean window
    missed rate over the maneuver rows of all blocks wins; candidates within 0.002 of the best
    are ordered by mean common-support window RMSE. If none is feasible, the one whose neutral
    difference leaves the margins least is taken and flagged as a fallback.
    """
    table = []
    for arm in candidates:
        missed, cs = [], []
        for block in blocks:
            for row in _maneuver_rows(block, "maneuver"):
                index = arm_index(row, arm)
                missed.append(np.nanmean(row.metrics["window_missed_rate"][index]))
                cs.append(np.nanmean(row.metrics["cs_window_rmse"][index]))
        table.append(
            {
                "name": arm,
                "violation": _neutral_violation(blocks, arm, two_sided=False),
                "missed": float(np.mean(missed)),
                "rmse": float(np.mean(cs)),
            }
        )
    feasible = [c for c in table if c["violation"] == 0.0]
    lines = [
        f"{c['name']}: neutral violation {c['violation']:.3g} margins, window missed "
        f"{c['missed']:.4g}, common-support window rmse {c['rmse']:.4g}"
        for c in table
    ]
    if feasible:
        best = min(c["missed"] for c in feasible)
        close = [c for c in feasible if c["missed"] <= best + 0.002]
        chosen = min(close, key=lambda c: (c["rmse"], c["missed"]))
        return Selection(chosen["name"], False, lines)
    chosen = min(table, key=lambda c: c["violation"])
    return Selection(chosen["name"], True, lines)


def _bias_violation(
    maneuver_blocks: Sequence[BlockResult], bias_blocks: Sequence[BlockResult], arm: str
) -> float:
    worst = 0.0
    for block in bias_blocks:
        row = _row(block, "bias 0 deg")
        for metric, margin in MARGINS.items():
            worst = max(worst, violation(paired(row, metric, arm), margin, two_sided=True))
    for block in maneuver_blocks:
        neutral = _row(block, "neutral")
        for row in _maneuver_rows(block, "maneuver"):
            shift = across_rows(row, neutral, "bias_run_mean_abs_error_deg", arm)
            worst = max(worst, violation(shift, B3_BIAS_BOUND_DEG, two_sided=False))
            for metric, margin in MARGINS.items():
                worst = max(worst, violation(paired(row, metric, arm), margin, two_sided=True))
    return worst


def select_bias_prior(
    maneuver_blocks: Sequence[BlockResult],
    bias_blocks: Sequence[BlockResult],
    candidates: Sequence[str],
) -> Selection:
    """Choose the bias estimate candidate (a prior) on the tuning results.

    Feasible = the neutral equivalence margins hold at b = 0 and criterion B3 (no bias read into
    maneuvers, no harm to the Phase 6 scores on the maneuver rows) holds on every tuned maneuver
    row. Among the feasible ones the largest mean RMSE improvement over the baseline at b of 0.5
    and 1 deg wins; with none feasible, the least violating one is taken and flagged.
    """
    table = []
    for arm in candidates:
        gains = []
        for block in bias_blocks:
            for label in ("bias 0.5 deg", "bias 1 deg"):
                row = _row(block, label)
                gains.append(-paired(row, "run_position_rmse", arm).mean)
        table.append(
            {
                "name": arm,
                "violation": _bias_violation(maneuver_blocks, bias_blocks, arm),
                "gain": float(np.nanmean(gains)),
            }
        )
    lines = [
        f"{c['name']}: violation {c['violation']:.3g} margins, mean rmse gain at 0.5 and 1 deg "
        f"{c['gain']:.4g} m"
        for c in table
    ]
    feasible = [c for c in table if c["violation"] == 0.0]
    if feasible:
        return Selection(max(feasible, key=lambda c: c["gain"])["name"], False, lines)
    return Selection(min(table, key=lambda c: c["violation"])["name"], True, lines)


# --- the criteria ------------------------------------------------------------------------


def _fmt(difference: PairedDifference) -> str:
    if difference.n_valid < 2:
        return "n/a"
    return f"{difference.mean:+.4g} [{difference.lower:+.4g}, {difference.upper:+.4g}]"


def criterion_a1(
    blocks: Sequence[BlockResult], arm: str, winnable: Sequence[str], group: str
) -> Criterion:
    """Primary criterion for an IMM arm on the maneuver rows of one group ("maneuver" or
    "held-out").

    PASS: window missed rate improves in at least two thirds of the winnable row-blocks, and no
    row-block (winnable or not) worsens in window missed rate, ghost rate or ID switches.
    Common-support window RMSE is listed as a secondary score.
    """
    name = f"A1 {arm} ({group} rows)"
    improved = worsened = total = 0
    lines = []
    secondary = []
    checks = 0
    for block in blocks:
        for row in _maneuver_rows(block, group):
            neutral = _row(block, neutral_label(row))
            label = block_label(block, row)
            degradation = across_rows(row, neutral, "window_missed_rate", BASELINE).mean
            missed = paired(row, "window_missed_rate", arm)
            verdicts = {
                "missed": judge(missed, practical_threshold("missed", degradation)),
                "ghost": judge(paired(row, "run_ghost_rate", arm), practical_threshold("ghost")),
                "id": judge(paired(row, "run_id_switches", arm), practical_threshold("id")),
            }
            checks += 3
            if "worsens" in verdicts.values():
                worsened += 1
            in_set = label in winnable
            if in_set:
                total += 1
                improved += int(verdicts["missed"] == "improves")
            rmse_deg = across_rows(row, neutral, "window_position_rmse", BASELINE).mean
            rmse = judge(paired_cs(row, arm, "window"), practical_threshold("rmse", rmse_deg))
            if rmse == "worsens":
                secondary.append(label)
            lines.append(
                f"{label}{' [winnable]' if in_set else ''}: d window missed {_fmt(missed)} "
                f"({verdicts['missed']}), ghost {verdicts['ghost']}, id {verdicts['id']}, "
                f"cs window rmse {rmse}"
            )
    lines.append(
        f"improved {improved} of {total} winnable row-blocks (need "
        f"{math.ceil(MISSED_RATIO_NEEDED * total)}); row-blocks with a worsening: {worsened}; "
        f"family of {checks} checks, unadjusted"
    )
    lines.append(f"secondary: cs window rmse worsens in {len(secondary)}: {secondary}")
    if total == 0:
        return Criterion(name, "N/A", lines)
    passed = improved >= math.ceil(MISSED_RATIO_NEEDED * total) and worsened == 0
    return Criterion(name, "PASS" if passed else "FAIL", lines)


def _aggregate(results: list[str]) -> str:
    if all(v == "PASS" for v in results):
        return "PASS"
    if any(v == "FAIL" for v in results):
        return "FAIL"
    return "UNDERPOWERED"


def criterion_a2(blocks: Sequence[BlockResult], arm: str) -> Criterion:
    """Neutral non-inferiority of an IMM arm: the upper bound of the paired difference to the
    baseline within the neutral margins on every neutral row."""
    results, lines = [], []
    for block in blocks:
        row = _row(block, "neutral")
        for metric, margin in MARGINS.items():
            diff = paired(row, metric, arm)
            verdict = non_inferior(diff, margin)
            results.append(verdict)
            lines.append(
                f"{block.layout} | {block.clutter:g} | {metric}: {_fmt(diff)} vs +{margin:g}: "
                f"{verdict}"
            )
    return Criterion(f"A2 {arm} (neutral non-inferiority)", _aggregate(results), lines)


def _drag_total(row: RowResult, arm: str, seeds_from: int = 0) -> np.ndarray:
    values = np.array(
        [row.metrics[f"missed_bin_{b:02d}"][arm_index(row, arm)] for b in DRAG_BINS]
    )
    return values.sum(axis=0)[seeds_from:]


def criterion_hdrag(blocks: Sequence[BlockResult], arms: Sequence[str]) -> Criterion:
    """The drag-tail hypothesis: removing the stiffness alone removes the tail.

    Missed rate summed over the bins of [25, 45) s, separated layout, turn 5 deg/s, both clutter
    rates. SUPPORTED if the high-Q EKF reduces it by at least half with an interval excluding
    zero in every such block; REFUTED if its interval includes zero (or is above) in every one;
    PARTIAL otherwise. The other arms are listed. Reported on all seeds and on the seeds from 10
    on (seeds 0-9 produced the hypothesis).
    """
    name = "H-drag (separated, turn 5 deg/s, missed rate summed over 25-45 s)"
    chosen = [b for b in blocks if b.layout == "separated"]
    lines = []
    outcomes: dict[int, list[str]] = {0: [], LATE_SEEDS_FROM: []}
    for block in chosen:
        try:
            row = _row(block, "turn 5 deg/s")
        except KeyError:
            continue
        if f"missed_bin_{DRAG_BINS[-1]:02d}" not in row.metrics:
            continue
        for start, label in ((0, "all seeds"), (LATE_SEEDS_FROM, f"seeds {LATE_SEEDS_FROM}+")):
            base = _drag_total(row, BASELINE, start)
            for arm in arms:
                if arm == BASELINE or arm not in row.arm_names:
                    continue
                diff = paired_difference(_drag_total(row, arm, start), base)
                reduction = -diff.mean / float(np.nanmean(base)) if np.nanmean(base) else np.nan
                lines.append(
                    f"clutter {block.clutter:g}, {label}, {arm}: baseline "
                    f"{np.nanmean(base):.4g}, d {_fmt(diff)}, reduction {reduction:.3g}"
                )
                if arm == "EKF high-Q":
                    supported = (
                        diff.n_valid >= 2 and diff.upper < 0 and reduction >= DRAG_REDUCTION_NEEDED
                    )
                    refuted = diff.n_valid < 2 or diff.upper >= 0
                    outcomes[start].append(
                        "SUPPORTED" if supported else "REFUTED" if refuted else "PARTIAL"
                    )
    if not outcomes[0]:
        return Criterion(name, "N/A", lines)

    def combine(values: list[str]) -> str:
        if values and all(v == "SUPPORTED" for v in values):
            return "SUPPORTED"
        if values and all(v == "REFUTED" for v in values):
            return "REFUTED"
        return "PARTIAL"

    lines.append(f"verdict on seeds {LATE_SEEDS_FROM}+: {combine(outcomes[LATE_SEEDS_FROM])}")
    return Criterion(name, combine(outcomes[0]), lines)


def criterion_b1(blocks: Sequence[BlockResult], arm: str = "EKF+bias") -> Criterion:
    """The bias estimate improves run RMSE and cross-range RMS at b = 0.5 and 1 deg in every
    block."""
    lines, results = [], []
    for block in blocks:
        neutral = _row(block, "bias 0 deg")
        for label in ("bias 0.5 deg", "bias 1 deg"):
            row = _row(block, label)
            for metric in ("run_position_rmse", "cross_rms"):
                degradation = across_rows(row, neutral, metric, BASELINE).mean
                diff = paired(row, metric, arm)
                verdict = judge(diff, practical_threshold("rmse", degradation))
                results.append(verdict == "improves")
                lines.append(
                    f"{block.layout} | {block.clutter:g} | {label} | {metric}: {_fmt(diff)} "
                    f"({verdict})"
                )
    verdict = "PASS" if all(results) else "FAIL"
    return Criterion(f"B1 {arm} improves at 0.5 and 1 deg", verdict, lines)


def criterion_b2(blocks: Sequence[BlockResult], arm: str = "EKF+bias") -> Criterion:
    """Neutral world (b = 0): the whole interval of the paired difference to the baseline lies
    within the neutral margins. Whether it includes zero is listed, not required."""
    lines, results = [], []
    for block in blocks:
        row = _row(block, "bias 0 deg")
        for metric, margin in MARGINS.items():
            diff = paired(row, metric, arm)
            verdict = equivalence(diff, margin)
            results.append(verdict)
            zero = diff.n_valid >= 2 and diff.lower <= 0.0 <= diff.upper
            lines.append(
                f"{block.layout} | {block.clutter:g} | {metric}: {_fmt(diff)} within +-{margin:g}: "
                f"{verdict}; interval includes 0: {zero}"
            )
    return Criterion(f"B2 {arm} neutral equivalence (b = 0)", _aggregate(results), lines)


def criterion_b3(blocks: Sequence[BlockResult], arm: str = "EKF+bias") -> Criterion:
    """Maneuvers are not read as bias: on every tuned maneuver row the run-mean |b_app| exceeds
    that of the neutral row by at most 0.05 deg (upper bound of the paired difference), and the
    bias arm meets the neutral margins against the baseline on that row."""
    lines, results = [], []
    for block in blocks:
        neutral = _row(block, "neutral")
        for row in _maneuver_rows(block, "maneuver"):
            shift = across_rows(row, neutral, "bias_run_mean_abs_error_deg", arm)
            verdicts = [non_inferior(shift, B3_BIAS_BOUND_DEG)]
            verdicts += [
                equivalence(paired(row, metric, arm), margin) for metric, margin in MARGINS.items()
            ]
            results.append(_aggregate(verdicts))
            lines.append(
                f"{block_label(block, row)}: d run-mean |b_app| {_fmt(shift)} deg vs "
                f"+{B3_BIAS_BOUND_DEG:g}, margins {verdicts[1:]} -> {results[-1]}"
            )
    return Criterion(f"B3 {arm} maneuvers are not bias", _aggregate(results), lines)


def hedge_hard_cases(blocks: Sequence[BlockResult]) -> Criterion:
    """Hedged check (descriptive): an IMM arm with the bias estimate is not worse in run RMSE
    than the better of its two single features by more than 1 m."""
    lines, bad = [], 0
    for block in blocks:
        for row in block.rows:
            for imm in ("IMM-A", "IMM-B"):
                combo = row.metrics["run_position_rmse"][arm_index(row, f"{imm}+bias")]
                best = np.minimum(
                    row.metrics["run_position_rmse"][arm_index(row, imm)],
                    row.metrics["run_position_rmse"][arm_index(row, "EKF+bias")],
                )
                diff = paired_difference(combo, best)
                verdict = judge(diff, practical_threshold("rmse"))
                bad += int(verdict == "worsens")
                lines.append(
                    f"{block.layout} | {block.clutter:g} | {row.label} | {imm}+bias vs better "
                    f"single: {_fmt(diff)} ({verdict})"
                )
    return Criterion(
        "Hard cases: IMM+bias against the better single feature (descriptive)",
        "PASS" if bad == 0 else "FAIL",
        lines,
    )


def runtime_summary(blocks: Sequence[BlockResult], title: str) -> Criterion:
    """Mean tracker runtime per trial of every arm, and relative to the baseline (reported)."""
    totals: dict[str, list[float]] = {}
    for block in blocks:
        for row in block.rows:
            for arm in row.arm_names:
                totals.setdefault(arm, []).append(
                    float(np.nanmean(row.metrics["runtime_s"][arm_index(row, arm)]))
                )
    base = float(np.mean(totals[BASELINE]))
    lines = [
        f"{arm}: {np.mean(values):.3f} s per trial, x{np.mean(values) / base:.2f} of {BASELINE}"
        for arm, values in totals.items()
    ]
    return Criterion(f"Runtime per trial, {title} (reported)", "REPORTED", lines)


def evaluate_all(
    maneuver: Sequence[BlockResult],
    bias: Sequence[BlockResult],
    hard: Sequence[BlockResult],
    parameters: Parameters,
) -> list[Criterion]:
    """Every pre-registered criterion on the evaluation results."""
    criteria: list[Criterion] = []
    for arm in ("IMM-A", "IMM-B"):
        criteria.append(criterion_a1(maneuver, arm, parameters.winnable, "maneuver"))
        criteria.append(criterion_a1(maneuver, arm, parameters.winnable, "held-out"))
        criteria.append(criterion_a2(maneuver, arm))
    criteria.append(criterion_hdrag(maneuver, ("EKF high-Q", "IMM-A", "IMM-B")))
    criteria.append(criterion_b1(bias))
    criteria.append(criterion_b2(bias))
    criteria.append(criterion_b3(maneuver))
    criteria.append(hedge_hard_cases(hard))
    for title, blocks in (("maneuver", maneuver), ("bias", bias), ("hard", hard)):
        criteria.append(runtime_summary(blocks, title))
    return criteria
