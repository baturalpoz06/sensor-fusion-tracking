"""Text output of the Phase 8b experiment: tables, headline numbers, criteria and tuning.

Every table lists numbers only; this module contains no prose about the results. Cells show the
mean over seeds with the half width of the 95% t interval. A paired cell is the per-seed
difference arm - baseline on the same row and seeds.
"""

from collections.abc import Mapping, Sequence

import numpy as np

from fusion.improvement_criteria import Criterion, Selection
from fusion.improvement_experiment import BASELINE, BlockResult, Parameters, RowResult
from fusion.mtt_experiment import SweepResult
from fusion.mtt_metrics import PairedDifference, paired_difference
from fusion.mtt_report import COLUMN_WIDTH, format_cell, format_median
from fusion.outage_report import format_table
from fusion.robustness_report import (
    DIAGNOSTIC_COLUMNS,
    DIAGNOSTIC_MEDIANS,
    SCORE_COLUMNS,
    SCORE_MEDIANS,
    WINDOW_COLUMNS,
    WINDOW_MEDIANS,
    format_paired_cell,
    has_window_scores,
)

LABEL_WIDTH = 52
MEDIAN_WIDTH = 16

# Paired columns: header, score name, kind. kind "direct" takes the difference arm - baseline of
# the score; "cs_run" / "cs_window" take the difference of the common-support RMSE of the arm and
# of the baseline on the pairs both matched.
PAIRED_SPECS = (
    ("d rmse [m]", "run_position_rmse", "direct"),
    ("d cs rmse [m]", "cs_run", "cs"),
    ("d missed", "run_missed_rate", "direct"),
    ("d win cs rmse [m]", "cs_window", "cs"),
    ("d win missed", "window_missed_rate", "direct"),
    ("d cross [m]", "cross_rms", "direct"),
    ("d NEES", "nees_mean", "direct"),
    ("d NEES med", "nees_median", "direct"),
    ("d ghost", "run_ghost_rate", "direct"),
    ("d id sw", "run_id_switches", "direct"),
    ("d runtime [s]", "runtime_s", "direct"),
)
LEGEND = (
    "d = arm - EKF (the Phase 8a tracker), per seed on the same row and the same data (paired);",
    "  cells are mean +- half width of the 95% t interval; [abs] = the EKF's own value.",
    "rmse is over matched (target, step) pairs at the 50 m match distance; cs rmse is over the",
    "  pairs that both the arm and the EKF matched (win = focus window); missed = share of",
    "  in-view target steps without a matched confirmed track (win = focus window).",
    "NEES, cross-range rms and the camera scores use the 200 m match distance.",
    "intervals are not adjusted for the many comparisons.",
)
PILOT_BANNER = (
    "PILOT: a does-it-run check with few seeds. No conclusions may be drawn from these numbers."
)
UNFROZEN_BANNER = (
    "UNFROZEN: the arm parameters are placeholders, not tuned values (see improvement_tune.txt)."
)


def flatten_block(block: BlockResult) -> tuple[SweepResult, list[str], list[int]]:
    """The (row, arm) pairs of a block as the rows of one sweep result.

    Returns:
        The sweep result, a label "row | arm" for each pair, and for each pair the index of the
        baseline pair of its row (its own index for a baseline pair).
    """
    names = list(block.rows[0].metrics)
    stack: dict[str, list[np.ndarray]] = {name: [] for name in names}
    labels, references = [], []
    for row in block.rows:
        first = len(labels)
        for a, arm in enumerate(row.arm_names):
            labels.append(f"{row.label} | {arm}")
            references.append(first)
            for name in names:
                stack[name].append(row.metrics[name][a])
    result = SweepResult(
        "row",
        np.arange(len(labels), dtype=float),
        {name: np.array(values) for name, values in stack.items()},
    )
    return result, labels, references


def _paired(result: SweepResult, i: int, reference: int, name: str, kind: str):
    if kind == "direct":
        return paired_difference(result.metrics[name][i], result.metrics[name][reference])
    return paired_difference(
        result.metrics[f"{name}_rmse"][i], result.metrics[f"{name}_rmse_ref"][i]
    )


def paired_table(
    block: BlockResult,
    specs: Sequence[tuple[str, str, str]] = PAIRED_SPECS,
    medians: bool = True,
    caption: str | None = None,
) -> list[str]:
    """Paired differences of every arm against the baseline of its row.

    The baseline pairs are not listed; after the mean columns come the medians of the
    differences (not the differences of the medians).
    """
    result, labels, references = flatten_block(block)
    header = f"{'row | arm':>{LABEL_WIDTH}}" + "".join(
        f"{text:>{COLUMN_WIDTH}}" for text, _, _ in specs
    )
    if medians:
        header += "".join(f"{'med ' + text[2:]:>{MEDIAN_WIDTH}}" for text, _, _ in specs)
    lines = [caption] if caption else []
    lines += [header, "-" * len(header)]
    for i, label in enumerate(labels):
        if references[i] == i:
            continue
        diffs: list[PairedDifference] = [
            _paired(result, i, references[i], name, kind) for _, name, kind in specs
        ]
        cells = "".join(f"{format_paired_cell(d):>{COLUMN_WIDTH}}" for d in diffs)
        if medians:
            cells += "".join(
                f"{(f'{d.median:+.3g}' if d.n_valid else 'n/a'):>{MEDIAN_WIDTH}}" for d in diffs
            )
        lines.append(f"{label:>{LABEL_WIDTH}}{cells}")
    return lines


def bin_lines(block: BlockResult, edges: Sequence[tuple[float, float]]) -> list[str]:
    """Missed rate and drag rate per time bin, mean over seeds, one line pair per (row, arm)."""
    names = [f"{a:g}-{b:g}" for a, b in edges]
    lines = [f"{'row | arm':>{LABEL_WIDTH}}  bins [s]: " + " ".join(f"{n:>7}" for n in names)]
    for row in block.rows:
        for a, arm in enumerate(row.arm_names):
            for kind in ("missed", "drag"):
                means = [
                    np.nanmean(row.metrics[f"{kind}_bin_{b:02d}"][a])
                    if np.isfinite(row.metrics[f"{kind}_bin_{b:02d}"][a]).any()
                    else np.nan
                    for b in range(len(edges))
                ]
                lines.append(
                    f"{row.label + ' | ' + arm:>{LABEL_WIDTH}}  {kind:>6}: "
                    + " ".join(f"{m:7.4f}" for m in means)
                )
    return lines


def mode_lines(block: BlockResult) -> list[str]:
    """Mode probabilities of the IMM arms: outside, inside the window, mean largest."""
    lines = [
        f"{'row | arm':>{LABEL_WIDTH}}{'non-ref outside':>{COLUMN_WIDTH}}"
        f"{'non-ref window':>{COLUMN_WIDTH}}{'mean max mu':>{COLUMN_WIDTH}}"
    ]
    for row in block.rows:
        for a, arm in enumerate(row.arm_names):
            if not np.isfinite(row.metrics["mode_max_mu"][a]).any():
                continue
            cells = "".join(
                f"{format_cell(row.metrics[name][a], 3):>{COLUMN_WIDTH}}"
                for name in ("mode_nonref_outside", "mode_nonref_window", "mode_max_mu")
            )
            lines.append(f"{row.label + ' | ' + arm:>{LABEL_WIDTH}}{cells}")
    return lines


def bias_lines(block: BlockResult) -> list[str]:
    """Bias estimate scores of the arms that have one, in degrees."""
    columns = (
        ("run-mean |err| [deg]", "bias_run_mean_abs_error_deg"),
        ("window max |b| [deg]", "bias_window_max_abs_estimate_deg"),
        ("final |err| [deg]", "bias_final_error_deg"),
        ("conv 0.25 share", "bias_converged_025"),
        ("conv 0.1 share", "bias_converged_010"),
        ("conv 0.25 med [s]", "bias_conv_025_s"),
        ("conv 0.1 med [s]", "bias_conv_010_s"),
    )
    lines = [
        f"{'row | arm':>{LABEL_WIDTH}}" + "".join(f"{h:>{COLUMN_WIDTH}}" for h, _ in columns)
    ]
    for row in block.rows:
        for a, arm in enumerate(row.arm_names):
            if not np.isfinite(row.metrics["bias_final_error_deg"][a]).any():
                continue
            cells = ""
            for _, name in columns:
                values = row.metrics[name][a]
                text = format_median(values) if name.endswith("_s") else format_cell(values, 3)
                cells += f"{text:>{COLUMN_WIDTH}}"
            lines.append(f"{row.label + ' | ' + arm:>{LABEL_WIDTH}}{cells}")
    return lines


def format_block(block: BlockResult, edges: Sequence[tuple[float, float]]) -> list[str]:
    """Every table of one block: scores, diagnostics, window scores, paired differences,
    time bins, mode probabilities and bias estimate scores."""
    result, labels, _ = flatten_block(block)

    def table(columns, medians, valid) -> list[str]:
        return format_table(
            result, columns, medians, valid, labels, "row | arm", None, 0.95, LABEL_WIDTH
        )

    sections = [
        (
            "scores (Phase 6)",
            table(SCORE_COLUMNS, SCORE_MEDIANS, ("run_position_rmse", "run_missed_rate")),
        ),
        (
            "error and camera diagnostics",
            table(DIAGNOSTIC_COLUMNS, DIAGNOSTIC_MEDIANS, ("cross_rms", "nees_mean")),
        ),
    ]
    if has_window_scores(result):
        sections.append(
            (
                "focus window scores",
                table(WINDOW_COLUMNS, WINDOW_MEDIANS, ("window_position_rmse", "window_cross_rms")),
            )
        )
    sections.append(("paired differences, arm - EKF", paired_table(block)))
    sections.append(("missed and drag rate per time bin", bin_lines(block, edges)))
    modes = mode_lines(block)
    if len(modes) > 1:
        sections.append(("IMM mode probabilities", modes))
    bias = bias_lines(block)
    if len(bias) > 1:
        sections.append(("bias estimate", bias))
    lines = [f"--- {block.title} ---"]
    for title, body in sections:
        lines += ["", title, *body]
    return lines


def headline_lines(
    groups: Mapping[str, Sequence[BlockResult]], banners: Sequence[str] = ()
) -> list[str]:
    """The compact numbers-only headline: paired differences per (row, arm) of every block."""
    title = "HEADLINE NUMBERS of the Phase 8b evaluation. No interpretation."
    lines = [*banners, title, *LEGEND, ""]
    specs = PAIRED_SPECS
    header = f"{'row | arm':>{LABEL_WIDTH}}" + "".join(f"{h:>{COLUMN_WIDTH}}" for h, _, _ in specs)
    lines += [header, "-" * len(header)]
    for group, blocks in groups.items():
        for block in blocks:
            lines.append(f"== {group}: {block.title}")
            for row in block.rows:
                lines.append(f"-- {row.label}")
                result, labels, references = flatten_block(BlockResult("", "", 0.0, [row]))
                for i, label in enumerate(labels):
                    if references[i] == i:
                        cells = "".join(
                            f"{_absolute(result, i, name, kind):>{COLUMN_WIDTH}}"
                            for _, name, kind in specs
                        )
                        lines.append(f"{label + ' [abs]':>{LABEL_WIDTH}}{cells}")
                        continue
                    diffs = [_paired(result, i, references[i], n, k) for _, n, k in specs]
                    cells = "".join(f"{format_paired_cell(d):>{COLUMN_WIDTH}}" for d in diffs)
                    lines.append(f"{label:>{LABEL_WIDTH}}{cells}")
    return lines


def _absolute(result: SweepResult, i: int, name: str, kind: str) -> str:
    key = name if kind == "direct" else f"{name}_rmse_ref"
    return format_cell(result.metrics[key][i], 3)


def criteria_lines(criteria: Sequence[Criterion], header: Sequence[str]) -> list[str]:
    """The verdict and the numbers of every criterion, as text."""
    lines = [*header, ""]
    for criterion in criteria:
        lines.append(f"== {criterion.name}: {criterion.verdict}")
        lines += [f"   {line}" for line in criterion.lines]
        lines.append("")
    lines.append("verdicts: " + "; ".join(f"{c.name.split(' ')[0]} {c.verdict}" for c in criteria))
    return lines


def parameters_text(parameters: Parameters) -> str:
    """The parameters as the Python expression to put into improvement_experiment.FROZEN."""
    winnable = ", ".join(repr(label) for label in parameters.winnable)
    return (
        "Parameters(\n"
        f"    ekf_high_accel_std={parameters.ekf_high_accel_std!r},\n"
        f"    imm_a_high_accel_std={parameters.imm_a_high_accel_std!r},\n"
        f"    imm_b_omega_deg={parameters.imm_b_omega_deg!r},\n"
        f"    imm_b_accel_std={parameters.imm_b_accel_std!r},\n"
        f"    bias_prior={parameters.bias_prior!r},\n"
        f"    winnable=({winnable}{',' if parameters.winnable else ''}),\n"
        f"    commit={parameters.commit!r},\n"
        ")"
    )


def tuning_lines(
    selections: Mapping[str, Selection],
    parameters: Parameters,
    winnable: Sequence[str],
    power: Sequence[str],
    sensitivity: Sequence[str] = (),
) -> list[str]:
    """The outcome of the tuning run: every candidate with its numbers, the choices, the
    winnable blocks, the projected interval widths and the frozen-parameter expression."""
    lines = ["TUNING RUN: seeds 1000-1019, tuned maneuver rows and bias rows only.", ""]
    for family, selection in selections.items():
        lines.append(
            f"== {family}: selected {selection.name}"
            f"{' (FALLBACK: no candidate met the neutral rule)' if selection.fallback else ''}"
        )
        lines += [f"   {line}" for line in selection.lines]
        lines.append("")
    lines.append(f"== winnable maneuver row-blocks ({len(winnable)})")
    lines += [f"   {label}" for label in winnable]
    lines += ["", "== projected 95% half widths at 50 seeds against the neutral margins"]
    lines += [f"   {line}" for line in power]
    if sensitivity:
        lines += ["", "== bias process noise sensitivity (descriptive)"]
        lines += [f"   {line}" for line in sensitivity]
    lines += ["", "FROZEN = " + parameters_text(parameters)]
    return lines


def timing_lines(rows: Sequence[RowResult]) -> list[str]:
    """Tracker runtime per trial of every arm in the timing run, and relative to the baseline."""
    lines = [
        f"{'row | arm':>{LABEL_WIDTH}}{'runtime [s]':>{COLUMN_WIDTH}}{'x baseline':>{COLUMN_WIDTH}}"
    ]
    for row in rows:
        base = float(np.nanmean(row.metrics["runtime_s"][row.arm_names.index(BASELINE)]))
        for a, arm in enumerate(row.arm_names):
            values = row.metrics["runtime_s"][a]
            lines.append(
                f"{row.label + ' | ' + arm:>{LABEL_WIDTH}}{format_cell(values, 3):>{COLUMN_WIDTH}}"
                f"{np.nanmean(values) / base:>{COLUMN_WIDTH}.2f}"
            )
    return lines
