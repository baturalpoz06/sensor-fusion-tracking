"""Text tables of a robustness sweep: scores, error diagnostics, window scores, paired differences.

One row per configuration of the sweep. Cells show the mean over seeds with the half-width of
the 95% Student t interval, and medians of the heavy-tailed scores. The paired table compares
every row with a reference row (the neutral one) seed by seed, which is the right comparison
for rows that share their random draws. This module only formats numbers; it contains no prose
about the results.
"""

from collections.abc import Mapping, Sequence

import numpy as np

from fusion.mtt_experiment import SweepResult
from fusion.mtt_metrics import PairedDifference, paired_difference
from fusion.mtt_report import COLUMN_WIDTH, MEDIAN_WIDTH, VALID_WIDTH, settings_caption
from fusion.outage_report import format_table
from fusion.robustness_experiment import RobustnessConfig

LABEL_WIDTH = 40

# Metric name -> (column header, significant digits of the mean).
SCORE_COLUMNS = {
    "run_position_rmse": ("rmse [m]", 3),
    "run_ghost_rate": ("ghost/step", 3),
    "run_false_track_rate": ("false/step", 3),
    "run_missed_rate": ("missed", 3),
    "run_id_switches": ("id sw", 3),
    "run_births_per_scan": ("births/scan", 3),
    "in_view_fraction": ("in view", 3),
    "min_target_separation": ("min sep [m]", 3),
}
SCORE_MEDIANS = {
    "run_position_rmse": "med rmse [m]",
    "run_ghost_rate": "med ghost",
    "run_missed_rate": "med missed",
    "run_id_switches": "med id sw",
}
DIAGNOSTIC_COLUMNS = {
    "match_fraction_50": ("matched 50 m", 3),
    "match_fraction_wide": ("matched wide", 3),
    "along_rms": ("along rms [m]", 3),
    "cross_rms": ("cross rms [m]", 3),
    "along_mean": ("along mean [m]", 3),
    "cross_mean": ("cross mean [m]", 3),
    "cross_bearing_rms": ("cross [mrad]", 3),
    "nees_mean": ("NEES", 3),
    "nees_outlier_fraction": ("NEES outliers", 3),
    "camera_accept_rate": ("cam accept", 3),
    "camera_true_accept_rate": ("cam true acc", 3),
    "camera_wrong_update_rate": ("cam wrong", 3),
    "camera_unattributed_update_rate": ("cam unattr", 3),
    "camera_skip_rate": ("cam skip", 3),
}
DIAGNOSTIC_MEDIANS = {
    "along_rms": "med along [m]",
    "cross_rms": "med cross [m]",
    "nees_mean": "med NEES",
}
WINDOW_COLUMNS = {
    "window_position_rmse": ("win rmse [m]", 3),
    "window_missed_rate": ("win missed", 3),
    "window_ghost_rate": ("win ghost", 3),
    "window_cross_rms": ("win cross [m]", 3),
    "window_nees_mean": ("win NEES", 3),
}
WINDOW_MEDIANS = {"window_position_rmse": "med win rmse", "window_cross_rms": "med win cross"}
# Metrics of the paired table: name -> header of the mean difference.
PAIRED_METRICS = {
    "run_position_rmse": "d rmse [m]",
    "run_missed_rate": "d missed",
    "cross_rms": "d cross [m]",
    "nees_mean": "d NEES",
    "camera_true_accept_rate": "d cam true acc",
}

PILOT_BANNER = (
    "PILOT: a does-it-run check with few seeds. No conclusions may be drawn from these numbers."
)
LEGEND = (
    "rows share their seeds, so truth noise, detections and clutter are identical across rows;",
    "  a row differs from the neutral row only by its disturbance or by the tracker's belief.",
    "scores (run_*) are the Phase 6 scores at the Phase 6 match distance. diagnostics use the",
    "  wide match distance, so that a track pulled off its target stays visible as an error",
    "  instead of turning into a miss and a ghost; read them with the matched shares.",
    "  matched 50 m / matched wide = share of in-view target steps with a matched track.",
    "rmse is taken over matched (target, step) pairs only: a row that loses its tracks (high",
    "  missed) can show a lower rmse; read it together with missed and matched.",
    "along / cross rms and means: error (estimate minus truth) relative to the line of sight",
    "  from the radar; cross is counter-clockwise positive, cross [mrad] is cross-range error",
    "  divided by range. a lever arm changes sign with the target bearing, so use cross rms.",
    "NEES: ideal 4 (median of chi-square 4 is 3.36); conditioned on a match within the wide",
    "  distance, so a very bad tracker can look better than it is; NEES means are heavy-tailed,",
    "  read them with med NEES and NEES outliers (share above the chi-square 0.999 quantile).",
    "cam accept = updates made / track-scan pairs offered; cam skip = tracks left out for",
    "  overlapping bearing gates; cam true acc = detected targets whose track took that target's",
    "  measurement; cam wrong = updates of matched tracks with clutter or another target's",
    "  measurement; cam unattr = updates by tracks matched to no target.",
    "cells are mean +- half-width of the 95% t interval across seeds over the seeds where the",
    "  score is defined; 'med' columns are medians; intervals are not adjusted for the many",
    "  comparisons, so an interval excluding zero is not a significance claim.",
    "paired table: per-seed difference row - reference (mean +- 95% t interval, median of the",
    "  differences, not the difference of the medians); dropped = seeds left out because a",
    "  value was undefined in either row (a setting that loses its tracks drops exactly its",
    "  worst seeds), listed per metric in the order of the columns.",
)


def robustness_caption(config: RobustnessConfig, scenario: str | None = None) -> str:
    """One line with the settings shared by the rows of a table.

    The truth disturbance and the belief of a row are named in its label; the line gives the
    scene settings and the match distances.

    Args:
        config: Configuration of any row (for the shared settings).
        scenario: Name of the scene (layout) of the table.
    """
    prefix = f"scenario {scenario}: " if scenario else ""
    return (
        f"{prefix}{settings_caption(config)}; Phase 6 scores match within "
        f"{config.match_distance:g} m, diagnostics within {config.diagnostic_match_distance:g} m"
    )


def _table(
    result: SweepResult,
    columns: Mapping[str, tuple[str, int]],
    medians: Mapping[str, str],
    valid_names: tuple[str, str],
    row_labels: Sequence[str],
    value_label: str,
    caption: str | None,
    confidence: float,
) -> list[str]:
    missing = [name for name in (*columns, *medians, *valid_names) if name not in result.metrics]
    if missing:
        raise ValueError(f"the result lacks the metrics {missing}")
    return format_table(
        result,
        columns,
        medians,
        valid_names,
        row_labels,
        value_label,
        caption,
        confidence,
        label_width=LABEL_WIDTH,
    )


def format_score_table(
    result: SweepResult,
    row_labels: Sequence[str],
    value_label: str = "row",
    caption: str | None = None,
    confidence: float = 0.95,
) -> list[str]:
    """Table of the Phase 6 scores (and the in-view share and target separation) per row.

    The last column counts the seeds with a defined position error and missed rate.
    """
    return _table(
        result,
        SCORE_COLUMNS,
        SCORE_MEDIANS,
        ("run_position_rmse", "run_missed_rate"),
        row_labels,
        value_label,
        caption,
        confidence,
    )


def format_diagnostic_table(
    result: SweepResult,
    row_labels: Sequence[str],
    value_label: str = "row",
    caption: str | None = None,
    confidence: float = 0.95,
) -> list[str]:
    """Table of the error and camera diagnostics per row.

    The last column counts the seeds with a defined cross-range error and NEES.
    """
    return _table(
        result,
        DIAGNOSTIC_COLUMNS,
        DIAGNOSTIC_MEDIANS,
        ("cross_rms", "nees_mean"),
        row_labels,
        value_label,
        caption,
        confidence,
    )


def has_window_scores(result: SweepResult) -> bool:
    """Whether any row has a window score (a focus window was set for it)."""
    return any(np.isfinite(result.metrics[name]).any() for name in WINDOW_COLUMNS)


def format_window_table(
    result: SweepResult,
    row_labels: Sequence[str],
    value_label: str = "row",
    caption: str | None = None,
    confidence: float = 0.95,
) -> list[str]:
    """Table of the scores over the focus window of each row (n/a for rows without one)."""
    return _table(
        result,
        WINDOW_COLUMNS,
        WINDOW_MEDIANS,
        ("window_position_rmse", "window_cross_rms"),
        row_labels,
        value_label,
        caption,
        confidence,
    )


def format_paired_cell(difference: PairedDifference, digits: int = 3) -> str:
    """Mean difference and interval half-width as "+mean +- half"; "n/a" if no seed is valid."""
    if difference.n_valid == 0:
        return "n/a"
    if difference.n_valid == 1:
        return f"{difference.mean:+.{digits}g} +- ?"
    half_width = (difference.upper - difference.lower) / 2.0
    return f"{difference.mean:+.{digits}g} +- {half_width:.2g}"


def format_paired_table(
    result: SweepResult,
    reference: int,
    row_labels: Sequence[str],
    metrics: Mapping[str, str] = PAIRED_METRICS,
    value_label: str = "row - reference",
    caption: str | None = None,
    confidence: float = 0.95,
) -> list[str]:
    """Table of the per-seed differences of every row against a reference row.

    For each metric: the mean difference with the half-width of its t interval, then (after
    all means) the median of the differences, and a last column with the number of seeds left
    out per metric.

    Args:
        result: Sweep result of the rows.
        reference: Index of the reference row (usually the neutral one); it is not listed.
        row_labels: Text of each row of the result, including the reference.
        metrics: Metric name -> header of its mean-difference column.
        value_label: Header of the first column.
        caption: Optional first line.
        confidence: Coverage of the Student t interval across seeds.

    Raises:
        ValueError: If row_labels does not match the rows, the reference is not a row, or a
            metric is missing from the result.
    """
    n_rows = len(result.values)
    if len(row_labels) != n_rows:
        raise ValueError(f"need {n_rows} row labels, got {len(row_labels)}")
    if not 0 <= reference < n_rows:
        raise ValueError(f"reference must be a row index in [0, {n_rows}), got {reference}")
    missing = [name for name in metrics if name not in result.metrics]
    if missing:
        raise ValueError(f"the result lacks the metrics {missing}")
    header = f"{value_label:>{LABEL_WIDTH}}" + "".join(
        f"{text:>{COLUMN_WIDTH}}" for text in metrics.values()
    )
    header += "".join(f"{'med ' + text:>{MEDIAN_WIDTH}}" for text in metrics.values())
    header += f"{'dropped':>{VALID_WIDTH}}"
    lines = [caption] if caption else []
    lines += [header, "-" * len(header)]
    for i, label in enumerate(row_labels):
        if i == reference:
            continue
        differences = [
            paired_difference(result.metrics[name][i], result.metrics[name][reference], confidence)
            for name in metrics
        ]
        cells = "".join(f"{format_paired_cell(d):>{COLUMN_WIDTH}}" for d in differences)
        medians = "".join(
            f"{(f'{d.median:+.3g}' if d.n_valid else 'n/a'):>{MEDIAN_WIDTH}}" for d in differences
        )
        dropped = "/".join(str(d.n_dropped) for d in differences)
        lines.append(f"{label:>{LABEL_WIDTH}}{cells}{medians}{dropped:>{VALID_WIDTH}}")
    return lines
