"""Text tables of a sweep: mean, confidence half-width and median of the metrics per value.

This module only formats numbers; it contains no prose about the results.
"""

from collections.abc import Sequence

import numpy as np

from fusion.mtt_experiment import SweepResult
from fusion.mtt_metrics import seed_confidence_interval

# Metric name -> (column header, significant digits of the mean).
COLUMNS = {
    "position_rmse": ("rmse [m]", 3),
    "ghost_rate": ("ghost/step", 3),
    "missed_rate": ("missed", 3),
    "confirmation_delay": ("delay [s]", 3),
    "confirmed_fraction": ("confirmed", 3),
    "id_switches": ("id sw", 3),
    "births_per_scan": ("births/scan", 3),
}
# Metrics that also get a median column, for heavy-tailed scores where a few seeds
# dominate the mean: metric name -> column header.
MEDIAN_COLUMNS = {
    "position_rmse": "med rmse [m]",
    "missed_rate": "med missed",
    "id_switches": "med id sw",
}
COLUMN_WIDTH = 23
MEDIAN_WIDTH = 14
VALID_WIDTH = 14


def format_cell(values: np.ndarray, digits: int, confidence: float = 0.95) -> str:
    """Mean and confidence half-width of one metric over seeds, as "mean +- half".

    "n/a" if no seed has a valid value; the half-width is "?" with a single valid seed.
    """
    interval = seed_confidence_interval(values, confidence)
    if interval.n_valid == 0:
        return "n/a"
    if interval.n_valid == 1:
        return f"{interval.mean:.{digits}g} +- ?"
    half_width = (interval.upper - interval.lower) / 2.0
    return f"{interval.mean:.{digits}g} +- {half_width:.2g}"


def format_median(values: np.ndarray, digits: int = 3) -> str:
    """Median over the valid (non-NaN) seeds, or "n/a" if there is none."""
    valid = np.asarray(values, dtype=float)
    valid = valid[np.isfinite(valid)]
    return f"{np.median(valid):.{digits}g}" if valid.size else "n/a"


def format_sweep_table(
    result: SweepResult,
    confidence: float = 0.95,
    value_label: str | None = None,
    caption: str | None = None,
    row_labels: Sequence[str] | None = None,
) -> list[str]:
    """Table lines: one row per swept value, one column per metric.

    After the mean columns come the medians of the heavy-tailed metrics (position
    error, missed rate, ID switches), then how many seeds were valid for the two
    metrics that can be undefined (position error, confirmation delay) as "rmse/delay".

    Args:
        result: Sweep result from fusion.mtt_experiment.sweep.
        confidence: Coverage of the Student t interval across seeds.
        value_label: Header of the first column; defaults to the parameter name.
        caption: Optional first line, e.g. the settings shared by all rows.
        row_labels: Optional text for the first column in place of the values.

    Raises:
        ValueError: If row_labels does not have one entry per value.
    """
    if row_labels is not None and len(row_labels) != len(result.values):
        raise ValueError(f"need {len(result.values)} row labels, got {len(row_labels)}")
    label = value_label or result.parameter
    header = f"{label:>24}" + "".join(
        f"{COLUMNS[name][0]:>{COLUMN_WIDTH}}" for name in result.metrics if name in COLUMNS
    )
    header += "".join(f"{text:>{MEDIAN_WIDTH}}" for text in MEDIAN_COLUMNS.values())
    header += f"{'valid seeds':>{VALID_WIDTH}}"
    lines = [caption] if caption else []
    lines += [header, "-" * len(header)]
    for i, value in enumerate(result.values):
        cells = "".join(
            f"{format_cell(result.metrics[name][i], COLUMNS[name][1], confidence):>{COLUMN_WIDTH}}"
            for name in result.metrics
            if name in COLUMNS
        )
        medians = "".join(
            f"{format_median(result.metrics[name][i]):>{MEDIAN_WIDTH}}" for name in MEDIAN_COLUMNS
        )
        valid = {
            name: seed_confidence_interval(result.metrics[name][i], confidence).n_valid
            for name in ("position_rmse", "confirmation_delay")
        }
        counts = f"{valid['position_rmse']}/{valid['confirmation_delay']}"
        first = row_labels[i] if row_labels is not None else f"{value:g}"
        lines.append(f"{first:>24}{cells}{medians}{counts:>{VALID_WIDTH}}")
    return lines
