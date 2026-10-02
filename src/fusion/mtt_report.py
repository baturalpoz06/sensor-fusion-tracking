"""Text report of a sweep: mean and confidence half-width of every metric per value."""

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
COLUMN_WIDTH = 19
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


def format_sweep_table(
    result: SweepResult, confidence: float = 0.95, value_label: str | None = None
) -> list[str]:
    """Table lines: one row per swept value, one column per metric.

    The last column gives how many seeds were valid for the two metrics that can
    be undefined (position error, confirmation delay) as "rmse/delay".

    Args:
        result: Sweep result from fusion.mtt_experiment.sweep.
        confidence: Coverage of the Student t interval across seeds.
        value_label: Header of the first column; defaults to the parameter name.
    """
    label = value_label or result.parameter
    header = f"{label:>20}" + "".join(
        f"{COLUMNS[name][0]:>{COLUMN_WIDTH}}" for name in result.metrics if name in COLUMNS
    )
    header += f"{'valid seeds':>{VALID_WIDTH}}"
    lines = [header, "-" * len(header)]
    for i, value in enumerate(result.values):
        cells = "".join(
            f"{format_cell(result.metrics[name][i], COLUMNS[name][1], confidence):>{COLUMN_WIDTH}}"
            for name in result.metrics
            if name in COLUMNS
        )
        valid = {
            name: seed_confidence_interval(result.metrics[name][i], confidence).n_valid
            for name in ("position_rmse", "confirmation_delay")
        }
        counts = f"{valid['position_rmse']}/{valid['confirmation_delay']}"
        lines.append(f"{value:>20g}{cells}{counts:>{VALID_WIDTH}}")
    return lines
