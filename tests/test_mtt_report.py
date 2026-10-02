"""Tests for the sweep text report."""

import numpy as np

from fusion.mtt_experiment import SweepResult
from fusion.mtt_metrics import MttMetrics, seed_confidence_interval
from fusion.mtt_report import COLUMNS, format_cell, format_sweep_table

VALUES = np.array([0.0, 5.0, 10.0, 15.0])


def make_result() -> SweepResult:
    rng = np.random.default_rng(0)
    metrics = {
        name: 1.0 + np.arange(len(VALUES))[:, None] + rng.normal(0.0, 0.3, size=(len(VALUES), 5))
        for name in MttMetrics._fields
    }
    return SweepResult("clutter_rate", VALUES, metrics)


def test_format_cell_shows_mean_and_half_width():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    interval = seed_confidence_interval(values)
    half = (interval.upper - interval.lower) / 2.0
    assert format_cell(values, 3) == f"{interval.mean:.3g} +- {half:.2g}"


def test_format_cell_marks_undefined_values():
    assert format_cell(np.array([np.nan, np.nan]), 3) == "n/a"
    assert format_cell(np.array([2.5, np.nan]), 3) == "2.5 +- ?"


def test_sweep_table_has_a_row_per_value_and_reports_valid_seeds():
    result = make_result()
    metrics = {name: values.copy() for name, values in result.metrics.items()}
    metrics["confirmation_delay"][2, :2] = np.nan  # 3 of 5 seeds valid
    metrics["position_rmse"][0] = np.nan  # no seed valid
    lines = format_sweep_table(SweepResult("clutter_rate", VALUES, metrics), value_label="lambda")
    assert len(lines) == 2 + len(VALUES)
    assert "lambda" in lines[0] and "valid seeds" in lines[0]
    assert all(COLUMNS[name][0] in lines[0] for name in MttMetrics._fields)
    assert lines[2].split()[-1] == "0/5" and "n/a" in lines[2]
    assert lines[4].split()[-1] == "5/3"
    assert [float(line.split()[0]) for line in lines[2:]] == list(VALUES)
