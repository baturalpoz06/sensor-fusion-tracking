"""Tests for the sweep text report."""

import numpy as np
import pytest

from fusion.mtt_experiment import MttConfig, SweepResult
from fusion.mtt_metrics import MttMetrics, seed_confidence_interval
from fusion.mtt_report import (
    COLUMNS,
    MEDIAN_COLUMNS,
    format_cell,
    format_median,
    format_sweep_table,
    settings_caption,
)

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


def test_format_median_skips_nan_and_marks_undefined():
    assert format_median(np.array([1.0, np.nan, 5.0, 100.0])) == "5"
    assert format_median(np.array([np.nan, np.nan])) == "n/a"
    assert format_median(np.array([])) == "n/a"


def test_sweep_table_adds_medians_of_the_heavy_tailed_metrics():
    result = make_result()
    metrics = {name: values.copy() for name, values in result.metrics.items()}
    metrics["position_rmse"][1] = [2.0, 2.1, 1.9, 2.0, 80.0]  # one outlier seed
    lines = format_sweep_table(SweepResult("clutter_rate", VALUES, metrics))
    assert all(text in lines[0] for text in MEDIAN_COLUMNS.values())
    row = lines[3]
    assert f"{format_median(metrics['position_rmse'][1]):>14}" in row
    assert format_median(metrics["position_rmse"][1]) == "2"
    # The mean cell shows what the median does not: the outlier.
    assert format_cell(metrics["position_rmse"][1], 3).startswith("17.6")


def test_sweep_table_caption_and_row_labels():
    result = make_result()
    labels = ["radar-only", "fused", "a", "b"]
    lines = format_sweep_table(result, caption="match distance 150 m", row_labels=labels)
    assert lines[0] == "match distance 150 m"
    assert len(lines) == 3 + len(VALUES)
    assert [line.split()[0] for line in lines[3:]] == labels
    with pytest.raises(ValueError, match="row labels"):
        format_sweep_table(result, row_labels=["only one"])


def test_settings_caption_lists_the_shared_settings():
    from dataclasses import replace

    from fusion.tracker.track import LifecycleConfig

    config = replace(
        MttConfig(),
        match_distance=150.0,
        lifecycle=LifecycleConfig(2, 4, 3),
        use_camera=False,
        radar_pd=0.8,
        camera_clutter_rate=7.0,
    )
    caption = settings_caption(config)
    for text in ["150 m (same at every row)", "2-of-4", "K=3", "99%", "camera off"]:
        assert text in caption
    assert "radar Pd 0.8" in caption and "camera Pd 0.9, clutter 7" in caption
    assert "replaces its own" in caption
