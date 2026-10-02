"""Tests for the sweep plots."""

import subprocess
import sys

import numpy as np
import pytest

from fusion.mtt_experiment import SweepResult
from fusion.mtt_metrics import MttMetrics, seed_confidence_interval
from fusion.mtt_plots import METRIC_TITLES, PARAMETER_LABELS, build_sweep_figure, plot_sweep

pytest.importorskip("matplotlib")

VALUES = np.array([0.0, 5.0, 10.0, 15.0])


def make_result(parameter="clutter_rate") -> SweepResult:
    rng = np.random.default_rng(0)
    metrics = {
        name: 1.0 + np.arange(len(VALUES))[:, None] + rng.normal(0.0, 0.3, size=(len(VALUES), 5))
        for name in MttMetrics._fields
    }
    return SweepResult(parameter, VALUES, metrics)


def visible_axes(fig):
    return [ax for ax in fig.axes if ax.get_visible()]


def test_one_visible_panel_per_metric_titled_and_labelled():
    fig = build_sweep_figure(make_result())
    axes = visible_axes(fig)
    assert len(axes) == len(MttMetrics._fields)
    assert [ax.get_title(loc="left") for ax in axes] == [
        METRIC_TITLES[n] for n in MttMetrics._fields
    ]
    assert all(ax.get_xlabel() == PARAMETER_LABELS["clutter_rate"] for ax in axes)


def test_each_panel_draws_the_seed_mean_and_its_confidence_band():
    result = make_result()
    fig = build_sweep_figure(result, confidence=0.9)
    for ax, name in zip(visible_axes(fig), MttMetrics._fields, strict=True):
        intervals = [seed_confidence_interval(row, 0.9) for row in result.metrics[name]]
        mean = [i.mean for i in intervals]
        np.testing.assert_allclose(ax.lines[0].get_ydata(), mean)
        np.testing.assert_allclose(ax.lines[0].get_xdata(), VALUES)

        # The band polygon must carry exactly the interval bounds.
        band_y = np.concatenate([p.vertices[:, 1] for p in ax.collections[0].get_paths()])
        bounds = [max(i.lower, 0.0) for i in intervals] + [i.upper for i in intervals]
        for bound in bounds:
            assert np.isclose(band_y, bound).any()
        assert ax.get_ylim()[0] == 0.0


def test_the_band_is_wider_for_a_higher_confidence_level():
    result = make_result()

    def band_height(confidence):
        ax = visible_axes(build_sweep_figure(result, confidence))[0]
        ys = np.concatenate([p.vertices[:, 1] for p in ax.collections[0].get_paths()])
        return ys.max() - ys.min()

    assert band_height(0.99) > band_height(0.8)


def test_values_without_an_interval_leave_a_gap_in_the_band():
    result = make_result()
    metrics = {name: values.copy() for name, values in result.metrics.items()}
    metrics["position_rmse"][1] = [2.0, np.nan, np.nan, np.nan, np.nan]  # one valid seed
    metrics["position_rmse"][3] = np.nan  # no valid seed
    fig = build_sweep_figure(SweepResult("pd", VALUES, metrics))
    ax = visible_axes(fig)[0]
    band_x = np.concatenate([p.vertices[:, 0] for p in ax.collections[0].get_paths()])
    assert not np.isclose(band_x, VALUES[1]).any() and not np.isclose(band_x, VALUES[3]).any()
    y = ax.lines[0].get_ydata()
    assert np.isfinite(y[1]) and np.isnan(y[3])  # a point for one seed, none for zero
    assert ax.get_xlabel() == PARAMETER_LABELS["pd"]


def test_title_is_optional():
    assert build_sweep_figure(make_result()).get_suptitle() == ""
    assert build_sweep_figure(make_result(), title="Sweep").get_suptitle() == "Sweep"


def test_plot_sweep_writes_an_image_and_creates_the_directory(tmp_path):
    path = tmp_path / "nested" / "sweep.png"
    assert plot_sweep(make_result(), path) == path
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_importing_the_plot_module_does_not_import_matplotlib():
    code = "import sys, fusion.mtt_plots; print('matplotlib' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_the_band_is_clipped_at_zero_for_nonnegative_scores():
    metrics = {name: values.copy() for name, values in make_result().metrics.items()}
    metrics["id_switches"][1] = [0.0, 0.0, 0.0, 0.0, 3.0]  # interval reaches below zero
    interval = seed_confidence_interval(metrics["id_switches"][1])
    assert interval.lower < 0.0
    fig = build_sweep_figure(SweepResult("pd", VALUES, metrics))
    ax = visible_axes(fig)[list(metrics).index("id_switches")]
    band_y = np.concatenate([p.vertices[:, 1] for p in ax.collections[0].get_paths()])
    assert band_y.min() >= 0.0
    assert np.isclose(band_y, interval.upper).any()


def test_unused_grid_cells_are_hidden():
    result = make_result()
    metrics = {name: values for name, values in result.metrics.items() if name != "births_per_scan"}
    fig = build_sweep_figure(SweepResult(result.parameter, result.values, metrics))
    assert len(visible_axes(fig)) == len(metrics) == 7
    assert len(fig.axes) == 8
