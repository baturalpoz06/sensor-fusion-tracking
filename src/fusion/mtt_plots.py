"""Sweep plots: one panel per metric, mean over seeds with a confidence band.

Requires matplotlib (the "plot" extra); it is imported only when a figure is built.
"""

import math
from pathlib import Path

import numpy as np

from fusion.mtt_experiment import SweepResult
from fusion.mtt_metrics import seed_confidence_interval

METRIC_TITLES = {
    "position_rmse": "Position RMSE [m]",
    "ghost_rate": "Ghost tracks per step (unmatched)",
    "missed_rate": "Missed-target rate",
    "confirmation_delay": "Confirmation delay [s]",
    "confirmed_fraction": "Targets ever confirmed",
    "false_track_rate": "False tracks per step (never matched)",
    "id_switches": "ID switches per run (after burn-in)",
    "births_per_scan": "Track births per radar scan",
}

PARAMETER_LABELS = {
    "clutter_rate": "Clutter rate, radar and camera [points per scan of each sensor]",
    "radar_clutter_rate": "Radar clutter rate [points per radar scan]",
    "camera_clutter_rate": "Camera clutter rate [points per camera scan, one per step]",
    "pd": "Detection probability, radar and camera",
    "radar_pd": "Radar detection probability",
    "camera_pd": "Camera detection probability",
}

# Neutral surface and ink tokens, and one series hue (single series per panel).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e6e5e1"
SERIES = "#2a78d6"
COLUMNS = 4


def build_sweep_figure(result: SweepResult, confidence: float = 0.95, title: str | None = None):
    """Figure with one panel per metric: seed mean as a line, confidence interval as a band.

    Values where fewer than 2 seeds are valid have no band, and values with no
    valid seed no point, so the line breaks instead of inventing numbers.

    Args:
        result: Sweep result from fusion.mtt_experiment.sweep.
        confidence: Coverage of the Student t band across seeds.
        title: Optional figure title.

    Returns:
        A matplotlib Figure (not attached to pyplot; nothing is shown or registered).
    """
    from matplotlib.figure import Figure

    names = list(result.metrics)
    rows = math.ceil(len(names) / COLUMNS)
    fig = Figure(figsize=(4.0 * COLUMNS, 3.1 * rows), facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(rows, COLUMNS, squeeze=False)
    for ax, name in zip(axes.flat, names, strict=False):
        _draw_panel(ax, result, name, confidence)
    for ax in axes.flat[len(names) :]:
        ax.set_visible(False)
    if title:
        fig.suptitle(title, color=INK, fontsize=12, x=0.01, ha="left")
    return fig


def _draw_panel(ax, result: SweepResult, name: str, confidence: float) -> None:
    intervals = [seed_confidence_interval(row, confidence) for row in result.metrics[name]]
    mean = np.array([i.mean for i in intervals])
    # The scores are non-negative: clip the band at zero instead of letting the axis cut it.
    lower = np.maximum([i.lower for i in intervals], 0.0)
    upper = np.array([i.upper for i in intervals])
    x = result.values

    ax.set_facecolor(SURFACE)
    ax.fill_between(
        x,
        lower,
        upper,
        where=np.isfinite(lower) & np.isfinite(upper),
        color=SERIES,
        alpha=0.2,
        linewidth=0,
    )
    ax.plot(x, mean, color=SERIES, linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE)
    ax.set_title(METRIC_TITLES.get(name, name), loc="left", fontsize=10, color=INK)
    ax.set_xlabel(
        PARAMETER_LABELS.get(result.parameter, result.parameter), fontsize=8, color=INK_MUTED
    )
    ax.set_ylim(bottom=0.0)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)


def plot_sweep(
    result: SweepResult,
    path: str | Path,
    confidence: float = 0.95,
    title: str | None = None,
) -> Path:
    """Save the sweep figure as an image; missing parent directories are created.

    Args:
        result: Sweep result from fusion.mtt_experiment.sweep.
        path: Output file; the extension picks the format (e.g. .png).
        confidence: Coverage of the Student t band across seeds.
        title: Optional figure title.

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    build_sweep_figure(result, confidence, title).savefig(path, dpi=150, facecolor=SURFACE)
    return path
