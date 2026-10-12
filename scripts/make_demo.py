"""Portfolio demo: a short animation of the tracker on three scenes, drawn from real outputs.

Usage:
    python scripts/make_demo.py [--out results/demo] [--scenes 1 2 3] [--fps 10]
        [--size 1280x720] [--formats gif webp] [--preview]

Every frame shows the recorded state of one simulation step; nothing is invented or interpolated.
Only public outputs are used: the simulation (truth and scans), the tracker's snapshots per step
(track id, status, state, covariance) and its documented camera log. The tracker does not report
which radar measurement each track took, so radar-to-track pairings are not drawn.

  1 Radar + camera fusion. Single target ('near', seed 0): ground truth, radar measurements
    (1 Hz), camera bearing rays (10 Hz, bearing only), the radar-only EKF and the fused EKF.
  2 Crossing targets. Four targets ('crossing', clutter 0, seed 35): ground truth, confirmed
    and tentative tracks with their ids, camera updates (the bearing each track took) and the
    camera updates withheld by the ambiguity rule.
  3 Radar outage. Three targets ('separated', clutter 0, seed 0), the radar silent from 20 s
    for 12 s, camera active throughout: the unaware and the outage-aware tracker side by side
    on identical data, with the 95% position ellipses of the actual covariance.

Playback is faster than real time (printed in each frame). A frame is drawn for a subset of the
steps; the radar detections of the steps in between stay visible, fading, so no scan is skipped.
Writes <out>/portfolio_demo.gif and .webp (draft assets; results/ is git-ignored) and a
provenance file; --preview also writes sample PNG frames of every scene to <out>/preview/.
"""

import argparse
import hashlib
import io
import math
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import NamedTuple

import numpy as np
from scipy.stats import chi2

from fusion.dropout import SingleOutage, apply_dropout
from fusion.experiment import SCENARIOS as SINGLE_TARGET_SCENARIOS
from fusion.experiment import (
    FusionConfig,
    evaluation_masks,
    make_rngs,
    position_rmse,
    simulate,
)
from fusion.mtt_experiment import SCENARIOS as MTT_SCENARIOS
from fusion.mtt_experiment import MttConfig
from fusion.mtt_metrics import MttMetrics, evaluate_mtt, match_tracks
from fusion.mtt_simulation import MttSimulation, make_mtt_rngs, simulate_mtt
from fusion.outage_experiment import OutageConfig
from fusion.provenance import provenance_lines
from fusion.sensors.camera import CameraModel
from fusion.sensors.radar import RadarModel, radar_to_cartesian
from fusion.tracker.multi_target import (
    CameraStepLog,
    TrackerConfig,
    TrackSnapshot,
    run_multi_target_tracking,
)
from fusion.tracker.track import TrackStatus
from fusion.tracking import TrackingResult, run_tracking_with_covariance

ROOT = Path(__file__).resolve().parents[1]
FUSION_SEED, CROSSING_SEED, OUTAGE_SEED = 0, 35, 0
OUTAGE_START, OUTAGE_LENGTH = 20.0, 12.0  # [s]
FPS = 10
SIZE = (1280, 720)
HOLD_FRAMES = 10  # copies of the last frame, so that the final state can be read
CONFIDENCE = 0.95  # of the drawn position ellipses
ELLIPSE_MAGNIFICATION = 20.0  # the outage ellipses are drawn this many times larger
RADAR_LIFETIME = 1.5  # [s] a radar detection stays visible, fading
CAMERA_LIFETIME = 0.5  # [s] a camera bearing (ray or tick) stays visible, fading
ZOOM_HALF_WIDTH = {"fusion": 30.0, "crossing": 100.0}  # [m]
# Playback segments (start [s], end [s], steps per frame): slower during the acquisition of the
# crossing scene and during the crossing itself.
FUSION_SEGMENTS = ((0.0, 60.01, 5),)
CROSSING_SEGMENTS = ((0.0, 4.0, 2), (4.0, 24.0, 5), (24.0, 36.0, 2), (36.0, 60.01, 5))
OUTAGE_SEGMENTS = ((8.0, 48.0, 3),)

# Neutral surface and ink tokens as in src/fusion/mtt_plots.py; one fixed hue per role.
SURFACE, INK, INK_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
TRUTH, FUSED, RADAR_ONLY = "#8a8984", "#2a78d6", "#d9822b"
RADAR_POINT, CAMERA_RAY, WARNING = "#7a3fb0", "#2e9e6b", "#c4403b"
ID_COLORS = ("#2a78d6", "#d9822b", "#2e9e6b", "#b03fa0", "#c4403b", "#1f9bb0", "#8c6d1f", "#5b5bd6")


# --- playback schedule and geometry --------------------------------------------------------


def frame_steps(segments: Sequence[tuple[float, float, int]], dt: float, n_steps: int) -> list[int]:
    """Steps that get a frame: every `stride` steps inside each segment [start, end).

    A segment starts at its first step at or after `start`, but after the last step of the
    previous segment; step n_steps is included if a segment reaches it.

    Raises:
        ValueError: If a stride is below 1.
    """
    steps: list[int] = []
    for start, end, stride in segments:
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        first = max(math.ceil(start / dt - 1e-9), steps[-1] + 1 if steps else 0)
        stop = min(math.ceil(end / dt - 1e-9), n_steps + 1)
        steps.extend(range(first, stop, stride))
    return steps


def segment_stride(segments: Sequence[tuple[float, float, int]], time: float) -> int:
    """Steps per frame of the segment that contains the time (the last one past the end)."""
    return next((s for start, end, s in segments if start <= time < end), segments[-1][2])


def window_steps(steps: Sequence[int], index: int) -> range:
    """Steps whose events belong to frame `index`: after the previous frame, up to this one."""
    k = steps[index]
    return range((steps[index - 1] if index else k - 1) + 1, k + 1)


def fade(age: float, lifetime: float) -> float:
    """Opacity of an observation of the given age [s]: 1 when new, 0 at the lifetime."""
    return float(np.clip(1.0 - age / lifetime, 0.0, 1.0))


def ellipse_geometry(
    cov2: np.ndarray, confidence: float = CONFIDENCE, magnification: float = 1.0
) -> tuple[float, float, float]:
    """Full width, full height and angle [deg] of the confidence ellipse of a 2x2 covariance.

    The ellipse {x : x^T C^-1 x <= chi2(confidence, 2)} has semi-axes sqrt(chi2 * eigenvalue);
    `magnification` scales it for display. The width lies along the major axis, and the angle
    is that axis counter-clockwise from +x.
    """
    cov = 0.5 * (np.asarray(cov2, dtype=float) + np.asarray(cov2, dtype=float).T)
    values, vectors = np.linalg.eigh(cov)
    major = vectors[:, -1]
    scale = math.sqrt(chi2.ppf(confidence, df=2)) * magnification
    return (
        2.0 * scale * math.sqrt(max(values[-1], 0.0)),
        2.0 * scale * math.sqrt(max(values[0], 0.0)),
        math.degrees(math.atan2(major[1], major[0])),
    )


def bearing_segment(bearing: float, start: float, end: float) -> np.ndarray:
    """Endpoints, shape (2, 2), of the part of a bearing ray from the origin between two ranges.

    A bearing-only measurement fixes the direction of the line of sight and nothing along it,
    so it is drawn as a line, never as a point.
    """
    direction = np.array([math.cos(bearing), math.sin(bearing)])
    return np.array([start * direction, end * direction])


# --- scenes: simulation and tracking with public outputs -------------------------------------


class TrackedRun(NamedTuple):
    """A tracker run as its public outputs report it.

    Attributes:
        history: history[k] holds the snapshots (id, status, state, covariance) after step k.
        camera_log: One CameraStepLog per step: the tracks offered the camera scan, the tracks
            withheld because their bearing gates overlap, and (track id, camera row) updates.
    """

    history: list[list[TrackSnapshot]]
    camera_log: list[CameraStepLog]


def track(
    sim: MttSimulation, config: TrackerConfig, radar_model: RadarModel,
    camera_model: CameraModel, radar_down: np.ndarray | None = None,
) -> TrackedRun:
    """run_multi_target_tracking, keeping the tracker's camera log next to the history."""
    run = run_multi_target_tracking(sim, config, radar_model, camera_model, radar_down)
    return TrackedRun(run.history, list(run.tracker.camera_log))


class FusionScene(NamedTuple):
    """Single-target scene: truth, measurements and the two EKF runs (steps 0..n)."""

    config: FusionConfig
    seed: int
    truth: np.ndarray  # (n + 1, 4)
    radar_z: np.ndarray  # (n + 1, 2); only the rows of the radar steps are measurements
    camera_z: np.ndarray  # (n + 1, 1)
    radar_only: TrackingResult
    fused: TrackingResult
    rmse: dict[str, float]  # at the radar steps after the burn-in: radar_only, fused, raw


def build_fusion_scene(seed: int = FUSION_SEED, config: FusionConfig | None = None) -> FusionScene:
    """Simulate the 'near' single-target scenario and run both filters on it."""
    config = FusionConfig() if config is None else config
    sim = simulate(SINGLE_TARGET_SCENARIOS["near"], config, make_rngs(seed))
    common = dict(
        dt=config.dt,
        radar_every=config.radar_every,
        radar_z=sim.radar_z,
        radar_model=RadarModel(config.radar_range_std, config.radar_bearing_std),
        accel_std=config.accel_std,
        velocity_std=config.velocity_std,
    )
    radar_only = run_tracking_with_covariance(sim.truth, use_camera=False, **common)
    fused = run_tracking_with_covariance(
        sim.truth, camera_z=sim.camera_z, camera_model=CameraModel(config.camera_bearing_std),
        use_camera=True, **common,
    )
    mask = evaluation_masks(len(sim.truth), config.radar_every, config.burn_in_steps)["radar"]
    rmse = {
        "radar_only": position_rmse(radar_only.estimates[:, :2], sim.truth, mask),
        "fused": position_rmse(fused.estimates[:, :2], sim.truth, mask),
        "raw": position_rmse(radar_to_cartesian(sim.radar_z), sim.truth, mask),
    }
    return FusionScene(config, seed, sim.truth, sim.radar_z, sim.camera_z, radar_only, fused, rmse)


class TrackingScene(NamedTuple):
    """Multi-target scene with one tracker run.

    Attributes:
        closest: Closest approach of targets 0 and 1 in meters and the time [s] it happens.
        matched_ids: Shape (n_targets, n + 1), id of the confirmed track matched to each target
            at each step (-1 if none); evaluation only, it uses the truth.
    """

    config: MttConfig
    seed: int
    sim: MttSimulation
    run: TrackedRun
    metrics: MttMetrics
    closest: tuple[float, float]
    matched_ids: np.ndarray


def build_crossing_scene(
    seed: int = CROSSING_SEED, duration: float | None = None, burn_in: float | None = None
) -> TrackingScene:
    """Simulate the 'crossing' scenario without clutter and track it."""
    changes = {"radar_clutter_rate": 0.0, "camera_clutter_rate": 0.0}
    if duration is not None:
        changes["duration"] = duration
    if burn_in is not None:
        changes["burn_in"] = burn_in
    config = replace(MttConfig(), **changes)
    sim = simulate_mtt(MTT_SCENARIOS["crossing"], config, make_mtt_rngs(seed))
    run = track(
        sim, config.tracker_config(),
        RadarModel(config.radar_range_std, config.radar_bearing_std),
        CameraModel(config.camera_bearing_std),
    )
    metrics = evaluate_mtt(
        sim.truth, run.history, dt=config.dt, fov=config.fov,
        max_distance=config.match_distance, burn_in_steps=config.burn_in_steps,
    )
    separation = np.hypot(*(sim.truth[0, :, :2] - sim.truth[1, :, :2]).T)
    closest = (float(separation.min()), float(separation.argmin() * config.dt))
    matched = match_tracks(sim.truth, run.history, config.match_distance).ids
    return TrackingScene(config, seed, sim, run, metrics, closest, matched)


class OutageScene(NamedTuple):
    """Radar outage scene: one simulation, one tracker run per policy on that same data."""

    config: OutageConfig
    seed: int
    sim: MttSimulation  # with the outage applied (empty radar scans while the radar is down)
    radar_down: np.ndarray  # (n + 1,) True at the steps in which the radar is silent
    runs: dict[str, TrackedRun]  # "unaware" and "aware"


def build_outage_scene(
    seed: int = OUTAGE_SEED, start: float = OUTAGE_START, length: float = OUTAGE_LENGTH,
    duration: float | None = None,
) -> OutageScene:
    """Simulate the 'separated' scenario with a radar outage and track it with both policies."""
    changes = {
        "radar_clutter_rate": 0.0,
        "camera_clutter_rate": 0.0,
        "dropout": SingleOutage(("radar",), start, length),
    }
    if duration is not None:
        changes["duration"] = duration
    config = replace(OutageConfig(), **changes)
    rngs = make_mtt_rngs(seed)
    sim = simulate_mtt(MTT_SCENARIOS["separated"], config, rngs)
    dropped = apply_dropout(sim, config.dropout.windows(config.dt, rngs["dropout"]), config.dt)
    radar_model = RadarModel(config.radar_range_std, config.radar_bearing_std)
    camera_model = CameraModel(config.camera_bearing_std)
    runs = {
        policy: track(dropped.sim, replace(config, outage_policy=policy).tracker_config(),
                      radar_model, camera_model, dropped.radar_down)
        for policy in ("unaware", "aware")
    }
    return OutageScene(config, seed, dropped.sim, dropped.radar_down, runs)


class TrackSeries(NamedTuple):
    """The confirmed part of one track's history: steps, position, covariance, major axis."""

    steps: np.ndarray
    xy: np.ndarray
    cov: np.ndarray
    major: np.ndarray


def confirmed_series(history: Sequence[Sequence[TrackSnapshot]]) -> dict[int, TrackSeries]:
    """Per track id, its confirmed snapshots in order, from the tracker's history."""
    rows: dict[int, list] = {}
    for step, snapshots in enumerate(history):
        for s in snapshots:
            if s.status is TrackStatus.CONFIRMED:
                rows.setdefault(s.track_id, []).append((step, s.x[:2], s.P[:2, :2]))
    series = {}
    for track_id, items in rows.items():
        cov = np.array([c for _, _, c in items])
        sym = 0.5 * (cov + cov.transpose(0, 2, 1))
        series[track_id] = TrackSeries(
            np.array([s for s, _, _ in items]), np.array([x for _, x, _ in items]), cov,
            np.sqrt(np.maximum(np.linalg.eigvalsh(sym)[:, -1], 0.0)),
        )
    return series


def id_colors(*series_sets: dict[int, TrackSeries]) -> dict[int, str]:
    """One color per track id over all given runs, in order of first appearance."""
    colors: dict[int, str] = {}
    for series in series_sets:
        for track_id in series:
            colors.setdefault(track_id, ID_COLORS[len(colors) % len(ID_COLORS)])
    return colors


# --- drawing helpers -----------------------------------------------------------------------------


def pyplot():
    """matplotlib.pyplot with the non-interactive Agg backend (imported on first use)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


class Style(NamedTuple):
    """Font sizes scaled to the frame height (720 px is the reference)."""

    title: float
    body: float
    small: float

    @classmethod
    def for_height(cls, height: int) -> "Style":
        s = height / 720.0
        return cls(15.0 * s, 10.5 * s, 8.5 * s)


def new_figure(size: tuple[int, int]):
    """An empty figure of exactly `size` pixels (dpi 100)."""
    return pyplot().figure(figsize=(size[0] / 100.0, size[1] / 100.0), dpi=100, facecolor=SURFACE)


def style_axes(ax, style: Style, xlabel: str = "", ylabel: str = "") -> None:
    """Quiet axes: light grid, no top/right spines, muted ticks."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelsize=style.small)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, fontsize=style.small, color=INK_MUTED)
    ax.set_ylabel(ylabel, fontsize=style.small, color=INK_MUTED)


def header(fig, style: Style, title: str, subtitle: str, note: str) -> None:
    """Scene title and subtitle at the top and a provenance note (1-2 lines) at the bottom."""
    fig.text(0.02, 0.953, title, fontsize=style.title, fontweight="bold", color=INK, va="center")
    fig.text(0.02, 0.912, subtitle, fontsize=style.small * 1.15, color=INK_MUTED, va="center")
    fig.text(0.02, 0.022, note, fontsize=style.small, color=INK_MUTED, va="center",
             linespacing=1.4)


def to_png(fig) -> bytes:
    """The figure as PNG bytes of exactly its pixel size."""
    from PIL import Image

    fig.canvas.draw()
    image = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", compress_level=1)
    return buffer.getvalue()


def padded_bounds(points: np.ndarray, margin: float):
    """(xlim, ylim) around points, shape (m, 2), with a relative margin on each side."""
    low, high = points.min(axis=0), points.max(axis=0)
    pad = margin * (high - low)
    return (low[0] - pad[0], high[0] + pad[0]), (low[1] - pad[1], high[1] + pad[1])


def text_lines(ax, lines: Sequence[tuple[str, str]], style: Style) -> None:
    """Lines of (text, color) stacked from the top of an axis-less panel."""
    ax.axis("off")
    step = 1.55 * style.body / (ax.get_window_extent().height / 100.0 * 72.0)
    for i, (text, color) in enumerate(lines):
        ax.text(0.0, 0.98 - i * step, text, fontsize=style.body, color=color, va="top",
                transform=ax.transAxes)


def label(ax, text: str, xy, offset: tuple[float, float], color: str, size: float,
          weight: str = "normal") -> None:
    """Text anchored at a data point with an offset in points; hidden if the point is outside."""
    ax.annotate(text, xy=xy, xytext=offset, textcoords="offset points", fontsize=size,
                color=color, fontweight=weight, zorder=8)


def legend(ax, handles: Sequence, style: Style, columns: int = 1) -> None:
    """A frameless legend filling an axis-less panel."""
    ax.axis("off")
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=style.small * 1.1,
              ncol=columns)


def handle(kind: str, color: str, text: str, **kwargs):
    """Legend entry: kind is 'line', 'marker' or 'hollow' (a marker with no fill)."""
    from matplotlib.lines import Line2D

    if kind == "line":
        return Line2D([], [], color=color, label=text, **kwargs)
    if kind == "hollow":
        return Line2D([], [], ls="", mfc="none", mec=color, label=text, **kwargs)
    return Line2D([], [], ls="", color=color, label=text, **kwargs)


# --- drawing: layers shared by the multi-target scenes ----------------------------------------


class MapState(NamedTuple):
    """What the multi-target layers need for one frame."""

    sim: MttSimulation
    run: TrackedRun
    series: dict[int, TrackSeries]
    colors: dict[int, str]
    dt: float
    k: int
    window: range


def draw_truth(ax, truth: np.ndarray, k: int, style: Style) -> None:
    """Faint ground-truth paths and the current true positions (T0, T1, ...)."""
    for i in range(truth.shape[0]):
        ax.plot(truth[i, :, 0], truth[i, :, 1], color=TRUTH, lw=0.8, alpha=0.4, zorder=1)
        ax.scatter(truth[i, k, 0], truth[i, k, 1], s=42, facecolors=SURFACE, edgecolors=INK,
                   linewidths=1.2, zorder=7)
        label(ax, f"T{i}", truth[i, k, :2], (7, -13), INK, style.small * 1.1)


def draw_radar_detections(ax, state: MapState, size: float = 34) -> None:
    """Radar detections of the last RADAR_LIFETIME seconds, fading, from the radar scans."""
    for s in range(max(0, state.k - int(RADAR_LIFETIME / state.dt)), state.k + 1):
        scan = state.sim.radar_scans[s]
        if scan is None or len(scan.z) == 0:
            continue
        pts = radar_to_cartesian(scan.z)
        ax.scatter(pts[:, 0], pts[:, 1], marker="s", s=size, facecolors="none",
                   edgecolors=RADAR_POINT, linewidths=1.2,
                   alpha=max(fade((state.k - s) * state.dt, RADAR_LIFETIME), 0.15), zorder=4)


def draw_camera_rays(ax, state: MapState, length: float) -> None:
    """Camera bearing rays of the frame's step, from the camera at the origin."""
    for z in state.sim.camera_scans[state.k].z[:, 0]:
        seg = bearing_segment(float(z), 0.0, length)
        ax.plot(seg[:, 0], seg[:, 1], color=CAMERA_RAY, lw=0.5, alpha=0.3, zorder=2)


def draw_camera_updates(ax, state: MapState, tick: float, width: float) -> None:
    """From the tracker's camera log, for the steps since the previous frame: a tick on the
    bearing each track took, and a ring around each track whose camera update was withheld."""
    for s in state.window:
        entry = state.run.camera_log[s]
        alpha = max(fade((state.k - s) * state.dt, CAMERA_LIFETIME), 0.2)
        by_id = {snap.track_id: snap for snap in state.run.history[s]}
        for track_id, row in entry.assigned:
            x = by_id[track_id].x
            distance = float(np.hypot(x[0], x[1]))
            seg = bearing_segment(float(state.sim.camera_scans[s].z[row, 0]),
                                  distance - tick, distance + tick)
            ax.plot(seg[:, 0], seg[:, 1], color=state.colors.get(track_id, INK), lw=width,
                    alpha=0.9 * alpha, zorder=6)
        for track_id in entry.skipped:
            x = by_id[track_id].x
            ax.scatter(x[0], x[1], s=190, facecolors="none", edgecolors=WARNING, linewidths=1.4,
                       alpha=alpha, zorder=6)


def draw_tracks(ax, state: MapState, style: Style, ellipse_magnification: float | None = None):
    """Confirmed tracks (trail, marker, id, optional position ellipse), tentative tracks hollow."""
    from matplotlib.patches import Ellipse

    for snap in state.run.history[state.k]:
        if snap.status is TrackStatus.TENTATIVE:
            ax.scatter(snap.x[0], snap.x[1], marker="D", s=26, facecolors="none",
                       edgecolors=INK_MUTED, linewidths=0.9, zorder=5)
            continue
        color = state.colors[snap.track_id]
        series = state.series[snap.track_id]
        count = int(np.searchsorted(series.steps, state.k, side="right"))
        ax.plot(series.xy[:count, 0], series.xy[:count, 1], color=color, lw=1.3, zorder=3)
        if ellipse_magnification is not None:
            width, height, angle = ellipse_geometry(snap.P[:2, :2], CONFIDENCE,
                                                    ellipse_magnification)
            ax.add_patch(Ellipse(snap.x[:2], width, height, angle=angle, facecolor=color,
                                 alpha=0.25, edgecolor=color, lw=1.0, zorder=4))
        ax.scatter(snap.x[0], snap.x[1], s=34, color=color, zorder=6)
        label(ax, f"ID {snap.track_id}", snap.x[:2], (7, 6), color, style.small * 1.15, "bold")


# --- drawing: scene 1 --------------------------------------------------------------------------


def draw_fusion_frame(fig, scene: FusionScene, steps: Sequence[int], index: int, speed: float,
                      size: tuple[int, int]) -> None:
    """Draw frame `index` of scene 1 onto `fig` (cleared first)."""
    style = Style.for_height(size[1])
    cfg, truth = scene.config, scene.truth
    k = steps[index]
    window = window_steps(steps, index)
    t = k * cfg.dt
    fig.clf()
    header(
        fig, style, "Scene 1 - Radar + camera fusion (single target)",
        f"Scenario 'near', seed {scene.seed} - radar every {cfg.radar_every * cfg.dt:g} s "
        f"(range {cfg.radar_range_std:g} m, bearing {math.degrees(cfg.radar_bearing_std):g} deg), "
        f"camera every {cfg.dt:g} s (bearing only, {math.degrees(cfg.camera_bearing_std):g} deg) "
        f"- playback x{speed:g}",
        "Simulated data from the fusion package, not real sensor data.",
    )
    ray_length = float(np.hypot(*truth[:, :2].T).max()) * 1.05
    radar_steps = [s for s in range(max(0, k - int(RADAR_LIFETIME / cfg.dt)), k + 1)
                   if s % cfg.radar_every == 0]

    # overview map
    ax = fig.add_axes([0.05, 0.40, 0.60, 0.46])
    style_axes(ax, style, "x [m]", "y [m]")
    ax.set_aspect("equal", adjustable="box")
    xlim, ylim = padded_bounds(np.vstack([truth[:, :2], [[0.0, 0.0]]]), 0.06)
    ax.set_xlim(*xlim)
    ax.set_ylim(min(ylim[0], -60.0), ylim[1] + 60.0)
    ax.plot(truth[:, 0], truth[:, 1], color=TRUTH, lw=0.8, alpha=0.4, zorder=1)
    ax.plot(0.0, 0.0, marker="^", color=INK, ms=7, zorder=6)
    label(ax, "radar + camera", (0.0, 0.0), (7, -14), INK, style.small)
    for s in window:
        a = fade((k - s) * cfg.dt, CAMERA_LIFETIME)
        seg = bearing_segment(float(scene.camera_z[s, 0]), 0.0, ray_length)
        ax.plot(seg[:, 0], seg[:, 1], color=CAMERA_RAY, lw=0.6, alpha=0.5 * a, zorder=2)
    for s in radar_steps:
        p = radar_to_cartesian(scene.radar_z[[s]])[0]
        ax.scatter(p[0], p[1], marker="s", s=34, facecolors="none", edgecolors=RADAR_POINT,
                   linewidths=1.3, alpha=max(fade((k - s) * cfg.dt, RADAR_LIFETIME), 0.15),
                   zorder=4)
    for run, color in ((scene.radar_only, RADAR_ONLY), (scene.fused, FUSED)):
        ax.plot(run.estimates[: k + 1, 0], run.estimates[: k + 1, 1], color=color, lw=1.2, zorder=3)
        ax.scatter(run.estimates[k, 0], run.estimates[k, 1], s=26, color=color, zorder=5)
    ax.scatter(truth[k, 0], truth[k, 1], s=60, facecolors=SURFACE, edgecolors=INK, linewidths=1.4,
               zorder=7)

    # zoom on the target, in offsets from the true position
    half = ZOOM_HALF_WIDTH["fusion"]
    zoom = fig.add_axes([0.72, 0.47, 0.26, 0.36])
    style_axes(zoom, style, "x offset from truth [m]", "y offset from truth [m]")
    zoom.set_aspect("equal", adjustable="box")
    zoom.set_xlim(-half, half)
    zoom.set_ylim(-half, half)
    zoom.set_title(f"Zoom on the target (+/-{half:g} m)", fontsize=style.body, color=INK,
                   loc="left")
    zoom.axhline(0.0, color=GRID, lw=0.8)
    zoom.axvline(0.0, color=GRID, lw=0.8)
    for s in window:  # a bearing ray, expressed relative to the true position
        u = bearing_segment(float(scene.camera_z[s, 0]), 0.0, 1.0)[1]
        reach = float(truth[k, :2] @ u)
        ends = np.array([(reach + d) * u - truth[k, :2] for d in (-3 * half, 3 * half)])
        zoom.plot(ends[:, 0], ends[:, 1], color=CAMERA_RAY, lw=1.0,
                  alpha=0.6 * fade((k - s) * cfg.dt, CAMERA_LIFETIME), zorder=2)
    for s in radar_steps:
        p = radar_to_cartesian(scene.radar_z[[s]])[0] - truth[k, :2]
        zoom.scatter(p[0], p[1], marker="s", s=60, facecolors="none", edgecolors=RADAR_POINT,
                     linewidths=1.6, alpha=max(fade((k - s) * cfg.dt, RADAR_LIFETIME), 0.2),
                     zorder=4)
    errors = {}
    for name, run, color in (("radar-only", scene.radar_only, RADAR_ONLY),
                             ("fused", scene.fused, FUSED)):
        offset = run.estimates[k, :2] - truth[k, :2]
        errors[name] = float(np.hypot(*offset))
        zoom.scatter(offset[0], offset[1], s=70, color=color, zorder=5)
    zoom.scatter(0.0, 0.0, marker="+", s=140, color=INK, linewidths=1.6, zorder=6)

    # error over time
    err = fig.add_axes([0.72, 0.12, 0.26, 0.23])
    style_axes(err, style, "time [s]", "position error [m]")
    times = np.arange(len(truth)) * cfg.dt
    curves = [(np.hypot(*(r.estimates[:, :2] - truth[:, :2]).T), c)
              for r, c in ((scene.radar_only, RADAR_ONLY), (scene.fused, FUSED))]
    top = max(float(e.max()) for e, _ in curves) * 1.08
    err.set_xlim(0.0, times[-1])
    err.set_ylim(0.0, top)
    err.axvspan(0.0, cfg.burn_in, color=GRID, alpha=0.7, lw=0)
    err.plot(times[:: cfg.radar_every], np.zeros(len(times[:: cfg.radar_every])), "|",
             color=RADAR_POINT, ms=6)
    for e, color in curves:
        err.plot(times[: k + 1], e[: k + 1], color=color, lw=1.1)
    err.axvline(t, color=INK_MUTED, lw=0.8)
    err.set_title("Position error over time", fontsize=style.body, color=INK, loc="left")

    # legend and numbers
    info = fig.add_axes([0.05, 0.07, 0.60, 0.26])
    legend(info, [
        handle("line", TRUTH, "ground truth", lw=1.2),
        handle("hollow", RADAR_POINT, "radar measurement (1 Hz)", marker="s"),
        handle("line", CAMERA_RAY, "camera bearing ray (10 Hz, no range)", lw=1.2),
        handle("line", RADAR_ONLY, "radar-only EKF", marker="o", ms=5),
        handle("line", FUSED, "fused EKF (radar + camera)", marker="o", ms=5),
    ], style, columns=2)
    info.text(0.0, 0.36, f"t = {t:4.1f} s     error now: radar-only {errors['radar-only']:4.1f} m, "
                         f"fused {errors['fused']:4.1f} m",
              fontsize=style.body, color=INK, va="top", transform=info.transAxes)
    info.text(0.0, 0.16, f"Seed {scene.seed} only, whole run, RMSE at radar steps after "
                         f"{cfg.burn_in:g} s: radar-only {scene.rmse['radar_only']:.2f} m, "
                         f"fused {scene.rmse['fused']:.2f} m (raw radar {scene.rmse['raw']:.2f} m)",
              fontsize=style.small * 1.1, color=INK_MUTED, va="top", transform=info.transAxes)
    info.text(0.0, 0.04, f"Grey band in the error plot: the {cfg.burn_in:g} s burn-in, excluded "
                         "from the RMSE. Purple ticks: radar scans.",
              fontsize=style.small * 1.1, color=INK_MUTED, va="top", transform=info.transAxes)


# --- drawing: scene 2 --------------------------------------------------------------------------


def draw_crossing_frame(fig, scene: TrackingScene, steps: Sequence[int], index: int, speed: float,
                        size: tuple[int, int], series: dict[int, TrackSeries],
                        colors: dict[int, str]) -> None:
    """Draw frame `index` of scene 2 onto `fig` (cleared first)."""
    style = Style.for_height(size[1])
    cfg, sim = scene.config, scene.sim
    k = steps[index]
    state = MapState(sim, scene.run, series, colors, cfg.dt, k, window_steps(steps, index))
    fig.clf()
    header(
        fig, style, "Scene 2 - Multi-target tracking, crossing targets",
        f"Scenario 'crossing', clutter 0, seed {scene.seed} - detection probability "
        f"{cfg.radar_pd:g} - 3-of-5 confirmation - playback x{speed:g}",
        "Simulated data from the fusion package, not real sensor data. T0, T1, ... are simulator "
        "labels; ID n is the tracker's own track id.\nThe tracker does not report which radar "
        "detection each track took, so radar-to-track pairings are not drawn.",
    )

    ax = fig.add_axes([0.04, 0.12, 0.38, 0.76])
    style_axes(ax, style, "x [m]", "y [m]")
    ax.set_aspect("equal", adjustable="box")
    xlim, ylim = padded_bounds(np.vstack([sim.truth[:, :, :2].reshape(-1, 2), [[0.0, 0.0]]]), 0.05)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    draw_truth(ax, sim.truth, k, style)
    ax.plot(0.0, 0.0, marker="^", color=INK, ms=7, zorder=6)
    label(ax, "radar + camera", (0.0, 0.0), (7, 4), INK, style.small)
    draw_camera_rays(ax, state, 3200.0)
    draw_radar_detections(ax, state)
    draw_camera_updates(ax, state, 50.0, 2.0)
    draw_tracks(ax, state, style)

    # zoom on the pair T0/T1
    mid = 0.5 * (sim.truth[0, k, :2] + sim.truth[1, k, :2])
    separation = float(np.hypot(*(sim.truth[0, k, :2] - sim.truth[1, k, :2])))
    half = ZOOM_HALF_WIDTH["crossing"]
    zoom = fig.add_axes([0.50, 0.46, 0.235, 0.38])
    style_axes(zoom, style, "x [m]", "y [m]")
    zoom.set_aspect("equal", adjustable="box")
    zoom.set_xlim(mid[0] - half, mid[0] + half)
    zoom.set_ylim(mid[1] - half, mid[1] + half)
    zoom.set_title(f"Zoom on T0/T1 (+/-{half:g} m around their midpoint)",
                   fontsize=style.small * 1.2, color=INK, loc="left")
    draw_truth(zoom, sim.truth, k, style)
    draw_radar_detections(zoom, state, 60)
    draw_camera_updates(zoom, state, 8.0, 2.4)
    draw_tracks(zoom, state, style)
    zoom.text(0.02, 0.03, f"T0-T1 separation {separation:5.1f} m"
              + ("" if separation < 2.0 * half else " (outside window)"),
              transform=zoom.transAxes, fontsize=style.small * 1.1, color=INK)

    # information
    snapshots = scene.run.history[k]
    entry = scene.run.camera_log[k]
    ids = ", ".join(str(s.track_id) for s in snapshots if s.status is TrackStatus.CONFIRMED)
    matched = []
    for i in range(sim.truth.shape[0]):
        found = scene.matched_ids[i, k]
        matched.append(f"T{i}: " + (f"ID {found}" if found >= 0 else "-"))
    text_lines(fig.add_axes([0.765, 0.46, 0.22, 0.38]), [
        (f"t = {k * cfg.dt:4.1f} s", INK),
        (f"confirmed tracks: {ids or 'none yet'}", INK),
        (f"tentative tracks: {sum(s.status is TrackStatus.TENTATIVE for s in snapshots)}", INK),
        (f"camera updates: {len(entry.assigned)}", INK),
        (f"camera withheld: {', '.join(map(str, entry.skipped)) or 'none'}", INK),
        ("", INK),
        ("Evaluation match (uses truth):", INK_MUTED),
        ("  ".join(matched[:2]), INK_MUTED),
        ("  ".join(matched[2:]), INK_MUTED),
    ], style)
    fig.text(0.50, 0.395,
             f"Whole run: T0-T1 closest approach {scene.closest[0]:.1f} m at "
             f"{scene.closest[1]:.1f} s; ID switches {scene.metrics.id_switches}",
             fontsize=style.small * 1.1, color=INK_MUTED, va="center")
    fig.text(0.50, 0.365, "(ID switches: truth-matched at 50 m, after the 5 s burn-in)",
             fontsize=style.small * 1.1, color=INK_MUTED, va="center")
    legend(fig.add_axes([0.50, 0.09, 0.48, 0.24]), [
        handle("marker", INK, "ground truth (T0, T1, ...)", marker="o", mfc=SURFACE, mec=INK),
        handle("hollow", RADAR_POINT, "radar detection (1 Hz)", marker="s"),
        handle("line", CAMERA_RAY, "camera bearing ray (10 Hz)", lw=1.0),
        handle("marker", FUSED, "confirmed track + id", marker="o"),
        handle("hollow", INK_MUTED, "tentative track", marker="D"),
        handle("line", INK, "camera update (tick on the bearing taken)", lw=2.4),
        handle("hollow", WARNING, "camera update withheld (gates overlap)", marker="o", ms=9),
    ], style, columns=2)


# --- drawing: scene 3 --------------------------------------------------------------------------


def draw_outage_frame(fig, scene: OutageScene, steps: Sequence[int], index: int, speed: float,
                      size: tuple[int, int], series: dict[str, dict[int, TrackSeries]],
                      colors: dict[int, str]) -> None:
    """Draw frame `index` of scene 3 onto `fig` (cleared first)."""
    style = Style.for_height(size[1])
    cfg, sim = scene.config, scene.sim
    k = steps[index]
    window = window_steps(steps, index)
    start, end = cfg.dropout.span
    fig.clf()
    header(
        fig, style, "Scene 3 - Radar outage: unaware vs outage-aware tracker",
        f"Scenario 'separated', clutter 0, seed {scene.seed} - radar silent {start:g}-{end:g} s, "
        f"camera active throughout - identical simulated data in both panels - playback x{speed:g}",
        f"Simulated data from the fusion package, not real sensor data. Ellipses: {CONFIDENCE:.0%} "
        f"confidence region of the track's position covariance P[:2,:2], drawn "
        f"{ELLIPSE_MAGNIFICATION:g}x larger than its true size.\n"
        "Strip: 1-sigma semi-major axis of the same covariance (true size). "
        "Triangles: scheduled radar scans (filled: delivered, hollow: radar down).",
    )
    ymax = 1.1 * max(float(s.major.max()) for per in series.values() for s in per.values())
    xlim, ylim = padded_bounds(np.vstack([sim.truth[:, :, :2].reshape(-1, 2), [[0.0, 0.0]]]), 0.05)
    first, last = steps[0] * cfg.dt, steps[-1] * cfg.dt
    down_since = None
    if scene.radar_down[k]:
        down_since = k - int(np.argmin(scene.radar_down[: k + 1][::-1]))
        down_since = (down_since if scene.radar_down[down_since] else down_since + 1)
    titles = {"unaware": "Unaware tracker: not told, lost scans count as misses",
              "aware": "Outage-aware tracker: told the radar is down"}
    for column, policy in enumerate(("unaware", "aware")):
        left = 0.05 + 0.475 * column
        run = scene.runs[policy]
        state = MapState(sim, run, series[policy], colors, cfg.dt, k, window)
        ax = fig.add_axes([left, 0.35, 0.43, 0.46])
        style_axes(ax, style, "x [m]", "y [m]")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        draw_truth(ax, sim.truth, k, style)
        ax.plot(0.0, 0.0, marker="^", color=INK, ms=6, zorder=6)
        draw_camera_rays(ax, state, 4000.0)
        draw_radar_detections(ax, state, 30)
        draw_tracks(ax, state, style, ELLIPSE_MAGNIFICATION)
        ax.set_title(titles[policy], fontsize=style.body, color=INK, loc="left")
        confirmed = [s.track_id for s in run.history[k] if s.status is TrackStatus.CONFIRMED]
        ax.text(0.02, 0.02, "confirmed tracks: " + (", ".join(f"ID {i}" for i in confirmed)
                                                      or "none"),
                transform=ax.transAxes, fontsize=style.small * 1.1, color=INK, va="bottom")

        strip = fig.add_axes([left, 0.125, 0.43, 0.15])
        style_axes(strip, style, "time [s]", "1-sigma major axis [m]")
        strip.set_xlim(first, last)
        strip.set_ylim(0.0, ymax)
        strip.axvspan(start, end, color=WARNING, alpha=0.10, lw=0)
        for track_id, ser in series[policy].items():
            count = int(np.searchsorted(ser.steps, k, side="right"))
            if count:
                strip.plot(ser.steps[:count] * cfg.dt, ser.major[:count], color=colors[track_id],
                           lw=1.3)
        scheduled = [s for s, scan in enumerate(sim.radar_scans) if scan is not None and s <= k]
        for flag, fill in ((False, "full"), (True, "none")):
            times = [s * cfg.dt for s in scheduled if bool(scene.radar_down[s]) is flag]
            strip.plot(times, np.full(len(times), ymax * 0.04), "^", ms=4, color=RADAR_POINT,
                       mfc=RADAR_POINT if fill == "full" else "none")
        strip.axvline(k * cfg.dt, color=INK_MUTED, lw=0.8)
    fig.text(0.05, 0.295, f"t = {k * cfg.dt:4.1f} s", fontsize=style.body, color=INK, va="center")
    if down_since is not None:
        fig.text(0.5, 0.872,
                 f"RADAR OUTAGE: no radar scan for {(k - down_since) * cfg.dt:4.1f} s - "
                 "the camera keeps delivering bearings", ha="center", va="center",
                 fontsize=style.small * 1.25, color=WARNING, fontweight="bold",
                 bbox={"facecolor": SURFACE, "edgecolor": WARNING, "boxstyle": "round,pad=0.3"})


# --- rendering and encoding --------------------------------------------------------------------


def scene_inputs(name: str, scene):
    """Playback segments, last step and extra drawing arguments of a scene."""
    if name == "fusion":
        return FUSION_SEGMENTS, len(scene.truth) - 1, {}
    if name == "crossing":
        series = confirmed_series(scene.run.history)
        return CROSSING_SEGMENTS, len(scene.sim.camera_scans) - 1, {
            "series": series, "colors": id_colors(series)}
    per_policy = {p: confirmed_series(r.history) for p, r in scene.runs.items()}
    return OUTAGE_SEGMENTS, len(scene.sim.camera_scans) - 1, {
        "series": per_policy, "colors": id_colors(*per_policy.values())}


def render_scene(name: str, scene, size: tuple[int, int], fps: int, hold: int = HOLD_FRAMES,
                 only: Sequence[int] | None = None) -> tuple[list[bytes], list[int]]:
    """Render the frames of a scene to PNG bytes; returns the frames and all frame steps.

    Args:
        name: "fusion", "crossing" or "outage".
        scene: The scene data from the matching build function.
        size: Frame size in pixels.
        fps: Frames per second (only for the printed playback speed).
        hold: Copies of the last frame appended (not with `only`).
        only: Frame indices to render; None renders every frame.
    """
    segments, last_step, extra = scene_inputs(name, scene)
    dt = scene.config.dt
    steps = frame_steps(segments, dt, last_step)
    draw = {"fusion": draw_fusion_frame, "crossing": draw_crossing_frame,
            "outage": draw_outage_frame}[name]
    fig = new_figure(size)
    frames = []
    for index in range(len(steps)) if only is None else only:
        speed = segment_stride(segments, steps[index] * dt) * dt * fps
        draw(fig, scene, steps, index, speed, size, **extra)
        frames.append(to_png(fig))
    if only is None:
        frames.extend([frames[-1]] * hold)
    pyplot().close(fig)
    return frames, steps


def encode_animations(frames: Sequence[bytes], out_dir: Path, name: str, fps: int,
                      formats: Sequence[str]) -> dict[str, Path]:
    """Write the frames as animated GIF and/or WebP with one fixed 255-color palette.

    A single palette, built from frames sampled over the whole animation, avoids color flicker
    between frames and keeps memory low (palette frames take one byte per pixel).
    """
    from PIL import Image

    unknown = [f for f in formats if f not in ("gif", "webp")]
    if unknown:
        raise ValueError(f"unknown format(s) {unknown}; use gif and/or webp")

    def open_frame(data: bytes):
        return Image.open(io.BytesIO(data)).convert("RGB")

    sample = [open_frame(frames[i]) for i in np.linspace(0, len(frames) - 1, 24).astype(int)]
    sheet = Image.new("RGB", (sample[0].width, sample[0].height * len(sample)))
    for i, image in enumerate(sample):
        sheet.paste(image, (0, i * image.height))
    palette = sheet.quantize(colors=255, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    images = [open_frame(d).quantize(palette=palette, dither=Image.Dither.NONE) for d in frames]
    out_dir.mkdir(parents=True, exist_ok=True)
    duration = round(1000 / fps)
    written = {}
    for fmt in formats:
        path = out_dir / f"{name}.{fmt}"
        options = ({"optimize": False, "disposal": 1} if fmt == "gif"
                   else {"lossless": True, "method": 4})
        images[0].save(path, save_all=True, append_images=images[1:], duration=duration, loop=0,
                       **options)
        written[fmt] = path
    return written


def parse_size(text: str) -> tuple[int, int]:
    """'1280x720' -> (1280, 720)."""
    width, _, height = text.lower().partition("x")
    return int(width), int(height)


def preview_indices(count: int) -> list[int]:
    """Frame indices of a scene for the preview images: start, 25%, 50%, 75% and the end."""
    return sorted({int(round(f * (count - 1))) for f in (0.0, 0.25, 0.5, 0.75, 1.0)})


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "demo")
    parser.add_argument("--name", default="portfolio_demo")
    parser.add_argument("--scenes", type=int, nargs="+", choices=(1, 2, 3), default=(1, 2, 3))
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--size", type=parse_size, default=SIZE, help="WIDTHxHEIGHT in pixels")
    parser.add_argument("--formats", nargs="+", choices=("gif", "webp"), default=("gif", "webp"))
    parser.add_argument("--preview", action="store_true", help="also write sample PNG frames")
    args = parser.parse_args(argv)

    builders = {1: ("fusion", build_fusion_scene), 2: ("crossing", build_crossing_scene),
                3: ("outage", build_outage_scene)}
    all_frames: list[bytes] = []
    facts: list[str] = []
    for number in args.scenes:
        name, build = builders[number]
        scene = build()
        frames, steps = render_scene(name, scene, args.size, args.fps)
        facts.append(f"scene {number}: {name}, seed {scene.seed}, {len(steps)} frames + "
                     f"{HOLD_FRAMES} hold, steps {steps[0]}..{steps[-1]}")
        print(facts[-1])
        if args.preview:
            (args.out / "preview").mkdir(parents=True, exist_ok=True)
            for i in preview_indices(len(steps)):
                (args.out / "preview" / f"{name}_{i:03d}.png").write_bytes(frames[i])
        all_frames.extend(frames)
    written = encode_animations(all_frames, args.out, args.name, args.fps, args.formats)
    seconds = len(all_frames) / args.fps
    lines = [*provenance_lines(sys.argv),
             f"frames: {len(all_frames)} at {args.fps} fps = {seconds:.1f} s, "
             f"size {args.size[0]}x{args.size[1]}", *facts]
    for path in written.values():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{path.name}: {path.stat().st_size / 1e6:.2f} MB, sha256 {digest}")
        print(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB)")
    (args.out / f"{args.name}_provenance.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(all_frames)} frames, {seconds:.1f} s at {args.fps} fps")


if __name__ == "__main__":
    main()
