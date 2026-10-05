"""Robustness metrics: the Phase 6 scores plus diagnostics of how a tracker degrades.

A run is scored twice against the truth. The Phase 6 scores (run_*) use the usual match distance
(50 m), so they stay comparable with every earlier phase. The diagnostics that describe an error
(the line-of-sight split, NEES, which measurements the camera updates used) use a wide match
distance: with the narrow one, a track that a bias or a lever arm pulls 60 m off its target is
no longer matched, and its error would vanish from the error scores and reappear only as a
missed target and a ghost. The matched shares of both distances are reported next to the errors.

The error of a matched track is split relative to the line of sight from the radar to the target
into an along-range part (radial) and a cross-range part (tangential, counter-clockwise
positive). A camera bias or a lever arm moves the estimate across the line of sight; a
maneuver mostly along it.

All scores are over the steps after the burn-in unless noted. Every field is a float, NaN where
it is undefined (no matched pair, no camera update, ...).
"""

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
from scipy.stats import chi2

from fusion.consistency import nees
from fusion.dropout import step_index
from fusion.mtt_metrics import MttMetrics, match_tracks, score_run, window_scores
from fusion.mtt_simulation import FieldOfView, Scan
from fusion.sensors.base import MIN_RANGE
from fusion.tracker.multi_target import CameraStepLog, TrackSnapshot

# A NEES above this quantile of the chi-square distribution (4 degrees of freedom) is an outlier.
OUTLIER_QUANTILE = 0.999
DIAGNOSTIC_FIELDS = (
    "match_fraction_50",
    "match_fraction_wide",
    "along_rms",
    "cross_rms",
    "along_mean",
    "cross_mean",
    "cross_bearing_rms",
    "nees_mean",
    "nees_median",
    "nees_outlier_fraction",
    "camera_accept_rate",
    "camera_skip_rate",
    "camera_true_accept_rate",
    "camera_wrong_update_rate",
    "camera_unattributed_update_rate",
    "in_view_fraction",
    "min_target_separation",
    "window_position_rmse",
    "window_missed_rate",
    "window_ghost_rate",
    "window_cross_rms",
    "window_nees_mean",
)
FIELDS = tuple(f"run_{name}" for name in MttMetrics._fields) + DIAGNOSTIC_FIELDS

RobustnessMetrics = NamedTuple("RobustnessMetrics", [(name, float) for name in FIELDS])
RobustnessMetrics.__doc__ = (
    "Scores of one run; every field is a float (NaN where undefined).\n\n"
    "run_ plus the Phase 6 metrics (50 m match distance; run_births_per_scan is filled by the\n"
    "caller). Diagnostics, all with the wide match distance unless noted: match_fraction_50 and\n"
    "match_fraction_wide (share of in-view target steps that have a match), along_rms and\n"
    "cross_rms (error relative to the line of sight from the radar, in meters), along_mean and\n"
    "cross_mean (signed, estimate minus truth; cross is counter-clockwise positive),\n"
    "cross_bearing_rms (cross-range error divided by range, in milliradians), nees_mean,\n"
    "nees_median and nees_outlier_fraction (state NEES of the matched tracks; ideal mean 4),\n"
    "camera_accept_rate (updates made per track offered a camera scan), camera_skip_rate\n"
    "(tracks left out for overlapping bearing gates), camera_true_accept_rate (detected\n"
    "targets whose track took that target's measurement), camera_wrong_update_rate (updates of\n"
    "matched tracks with clutter or another target's measurement),\n"
    "camera_unattributed_update_rate (updates of tracks matched to no target),\n"
    "in_view_fraction (share of target steps in the\n"
    "field of view), min_target_separation (closest approach of two targets over the whole run,\n"
    "meters) and the window_ scores of the focus window (Phase 6 scores at 50 m, cross_rms and\n"
    "nees_mean wide), NaN without a focus window."
)


def los_error_split(error_xy: np.ndarray, truth_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split position errors into along-range and cross-range parts of the line of sight.

    With u the unit vector from the radar at the origin to the true position and u_perp = u
    rotated by +90 degrees, along = e . u and cross = e . u_perp for the error e = estimate -
    truth. A bearing bias of +b therefore gives a positive cross error of about b * range.

    Args:
        error_xy: Shape (n, 2) position errors (estimate minus truth) in meters.
        truth_xy: Shape (n, 2) true positions in meters.

    Returns:
        (along, cross), each shape (n,); NaN where the target is closer than MIN_RANGE to the
        radar and the line of sight is undefined.
    """
    error_xy = np.asarray(error_xy, dtype=float)
    truth_xy = np.asarray(truth_xy, dtype=float)
    if error_xy.shape != truth_xy.shape or error_xy.ndim != 2 or error_xy.shape[1] != 2:
        raise ValueError(
            f"error_xy and truth_xy must both have shape (n, 2), got {error_xy.shape} "
            f"and {truth_xy.shape}"
        )
    distance = np.hypot(truth_xy[:, 0], truth_xy[:, 1])
    valid = distance >= MIN_RANGE
    safe = np.where(valid, distance, 1.0)
    ux, uy = truth_xy[:, 0] / safe, truth_xy[:, 1] / safe
    along = error_xy[:, 0] * ux + error_xy[:, 1] * uy
    cross = -error_xy[:, 0] * uy + error_xy[:, 1] * ux
    return np.where(valid, along, np.nan), np.where(valid, cross, np.nan)


def _rms(values: np.ndarray) -> float:
    valid = values[np.isfinite(values)]
    return float(np.sqrt(np.mean(valid**2))) if valid.size else np.nan


def _mean(values: np.ndarray) -> float:
    valid = values[np.isfinite(values)]
    return float(np.mean(valid)) if valid.size else np.nan


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else np.nan


def match_fraction(ids: np.ndarray, in_view: np.ndarray, after: np.ndarray) -> float:
    """Share of the in-view (target, step) pairs after the burn-in that have a matched track."""
    counted = in_view & after[None, :]
    return float(np.mean(ids[counted] >= 0)) if counted.any() else np.nan


def min_separation(truth: np.ndarray) -> float:
    """Closest approach of any two targets over the whole run, in meters; NaN for one target."""
    n_targets = truth.shape[0]
    if n_targets < 2:
        return np.nan
    best = np.inf
    for i in range(n_targets):
        for j in range(i + 1, n_targets):
            best = min(best, float(np.hypot(*(truth[i, :, :2] - truth[j, :, :2]).T).min()))
    return best


class CameraRates(NamedTuple):
    """Camera update diagnostics of a run (see camera_rates)."""

    accept: float
    skip: float
    true_accept: float
    wrong_update: float
    unattributed_update: float


def camera_rates(
    ids: np.ndarray,
    camera_log: Sequence[CameraStepLog],
    camera_scans: Sequence[Scan],
    burn_in_steps: int,
) -> CameraRates:
    """Camera accept rates over the steps after the burn-in.

    Tracks are attributed to targets by the match (ids[target, step] is the track matched to
    the target); a track matched to no target at a step is unattributed there.

    accept: updates made / tracks offered a camera scan (confirmed, bearing gate not
    overlapping another). skip: tracks left out for overlapping gates / (offered + left out).
    true_accept: over the (target, step) pairs in which the camera detected the target and
    its matched track was offered the scan, the share in which that track took a measurement
    of this very target. wrong_update: of the updates of matched tracks, the share that used
    clutter or another target's measurement. unattributed_update: of all updates, the share
    made by tracks matched to no target.

    Args:
        ids: Shape (n_targets, n_steps) matched track ids, -1 where unmatched.
        camera_log: One CameraStepLog per step.
        camera_scans: One camera Scan per step (its origin column names the source of each row).
        burn_in_steps: Initial steps that are left out.

    Returns:
        CameraRates; each NaN where its denominator is zero.
    """
    n_targets, n_steps = ids.shape
    if len(camera_log) != n_steps or len(camera_scans) != n_steps:
        raise ValueError(
            f"need one log entry and one scan per step ({n_steps}), got {len(camera_log)} "
            f"and {len(camera_scans)}"
        )
    offered = skipped = accepted = 0
    pairs = hits = 0
    attributed_updates = wrong = unattributed = 0
    for k in range(burn_in_steps, n_steps):
        entry, scan = camera_log[k], camera_scans[k]
        offered += len(entry.usable)
        skipped += len(entry.skipped)
        accepted += len(entry.assigned)
        taken = dict(entry.assigned)  # track id -> row of the scan
        owner = {int(ids[i, k]): i for i in range(n_targets) if ids[i, k] >= 0}
        for track_id, row in entry.assigned:
            if track_id in owner:
                attributed_updates += 1
                wrong += int(scan.origin[row] != owner[track_id])
            else:
                unattributed += 1
        for target in range(n_targets):
            track_id = int(ids[target, k])
            if track_id < 0 or track_id not in entry.usable or target not in scan.origin:
                continue
            pairs += 1
            row = taken.get(track_id)
            hits += int(row is not None and scan.origin[row] == target)
    return CameraRates(
        _ratio(accepted, offered),
        _ratio(skipped, offered + skipped),
        _ratio(hits, pairs),
        _ratio(wrong, attributed_updates),
        _ratio(unattributed, accepted),
    )


class MatchedErrors(NamedTuple):
    """Errors of the matched (target, step) pairs after the burn-in.

    Attributes:
        steps: Shape (m,) step of each pair.
        along: Shape (m,) along-range error (estimate minus truth) in meters.
        cross: Shape (m,) cross-range error in meters.
        range_: Shape (m,) true range in meters.
        nees: Shape (m,) state NEES of the matched track.
    """

    steps: np.ndarray
    along: np.ndarray
    cross: np.ndarray
    range_: np.ndarray
    nees: np.ndarray


def matched_errors(
    truth: np.ndarray,
    history: Sequence[Sequence[TrackSnapshot]],
    ids: np.ndarray,
    burn_in_steps: int,
) -> MatchedErrors:
    """Line-of-sight errors and NEES of every matched pair after the burn-in."""
    present = [{s.track_id: s for s in step} for step in history]
    n_steps = truth.shape[1]
    wanted = (ids >= 0) & (np.arange(n_steps) >= burn_in_steps)[None, :]
    targets, steps = np.nonzero(wanted)
    if targets.size == 0:
        empty = np.zeros(0)
        return MatchedErrors(steps, empty, empty, empty, empty)
    snapshots = [present[k][int(ids[i, k])] for i, k in zip(targets, steps, strict=True)]
    estimate = np.array([s.x for s in snapshots])
    covariance = np.array([s.P for s in snapshots])
    truth_pairs = truth[targets, steps]
    along, cross = los_error_split(estimate[:, :2] - truth_pairs[:, :2], truth_pairs[:, :2])
    return MatchedErrors(
        steps,
        along,
        cross,
        np.hypot(truth_pairs[:, 0], truth_pairs[:, 1]),
        nees(truth_pairs - estimate, covariance),
    )


def evaluate_robustness(
    truth: np.ndarray,
    history: Sequence[Sequence[TrackSnapshot]],
    camera_log: Sequence[CameraStepLog],
    camera_scans: Sequence[Scan],
    *,
    dt: float,
    fov: FieldOfView,
    max_distance: float,
    wide_distance: float,
    burn_in_steps: int,
    focus_window: tuple[float, float] | None = None,
) -> RobustnessMetrics:
    """Score one run of the tracker against the truth: Phase 6 scores and robustness diagnostics.

    Args:
        truth: Shape (n_targets, n_steps, 4) true states.
        history: n_steps lists of track snapshots.
        camera_log: The tracker's camera_log, one entry per step.
        camera_scans: The simulation's camera scans, one per step (for the origin of each row).
        dt: Time step in seconds.
        fov: Field of view; targets outside it do not count as missed.
        max_distance: Match distance of the Phase 6 scores in meters.
        wide_distance: Match distance of the error diagnostics in meters (>= max_distance).
        burn_in_steps: Initial steps left out of all scores except the delay and the confirmed
            fraction of the Phase 6 scores.
        focus_window: Optional (start, end) in seconds: the window_ scores cover its steps.

    Returns:
        RobustnessMetrics with run_births_per_scan left at NaN.

    Raises:
        ValueError: If wide_distance < max_distance, or the log or scans do not have one entry
            per step.
    """
    if wide_distance < max_distance:
        raise ValueError(f"wide_distance must be >= max_distance, got {wide_distance}")
    truth = np.asarray(truth, dtype=float)
    n_targets, n_steps = truth.shape[:2]
    steps = np.arange(n_steps)
    after = steps >= burn_in_steps
    in_view = fov.contains(truth[:, :, :2].reshape(-1, 2)).reshape(n_targets, n_steps)

    narrow = match_tracks(truth, history, max_distance)
    run = score_run(narrow, truth, history, dt=dt, fov=fov, burn_in_steps=burn_in_steps)
    wide = match_tracks(truth, history, wide_distance)
    errors = matched_errors(truth, history, wide.ids, burn_in_steps)
    rates = camera_rates(wide.ids, camera_log, camera_scans, burn_in_steps)

    values = {name: np.nan for name in FIELDS}
    for name, value in zip(MttMetrics._fields, run, strict=True):
        values[f"run_{name}"] = float(value)
    bearing_error = errors.cross / errors.range_
    values.update(
        match_fraction_50=match_fraction(narrow.ids, in_view, after),
        match_fraction_wide=match_fraction(wide.ids, in_view, after),
        along_rms=_rms(errors.along),
        cross_rms=_rms(errors.cross),
        along_mean=_mean(errors.along),
        cross_mean=_mean(errors.cross),
        cross_bearing_rms=1e3 * _rms(bearing_error),
        camera_accept_rate=rates.accept,
        camera_skip_rate=rates.skip,
        camera_true_accept_rate=rates.true_accept,
        camera_wrong_update_rate=rates.wrong_update,
        camera_unattributed_update_rate=rates.unattributed_update,
        in_view_fraction=float(np.mean(in_view[:, after])) if after.any() else np.nan,
        min_target_separation=min_separation(truth),
    )
    if errors.nees.size:
        values["nees_mean"] = float(errors.nees.mean())
        values["nees_median"] = float(np.median(errors.nees))
        values["nees_outlier_fraction"] = float(
            np.mean(errors.nees > chi2.ppf(OUTLIER_QUANTILE, df=truth.shape[2]))
        )
    if focus_window is not None:
        mask = (steps >= step_index(focus_window[0], dt)) & (
            steps < step_index(focus_window[1], dt)
        )
        rmse, ghost, missed = window_scores(narrow, in_view, mask)
        values.update(window_position_rmse=rmse, window_ghost_rate=ghost, window_missed_rate=missed)
        inside = np.isin(errors.steps, np.flatnonzero(mask))
        values["window_cross_rms"] = _rms(errors.cross[inside])
        if inside.any():
            values["window_nees_mean"] = float(errors.nees[inside].mean())
    return RobustnessMetrics(**values)
