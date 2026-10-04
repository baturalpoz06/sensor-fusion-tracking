"""Outage metrics: the Phase 6 scores per window, plus what happens around an outage.

The run is cut at the outage span (start, end) into three windows, each a half-open range of
steps: before [burn-in, start), during [start, end) and after [end, end + after window),
the last clipped at the end of the run. All Phase 6 scores are computed per window, and
these outage-specific scores are added:

    reacquisition: whether and how fast a target that was tracked just before the outage is
        matched again after it (time, share of targets, and a censored mean);
    identity_kept: whether the first track after the outage has the id of the track before it;
    nees: normalized estimation error squared of the track that followed a target into
        the outage, on the outage steps (ideal value: the state dimension);
    ghost lifetime: how long the track of a target that ceased to exist survives.

The scores of one run are the fields of OutageMetrics (see FIELDS).
"""

from collections.abc import Mapping, Sequence
from typing import NamedTuple

import numpy as np
from scipy.stats import chi2

from fusion.consistency import nees
from fusion.dropout import step_index
from fusion.mtt_metrics import (
    MttMetrics,
    false_track_counts,
    match_tracks,
    score_run,
    window_scores,
)
from fusion.mtt_simulation import FieldOfView
from fusion.tracker.multi_target import TrackSnapshot

WINDOWS = ("before", "during", "after")
# Phase 6 scores computed per window (id_switches and false_track_rate are redefined, see
# evaluate_outage).
WINDOW_METRICS = (
    "position_rmse",
    "ghost_rate",
    "false_track_rate",
    "missed_rate",
    "id_switches",
    "births_per_scan",
)
OUTAGE_FIELDS = (
    "reacquisition_time",
    "reacquired_fraction",
    "reacquisition_time_censored",
    "identity_kept",
    "nees_outage",
    "nees_samples",
    "nees_outlier_fraction",
    "nees_before",
    "ghost_lifetime",
    "ghost_censored",
    "coast_deletions",
    "tentative_drops",
    "after_length",
)
FIELDS = (
    tuple(f"{window}_{metric}" for window in WINDOWS for metric in WINDOW_METRICS)
    + tuple(f"run_{metric}" for metric in MttMetrics._fields)
    + OUTAGE_FIELDS
)
# A NEES above this quantile of the chi-square distribution counts as an outlier.
OUTLIER_QUANTILE = 0.999

OutageMetrics = NamedTuple("OutageMetrics", [(name, float) for name in FIELDS])
OutageMetrics.__doc__ = (
    "Scores of one run around an outage; every field is a float (NaN where undefined).\n\n"
    "Fields: before_/during_/after_ plus the window metrics (position_rmse, ghost_rate,\n"
    "false_track_rate, missed_rate, id_switches, births_per_scan); run_ plus the Phase 6\n"
    "metrics of the whole run; the outage scores reacquisition_time, reacquired_fraction,\n"
    "reacquisition_time_censored, identity_kept, nees_outage, nees_samples,\n"
    "nees_outlier_fraction, nees_before, ghost_lifetime, ghost_censored; after_length, the\n"
    "seconds of the after window actually scored (it is clipped at the end of the run); and\n"
    "the tracker counters coast_deletions and tentative_drops, which the caller fills in\n"
    "(NaN here)."
)


class OutageWindows(NamedTuple):
    """The three windows of a run as step masks.

    Attributes:
        before: Shape (n_steps,) boolean, steps [burn-in, start).
        during: Steps [start, end).
        after: Steps [end, end + after window), clipped at the end of the run.
        start_step: First step of the outage.
        end_step: First step after the outage.
    """

    before: np.ndarray
    during: np.ndarray
    after: np.ndarray
    start_step: int
    end_step: int

    def masks(self) -> dict[str, np.ndarray]:
        """The windows by name, in the order of WINDOWS."""
        return {"before": self.before, "during": self.during, "after": self.after}


def outage_windows(
    n_steps: int,
    dt: float,
    burn_in_steps: int,
    span: tuple[float, float] | None,
    after_window: float,
) -> OutageWindows:
    """Cut a run into the before, during and after windows.

    Args:
        n_steps: Number of steps of the run.
        dt: Time step in seconds.
        burn_in_steps: Initial steps excluded from the before window.
        span: (start, end) of the outage in seconds, or None for a run without an outage
            (then the before window is the whole run after the burn-in and the other two
            are empty). A span of zero length gives an empty during window.
        after_window: Length of the after window in seconds.
    """
    steps = np.arange(n_steps)
    if span is None:
        start_step = end_step = n_steps
    else:
        start_step = min(step_index(span[0], dt), n_steps)
        end_step = min(step_index(span[1], dt), n_steps)
    after_end = min(end_step + step_index(after_window, dt), n_steps)
    return OutageWindows(
        before=(steps >= burn_in_steps) & (steps < start_step),
        during=(steps >= start_step) & (steps < end_step),
        after=(steps >= end_step) & (steps < after_end),
        start_step=start_step,
        end_step=end_step,
    )


def window_id_switches(ids: np.ndarray, mask: np.ndarray, carry_over: bool) -> float:
    """Identity changes of the matched track within a window, summed over targets.

    With carry_over the last valid id before the window counts as the id the window starts
    from, so that a track lost before the window and a different one found in it is a
    switch (the Phase 6 count within a window would miss it). NaN for an empty window.

    Args:
        ids: Shape (n_targets, n_steps) matched track ids, -1 where unmatched.
        mask: Shape (n_steps,) boolean selection of one contiguous window.
        carry_over: Whether to start from the last valid id before the window.
    """
    steps = np.flatnonzero(mask)
    if steps.size == 0:
        return np.nan
    total = 0
    for row in np.asarray(ids):
        valid = row[steps]
        valid = valid[valid >= 0]
        if carry_over:
            earlier = row[: steps[0]]
            earlier = earlier[earlier >= 0]
            if earlier.size:
                valid = np.concatenate([earlier[-1:], valid])
        total += int(np.sum(valid[1:] != valid[:-1]))
    return float(total)


def window_false_track_rate(
    history: Sequence[Sequence[TrackSnapshot]], ids: np.ndarray, mask: np.ndarray
) -> float:
    """Mean confirmed tracks per step in the window that were never matched up to its end.

    Unlike the whole-run Phase 6 definition this uses only the matches up to the end of
    the window, so the score of a window does not depend on what happens after it.
    """
    steps = np.flatnonzero(mask)
    if steps.size == 0:
        return np.nan
    last = steps[-1] + 1
    seen = ids[:, :last]
    matched = {int(i) for i in np.unique(seen[seen >= 0])}
    return float(np.mean(false_track_counts(history[steps[0] : last], matched)))


def window_births_per_scan(
    first_seen: Mapping[int, int], scheduled: np.ndarray, mask: np.ndarray
) -> float:
    """Tracks first reported in the window per scheduled radar scan of the window.

    Scans lost to an outage count as scheduled, so unaware and aware trackers are
    comparable. NaN if the window has no scheduled scan.
    """
    scans = int(np.sum(scheduled & mask))
    if scans == 0:
        return np.nan
    return sum(1 for step in first_seen.values() if mask[step]) / scans


def anchor_ids(ids: np.ndarray, alive: np.ndarray, start_step: int) -> dict[int, int]:
    """Target -> id of the track matched to it at the last step before the outage.

    A target without a match there was not being tracked when the outage began and has
    no anchor.
    """
    if not 1 <= start_step <= ids.shape[1]:
        return {}
    k = start_step - 1
    return {
        int(target): int(ids[target, k])
        for target in range(ids.shape[0])
        if ids[target, k] >= 0 and alive[target, k]
    }


def _nees_samples(
    truth: np.ndarray,
    present: Sequence[Mapping[int, TrackSnapshot]],
    anchors: Mapping[int, int],
    alive: np.ndarray,
    steps: np.ndarray,
) -> np.ndarray:
    """NEES of the anchor tracks at the given steps, while they exist and the target does."""
    errors, covariances = [], []
    for k in steps:
        for target, track_id in anchors.items():
            snapshot = present[k].get(track_id)
            if snapshot is not None and alive[target, k]:
                errors.append(truth[target, k] - snapshot.x)
                covariances.append(snapshot.P)
    if not errors:
        return np.zeros(0)
    return nees(np.array(errors), np.array(covariances))


def _ghost_lifetimes(
    ids: np.ndarray,
    present: Sequence[Mapping[int, TrackSnapshot]],
    vanish: Mapping[int, float],
    dt: float,
) -> tuple[float, float]:
    """Mean lifetime of the tracks of vanished targets, and the share still alive at the end.

    The lifetime runs from the first step without the target to the first step without the
    track; a track that survives to the last step counts until the end (censored). Only
    targets that were matched at their last existing step have a track to follow.
    """
    n_steps = ids.shape[1]
    lifetimes, censored = [], []
    for target, time in vanish.items():
        end = step_index(time, dt)
        if not 1 <= end < n_steps or ids[target, end - 1] < 0:
            continue
        track_id = int(ids[target, end - 1])
        gone = next((k for k in range(end, n_steps) if track_id not in present[k]), None)
        lifetimes.append(((n_steps if gone is None else gone) - end) * dt)
        censored.append(gone is None)
    if not lifetimes:
        return np.nan, np.nan
    return float(np.mean(lifetimes)), float(np.mean(censored))


def evaluate_outage(
    truth: np.ndarray,
    history: Sequence[Sequence[TrackSnapshot]],
    *,
    dt: float,
    fov: FieldOfView,
    max_distance: float,
    burn_in_steps: int,
    span: tuple[float, float] | None,
    after_window: float,
    radar_down: np.ndarray,
    scheduled: np.ndarray,
    alive: np.ndarray | None = None,
    vanish: Mapping[int, float] | None = None,
) -> OutageMetrics:
    """Score one run around an outage.

    Per window: position_rmse, ghost_rate, missed_rate (targets in view and existing),
    births_per_scan, and two scores defined for a window: id_switches counts a change
    from the last id before the window (carry-over; the before window has none, like
    Phase 6 after the burn-in), and false_track_rate counts the confirmed tracks never
    matched up to the end of the window. Phase 6 scores of the whole run are repeated as
    run_*. The anchor track of a target is the track matched to it at the last step before
    the outage. Eligible for reacquisition are the anchored targets that exist and are in
    view at the first step after the outage.

    reacquisition_time is the mean time to the first match after the outage over the
    eligible targets that were matched again within the after window, reacquired_fraction
    the share of eligible targets that were, and reacquisition_time_censored the mean over
    all eligible targets with the after window length (a lower bound) for those that were
    not. identity_kept is the share of eligible targets whose first match after the outage
    has the anchor id (a target never matched again counts as not kept).

    nees_outage averages the NEES of the anchor tracks over the steps of the during window
    at which the radar is down, as long as the track exists, whatever its distance from
    the target; nees_before does the same over the before window (the control: the ideal
    value is the state dimension, and a consistent filter is slightly above it in practice).
    nees_outlier_fraction is the share of the outage samples above the chi-square quantile
    OUTLIER_QUANTILE, nees_samples their number.

    ghost_lifetime and ghost_censored describe the targets in `vanish`, see _ghost_lifetimes.

    Args:
        truth: Shape (n_targets, n_steps, 4) true states.
        history: n_steps lists of track snapshots.
        dt: Time step in seconds.
        fov: Field of view; targets outside it do not count as missed.
        max_distance: Largest position error in meters of a valid match.
        burn_in_steps: Initial steps excluded from the before window and the run scores.
        span: (start, end) of the outage in seconds, or None if there is none.
        after_window: Length of the after window in seconds.
        radar_down: Shape (n_steps,) boolean, steps at which the radar is silent.
        scheduled: Shape (n_steps,) boolean, steps with a scheduled radar scan.
        alive: Optional shape (n_targets, n_steps) mask of existing targets.
        vanish: Optional target index -> time in seconds at which it ceases to exist.

    Returns:
        OutageMetrics with coast_deletions, tentative_drops and run_births_per_scan left at NaN.
    """
    truth = np.asarray(truth, dtype=float)
    n_targets, n_steps = truth.shape[:2]
    alive_mask = np.ones((n_targets, n_steps), dtype=bool) if alive is None else alive
    match = match_tracks(truth, history, max_distance, alive)
    windows = outage_windows(n_steps, dt, burn_in_steps, span, after_window)
    in_view = fov.contains(truth[:, :, :2].reshape(-1, 2)).reshape(n_targets, n_steps)
    in_view = in_view & alive_mask

    first_seen: dict[int, int] = {}
    for k, step in enumerate(history):
        for snapshot in step:
            first_seen.setdefault(snapshot.track_id, k)
    present = [{s.track_id: s for s in step} for step in history]

    values: dict[str, float] = {name: np.nan for name in FIELDS}
    values["after_length"] = float(np.sum(windows.after) * dt)
    for window, mask in windows.masks().items():
        rmse, ghost, missed = window_scores(match, in_view, mask)
        values[f"{window}_position_rmse"] = rmse
        values[f"{window}_ghost_rate"] = ghost
        values[f"{window}_missed_rate"] = missed
        values[f"{window}_false_track_rate"] = window_false_track_rate(history, match.ids, mask)
        values[f"{window}_id_switches"] = window_id_switches(
            match.ids, mask, carry_over=window != "before"
        )
        values[f"{window}_births_per_scan"] = window_births_per_scan(first_seen, scheduled, mask)

    run = score_run(
        match, truth, history, dt=dt, fov=fov, burn_in_steps=burn_in_steps, alive=alive
    )
    for name, value in zip(MttMetrics._fields, run, strict=True):
        values[f"run_{name}"] = float(value)

    anchors = {} if span is None else anchor_ids(match.ids, alive_mask, windows.start_step)
    if anchors:
        before = _nees_samples(truth, present, anchors, alive_mask, np.flatnonzero(windows.before))
        during_steps = np.flatnonzero(windows.during & np.asarray(radar_down, dtype=bool))
        during = _nees_samples(truth, present, anchors, alive_mask, during_steps)
        if before.size:
            values["nees_before"] = float(before.mean())
        if during.size:
            threshold = chi2.ppf(OUTLIER_QUANTILE, df=truth.shape[2])
            values["nees_outage"] = float(during.mean())
            values["nees_outlier_fraction"] = float(np.mean(during > threshold))
        values["nees_samples"] = float(during.size)
        values.update(_reacquisition(match.ids, anchors, in_view, alive_mask, windows, dt))
    elif span is not None:
        values["nees_samples"] = 0.0
    if vanish:
        lifetime, censored = _ghost_lifetimes(match.ids, present, vanish, dt)
        values["ghost_lifetime"], values["ghost_censored"] = lifetime, censored
    return OutageMetrics(**values)


def _reacquisition(
    ids: np.ndarray,
    anchors: Mapping[int, int],
    in_view: np.ndarray,
    alive: np.ndarray,
    windows: OutageWindows,
    dt: float,
) -> dict[str, float]:
    """Reacquisition time, share and censored mean, and identity_kept."""
    end, after_steps = windows.end_step, np.flatnonzero(windows.after)
    eligible = [
        target
        for target in anchors
        if end < ids.shape[1] and alive[target, end] and in_view[target, end]
    ]
    if not eligible or after_steps.size == 0:
        return {}
    window_length = after_steps.size * dt
    times, censored, kept = [], [], []
    for target in eligible:
        matched = np.flatnonzero(ids[target, after_steps] >= 0)
        if matched.size:
            step = after_steps[matched[0]]
            times.append((step - end) * dt)
            censored.append((step - end) * dt)
            kept.append(float(ids[target, step] == anchors[target]))
        else:
            censored.append(window_length)
            kept.append(0.0)
    return {
        "reacquisition_time": float(np.mean(times)) if times else np.nan,
        "reacquired_fraction": len(times) / len(eligible),
        "reacquisition_time_censored": float(np.mean(censored)),
        "identity_kept": float(np.mean(kept)),
    }
