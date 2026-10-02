"""Multi-target tracking metrics: truth matching, error, ghosts, misses, ID switches."""

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
from scipy.stats import t as student_t

from fusion.association.assignment import gated_assignment
from fusion.mtt_simulation import FieldOfView
from fusion.tracker.multi_target import TrackSnapshot
from fusion.tracker.track import TrackStatus


class MttMetrics(NamedTuple):
    """Scores of one tracking run, over the steps after the burn-in unless noted.

    Attributes:
        position_rmse: Root mean square position error in meters over all matched
            (target, step) pairs; NaN if there are none.
        ghost_rate: Mean number of confirmed tracks per step that match no target
            (farther than max_distance from every unclaimed target). This includes
            tracks that follow a real target but are pulled off it, and duplicates.
        missed_rate: Share of (target, step) pairs inside the field of view that
            have no matching confirmed track; NaN if no target is ever in view. It
            includes the time before a target's first confirmation when that comes
            after the burn-in (at low detection probability), so read it together
            with confirmation_delay.
        confirmation_delay: Mean over the targets that were ever matched of the
            time in seconds from the start to the first match (all steps count);
            NaN if no target was matched. Read it together with confirmed_fraction.
        confirmed_fraction: Share of targets that were matched at least once (all steps).
        id_switches: Number of times a target is matched to a different track id
            than the last one it was matched to, summed over targets.
        false_track_rate: Mean number of confirmed tracks per step that are never
            matched to any target during the whole run (clutter-born tracks). It is
            at most ghost_rate; the difference is the off-target rate, tracks that
            were on a target at some step but are unmatched at this one.
        births_per_scan: Diagnostic, tracks started per radar scan; NaN unless the
            experiment fills it from the tracker counters.
    """

    position_rmse: float
    ghost_rate: float
    missed_rate: float
    confirmation_delay: float
    confirmed_fraction: float
    id_switches: int
    false_track_rate: float = float("nan")
    births_per_scan: float = float("nan")


class Matching(NamedTuple):
    """Per-step assignment of confirmed tracks to true targets.

    Attributes:
        ids: Shape (n_targets, n_steps) id of the matched confirmed track, -1 if none.
        squared_error: Shape (n_targets, n_steps) squared position error of the
            match, NaN where there is none.
        n_confirmed: Shape (n_steps,) number of confirmed tracks.
        n_matched: Shape (n_steps,) number of targets with a match.
    """

    ids: np.ndarray
    squared_error: np.ndarray
    n_confirmed: np.ndarray
    n_matched: np.ndarray


def match_tracks(
    truth: np.ndarray,
    history: Sequence[Sequence[TrackSnapshot]],
    max_distance: float,
) -> Matching:
    """Match the confirmed tracks to the true targets at every step.

    At each step the pairing is the global assignment (the same gated Hungarian
    algorithm as in the tracker) that maximizes the number of pairs closer than
    max_distance and then minimizes the total squared distance, so every track
    explains at most one target and vice versa. Tentative tracks are ignored.
    All targets can be matched, also outside the field of view.

    Args:
        truth: Shape (n_targets, n_steps, 4) true states.
        history: n_steps lists of track snapshots.
        max_distance: Largest position error in meters of a valid match.

    Raises:
        ValueError: If history does not have one entry per step, or max_distance <= 0.
    """
    truth = np.asarray(truth, dtype=float)
    n_targets, n_steps = truth.shape[:2]
    if len(history) != n_steps:
        raise ValueError(f"history has {len(history)} steps, truth has {n_steps}")
    if max_distance <= 0.0:
        raise ValueError(f"max_distance must be > 0, got {max_distance}")

    ids = np.full((n_targets, n_steps), -1, dtype=int)
    squared_error = np.full((n_targets, n_steps), np.nan)
    n_confirmed = np.zeros(n_steps, dtype=int)
    n_matched = np.zeros(n_steps, dtype=int)
    for k, snapshots in enumerate(history):
        confirmed = [s for s in snapshots if s.status is TrackStatus.CONFIRMED]
        n_confirmed[k] = len(confirmed)
        if not confirmed or n_targets == 0:
            continue
        positions = np.array([s.x[:2] for s in confirmed])
        diff = truth[:, k, None, :2] - positions[None, :, :]
        cost = np.sum(diff**2, axis=2)
        result = gated_assignment(cost, cost <= max_distance**2)
        for target, track in result.pairs:
            ids[target, k] = confirmed[track].track_id
            squared_error[target, k] = cost[target, track]
        n_matched[k] = len(result.pairs)
    return Matching(ids, squared_error, n_confirmed, n_matched)


def false_track_counts(
    history: Sequence[Sequence[TrackSnapshot]], matched_ids: set[int]
) -> np.ndarray:
    """Number of confirmed tracks per step whose id was never matched to a target.

    Args:
        history: n_steps lists of track snapshots.
        matched_ids: Ids of the tracks that were matched to a target at some step.

    Returns:
        Shape (n_steps,) counts.
    """
    return np.array(
        [
            sum(s.status is TrackStatus.CONFIRMED and s.track_id not in matched_ids for s in step)
            for step in history
        ],
        dtype=int,
    )


def count_id_switches(ids: np.ndarray) -> int:
    """Number of identity changes of the matched track, summed over targets.

    A target's id sequence is compared with its last valid id: gaps (-1) do not
    count, but 1 -> gap -> 2 is a switch.

    Args:
        ids: Shape (n_targets, n_steps) matched track ids, -1 where unmatched.
    """
    switches = 0
    for row in np.asarray(ids):
        valid = row[row >= 0]
        switches += int(np.sum(valid[1:] != valid[:-1]))
    return switches


def first_match_times(ids: np.ndarray, dt: float) -> np.ndarray:
    """Time in seconds of the first match of each target; NaN if it never matched.

    Args:
        ids: Shape (n_targets, n_steps) matched track ids, -1 where unmatched.
        dt: Time step in seconds.
    """
    ids = np.asarray(ids)
    matched = ids >= 0
    first = np.argmax(matched, axis=1) if ids.shape[1] else np.zeros(len(ids), dtype=int)
    return np.where(matched.any(axis=1), first * dt, np.nan)


def evaluate_mtt(
    truth: np.ndarray,
    history: Sequence[Sequence[TrackSnapshot]],
    *,
    dt: float,
    fov: FieldOfView,
    max_distance: float,
    burn_in_steps: int,
) -> MttMetrics:
    """Score one run of the multi-target tracker against the truth.

    Args:
        truth: Shape (n_targets, n_steps, 4) true states.
        history: n_steps lists of track snapshots.
        dt: Time step in seconds.
        fov: Field of view; targets outside it do not count as missed.
        max_distance: Largest position error in meters of a valid match.
        burn_in_steps: Initial steps excluded from all metrics except the delay
            and the confirmed fraction.

    Returns:
        MttMetrics with births_per_scan left at NaN.
    """
    truth = np.asarray(truth, dtype=float)
    n_targets, n_steps = truth.shape[:2]
    match = match_tracks(truth, history, max_distance)
    after = np.arange(n_steps) >= burn_in_steps

    squared = match.squared_error[:, after]
    position_rmse = float(np.sqrt(np.nanmean(squared))) if np.isfinite(squared).any() else np.nan
    ghost_rate = (
        float(np.mean((match.n_confirmed - match.n_matched)[after])) if after.any() else np.nan
    )

    in_view = fov.contains(truth[:, :, :2].reshape(-1, 2)).reshape(n_targets, n_steps)
    counted = in_view & after
    missed_rate = float(np.mean(match.ids[counted] < 0)) if counted.any() else np.nan

    matched_ids = {int(i) for i in np.unique(match.ids[match.ids >= 0])}
    false_counts = false_track_counts(history, matched_ids)
    false_track_rate = float(np.mean(false_counts[after])) if after.any() else np.nan

    delays = first_match_times(match.ids, dt)
    ever_matched = np.isfinite(delays)
    return MttMetrics(
        position_rmse,
        ghost_rate,
        missed_rate,
        float(np.mean(delays[ever_matched])) if ever_matched.any() else np.nan,
        float(np.mean(ever_matched)) if n_targets else np.nan,
        count_id_switches(match.ids[:, after]),
        false_track_rate,
    )


class SeedInterval(NamedTuple):
    """Mean over seeds with a confidence interval.

    Attributes:
        mean: Mean of the valid (non-NaN) values; NaN if there are none.
        lower: Lower bound of the interval; NaN with fewer than 2 valid values.
        upper: Upper bound of the interval; NaN with fewer than 2 valid values.
        n_valid: Number of valid values.
    """

    mean: float
    lower: float
    upper: float
    n_valid: int


def seed_confidence_interval(values: np.ndarray, confidence: float = 0.95) -> SeedInterval:
    """Student t confidence interval of the mean over seeds, skipping NaN values.

    A metric can be undefined for a seed (for example the position error when
    no target was matched). Those seeds are left out and reported through
    n_valid. With fewer than 2 valid seeds the interval is undefined and NaN is
    returned instead of raising, so a sweep over extreme settings still completes.

    Args:
        values: Shape (n_seeds,) one metric value per seed, NaN where undefined.
        confidence: Coverage of the interval, strictly between 0 and 1.

    Raises:
        ValueError: If confidence is not in (0, 1).
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    valid = np.asarray(values, dtype=float)
    valid = valid[np.isfinite(valid)]
    n = len(valid)
    if n == 0:
        return SeedInterval(np.nan, np.nan, np.nan, 0)
    mean = float(valid.mean())
    if n < 2:
        return SeedInterval(mean, np.nan, np.nan, n)
    half_width = float(student_t.ppf(0.5 + confidence / 2.0, df=n - 1)) * float(
        valid.std(ddof=1) / np.sqrt(n)
    )
    return SeedInterval(mean, mean - half_width, mean + half_width, n)
