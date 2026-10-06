"""Metrics of the Phase 8b before / after evaluation: time bins, common support, modes, bias.

These functions score one tracker run that has already been matched to the truth (the matching
of fusion.mtt_metrics at the Phase 6 distance and at the wide diagnostic distance). They read
the diagnostic logs of the tracker but never feed anything back.

Definitions:
    missed: an in-view (target, step) pair without a matched confirmed track at the Phase 6
        distance; its rate is taken over the in-view pairs of a time bin.
    drag: a missed pair whose target was last matched (at the Phase 6 distance) to a track that
        is still matched to it at the wide distance: the track is alive but pulled off the
        target. A track that was lost and replaced by a new one near the target (another id) is
        not drag.
    common support: the (target, step) pairs that every compared arm matched at the Phase 6
        distance, so that an RMSE is not improved or worsened by which pairs an arm kept.
"""

from collections.abc import Sequence

import numpy as np

from fusion.dropout import step_index
from fusion.tracker.camera_bias import BiasLogEntry
from fusion.tracker.multi_target import ModeLogEntry


def window_mask(n_steps: int, window: tuple[float, float], dt: float) -> np.ndarray:
    """Boolean mask of the steps in [start, end) seconds (the focus-window convention)."""
    steps = np.arange(n_steps)
    return (steps >= step_index(window[0], dt)) & (steps < step_index(window[1], dt))


def drag_mask(narrow_ids: np.ndarray, wide_ids: np.ndarray, in_view: np.ndarray) -> np.ndarray:
    """In-view (target, step) pairs that are missed at the narrow distance but dragged.

    Args:
        narrow_ids: Shape (n_targets, n_steps) matched track ids at the Phase 6 distance, -1 if
            none.
        wide_ids: Same at the wide distance.
        in_view: Shape (n_targets, n_steps) targets inside the field of view.

    Returns:
        Shape (n_targets, n_steps) boolean mask. Unmatched in view, and the track of the last
        narrow match of this target is matched to it at the wide distance.
    """
    narrow_ids = np.asarray(narrow_ids)
    wide_ids = np.asarray(wide_ids)
    drag = np.zeros(narrow_ids.shape, dtype=bool)
    for i in range(narrow_ids.shape[0]):
        last = -1
        for k in range(narrow_ids.shape[1]):
            if narrow_ids[i, k] >= 0:
                last = int(narrow_ids[i, k])
            elif in_view[i, k] and last >= 0 and wide_ids[i, k] == last:
                drag[i, k] = True
    return drag


def bin_edges(n_steps: int, dt: float, start: float, width: float) -> list[tuple[int, int]]:
    """Step ranges [first, last) of consecutive bins of `width` seconds from `start` to the end.

    The last bin may be shorter. Start is typically the burn-in, so that the confirmation time
    of the first tracks does not appear in the rates.
    """
    first = step_index(start, dt)
    size = max(step_index(width, dt), 1)
    return [(k, min(k + size, n_steps)) for k in range(first, n_steps, size)]


def binned_rates(
    narrow_ids: np.ndarray,
    wide_ids: np.ndarray,
    in_view: np.ndarray,
    bins: Sequence[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Missed rate and drag rate in each time bin.

    Args:
        narrow_ids: Shape (n_targets, n_steps) matched ids at the Phase 6 distance.
        wide_ids: Same at the wide distance.
        in_view: Shape (n_targets, n_steps) targets inside the field of view.
        bins: Step ranges from bin_edges.

    Returns:
        (missed, drag), each shape (n_bins,): the share of the bin's in-view pairs that are
        missed, respectively dragged; NaN for a bin without in-view pairs.
    """
    missed_pairs = in_view & (np.asarray(narrow_ids) < 0)
    drag_pairs = drag_mask(narrow_ids, wide_ids, in_view)
    missed = np.full(len(bins), np.nan)
    drag = np.full(len(bins), np.nan)
    for b, (first, last) in enumerate(bins):
        counted = int(in_view[:, first:last].sum())
        if counted:
            missed[b] = missed_pairs[:, first:last].sum() / counted
            drag[b] = drag_pairs[:, first:last].sum() / counted
    return missed, drag


def common_support_rmse(
    squared_errors: Sequence[np.ndarray], mask: np.ndarray
) -> tuple[list[float], int]:
    """Position RMSE of each arm over the pairs that all arms matched.

    Args:
        squared_errors: One shape (n_targets, n_steps) array per arm, the squared position
            error of the match, NaN where the arm has none (Matching.squared_error).
        mask: Shape (n_steps,) boolean selection of the steps.

    Returns:
        (rmse per arm, number of common pairs); NaN values if there are none.
    """
    stack = np.array([np.asarray(e)[:, mask] for e in squared_errors])
    common = np.isfinite(stack).all(axis=0)
    n = int(common.sum())
    if n == 0:
        return [float("nan")] * len(squared_errors), 0
    return [float(np.sqrt(np.mean(e[common]))) for e in stack], n


def mode_probabilities(
    mode_log: Sequence[Sequence[ModeLogEntry]], ids: np.ndarray, n_modes: int
) -> np.ndarray:
    """Mode probabilities of the track matched to each target at each step.

    Args:
        mode_log: The tracker's mode_log, one tuple of entries per step.
        ids: Shape (n_targets, n_steps) matched track ids, -1 where unmatched.
        n_modes: Number of modes.

    Returns:
        Shape (n_targets, n_steps, n_modes), NaN where the target has no matched IMM track.
    """
    ids = np.asarray(ids)
    out = np.full((*ids.shape, n_modes), np.nan)
    for k, entries in enumerate(mode_log):
        by_id = {e.track_id: e.mu for e in entries}
        for i in range(ids.shape[0]):
            mu = by_id.get(int(ids[i, k])) if ids[i, k] >= 0 else None
            if mu is not None:
                out[i, k] = mu
    return out


def mode_summary(probabilities: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """Mean probability outside the first (reference) mode, and mean of the largest one.

    Args:
        probabilities: Result of mode_probabilities.
        mask: Shape (n_steps,) boolean selection of the steps.

    Returns:
        (non_reference, max_mu), each NaN if no matched IMM track falls in the selection.
    """
    selected = probabilities[:, mask, :]
    valid = np.isfinite(selected).all(axis=-1)
    if not valid.any():
        return float("nan"), float("nan")
    chosen = selected[valid]
    return float(np.mean(1.0 - chosen[:, 0])), float(np.mean(chosen.max(axis=1)))


def bias_errors(bias_log: Sequence[BiasLogEntry], true_bias: float) -> np.ndarray:
    """Absolute error |b_app - b| of the applied bias estimate after every step, in radians."""
    return np.abs(np.array([e.b_app for e in bias_log]) - true_bias)


def convergence_time(errors: np.ndarray, dt: float, threshold: float) -> float:
    """First time in seconds after which the error stays within the threshold to the end.

    Args:
        errors: Shape (n_steps,) absolute errors.
        dt: Time step in seconds.
        threshold: Largest accepted error, same unit as errors.

    Returns:
        0 if the error never exceeds the threshold, NaN if it still does at the last step,
        otherwise the time of the first step of the final stretch within it.
    """
    errors = np.asarray(errors, dtype=float)
    above = np.flatnonzero(~(errors <= threshold))
    if above.size == 0:
        return 0.0
    if above[-1] == len(errors) - 1:
        return float("nan")
    return float((above[-1] + 1) * dt)


def bias_summary(
    errors: np.ndarray,
    bias_log: Sequence[BiasLogEntry],
    mask: np.ndarray,
    burn_in_steps: int,
) -> dict[str, float]:
    """Scalar scores of a bias estimate run.

    Args:
        errors: Result of bias_errors, shape (n_steps,).
        bias_log: The tracker's bias_log (for the applied estimate itself).
        mask: Shape (n_steps,) boolean selection (the focus window) for the window maximum.
        burn_in_steps: Steps excluded from the run mean.

    Returns:
        run_mean_abs_error, window_max_abs_estimate, final_error (radians).
    """
    applied = np.abs(np.array([e.b_app for e in bias_log]))
    return {
        "run_mean_abs_error": float(np.mean(errors[burn_in_steps:])),
        "window_max_abs_estimate": float(applied[mask].max()) if mask.any() else float("nan"),
        "final_error": float(errors[-1]),
    }
