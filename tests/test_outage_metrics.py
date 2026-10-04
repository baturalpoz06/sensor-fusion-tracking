"""Tests for the outage metrics, on hand-built track histories.

Each test docstring names the condition that makes it fail without the feature.
"""

import numpy as np
import pytest
from scipy.stats import chi2

from fusion.mtt_metrics import MttMetrics, evaluate_mtt, match_tracks
from fusion.mtt_simulation import FieldOfView
from fusion.outage_metrics import (
    FIELDS,
    OUTLIER_QUANTILE,
    OutageMetrics,
    evaluate_outage,
    outage_windows,
    window_id_switches,
)
from fusion.tracker.multi_target import TrackSnapshot
from fusion.tracker.track import TrackStatus

DT = 0.1
N = 100
BURN_IN = 10
FOV = FieldOfView(range_min=50.0, range_max=3000.0)
CONFIRMED, TENTATIVE = TrackStatus.CONFIRMED, TrackStatus.TENTATIVE
SPAN = (3.0, 5.0)  # outage steps 30..49
AFTER = 2.0  # after window: steps 50..69
SCHEDULED = np.arange(N) % 10 == 0
STARTS = [
    [100.0, 500.0, 10.0, 0.0],
    [-800.0, -900.0, 0.0, 12.0],
    [1500.0, 1500.0, -8.0, -5.0],
    [-1500.0, 1200.0, 5.0, -9.0],
]
P_UNIT = np.diag([4.0, 4.0, 1.0, 1.0])


def straight_truth(n_targets: int) -> np.ndarray:
    t = np.arange(N) * DT
    states = []
    for start in np.array(STARTS[:n_targets]):
        xy = start[:2] + np.outer(t, start[2:])
        states.append(np.column_stack([xy, np.tile(start[2:], (N, 1))]))
    return np.stack(states)


def snap(track_id, state, cov=None, status=CONFIRMED) -> TrackSnapshot:
    state = np.asarray(state, dtype=float)
    return TrackSnapshot(track_id, status, state, P_UNIT if cov is None else cov)


def build_history(rule) -> list[list[TrackSnapshot]]:
    """History from rule(k) -> list of snapshots at step k."""
    return [rule(k) for k in range(N)]


def down_flags(first: int, last: int) -> np.ndarray:
    flags = np.zeros(N, dtype=bool)
    flags[first:last] = True
    return flags


def score(truth, history, **overrides) -> OutageMetrics:
    arguments = {
        "dt": DT,
        "fov": FOV,
        "max_distance": 50.0,
        "burn_in_steps": BURN_IN,
        "span": SPAN,
        "after_window": AFTER,
        "radar_down": down_flags(30, 50),
        "scheduled": SCHEDULED,
    }
    return evaluate_outage(truth, history, **{**arguments, **overrides})


def perfect(truth, k, offset=(0.0, 0.0), ids=None):
    ids = range(len(truth)) if ids is None else ids
    return [snap(i, truth[j, k] + np.array([*offset, 0.0, 0.0])) for j, i in enumerate(ids)]


# --- structure and windows -----------------------------------------------------------


def test_outage_metrics_have_the_window_run_and_outage_fields():
    """Fails if a window or a Phase 6 metric is missing from the result."""
    assert OutageMetrics._fields == FIELDS
    for window in ("before", "during", "after"):
        assert f"{window}_position_rmse" in FIELDS and f"{window}_id_switches" in FIELDS
    assert all(f"run_{name}" in FIELDS for name in MttMetrics._fields)
    assert len(set(FIELDS)) == len(FIELDS)


def steps_of(mask):
    steps = np.flatnonzero(mask)
    return (int(steps[0]), int(steps[-1])) if steps.size else None


def test_windows_are_half_open_and_partition_the_run_around_the_span():
    """Fails with closed intervals or a shifted boundary: a step would lie in two windows."""
    windows = outage_windows(N, DT, BURN_IN, SPAN, AFTER)
    assert (windows.start_step, windows.end_step) == (30, 50)
    assert steps_of(windows.before) == (10, 29)
    assert steps_of(windows.during) == (30, 49)
    assert steps_of(windows.after) == (50, 69)
    assert not (windows.before & windows.during).any()
    assert not (windows.during & windows.after).any()


@pytest.mark.parametrize(
    ("span", "after", "before", "during", "after_steps"),
    [
        ((8.0, 9.0), 5.0, (10, 79), (80, 89), (90, 99)),  # the after window is clipped
        ((3.0, 3.0), 2.0, (10, 29), None, (30, 49)),  # a zero-length outage has no during window
        ((0.5, 5.0), 2.0, None, (5, 49), (50, 69)),  # the outage starts inside the burn-in
        (None, 2.0, (10, 99), None, None),  # no outage: everything is "before"
        ((9.5, 20.0), 2.0, (10, 94), (95, 99), None),  # the outage runs past the end of the run
    ],
)
def test_window_edge_cases(span, after, before, during, after_steps):
    """Fails if clipping, an empty outage, a burn-in overlap or a missing span is mishandled."""
    windows = outage_windows(N, DT, BURN_IN, span, after)
    assert steps_of(windows.before) == before
    assert steps_of(windows.during) == during
    assert steps_of(windows.after) == after_steps


# --- scores per window ---------------------------------------------------------------


def test_window_scores_follow_the_window_and_the_run_scores_equal_phase_6_bitwise():
    """Fails if the error leaks across windows or the run scores differ from evaluate_mtt."""
    truth = straight_truth(1)
    history = build_history(
        lambda k: perfect(truth, k, offset=(3.0, 4.0) if 30 <= k < 50 else (0.0, 0.0))
    )
    result = score(truth, history)
    assert result.before_position_rmse == pytest.approx(0.0, abs=1e-9)
    assert result.during_position_rmse == pytest.approx(5.0)
    assert result.after_position_rmse == pytest.approx(0.0, abs=1e-9)
    assert result.run_position_rmse == pytest.approx(5.0 * np.sqrt(20 / 90))
    assert result.before_ghost_rate == 0.0 and result.during_missed_rate == 0.0

    reference = evaluate_mtt(
        truth, history, dt=DT, fov=FOV, max_distance=50.0, burn_in_steps=BURN_IN
    )
    for name in MttMetrics._fields:
        np.testing.assert_equal(getattr(result, f"run_{name}"), getattr(reference, name))


def test_a_run_without_an_outage_scores_everything_as_before():
    """Fails if a missing outage produces windows or NaN-free outage scores out of nothing."""
    truth = straight_truth(1)
    result = score(truth, build_history(lambda k: perfect(truth, k)), span=None)
    assert result.before_position_rmse == pytest.approx(0.0, abs=1e-9)
    for name in ("during_position_rmse", "after_ghost_rate", "during_missed_rate"):
        assert np.isnan(getattr(result, name))
    assert np.isnan(result.reacquired_fraction) and np.isnan(result.nees_samples)


def test_a_target_that_ceased_to_exist_is_neither_matched_nor_missed():
    """Fails if a vanished target is still matched to the track next to it, or counted as missed."""
    truth = straight_truth(2)
    alive = np.ones((2, N), dtype=bool)
    alive[1, 60:] = False
    history = build_history(lambda k: perfect(truth, k))  # the track of target 1 stays

    matching = match_tracks(truth, history, 50.0, alive)
    assert (matching.ids[1, :60] == 1).all() and (matching.ids[1, 60:] == -1).all()
    assert (match_tracks(truth, history, 50.0).ids[1, 60:] == 1).all()  # precondition

    result = score(truth, history, alive=alive)
    assert result.after_missed_rate == 0.0  # the vanished target is not counted as missed
    assert result.after_ghost_rate == pytest.approx(0.5)  # its track is unexplained from step 60


def test_match_with_every_target_alive_is_identical_to_the_default():
    """Fails if the subset path reorders the cost matrix or changes tie-breaks."""
    truth = straight_truth(3)
    history = build_history(
        lambda k: perfect(truth, k, offset=(2.0 * (k % 5), 1.0))
        + [snap(9, truth[0, k] + np.array([30.0, 0.0, 0.0, 0.0]))]
    )
    default = match_tracks(truth, history, 50.0)
    explicit = match_tracks(truth, history, 50.0, np.ones((3, N), dtype=bool))
    for a, b in zip(default, explicit, strict=True):
        np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match="alive"):
        match_tracks(truth, history, 50.0, np.ones((2, N), dtype=bool))


def test_id_switches_start_from_the_last_id_before_the_window():
    """Fails if a window counts only changes inside it: the outage switch would be lost."""
    ids = np.full((1, N), -1)
    ids[0, :30] = 0  # the track before the outage
    ids[0, 55:] = 5  # a new track after it (nothing is matched in between)
    windows = outage_windows(N, DT, BURN_IN, SPAN, AFTER)
    assert window_id_switches(ids, windows.before, carry_over=False) == 0
    assert window_id_switches(ids, windows.during, carry_over=True) == 0
    assert window_id_switches(ids, windows.after, carry_over=True) == 1
    assert window_id_switches(ids, windows.after, carry_over=False) == 0  # the Phase 6 style
    assert np.isnan(window_id_switches(ids, np.zeros(N, dtype=bool), carry_over=True))

    ids[0, 60:65] = 6  # a change inside the window counts too
    assert window_id_switches(ids, windows.after, carry_over=True) == 3


def test_a_track_matched_only_after_a_window_is_false_in_that_window():
    """Fails if a window's false-track rate uses the matches of the whole run.

    Track 9 sits 500 m from the target until step 29 and is the target's track from
    step 60 on. Within the before window it is unmatched (false); the whole-run definition,
    which sees the later match, calls it not false.
    """
    truth = straight_truth(1)

    def rule(k):
        tracks = [snap(0, truth[0, k])] if k < 15 or 30 <= k < 60 else []
        if 15 <= k < 30:
            return [snap(0, truth[0, k]), snap(9, truth[0, k] + np.array([500.0, 0, 0, 0]))]
        if k >= 60:
            return [snap(9, truth[0, k])]
        return tracks

    result = score(truth, build_history(rule))
    assert result.before_false_track_rate == pytest.approx(15 / 20)  # steps 15..29 of 10..29
    assert result.run_false_track_rate == 0.0  # the whole run sees track 9 matched later
    assert result.during_false_track_rate == 0.0 and result.after_false_track_rate == 0.0


def test_births_are_counted_per_scheduled_scan_of_each_window():
    """Fails if births are assigned to the wrong window or lost scans are left out of the rate."""
    truth = straight_truth(1)
    born = {1: 15, 2: 45, 3: 60, 4: 60}  # track id -> first step

    def rule(k):
        far = np.array([2500.0, 2500.0, 0.0, 0.0])
        return [snap(0, truth[0, k])] + [
            snap(i, far + i, status=TENTATIVE) for i, first in born.items() if k >= first
        ]

    result = score(truth, build_history(rule))
    assert result.before_births_per_scan == pytest.approx(1 / 2)  # scans 10, 20; track 1
    assert result.during_births_per_scan == pytest.approx(1 / 2)  # scans 30, 40 (lost); track 2
    assert result.after_births_per_scan == pytest.approx(2 / 2)  # scans 50, 60; tracks 3, 4


# --- reacquisition and identity ------------------------------------------------------


def test_reacquisition_time_fraction_censoring_and_identity():
    """Fails if survivors are not counted, a target never found is hidden, or ids are not compared.

    Target 0 keeps its track (0 s, identity kept); target 1 is found again after 0.5 s by a
    new track (identity lost); target 2 is never found (counts as lost, lowers the
    fraction, enters the censored mean with the window length); target 3 was not tracked
    when the outage began, so it is not eligible even though it is tracked afterwards.
    """
    truth = straight_truth(4)

    def rule(k):
        tracks = [snap(0, truth[0, k])]
        if k < 30:
            tracks += [snap(1, truth[1, k]), snap(2, truth[2, k])]
        if k >= 55:
            tracks.append(snap(11, truth[1, k]))
        if k >= 40:
            tracks.append(snap(3, truth[3, k]))
        return tracks

    result = score(truth, build_history(rule))
    window_length = 2.0  # steps 50..69
    assert result.reacquisition_time == pytest.approx((0.0 + 0.5) / 2)
    assert result.reacquired_fraction == pytest.approx(2 / 3)
    assert result.reacquisition_time_censored == pytest.approx((0.0 + 0.5 + window_length) / 3)
    assert result.identity_kept == pytest.approx(1 / 3)


def test_no_eligible_target_leaves_the_reacquisition_scores_undefined():
    """Fails if an empty set of eligible targets gives 0 or a division error instead of NaN."""
    truth = straight_truth(1)
    result = score(truth, build_history(lambda k: []))
    for name in ("reacquisition_time", "reacquired_fraction", "identity_kept"):
        assert np.isnan(getattr(result, name))


# --- NEES ----------------------------------------------------------------------------


def error_track(truth, k, error, cov=None, track_id=0):
    """Snapshot whose estimate is off from the truth by `error` (error = truth - estimate)."""
    return snap(track_id, truth[0, k] - np.asarray(error, dtype=float), cov)


def nees_history(truth, before_error, down_error, up_error):
    def rule(k):
        error = before_error if k < 30 else down_error if k < 40 else up_error
        return [error_track(truth, k, error)]

    return build_history(rule)


def test_nees_is_taken_on_the_radar_down_steps_of_the_outage_only():
    """Fails if steps without an outage or the wrong window enter the outage mean.

    Hand values with P = diag(4, 4, 1, 1): error (2, 0, 0, 0.5) gives 1 + 0.25 = 1.25,
    (2, 2, 1, 0) gives 3, (20, 0, 0, 0) gives 100. Radar down at steps 30..39 only.
    """
    truth = straight_truth(1)
    history = nees_history(truth, (2, 2, 1, 0), (2, 0, 0, 0.5), (20, 0, 0, 0))
    result = score(truth, history, radar_down=down_flags(30, 40))
    assert result.nees_outage == pytest.approx(1.25)
    assert result.nees_samples == 10
    assert result.nees_before == pytest.approx(3.0)
    assert result.nees_outlier_fraction == 0.0


def test_nees_counts_outliers_above_the_chi_square_quantile():
    """Fails if the outlier share is not computed from the NEES values themselves."""
    truth = straight_truth(1)

    def rule(k):
        error = (10, 0, 0, 0) if k < 35 else (2, 0, 0, 0.5)  # NEES 25 and 1.25
        return [error_track(truth, k, error if k >= 30 else (0, 0, 0, 0))]

    result = score(truth, build_history(rule), radar_down=down_flags(30, 40))
    assert 25.0 > chi2.ppf(OUTLIER_QUANTILE, 4) > 1.25
    assert result.nees_outlier_fraction == pytest.approx(0.5)
    assert result.nees_outage == pytest.approx((25.0 * 5 + 1.25 * 5) / 10)


def test_a_track_far_from_its_target_is_still_scored():
    """Fails if the NEES is only taken where the track is within the match distance.

    The track is 80 m off, beyond the 50 m match distance, so it is not matched; with
    P = diag(1600, 1600, 1, 1) its NEES is 80^2 / 1600 = 4.
    """
    truth = straight_truth(1)
    wide = np.diag([1600.0, 1600.0, 1.0, 1.0])

    def rule(k):
        return [error_track(truth, k, (80, 0, 0, 0) if k >= 30 else (0, 0, 0, 0), wide)]

    result = score(truth, build_history(rule))
    assert result.during_missed_rate == 1.0  # precondition: unmatched during the outage
    assert result.nees_outage == pytest.approx(4.0)
    assert result.nees_samples == 20


def test_a_deleted_anchor_track_stops_contributing_to_the_nees():
    """Fails if steps without the anchor track are scored with another track or counted."""
    truth = straight_truth(1)

    def rule(k):
        return [error_track(truth, k, (2, 0, 0, 0.5))] if k < 35 else []

    result = score(truth, build_history(rule), radar_down=down_flags(30, 40))
    assert result.nees_samples == 5 and result.nees_outage == pytest.approx(1.25)


def test_a_target_without_an_anchor_has_no_nees():
    """Fails if a target that was not tracked at the outage start gets an outage NEES."""
    truth = straight_truth(1)

    def rule(k):
        return [error_track(truth, k, (2, 0, 0, 0.5))] if k >= 30 else []

    result = score(truth, build_history(rule))
    assert np.isnan(result.nees_outage) and result.nees_samples == 0


# --- ghosts --------------------------------------------------------------------------


def ghost_scene(last_step_of_track):
    truth = straight_truth(1)
    alive = np.ones((1, N), dtype=bool)
    alive[0, 60:] = False
    history = build_history(
        lambda k: [snap(0, truth[0, k])] if k <= last_step_of_track else []
    )
    return truth, alive, history


def test_ghost_lifetime_runs_from_the_end_of_the_target_to_the_end_of_its_track():
    """Fails if the lifetime is measured from the wrong step or a survivor is not censored."""
    truth, alive, history = ghost_scene(84)  # the track disappears at step 85
    result = score(truth, history, alive=alive, vanish={0: 6.0})
    assert result.ghost_lifetime == pytest.approx((85 - 60) * DT)
    assert result.ghost_censored == 0.0
    assert result.after_ghost_rate == pytest.approx(0.5)  # unexplained over steps 60..69 of 50..69

    truth, alive, history = ghost_scene(N)  # never deleted
    result = score(truth, history, alive=alive, vanish={0: 6.0})
    assert result.ghost_lifetime == pytest.approx((N - 60) * DT) and result.ghost_censored == 1.0


def test_a_vanished_target_without_a_track_has_no_ghost_lifetime():
    """Fails if a target that was not tracked when it vanished gets a lifetime of zero."""
    truth, alive, history = ghost_scene(55)  # the track ended before the target did
    result = score(truth, history, alive=alive, vanish={0: 6.0})
    assert np.isnan(result.ghost_lifetime) and np.isnan(result.ghost_censored)
    assert np.isnan(score(truth, history).ghost_lifetime)  # nothing vanishes
