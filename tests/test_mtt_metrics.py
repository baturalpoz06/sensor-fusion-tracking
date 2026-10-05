"""Tests for the multi-target tracking metrics."""

import numpy as np
import pytest
from scipy import stats

from fusion.mtt_metrics import (
    MttMetrics,
    count_id_switches,
    evaluate_mtt,
    false_track_counts,
    first_match_times,
    match_tracks,
    paired_difference,
    seed_confidence_interval,
)
from fusion.mtt_simulation import FieldOfView
from fusion.tracker.multi_target import TrackSnapshot
from fusion.tracker.track import TrackStatus

DT = 0.1
FOV = FieldOfView(range_min=50.0, range_max=3000.0)
CONFIRMED, TENTATIVE = TrackStatus.CONFIRMED, TrackStatus.TENTATIVE
N_STEPS = 40


def make_truth(starts) -> np.ndarray:
    """Straight-line truth (n_targets, N_STEPS, 4): starts are [x, y, vx, vy]."""
    t = np.arange(N_STEPS) * DT
    states = []
    for start in starts:
        xy = start[:2] + np.outer(t, start[2:])
        states.append(np.column_stack([xy, np.tile(start[2:], (N_STEPS, 1))]))
    return np.stack(states)


def snapshot(track_id, position, status=CONFIRMED) -> TrackSnapshot:
    x = np.array([position[0], position[1], 0.0, 0.0])
    return TrackSnapshot(track_id, status, x, np.eye(4))


def perfect_history(truth, ids=None, offset=(0.0, 0.0)):
    ids = ids if ids is not None else range(len(truth))
    return [
        [snapshot(i, truth[j, k, :2] + offset) for j, i in enumerate(ids)] for k in range(N_STEPS)
    ]


TWO = make_truth([np.array([100.0, 500.0, 10.0, 0.0]), np.array([-800.0, -900.0, 0.0, 12.0])])


def evaluate(truth, history, burn_in_steps=0, max_distance=50.0):
    return evaluate_mtt(
        truth, history, dt=DT, fov=FOV, max_distance=max_distance, burn_in_steps=burn_in_steps
    )


# --- matching ------------------------------------------------------------------------


def test_match_tracks_pairs_each_target_with_its_own_track():
    match = match_tracks(TWO, perfect_history(TWO, ids=[7, 3]), max_distance=50.0)
    assert (match.ids[0] == 7).all() and (match.ids[1] == 3).all()
    np.testing.assert_allclose(match.squared_error, 0.0, atol=1e-12)
    assert (match.n_confirmed == 2).all() and (match.n_matched == 2).all()


def test_one_track_serves_only_the_nearer_of_two_close_targets():
    truth = make_truth([np.array([500.0, 0.0, 0.0, 0.0]), np.array([520.0, 0.0, 0.0, 0.0])])
    history = [[snapshot(4, (503.0, 0.0))] for _ in range(N_STEPS)]
    match = match_tracks(truth, history, max_distance=50.0)
    assert (match.ids[0] == 4).all() and (match.ids[1] == -1).all()
    assert (match.n_matched == 1).all()


@pytest.mark.parametrize(("offset", "matched"), [(49.99, True), (50.0, True), (50.01, False)])
def test_a_match_must_be_within_the_distance_and_exactly_at_it_counts(offset, matched):
    truth = make_truth([np.array([500.0, 0.0, 0.0, 0.0])])
    history = [[snapshot(1, (500.0 + offset, 0.0))] for _ in range(N_STEPS)]
    match = match_tracks(truth, history, max_distance=50.0)
    assert (match.ids[0] == 1).all() == matched
    assert (match.ids[0] == -1).all() == (not matched)


def test_tentative_tracks_are_neither_matched_nor_ghosts():
    truth = make_truth([np.array([500.0, 0.0, 0.0, 0.0])])
    history = [
        [snapshot(1, (500.0, 0.0), TENTATIVE), snapshot(2, (900.0, 0.0), TENTATIVE)]
        for _ in range(N_STEPS)
    ]
    match = match_tracks(truth, history, 50.0)
    assert (match.ids == -1).all() and (match.n_confirmed == 0).all()


def test_match_tracks_validates_its_inputs():
    with pytest.raises(ValueError, match="history"):
        match_tracks(TWO, perfect_history(TWO)[:-1], 50.0)
    with pytest.raises(ValueError, match="max_distance"):
        match_tracks(TWO, perfect_history(TWO), 0.0)


def test_no_targets_and_no_tracks_are_handled():
    empty_truth = np.zeros((0, N_STEPS, 4))
    match = match_tracks(empty_truth, [[snapshot(1, (500.0, 0.0))] for _ in range(N_STEPS)], 50.0)
    assert match.ids.shape == (0, N_STEPS) and (match.n_confirmed == 1).all()
    match = match_tracks(TWO, [[] for _ in range(N_STEPS)], 50.0)
    assert (match.ids == -1).all()


# --- id switches and delays ----------------------------------------------------------


@pytest.mark.parametrize(
    ("row", "switches"),
    [
        ([1, 1, 1], 0),
        ([1, 1, 2, 2], 1),
        ([1, -1, -1, 1], 0),
        ([1, -1, 2], 1),
        ([-1, -1, 3, 3, 4, -1, 3], 2),
        ([-1, -1], 0),
    ],
)
def test_count_id_switches(row, switches):
    assert count_id_switches(np.array([row])) == switches


def test_id_switches_are_summed_over_targets():
    assert count_id_switches(np.array([[1, 2, 1], [5, 5, 6]])) == 3


def test_first_match_times():
    ids = np.array([[-1, -1, 4, 4], [2, 2, 2, 2], [-1, -1, -1, -1]])
    delays = first_match_times(ids, dt=0.5)
    np.testing.assert_array_equal(delays[:2], [1.0, 0.0])
    assert np.isnan(delays[2])


# --- evaluate_mtt --------------------------------------------------------------------


def test_perfect_tracking_scores_perfectly():
    metrics = evaluate(TWO, perfect_history(TWO))
    assert isinstance(metrics, MttMetrics)
    assert metrics.position_rmse == pytest.approx(0.0, abs=1e-9)
    assert metrics.ghost_rate == 0.0
    assert metrics.missed_rate == 0.0
    assert metrics.confirmation_delay == 0.0
    assert metrics.confirmed_fraction == 1.0
    assert metrics.id_switches == 0
    assert np.isnan(metrics.births_per_scan)


def test_position_rmse_is_the_offset_of_the_tracks():
    metrics = evaluate(TWO, perfect_history(TWO, offset=(3.0, 4.0)))
    assert metrics.position_rmse == pytest.approx(5.0)


def test_an_unexplained_confirmed_track_is_a_ghost_every_step():
    history = perfect_history(TWO)
    for k in range(N_STEPS):
        history[k].append(snapshot(99, (2000.0, 2000.0)))
    metrics = evaluate(TWO, history)
    assert metrics.ghost_rate == pytest.approx(1.0)
    assert metrics.missed_rate == 0.0
    assert metrics.position_rmse == pytest.approx(0.0, abs=1e-9)


def test_a_ghost_that_lives_half_of_the_steps_scores_one_half():
    history = perfect_history(TWO)
    for k in range(N_STEPS // 2):
        history[k].append(snapshot(99, (2000.0, 2000.0)))
    assert evaluate(TWO, history).ghost_rate == pytest.approx(0.5)


def test_a_target_without_a_track_is_missed():
    history = perfect_history(TWO, ids=[0, 1])
    history = [[s for s in step if s.track_id == 0] for step in history]
    metrics = evaluate(TWO, history)
    assert metrics.missed_rate == pytest.approx(0.5)
    assert metrics.confirmed_fraction == pytest.approx(0.5)
    assert metrics.confirmation_delay == 0.0  # only the matched target contributes
    assert metrics.ghost_rate == 0.0


def test_a_track_that_starts_late_delays_confirmation_and_counts_as_missed_until_then():
    history = perfect_history(TWO)
    for k in range(10):
        history[k] = [s for s in history[k] if s.track_id == 1]
    metrics = evaluate(TWO, history)
    assert metrics.confirmation_delay == pytest.approx(0.5 * 10 * DT)
    assert metrics.missed_rate == pytest.approx(10 / (2 * N_STEPS))


def test_targets_outside_the_field_of_view_do_not_count_as_missed():
    outside = make_truth([np.array([5000.0, 0.0, 0.0, 0.0])])
    truth = np.concatenate([TWO, outside])
    history = perfect_history(TWO, ids=[0, 1])
    assert evaluate(truth, history).missed_rate == 0.0
    assert np.isnan(evaluate(outside, [[] for _ in range(N_STEPS)]).missed_rate)


def test_burn_in_excludes_early_errors_but_not_the_delay():
    history = perfect_history(TWO, offset=(0.0, 0.0))
    for k in range(10):
        history[k] = [snapshot(0, TWO[0, k, :2] + 30.0), snapshot(1, TWO[1, k, :2])]
    with_burn_in = evaluate(TWO, history, burn_in_steps=10)
    assert with_burn_in.position_rmse == pytest.approx(0.0, abs=1e-9)
    assert evaluate(TWO, history, burn_in_steps=0).position_rmse > 5.0
    assert with_burn_in.confirmation_delay == 0.0


def test_id_switch_is_counted_when_a_target_changes_track():
    history = perfect_history(TWO, ids=[0, 1])
    for k in range(20, N_STEPS):
        history[k] = [snapshot(5, TWO[0, k, :2]), snapshot(1, TWO[1, k, :2])]
    assert evaluate(TWO, history).id_switches == 1
    # The switch happens before the burn-in ends: not counted.
    assert evaluate(TWO, history, burn_in_steps=25).id_switches == 0


def test_no_valid_matches_gives_nan_error_and_nan_delay():
    metrics = evaluate(TWO, [[] for _ in range(N_STEPS)])
    assert np.isnan(metrics.position_rmse) and np.isnan(metrics.confirmation_delay)
    assert metrics.missed_rate == 1.0 and metrics.confirmed_fraction == 0.0


# --- seed confidence interval --------------------------------------------------------


def test_seed_confidence_interval_matches_student_t():
    values = np.array([1.0, 2.0, 4.0, 3.0, 5.5, 2.5])
    result = seed_confidence_interval(values, confidence=0.9)
    low, high = stats.t.interval(0.9, df=5, loc=values.mean(), scale=stats.sem(values))
    assert result.mean == pytest.approx(values.mean())
    assert result.lower == pytest.approx(low) and result.upper == pytest.approx(high)
    assert result.n_valid == 6


def test_seed_confidence_interval_skips_nan():
    values = np.array([1.0, np.nan, 3.0, 5.0, np.nan])
    result = seed_confidence_interval(values)
    clean = seed_confidence_interval(np.array([1.0, 3.0, 5.0]))
    assert result == clean and result.n_valid == 3


@pytest.mark.parametrize(
    ("values", "n_valid", "mean_is_nan"),
    [([np.nan, np.nan], 0, True), ([2.0, np.nan], 1, False), ([], 0, True)],
)
def test_too_few_valid_seeds_give_a_nan_band_instead_of_an_error(values, n_valid, mean_is_nan):
    result = seed_confidence_interval(np.array(values))
    assert result.n_valid == n_valid
    assert np.isnan(result.lower) and np.isnan(result.upper)
    assert np.isnan(result.mean) == mean_is_nan


def test_seed_confidence_interval_rejects_an_invalid_confidence():
    with pytest.raises(ValueError, match="confidence"):
        seed_confidence_interval(np.array([1.0, 2.0]), confidence=1.0)


# --- false tracks versus off-target tracks -------------------------------------------


def test_false_track_counts_ignore_matched_ids_and_tentative_tracks():
    history = [
        [snapshot(1, (0.0, 0.0)), snapshot(2, (0.0, 0.0)), snapshot(3, (0.0, 0.0), TENTATIVE)],
        [snapshot(2, (0.0, 0.0))],
        [],
    ]
    np.testing.assert_array_equal(false_track_counts(history, {1}), [1, 1, 0])
    np.testing.assert_array_equal(false_track_counts(history, {1, 2}), [0, 0, 0])


def test_a_never_matched_far_track_is_both_a_ghost_and_a_false_track():
    history = perfect_history(TWO)
    for k in range(N_STEPS):
        history[k].append(snapshot(99, (2000.0, 2000.0)))
    metrics = evaluate(TWO, history)
    assert metrics.ghost_rate == pytest.approx(1.0)
    assert metrics.false_track_rate == pytest.approx(1.0)


def test_a_track_pulled_off_its_target_is_a_ghost_but_not_a_false_track():
    history = perfect_history(TWO, ids=[0, 1])
    for k in range(20, N_STEPS):  # track 0 leaves target 0 by 300 m for the second half
        history[k][0] = snapshot(0, TWO[0, k, :2] + np.array([300.0, 0.0]))
    metrics = evaluate(TWO, history)
    assert metrics.ghost_rate == pytest.approx(0.5)  # unmatched in 20 of 40 steps
    assert metrics.missed_rate == pytest.approx(0.25)  # target 0 unmatched in 20 of 80 pairs
    assert metrics.false_track_rate == 0.0  # it was on target 0 before
    assert metrics.id_switches == 0


def test_a_duplicate_track_next_to_a_target_is_a_false_track():
    history = perfect_history(TWO, ids=[0, 1])
    for k in range(N_STEPS):
        history[k].append(snapshot(7, TWO[0, k, :2] + np.array([10.0, 0.0])))
    metrics = evaluate(TWO, history)
    # Track 0 is closer and claims target 0; track 7 never matches anything.
    assert metrics.false_track_rate == pytest.approx(1.0)
    assert metrics.ghost_rate == pytest.approx(1.0)
    assert metrics.id_switches == 0


def test_false_tracks_are_counted_only_after_the_burn_in():
    history = perfect_history(TWO)
    for k in range(10):
        history[k].append(snapshot(99, (2000.0, 2000.0)))
    assert evaluate(TWO, history, burn_in_steps=0).false_track_rate == pytest.approx(10 / N_STEPS)
    assert evaluate(TWO, history, burn_in_steps=10).false_track_rate == 0.0


# --- paired difference ---------------------------------------------------------------------


def test_paired_difference_of_known_values():
    """Fails if the difference has the wrong sign or the interval is not on the differences."""
    a = np.array([3.0, 5.0, 4.0, 8.0])
    b = np.array([1.0, 4.0, 4.0, 5.0])
    result = paired_difference(a, b)
    np.testing.assert_allclose(result.mean, 1.5)  # a - b = 2, 1, 0, 3
    assert result.median == pytest.approx(1.5) and result.n_valid == 4 and result.n_dropped == 0
    expected = seed_confidence_interval(a - b)
    assert (result.lower, result.upper) == pytest.approx((expected.lower, expected.upper))
    reverse = paired_difference(b, a)
    assert reverse.mean == pytest.approx(-1.5) and reverse.lower == pytest.approx(-result.upper)


def test_paired_difference_drops_and_counts_seeds_where_either_value_is_undefined():
    """Fails if a pair with a NaN is kept, or the dropped seeds (survivorship) are not counted."""
    a = np.array([2.0, np.nan, 4.0, 6.0, np.inf])
    b = np.array([1.0, 3.0, np.nan, 2.0, 1.0])
    result = paired_difference(a, b)
    assert result.n_valid == 2 and result.n_dropped == 3
    assert result.mean == pytest.approx(2.5) and result.median == pytest.approx(2.5)


def test_a_paired_difference_is_tighter_than_two_separate_intervals_for_a_shared_seed_effect():
    """Fails if pairing is not done per seed: a shift shared by both settings must cancel."""
    seed_effect = np.random.default_rng(0).normal(0.0, 10.0, 50)
    result = paired_difference(seed_effect + 1.0, seed_effect)
    assert result.mean == pytest.approx(1.0)
    assert result.upper - result.lower < 1e-9  # no variation left in the differences


def test_paired_difference_with_too_few_pairs_has_no_interval_and_checks_its_input():
    """Fails if one pair claims an interval, or mismatched inputs are accepted."""
    one = paired_difference(np.array([3.0, np.nan]), np.array([1.0, 2.0]))
    assert one.n_valid == 1 and one.mean == 2.0 and np.isnan(one.lower) and np.isnan(one.upper)
    none = paired_difference(np.array([np.nan]), np.array([1.0]))
    assert none.n_valid == 0 and np.isnan(none.mean) and np.isnan(none.median)
    with pytest.raises(ValueError, match="same shape"):
        paired_difference(np.zeros(3), np.zeros(4))
    with pytest.raises(ValueError, match="confidence"):
        paired_difference(np.zeros(3), np.ones(3), confidence=1.5)
