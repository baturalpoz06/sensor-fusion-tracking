"""Tests of the Phase 8b metrics against hand-built matchings.

Each test docstring names the defect that makes it fail.
"""

import numpy as np
import pytest

from fusion.improvement_metrics import (
    bias_errors,
    bias_summary,
    bin_edges,
    binned_rates,
    common_support_rmse,
    convergence_time,
    drag_mask,
    mode_probabilities,
    mode_summary,
    window_mask,
)
from fusion.tracker.camera_bias import BiasLogEntry
from fusion.tracker.multi_target import ModeLogEntry

DT = 0.1


def hand_built_matching():
    """One target, 100 steps (10 s). Narrow id 7 until step 44, lost from 45 to 79, then id 8.

    Wide distance: id 7 is still on the target at steps 45-59 (dragged), at 60-70 a different
    track (id 9, a re-birth) is near it, from 71 nothing. Steps 75-79 are outside the view.
    """
    narrow = np.full((1, 100), -1)
    narrow[0, :45] = 7
    narrow[0, 80:] = 8
    wide = narrow.copy()
    wide[0, 45:60] = 7
    wide[0, 60:71] = 9
    in_view = np.ones((1, 100), dtype=bool)
    in_view[0, 75:80] = False
    return narrow, wide, in_view


def test_bins_start_at_the_burn_in_and_cover_the_run():
    """Fails if bins start at zero (confirmation would be counted) or leave steps out."""
    edges = bin_edges(100, DT, 2.0, 2.0)
    assert edges == [(20, 40), (40, 60), (60, 80), (80, 100)]
    assert bin_edges(105, DT, 2.0, 2.0)[-1] == (100, 105)


def test_drag_needs_the_same_track_to_be_alive_within_the_wide_distance():
    """Fails if a re-born track near the target counts as drag, or a dragged track does not."""
    narrow, wide, in_view = hand_built_matching()
    drag = drag_mask(narrow, wide, in_view)
    assert drag[0, 45:60].all()
    assert not drag[0, :45].any() and not drag[0, 60:].any()


def test_missed_and_drag_rates_per_bin_match_the_hand_count():
    """Fails if the rates use the wrong denominator or leave out-of-view pairs in.

    Bin [40, 60): steps 45-59 missed (15 of 20), all dragged. Bin [60, 80): steps 60-79 are
    unmatched, steps 75-79 out of view, so 15 of 15 missed and none dragged.
    """
    narrow, wide, in_view = hand_built_matching()
    missed, drag = binned_rates(narrow, wide, in_view, bin_edges(100, DT, 2.0, 2.0))
    np.testing.assert_allclose(missed, [0.0, 0.75, 1.0, 0.0])
    np.testing.assert_allclose(drag, [0.0, 0.75, 0.0, 0.0])


def test_a_bin_without_targets_in_view_has_no_rate():
    """Fails if an empty bin reports 0 instead of NaN."""
    narrow, wide, _ = hand_built_matching()
    nothing = np.zeros((1, 100), dtype=bool)
    missed, drag = binned_rates(narrow, wide, nothing, [(20, 40)])
    assert np.isnan(missed).all() and np.isnan(drag).all()


def test_the_rmse_is_taken_over_the_pairs_that_every_arm_matched():
    """Fails if an arm's RMSE keeps the pairs the other arm lost.

    Arm A matched steps 0-3 (errors 1, 1, 1, 5 m), arm B only steps 0-2 (errors 2 m): the
    common pairs are 0-2, so A's RMSE is 1 (not sqrt(28/4)), B's is 2.
    """
    a = np.array([[1.0, 1.0, 1.0, 25.0]])
    b = np.array([[4.0, 4.0, 4.0, np.nan]])
    rmse, n = common_support_rmse([a, b], np.ones(4, dtype=bool))
    assert n == 3
    np.testing.assert_allclose(rmse, [1.0, 2.0])
    only, n_masked = common_support_rmse([a, b], np.array([False, False, False, True]))
    assert n_masked == 0 and np.isnan(only).all()


def test_the_window_mask_is_the_half_open_focus_window():
    """Fails if the window includes its end step or misses its start step."""
    mask = window_mask(100, (2.0, 5.0), DT)
    assert mask.sum() == 30 and mask[20] and mask[49] and not mask[50] and not mask[19]


def test_mode_probabilities_follow_the_matched_track_of_each_target():
    """Fails if probabilities of unmatched or other tracks are attributed to a target."""
    log = [
        (ModeLogEntry(3, (0.9, 0.1), 0.0), ModeLogEntry(4, (0.2, 0.8), 0.0)),
        (ModeLogEntry(3, (0.7, 0.3), 0.0),),
    ]
    ids = np.array([[3, 3], [4, -1]])
    mu = mode_probabilities(log, ids, 2)
    np.testing.assert_array_equal(mu[0, 0], [0.9, 0.1])
    np.testing.assert_array_equal(mu[0, 1], [0.7, 0.3])
    np.testing.assert_array_equal(mu[1, 0], [0.2, 0.8])
    assert np.isnan(mu[1, 1]).all()


def test_the_mode_summary_averages_the_matched_pairs_in_the_selection():
    """Fails if NaN pairs enter the mean, or the reference mode is not the first one."""
    mu = np.array([[[0.9, 0.1], [0.5, 0.5]], [[0.2, 0.8], [np.nan, np.nan]]])
    non_reference, largest = mode_summary(mu, np.array([True, True]))
    assert non_reference == pytest.approx(np.mean([0.1, 0.5, 0.8]))
    assert largest == pytest.approx(np.mean([0.9, 0.5, 0.8]))
    nothing = mode_summary(mu, np.array([False, False]))
    assert np.isnan(nothing).all()


def test_convergence_time_is_the_start_of_the_final_stretch_within_the_threshold():
    """Fails if the first dip below the threshold counts, or a run that ends above it converged."""
    errors = np.array([1.0, 0.9, 0.5, 0.2, 0.05, 0.3, 0.05, 0.05])
    assert convergence_time(errors, DT, 0.1) == pytest.approx(0.6)
    assert convergence_time(np.array([0.05, 0.02]), DT, 0.1) == 0.0
    assert np.isnan(convergence_time(np.array([0.05, 0.5]), DT, 0.1))
    assert np.isnan(convergence_time(np.array([0.05, np.nan]), DT, 0.1))


def entry(b_app: float) -> BiasLogEntry:
    return BiasLogEntry(0.0, 0.0, 1.0, b_app, 0.0, 0, 0)


def test_bias_errors_and_summary_on_a_synthetic_log():
    """Fails if the error is not |b_app - b|, the run mean includes the burn-in, or the window
    maximum looks outside the window."""
    log = [entry(b) for b in (0.0, 0.5, 0.9, 1.1, 1.0, 1.0)]
    errors = bias_errors(log, 1.0)
    np.testing.assert_allclose(errors, [1.0, 0.5, 0.1, 0.1, 0.0, 0.0])
    mask = np.array([False, False, True, True, False, False])
    summary = bias_summary(errors, log, mask, burn_in_steps=2)
    assert summary["run_mean_abs_error"] == pytest.approx(np.mean([0.1, 0.1, 0.0, 0.0]))
    assert summary["window_max_abs_estimate"] == pytest.approx(1.1)
    assert summary["final_error"] == 0.0
