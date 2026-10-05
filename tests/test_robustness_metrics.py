"""Tests for the robustness metrics: line-of-sight split, NEES, camera accept rates, windows.

Each test docstring names the condition that makes it fail without the feature.
"""

import numpy as np
import pytest

from fusion.mtt_simulation import FieldOfView, Scan
from fusion.robustness_metrics import (
    DIAGNOSTIC_FIELDS,
    FIELDS,
    RobustnessMetrics,
    camera_rates,
    evaluate_robustness,
    los_error_split,
    match_fraction,
    min_separation,
)
from fusion.tracker.multi_target import CameraStepLog, TrackSnapshot
from fusion.tracker.track import TrackStatus

DT = 0.1
N_STEPS = 40
SIGMA = 10.0
EMPTY_SCAN = Scan(np.zeros((0, 1)), np.zeros(0, dtype=int))
QUIET_LOG = [CameraStepLog()] * N_STEPS
QUIET_SCANS = [EMPTY_SCAN] * N_STEPS


def make_truth() -> np.ndarray:
    """Two stationary targets: one on the +x axis, one on the +y axis, 1000 m away."""
    truth = np.zeros((2, N_STEPS, 4))
    truth[0, :, :2] = [1000.0, 0.0]
    truth[1, :, :2] = [0.0, 1000.0]
    return truth


def make_history(offsets, steps=range(N_STEPS)):
    """Confirmed tracks 0 and 1 on the targets, displaced by the given (dx, dy) offsets."""
    truth = make_truth()
    covariance = SIGMA**2 * np.eye(4)
    history = []
    for k in range(N_STEPS):
        step = []
        if k in steps:
            for track, (dx, dy) in enumerate(offsets):
                x = truth[track, k].copy()
                x[:2] += [dx, dy]
                step.append(TrackSnapshot(track, TrackStatus.CONFIRMED, x, covariance))
        history.append(step)
    return history


def evaluate(history, **overrides):
    settings = dict(
        dt=DT,
        fov=FieldOfView(),
        max_distance=50.0,
        wide_distance=200.0,
        burn_in_steps=5,
    )
    settings.update(overrides)
    return evaluate_robustness(make_truth(), history, QUIET_LOG, QUIET_SCANS, **settings)


# --- line-of-sight split ---------------------------------------------------------------


def test_a_radial_error_is_pure_along_range_and_a_tangential_one_pure_cross_range():
    """Fails if the axes of the split are swapped or not aligned with the line of sight."""
    truth = np.array([[1000.0, 0.0], [0.0, 1000.0], [600.0, 800.0], [-500.0, -500.0]])
    unit = truth / np.hypot(*truth.T)[:, None]
    tangent = np.column_stack([-unit[:, 1], unit[:, 0]])
    along, cross = los_error_split(7.0 * unit, truth)
    np.testing.assert_allclose(along, 7.0, atol=1e-12)
    np.testing.assert_allclose(cross, 0.0, atol=1e-12)
    along, cross = los_error_split(-3.0 * tangent, truth)
    np.testing.assert_allclose(along, 0.0, atol=1e-12)
    np.testing.assert_allclose(cross, -3.0, atol=1e-12)


def test_a_positive_bearing_error_is_a_positive_cross_range_error():
    """Fails if the sign convention is lost: counter-clockwise (positive bearing) is positive.

    A positive bearing bias rotates the estimate counter-clockwise about the radar.
    """
    truth = np.array([[1000.0, 0.0], [0.0, 1000.0], [-800.0, 600.0]])
    bias = 0.01
    rotated = np.column_stack(
        [
            np.cos(bias) * truth[:, 0] - np.sin(bias) * truth[:, 1],
            np.sin(bias) * truth[:, 0] + np.cos(bias) * truth[:, 1],
        ]
    )
    along, cross = los_error_split(rotated - truth, truth)
    np.testing.assert_allclose(cross, 0.01 * np.hypot(*truth.T), rtol=1e-3)
    assert (cross > 0).all()
    assert (np.abs(along) < 0.01 * np.abs(cross)).all()


def test_the_split_is_undefined_at_the_radar_and_checks_its_shapes():
    """Fails if a target at the radar gives a finite split or a bad shape is accepted."""
    along, cross = los_error_split(np.array([[1.0, 1.0]]), np.array([[0.0, 0.0]]))
    assert np.isnan(along).all() and np.isnan(cross).all()
    with pytest.raises(ValueError, match="shape"):
        los_error_split(np.zeros((3, 2)), np.zeros((2, 2)))


# --- scores of a run -------------------------------------------------------------------


def test_the_fields_are_the_phase_6_scores_and_the_diagnostics():
    """Fails if a field is added or dropped without the named tuple following."""
    assert FIELDS[: len(FIELDS) - len(DIAGNOSTIC_FIELDS)][0] == "run_position_rmse"
    assert RobustnessMetrics._fields == FIELDS
    assert len(set(FIELDS)) == len(FIELDS)


def test_the_error_scores_of_a_crafted_run_are_the_known_values():
    """Fails if the split, the rms, the signed mean, the bearing error or the NEES is wrong.

    Track 0 is 10 m across the line of sight of target 0; track 1 is 20 m beyond target 1.
    Their covariance is 10^2 I, so the NEES is 1 and 4.
    """
    metrics = evaluate(make_history([(0.0, 10.0), (0.0, 20.0)]))
    assert metrics.match_fraction_50 == 1.0 and metrics.match_fraction_wide == 1.0
    assert metrics.along_rms == pytest.approx(np.sqrt((0.0 + 400.0) / 2))
    assert metrics.cross_rms == pytest.approx(np.sqrt((100.0 + 0.0) / 2))
    assert metrics.along_mean == pytest.approx(10.0)
    assert metrics.cross_mean == pytest.approx(5.0)
    assert metrics.cross_bearing_rms == pytest.approx(1e3 * np.sqrt((0.01**2 + 0.0) / 2))
    assert metrics.nees_mean == pytest.approx(2.5) and metrics.nees_median == pytest.approx(2.5)
    assert metrics.nees_outlier_fraction == 0.0
    assert metrics.run_position_rmse == pytest.approx(np.sqrt((100.0 + 400.0) / 2))
    assert metrics.run_missed_rate == 0.0 and metrics.in_view_fraction == 1.0


def test_steps_before_the_burn_in_do_not_count():
    """Fails if the diagnostics include the burn-in, where the tracker has not settled."""
    history = make_history([(0.0, 10.0), (0.0, 20.0)])
    bad = [
        TrackSnapshot(s.track_id, s.status, s.x + [0.0, 90.0, 0.0, 0.0], s.P) for s in history[2]
    ]
    history[2] = bad  # inside the burn-in of 5 steps
    clean = evaluate(make_history([(0.0, 10.0), (0.0, 20.0)]))
    dirty = evaluate(history)
    assert tuple(dirty) == pytest.approx(tuple(clean), nan_ok=True)


def test_a_track_pulled_beyond_the_phase_6_distance_still_shows_up_as_an_error():
    """Fails if the error diagnostics use the narrow match distance and lose a biased track.

    Track 0 is 80 m off its target: outside 50 m (a miss and a ghost at the Phase 6 distance)
    but inside the wide match, where its cross-range error is counted.
    """
    metrics = evaluate(make_history([(0.0, 80.0), (0.0, 20.0)]))
    assert metrics.match_fraction_50 == pytest.approx(0.5)
    assert metrics.match_fraction_wide == 1.0
    assert metrics.run_missed_rate == pytest.approx(0.5)
    assert metrics.cross_rms == pytest.approx(np.sqrt(80.0**2 / 2))
    assert metrics.cross_mean == pytest.approx(40.0)
    assert metrics.nees_mean == pytest.approx((64.0 + 4.0) / 2)


def test_a_run_without_matched_tracks_has_undefined_error_scores_but_a_missed_rate():
    """Fails if an empty run gives zeros (a perfect score) instead of NaN for the errors."""
    metrics = evaluate(make_history([], steps=range(0)))
    for name in ("along_rms", "cross_rms", "nees_mean", "nees_median", "cross_bearing_rms"):
        assert np.isnan(getattr(metrics, name)), name
    assert metrics.run_missed_rate == 1.0
    assert metrics.match_fraction_wide == 0.0


def test_nees_outliers_are_counted_against_the_four_dof_quantile():
    """Fails if the outlier threshold is not the chi-square 0.999 quantile of 4 dof (18.47)."""
    history = make_history([(0.0, 0.0), (0.0, 0.0)])
    far = []
    for step in history:
        # 50 m off: NEES = 25, above 18.47, but inside the wide match distance.
        first = step[0]
        moved = TrackSnapshot(first.track_id, first.status, first.x + [0, 50.0, 0, 0], first.P)
        far.append([moved, *step[1:]])
    metrics = evaluate(far, max_distance=60.0)
    assert metrics.nees_outlier_fraction == pytest.approx(0.5)


def test_the_wide_distance_must_not_be_smaller_than_the_phase_6_distance():
    """Fails if the diagnostics could use a narrower match than the scores they explain."""
    with pytest.raises(ValueError, match="wide_distance"):
        evaluate(make_history([(0.0, 0.0), (0.0, 0.0)]), wide_distance=20.0)


def test_in_view_fraction_and_match_fraction_count_only_targets_in_the_field_of_view():
    """Fails if a target outside the field of view counts as missed or as in view."""
    fov = FieldOfView(range_min=50.0, range_max=900.0)  # both targets are at 1000 m: outside
    metrics = evaluate(make_history([(0.0, 0.0), (0.0, 0.0)]), fov=fov)
    assert metrics.in_view_fraction == 0.0
    assert np.isnan(metrics.match_fraction_wide) and np.isnan(metrics.run_missed_rate)
    ids = np.array([[0, -1, -1, -1], [-1, -1, -1, -1]])
    in_view = np.array([[True, True, True, False], [False, False, True, True]])
    after = np.array([False, True, True, True])
    assert match_fraction(ids, in_view, after) == pytest.approx(0.0)
    assert match_fraction(ids, in_view, np.array([True, True, True, True])) == pytest.approx(0.2)


def test_the_closest_approach_of_two_targets_is_found_over_the_whole_run():
    """Fails if the separation is only checked at one step, or defined for a single target."""
    truth = np.zeros((3, 4, 4))
    truth[0, :, 0] = [0, 10, 20, 30]
    truth[1, :, 0] = [100, 50, 24, 100]
    truth[2, :, 1] = 1000.0
    assert min_separation(truth) == pytest.approx(4.0)
    assert np.isnan(min_separation(truth[:1]))


# --- focus window ----------------------------------------------------------------------


def test_the_focus_window_scores_only_its_steps_and_is_nan_without_one():
    """Fails if the window scores cover the whole run, or exist when no window is given."""
    history = make_history([(0.0, 10.0), (0.0, 20.0)])
    plain = evaluate(history)
    for name in ("window_position_rmse", "window_cross_rms", "window_nees_mean"):
        assert np.isnan(getattr(plain, name)), name
    # Inside the window [2 s, 3 s) = steps 20..29, track 0 is 30 m across instead of 10 m.
    for k in range(20, 30):
        history[k][0] = TrackSnapshot(
            0, TrackStatus.CONFIRMED, history[k][0].x + [0.0, 20.0, 0.0, 0.0], history[k][0].P
        )
    metrics = evaluate(history, focus_window=(2.0, 3.0))
    assert metrics.window_cross_rms == pytest.approx(np.sqrt((30.0**2 + 0.0) / 2))
    assert metrics.window_nees_mean == pytest.approx((9.0 + 4.0) / 2)
    assert metrics.window_position_rmse == pytest.approx(np.sqrt((30.0**2 + 20.0**2) / 2))
    assert metrics.window_missed_rate == 0.0 and metrics.window_ghost_rate == 0.0
    assert metrics.cross_rms < metrics.window_cross_rms  # the whole run is calmer


# --- camera accept rates ---------------------------------------------------------------


def scan_with(origins):
    return Scan(np.zeros((len(origins), 1)), np.array(origins, dtype=int))


def crafted_camera_run():
    """Five steps, each a different case; ids: target 0 -> track 0, target 1 -> track 1."""
    log = [
        CameraStepLog((0, 1), (), ((0, 0), (1, 1))),  # A: both take their own measurement
        CameraStepLog((0, 1), (), ((0, 0), (1, 2))),  # B: track 0 takes clutter (row 0)
        CameraStepLog((0, 7), (), ((7, 0),)),  # C: a track matched to no target is updated
        CameraStepLog((), (0, 1), ()),  # D: both left out for overlapping gates
        CameraStepLog((0, 1), (), ()),  # E: offered, no target detected, no update
    ]
    scans = [
        scan_with([0, 1]),
        scan_with([-1, 0, 1]),
        scan_with([-1]),
        scan_with([0, 1]),
        scan_with([-1]),
    ]
    ids = np.array([[0] * 5, [1] * 5])
    return ids, log, scans


def test_camera_rates_of_a_crafted_run_are_the_hand_counted_values():
    """Fails if the accept, skip, true-accept, wrong-update or unattributed rate is miscounted.

    Offered: 2 + 2 + 2 + 0 + 2 = 8, accepted 2 + 2 + 1 = 5, skipped 2. Detected targets with
    a usable track: A 2, B 2 (hits 2 + 1), none in C or E: 3 of 4. Updates of matched tracks:
    4 (A, B), one of them with clutter. One of the 5 updates is by an unmatched track.
    """
    ids, log, scans = crafted_camera_run()
    rates = camera_rates(ids, log, scans, burn_in_steps=0)
    assert rates.accept == pytest.approx(5 / 8)
    assert rates.skip == pytest.approx(2 / 10)
    assert rates.true_accept == pytest.approx(3 / 4)
    assert rates.wrong_update == pytest.approx(1 / 4)
    assert rates.unattributed_update == pytest.approx(1 / 5)


def test_camera_rates_skip_the_burn_in_and_are_undefined_without_data():
    """Fails if the burn-in steps count, or an empty denominator gives a number."""
    ids, log, scans = crafted_camera_run()
    after_a = camera_rates(ids, log, scans, burn_in_steps=1)
    assert after_a.accept == pytest.approx(3 / 6)  # step A (2 accepted of 2) is gone
    nothing = camera_rates(ids, log, scans, burn_in_steps=5)
    assert all(np.isnan(value) for value in nothing)


def test_camera_rates_need_one_log_entry_and_one_scan_per_step():
    """Fails if a log that is shifted against the steps is accepted silently."""
    ids, log, scans = crafted_camera_run()
    with pytest.raises(ValueError, match="one log entry and one scan per step"):
        camera_rates(ids, log[:-1], scans, 0)
    with pytest.raises(ValueError, match="one log entry and one scan per step"):
        camera_rates(ids, log, scans[:-1], 0)
