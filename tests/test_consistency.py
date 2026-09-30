"""Tests for the NEES computation and its chi-square band."""

import numpy as np
import pytest

from fusion.consistency import (
    nees,
    nees_band,
    nees_by_component,
    seed_mean_nees,
    summarize_nees,
)
from fusion.filters.kalman import KalmanFilter
from fusion.scenario import constant_velocity_trajectory
from fusion.sensors.position import position_measure

DT = 0.1
ACCEL_STD = 0.5
POS_STD = 5.0
VELOCITY_STD = 20.0
N_STEPS = 200
N_SEEDS = 50


def test_nees_hand_computed():
    errors = np.array([[1.0, 1.0], [2.0, 0.0]])
    covariances = np.array([[[2.0, 1.0], [1.0, 2.0]], [[4.0, 0.0], [0.0, 9.0]]])
    # P = [[2, 1], [1, 2]] has inverse [[2, -1], [-1, 2]] / 3, so e = [1, 1] gives 2/3.
    # Diagonal P = diag(4, 9) with e = [2, 0] gives 2^2 / 4 = 1.
    np.testing.assert_allclose(nees(errors, covariances), [2.0 / 3.0, 1.0])


def test_nees_of_identity_covariance_is_squared_norm():
    errors = np.array([[3.0, 4.0, 0.0, 0.0]])
    np.testing.assert_allclose(nees(errors, np.eye(4)[None]), [25.0])


def test_nees_rejects_inconsistent_shapes():
    with pytest.raises(ValueError, match="errors"):
        nees(np.zeros(4), np.eye(4)[None])
    with pytest.raises(ValueError, match="covariances"):
        nees(np.zeros((2, 4)), np.zeros((2, 3, 3)))


def test_nees_by_component_uses_the_matching_blocks():
    truth = np.array([[10.0, 20.0, 3.0, 4.0]])
    estimates = np.array([[9.0, 20.0, 3.0, 2.0]])  # e = [1, 0, 0, 2]
    covariances = np.array(
        [
            [
                [2.0, 0.5, 0.3, 0.0],
                [0.5, 1.0, 0.0, 0.3],
                [0.3, 0.0, 4.0, 1.0],
                [0.0, 0.3, 1.0, 2.0],
            ]
        ]
    )
    result = nees_by_component(truth, estimates, covariances)

    # Position block [[2, .5], [.5, 1]], e = [1, 0]: e^T P^-1 e = P^-1[0, 0] = 1 / (2 - .25).
    np.testing.assert_allclose(result["position"], [1.0 / 1.75])
    # Velocity block [[4, 1], [1, 2]], e = [0, 2]: 4 * P^-1[1, 1] = 4 * 4 / 7.
    np.testing.assert_allclose(result["velocity"], [16.0 / 7.0])
    expected_state = np.array([1.0, 0.0, 0.0, 2.0]) @ np.linalg.inv(covariances[0]) @ [1, 0, 0, 2]
    np.testing.assert_allclose(result["state"], [expected_state])


def test_nees_band_matches_closed_form_for_two_degrees_of_freedom():
    # chi2 with 2 dof is exponential: quantile q = -2 ln(1 - p).
    lower, upper = nees_band(dim=2, n_runs=1)
    assert lower == pytest.approx(-2.0 * np.log(1.0 - 0.025))
    assert upper == pytest.approx(-2.0 * np.log(1.0 - 0.975))


@pytest.mark.parametrize("dim", [2, 4])
def test_nees_band_contains_the_expected_value_and_narrows_with_more_runs(dim):
    widths = []
    for n_runs in (1, 10, 50, 500):
        lower, upper = nees_band(dim, n_runs)
        assert lower < dim < upper
        widths.append(upper - lower)
    assert all(a > b for a, b in zip(widths, widths[1:], strict=False))


def test_nees_band_rejects_invalid_arguments():
    with pytest.raises(ValueError):
        nees_band(dim=0, n_runs=10)
    with pytest.raises(ValueError):
        nees_band(dim=2, n_runs=10, confidence=1.0)


def test_summarize_nees_hand_computed():
    lower, upper = nees_band(dim=2, n_runs=50)
    mean_nees = np.array([100.0, lower - 0.1, 2.0, upper + 0.1, 2.5, 2.0])
    mask = np.array([False, True, True, True, True, True])  # first (huge) step is excluded
    summary = summarize_nees(mean_nees, dim=2, n_runs=50, mask=mask)
    assert summary.lower == pytest.approx(lower)
    assert summary.upper == pytest.approx(upper)
    assert summary.mean == pytest.approx(np.mean(mean_nees[1:]))
    assert summary.inside == pytest.approx(3 / 5)
    assert summary.above == pytest.approx(1 / 5)
    assert summary.below == pytest.approx(1 / 5)
    with pytest.raises(ValueError, match="no steps"):
        summarize_nees(mean_nees, dim=2, n_runs=50, mask=np.zeros(6, dtype=bool))


def linear_kf_nees(seed: int, filter_pos_std: float) -> dict[str, np.ndarray]:
    """NEES of a linear KF on a simulated run whose measurement noise is POS_STD.

    The initial error is drawn from N(0, P0), so a filter with the true noise
    level is consistent from step 0 on and needs no burn-in.
    """
    trajectory_ss, sensor_ss, init_ss = np.random.SeedSequence(seed).spawn(3)
    truth = constant_velocity_trajectory(
        np.array([0.0, 0.0, 15.0, -8.0]),
        dt=DT,
        n_steps=N_STEPS,
        accel_std=ACCEL_STD,
        rng=np.random.default_rng(trajectory_ss),
    )
    measurements = position_measure(truth, POS_STD, rng=np.random.default_rng(sensor_ss))

    P0 = np.diag([POS_STD**2, POS_STD**2, VELOCITY_STD**2, VELOCITY_STD**2])  # noqa: N806
    x0 = truth[0] + np.random.default_rng(init_ss).multivariate_normal(np.zeros(4), P0)
    kf = KalmanFilter(dt=DT, accel_std=ACCEL_STD, pos_std=filter_pos_std, x0=x0, P0=P0)

    estimates = np.zeros((N_STEPS + 1, 4))
    covariances = np.zeros((N_STEPS + 1, 4, 4))
    for k in range(N_STEPS + 1):
        if k > 0:
            kf.predict()
        kf.update(measurements[k])
        estimates[k] = kf.x
        covariances[k] = kf.P
    return nees_by_component(truth, estimates, covariances)


def summaries(filter_pos_std: float):
    trials = [linear_kf_nees(seed, filter_pos_std) for seed in range(N_SEEDS)]
    dims = {"state": 4, "position": 2, "velocity": 2}
    mask = np.ones(N_STEPS + 1, dtype=bool)
    return {
        name: summarize_nees(np.mean([t[name] for t in trials], axis=0), dim, N_SEEDS, mask)
        for name, dim in dims.items()
    }


def test_honest_linear_filter_looks_honest():
    # Observed inside the band: state 98%, position 94%, velocity 92% (about 95% expected).
    for name, summary in summaries(POS_STD).items():
        assert summary.inside > 0.85, name
        assert summary.above < 0.15, name
        assert summary.below < 0.15, name
        assert summary.lower < summary.mean < summary.upper, name


def test_overconfident_filter_is_above_the_band():
    # The filter believes its measurements are 10x more precise than they are.
    # Observed: 100% of the steps above the band, mean NEES about 60x the expected value.
    for name, summary in summaries(POS_STD / 10.0).items():
        assert summary.above > 0.95, name
        assert summary.mean > 5.0 * summary.upper, name


def test_underconfident_filter_is_below_the_band():
    # The filter believes its measurements are 10x noisier than they are (observed ~95% below).
    for name, summary in summaries(POS_STD * 10.0).items():
        assert summary.below > 0.85, name
        assert summary.mean < summary.lower, name


# Student t 97.5% quantile with 2 degrees of freedom (table value 4.303).
T_975_DF2 = 4.302652729911275


def test_seed_mean_nees_hand_computed():
    nees_per_seed = np.array(
        [
            [100.0, 1.0, 3.0],
            [100.0, 2.0, 4.0],
            [100.0, 3.0, 5.0],
        ]
    )
    mask = np.array([False, True, True])  # the huge first step is excluded
    # Per-seed time averages 2, 3, 4: mean 3, sample std 1, standard error 1 / sqrt(3).
    half_width = T_975_DF2 / np.sqrt(3.0)

    result = seed_mean_nees(nees_per_seed, dim=2, mask=mask)
    assert result.mean == pytest.approx(3.0)
    assert result.lower == pytest.approx(3.0 - half_width)
    assert result.upper == pytest.approx(3.0 + half_width)
    assert result.verdict == "consistent"  # 2 is inside [0.52, 5.48]

    # The same interval judged against a larger expected value lies below it.
    assert seed_mean_nees(nees_per_seed, dim=6, mask=mask).verdict == "underconfident"
    # Ten times larger NEES: interval [5.2, 54.8] lies above 4.
    assert seed_mean_nees(10.0 * nees_per_seed, dim=4, mask=mask).verdict == "overconfident"


def test_seed_mean_nees_rejects_invalid_arguments():
    values = np.ones((3, 4))
    mask = np.ones(4, dtype=bool)
    with pytest.raises(ValueError, match="n_seeds"):
        seed_mean_nees(np.ones(4), dim=2, mask=mask)
    with pytest.raises(ValueError, match="at least 2 seeds"):
        seed_mean_nees(np.ones((1, 4)), dim=2, mask=mask)
    with pytest.raises(ValueError, match="mask must have shape"):
        seed_mean_nees(values, dim=2, mask=np.ones(3, dtype=bool))
    with pytest.raises(ValueError, match="no steps"):
        seed_mean_nees(values, dim=2, mask=np.zeros(4, dtype=bool))
    with pytest.raises(ValueError, match="confidence"):
        seed_mean_nees(values, dim=2, mask=mask, confidence=1.0)


def seed_mean_tests(filter_pos_std: float):
    trials = [linear_kf_nees(seed, filter_pos_std) for seed in range(N_SEEDS)]
    dims = {"state": 4, "position": 2, "velocity": 2}
    mask = np.ones(N_STEPS + 1, dtype=bool)
    return {
        name: seed_mean_nees(np.stack([t[name] for t in trials]), dim, mask)
        for name, dim in dims.items()
    }


def test_honest_linear_filter_interval_contains_the_expected_value():
    # Observed: state 4.22 [3.76, 4.67], position 2.16 [1.90, 2.42], velocity 2.17 [1.94, 2.41].
    for (name, result), dim in zip(seed_mean_tests(POS_STD).items(), (4, 2, 2), strict=True):
        assert result.lower < dim < result.upper, name
        assert result.verdict == "consistent", name


def test_overconfident_filter_interval_is_above_the_expected_value():
    # The filter believes its measurements are 10x more precise than they are.
    # Observed lower bounds: state 213, position 154, velocity 64.
    for name, result in seed_mean_tests(POS_STD / 10.0).items():
        assert result.verdict == "overconfident", name


def test_underconfident_filter_interval_is_below_the_expected_value():
    # The filter believes its measurements are 10x noisier than they are.
    # Observed upper bounds: state 2.47, position 0.34, velocity 1.14.
    for name, result in seed_mean_tests(POS_STD * 10.0).items():
        assert result.verdict == "underconfident", name
