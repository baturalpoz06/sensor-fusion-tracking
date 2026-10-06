"""Oracle tests of the global camera-bias estimator on synthetic radar-camera pairs.

Each test docstring names the defect that makes it fail.
"""

import math
import pickle

import numpy as np
import pytest

from fusion.tracker.camera_bias import CameraBiasConfig, CameraBiasEstimator

DEG = float(np.pi / 180.0)
RADAR_STD = 2.0 * DEG
CAMERA_STD = 0.1 * DEG
PAIR_VAR = RADAR_STD**2 + CAMERA_STD**2
SLAB = 1.0 * DEG


def make(mode="spike_slab", **kwargs) -> CameraBiasEstimator:
    config = CameraBiasConfig(mode=mode, process_noise=0.0, **kwargs)
    return CameraBiasEstimator(config, RADAR_STD, CAMERA_STD)


def pairs(bias_deg: float, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return bias_deg * DEG + rng.normal(0.0, math.sqrt(PAIR_VAR), n)


def log_normal_pdf(x: float, variance: float) -> float:
    return -0.5 * (x * x / variance + math.log(2 * math.pi * variance))


def test_a_known_bias_is_recovered_with_the_closed_form_posterior_std():
    """Fails if the estimate is biased, or its std differs from 1/sqrt(1/s0^2 + n/sd^2)."""
    n = 2000
    est = make("always")
    est.update(list(pairs(0.5, n)))
    b, var = est.estimate
    # The 99% gate (5.8 deg on a 2 deg noise) legitimately rejects a few tails: the closed form
    # counts the pairs that were used.
    assert est.used + est.rejected == n and est.rejected < 0.02 * n
    sigma = 1.0 / math.sqrt(1.0 / SLAB**2 + est.used / PAIR_VAR)
    assert var == pytest.approx(sigma**2, rel=1e-9)
    assert abs(b - 0.5 * DEG) < 3.0 * sigma


def test_the_spike_and_slab_posterior_goes_to_one_for_a_real_bias():
    """Fails if H1 is not preferred over H0 when a bias of 0.5 deg is present."""
    est = make("spike_slab", prior_h1=0.01)
    est.update(list(pairs(0.5, 1500)))
    assert est.p1 > 0.99
    b, _ = est.estimate
    assert abs(b - 0.5 * DEG) < 0.1 * DEG


def test_the_posterior_equals_the_closed_form_bayes_factor_without_drift():
    """Fails if the sequential likelihoods are not those of the batch marginals.

    With q = 0 the sample mean is sufficient: BF = N(ybar; 0, s0^2 + sd^2/n) / N(ybar; 0,
    sd^2/n), and the log posterior odds are the log prior odds plus log BF.
    """
    n = 60
    values = pairs(0.3, n, seed=4)
    est = make("spike_slab", prior_h1=0.2)
    est.update(list(values))
    ybar = float(values.mean())
    log_bf = log_normal_pdf(ybar, SLAB**2 + PAIR_VAR / n) - log_normal_pdf(ybar, PAIR_VAR / n)
    expected = math.log(0.2 / 0.8) + log_bf
    assert est._log_odds == pytest.approx(expected, abs=1e-8)
    assert est.p1 == pytest.approx(1.0 / (1.0 + math.exp(-expected)), rel=1e-9)


def test_without_bias_and_without_noise_the_applied_estimate_is_exactly_zero():
    """Fails if a zero difference moves the applied bias (no bias, no drift)."""
    for mode in ("spike_slab", "always"):
        est = make(mode)
        est.update([0.0] * 500)
        assert est.estimate[0] == 0.0
        assert est.rejected == 0


def test_noisy_pairs_without_bias_keep_the_estimate_small_and_h1_unlikely():
    """Fails if noise alone is read as bias: |b1| within 3 sigma, mean P1 below the prior."""
    n = 146
    sigma = 1.0 / math.sqrt(1.0 / SLAB**2 + n / PAIR_VAR)
    p1s = []
    for seed in range(40):
        est = make("spike_slab", prior_h1=0.5)
        est.update(list(pairs(0.0, n, seed)))
        p1s.append(est.p1)
        assert abs(est._b1) < 4.0 * sigma
        assert abs(est.estimate[0]) <= abs(est._b1)
    assert float(np.mean(p1s)) < 0.5


def test_the_outlier_gate_rejects_a_wild_pair_and_leaves_the_estimate_unchanged():
    """Fails if a 20 deg pair is used, or the gate is centered on the current estimate."""
    est = make("spike_slab")
    est.update(list(pairs(0.5, 50)))
    before = (est._b1, est._var1, est._log_odds, est.used)
    est.update([20.0 * DEG])
    assert est.rejected == 1
    assert (est._b1, est._var1, est._log_odds, est.used) == before
    est.update([-20.0 * DEG])
    assert est.rejected == 2


def test_the_gate_is_centered_on_zero_so_a_correct_pair_is_not_lost_with_a_wrong_estimate():
    """Fails if the gate follows b_hat: a pair at the true bias must pass a stale estimate."""
    est = make("always")
    est._b1 = -4.0 * DEG  # a badly wrong estimate
    assert est.gate_half_width > 5.0 * DEG
    est.update([0.5 * DEG])
    assert est.rejected == 0 and est.used == 1


def test_a_difference_across_the_branch_cut_is_wrapped_before_it_is_used():
    """Fails if a bearing difference of 2 pi - small is treated as a huge outlier."""
    est = make("always")
    est.update([2.0 * math.pi - 0.001])
    assert est.rejected == 0 and est.used == 1
    assert est._b1 == pytest.approx(-0.001 * (SLAB**2 / (SLAB**2 + PAIR_VAR)), rel=1e-9)


def test_the_mixture_mean_and_variance_follow_the_posterior_probability():
    """Fails if the applied estimate is b1 or ignores the spike: hand numbers for P1 = 0.3."""
    est = make("spike_slab")
    est._log_odds = math.log(0.3 / 0.7)
    est._b1, est._var1 = 0.01, 1e-4
    b, var = est.estimate
    assert b == pytest.approx(0.3 * 0.01, rel=1e-12)
    assert var == pytest.approx(0.3 * 1e-4 + 0.3 * 0.7 * 1e-4, rel=1e-12)


def test_extreme_log_odds_do_not_overflow():
    """Fails if the posterior is computed as exp(l) / (1 + exp(l)) and overflows."""
    est = make("spike_slab")
    for odds, p1 in ((900.0, 1.0), (-900.0, 0.0)):
        est._log_odds = odds
        assert est.p1 == p1
        assert np.isfinite(est.estimate).all()


def test_the_random_walk_widens_the_h1_variance_and_a_zero_walk_does_not():
    """Fails if predict does not add q * dt, or adds it when q = 0."""
    walking = CameraBiasEstimator(CameraBiasConfig(process_noise=1e-8), RADAR_STD, CAMERA_STD)
    still = make("spike_slab")
    for est in (walking, still):
        est.update(list(pairs(0.0, 100)))
    var_before = walking._var1
    walking.predict(10.0)
    still_before = still._var1
    still.predict(10.0)
    assert walking._var1 == pytest.approx(var_before + 1e-7, rel=1e-12)
    assert still._var1 == still_before


def test_the_oracle_applies_the_true_bias_with_no_variance_and_ignores_the_data():
    """Fails if the oracle arm learns from pairs or reports a variance."""
    est = CameraBiasEstimator(
        CameraBiasConfig(mode="oracle", oracle_bias=0.5 * DEG), RADAR_STD, CAMERA_STD
    )
    est.update(list(pairs(2.0, 100)))
    est.predict(5.0)
    assert est.estimate == (0.5 * DEG, 0.0)
    assert est.used == 0


def test_the_log_entry_reports_the_state():
    """Fails if the log entry is not the applied estimate and counters."""
    est = make("always")
    est.update([0.001, 20.0 * DEG])
    entry = est.log_entry()
    assert entry.used == 1 and entry.rejected == 1 and entry.p1 == 1.0
    assert entry.b_app == est.estimate[0] and entry.sd_app == pytest.approx(math.sqrt(est._var1))


def test_the_default_noise_is_a_tenth_of_a_degree_per_ten_minutes():
    """Fails if the default process noise is not (0.1 deg)^2 per 600 s, in radians."""
    assert CameraBiasConfig().process_noise == pytest.approx((0.1 * DEG) ** 2 / 600.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "bayes"},
        {"prior_h1": 0.0},
        {"prior_h1": 1.0},
        {"sigma_slab": 0.0},
        {"process_noise": -1.0},
        {"gate_probability": 1.0},
        {"oracle_bias": float("nan")},
    ],
)
def test_invalid_configurations_are_rejected(kwargs):
    """Fails if a bad parameter is accepted."""
    with pytest.raises(ValueError):
        CameraBiasConfig(**kwargs)


def test_the_configuration_is_hashable_and_picklable():
    """Fails if a process pool could not receive the configuration."""
    config = CameraBiasConfig(prior_h1=0.1)
    assert pickle.loads(pickle.dumps(config)) == config
    assert hash(config) == hash(pickle.loads(pickle.dumps(config)))
