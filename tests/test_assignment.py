"""Tests for gated Hungarian assignment."""

import itertools

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from fusion.association.assignment import gated_assignment

INF = np.inf


def assign(cost, threshold):
    cost = np.array(cost, dtype=float)
    return gated_assignment(cost, cost <= threshold)


def best_partial_matching(cost: np.ndarray, in_gate: np.ndarray):
    """Brute force: most in-gate pairs, then lowest total cost."""
    n, m = cost.shape
    best = (0, 0.0)
    for k in range(1, min(n, m) + 1):
        for rows in itertools.combinations(range(n), k):
            for cols in itertools.permutations(range(m), k):
                if all(in_gate[r, c] for r, c in zip(rows, cols, strict=True)):
                    total = sum(cost[r, c] for r, c in zip(rows, cols, strict=True))
                    if k > best[0] or (k == best[0] and total < best[1]):
                        best = (k, total)
    return best


def test_simple_optimal_assignment():
    result = assign([[1.0, 5.0], [6.0, 2.0]], threshold=10.0)
    assert result.pairs == [(0, 0), (1, 1)]
    assert result.unassigned_rows == [] and result.unassigned_cols == []


def test_global_optimum_beats_greedy_nearest_neighbour():
    # Greedy gives row 0 its best column 0 and leaves row 1 with 100 (total 101);
    # the global optimum is (0, 1) + (1, 0) = 3.
    result = assign([[1.0, 2.0], [1.0, 100.0]], threshold=200.0)
    assert result.pairs == [(0, 1), (1, 0)]


def test_forced_out_of_gate_pair_is_rolled_back():
    # The square solver must pair row 1 with column 1, which has no in-gate entry.
    result = assign([[1.0, INF], [INF, INF]], threshold=10.0)
    assert result.pairs == [(0, 0)]
    assert result.unassigned_rows == [1]
    assert result.unassigned_cols == [1]


def test_maximum_number_of_in_gate_pairs_beats_lower_cost():
    # (0, 1) + (1, 0) is cheaper only if the out-of-gate (1, 0) is used; the
    # gated optimum is the two in-gate pairs (0, 0) + (1, 1), total 19.8.
    result = assign([[9.9, 0.1], [INF, 9.9]], threshold=10.0)
    assert result.pairs == [(0, 0), (1, 1)]


def test_a_sentinel_equal_to_the_threshold_would_fail_that_case():
    # Same matrix with the naive sentinel BIG = threshold: the solver prefers
    # 0.1 + 10 over 9.9 + 9.9, and after rollback only one pair is left.
    cost = np.array([[9.9, 0.1], [INF, 9.9]])
    naive = np.where(cost <= 10.0, cost, 10.0)
    rows, cols = linear_sum_assignment(naive)
    survivors = [(r, c) for r, c in zip(rows, cols, strict=True) if cost[r, c] <= 10.0]
    assert len(survivors) == 1
    assert len(gated_assignment(cost, cost <= 10.0).pairs) == 2


def test_finite_out_of_gate_costs_are_not_used():
    # Raw Hungarian would pick (0, 0) + (1, 1) = 5400 and lose a pair on rollback.
    # The best gated outcome is the single cheapest in-gate pair (1, 0) = 2000.
    result = assign([[2400.0, 1e5], [2000.0, 3000.0]], threshold=2500.0)
    assert result.pairs == [(1, 0)]
    assert result.unassigned_rows == [0]
    assert result.unassigned_cols == [1]


def test_nan_entries_do_not_crash_and_are_never_assigned():
    cost = np.array([[np.nan, 1.0], [2.0, np.nan]])
    result = gated_assignment(cost, np.isfinite(cost))
    assert result.pairs == [(0, 1), (1, 0)]
    # An in-gate flag on a non-finite cost is ignored.
    assert gated_assignment(cost, np.ones_like(cost, dtype=bool)).pairs == [(0, 1), (1, 0)]
    assert gated_assignment(np.full((2, 2), np.nan), np.ones((2, 2), dtype=bool)).pairs == []


@pytest.mark.parametrize("shape", [(3, 2), (2, 3), (1, 4), (4, 1)])
def test_rectangular_matrices(shape):
    rng = np.random.default_rng(0)
    cost = rng.uniform(0.0, 5.0, size=shape)
    result = gated_assignment(cost, np.ones(shape, dtype=bool))
    n_pairs = min(shape)
    assert len(result.pairs) == n_pairs
    assert len(result.unassigned_rows) == shape[0] - n_pairs
    assert len(result.unassigned_cols) == shape[1] - n_pairs


@pytest.mark.parametrize("shape", [(0, 3), (2, 0), (0, 0)])
def test_empty_matrices(shape):
    result = gated_assignment(np.zeros(shape), np.zeros(shape, dtype=bool))
    assert result.pairs == []
    assert result.unassigned_rows == list(range(shape[0]))
    assert result.unassigned_cols == list(range(shape[1]))


def test_everything_out_of_gate_is_unassigned():
    result = assign([[50.0, 60.0], [70.0, INF]], threshold=10.0)
    assert result.pairs == []
    assert result.unassigned_rows == [0, 1]
    assert result.unassigned_cols == [0, 1]


def test_cost_exactly_at_the_gate_is_accepted():
    assert assign([[10.0]], threshold=10.0).pairs == [(0, 0)]
    assert assign([[np.nextafter(10.0, 11.0)]], threshold=10.0).pairs == []


def test_negative_costs_are_handled():
    # d^2 + ln det S can be negative. The in-gate maximum-cardinality answer must hold.
    cost = np.array([[-5.0, INF], [-4.0, -3.0]])
    result = gated_assignment(cost, np.isfinite(cost))
    assert result.pairs == [(0, 0), (1, 1)]
    cost = np.array([[-5.0, -1.0], [-2.0, -6.0]])
    assert gated_assignment(cost, np.ones((2, 2), dtype=bool)).pairs == [(0, 0), (1, 1)]


def test_matches_brute_force_on_random_gated_matrices():
    rng = np.random.default_rng(42)
    for _ in range(300):
        n, m = rng.integers(1, 5, size=2)
        cost = rng.uniform(-3.0, 12.0, size=(n, m))
        in_gate = cost <= 9.0
        result = gated_assignment(cost, in_gate)

        expected_pairs, expected_cost = best_partial_matching(cost, in_gate)
        assert len(result.pairs) == expected_pairs
        assert sum(cost[r, c] for r, c in result.pairs) == pytest.approx(expected_cost)
        assert all(in_gate[r, c] for r, c in result.pairs)
        assert len({r for r, _ in result.pairs}) == len(result.pairs)
        assert len({c for _, c in result.pairs}) == len(result.pairs)


def test_shapes_are_validated():
    with pytest.raises(ValueError, match="equal shape"):
        gated_assignment(np.zeros((2, 2)), np.zeros((2, 3), dtype=bool))
    with pytest.raises(ValueError, match="2-D"):
        gated_assignment(np.zeros(3), np.zeros(3, dtype=bool))
