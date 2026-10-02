"""Gated global assignment of measurements to tracks (Hungarian algorithm)."""

from typing import NamedTuple

import numpy as np
from scipy.optimize import linear_sum_assignment


class Assignment(NamedTuple):
    """Result of a gated assignment.

    Attributes:
        pairs: (row, col) pairs, sorted by row; every pair is inside the gate.
        unassigned_rows: Rows without a partner, ascending.
        unassigned_cols: Columns without a partner, ascending.
    """

    pairs: list[tuple[int, int]]
    unassigned_rows: list[int]
    unassigned_cols: list[int]


def gated_assignment(cost: np.ndarray, in_gate: np.ndarray) -> Assignment:
    """Assign rows to columns at minimum total cost, using only in-gate pairs.

    The Hungarian algorithm needs a finite entry for every pair, so every pair
    outside the gate (inf, NaN, or a finite cost that failed the gate) gets a
    sentinel BIG. The sentinel is larger than any possible sum of in-gate
    costs, so the solver first maximizes the number of in-gate pairs and then
    minimizes their total cost. Whatever pair still lands on a sentinel is
    rolled back: both its row and its column are reported as unassigned.

    In-gate costs are shifted to be non-negative first (a cost such as
    d^2 + ln det S can be negative); a common shift does not change the optimum
    among assignments with the same number of pairs.

    Args:
        cost: Shape (n, m) costs. Only entries where in_gate is True are read.
        in_gate: Shape (n, m) boolean gate decision. An in-gate pair with a
            non-finite cost is treated as out of the gate.

    Returns:
        Assignment with the in-gate pairs and the unassigned rows and columns.

    Raises:
        ValueError: If cost and in_gate are not 2-D arrays of the same shape.
    """
    cost = np.asarray(cost, dtype=float)
    in_gate = np.asarray(in_gate, dtype=bool)
    if cost.ndim != 2 or in_gate.shape != cost.shape:
        raise ValueError(
            f"cost and in_gate must be 2-D with equal shape, got {cost.shape} and {in_gate.shape}"
        )
    n_rows, n_cols = cost.shape
    usable = in_gate & np.isfinite(cost)

    pairs: list[tuple[int, int]] = []
    if usable.any():
        shifted = cost - cost[usable].min()
        n_pairs = min(n_rows, n_cols)
        big = (n_pairs + 1) * shifted[usable].max() + 1.0
        rows, cols = linear_sum_assignment(np.where(usable, shifted, big))
        pairs = [(int(r), int(c)) for r, c in zip(rows, cols, strict=True) if usable[r, c]]

    assigned_rows = {r for r, _ in pairs}
    assigned_cols = {c for _, c in pairs}
    return Assignment(
        sorted(pairs),
        [r for r in range(n_rows) if r not in assigned_rows],
        [c for c in range(n_cols) if c not in assigned_cols],
    )
