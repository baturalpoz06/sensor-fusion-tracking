"""Filter consistency: NEES and its chi-square acceptance band."""

from typing import NamedTuple

import numpy as np
from scipy.stats import chi2
from scipy.stats import t as student_t

# State components scored separately: name -> slice of [x, y, vx, vy].
COMPONENTS = {
    "state": slice(0, 4),
    "position": slice(0, 2),
    "velocity": slice(2, 4),
}


def nees(errors: np.ndarray, covariances: np.ndarray) -> np.ndarray:
    """Normalized estimation error squared, e^T P^-1 e, at every step.

    Args:
        errors: Shape (n, d) estimation errors (truth - estimate).
        covariances: Shape (n, d, d) filter covariances at the same steps.

    Returns:
        Shape (n,) NEES values. If the filter is consistent, each has mean d.

    Raises:
        ValueError: On inconsistent shapes.
    """
    errors = np.asarray(errors, dtype=float)
    covariances = np.asarray(covariances, dtype=float)
    if errors.ndim != 2:
        raise ValueError(f"errors must have shape (n, d), got {errors.shape}")
    n, d = errors.shape
    if covariances.shape != (n, d, d):
        raise ValueError(f"covariances must have shape ({n}, {d}, {d}), got {covariances.shape}")
    # Solve P y = e instead of forming P^-1.
    solved = np.linalg.solve(covariances, errors[..., None])[..., 0]
    return np.einsum("nd,nd->n", errors, solved)


def nees_by_component(
    truth: np.ndarray, estimates: np.ndarray, covariances: np.ndarray
) -> dict[str, np.ndarray]:
    """NEES of the whole state, the position only and the velocity only.

    The partial variants use the matching diagonal blocks of P: the top-left
    2x2 block for position, the bottom-right 2x2 block for velocity.

    Args:
        truth: Shape (n, 4) true states.
        estimates: Shape (n, 4) estimates.
        covariances: Shape (n, 4, 4) covariances.

    Returns:
        {"state": (n,), "position": (n,), "velocity": (n,)} NEES arrays.
    """
    errors = np.asarray(truth, dtype=float) - np.asarray(estimates, dtype=float)
    covariances = np.asarray(covariances, dtype=float)
    return {
        name: nees(errors[:, part], covariances[:, part, part])
        for name, part in COMPONENTS.items()
    }


def nees_band(dim: int, n_runs: int, confidence: float = 0.95) -> tuple[float, float]:
    """Two-sided chi-square band for the mean NEES of n_runs independent runs.

    The sum of n_runs independent NEES values of a consistent filter is
    chi-square distributed with n_runs * dim degrees of freedom, so their mean
    lies in [q_low, q_high] / n_runs with probability `confidence`.

    Args:
        dim: Dimension of the error vector (4 for the state, 2 for position or velocity).
        n_runs: Number of independent runs averaged at each step.
        confidence: Probability mass of the band.

    Returns:
        (lower, upper) bounds on the seed-averaged NEES at a single step.
    """
    if dim < 1 or n_runs < 1:
        raise ValueError(f"dim and n_runs must be >= 1, got {dim} and {n_runs}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    tail = (1.0 - confidence) / 2.0
    lower, upper = chi2.ppf([tail, 1.0 - tail], df=n_runs * dim)
    return float(lower / n_runs), float(upper / n_runs)


class NeesSummary(NamedTuple):
    """Summary of the seed-averaged NEES over the selected steps.

    Attributes:
        mean: Mean over the selected steps of the seed-averaged NEES. Steps are
            correlated in time, so it has no chi-square band; seed_mean_nees
            tests it with a confidence interval across seeds instead.
        lower: Lower band bound for one step.
        upper: Upper band bound for one step.
        inside: Fraction of selected steps within [lower, upper].
        above: Fraction of selected steps above the band.
        below: Fraction of selected steps below the band.
    """

    mean: float
    lower: float
    upper: float
    inside: float
    above: float
    below: float


def summarize_nees(
    mean_nees: np.ndarray,
    dim: int,
    n_runs: int,
    mask: np.ndarray,
    confidence: float = 0.95,
) -> NeesSummary:
    """Compare the step-wise seed-averaged NEES with its chi-square band.

    For a consistent filter about (1 - confidence) of the steps are expected
    outside the band by chance alone.

    Args:
        mean_nees: Shape (n,) NEES averaged over seeds at each step.
        dim: Dimension of the error vector.
        n_runs: Number of seeds that were averaged.
        mask: Shape (n,) boolean selection of steps; must select at least one.
        confidence: Probability mass of the band.
    """
    selected = np.asarray(mean_nees, dtype=float)[np.asarray(mask, dtype=bool)]
    if selected.size == 0:
        raise ValueError("mask selects no steps")
    lower, upper = nees_band(dim, n_runs, confidence)
    above = float(np.mean(selected > upper))
    below = float(np.mean(selected < lower))
    return NeesSummary(float(selected.mean()), lower, upper, 1.0 - above - below, above, below)


class SeedMeanNees(NamedTuple):
    """Confidence interval for the expected time-averaged NEES, built across seeds.

    Attributes:
        mean: Mean over seeds of each seed's NEES averaged over the selected steps.
        lower: Lower bound of the confidence interval for that mean.
        upper: Upper bound of the confidence interval for that mean.
        verdict: "consistent" if the interval contains dim, "overconfident" if it
            lies above dim, "underconfident" if it lies below.
    """

    mean: float
    lower: float
    upper: float
    verdict: str


def seed_mean_nees(
    nees_per_seed: np.ndarray,
    dim: int,
    mask: np.ndarray,
    confidence: float = 0.95,
) -> SeedMeanNees:
    """Test the mean NEES against dim with a statistic that is independent across seeds.

    Steps of one run are correlated in time, so step-wise band counts are not
    independent events. Seeds are: each seed's NEES is first averaged over the
    selected steps, and the interval is built from the spread of those averages
    across seeds. A consistent filter has expected NEES dim at every step, so
    its time average also has expectation dim.

    The interval uses the Student t distribution with n_seeds - 1 degrees of
    freedom: the standard error comes from the same seeds, and t accounts for
    that estimate. For hundreds of seeds it equals the normal interval to three
    digits; for a few dozen it is slightly wider and so avoids false alarms.

    Args:
        nees_per_seed: Shape (n_seeds, n) NEES of every seed at every step.
        dim: Dimension of the error vector (the expected NEES of a consistent filter).
        mask: Shape (n,) boolean selection of steps; must select at least one.
        confidence: Coverage of the interval.

    Returns:
        SeedMeanNees with the mean, the interval and the verdict.

    Raises:
        ValueError: On inconsistent shapes, fewer than 2 seeds, an empty mask or
            an invalid confidence.
    """
    values = np.asarray(nees_per_seed, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if values.ndim != 2:
        raise ValueError(f"nees_per_seed must have shape (n_seeds, n), got {values.shape}")
    n_seeds, n = values.shape
    if n_seeds < 2:
        raise ValueError(f"need at least 2 seeds for a confidence interval, got {n_seeds}")
    if mask.shape != (n,):
        raise ValueError(f"mask must have shape ({n},), got {mask.shape}")
    if not np.any(mask):
        raise ValueError("mask selects no steps")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    per_seed = values[:, mask].mean(axis=1)
    mean = float(per_seed.mean())
    standard_error = float(per_seed.std(ddof=1) / np.sqrt(n_seeds))
    half_width = float(student_t.ppf(0.5 + confidence / 2.0, df=n_seeds - 1)) * standard_error
    lower, upper = mean - half_width, mean + half_width

    if lower > dim:
        verdict = "overconfident"
    elif upper < dim:
        verdict = "underconfident"
    else:
        verdict = "consistent"
    return SeedMeanNees(mean, lower, upper, verdict)
