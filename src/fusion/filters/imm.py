"""Interacting multiple model (IMM) filter over linear motion modes with a shared 4-D state.

Every mode is a linear motion model of the state [x, y, vx, vy]: constant velocity (CV) with some
process noise, or a coordinated turn (CT) at a fixed known turn rate. Because all modes share
the state, mixing is the textbook formula and every mode is updated by the same EKF update.

One cycle: `predict` mixes the mode estimates with the mixing probabilities, propagates every
mode, and sets the mode probabilities to the predicted ones; `update` updates every mode with the
measurement and reweights the probabilities by the mode likelihoods in the log domain.

With a single mode the filter is the plain EKF, bit for bit: that case skips mixing and
combination (sums and einsums would turn signed zeros into positive zeros and change the bytes).
"""

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
from scipy.special import logsumexp

from fusion.filters.base import cv_process_noise, cv_transition
from fusion.filters.ekf import ekf_update
from fusion.sensors.base import MeasurementModel


class MotionMode(NamedTuple):
    """One linear motion mode.

    Attributes:
        name: Label of the mode.
        F: Shape (4, 4) state transition matrix over one step.
        Q: Shape (4, 4) process noise covariance over one step.
    """

    name: str
    F: np.ndarray  # noqa: N815 - standard notation
    Q: np.ndarray  # noqa: N815 - standard notation


def cv_mode(dt: float, accel_std: float, name: str = "cv") -> MotionMode:
    """Constant-velocity mode with white-noise acceleration of the given std."""
    return MotionMode(name, cv_transition(dt), cv_process_noise(dt, accel_std))


def ct_transition(dt: float, omega: float) -> np.ndarray:
    """Shape (4, 4) transition of a coordinated turn at the known rate omega (rad/s).

    Positive omega turns counter-clockwise (to the left), as in fusion.maneuvers. The sinc forms
    are finite for omega -> 0, where the matrix is the constant-velocity one.
    """
    theta = omega * dt
    half = np.sin(theta / 2.0)
    sin_over_omega = dt * np.sinc(theta / np.pi)  # sin(theta) / omega
    one_minus_cos_over_omega = dt * half * np.sinc(theta / (2.0 * np.pi))  # (1 - cos) / omega
    cos_theta = 1.0 - 2.0 * half**2
    sin_theta = np.sin(theta)
    return np.array(
        [
            [1.0, 0.0, sin_over_omega, -one_minus_cos_over_omega],
            [0.0, 1.0, one_minus_cos_over_omega, sin_over_omega],
            [0.0, 0.0, cos_theta, -sin_theta],
            [0.0, 0.0, sin_theta, cos_theta],
        ]
    )


def ct_mode(dt: float, omega: float, accel_std: float, name: str | None = None) -> MotionMode:
    """Coordinated-turn mode at a fixed turn rate, with white-noise acceleration.

    The noise matrix is that of the constant-velocity model: the random acceleration is added
    after the rotation, as in the simulator of the true trajectories.
    """
    return MotionMode(
        name if name is not None else f"ct{np.rad2deg(omega):+g}",
        ct_transition(dt, omega),
        cv_process_noise(dt, accel_std),
    )


def uniform_transition(stay_per_scan: float, n_modes: int, steps_per_scan: int) -> np.ndarray:
    """Per-step mode transition matrix from a stay probability given per radar scan.

    The scan matrix has stay_per_scan on the diagonal and equal probabilities elsewhere. Such a
    matrix has the eigenvalue 1 and the eigenvalue lam = (M p - 1) / (M - 1) (M - 1 times), so
    its n-th root has the same form with lam_s = lam^(1/n); the step matrix raised to
    steps_per_scan reproduces the scan matrix.

    Args:
        stay_per_scan: Probability of keeping the mode over one radar scan, in (1/M, 1].
        n_modes: Number of modes M.
        steps_per_scan: Filter steps per radar scan n (>= 1).

    Returns:
        Shape (M, M) row-stochastic matrix; [[1.0]] for a single mode.

    Raises:
        ValueError: If n_modes < 1, steps_per_scan < 1 or the stay probability is out of range.
    """
    if n_modes < 1 or steps_per_scan < 1:
        raise ValueError(
            f"n_modes and steps_per_scan must be >= 1, got {n_modes} and {steps_per_scan}"
        )
    if n_modes == 1:
        return np.ones((1, 1))
    if not 1.0 / n_modes < stay_per_scan <= 1.0:
        raise ValueError(
            f"stay_per_scan must be in (1/{n_modes}, 1], got {stay_per_scan}"
        )
    lam = (n_modes * stay_per_scan - 1.0) / (n_modes - 1.0)
    lam_step = lam ** (1.0 / steps_per_scan)
    stay = (1.0 + (n_modes - 1.0) * lam_step) / n_modes
    off = (1.0 - stay) / (n_modes - 1.0)
    return off * np.ones((n_modes, n_modes)) + (stay - off) * np.eye(n_modes)


def mix_modes(
    transition: np.ndarray,
    mu: np.ndarray,
    x_modes: np.ndarray,
    P_modes: np.ndarray,  # noqa: N803 - standard notation
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """IMM mixing step.

    With c_j = sum_i pi_ij mu_i and mixing weights w_ij = pi_ij mu_i / c_j, mode j starts from
        x0_j = sum_i w_ij x_i,   P0_j = sum_i w_ij (P_i + (x_i - x0_j)(x_i - x0_j)^T).
    A mode with c_j = 0 (only possible with a transition matrix that has no floor) mixes with
    the weights mu.

    Args:
        transition: Shape (M, M) matrix, row i = transitions out of mode i.
        mu: Shape (M,) mode probabilities.
        x_modes: Shape (M, 4) mode states.
        P_modes: Shape (M, 4, 4) mode covariances.

    Returns:
        (c, x0, P0): predicted mode probabilities (M,), mixed states (M, 4), covariances (M, 4, 4).
    """
    c = transition.T @ mu
    weights = np.where(c > 0.0, transition * mu[:, None] / np.where(c > 0.0, c, 1.0), mu[:, None])
    x0 = weights.T @ x_modes
    diff = x_modes[:, None, :] - x0[None, :, :]  # (i, j, 4)
    spread = np.einsum("ij,ija,ijb->jab", weights, diff, diff)
    P0 = np.einsum("ij,iab->jab", weights, P_modes) + spread  # noqa: N806
    return c, x0, P0


def combine_modes(
    mu: np.ndarray,
    x_modes: np.ndarray,
    P_modes: np.ndarray,  # noqa: N803 - standard notation
) -> tuple[np.ndarray, np.ndarray, float]:
    """Moment-matched combination of the modes.

    x = sum mu_j x_j,  P = sum mu_j (P_j + (x_j - x)(x_j - x)^T). The spread term is exactly
    symmetric and positive semi-definite; P is not symmetrized further.

    Returns:
        (x, P, spread_trace): combined state (4,), covariance (4, 4) and the trace of the
        spread-of-the-means part of P.
    """
    x = mu @ x_modes
    diff = x_modes - x[None, :]
    spread = np.einsum("j,ja,jb->ab", mu, diff, diff)
    P = np.einsum("j,jab->ab", mu, P_modes) + spread  # noqa: N806
    return x, P, float(np.trace(spread))


def mode_log_likelihood(residual: np.ndarray, S: np.ndarray) -> float:  # noqa: N803
    """Gaussian log-likelihood of an innovation: -(d^2 + ln det S + m ln 2 pi) / 2.

    S is symmetrized and factorized by Cholesky. Returns -inf if S is not positive definite or
    the result is not finite.
    """
    sym = 0.5 * (S + S.T)
    try:
        chol = np.linalg.cholesky(sym)
    except np.linalg.LinAlgError:
        return float("-inf")
    y = np.linalg.solve(chol, residual)
    value = -0.5 * (
        float(y @ y)
        + 2.0 * float(np.sum(np.log(np.diag(chol))))
        + len(residual) * np.log(2 * np.pi)
    )
    return value if np.isfinite(value) else float("-inf")


def update_mode_probabilities(
    predicted: np.ndarray, log_likelihood: np.ndarray
) -> tuple[np.ndarray, bool]:
    """Posterior mode probabilities mu_j ~ c_j L_j, computed in the log domain.

    Args:
        predicted: Shape (M,) predicted probabilities c.
        log_likelihood: Shape (M,) log-likelihoods; NaN counts as -inf.

    Returns:
        (mu, ok). If no mode has a finite posterior weight (all likelihoods -inf), mu is the
        predicted probabilities and ok is False.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        log_post = np.log(predicted) + np.where(np.isnan(log_likelihood), -np.inf, log_likelihood)
        total = logsumexp(log_post)
    if not np.isfinite(total):
        return predicted.copy(), False
    mu = np.exp(log_post - total)
    return mu / mu.sum(), True


class IMMFilter:
    """IMM filter with the interface the tracker needs from a track filter.

    Attributes:
        x: Shape (4,) combined state estimate.
        P: Shape (4, 4) combined covariance.
        mu: Shape (M,) mode probabilities (a copy).
        likelihood_failures: Updates in which no mode had a usable likelihood (mu kept).
        spread_trace: Trace of the spread-of-the-means part of the combined covariance.
    """

    def __init__(
        self,
        modes: Sequence[MotionMode],
        transition: np.ndarray,
        x0: np.ndarray,
        P0: np.ndarray,  # noqa: N803 - standard notation
        mu0: np.ndarray | None = None,
    ) -> None:
        """Start every mode from the same initial estimate.

        Args:
            modes: The M motion modes.
            transition: Shape (M, M) per-step transition matrix (see uniform_transition).
            x0: Shape (4,) initial state.
            P0: Shape (4, 4) initial covariance.
            mu0: Shape (M,) initial mode probabilities; uniform by default.

        Raises:
            ValueError: If a shape is wrong, or transition / mu0 are not probabilities.
        """
        self._modes = tuple(modes)
        n = len(self._modes)
        if n < 1:
            raise ValueError("an IMM filter needs at least one mode")
        x0 = np.array(x0, dtype=float)
        P0 = np.array(P0, dtype=float)  # noqa: N806
        if x0.shape != (4,):
            raise ValueError(f"x0 must have shape (4,), got {x0.shape}")
        if P0.shape != (4, 4):
            raise ValueError(f"P0 must have shape (4, 4), got {P0.shape}")
        self._pi = np.array(transition, dtype=float)
        if self._pi.shape != (n, n) or (self._pi < 0.0).any():
            raise ValueError(f"transition must be a non-negative ({n}, {n}) matrix")
        if not np.allclose(self._pi.sum(axis=1), 1.0, atol=1e-9):
            raise ValueError("every row of the transition matrix must sum to 1")
        mu = np.full(n, 1.0 / n) if mu0 is None else np.array(mu0, dtype=float)
        if mu.shape != (n,) or (mu < 0.0).any() or not np.isclose(mu.sum(), 1.0, atol=1e-9):
            raise ValueError(f"mu0 must be {n} probabilities that sum to 1")
        self._mu = mu
        self._x_modes = np.array([x0 for _ in range(n)])
        self._P_modes = np.array([P0 for _ in range(n)])  # noqa: N806
        self.likelihood_failures = 0
        self._combine()

    @property
    def x(self) -> np.ndarray:
        """Combined state estimate, shape (4,)."""
        return self._x

    @property
    def P(self) -> np.ndarray:  # noqa: N802 - standard notation
        """Combined covariance, shape (4, 4)."""
        return self._P

    @property
    def mu(self) -> np.ndarray:
        """Mode probabilities, shape (M,) (a copy)."""
        return self._mu.copy()

    @property
    def modes(self) -> tuple[MotionMode, ...]:
        """The motion modes."""
        return self._modes

    @property
    def x_modes(self) -> np.ndarray:
        """Per-mode states, shape (M, 4) (a copy)."""
        return self._x_modes.copy()

    @property
    def P_modes(self) -> np.ndarray:  # noqa: N802 - standard notation
        """Per-mode covariances, shape (M, 4, 4) (a copy)."""
        return self._P_modes.copy()

    def _combine(self) -> None:
        if len(self._modes) == 1:
            self._x, self._P, self.spread_trace = self._x_modes[0], self._P_modes[0], 0.0
        else:
            self._x, self._P, self.spread_trace = combine_modes(
                self._mu, self._x_modes, self._P_modes
            )

    def predict(self) -> None:
        """Mix the modes, propagate each one step and set mu to the predicted probabilities."""
        if len(self._modes) == 1:
            mode = self._modes[0]
            self._x_modes[0] = mode.F @ self._x_modes[0]
            self._P_modes[0] = mode.F @ self._P_modes[0] @ mode.F.T + mode.Q
        else:
            c, x0, P0 = mix_modes(self._pi, self._mu, self._x_modes, self._P_modes)  # noqa: N806
            for j, mode in enumerate(self._modes):
                self._x_modes[j] = mode.F @ x0[j]
                self._P_modes[j] = mode.F @ P0[j] @ mode.F.T + mode.Q
            self._mu = c
        self._combine()

    def update(self, z: np.ndarray, model: MeasurementModel) -> None:
        """Update every mode with a measurement and reweight the mode probabilities.

        Raises:
            ValueError: If z has the wrong shape or the model cannot be linearized at the state
                of some mode. Nothing is changed in that case.
        """
        pairs = zip(self._x_modes, self._P_modes, strict=True)
        results = [ekf_update(x, cov, z, model) for x, cov in pairs]
        if len(self._modes) == 1:
            self._x_modes[0], self._P_modes[0] = results[0].x, results[0].P
        else:
            log_likelihood = np.array(
                [mode_log_likelihood(r.residual, r.S) for r in results]
            )
            mu, ok = update_mode_probabilities(self._mu, log_likelihood)
            if not ok:
                self.likelihood_failures += 1
            for j, r in enumerate(results):
                self._x_modes[j], self._P_modes[j] = r.x, r.P
            self._mu = mu
        self._combine()
