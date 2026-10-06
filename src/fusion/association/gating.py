"""Measurement gating: innovations, Mahalanobis distance and gated cost matrices."""

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
from scipy.stats import chi2

from fusion.angles import wrap_angle
from fusion.filters.base import TrackFilter
from fusion.sensors.base import MIN_RANGE, MeasurementModel


def gate_threshold(dof: int, probability: float) -> float:
    """Chi-square threshold on the squared Mahalanobis distance.

    A measurement of a correctly modelled target falls inside the gate with
    the given probability. For 99%: 9.21 with 2 degrees of freedom (radar),
    6.63 with 1 (camera).

    Args:
        dof: Measurement dimension (degrees of freedom).
        probability: Gate probability, strictly between 0 and 1.

    Raises:
        ValueError: If dof < 1 or the probability is not in (0, 1).
    """
    if dof < 1:
        raise ValueError(f"dof must be >= 1, got {dof}")
    if not 0.0 < probability < 1.0:
        raise ValueError(f"probability must be in (0, 1), got {probability}")
    return float(chi2.ppf(probability, df=dof))


class Innovation(NamedTuple):
    """Measurement residual and its covariance.

    Attributes:
        residual: Shape (m,) innovation z - h(x), angle elements wrapped.
        covariance: Shape (m, m) innovation covariance S = H P H^T + R.
    """

    residual: np.ndarray
    covariance: np.ndarray


def innovation_covariance(
    x: np.ndarray,
    P: np.ndarray,  # noqa: N803 - standard notation
    model: MeasurementModel,
) -> np.ndarray:
    """Innovation covariance S = H P H^T + R, symmetrized as (S + S^T) / 2.

    Raises:
        ValueError: If the model cannot be linearized at x (target at the sensor).
    """
    H = model.jacobian(x)  # noqa: N806 - standard notation
    S = H @ P @ H.T + model.R  # noqa: N806 - standard notation
    return 0.5 * (S + S.T)


def innovation(
    x: np.ndarray,
    P: np.ndarray,  # noqa: N803 - standard notation
    z: np.ndarray,
    model: MeasurementModel,
) -> Innovation:
    """Residual and covariance of measurement z against the estimate (x, P).

    Computed exactly as ExtendedKalmanFilter.update does: angle elements of the
    residual are wrapped to [-pi, pi] before anything else uses them, so a
    bearing on either side of the -x axis is not mistaken for a 2*pi jump.

    Raises:
        ValueError: If z has the wrong shape, or the model cannot be linearized at x.
    """
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    m = model.R.shape[0]
    if z.shape != (m,):
        raise ValueError(f"z must have shape ({m},), got {z.shape}")
    S = innovation_covariance(x, np.asarray(P, dtype=float), model)  # noqa: N806
    residual = z - model.h(x)
    angles = list(model.angle_indices)
    residual[angles] = wrap_angle(residual[angles])
    return Innovation(residual, S)


def mahalanobis_squared(inn: Innovation) -> float:
    """Squared Mahalanobis distance nu^T S^-1 nu, by Cholesky solve.

    Returns:
        The distance, or inf if S is not positive definite or the result is not finite.
    """
    try:
        chol = np.linalg.cholesky(inn.covariance)
    except np.linalg.LinAlgError:
        return float("inf")
    y = np.linalg.solve(chol, inn.residual)
    d2 = float(y @ y)
    return d2 if np.isfinite(d2) else float("inf")


class GatedCosts(NamedTuple):
    """Association costs of every track-measurement pair.

    Attributes:
        cost: Shape (n_tracks, n_meas) cost d^2 + ln det S. Inf, or NaN for a
            non-finite estimate, for a track that could not be gated (too close to the
            sensor, S not positive definite); never in the gate then.
        in_gate: Shape (n_tracks, n_meas) boolean, True if d^2 <= threshold.
    """

    cost: np.ndarray
    in_gate: np.ndarray


def gated_costs(
    filters: Sequence[TrackFilter],
    measurements: np.ndarray,
    model: MeasurementModel,
    threshold: float,
    min_range: float,
) -> GatedCosts:
    """Gate and score every track against every measurement of one sensor.

    The gate is on the squared Mahalanobis distance d^2 (chi-square threshold).
    The cost is d^2 + ln det S, the standard likelihood-based association cost:
    with d^2 alone a track with a wide innovation covariance would be favored
    for every measurement. Callers must assign with `in_gate`, not with the cost.

    H, S and its Cholesky factor do not depend on the measurement, so they are
    computed once per track; residuals and distances are vectorized over the
    measurements. Filters are only read, never modified.

    A track whose predicted range is below max(min_range, MIN_RANGE) is out of
    the gate for every measurement: close to the sensor the bearing Jacobian
    explodes, S becomes huge and the gate would accept everything.

    Args:
        filters: Predicted track filters, each with x (4,) and P (4, 4).
        measurements: Shape (n_meas, m) measurements of the sensor model.
        model: Measurement model of the sensor.
        threshold: Gate on d^2, e.g. gate_threshold(m, 0.99).
        min_range: Minimum predicted range in meters at which a track is gated.

    Returns:
        GatedCosts with (n_tracks, n_meas) arrays. A NaN measurement is never in the gate.

    Raises:
        ValueError: If measurements does not have shape (n_meas, m).
    """
    z = np.asarray(measurements, dtype=float)
    m = model.R.shape[0]
    if z.ndim != 2 or z.shape[1] != m:
        raise ValueError(f"measurements must have shape (n, {m}), got {z.shape}")

    n_tracks, n_meas = len(filters), z.shape[0]
    cost = np.full((n_tracks, n_meas), np.inf)
    in_gate = np.zeros((n_tracks, n_meas), dtype=bool)
    guard = max(min_range, MIN_RANGE)
    angles = list(model.angle_indices)

    for i, track_filter in enumerate(filters):
        x = track_filter.x
        if n_meas == 0 or np.hypot(x[0], x[1]) < guard:
            continue
        S = innovation_covariance(x, track_filter.P, model)  # noqa: N806
        try:
            chol = np.linalg.cholesky(S)
        except np.linalg.LinAlgError:
            continue
        residuals = z - model.h(x)
        residuals[:, angles] = wrap_angle(residuals[:, angles])
        y = np.linalg.solve(chol, residuals.T)
        d2 = np.einsum("ij,ij->j", y, y)
        cost[i] = d2 + 2.0 * np.sum(np.log(np.diag(chol)))
        in_gate[i] = d2 <= threshold
    return GatedCosts(cost, in_gate)
