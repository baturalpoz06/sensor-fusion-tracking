"""Extended Kalman filter with a 2D constant-velocity motion model."""

from typing import NamedTuple

import numpy as np

from fusion.angles import wrap_angle
from fusion.filters.base import CVFilterBase
from fusion.sensors.base import MeasurementModel


class EkfStep(NamedTuple):
    """Result of one EKF measurement update.

    Attributes:
        x: Shape (4,) corrected state.
        P: Shape (4, 4) corrected covariance (Joseph form).
        residual: Shape (m,) innovation z - h(x) with angle elements wrapped.
        S: Shape (m, m) innovation covariance H P H^T + R (not symmetrized).
        K: Shape (4, m) Kalman gain.
    """

    x: np.ndarray
    P: np.ndarray  # noqa: N815 - standard notation
    residual: np.ndarray
    S: np.ndarray  # noqa: N815 - standard notation
    K: np.ndarray  # noqa: N815 - standard notation


def ekf_update(
    x: np.ndarray,
    P: np.ndarray,  # noqa: N803 - standard notation
    z: np.ndarray,
    model: MeasurementModel,
) -> EkfStep:
    """Pure EKF measurement update: nothing is modified, the corrected estimate is returned.

    Args:
        x: Shape (4,) predicted state.
        P: Shape (4, 4) predicted covariance.
        z: Shape (m,) measurement, matching the model's R.
        model: Measurement model providing h, its Jacobian, R and angle indices.

    Raises:
        ValueError: If z has the wrong shape, or the model cannot be linearized at x.
    """
    z = np.asarray(z, dtype=float)
    m = model.R.shape[0]
    if z.shape != (m,):
        raise ValueError(f"z must have shape ({m},), got {z.shape}")

    # Linearize first: this is where an undefined geometry raises.
    H = model.jacobian(x)  # noqa: N806 - standard notation

    innovation = z - model.h(x)
    # Angles are only meaningful modulo 2*pi; without wrapping, a target
    # crossing the -x axis would produce an innovation of about 2*pi.
    angles = list(model.angle_indices)
    innovation[angles] = wrap_angle(innovation[angles])

    S = H @ P @ H.T + model.R  # noqa: N806 - standard notation
    # K = P H^T S^-1, computed as a linear solve since S is symmetric.
    K = np.linalg.solve(S, H @ P).T  # noqa: N806 - standard notation

    x_new = x + K @ innovation

    # Joseph form keeps P symmetric and positive semi-definite under rounding.
    I_KH = np.eye(4) - K @ H  # noqa: N806 - standard notation
    P_new = I_KH @ P @ I_KH.T + K @ model.R @ K.T  # noqa: N806 - standard notation
    return EkfStep(x_new, P_new, innovation, S, K)


class ExtendedKalmanFilter(CVFilterBase):
    """Extended Kalman filter for state [x, y, vx, vy] and nonlinear measurements.

    The motion model is linear (see CVFilterBase), so prediction is exact. The
    measurement model z = h(x) + v is linearized around the current estimate
    at each update, so one filter can take measurements from different sensors.

    Attributes:
        x: Shape (4,) state estimate.
        P: Shape (4, 4) state covariance.
        F: Shape (4, 4) state transition matrix.
        Q: Shape (4, 4) process noise covariance.
    """

    def update(self, z: np.ndarray, model: MeasurementModel) -> None:
        """Correct the estimate with a measurement from the given sensor model.

        Args:
            z: Shape (m,) measurement, matching the model's R.
            model: Measurement model providing h, its Jacobian, R and angle indices.

        Raises:
            ValueError: If z has the wrong shape, or if the model cannot be
                linearized at the current estimate (e.g. zero range). The
                estimate is left unchanged in that case.
        """
        self.x, self.P = ekf_update(self.x, self.P, z, model)[:2]
