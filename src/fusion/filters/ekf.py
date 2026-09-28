"""Extended Kalman filter with a 2D constant-velocity motion model."""

import numpy as np

from fusion.angles import wrap_angle
from fusion.filters.base import CVFilterBase
from fusion.sensors.base import MeasurementModel


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
        z = np.asarray(z, dtype=float)
        m = model.R.shape[0]
        if z.shape != (m,):
            raise ValueError(f"z must have shape ({m},), got {z.shape}")

        # Linearize first: this is where an undefined geometry raises, before
        # any state is modified.
        H = model.jacobian(self.x)  # noqa: N806 - standard notation

        innovation = z - model.h(self.x)
        # Angles are only meaningful modulo 2*pi; without wrapping, a target
        # crossing the -x axis would produce an innovation of about 2*pi.
        angles = list(model.angle_indices)
        innovation[angles] = wrap_angle(innovation[angles])

        S = H @ self.P @ H.T + model.R  # noqa: N806 - standard notation
        # K = P H^T S^-1, computed as a linear solve since S is symmetric.
        K = np.linalg.solve(S, H @ self.P).T  # noqa: N806 - standard notation

        self.x = self.x + K @ innovation

        # Joseph form keeps P symmetric and positive semi-definite under rounding.
        I_KH = np.eye(4) - K @ H  # noqa: N806 - standard notation
        self.P = I_KH @ self.P @ I_KH.T + K @ model.R @ K.T
