"""EKF that also tracks the sensitivity of its estimate to a constant camera bearing bias.

The filter itself runs exactly as the plain EKF (its x and P are bit for bit those of
ExtendedKalmanFilter). Alongside, it carries V = d x / d b, the derivative of the state
estimate with respect to a constant bias b added to every camera bearing. With a measurement
z = h(x) + J_b b (J_b = 1 for the camera, 0 for the radar) the recursion is

    predict:  V <- F V
    update:   V <- (I - K H) V + K J_b

with K and H of that update. A separately estimated bias b_hat is then applied to the output
only: x_hat = x - V b_hat, P_hat = P + V var_b V^T (two-stage output correction), so the shared
estimate is never fed back into the filter.
"""

import numpy as np

from fusion.filters.ekf import ExtendedKalmanFilter, ekf_update
from fusion.sensors.base import MeasurementModel
from fusion.sensors.camera import CameraModel


def bias_jacobian(model: MeasurementModel) -> np.ndarray:
    """Shape (m,) derivative of the measurement with respect to the camera bearing bias."""
    m = model.R.shape[0]
    return np.ones(m) if isinstance(model, CameraModel) else np.zeros(m)


def correct_output(
    x: np.ndarray,
    P: np.ndarray,  # noqa: N803 - standard notation
    V: np.ndarray,  # noqa: N803 - standard notation
    bias: float,
    bias_variance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a bias estimate to an estimate and its sensitivity: x - V b, P + V var V^T.

    Args:
        x: Shape (4,) state estimate computed from the biased camera data.
        P: Shape (4, 4) its covariance.
        V: Shape (4,) sensitivity d x / d b.
        bias: Estimated bias in radians.
        bias_variance: Variance of the bias estimate in rad^2.
    """
    return x - V * bias, P + bias_variance * np.outer(V, V)


class SensitivityEKF(ExtendedKalmanFilter):
    """Extended Kalman filter with the sensitivity V = d x / d b of its estimate.

    Attributes:
        x: Shape (4,) state estimate (identical to that of ExtendedKalmanFilter).
        P: Shape (4, 4) state covariance (identical).
        V: Shape (4,) derivative of x with respect to a constant camera bearing bias, in
            meters (or m/s) per radian; zero at birth.
    """

    def __init__(
        self,
        dt: float,
        accel_std: float,
        x0: np.ndarray,
        P0: np.ndarray,  # noqa: N803 - standard notation
    ) -> None:
        super().__init__(dt, accel_std, x0, P0)
        self.V = np.zeros(4)

    def predict(self) -> None:
        """Propagate the state, covariance and sensitivity one step."""
        super().predict()
        self.V = self.F @ self.V

    def update(self, z: np.ndarray, model: MeasurementModel) -> None:
        """Correct the estimate and the sensitivity with a measurement.

        Raises:
            ValueError: As ExtendedKalmanFilter.update; nothing is changed then.
        """
        step = ekf_update(self.x, self.P, z, model)
        H = model.jacobian(self.x)  # noqa: N806 - standard notation
        self.V = (np.eye(4) - step.K @ H) @ self.V + step.K @ bias_jacobian(model)
        self.x, self.P = step.x, step.P
