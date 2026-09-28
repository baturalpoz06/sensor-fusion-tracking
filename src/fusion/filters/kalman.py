"""Linear Kalman filter with a 2D constant-velocity motion model."""

import numpy as np


class KalmanFilter:
    """Linear Kalman filter for state [x, y, vx, vy] and position measurements [x, y].

    Motion model (constant velocity, white-noise acceleration held over each step):
        x_k+1 = F x_k + G a_k,   a_k ~ N(0, accel_std^2 I)
    Measurement model:
        z_k = H x_k + v_k,       v_k ~ N(0, pos_std^2 I)

    Attributes:
        x: Shape (4,) state estimate.
        P: Shape (4, 4) state covariance.
        F: Shape (4, 4) state transition matrix.
        Q: Shape (4, 4) process noise covariance.
        H: Shape (2, 4) measurement matrix.
        R: Shape (2, 2) measurement noise covariance.
    """

    def __init__(
        self,
        dt: float,
        accel_std: float,
        pos_std: float,
        x0: np.ndarray,
        P0: np.ndarray,  # noqa: N803 - standard Kalman filter notation
    ) -> None:
        """Set up the model matrices and the initial estimate.

        Args:
            dt: Time step in seconds.
            accel_std: Standard deviation of the random acceleration in m/s^2.
            pos_std: Standard deviation of the position measurement noise in meters.
            x0: Shape (4,) initial state estimate.
            P0: Shape (4, 4) initial state covariance.
        """
        self.x = np.array(x0, dtype=float)
        self.P = np.array(P0, dtype=float)
        if self.x.shape != (4,):
            raise ValueError(f"x0 must have shape (4,), got {self.x.shape}")
        if self.P.shape != (4, 4):
            raise ValueError(f"P0 must have shape (4, 4), got {self.P.shape}")

        self.F = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        # Acceleration held constant over one step moves position by a*dt^2/2 and
        # velocity by a*dt, matching the simulator in fusion.scenario.
        G = np.array(  # noqa: N806 - standard Kalman filter notation
            [
                [0.5 * dt**2, 0.0],
                [0.0, 0.5 * dt**2],
                [dt, 0.0],
                [0.0, dt],
            ]
        )
        self.Q = accel_std**2 * G @ G.T

        self.H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        )
        self.R = pos_std**2 * np.eye(2)

    def predict(self) -> None:
        """Propagate the state and covariance one time step forward."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: np.ndarray) -> None:
        """Correct the estimate with a position measurement.

        Args:
            z: Shape (2,) measurement [x, y].
        """
        z = np.asarray(z, dtype=float)
        innovation = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R  # noqa: N806 - standard notation
        # K = P H^T S^-1, computed as a linear solve since S is symmetric.
        K = np.linalg.solve(S, self.H @ self.P).T  # noqa: N806 - standard notation

        self.x = self.x + K @ innovation

        # Joseph form keeps P symmetric and positive semi-definite under rounding.
        I_KH = np.eye(4) - K @ self.H  # noqa: N806 - standard notation
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
