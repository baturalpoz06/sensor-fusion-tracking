"""Shared constant-velocity motion model for the Kalman filter family."""

import numpy as np


def cv_transition(dt: float) -> np.ndarray:
    """Shape (4, 4) constant-velocity transition matrix for state [x, y, vx, vy]."""
    return np.array(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def cv_process_noise(dt: float, accel_std: float) -> np.ndarray:
    """Shape (4, 4) process noise of a random acceleration held constant over one step.

    The acceleration moves position by a*dt^2/2 and velocity by a*dt, matching the simulator
    in fusion.scenario.
    """
    G = np.array(  # noqa: N806 - standard Kalman filter notation
        [
            [0.5 * dt**2, 0.0],
            [0.0, 0.5 * dt**2],
            [dt, 0.0],
            [0.0, dt],
        ]
    )
    return accel_std**2 * G @ G.T


class CVFilterBase:
    """State, covariance and prediction step for a 2D constant-velocity model.

    Motion model (constant velocity, white-noise acceleration held over each step):
        x_k+1 = F x_k + G a_k,   a_k ~ N(0, accel_std^2 I)

    Subclasses add a measurement update.

    Attributes:
        x: Shape (4,) state estimate [x, y, vx, vy].
        P: Shape (4, 4) state covariance.
        F: Shape (4, 4) state transition matrix.
        Q: Shape (4, 4) process noise covariance.
    """

    def __init__(
        self,
        dt: float,
        accel_std: float,
        x0: np.ndarray,
        P0: np.ndarray,  # noqa: N803 - standard Kalman filter notation
    ) -> None:
        """Set up the motion model and the initial estimate.

        Args:
            dt: Time step in seconds.
            accel_std: Standard deviation of the random acceleration in m/s^2.
            x0: Shape (4,) initial state estimate.
            P0: Shape (4, 4) initial state covariance.
        """
        self.x = np.array(x0, dtype=float)
        self.P = np.array(P0, dtype=float)
        if self.x.shape != (4,):
            raise ValueError(f"x0 must have shape (4,), got {self.x.shape}")
        if self.P.shape != (4, 4):
            raise ValueError(f"P0 must have shape (4, 4), got {self.P.shape}")

        self.F = cv_transition(dt)
        self.Q = cv_process_noise(dt, accel_std)

    def predict(self) -> None:
        """Propagate the state and covariance one time step forward."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
