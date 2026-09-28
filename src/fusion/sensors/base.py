"""Common interface for nonlinear measurement models used by the EKF."""

from typing import Protocol

import numpy as np

# Below this distance from the sensor the bearing is undefined and the
# Jacobians (which divide by r and r^2) are numerically meaningless.
MIN_RANGE = 1e-6


class MeasurementModel(Protocol):
    """Nonlinear measurement model z = h(x) + v, v ~ N(0, R).

    Attributes:
        R: Shape (m, m) measurement noise covariance.
        angle_indices: Indices of measurement elements that are angles and
            must be wrapped when differenced.
    """

    R: np.ndarray
    angle_indices: tuple[int, ...]

    def h(self, x: np.ndarray) -> np.ndarray:
        """Expected measurement, shape (m,), for state x of shape (4,)."""
        ...

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """Jacobian dh/dx evaluated at x, shape (m, 4)."""
        ...


def planar_range(x: np.ndarray) -> float:
    """Distance from the sensor at the origin to the target position.

    Args:
        x: Shape (4,) state [x, y, vx, vy].

    Returns:
        The range sqrt(x^2 + y^2) in meters.

    Raises:
        ValueError: If the range is below MIN_RANGE, where the bearing is undefined.
    """
    r = float(np.hypot(x[0], x[1]))
    if r < MIN_RANGE:
        raise ValueError(
            f"Target is at the sensor position (range {r:.3g} m < {MIN_RANGE} m); "
            "bearing and its Jacobian are undefined"
        )
    return r
