"""Camera measurement model: bearing-only with Gaussian noise."""

import numpy as np

from fusion.sensors.base import planar_range


def camera_measure(
    states: np.ndarray,
    bearing_std: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate noisy camera measurements from true target states.

    The camera sits at the origin and measures:
        bearing = angle from the +x axis, counter-clockwise, in radians

    Args:
        states: Shape (n, 4) true states [x, y, vx, vy]. Velocities are ignored.
        bearing_std: Standard deviation of bearing noise in radians.
        rng: Random generator for reproducible noise.

    Returns:
        Shape (n, 1) array of [bearing] measurements.
    """
    if rng is None:
        rng = np.random.default_rng()

    x = states[:, 0]
    y = states[:, 1]
    true_bearing = np.arctan2(y, x)

    n = len(states)
    noisy_bearing = true_bearing + rng.normal(0.0, bearing_std, size=n)

    return noisy_bearing.reshape(n, 1)


class CameraModel:
    """Camera model for the EKF: z = [bearing] = h(x) + v.

    Range is not observable from a single camera, so this model is meant to be
    combined with other sensors rather than used alone.

    Attributes:
        R: Shape (1, 1) measurement noise covariance.
        angle_indices: (0,), the bearing is the only measurement element.
    """

    angle_indices = (0,)

    def __init__(self, bearing_std: float) -> None:
        """Set the measurement noise.

        Args:
            bearing_std: Standard deviation of bearing noise in radians.
        """
        self.R = np.array([[bearing_std**2]])

    def h(self, x: np.ndarray) -> np.ndarray:
        """Expected measurement [arctan2(y, x)] for state x."""
        return np.array([np.arctan2(x[1], x[0])])

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """Jacobian of h at x, shape (1, 4). Velocity columns are zero.

        With r^2 = x^2 + y^2:  d bearing / d(x, y) = [-y / r^2, x / r^2]

        Raises:
            ValueError: If the target is at the camera position (r ~ 0).
        """
        r = planar_range(x)
        return np.array([[-x[1] / r**2, x[0] / r**2, 0.0, 0.0]])
