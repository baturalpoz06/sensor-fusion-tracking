"""Camera measurement model: bearing-only with Gaussian noise."""

import numpy as np

from fusion.sensors.base import planar_range


def camera_measure(
    states: np.ndarray,
    bearing_std: float,
    rng: np.random.Generator | None = None,
    *,
    position: tuple[float, float] = (0.0, 0.0),
    bias: float = 0.0,
) -> np.ndarray:
    """Generate noisy camera measurements from true target states.

    The camera sits at `position` (the origin by default) and measures:
        bearing = angle from the +x axis, counter-clockwise, in radians, plus a constant bias

    The bias is added without wrapping, like the noise: the result may leave (-pi, pi], and
    the filter wraps its innovations. The defaults are neutral: the arithmetic is the same
    as for a camera at the origin without bias, and the random draw is unchanged.

    Args:
        states: Shape (n, 4) true states [x, y, vx, vy]. Velocities are ignored.
        bearing_std: Standard deviation of bearing noise in radians.
        rng: Random generator for reproducible noise.
        position: Camera position (x, y) in meters.
        bias: Constant bearing bias in radians (positive is counter-clockwise).

    Returns:
        Shape (n, 1) array of [bearing] measurements.
    """
    if rng is None:
        rng = np.random.default_rng()

    x = states[:, 0]
    y = states[:, 1]
    true_bearing = np.arctan2(y - position[1], x - position[0])

    n = len(states)
    noisy_bearing = true_bearing + bias + rng.normal(0.0, bearing_std, size=n)

    return noisy_bearing.reshape(n, 1)


class CameraModel:
    """Camera model for the EKF: z = [bearing] = h(x) + v.

    Range is not observable from a single camera, so this model is meant to be
    combined with other sensors rather than used alone.

    Attributes:
        R: Shape (1, 1) measurement noise covariance.
        angle_indices: (0,), the bearing is the only measurement element.
        position: Camera position (x, y) in meters that the model assumes.
    """

    angle_indices = (0,)

    def __init__(self, bearing_std: float, position: tuple[float, float] = (0.0, 0.0)) -> None:
        """Set the measurement noise and the assumed camera position.

        Args:
            bearing_std: Standard deviation of bearing noise in radians.
            position: Camera position (x, y) in meters; the origin by default.
        """
        self.R = np.array([[bearing_std**2]])
        self.position = (float(position[0]), float(position[1]))

    def h(self, x: np.ndarray) -> np.ndarray:
        """Expected measurement [arctan2(y - cy, x - cx)] for state x."""
        return np.array([np.arctan2(x[1] - self.position[1], x[0] - self.position[0])])

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """Jacobian of h at x, shape (1, 4). Velocity columns are zero.

        With dx = x - cx, dy = y - cy and r^2 = dx^2 + dy^2:
            d bearing / d(x, y) = [-dy / r^2, dx / r^2]

        Raises:
            ValueError: If the target is at the camera position (r ~ 0).
        """
        dx = x[0] - self.position[0]
        dy = x[1] - self.position[1]
        r = planar_range(np.array([dx, dy]))
        return np.array([[-dy / r**2, dx / r**2, 0.0, 0.0]])
