"""Radar measurement model: range and bearing with Gaussian noise."""

import numpy as np

from fusion.sensors.base import planar_range


def radar_measure(
    states: np.ndarray,
    range_std: float,
    bearing_std: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate noisy radar measurements from true target states.

    The radar sits at the origin and measures:
        range   = distance to the target in meters
        bearing = angle from the +x axis, counter-clockwise, in radians

    Args:
        states: Shape (n, 4) true states [x, y, vx, vy]. Velocities are ignored.
        range_std: Standard deviation of range noise in meters.
        bearing_std: Standard deviation of bearing noise in radians.
        rng: Random generator for reproducible noise.

    Returns:
        Shape (n, 2) array of [range, bearing] measurements.
    """
    if rng is None:
        rng = np.random.default_rng()

    x = states[:, 0]
    y = states[:, 1]
    true_range = np.hypot(x, y)
    true_bearing = np.arctan2(y, x)

    n = len(states)
    noisy_range = true_range + rng.normal(0.0, range_std, size=n)
    noisy_bearing = true_bearing + rng.normal(0.0, bearing_std, size=n)

    return np.column_stack([noisy_range, noisy_bearing])


class RadarModel:
    """Radar model for the EKF: z = [range, bearing] = h(x) + v.

    Attributes:
        R: Shape (2, 2) measurement noise covariance.
        angle_indices: (1,), the bearing is the only angle element.
    """

    angle_indices = (1,)

    def __init__(self, range_std: float, bearing_std: float) -> None:
        """Set the measurement noise.

        Args:
            range_std: Standard deviation of range noise in meters.
            bearing_std: Standard deviation of bearing noise in radians.
        """
        self.R = np.diag([range_std**2, bearing_std**2])

    def h(self, x: np.ndarray) -> np.ndarray:
        """Expected measurement [sqrt(x^2 + y^2), arctan2(y, x)] for state x."""
        return np.array([np.hypot(x[0], x[1]), np.arctan2(x[1], x[0])])

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """Jacobian of h at x, shape (2, 4). Velocity columns are zero.

        With r = sqrt(x^2 + y^2):
            d range / d(x, y)   = [x / r, y / r]
            d bearing / d(x, y) = [-y / r^2, x / r^2]

        Raises:
            ValueError: If the target is at the radar position (r ~ 0).
        """
        r = planar_range(x)
        return np.array(
            [
                [x[0] / r, x[1] / r, 0.0, 0.0],
                [-x[1] / r**2, x[0] / r**2, 0.0, 0.0],
            ]
        )


def radar_initial_estimate(
    z: np.ndarray,
    range_std: float,
    bearing_std: float,
    velocity_std: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build an initial state and covariance from the first radar measurement.

    Position comes from the polar-to-Cartesian conversion; velocity is unknown,
    so it starts at zero with a large uncertainty. The position covariance is the
    measurement noise propagated through the conversion, which is elongated
    across the line of sight by roughly range * bearing_std.

    Args:
        z: Shape (2,) measurement [range, bearing].
        range_std: Standard deviation of range noise in meters.
        bearing_std: Standard deviation of bearing noise in radians.
        velocity_std: Prior standard deviation of each velocity component in m/s.

    Returns:
        Tuple (x0, P0) with shapes (4,) and (4, 4).
    """
    r, b = float(z[0]), float(z[1])
    x0 = np.array([r * np.cos(b), r * np.sin(b), 0.0, 0.0])

    # Jacobian of [r cos b, r sin b] with respect to [r, b].
    J = np.array(  # noqa: N806 - standard notation
        [
            [np.cos(b), -r * np.sin(b)],
            [np.sin(b), r * np.cos(b)],
        ]
    )
    P0 = np.zeros((4, 4))  # noqa: N806 - standard notation
    P0[:2, :2] = J @ np.diag([range_std**2, bearing_std**2]) @ J.T
    P0[2:, 2:] = velocity_std**2 * np.eye(2)
    return x0, P0