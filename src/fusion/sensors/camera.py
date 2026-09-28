"""Camera measurement model: bearing-only with Gaussian noise."""

import numpy as np


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
