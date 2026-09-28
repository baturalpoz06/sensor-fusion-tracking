"""Position measurement model: direct [x, y] with Gaussian noise."""

import numpy as np


def position_measure(
    states: np.ndarray,
    pos_std: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate noisy Cartesian position measurements from true target states.

    The sensor measures the target position directly, for example a GPS-like
    sensor or a pre-processed detection in a common Cartesian frame.

    Args:
        states: Shape (n, 4) true states [x, y, vx, vy]. Velocities are ignored.
        pos_std: Standard deviation of the noise on each axis in meters.
        rng: Random generator for reproducible noise.

    Returns:
        Shape (n, 2) array of [x, y] measurements.
    """
    if rng is None:
        rng = np.random.default_rng()

    x = states[:, 0]
    y = states[:, 1]

    n = len(states)
    noisy_x = x + rng.normal(0.0, pos_std, size=n)
    noisy_y = y + rng.normal(0.0, pos_std, size=n)

    return np.column_stack([noisy_x, noisy_y])
