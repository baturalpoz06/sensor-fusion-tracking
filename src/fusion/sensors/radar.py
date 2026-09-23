"""Radar measurement model: range and bearing with Gaussian noise."""

import numpy as np


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