"""Angle helpers."""

import numpy as np


def wrap_angle(a: float | np.ndarray) -> float | np.ndarray:
    """Wrap angles to the interval (-pi, pi].

    Args:
        a: Angle or array of angles in radians.

    Returns:
        The equivalent angle(s) in (-pi, pi], same shape as the input.
    """
    return np.arctan2(np.sin(a), np.cos(a))
