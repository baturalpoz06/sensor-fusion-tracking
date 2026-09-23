"""Ground-truth target motion for simulation."""

import numpy as np


def constant_velocity_trajectory(
    initial_state: np.ndarray,
    dt: float,
    n_steps: int,
    accel_std: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate a 2D target moving with (nearly) constant velocity.

    State vector: [x, y, vx, vy] in meters and meters/second.

    Args:
        initial_state: Shape (4,) starting state.
        dt: Time step in seconds.
        n_steps: Number of steps to simulate after the initial state.
        accel_std: Standard deviation of random acceleration in m/s^2.
            Zero produces a perfectly straight line.
        rng: Random generator, pass a seeded one for reproducible runs.

    Returns:
        Array of shape (n_steps + 1, 4) with the state at every time step.
    """
    if rng is None:
        rng = np.random.default_rng()

    states = np.zeros((n_steps + 1, 4))
    states[0] = initial_state

    for k in range(n_steps):
        x, y, vx, vy = states[k]
        ax, ay = rng.normal(0.0, accel_std, size=2)
        states[k + 1] = [
            x + vx * dt + 0.5 * ax * dt**2,
            y + vy * dt + 0.5 * ay * dt**2,
            vx + ax * dt,
            vy + ay * dt,
        ]

    return states