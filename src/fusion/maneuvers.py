"""Truth-side disturbances: target maneuvers, and the camera's bias, offset and time delay.

Everything here describes the real world of a simulation, never what the tracker believes.
A `TruthDisturbance` is passed to `simulate_mtt`; its defaults are neutral and reproduce the
undisturbed simulation bit for bit.

Motion. A target moves with piecewise-constant acceleration over each step, as in
`fusion.scenario.constant_velocity_trajectory`. A coordinated turn adds a constant turn rate
omega (positive is counter-clockwise, to the left) over a window of steps. Within step k, at
time tau after its start, the state is

    v(tau) = R(omega * tau) v_k + a * tau
    p(tau) = p_k + C(omega, tau) v_k + 0.5 * a * tau^2

with a the total acceleration of the step (random plus scheduled) and C the exact
coordinated-turn displacement. This is the definition of the true trajectory between grid
points: it is continuous, p' = v holds exactly, and it equals the grid states at tau = 0 and
tau = dt. For a turn or an acceleration alone it is the exact solution of the motion; with
both together it differs from the solution of v' = omega J v + a by a splitting error of
about 1e-3 m/s per step at 20 deg/s, which is negligible for bearings. The sub-step states
serve the camera time offset: the camera measures the truth at t - delay.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

# Slack on the ratio time / dt when a time is matched to a step; equal to
# fusion.dropout.STEP_TOLERANCE (that module cannot be imported here: it imports the simulation).
GRID_TOLERANCE = 1e-9


@dataclass(frozen=True)
class Turn:
    """Coordinated turn of one target: a constant turn rate over [start, end).

    Attributes:
        start: First maneuvering time in seconds (>= 0, on the step grid).
        end: First time after the maneuver in seconds (> start, on the step grid).
        turn_rate: Turn rate in rad/s; positive turns counter-clockwise (to the left).
    """

    start: float
    end: float
    turn_rate: float

    def __post_init__(self) -> None:
        _check_window(self.start, self.end)
        if not math.isfinite(self.turn_rate):
            raise ValueError(f"turn_rate must be finite, got {self.turn_rate}")


@dataclass(frozen=True)
class Acceleration:
    """Sudden acceleration of one target along its heading, over [start, end).

    The direction is the heading (velocity direction) that the target has at the first step of
    the window; it stays fixed for the whole window.

    Attributes:
        start: First accelerating time in seconds (>= 0, on the step grid).
        end: First time after the maneuver in seconds (> start, on the step grid).
        along: Acceleration in m/s^2 along the heading (negative brakes).
    """

    start: float
    end: float
    along: float

    def __post_init__(self) -> None:
        _check_window(self.start, self.end)
        if not math.isfinite(self.along):
            raise ValueError(f"along must be finite, got {self.along}")


Maneuver = Turn | Acceleration


@dataclass(frozen=True)
class RandomManeuvers:
    """Random turns of every target, drawn from the "maneuver" stream.

    From `start` on, time is cut into segments of `segment` seconds. In each segment of each
    target a turn happens with the given probability, at a turn rate uniform in
    [-max_turn_rate, max_turn_rate]. Two uniforms are drawn per target and segment whatever
    the parameters, so the draws of different settings are identical.

    Attributes:
        start: Start of the first segment in seconds (>= 0, on the step grid).
        segment: Length of a segment in seconds (> 0, on the step grid).
        max_turn_rate: Largest absolute turn rate in rad/s (>= 0).
        probability: Probability that a segment contains a turn, in [0, 1].
    """

    start: float
    segment: float
    max_turn_rate: float
    probability: float = 0.5

    def __post_init__(self) -> None:
        if not (math.isfinite(self.start) and self.start >= 0.0):
            raise ValueError(f"start must be finite and >= 0, got {self.start}")
        if not (math.isfinite(self.segment) and self.segment > 0.0):
            raise ValueError(f"segment must be finite and > 0, got {self.segment}")
        if not (math.isfinite(self.max_turn_rate) and self.max_turn_rate >= 0.0):
            raise ValueError(f"max_turn_rate must be finite and >= 0, got {self.max_turn_rate}")
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(f"probability must be in [0, 1], got {self.probability}")


@dataclass(frozen=True)
class TruthDisturbance:
    """Everything that makes the simulated world differ from the tracker's default beliefs.

    All defaults are neutral: a camera at the radar, no bias, no time offset, no maneuvers.

    Attributes:
        camera_position: True camera position (x, y) in meters; the radar is at the origin.
        camera_bias: Constant bearing bias of the camera in radians (positive is
            counter-clockwise).
        camera_time_offset: Seconds by which the camera data lag: a measurement stamped t is
            generated from the truth at t - camera_time_offset (>= 0).
        maneuvers: (target index, maneuver) pairs; the maneuvers of one target must not overlap.
        random_maneuvers: Random turns of all targets, or None.
    """

    camera_position: tuple[float, float] = (0.0, 0.0)
    camera_bias: float = 0.0
    camera_time_offset: float = 0.0
    maneuvers: tuple[tuple[int, Maneuver], ...] = ()
    random_maneuvers: RandomManeuvers | None = None

    def __post_init__(self) -> None:
        position_ok = len(self.camera_position) == 2 and all(
            math.isfinite(c) for c in self.camera_position
        )
        if not position_ok:
            raise ValueError(
                f"camera_position must be two finite numbers, got {self.camera_position}"
            )
        if not math.isfinite(self.camera_bias):
            raise ValueError(f"camera_bias must be finite, got {self.camera_bias}")
        if not (math.isfinite(self.camera_time_offset) and self.camera_time_offset >= 0.0):
            raise ValueError(
                f"camera_time_offset must be finite and >= 0, got {self.camera_time_offset}"
            )
        for target, maneuver in self.maneuvers:
            if not isinstance(target, int) or target < 0:
                raise ValueError(f"target index must be an int >= 0, got {target!r}")
            if not isinstance(maneuver, Turn | Acceleration):
                raise ValueError(f"expected a Turn or an Acceleration, got {maneuver!r}")
        if self.random_maneuvers is not None and not isinstance(
            self.random_maneuvers, RandomManeuvers
        ):
            raise ValueError(
                f"random_maneuvers must be RandomManeuvers, got {self.random_maneuvers!r}"
            )


def _check_window(start: float, end: float) -> None:
    if not (math.isfinite(start) and math.isfinite(end)):
        raise ValueError(f"start and end must be finite, got {start} and {end}")
    if not 0.0 <= start < end:
        raise ValueError(f"need 0 <= start < end, got {start} and {end}")


def _grid_step(time: float, dt: float, what: str) -> int:
    """Step index of a time that must lie on the step grid (up to GRID_TOLERANCE)."""
    ratio = time / dt
    step = round(ratio)
    if abs(ratio - step) > GRID_TOLERANCE:
        raise ValueError(f"{what} {time} s is not a multiple of dt = {dt} s")
    return step


class Trajectory(NamedTuple):
    """A simulated trajectory with what is needed to evaluate it between grid points.

    Attributes:
        states: Shape (n + 1, 4) states [x, y, vx, vy] at the grid times.
        accel: Shape (n, 2) total acceleration of each step (random plus scheduled).
        turn_rate: Shape (n,) turn rate of each step in rad/s (0 where the target goes straight).
    """

    states: np.ndarray
    accel: np.ndarray
    turn_rate: np.ndarray


def _advance(
    state: np.ndarray, accel: np.ndarray, turn_rate: np.ndarray, tau: np.ndarray
) -> np.ndarray:
    """State tau seconds after `state`, under the given acceleration and turn rate.

    Works on arrays (shapes (m, 4), (m, 2), (m,), (m,)) and on one state (shapes (4,), (2,)
    and scalars). The sinc forms are finite for turn_rate -> 0, where the result equals the
    straight-line step; for turn_rate = 0 every operation is exactly that of a
    constant-velocity step.
    """
    state = np.asarray(state, dtype=float)
    accel = np.asarray(accel, dtype=float)
    x, y, vx, vy = state[..., 0], state[..., 1], state[..., 2], state[..., 3]
    ax, ay = accel[..., 0], accel[..., 1]
    theta = turn_rate * tau
    half = np.sin(theta / 2.0)
    sin_over_omega = tau * np.sinc(theta / np.pi)  # sin(theta) / omega
    one_minus_cos_over_omega = tau * half * np.sinc(theta / (2.0 * np.pi))  # (1 - cos) / omega
    cos_theta = 1.0 - 2.0 * half**2
    sin_theta = np.sin(theta)
    return np.stack(
        [
            x + sin_over_omega * vx - one_minus_cos_over_omega * vy + 0.5 * ax * tau**2,
            y + one_minus_cos_over_omega * vx + sin_over_omega * vy + 0.5 * ay * tau**2,
            cos_theta * vx - sin_theta * vy + ax * tau,
            sin_theta * vx + cos_theta * vy + ay * tau,
        ],
        axis=-1,
    )


def _step_ranges(
    schedule: Sequence[Maneuver], dt: float, n_steps: int
) -> list[tuple[Maneuver, int, int]]:
    """Each maneuver with its step range [first, last), clipped to the run; overlaps are refused."""
    ranges = []
    for maneuver in schedule:
        first = _grid_step(maneuver.start, dt, "maneuver start")
        last = _grid_step(maneuver.end, dt, "maneuver end")
        ranges.append((maneuver, first, min(last, n_steps)))
    ordered = sorted(ranges, key=lambda item: item[1])
    for (_, _, last), (_, first, _) in zip(ordered, ordered[1:], strict=False):
        if first < last:
            raise ValueError("the maneuvers of one target must not overlap")
    return ranges


def maneuvering_trajectory(
    initial_state: np.ndarray,
    dt: float,
    n_steps: int,
    accel_std: float,
    rng: np.random.Generator,
    schedule: Sequence[Maneuver] = (),
) -> Trajectory:
    """Simulate one target with random acceleration and scheduled maneuvers.

    The random draws are those of `constant_velocity_trajectory`: two normals per step from
    `rng`, in the same order, whatever the schedule. Without a schedule the states are
    identical to its result.

    Args:
        initial_state: Shape (4,) starting state [x, y, vx, vy].
        dt: Time step in seconds.
        n_steps: Number of steps after the initial state.
        accel_std: Standard deviation of the random acceleration in m/s^2.
        rng: Random generator.
        schedule: Maneuvers of this target; start and end times must lie on the step grid.

    Returns:
        The Trajectory with states of shape (n_steps + 1, 4).

    Raises:
        ValueError: If a maneuver is off the step grid, maneuvers overlap, or an
            acceleration starts while the target is at rest (no heading).
    """
    ranges = _step_ranges(schedule, dt, n_steps)
    turn_rate = np.zeros(n_steps)
    accelerating = np.zeros(n_steps, dtype=bool)
    along = np.zeros(n_steps)
    window_start = np.zeros(n_steps, dtype=int)
    for maneuver, first, last in ranges:
        if isinstance(maneuver, Turn):
            turn_rate[first:last] = maneuver.turn_rate
        else:
            accelerating[first:last] = True
            along[first:last] = maneuver.along
            window_start[first:last] = first

    states = np.zeros((n_steps + 1, 4))
    states[0] = initial_state
    accel = np.zeros((n_steps, 2))
    direction = np.zeros(2)
    for k in range(n_steps):
        ax, ay = rng.normal(0.0, accel_std, size=2)
        extra_x = extra_y = 0.0
        if accelerating[k]:
            if k == window_start[k]:
                speed = float(np.hypot(states[k, 2], states[k, 3]))
                if speed == 0.0:
                    raise ValueError("an acceleration along the heading needs a moving target")
                direction = states[k, 2:] / speed
            extra_x, extra_y = along[k] * direction[0], along[k] * direction[1]
        accel[k] = ax + extra_x, ay + extra_y
        states[k + 1] = _advance(states[k], accel[k], turn_rate[k], dt)
    return Trajectory(states, accel, turn_rate)


def _evaluate(trajectory: Trajectory, dt: float, index: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """States tau seconds after grid step `index`; before the start by straight-line extrapolation.

    An index below 0 means a time before the start of the run; the state there is the initial
    state moved on at constant velocity, by index * dt + tau seconds (negative).
    """
    states = trajectory.states
    n_steps = len(trajectory.accel)
    if np.any(index > n_steps) or np.any((index == n_steps) & (tau != 0.0)):
        raise ValueError("a time after the end of the run was requested")
    clipped = np.clip(index, 0, n_steps)
    step = np.clip(index, 0, n_steps - 1)
    inside = _advance(states[step], trajectory.accel[step], trajectory.turn_rate[step], tau)
    result = np.where((tau == 0.0)[..., None], states[clipped], inside)
    seconds = index * dt + tau
    start = states[0]
    extrapolated = np.stack(
        [
            start[0] + start[2] * seconds,
            start[1] + start[3] * seconds,
            np.full_like(seconds, start[2]),
            np.full_like(seconds, start[3]),
        ],
        axis=-1,
    )
    return np.where((index < 0)[..., None], extrapolated, result)


def state_at(trajectory: Trajectory, dt: float, times: np.ndarray) -> np.ndarray:
    """True states at arbitrary times, between the grid points of the trajectory.

    A time that is a whole number of steps up to rounding (0.3 / 0.1) returns the grid state
    bit for bit. Before time zero the target is extrapolated at its initial velocity.

    Args:
        trajectory: Result of maneuvering_trajectory.
        dt: Time step in seconds.
        times: Shape (m,) times in seconds, at most the end of the run.

    Returns:
        Shape (m, 4) states.

    Raises:
        ValueError: If a time is after the end of the run.
    """
    times = np.atleast_1d(np.asarray(times, dtype=float))
    index = np.floor(times / dt + GRID_TOLERANCE).astype(int)
    tau = times - index * dt
    tau = np.where(np.abs(tau) < GRID_TOLERANCE * dt, 0.0, tau)
    return _evaluate(trajectory, dt, index, tau)


def delayed_states(trajectory: Trajectory, dt: float, delay: float) -> np.ndarray:
    """The truth as seen with a time delay: at every grid time t_k the state at t_k - delay.

    The delay is split into whole steps and a remainder, m = ceil(delay / dt) and
    tau = (m - delay / dt) * dt in [0, dt), so that t_k - delay = t_(k-m) + tau. A delay of
    whole steps reads the grid states themselves; otherwise the states are evaluated from the
    motion model (see the module docstring), not interpolated between samples. For
    t_k < delay the target is extrapolated back at its initial velocity.

    Args:
        trajectory: Result of maneuvering_trajectory.
        dt: Time step in seconds.
        delay: Delay in seconds (>= 0).

    Returns:
        Shape (n + 1, 4) states, one per grid time.

    Raises:
        ValueError: If delay is negative.
    """
    if delay < 0.0:
        raise ValueError(f"delay must be >= 0, got {delay}")
    ratio = delay / dt
    whole = math.ceil(ratio - GRID_TOLERANCE)
    tau = (whole - ratio) * dt
    tau = 0.0 if abs(tau) < GRID_TOLERANCE * dt else tau
    index = np.arange(len(trajectory.states)) - whole
    return _evaluate(trajectory, dt, index, np.full(len(index), tau))


def draw_random_schedule(
    spec: RandomManeuvers, n_targets: int, dt: float, n_steps: int, rng: np.random.Generator
) -> list[list[Turn]]:
    """Draw the random turns of every target from the "maneuver" stream.

    The uniforms have the fixed shape (n_targets, n_segments, 2): the first decides whether the
    segment contains a turn, the second gives its rate.

    Args:
        spec: Random maneuver settings.
        n_targets: Number of targets.
        dt: Time step in seconds.
        n_steps: Number of steps of the run.
        rng: Generator of the "maneuver" stream.

    Returns:
        One list of turns per target.
    """
    first = _grid_step(spec.start, dt, "random maneuver start")
    length = _grid_step(spec.segment, dt, "random maneuver segment")
    n_segments = max((n_steps - first) // length, 0)
    uniforms = rng.random((n_targets, n_segments, 2))
    schedule: list[list[Turn]] = [[] for _ in range(n_targets)]
    for target in range(n_targets):
        for j in range(n_segments):
            if uniforms[target, j, 0] < spec.probability:
                rate = (2.0 * uniforms[target, j, 1] - 1.0) * spec.max_turn_rate
                begin = (first + j * length) * dt
                schedule[target].append(Turn(begin, begin + spec.segment, float(rate)))
    return schedule


def resolve_schedules(
    disturbance: TruthDisturbance,
    n_targets: int,
    dt: float,
    n_steps: int,
    rng: np.random.Generator,
) -> list[list[Maneuver]]:
    """The maneuvers of every target: the explicit ones plus the random turns.

    The generator is read only if the disturbance has random maneuvers.

    Raises:
        ValueError: If a maneuver names a target that does not exist.
    """
    schedules: list[list[Maneuver]] = [[] for _ in range(n_targets)]
    for target, maneuver in disturbance.maneuvers:
        if target >= n_targets:
            raise ValueError(f"unknown target {target}, the simulation has {n_targets}")
        schedules[target].append(maneuver)
    if disturbance.random_maneuvers is not None:
        drawn = draw_random_schedule(disturbance.random_maneuvers, n_targets, dt, n_steps, rng)
        for target, turns in enumerate(drawn):
            schedules[target].extend(turns)
    return schedules
