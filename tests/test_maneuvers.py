"""Tests for the truth-side maneuvers: turns, accelerations, sub-step evaluation, time delay.

Each test docstring names the condition that makes it fail without the feature.
"""

import numpy as np
import pytest

from fusion.dropout import STEP_TOLERANCE
from fusion.maneuvers import (
    GRID_TOLERANCE,
    Acceleration,
    RandomManeuvers,
    TruthDisturbance,
    Turn,
    _advance,
    delayed_states,
    draw_random_schedule,
    maneuvering_trajectory,
    resolve_schedules,
    state_at,
)
from fusion.scenario import constant_velocity_trajectory

DT = 0.1
START = np.array([-450.0, 1000.0, 15.0, 3.0])


def heading(states: np.ndarray) -> np.ndarray:
    return np.arctan2(states[..., 3], states[..., 2])


def speed(states: np.ndarray) -> np.ndarray:
    return np.hypot(states[..., 2], states[..., 3])


class ScriptedRng:
    """Stands in for a generator: returns the given acceleration rows one step at a time."""

    def __init__(self, rows):
        self.rows = iter(rows)

    def normal(self, loc, scale, size):
        return np.array(next(self.rows), dtype=float)


# --- neutrality ------------------------------------------------------------------------


@pytest.mark.parametrize("accel_std", [0.5, 0.0])
def test_no_schedule_reproduces_the_constant_velocity_trajectory_bitwise(accel_std):
    """Fails if the maneuvering step changes a single bit of an undisturbed trajectory."""
    reference = constant_velocity_trajectory(START, DT, 600, accel_std, np.random.default_rng(3))
    result = maneuvering_trajectory(START, DT, 600, accel_std, np.random.default_rng(3))
    np.testing.assert_array_equal(result.states, reference)


@pytest.mark.parametrize(
    "schedule",
    [[], [Turn(5.0, 15.0, 0.2)], [Acceleration(5.0, 8.0, 2.0)]],
    ids=["none", "turn", "acceleration"],
)
def test_the_schedule_does_not_change_the_random_draws(schedule):
    """Fails if a maneuver consumes or skips draws of the trajectory stream."""
    a, b = np.random.default_rng(4), np.random.default_rng(4)
    constant_velocity_trajectory(START, DT, 200, 0.5, a)
    maneuvering_trajectory(START, DT, 200, 0.5, b, schedule)
    assert a.bit_generator.state == b.bit_generator.state


def test_a_zero_turn_rate_equals_going_straight_bitwise():
    """Fails if a turn of rate 0 (the neutral row of the omega sweep) differs from no turn."""
    plain = maneuvering_trajectory(START, DT, 300, 0.5, np.random.default_rng(1))
    zero = maneuvering_trajectory(START, DT, 300, 0.5, np.random.default_rng(1), [Turn(5, 20, 0.0)])
    np.testing.assert_array_equal(zero.states, plain.states)


# --- coordinated turn ------------------------------------------------------------------


@pytest.mark.parametrize("rate_deg", [5.0, 10.0, 20.0, -10.0])
def test_a_turn_keeps_the_speed_and_changes_the_heading_by_omega_times_time(rate_deg):
    """Fails if the turn changes the speed, has the wrong rate or the wrong direction.

    Positive rates turn left (counter-clockwise). The first turned state is the one after the
    first step of the window, so the heading is constant up to the window start.
    """
    omega = np.deg2rad(rate_deg)
    trajectory = maneuvering_trajectory(
        START, DT, 400, 0.0, np.random.default_rng(0), [Turn(15.0, 25.0, omega)]
    )
    states = trajectory.states
    np.testing.assert_allclose(speed(states), speed(START), rtol=1e-12)
    np.testing.assert_array_equal(
        states[:151], maneuvering_trajectory(START, DT, 150, 0.0, np.random.default_rng(0)).states
    )
    change = np.angle(np.exp(1j * (heading(states[250]) - heading(states[150]))))
    assert change == pytest.approx(np.angle(np.exp(1j * omega * 10.0)), abs=1e-12)
    np.testing.assert_allclose(heading(states[300]), heading(states[250]), atol=1e-12)


@pytest.mark.parametrize("rate_deg", [5.0, 10.0, 20.0])
def test_a_full_turn_circle_closes(rate_deg):
    """Fails if the turn displacement is not the exact arc: the target must return to its start.

    The window lasts one full revolution, 2 pi / omega, a whole number of steps for these rates.
    """
    omega = np.deg2rad(rate_deg)
    n = round(2.0 * np.pi / omega / DT)
    trajectory = maneuvering_trajectory(
        START, DT, n, 0.0, np.random.default_rng(0), [Turn(0.0, n * DT, omega)]
    )
    np.testing.assert_allclose(trajectory.states[-1], START, atol=1e-9)
    radius = speed(START) / omega
    centre = START[:2] + radius * np.array([-np.sin(heading(START)), np.cos(heading(START))])
    distance = np.hypot(*(trajectory.states[:, :2] - centre).T)
    np.testing.assert_allclose(distance, radius, rtol=1e-12)


def test_the_turn_step_is_finite_and_straight_for_vanishing_rates():
    """Fails if the turn displacement is a 0/0 at omega -> 0 (sin(x)/omega without a limit)."""
    state, accel = np.array([10.0, 5.0, 3.0, -4.0]), np.array([0.2, -0.1])
    straight = _advance(state, accel, 0.0, DT)
    assert np.isfinite(straight).all()
    for omega in (1e-300, 1e-12, -1e-12):  # the real effect is about omega * dt * speed
        np.testing.assert_allclose(_advance(state, accel, omega, DT), straight, rtol=1e-12)


# --- acceleration ----------------------------------------------------------------------


def test_an_acceleration_changes_the_velocity_and_position_along_the_start_heading():
    """Fails if the acceleration has the wrong size, direction or window.

    The heading is the realized one at the first step of the window, so the target here has
    random acceleration before it and the direction is read from its velocity at that time.
    """
    trajectory = maneuvering_trajectory(
        START, DT, 300, 0.5, np.random.default_rng(2), [Acceleration(10.0, 13.0, 2.0)]
    )
    plain = maneuvering_trajectory(START, DT, 300, 0.5, np.random.default_rng(2))
    np.testing.assert_array_equal(trajectory.states[:101], plain.states[:101])
    k0, k1 = 100, 130
    direction = trajectory.states[k0, 2:] / speed(trajectory.states[k0])
    expected_dv = 2.0 * 3.0 * direction
    delta_v = (trajectory.states[k1, 2:] - trajectory.states[k0, 2:]) - (
        plain.states[k1, 2:] - plain.states[k0, 2:]
    )
    np.testing.assert_allclose(delta_v, expected_dv, atol=1e-12)
    delta_p = (trajectory.states[k1, :2] - plain.states[k1, :2]) - (
        trajectory.states[k0, :2] - plain.states[k0, :2]
    )
    # The extra velocity of the window start is zero, so only 0.5 A T^2 is gained over the window.
    np.testing.assert_allclose(delta_p, 0.5 * 2.0 * 3.0**2 * direction, atol=1e-10)


def test_an_acceleration_from_rest_is_refused():
    """Fails if a window without a heading silently divides by a zero speed."""
    with pytest.raises(ValueError, match="moving target"):
        maneuvering_trajectory(
            np.array([0.0, 100.0, 0.0, 0.0]), DT, 50, 0.0, np.random.default_rng(0),
            [Acceleration(1.0, 2.0, 1.0)],
        )  # fmt: skip


# --- evaluation between grid points ----------------------------------------------------


def test_state_at_grid_times_returns_the_grid_states_bitwise_also_after_rounding():
    """Fails if a time like 0.3 s (3 steps up to rounding) is evaluated in the previous step."""
    trajectory = maneuvering_trajectory(
        START, DT, 100, 0.5, np.random.default_rng(1), [Turn(2.0, 6.0, 0.2)]
    )
    steps = np.arange(101)
    times = steps * DT
    assert 0.3 / 0.1 != 3.0  # the rounding trap exists
    np.testing.assert_array_equal(state_at(trajectory, DT, times), trajectory.states)
    np.testing.assert_array_equal(state_at(trajectory, DT, np.array([0.1 + 0.2]))[0],
                                  trajectory.states[3])  # fmt: skip


@pytest.mark.parametrize(
    "schedule",
    [[Turn(1.0, 4.0, 0.35)], [Acceleration(1.0, 3.0, 2.5)], []],
    ids=["turn", "acceleration", "straight"],
)
def test_sub_step_states_equal_a_simulation_with_a_quarter_step(schedule):
    """Fails if sub-step evaluation is not the motion model: turn only or acceleration only.

    The target moves exactly under these models, so four quarter steps give the same state as
    one step evaluated at the quarter points (no random acceleration: it is attached to whole
    steps, so it would not be the same model at another step length).
    """
    coarse = maneuvering_trajectory(START, DT, 60, 0.0, np.random.default_rng(0), schedule)
    fine = maneuvering_trajectory(START, DT / 4, 240, 0.0, np.random.default_rng(0), schedule)
    times = np.arange(241) * (DT / 4)
    np.testing.assert_allclose(state_at(coarse, DT, times), fine.states, atol=1e-11)


def test_sub_step_states_with_a_turn_and_random_acceleration_are_consistent():
    """Fails if the sub-step motion is not continuous, p' = v, or does not end at the next state."""
    trajectory = maneuvering_trajectory(
        START, DT, 120, 0.5, np.random.default_rng(6), [Turn(2.0, 9.0, np.deg2rad(20.0))]
    )
    states = trajectory.states
    # Continuity at the grid points: just before a point the state is almost the point itself.
    eps = 1e-9
    before = state_at(trajectory, DT, np.arange(1, 121) * DT - eps)
    np.testing.assert_allclose(before, states[1:], atol=1e-6)
    # dp/dt = v inside steps, by central differences.
    h = 1e-5
    mid = (np.arange(0, 120) + 0.37) * DT
    velocity = (
        state_at(trajectory, DT, mid + h)[:, :2] - state_at(trajectory, DT, mid - h)[:, :2]
    ) / (2 * h)
    np.testing.assert_allclose(velocity, state_at(trajectory, DT, mid)[:, 2:], atol=1e-6)


def test_state_at_before_the_start_extrapolates_and_after_the_end_is_refused():
    """Fails if a negative time wraps around to the end of the array, or a late time is accepted."""
    trajectory = maneuvering_trajectory(START, DT, 20, 0.5, np.random.default_rng(0))
    early = state_at(trajectory, DT, np.array([-0.25, -0.05]))
    np.testing.assert_allclose(early[:, :2], START[:2] + np.outer([-0.25, -0.05], START[2:]))
    np.testing.assert_array_equal(early[:, 2:], [START[2:], START[2:]])
    with pytest.raises(ValueError, match="after the end"):
        state_at(trajectory, DT, np.array([2.0 + 0.01]))


# --- camera time delay -----------------------------------------------------------------


def test_the_delay_is_evaluated_at_t_minus_delay_from_hand_computed_states():
    """Fails if the acceleration of a step is attached to the wrong interval, or the delay has the
    wrong sign or rounding (floor instead of ceil, t + delay).

    Scripted accelerations: 0 in step 0, (2, 0) in step 1. The target starts at x = 0 with
    10 m/s, so at 0.1 s it is at x = 1 with 10 m/s, and in step 1, tau seconds in, at
    x = 1 + 10 tau + tau^2. Delay 0.075 s at t = 0.2 s is 0.125 s (tau = 0.025); delay 0.025 s
    at t = 0.2 s is 0.175 s (tau = 0.075).
    """
    rows = [(0.0, 0.0), (2.0, 0.0), (0.0, 0.0), (0.0, 0.0)]
    trajectory = maneuvering_trajectory(
        np.array([0.0, 0.0, 10.0, 0.0]), DT, 4, 1.0, ScriptedRng(rows)
    )
    delayed = delayed_states(trajectory, DT, 0.075)
    np.testing.assert_allclose(delayed[2], [1.0 + 0.25 + 0.000625, 0.0, 10.05, 0.0], atol=1e-12)
    delayed = delayed_states(trajectory, DT, 0.025)
    np.testing.assert_allclose(delayed[2], [1.0 + 0.75 + 0.005625, 0.0, 10.15, 0.0], atol=1e-12)
    # A delay of 0.025 s at step 1 (t = 0.1 s) reaches into step 0, which has no acceleration.
    np.testing.assert_allclose(delayed[1], [0.75, 0.0, 10.0, 0.0], atol=1e-12)


@pytest.mark.parametrize("delay_ms, steps_back", [(0, 0), (100, 1), (200, 2)])
def test_a_delay_of_whole_steps_reads_the_grid_states_bitwise(delay_ms, steps_back):
    """Fails if a delay of 100 or 200 ms goes through the sub-step formula instead of the grid."""
    trajectory = maneuvering_trajectory(
        START, DT, 100, 0.5, np.random.default_rng(2), [Turn(2.0, 6.0, 0.3)]
    )
    delayed = delayed_states(trajectory, DT, delay_ms / 1000.0)
    k = np.arange(steps_back, 101)
    np.testing.assert_array_equal(delayed[k], trajectory.states[k - steps_back])


@pytest.mark.parametrize("delay_ms", [25, 50, 100, 200])
def test_the_delayed_state_is_the_state_at_t_minus_delay(delay_ms):
    """Fails if delayed_states disagrees with state_at at the shifted times."""
    delay = delay_ms / 1000.0
    trajectory = maneuvering_trajectory(
        START, DT, 100, 0.5, np.random.default_rng(2), [Turn(2.0, 6.0, 0.3)]
    )
    times = np.arange(101) * DT - delay
    np.testing.assert_allclose(
        delayed_states(trajectory, DT, delay), state_at(trajectory, DT, times), atol=1e-12
    )


def test_steps_before_the_delay_extrapolate_the_initial_velocity_instead_of_wrapping():
    """Fails if t - delay < 0 indexes the end of the array (numpy negative indices wrap)."""
    trajectory = maneuvering_trajectory(START, DT, 30, 0.5, np.random.default_rng(2))
    delayed = delayed_states(trajectory, DT, 0.2)
    for k in (0, 1):
        seconds = k * DT - 0.2
        np.testing.assert_allclose(delayed[k, :2], START[:2] + seconds * START[2:], atol=1e-12)
        np.testing.assert_array_equal(delayed[k, 2:], START[2:])
    np.testing.assert_array_equal(delayed[2], trajectory.states[0])


def test_a_negative_delay_is_refused():
    """Fails if a delay into the future (which the truth cannot provide) is accepted."""
    trajectory = maneuvering_trajectory(START, DT, 5, 0.0, np.random.default_rng(0))
    with pytest.raises(ValueError, match="delay"):
        delayed_states(trajectory, DT, -0.1)


# --- schedules -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "schedule",
    [
        [Turn(1.05, 2.0, 0.1)],  # start off the grid
        [Turn(1.0, 2.05, 0.1)],  # end off the grid
        [Turn(1.0, 3.0, 0.1), Acceleration(2.0, 4.0, 1.0)],  # overlap
        [Turn(1.0, 3.0, 0.1), Turn(2.5, 4.0, 0.2)],
    ],
)
def test_maneuvers_off_the_grid_or_overlapping_are_refused(schedule):
    """Fails if a window is silently rounded to the grid or two maneuvers are mixed."""
    with pytest.raises(ValueError):
        maneuvering_trajectory(START, DT, 100, 0.0, np.random.default_rng(0), schedule)


def test_touching_maneuvers_are_allowed_and_a_window_may_run_past_the_end():
    """Fails if the half-open windows [start, end) are treated as closed, or the clip is missing."""
    schedule = [Turn(1.0, 2.0, 0.1), Acceleration(2.0, 3.0, 1.0), Turn(9.0, 50.0, 0.1)]
    result = maneuvering_trajectory(START, DT, 100, 0.0, np.random.default_rng(0), schedule)
    assert result.states.shape == (101, 4)
    assert result.turn_rate[10] == 0.1 and result.turn_rate[9] == 0.0
    assert result.turn_rate[99] == 0.1


@pytest.mark.parametrize(
    "build",
    [
        lambda: Turn(2.0, 1.0, 0.1),
        lambda: Turn(-1.0, 1.0, 0.1),
        lambda: Turn(0.0, 1.0, float("nan")),
        lambda: Acceleration(0.0, 1.0, float("inf")),
        lambda: RandomManeuvers(0.0, 0.0, 0.1),
        lambda: RandomManeuvers(0.0, 5.0, -0.1),
        lambda: RandomManeuvers(0.0, 5.0, 0.1, probability=1.5),
        lambda: TruthDisturbance(camera_time_offset=-0.01),
        lambda: TruthDisturbance(camera_bias=float("nan")),
        lambda: TruthDisturbance(camera_position=(1.0,)),
        lambda: TruthDisturbance(maneuvers=((0, "turn"),)),
        lambda: TruthDisturbance(maneuvers=((-1, Turn(0.0, 1.0, 0.1)),)),
        lambda: TruthDisturbance(random_maneuvers="random"),
    ],
)
def test_invalid_maneuvers_and_disturbances_are_rejected_when_built(build):
    """Fails if a bad setting is only noticed in a worker process in the middle of a sweep."""
    with pytest.raises(ValueError):
        build()


def test_the_default_disturbance_is_neutral():
    """Fails if a default silently adds a bias, an offset, a delay or a maneuver."""
    d = TruthDisturbance()
    assert d.camera_position == (0.0, 0.0) and d.camera_bias == 0.0
    assert d.camera_time_offset == 0.0 and d.maneuvers == () and d.random_maneuvers is None


def test_an_unknown_target_in_the_schedule_is_refused():
    """Fails if a maneuver of a target that does not exist is silently dropped."""
    disturbance = TruthDisturbance(maneuvers=((3, Turn(1.0, 2.0, 0.1)),))
    with pytest.raises(ValueError, match="unknown target"):
        resolve_schedules(disturbance, 2, DT, 100, np.random.default_rng(0))


def test_resolve_schedules_collects_the_maneuvers_by_target():
    """Fails if maneuvers land on the wrong target."""
    turn, brake = Turn(1.0, 2.0, 0.1), Acceleration(3.0, 4.0, -1.0)
    disturbance = TruthDisturbance(maneuvers=((1, turn), (0, brake)))
    schedules = resolve_schedules(disturbance, 3, DT, 100, np.random.default_rng(0))
    assert schedules == [[brake], [turn], []]


# --- random maneuvers ------------------------------------------------------------------


def test_random_maneuvers_read_the_stream_only_when_they_are_set():
    """Fails if the maneuver stream is consumed by a disturbance without random maneuvers."""
    a, b = np.random.default_rng(8), np.random.default_rng(8)
    resolve_schedules(TruthDisturbance(maneuvers=((0, Turn(1.0, 2.0, 0.1)),)), 2, DT, 100, a)
    assert a.bit_generator.state == b.bit_generator.state
    resolve_schedules(
        TruthDisturbance(random_maneuvers=RandomManeuvers(1.0, 2.0, 0.1)), 2, DT, 100, a
    )
    assert a.bit_generator.state != b.bit_generator.state


def test_random_draws_have_a_fixed_shape_whatever_the_parameters():
    """Fails if the number of uniforms drawn depends on the probability or the rate.

    That would shift the stream between rows of a sweep and break the common random numbers.
    """
    states = []
    for probability, max_rate in ((0.0, 0.1), (0.5, 0.2), (1.0, 0.4)):
        rng = np.random.default_rng(9)
        draw_random_schedule(RandomManeuvers(1.0, 2.0, max_rate, probability), 3, DT, 100, rng)
        states.append(rng.bit_generator.state)
    assert states[0] == states[1] == states[2]


def test_random_turns_follow_the_draws_and_scale_with_the_maximum_rate():
    """Fails if the segments, the turn decision or the rate mapping are wrong."""
    spec = RandomManeuvers(1.0, 2.0, 0.2, probability=1.0)
    uniforms = np.random.default_rng(5).random((2, 4, 2))
    schedule = draw_random_schedule(spec, 2, DT, 90, np.random.default_rng(5))
    assert [len(turns) for turns in schedule] == [4, 4]  # (9 s - 1 s) / 2 s segments
    for target, turns in enumerate(schedule):
        for j, turn in enumerate(turns):
            assert turn.start == pytest.approx(1.0 + 2.0 * j)
            assert turn.end == pytest.approx(3.0 + 2.0 * j)
            assert turn.turn_rate == pytest.approx((2.0 * uniforms[target, j, 1] - 1.0) * 0.2)
    double = draw_random_schedule(
        RandomManeuvers(1.0, 2.0, 0.4, probability=1.0), 2, DT, 90, np.random.default_rng(5)
    )
    assert double[1][2].turn_rate == pytest.approx(2.0 * schedule[1][2].turn_rate)
    none = draw_random_schedule(
        RandomManeuvers(1.0, 2.0, 0.2, probability=0.0), 2, DT, 90, np.random.default_rng(5)
    )
    assert none == [[], []]


def test_random_turns_are_deterministic_per_seed():
    """Fails if the schedule does not follow the seed."""
    spec = RandomManeuvers(1.0, 2.0, 0.2)
    a = draw_random_schedule(spec, 3, DT, 100, np.random.default_rng(1))
    b = draw_random_schedule(spec, 3, DT, 100, np.random.default_rng(1))
    c = draw_random_schedule(spec, 3, DT, 100, np.random.default_rng(2))
    assert a == b and a != c


def test_the_grid_tolerance_equals_the_dropout_step_tolerance():
    """Fails if the two copies of the tolerance drift apart (they cannot share one: a cycle)."""
    assert GRID_TOLERANCE == STEP_TOLERANCE
