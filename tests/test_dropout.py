"""Tests for sensor dropout: windows, generators, masking and vanishing targets.

Each test docstring names the condition that makes it fail without the feature.
"""

import math
from dataclasses import replace

import numpy as np
import pytest

from fusion.dropout import (
    DropoutWindow,
    MarkovBursts,
    PeriodicFlicker,
    SingleOutage,
    apply_dropout,
    down_steps,
    end_targets,
    step_index,
)
from fusion.mtt_experiment import SCENARIOS
from fusion.mtt_simulation import SimulationConfig, make_mtt_rngs, simulate_mtt

DT = 0.1
SCENE = SimulationConfig(duration=20.0, radar_clutter_rate=5.0, camera_clutter_rate=5.0)
N_POINTS = SCENE.n_steps + 1
RADAR = ("radar",)
BLACKOUT = ("radar", "camera")

# Statistical tests: seeds averaged, and the width of the acceptance interval in standard
# errors (a correct generator fails a 4-sigma test about once in 16,000 runs; the seeds are
# fixed, so the outcome is deterministic).
N_SEEDS = 2000
SIGMAS = 4.0
# The burst schedules of two off fractions must overlap far more than independent ones
# (share 0.3 for off fractions 0.1 and 0.3); measured about 0.78.
MIN_COUPLED_OVERLAP = 0.6
# Step grid of the 20 s - 50 s span used by the statistical tests: 60 s at dt 0.1.
STAT_POINTS = 601


def assert_same_scan(a, b):
    assert (a is None) == (b is None)
    if a is not None:
        np.testing.assert_array_equal(a.z, b.z)
        np.testing.assert_array_equal(a.origin, b.origin)


def assert_same_sim(a, b):
    np.testing.assert_array_equal(a.truth, b.truth)
    for sa, sb in zip(a.radar_scans + a.camera_scans, b.radar_scans + b.camera_scans, strict=True):
        assert_same_scan(sa, sb)


# --- windows and step indices --------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sensor": "lidar", "start": 1.0, "end": 2.0},
        {"sensor": "radar", "start": -1.0, "end": 2.0},
        {"sensor": "radar", "start": 2.0, "end": 2.0},
        {"sensor": "radar", "start": 3.0, "end": 2.0},
        {"sensor": "radar", "start": 1.0, "end": float("nan")},
    ],
)
def test_invalid_windows_are_rejected(kwargs):
    """Fails if a malformed window is accepted and then silently masks nothing or everything."""
    with pytest.raises(ValueError):
        DropoutWindow(**kwargs)


def test_step_index_survives_float_ratios_that_break_a_plain_ceiling():
    """Fails with ceil(time / dt) (7.000000000000001 -> 8) or a k * dt >= start comparison."""
    assert 3 * 0.3 < 0.9  # a comparison of k * dt with the start would skip step 3
    assert math.ceil(0.14 / 0.02) == 8  # a plain ceiling would start the window one step late
    assert step_index(0.9, 0.3) == 3
    assert step_index(0.14, 0.02) == 7
    assert step_index(0.0, 0.1) == 0


def test_down_steps_are_the_half_open_window_on_the_step_grid():
    """Fails if the window is closed at the end or shifted by one step."""
    window = DropoutWindow("radar", 20.0, 24.0)
    mask = down_steps([window], "radar", DT, 601)
    np.testing.assert_array_equal(np.flatnonzero(mask), np.arange(200, 240))


@pytest.mark.parametrize(
    ("dt", "start", "end", "expected"),
    [(0.3, 0.9, 1.5, [3, 4]), (0.02, 0.14, 0.2, [7, 8, 9])],
)
def test_down_steps_at_float_edges(dt, start, end, expected):
    """Fails if the window edges are found by comparing k * dt with start and end."""
    mask = down_steps([DropoutWindow("radar", start, end)], "radar", dt, 20)
    np.testing.assert_array_equal(np.flatnonzero(mask), expected)


def test_down_steps_clip_to_the_simulation_and_separate_the_sensors():
    """Fails if a window runs past the array, or one sensor's window silences the other."""
    windows = [DropoutWindow("radar", 5.0, 100.0), DropoutWindow("camera", 1.0, 2.0)]
    radar = down_steps(windows, "radar", DT, 61)
    assert np.flatnonzero(radar)[[0, -1]].tolist() == [50, 60]
    camera = down_steps(windows, "camera", DT, 61)
    np.testing.assert_array_equal(np.flatnonzero(camera), np.arange(10, 20))
    assert not down_steps([], "radar", DT, 61).any()
    with pytest.raises(ValueError):
        down_steps(windows, "lidar", DT, 61)


def test_overlapping_windows_are_united():
    """Fails if a later window overwrites an earlier one instead of adding to it."""
    windows = [DropoutWindow("radar", 1.0, 3.0), DropoutWindow("radar", 2.0, 4.0)]
    np.testing.assert_array_equal(
        np.flatnonzero(down_steps(windows, "radar", DT, 61)), np.arange(10, 40)
    )


# --- generators ----------------------------------------------------------------------


def test_single_outage_windows_span_and_zero_duration():
    """Fails if a zero duration produces a window, or a blackout does not cover both sensors."""
    outage = SingleOutage(BLACKOUT, 20.0, 4.0)
    assert outage.windows(DT) == (
        DropoutWindow("radar", 20.0, 24.0),
        DropoutWindow("camera", 20.0, 24.0),
    )
    assert outage.span == (20.0, 24.0)
    none = SingleOutage(RADAR, 20.0, 0.0)
    assert none.windows(DT) == () and none.span == (20.0, 20.0)


@pytest.mark.parametrize(
    "build",
    [
        lambda: SingleOutage(RADAR, -1.0, 2.0),
        lambda: SingleOutage(RADAR, 1.0, -2.0),
        lambda: SingleOutage((), 1.0, 2.0),
        lambda: SingleOutage(("radar", "radar"), 1.0, 2.0),
        lambda: SingleOutage(("lidar",), 1.0, 2.0),
        lambda: SingleOutage(["radar"], 1.0, 2.0),
        lambda: PeriodicFlicker(RADAR, 20.0, 30.0, 10.0, 0.0),
        lambda: PeriodicFlicker(RADAR, 20.0, 30.0, 10.0, 11.0),
        lambda: PeriodicFlicker(RADAR, 20.0, 30.0, 10.0, 3.0, phase=10.0),
        lambda: PeriodicFlicker(RADAR, 20.0, 0.0, 10.0, 3.0),
        lambda: MarkovBursts(RADAR, 20.0, 30.0, 1.0, 2.0),
        lambda: MarkovBursts(RADAR, 20.0, 30.0, -0.1, 2.0),
        lambda: MarkovBursts(RADAR, 20.0, 30.0, 0.2, 0.0),
    ],
)
def test_invalid_specs_are_rejected(build):
    """Fails if an impossible outage specification is accepted."""
    with pytest.raises(ValueError):
        build()


def test_periodic_flicker_lists_its_windows_with_clipping_and_phase():
    """Fails if windows drift off the period, are not clipped to the span, or ignore the phase."""
    flicker = PeriodicFlicker(RADAR, 20.0, 30.0, 10.0, 3.0)
    expected = [(20.0, 23.0), (30.0, 33.0), (40.0, 43.0)]
    assert [(w.start, w.end) for w in flicker.windows(DT)] == pytest.approx(expected)
    assert flicker.span == (20.0, 50.0)

    clipped = PeriodicFlicker(RADAR, 20.0, 22.0, 10.0, 3.0).windows(DT)
    assert (clipped[-1].start, clipped[-1].end) == pytest.approx((40.0, 42.0))

    shifted = PeriodicFlicker(RADAR, 20.0, 30.0, 10.0, 3.0, phase=2.5).windows(DT)
    assert [w.start for w in shifted] == pytest.approx([22.5, 32.5, 42.5])


def test_periodic_flicker_on_every_sensor_gives_each_the_same_schedule():
    """Fails if a multi-sensor spec drops a sensor or draws different times per sensor."""
    windows = PeriodicFlicker(BLACKOUT, 20.0, 30.0, 10.0, 1.0).windows(DT)
    by_sensor = {s: [(w.start, w.end) for w in windows if w.sensor == s] for s in BLACKOUT}
    assert by_sensor["radar"] == by_sensor["camera"] and len(by_sensor["radar"]) == 3


def bursts(off_fraction, seed, sensors=RADAR, mean_burst=2.0, span=30.0):
    spec = MarkovBursts(sensors, 20.0, span, off_fraction, mean_burst)
    return spec.windows(DT, np.random.default_rng(seed))


def test_markov_windows_are_deterministic_per_seed_and_inside_the_span():
    """Fails if the chain uses global randomness, or bursts leak outside the span."""
    a, b, c = bursts(0.2, 1), bursts(0.2, 1), bursts(0.2, 2)
    assert a == b and a != c and len(a) > 0
    assert all(20.0 - 1e-9 <= w.start < w.end <= 50.0 + 1e-9 for w in a)


def test_markov_blackout_shares_one_burst_schedule_across_sensors():
    """Fails if every sensor gets its own independent chain in a total blackout."""
    windows = bursts(0.3, 4, sensors=BLACKOUT)
    by_sensor = {s: [(w.start, w.end) for w in windows if w.sensor == s] for s in BLACKOUT}
    assert by_sensor["radar"] == by_sensor["camera"] and by_sensor["radar"]


def test_markov_without_off_time_has_no_windows_and_needs_a_generator():
    """Fails if a zero off fraction still produces bursts or a missing generator is ignored."""
    assert bursts(0.0, 1) == ()
    with pytest.raises(ValueError, match="generator"):
        MarkovBursts(RADAR, 20.0, 30.0, 0.2, 2.0).windows(DT)


def test_markov_rejects_parameters_the_chain_cannot_realize():
    """Fails if a burst shorter than a step or an unreachable off fraction is silently clipped."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="mean_burst"):
        MarkovBursts(RADAR, 20.0, 30.0, 0.2, 0.05).windows(DT, rng)
    with pytest.raises(ValueError, match="too high"):
        MarkovBursts(RADAR, 20.0, 30.0, 0.9, 0.5).windows(DT, rng)


def test_markov_off_fraction_and_burst_length_match_the_requested_statistics():
    """Fails if p and q are swapped (off fraction 1 - pi) or q is per second instead of per step.

    The off fraction is judged against the spread across seeds (steps of one run are
    correlated, so the binomial standard error is about five times too small). The burst
    length is estimated without the bias of bursts cut by the span end: q_hat = off -> on
    transitions / off steps with a successor, with a delta-method standard error over seeds.
    """
    off_fraction, mean_burst = 0.2, 2.0
    fractions, transitions, exposure = [], [], []
    for seed in range(N_SEEDS):
        windows = bursts(off_fraction, seed, mean_burst=mean_burst)
        mask = down_steps(windows, "radar", DT, STAT_POINTS)[200:500]
        fractions.append(mask.mean())
        transitions.append(np.sum(mask[:-1] & ~mask[1:]))
        exposure.append(mask[:-1].sum())

    fractions = np.array(fractions)
    standard_error = fractions.std(ddof=1) / np.sqrt(N_SEEDS)
    assert abs(fractions.mean() - off_fraction) < SIGMAS * standard_error

    transitions, exposure = np.array(transitions), np.array(exposure)
    q_hat = transitions.sum() / exposure.sum()
    q_error = np.sqrt(np.sum((transitions - q_hat * exposure) ** 2)) / exposure.sum()
    q_true = DT / mean_burst
    assert abs(q_hat - q_true) < SIGMAS * q_error


def test_markov_bursts_of_different_off_fractions_are_coupled():
    """Fails if the uniforms are redrawn per off fraction instead of being shared."""
    shares = []
    for seed in range(200):
        low = down_steps(bursts(0.1, seed), "radar", DT, STAT_POINTS)
        high = down_steps(bursts(0.3, seed), "radar", DT, STAT_POINTS)
        if low.any():
            shares.append((low & high).sum() / low.sum())
    assert np.mean(shares) > MIN_COUPLED_OVERLAP


# --- masking -------------------------------------------------------------------------


@pytest.fixture(scope="module")
def full_sim():
    return simulate_mtt(SCENARIOS["separated"], SCENE, make_mtt_rngs(3))


def test_masking_changes_only_the_silent_scans_and_flags_every_silent_step(full_sim):
    """Fails if masking touches truth or scans of up steps, or flags only scheduled scan steps."""
    window = DropoutWindow("radar", 5.0, 10.0)
    dropped = apply_dropout(full_sim, [window], DT)
    radar_down = dropped.radar_down
    np.testing.assert_array_equal(np.flatnonzero(radar_down), np.arange(50, 100))
    assert not dropped.camera_down.any()
    np.testing.assert_array_equal(dropped.sim.truth, full_sim.truth)

    # Precondition: the silenced scans really held something to remove.
    assert any(
        len(scan.z) > 0 for k, scan in enumerate(full_sim.radar_scans) if scan and radar_down[k]
    )
    for k, (before, after) in enumerate(
        zip(full_sim.radar_scans, dropped.sim.radar_scans, strict=True)
    ):
        if before is None:
            assert after is None  # steps without a scheduled scan stay None, flag or not
        elif radar_down[k]:
            assert after.z.shape == (0, 2) and after.origin.shape == (0,)
        else:
            assert_same_scan(before, after)
    assert radar_down[55] and full_sim.radar_scans[55] is None
    for before, after in zip(full_sim.camera_scans, dropped.sim.camera_scans, strict=True):
        assert_same_scan(before, after)


def test_camera_masking_leaves_the_radar_alone_and_empties_camera_scans(full_sim):
    """Fails if a camera window silences the radar or leaves a camera scan with clutter."""
    dropped = apply_dropout(full_sim, [DropoutWindow("camera", 5.0, 10.0)], DT)
    assert not dropped.radar_down.any()
    for before, after in zip(full_sim.radar_scans, dropped.sim.radar_scans, strict=True):
        assert_same_scan(before, after)
    assert sum(len(s.z) for s in full_sim.camera_scans[50:100]) > 0  # precondition
    for scan in dropped.sim.camera_scans[50:100]:
        assert scan.z.shape == (0, 1)


def test_an_outage_removes_clutter_but_a_zero_detection_probability_does_not():
    """Fails if the outage is masked through Pd (clutter would remain in the silent scans)."""
    pd_zero = simulate_mtt(
        SCENARIOS["separated"], replace(SCENE, radar_pd=0.0), make_mtt_rngs(3)
    )
    # The window is half-open, so it must reach past the last step to silence it.
    window = DropoutWindow("radar", 0.0, SCENE.duration + 1.0)
    scheduled = [k for k, s in enumerate(pd_zero.radar_scans) if s is not None]
    cluttered = [k for k in scheduled if len(pd_zero.radar_scans[k].z) > 0]
    assert cluttered  # precondition: with Pd = 0 the scans still hold clutter
    assert all((pd_zero.radar_scans[k].origin == -1).all() for k in cluttered)

    silent = apply_dropout(pd_zero, [window], DT).sim
    assert all(len(silent.radar_scans[k].z) == 0 for k in scheduled)


def test_drawing_the_windows_leaves_the_simulation_draws_unchanged():
    """Fails if the dropout draws come from a stream shared with the simulation."""
    rngs = make_mtt_rngs(5)
    MarkovBursts(BLACKOUT, 5.0, 10.0, 0.3, 2.0).windows(DT, rngs["dropout"])
    assert_same_sim(
        simulate_mtt(SCENARIOS["separated"], SCENE, rngs),
        simulate_mtt(SCENARIOS["separated"], SCENE, make_mtt_rngs(5)),
    )


# --- vanishing targets ---------------------------------------------------------------


@pytest.fixture(scope="module")
def sure_sim():
    scene = replace(SCENE, radar_pd=1.0, camera_pd=1.0)
    return simulate_mtt(SCENARIOS["separated"], scene, make_mtt_rngs(3))


def test_end_targets_removes_only_that_target_from_its_end_step_and_keeps_clutter(sure_sim):
    """Fails if detections vanish early, other targets lose detections, or clutter is removed."""
    trimmed, alive = end_targets(sure_sim, {0: 10.0}, DT)
    np.testing.assert_array_equal(trimmed.truth, sure_sim.truth)

    expected = np.ones_like(alive)
    expected[0, 100:] = False
    np.testing.assert_array_equal(alive, expected)

    # Precondition: target 0 was detected after its end step in the original.
    assert any(
        (s.origin == 0).any() for k, s in enumerate(sure_sim.radar_scans) if s and k >= 100
    )
    for k, (before, after) in enumerate(
        zip(
            sure_sim.radar_scans + sure_sim.camera_scans,
            trimmed.radar_scans + trimmed.camera_scans,
            strict=True,
        )
    ):
        if before is None:
            assert after is None
            continue
        step = k % N_POINTS
        keep = (before.origin != 0) if step >= 100 else np.ones(len(before.origin), dtype=bool)
        np.testing.assert_array_equal(after.z, before.z[keep])
        np.testing.assert_array_equal(after.origin, before.origin[keep])
        assert (after.origin == -1).sum() == (before.origin == -1).sum()


def test_end_targets_rejects_unknown_targets_and_negative_times(sure_sim):
    """Fails if a typo in the target index or time silently changes nothing."""
    with pytest.raises(ValueError, match="unknown target"):
        end_targets(sure_sim, {7: 10.0}, DT)
    with pytest.raises(ValueError, match="end time"):
        end_targets(sure_sim, {0: -1.0}, DT)
