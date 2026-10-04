"""Sensor dropout: outage windows, their generators and masking of generated scans.

A dropout silences a sensor: during an outage it delivers neither target detections nor
clutter. Outages are applied after the full scans were generated (apply_dropout), so the
truth, the noise, the detection draws and the clutter draws are identical across outage
settings; only the masked scans differ. This differs from a detection probability of
zero, where clutter still arrives.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from fusion.mtt_simulation import MttSimulation, Scan

SENSORS = ("radar", "camera")
# Number of measurement values per scan point of each sensor, for empty scans.
SCAN_DIMENSIONS = {"radar": 2, "camera": 1}
# Slack on the ratio time / dt when a time is converted to a step index, so that a time
# that is a whole number of steps up to rounding (0.3 / 0.1) is not pushed to the next step.
STEP_TOLERANCE = 1e-9


@dataclass(frozen=True)
class DropoutWindow:
    """Interval in which one sensor is silent: half-open [start, end), in seconds.

    Attributes:
        sensor: "radar" or "camera".
        start: First silent time in seconds (>= 0).
        end: First time after the outage in seconds (> start).
    """

    sensor: str
    start: float
    end: float

    def __post_init__(self) -> None:
        if self.sensor not in SENSORS:
            raise ValueError(f"sensor must be one of {SENSORS}, got {self.sensor!r}")
        if not (np.isfinite(self.start) and np.isfinite(self.end)):
            raise ValueError(f"start and end must be finite, got {self.start} and {self.end}")
        if not 0.0 <= self.start < self.end:
            raise ValueError(f"need 0 <= start < end, got {self.start} and {self.end}")


def step_index(time: float, dt: float) -> int:
    """Index of the first step at or after the given time: ceil(time / dt), robust to rounding.

    Step k is at time k * dt, so step_index(start, dt) is the first step of a window that
    starts at `start` and step_index(end, dt) is the first step after it.
    """
    return math.ceil(time / dt - STEP_TOLERANCE)


def down_steps(
    windows: Sequence[DropoutWindow], sensor: str, dt: float, n_points: int
) -> np.ndarray:
    """Boolean mask of the steps at which the sensor is silent.

    Step k (time k * dt) is silent iff it lies in [start, end) of a window of this sensor.
    Windows are clipped to the n_points steps of the simulation.

    Args:
        windows: Dropout windows of any sensor.
        sensor: The sensor whose mask is wanted.
        dt: Time step in seconds.
        n_points: Number of steps of the simulation.

    Returns:
        Shape (n_points,) boolean array.
    """
    if sensor not in SENSORS:
        raise ValueError(f"sensor must be one of {SENSORS}, got {sensor!r}")
    mask = np.zeros(n_points, dtype=bool)
    for window in windows:
        if window.sensor == sensor:
            first = max(step_index(window.start, dt), 0)
            last = min(step_index(window.end, dt), n_points)
            mask[first:last] = True
    return mask


def _check_sensors(sensors: tuple[str, ...]) -> None:
    if not isinstance(sensors, tuple) or not sensors:
        raise ValueError(f"sensors must be a non-empty tuple, got {sensors!r}")
    if len(set(sensors)) != len(sensors) or any(s not in SENSORS for s in sensors):
        raise ValueError(f"sensors must be distinct names from {SENSORS}, got {sensors}")


def _for_sensors(
    sensors: tuple[str, ...], intervals: Sequence[tuple[float, float]]
) -> tuple[DropoutWindow, ...]:
    return tuple(DropoutWindow(s, start, end) for s in sensors for start, end in intervals)


@dataclass(frozen=True)
class SingleOutage:
    """One outage of the given sensors; several sensors silent together is a total blackout.

    Attributes:
        sensors: Silenced sensors, a tuple of names from SENSORS.
        start: Start time in seconds (>= 0).
        duration: Length in seconds (>= 0); zero means no outage.
    """

    sensors: tuple[str, ...]
    start: float
    duration: float

    def __post_init__(self) -> None:
        _check_sensors(self.sensors)
        if self.start < 0.0 or self.duration < 0.0:
            raise ValueError(
                f"start and duration must be >= 0, got {self.start} and {self.duration}"
            )

    @property
    def span(self) -> tuple[float, float]:
        """Interval (start, end) in seconds that the outage metrics refer to."""
        return self.start, self.start + self.duration

    def windows(
        self, dt: float, rng: np.random.Generator | None = None
    ) -> tuple[DropoutWindow, ...]:
        """The outage as windows. No random draws: dt and rng only give a common interface."""
        if self.duration == 0.0:
            return ()
        return _for_sensors(self.sensors, [(self.start, self.start + self.duration)])


@dataclass(frozen=True)
class PeriodicFlicker:
    """Regular on/off flicker: an outage of off_time at the start of every period.

    Window i is [start + phase + i * period, start + phase + i * period + off_time),
    clipped to the span. With phase 0 and a period that is a multiple of the radar scan
    period the windows are aligned with the radar scan grid, so an off time shorter than
    that period loses exactly one scan; a phase shifts them against the grid.

    Attributes:
        sensors: Flickering sensors, a tuple of names from SENSORS.
        start: Start of the span in seconds (>= 0).
        span_length: Length of the span in seconds (> 0).
        period: Time between window starts in seconds (> 0).
        off_time: Length of each outage in seconds, in (0, period].
        phase: Offset of the first window from the span start in seconds, in [0, period).
    """

    sensors: tuple[str, ...]
    start: float
    span_length: float
    period: float
    off_time: float
    phase: float = 0.0

    def __post_init__(self) -> None:
        _check_sensors(self.sensors)
        if self.start < 0.0 or self.span_length <= 0.0 or self.period <= 0.0:
            raise ValueError("need start >= 0, span_length > 0 and period > 0")
        if not 0.0 < self.off_time <= self.period:
            raise ValueError(f"off_time must be in (0, period], got {self.off_time}")
        if not 0.0 <= self.phase < self.period:
            raise ValueError(f"phase must be in [0, period), got {self.phase}")

    @property
    def span(self) -> tuple[float, float]:
        """Interval (start, end) in seconds that the outage metrics refer to."""
        return self.start, self.start + self.span_length

    def windows(
        self, dt: float, rng: np.random.Generator | None = None
    ) -> tuple[DropoutWindow, ...]:
        """The flicker windows. No random draws."""
        span_end = self.start + self.span_length
        intervals = []
        i = 0
        while (first := self.start + self.phase + i * self.period) < span_end:
            intervals.append((first, min(first + self.off_time, span_end)))
            i += 1
        return _for_sensors(self.sensors, intervals)


@dataclass(frozen=True)
class MarkovBursts:
    """Random bursts: a two-state (on / off) Markov chain on the simulation steps.

    With step probabilities q (off -> on) and p (on -> off) the chain has the stationary
    off fraction p / (p + q) and geometric burst lengths of mean dt / q. Given the off
    fraction pi and the mean burst length, q = dt / mean_burst and p = q * pi / (1 - pi).
    The first step is drawn from the stationary law. Bursts cut by the span end are
    shorter than the chain would make them.

    The chain reads one fixed-shape array of uniforms (one per span step) whatever the
    off fraction, like the detection uniforms of simulate_mtt, so the bursts of different
    off fractions are positively coupled and a sweep over pi compares like with like.

    Attributes:
        sensors: Affected sensors; all of them share one burst schedule.
        start: Start of the span in seconds (>= 0).
        span_length: Length of the span in seconds (> 0).
        off_fraction: Stationary share of silent time, in [0, 1).
        mean_burst: Mean burst length in seconds (>= dt, checked when windows are drawn).
    """

    sensors: tuple[str, ...]
    start: float
    span_length: float
    off_fraction: float
    mean_burst: float

    def __post_init__(self) -> None:
        _check_sensors(self.sensors)
        if self.start < 0.0 or self.span_length <= 0.0:
            raise ValueError("need start >= 0 and span_length > 0")
        if not 0.0 <= self.off_fraction < 1.0:
            raise ValueError(f"off_fraction must be in [0, 1), got {self.off_fraction}")
        if self.mean_burst <= 0.0:
            raise ValueError(f"mean_burst must be > 0, got {self.mean_burst}")

    @property
    def span(self) -> tuple[float, float]:
        """Interval (start, end) in seconds that the outage metrics refer to."""
        return self.start, self.start + self.span_length

    def transition_probabilities(self, dt: float) -> tuple[float, float]:
        """Step probabilities (p, q): on -> off and off -> on.

        Raises:
            ValueError: If the burst length is shorter than a step or the off fraction is
                too high for it (p would exceed 1).
        """
        q = dt / self.mean_burst
        if q > 1.0:
            raise ValueError(f"mean_burst must be >= dt, got {self.mean_burst} < {dt}")
        p = q * self.off_fraction / (1.0 - self.off_fraction)
        if p > 1.0:
            raise ValueError(
                f"off_fraction {self.off_fraction} is too high for mean_burst {self.mean_burst} "
                f"at dt {dt} (on -> off probability {p:.3g} > 1)"
            )
        return p, q

    def windows(
        self, dt: float, rng: np.random.Generator | None = None
    ) -> tuple[DropoutWindow, ...]:
        """Draw the burst windows.

        Args:
            dt: Time step in seconds.
            rng: Generator of the "dropout" stream; required.
        """
        if rng is None:
            raise ValueError("MarkovBursts needs a random generator")
        p, q = self.transition_probabilities(dt)
        first = step_index(self.start, dt)
        n_steps = step_index(self.start + self.span_length, dt) - first
        if n_steps <= 0:
            return ()
        uniforms = rng.random(n_steps)

        off = np.zeros(n_steps, dtype=bool)
        state = bool(uniforms[0] < self.off_fraction)
        off[0] = state
        for i in range(1, n_steps):
            state = not (uniforms[i] < q) if state else bool(uniforms[i] < p)
            off[i] = state

        edges = np.diff(np.concatenate([[0], off.astype(int), [0]]))
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        return _for_sensors(
            self.sensors,
            [((first + a) * dt, (first + b) * dt) for a, b in zip(starts, ends, strict=True)],
        )


DropoutSpec = SingleOutage | PeriodicFlicker | MarkovBursts


class DroppedSimulation(NamedTuple):
    """A simulation with its outages applied.

    Attributes:
        sim: The simulation whose scans at silent steps are empty.
        radar_down: Shape (n_points,) True at every step in which the radar is silent,
            scheduled scan step or not.
        camera_down: Shape (n_points,) True at every step in which the camera is silent.
    """

    sim: MttSimulation
    radar_down: np.ndarray
    camera_down: np.ndarray


def _empty_scan(sensor: str) -> Scan:
    return Scan(np.zeros((0, SCAN_DIMENSIONS[sensor])), np.zeros(0, dtype=int))


def apply_dropout(
    sim: MttSimulation, windows: Sequence[DropoutWindow], dt: float
) -> DroppedSimulation:
    """Silence the sensors in their windows: scans at silent steps become empty.

    A silent sensor delivers no detections and no clutter. Radar steps without a scheduled
    scan stay None. Truth and every scan at a step in which the sensor is up are the same
    data as before.

    Args:
        sim: Simulation with full scans.
        windows: Dropout windows.
        dt: Time step in seconds.
    """
    n_points = sim.truth.shape[1]
    radar_down = down_steps(windows, "radar", dt, n_points)
    camera_down = down_steps(windows, "camera", dt, n_points)
    radar_scans = [
        scan if scan is None or not down else _empty_scan("radar")
        for scan, down in zip(sim.radar_scans, radar_down, strict=True)
    ]
    camera_scans = [
        _empty_scan("camera") if down else scan
        for scan, down in zip(sim.camera_scans, camera_down, strict=True)
    ]
    return DroppedSimulation(
        sim._replace(radar_scans=radar_scans, camera_scans=camera_scans), radar_down, camera_down
    )


def end_targets(
    sim: MttSimulation, end_times: Mapping[int, float], dt: float
) -> tuple[MttSimulation, np.ndarray]:
    """Make targets cease to exist: remove their detections from their end step on.

    The truth array is unchanged (the targets keep their simulated states); the returned
    mask says which targets exist at each step. Clutter is kept, and nothing is drawn, so
    all random draws stay the same.

    Args:
        sim: Simulation with full scans.
        end_times: Target index -> first time in seconds at which the target no longer exists.
        dt: Time step in seconds.

    Returns:
        The simulation without the later detections and the shape (n_targets, n_points)
        boolean mask of existing targets.

    Raises:
        ValueError: On an unknown target index or a negative end time.
    """
    n_targets, n_points = sim.truth.shape[:2]
    alive = np.ones((n_targets, n_points), dtype=bool)
    end_steps = {}
    for target, time in end_times.items():
        if not 0 <= target < n_targets:
            raise ValueError(f"unknown target {target}, the simulation has {n_targets}")
        if time < 0.0:
            raise ValueError(f"end time must be >= 0, got {time}")
        end_steps[target] = step_index(time, dt)
        alive[target, end_steps[target] :] = False

    def trimmed(scan: Scan | None, k: int) -> Scan | None:
        if scan is None:
            return None
        gone = [t for t, end in end_steps.items() if k >= end]
        keep = ~np.isin(scan.origin, gone)
        return Scan(scan.z[keep], scan.origin[keep])

    return (
        sim._replace(
            radar_scans=[trimmed(s, k) for k, s in enumerate(sim.radar_scans)],
            camera_scans=[trimmed(s, k) for k, s in enumerate(sim.camera_scans)],
        ),
        alive,
    )
