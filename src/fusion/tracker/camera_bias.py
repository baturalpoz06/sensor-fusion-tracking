"""Global estimate of the camera's constant bearing bias, from simultaneous radar-camera pairs.

A constant bearing bias rotates the camera frame, and a rotated constant-velocity or
coordinated-turn trajectory is still of that kind: target kinematics carry no information on the
bias (for a camera at the radar and no time offset). The only information is the difference of
the two sensors' bearings of the same target at the same instant,

    delta = wrap(z_camera - z_radar) = b + v_camera - v_radar,

whose variance is the sum of the two bearing variances. It does not depend on any track
estimate, so using it involves no double counting and no feedback, and target motion cancels in
it, so a maneuver cannot be read as bias. One scalar estimate is shared by all tracks.

Three application rules share the pairs:
    "spike_slab": two hypotheses, H0 "b = 0" and H1 "b ~ N(0, sigma_slab^2), random walk". The
        posterior probability P1 of H1 follows from the accumulated log-likelihood ratio (log
        domain), and the applied estimate is the mixture mean P1 * b1 with its mixture variance.
    "always": the H1 filter alone, applied with full weight.
    "oracle": a fixed known bias with zero variance (an upper bound, for ablations).

All angles are in radians.
"""

import math
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from scipy.special import expit

from fusion.angles import wrap_angle
from fusion.association.gating import gate_threshold

BIAS_MODES = ("spike_slab", "always", "oracle")
VARIANCE_FLOOR = 1e-18  # rad^2, keeps the scalar filter variance positive


@dataclass(frozen=True)
class CameraBiasConfig:
    """Parameters of the bias estimate.

    Attributes:
        mode: "spike_slab", "always" or "oracle" (see the module docstring).
        prior_h1: Prior probability of H1 in (0, 1), used by "spike_slab".
        sigma_slab: Prior std of the bias under H1 in radians (the mounting tolerance of an
            uncalibrated camera); also the initial std of the "always" filter.
        process_noise: Random walk variance of the bias per second in rad^2/s.
        gate_probability: Probability mass of the outlier gate on a pair.
        oracle_bias: The fixed bias applied by "oracle", in radians.
    """

    mode: str = "spike_slab"
    prior_h1: float = 0.5
    sigma_slab: float = float(np.deg2rad(1.0))
    process_noise: float = float(np.deg2rad(0.1) ** 2 / 600.0)
    gate_probability: float = 0.99
    oracle_bias: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in BIAS_MODES:
            raise ValueError(f"mode must be one of {BIAS_MODES}, got {self.mode!r}")
        if not 0.0 < self.prior_h1 < 1.0:
            raise ValueError(f"prior_h1 must be in (0, 1), got {self.prior_h1}")
        if not (math.isfinite(self.sigma_slab) and self.sigma_slab > 0.0):
            raise ValueError(f"sigma_slab must be finite and > 0, got {self.sigma_slab}")
        if not (math.isfinite(self.process_noise) and self.process_noise >= 0.0):
            raise ValueError(f"process_noise must be finite and >= 0, got {self.process_noise}")
        if not 0.0 < self.gate_probability < 1.0:
            raise ValueError(f"gate_probability must be in (0, 1), got {self.gate_probability}")
        if not math.isfinite(self.oracle_bias):
            raise ValueError(f"oracle_bias must be finite, got {self.oracle_bias}")


class BiasLogEntry(NamedTuple):
    """State of the bias estimate after one step; a diagnostic, never read by the tracker.

    Attributes:
        b1: Mean of the H1 filter in radians.
        sd1: Std of the H1 filter in radians.
        p1: Posterior probability of H1 (1 for "always" and "oracle").
        b_app: Applied estimate in radians.
        sd_app: Std of the applied estimate in radians.
        used: Pairs used so far.
        rejected: Pairs rejected by the outlier gate so far.
    """

    b1: float
    sd1: float
    p1: float
    b_app: float
    sd_app: float
    used: int
    rejected: int


class CameraBiasEstimator:
    """Scalar bias estimate fed with radar-camera bearing differences.

    Attributes:
        used: Pairs that updated the estimate.
        rejected: Pairs rejected by the outlier gate.
    """

    def __init__(
        self, config: CameraBiasConfig, radar_bearing_std: float, camera_bearing_std: float
    ) -> None:
        """Set up the estimate from the believed bearing noise of the two sensors.

        Args:
            config: Estimator parameters.
            radar_bearing_std: Believed radar bearing std in radians.
            camera_bearing_std: Believed camera bearing std in radians.
        """
        self.config = config
        self._pair_variance = radar_bearing_std**2 + camera_bearing_std**2
        # Fixed gate centered on zero, wide enough for the slab, identical for both hypotheses.
        self._gate_half_width = math.sqrt(
            gate_threshold(1, config.gate_probability)
            * (self._pair_variance + config.sigma_slab**2)
        )
        self._b1 = 0.0
        self._var1 = config.sigma_slab**2
        self._log_odds = math.log(config.prior_h1 / (1.0 - config.prior_h1))
        self.used = 0
        self.rejected = 0

    @property
    def pair_variance(self) -> float:
        """Variance of one radar-camera bearing difference, in rad^2."""
        return self._pair_variance

    @property
    def gate_half_width(self) -> float:
        """Half width of the outlier gate on a pair, in radians."""
        return self._gate_half_width

    @property
    def p1(self) -> float:
        """Posterior probability of H1."""
        if self.config.mode == "spike_slab":
            return float(expit(self._log_odds))
        return 1.0

    @property
    def estimate(self) -> tuple[float, float]:
        """Applied bias and its variance: (b_app, var_app) in radians and rad^2.

        spike_slab: b_app = P1 b1, var_app = P1 var1 + P1 (1 - P1) b1^2 (mixture of H0 with
        b = 0 and H1 with N(b1, var1)). always: (b1, var1). oracle: (oracle_bias, 0).
        """
        if self.config.mode == "oracle":
            return self.config.oracle_bias, 0.0
        if self.config.mode == "always":
            return self._b1, self._var1
        p1 = self.p1
        return p1 * self._b1, p1 * self._var1 + p1 * (1.0 - p1) * self._b1**2

    def predict(self, dt: float) -> None:
        """Let the H1 random walk drift for dt seconds."""
        if self.config.mode == "oracle":
            return
        self._var1 += self.config.process_noise * dt

    def update(self, deltas: list[float]) -> None:
        """Take radar-camera bearing differences (camera minus radar, in radians), in order.

        A difference outside the gate is counted as rejected and used by neither hypothesis.
        """
        if self.config.mode == "oracle":
            return
        for raw in deltas:
            delta = float(wrap_angle(raw))
            if not math.isfinite(delta) or abs(delta) > self._gate_half_width:
                self.rejected += 1
                continue
            self.used += 1
            innovation_var = self._var1 + self._pair_variance
            if self.config.mode == "spike_slab":
                log_h1 = -0.5 * (
                    (delta - self._b1) ** 2 / innovation_var + math.log(innovation_var)
                )
                log_h0 = -0.5 * (delta**2 / self._pair_variance + math.log(self._pair_variance))
                self._log_odds += log_h1 - log_h0
            gain = self._var1 / innovation_var
            self._b1 += gain * (delta - self._b1)
            self._var1 = max((1.0 - gain) * self._var1, VARIANCE_FLOOR)

    def log_entry(self) -> BiasLogEntry:
        """The current state as a log entry."""
        b_app, var_app = self.estimate
        return BiasLogEntry(
            self._b1,
            math.sqrt(self._var1),
            self.p1,
            b_app,
            math.sqrt(var_app),
            self.used,
            self.rejected,
        )
