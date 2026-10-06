"""Record the Phase 8b golden fixture: exact results of the IMM and bias features as a guard.

Usage:
    python scripts/record_phase8b_golden.py [--out tests/data/phase8b_golden.json]

For two scenes (separated and crossing, clutter 5, 30 s) with a camera bias of 0.5 deg and a
10 deg/s turn of every target, and two seeds from the test range (2000 and up), it runs nine
tracker variants with explicit parameters (they do not depend on the tuned values frozen in
fusion.improvement_experiment): the EKF, the EKF with a higher process noise, IMM-A, IMM-B, the
bias estimate in three forms (spike and slab, always applied, the true bias) and both IMMs with
the spike-and-slab bias estimate. It stores the Robustness metrics as exact float hex strings, a
SHA-256 digest of every step's tracks (as reported, so bias-corrected where the estimate is on),
the bias estimate at a few steps, a digest of the mode probabilities, and a digest of the
simulated data. The numpy and scipy versions and the platform of the recording are stored too.

The regression test recomputes everything with compute_golden() and compares it (bitwise where
the environment matches, with a tight tolerance and without the digests elsewhere).
"""

import argparse
import hashlib
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import scipy

from fusion.filters.imm import ModeSpec
from fusion.improvement_experiment import Arm, imm_a_modes, imm_b_modes
from fusion.maneuvers import TruthDisturbance, Turn
from fusion.mtt_experiment import SCENARIOS
from fusion.mtt_simulation import make_mtt_rngs, simulate_mtt
from fusion.robustness_experiment import RobustnessConfig
from fusion.robustness_metrics import FIELDS, evaluate_robustness
from fusion.tracker.camera_bias import CameraBiasConfig
from fusion.tracker.multi_target import run_multi_target_tracking

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "tests" / "data" / "phase8b_golden.json"
DEG = float(np.pi / 180.0)
SEEDS = (2000, 2001)
SCENE_NAMES = ("separated", "crossing")
BIAS = 0.5 * DEG
BIAS_SAMPLE_EVERY = 50  # steps between recorded bias estimates (and the last step)
CONFIG = RobustnessConfig(
    duration=30.0, burn_in=5.0, radar_clutter_rate=5.0, camera_clutter_rate=5.0
)
SPIKE_SLAB = CameraBiasConfig(mode="spike_slab", prior_h1=0.5)


def _load_phase7_recorder():
    path = ROOT / "scripts" / "record_phase7_golden.py"
    spec = importlib.util.spec_from_file_location("record_phase7_golden", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PHASE7 = _load_phase7_recorder()

ARMS = {
    "ekf": Arm("ekf"),
    "ekf_high_q_2": Arm("ekf_high_q_2", accel_std=2.0),
    "imm_a_3": Arm("imm_a_3", modes=imm_a_modes(3.0)),
    "imm_b_10_1": Arm("imm_b_10_1", modes=imm_b_modes(10.0, 1.0)),
    "ekf_bias": Arm("ekf_bias", camera_bias=SPIKE_SLAB),
    "ekf_always": Arm("ekf_always", camera_bias=CameraBiasConfig(mode="always")),
    "ekf_oracle": Arm(
        "ekf_oracle", camera_bias=CameraBiasConfig(mode="oracle", oracle_bias=BIAS)
    ),
    "imm_a_bias": Arm("imm_a_bias", modes=imm_a_modes(3.0), camera_bias=SPIKE_SLAB),
    "imm_b_bias": Arm("imm_b_bias", modes=imm_b_modes(10.0, 1.0), camera_bias=SPIKE_SLAB),
}
assert all(isinstance(spec, ModeSpec) for arm in ARMS.values() for spec in arm.modes or ())


def scene_config(scene: str) -> RobustnessConfig:
    """The configuration of a scene: a 0.5 deg camera bias and a 10 deg/s turn of every target."""
    n = len(SCENARIOS[scene])
    turns = tuple((i, Turn(15.0, 25.0, 10.0 * DEG)) for i in range(n))
    truth = TruthDisturbance(camera_bias=BIAS, maneuvers=turns)
    return replace(CONFIG, truth=truth)


def mode_digest(mode_log) -> str:
    """SHA-256 over the mode probabilities of every IMM track at every step."""
    digest = hashlib.sha256()
    for step in mode_log:
        digest.update(f"step:{len(step)}".encode())
        for entry in step:
            digest.update(f"{entry.track_id}".encode())
            digest.update(np.asarray(entry.mu, dtype=float).tobytes())
    return digest.hexdigest()


def trial_record(sim, config: RobustnessConfig, arm: Arm) -> dict:
    """Recorded quantities of one arm on one simulation."""
    base, radar_model, camera_model = config.tracker_setup()
    tracker_config = arm.tracker_config(base, config.radar_every)
    run = run_multi_target_tracking(sim, tracker_config, radar_model, camera_model)
    metrics = evaluate_robustness(
        sim.truth,
        run.history,
        run.tracker.camera_log,
        sim.camera_scans,
        dt=config.dt,
        fov=config.fov,
        max_distance=config.match_distance,
        wide_distance=config.diagnostic_match_distance,
        burn_in_steps=config.burn_in_steps,
        focus_window=(15.0, config.duration),
    )
    metrics = metrics._replace(run_births_per_scan=run.tracker.births / run.tracker.radar_scans)
    record = {
        "metrics": [float(value).hex() for value in metrics],
        "history_sha256": PHASE7.PHASE6.history_digest(run.history),
    }
    log = run.tracker.bias_log
    if log:
        steps = sorted({*range(0, len(log), BIAS_SAMPLE_EVERY), len(log) - 1})
        record["bias_samples"] = [float(log[k].b_app).hex() for k in steps]
        record["bias_p1_last"] = [float(log[-1].p1).hex()]
    if run.tracker.mode_log:
        record["mode_sha256"] = mode_digest(run.tracker.mode_log)
    return record


def compute_golden() -> dict:
    """Recompute every recorded quantity with the current code."""
    trials, simulations = {}, {}
    for scene in SCENE_NAMES:
        config = scene_config(scene)
        for seed in SEEDS:
            sim = simulate_mtt(SCENARIOS[scene], config, make_mtt_rngs(seed), config.truth)
            simulations[f"{scene}/{seed}"] = PHASE7.simulation_digest(sim)
            for name, arm in ARMS.items():
                trials[f"{scene}/{name}/{seed}"] = trial_record(sim, config, arm)
    return {
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": sys.platform,
        "metric_fields": list(FIELDS),
        "trials": trials,
        "simulations": simulations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    golden = compute_golden()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(golden, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.out} ({len(golden['trials'])} trials)")


if __name__ == "__main__":
    main()
