"""Record the Phase 7 golden fixture: exact outage results used as a regression guard.

Usage:
    python scripts/record_phase7_golden.py [--out tests/data/phase7_golden.json]

The fixture was recorded at commit 496d659, after Phase 7 and before any Phase 8 change. For a
grid of outage scenes it stores the OutageMetrics of run_outage_trial as exact float hex
strings, a SHA-256 digest of every step's tracks, and two digests of the simulated data: the
generated scene (truth and every scan, before any masking) and the scene the tracker was given
(after vanishing targets and outages were applied). It also stores the simulation digests of
the Phase 6 golden grid and a small outage sweep. The data digests pin the simulator on their
own, so a change to the generated data is told apart from a change to the tracker.

The regression test recomputes everything with compute_golden() and compare_sweep() and
compares it (exactly where the numpy, scipy and platform of the recording match, with a tight
tolerance elsewhere, without the digests).
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

from fusion.dropout import (
    MarkovBursts,
    PeriodicFlicker,
    SingleOutage,
    apply_dropout,
    end_targets,
)
from fusion.mtt_experiment import SCENARIOS
from fusion.mtt_simulation import MttSimulation, make_mtt_rngs, simulate_mtt
from fusion.outage_experiment import (
    OUTAGE_SCENARIOS,
    SENSOR_SETS,
    OutageConfig,
    duration_points,
    outage_sweep,
    run_outage_trial,
)
from fusion.outage_metrics import OutageMetrics
from fusion.sensors.camera import CameraModel
from fusion.sensors.radar import RadarModel
from fusion.tracker.multi_target import run_multi_target_tracking

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "tests" / "data" / "phase7_golden.json"


def _load_phase6_recorder():
    path = ROOT / "scripts" / "record_phase6_golden.py"
    spec = importlib.util.spec_from_file_location("record_phase6_golden", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PHASE6 = _load_phase6_recorder()

SEEDS = (0, 1)
RADAR = SENSOR_SETS["radar"]
BLACKOUT = SENSOR_SETS["blackout"]
START = 10.0
# Phase 6 scene defaults (clutter 5 per scan and sensor, Pd 0.9); cases named clutter3 lower it.
BASE = OutageConfig(duration=30.0, burn_in=5.0, after_window=10.0)
# Each case overrides the base configuration; together they reach every outage code path:
# unaware and aware policies, dropped and frozen tentatives, all three dropout generators
# (the random bursts read the "dropout" stream), coast deletions and a vanishing target.
CASES = {
    "radar4_unaware": {"dropout": SingleOutage(RADAR, START, 4.0)},
    "radar4_aware": {"dropout": SingleOutage(RADAR, START, 4.0), "outage_policy": "aware"},
    "radar8_aware_clutter3": {
        "dropout": SingleOutage(RADAR, START, 8.0),
        "outage_policy": "aware",
        "radar_clutter_rate": 3.0,
        "camera_clutter_rate": 3.0,
    },
    "blackout4_aware_freeze_clutter3": {
        "dropout": SingleOutage(BLACKOUT, START, 4.0),
        "outage_policy": "aware",
        "aware_tentatives": "freeze",
        "radar_clutter_rate": 3.0,
        "camera_clutter_rate": 3.0,
    },
    "blackout8_aware_coast2": {
        "dropout": SingleOutage(BLACKOUT, START, 8.0),
        "outage_policy": "aware",
        "max_coast_time": 2.0,
    },
    "flicker_aware_clutter3": {
        "dropout": PeriodicFlicker(RADAR, START, 10.0, 4.0, 1.0),
        "outage_policy": "aware",
        "radar_clutter_rate": 3.0,
        "camera_clutter_rate": 3.0,
    },
    "bursts_unaware_clutter3": {
        "dropout": MarkovBursts(RADAR, START, 10.0, 0.3, 2.0),
        "radar_clutter_rate": 3.0,
        "camera_clutter_rate": 3.0,
    },
    "vanish_aware": {
        "dropout": SingleOutage(RADAR, START, 12.0),
        "outage_policy": "aware",
        "max_coast_time": 5.0,
        "vanish": ((0, START + 2.0),),
    },
}
# Scenes of each case: the vanishing case runs only in its own scenario.
CASE_SCENARIOS = {
    case: ("vanishing",) if case.startswith("vanish") else ("crossing", "separated")
    for case in CASES
}
SWEEP_SEEDS = (0, 1)
SWEEP_DURATIONS = (0.0, 4.0)


def case_config(case: str) -> OutageConfig:
    return replace(BASE, **CASES[case])


def simulation_digest(sim: MttSimulation) -> str:
    """SHA-256 over the truth and every scan (measurements and origins; a missing scan marked)."""
    digest = hashlib.sha256()
    truth = np.ascontiguousarray(sim.truth)
    digest.update(f"truth:{truth.shape}".encode())
    digest.update(truth.tobytes())
    for name, scans in (("radar", sim.radar_scans), ("camera", sim.camera_scans)):
        digest.update(f"{name}:{len(scans)}".encode())
        for scan in scans:
            if scan is None:
                digest.update(b"none")
                continue
            z = np.ascontiguousarray(scan.z, dtype=float)
            origin = np.ascontiguousarray(scan.origin, dtype=np.int64)
            digest.update(f"scan:{z.shape}".encode())
            digest.update(z.tobytes())
            digest.update(origin.tobytes())
    return digest.hexdigest()


def run_digests(initial_states: np.ndarray, config: OutageConfig, seed: int) -> dict[str, str]:
    """Data and history digests of one trial, following the steps of run_outage_trial."""
    rngs = make_mtt_rngs(seed)
    sim = simulate_mtt(initial_states, config, rngs)
    generated = simulation_digest(sim)
    vanish = dict(config.vanish)
    if vanish:
        sim, _ = end_targets(sim, vanish, config.dt)
    windows = () if config.dropout is None else config.dropout.windows(config.dt, rngs["dropout"])
    dropped = apply_dropout(sim, windows, config.dt)
    radar_model = RadarModel(config.radar_range_std, config.radar_bearing_std)
    camera_model = CameraModel(config.camera_bearing_std) if config.use_camera else None
    run = run_multi_target_tracking(
        dropped.sim, config.tracker_config(), radar_model, camera_model, dropped.radar_down
    )
    return {
        "simulation_sha256": generated,
        "masked_simulation_sha256": simulation_digest(dropped.sim),
        "history_sha256": PHASE6.history_digest(run.history),
    }


def phase6_simulation_digests() -> dict[str, str]:
    """Digest of the generated scene of every trial of the Phase 6 golden grid."""
    digests = {}
    for scenario in PHASE6.SCENARIO_NAMES:
        for case in PHASE6.CASES:
            config = PHASE6.case_config(case)
            for seed in PHASE6.SEEDS:
                sim = simulate_mtt(SCENARIOS[scenario], config, make_mtt_rngs(seed))
                digests[f"{scenario}/{case}/{seed}"] = simulation_digest(sim)
    return digests


def compute_sweep(workers: int = 1) -> dict[str, list[list[str]]]:
    """A small radar-outage sweep with the aware policy, as exact text."""
    base = replace(case_config("radar8_aware_clutter3"), dropout=None)
    points = duration_points(base, RADAR, START, SWEEP_DURATIONS)
    result = outage_sweep(
        OUTAGE_SCENARIOS["separated"], points, SWEEP_DURATIONS, "outage", SWEEP_SEEDS, workers
    )
    return PHASE6.sweep_hex(result)


def compute_golden() -> dict:
    """Recompute every recorded quantity with the current code (the sweep with one worker)."""
    trials = {}
    for case in CASES:
        config = case_config(case)
        for scenario in CASE_SCENARIOS[case]:
            states = OUTAGE_SCENARIOS[scenario]
            for seed in SEEDS:
                trials[f"{scenario}/{case}/{seed}"] = {
                    "metrics": PHASE6.metrics_hex(run_outage_trial(states, config, seed)),
                    **run_digests(states, config, seed),
                }
    return {
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": sys.platform,
        "metric_fields": list(OutageMetrics._fields),
        "trials": trials,
        "phase6_simulations": phase6_simulation_digests(),
        "outage_sweep": compute_sweep(),
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
