"""Record the Phase 6 golden fixture: exact multi-target results used as a regression guard.

Usage:
    python scripts/record_phase6_golden.py [--out tests/data/phase6_golden.json]

The fixture was recorded at commit 0067cac, before any Phase 7 change. It stores, for a grid
of scenes, the Phase 6 metrics as exact float hex strings and a SHA-256 digest of every
step's track ids, statuses, states and covariances, plus a small sweep and a fused vs
radar-only comparison. The regression test recomputes the same quantities with
compute_golden() and demands exact equality, so any change to a code path shared with
Phase 6 that alters even one bit is caught.
"""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import scipy

from fusion.mtt_experiment import SCENARIOS, MttConfig, compare_fusion, run_mtt_trial, sweep
from fusion.mtt_metrics import MttMetrics
from fusion.mtt_simulation import make_mtt_rngs, simulate_mtt
from fusion.sensors.camera import CameraModel
from fusion.sensors.radar import RadarModel
from fusion.tracker.multi_target import run_multi_target_tracking

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "tests" / "data" / "phase6_golden.json"

SCENARIO_NAMES = ("crossing", "separated")
SEEDS = (0, 1, 2)
DURATION = 30.0
BURN_IN = 5.0
# Each case overrides the base configuration: clutter rates, detection probabilities, camera use.
CASES = {
    "clean": {"radar_clutter_rate": 0.0, "camera_clutter_rate": 0.0},
    "clutter5": {"radar_clutter_rate": 5.0, "camera_clutter_rate": 5.0},
    "clutter5_radar_only": {
        "radar_clutter_rate": 5.0,
        "camera_clutter_rate": 5.0,
        "use_camera": False,
    },
    "low_pd": {
        "radar_pd": 0.6,
        "camera_pd": 0.6,
        "radar_clutter_rate": 2.0,
        "camera_clutter_rate": 2.0,
    },
}
SWEEP_SEEDS = (0, 1)
SWEEP_VALUES = (0.0, 5.0)


def case_config(case: str) -> MttConfig:
    return replace(MttConfig(duration=DURATION, burn_in=BURN_IN), **CASES[case])


def metrics_hex(metrics: MttMetrics) -> list[str]:
    """Exact text form of every metric (floats as hex, NaN as 'nan')."""
    return [float(value).hex() for value in metrics]


def history_digest(history) -> str:
    """SHA-256 over the ids, statuses, states and covariances of every step."""
    digest = hashlib.sha256()
    for step in history:
        digest.update(f"step:{len(step)}".encode())
        for snapshot in step:
            digest.update(f"{snapshot.track_id}:{snapshot.status.value}".encode())
            digest.update(np.ascontiguousarray(snapshot.x).tobytes())
            digest.update(np.ascontiguousarray(snapshot.P).tobytes())
    return digest.hexdigest()


def run_digest(initial_states: np.ndarray, config: MttConfig, seed: int) -> str:
    sim = simulate_mtt(initial_states, config, make_mtt_rngs(seed))
    radar_model = RadarModel(config.radar_range_std, config.radar_bearing_std)
    camera_model = CameraModel(config.camera_bearing_std) if config.use_camera else None
    run = run_multi_target_tracking(sim, config.tracker_config(), radar_model, camera_model)
    return history_digest(run.history)


def sweep_hex(result) -> dict[str, list[list[str]]]:
    return {
        name: [[float(v).hex() for v in row] for row in values]
        for name, values in result.metrics.items()
    }


def compute_sweeps(workers: int = 1) -> dict[str, dict[str, list[list[str]]]]:
    """A small clutter sweep and a fused vs radar-only comparison, as exact text."""
    config = replace(case_config("clutter5"), duration=20.0, burn_in=2.0)
    states = SCENARIOS["separated"]
    return {
        "sweep_clutter_rate": sweep_hex(
            sweep(states, config, "clutter_rate", SWEEP_VALUES, SWEEP_SEEDS, workers)
        ),
        "compare_fusion": sweep_hex(compare_fusion(states, config, SWEEP_SEEDS, workers)),
    }


def compute_golden() -> dict:
    """Recompute every recorded quantity with the current code."""
    trials = {}
    for scenario in SCENARIO_NAMES:
        for case in CASES:
            config = case_config(case)
            for seed in SEEDS:
                key = f"{scenario}/{case}/{seed}"
                states = SCENARIOS[scenario]
                trials[key] = {
                    "metrics": metrics_hex(run_mtt_trial(states, config, seed)),
                    "history_sha256": run_digest(states, config, seed),
                }
    return {
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "metric_fields": list(MttMetrics._fields),
        "trials": trials,
        **compute_sweeps(),
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
