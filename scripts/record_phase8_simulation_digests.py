"""Record simulation-only digests of disturbed scenes, for Phase 8 before / after comparisons.

Usage:
    python scripts/record_phase8_simulation_digests.py [--out PATH]

The fixture was recorded right after the truth disturbances were wired into the simulator (before
any Phase 8b change). For a few disturbed scenes it stores a SHA-256 digest of the generated truth
and of every scan (measurements and origins), plus two checksums as exact float hex strings (the
final truth states and the sum of all camera measurements). It stores no tracker result: its
purpose is to guarantee that a later change of the tracker is compared on exactly the same data,
and that the disturbances themselves keep generating the same world.

The regression test recomputes everything with compute_digests() and compares it (the digests
bitwise where the numpy, scipy and platform of the recording match, the checksums with a tight
tolerance elsewhere).
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import scipy

from fusion.maneuvers import Acceleration, RandomManeuvers, TruthDisturbance, Turn
from fusion.mtt_experiment import SCENARIOS, MttConfig
from fusion.mtt_simulation import make_mtt_rngs, simulate_mtt

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "tests" / "data" / "phase8_simulation_digests.json"


def _load_phase7_recorder():
    path = ROOT / "scripts" / "record_phase7_golden.py"
    spec = importlib.util.spec_from_file_location("record_phase7_golden", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PHASE7 = _load_phase7_recorder()

SEEDS = (0, 1, 2)
CONFIG = MttConfig()  # 60 s, clutter 5 per scan and sensor, Pd 0.9: the Phase 6 defaults
DEG = float(np.pi / 180.0)


def _turns(n_targets: int, rate_deg: float) -> tuple:
    return tuple((i, Turn(15.0, 25.0, rate_deg * DEG)) for i in range(n_targets))


# Scene name -> (scenario, disturbance). Together they reach every disturbance hook.
SCENES = {
    "separated/bias_0.5deg": ("separated", TruthDisturbance(camera_bias=0.5 * DEG)),
    "crossing/offset_50ms": ("crossing", TruthDisturbance(camera_time_offset=0.05)),
    "separated/offset_25ms": ("separated", TruthDisturbance(camera_time_offset=0.025)),
    "crossing/lever_20m_x": ("crossing", TruthDisturbance(camera_position=(20.0, 0.0))),
    "separated/lever_20m_y": ("separated", TruthDisturbance(camera_position=(0.0, 20.0))),
    "separated/turn_10deg": ("separated", TruthDisturbance(maneuvers=_turns(3, 10.0))),
    "crossing/turn_20deg": ("crossing", TruthDisturbance(maneuvers=_turns(4, 20.0))),
    "separated/acceleration_2": (
        "separated",
        TruthDisturbance(maneuvers=tuple((i, Acceleration(15.0, 18.0, 2.0)) for i in range(3))),
    ),
    "separated/random_10deg": (
        "separated",
        TruthDisturbance(random_maneuvers=RandomManeuvers(10.0, 5.0, 10.0 * DEG)),
    ),
    "crossing/combined": (
        "crossing",
        TruthDisturbance(
            camera_position=(5.0, 5.0),
            camera_bias=0.25 * DEG,
            camera_time_offset=0.025,
            maneuvers=((0, Turn(15.0, 25.0, 5.0 * DEG)),),
        ),
    ),
}


def camera_checksum(sim) -> float:
    """Sum of every camera measurement of the run (clutter included)."""
    return float(sum(float(scan.z.sum()) for scan in sim.camera_scans))


def compute_digests() -> dict:
    """Recompute every recorded quantity with the current code."""
    scenes = {}
    for name, (scenario, disturbance) in SCENES.items():
        for seed in SEEDS:
            sim = simulate_mtt(SCENARIOS[scenario], CONFIG, make_mtt_rngs(seed), disturbance)
            scenes[f"{name}/{seed}"] = {
                "simulation_sha256": PHASE7.simulation_digest(sim),
                "truth_end": [float(v).hex() for v in sim.truth[:, -1, :].ravel()],
                "camera_sum": [camera_checksum(sim).hex()],
            }
    return {
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": sys.platform,
        "scenes": scenes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    digests = compute_digests()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(digests, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.out} ({len(digests['scenes'])} scenes)")


if __name__ == "__main__":
    main()
