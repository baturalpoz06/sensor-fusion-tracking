"""Phase 7 regression guard: outage results and simulated data must stay as recorded.

The fixture tests/data/phase7_golden.json was recorded at commit 496d659 by
scripts/record_phase7_golden.py, before any Phase 8 change. As for Phase 6, the comparison is
bitwise where the recording environment (numpy, scipy and platform) matches, digests included,
and within a tight tolerance elsewhere without the digests; the test never skips.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy

from fusion.mtt_simulation import RNG_STREAMS
from tests.test_phase6_regression import PHASE6_STREAMS, assert_same

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = ROOT / "tests" / "data" / "phase7_golden.json"
RECORDER_PATH = ROOT / "scripts" / "record_phase7_golden.py"

# The generators Phase 7 derived from a seed, in order. New streams may only be appended.
PHASE7_STREAMS = (*PHASE6_STREAMS, "dropout")
DIGESTS = ("simulation_sha256", "masked_simulation_sha256", "history_sha256")


def load_recorder():
    spec = importlib.util.spec_from_file_location("record_phase7_golden", RECORDER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def golden() -> dict:
    data = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    recorded = (data["numpy"], data["scipy"], data["platform"])
    data["exact"] = recorded == (np.__version__, scipy.__version__, sys.platform)
    return data


@pytest.fixture(scope="module")
def recorder():
    return load_recorder()


@pytest.fixture(scope="module")
def current(recorder) -> dict:
    return recorder.compute_golden()


def test_every_recorded_outage_trial_is_reproduced(golden, current):
    """Fails if any code shared with Phase 7 (outage masking, policies, metrics) changes a result.

    Bitwise, with the data and history digests, where the recording environment matches.
    """
    assert current["metric_fields"] == golden["metric_fields"]
    assert set(current["trials"]) == set(golden["trials"])
    for key, expected in golden["trials"].items():
        assert_same(current["trials"][key]["metrics"], expected["metrics"], golden["exact"], key)
        if golden["exact"]:
            for digest in DIGESTS:
                assert current["trials"][key][digest] == expected[digest], f"{key} {digest}"


def test_the_simulated_data_of_the_phase_6_grid_is_unchanged(golden, current):
    """Fails if the simulator generates different truth or scans for the Phase 6 scenes.

    Pins the data on its own, so a simulator change is told apart from a tracker change.
    Only compared where the recording environment matches (the digests are bitwise).
    """
    assert set(current["phase6_simulations"]) == set(golden["phase6_simulations"])
    if golden["exact"]:
        assert current["phase6_simulations"] == golden["phase6_simulations"]


@pytest.mark.parametrize("workers", [1, 2])
def test_the_outage_sweep_is_reproduced_with_any_worker_count(golden, recorder, workers):
    """Fails if the outage sweep machinery changes a result or depends on the worker count."""
    current = recorder.compute_sweep(workers)
    assert set(current) == set(golden["outage_sweep"])
    for metric, rows in golden["outage_sweep"].items():
        assert_same(current[metric], rows, golden["exact"], f"outage_sweep/{metric}")


def test_fixture_covers_the_outage_code_paths(golden):
    """Fails if the fixture is regenerated without the cases that reach each outage path.

    Coast deletions, dropped tentatives, frozen tentatives (no drops) and a ghost lifetime
    must all occur somewhere in the recorded trials.
    """
    fields = golden["metric_fields"]
    metrics = {
        key: dict(zip(fields, (float.fromhex(v) for v in trial["metrics"]), strict=True))
        for key, trial in golden["trials"].items()
    }
    assert any(m["coast_deletions"] > 0 for m in metrics.values())
    assert any(m["tentative_drops"] > 0 for m in metrics.values())
    assert all(m["tentative_drops"] == 0 for key, m in metrics.items() if "freeze" in key)
    assert any(np.isfinite(m["ghost_lifetime"]) for m in metrics.values())
    scenarios = {key.split("/")[0] for key in metrics}
    assert scenarios == {"crossing", "separated", "vanishing"}


def test_phase_7_random_streams_keep_their_draws_when_streams_are_appended():
    """Fails if a new stream is inserted before the Phase 7 ones instead of appended."""
    assert tuple(RNG_STREAMS[: len(PHASE7_STREAMS)]) == PHASE7_STREAMS
