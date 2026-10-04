"""Phase 6 regression guard: results must stay bitwise identical to the recorded fixture.

The fixture tests/data/phase6_golden.json was recorded at commit 0067cac by
scripts/record_phase6_golden.py, before any Phase 7 change.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import scipy

from fusion.mtt_simulation import RNG_STREAMS, make_mtt_rngs

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = ROOT / "tests" / "data" / "phase6_golden.json"
RECORDER_PATH = ROOT / "scripts" / "record_phase6_golden.py"

# The generators Phase 6 derived from a seed, in order. New streams may only be appended.
PHASE6_STREAMS = (
    "trajectory",
    "radar_noise",
    "radar_detection",
    "radar_clutter",
    "radar_shuffle",
    "camera_noise",
    "camera_detection",
    "camera_clutter",
    "camera_shuffle",
)


def load_recorder():
    spec = importlib.util.spec_from_file_location("record_phase6_golden", RECORDER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def golden() -> dict:
    data = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    if (data["numpy"], data["scipy"]) != (np.__version__, scipy.__version__):
        pytest.skip(
            f"fixture recorded with numpy {data['numpy']} / scipy {data['scipy']}, "
            f"running numpy {np.__version__} / scipy {scipy.__version__}: exact floats may differ"
        )
    return data


@pytest.fixture(scope="module")
def recorder():
    return load_recorder()


def test_every_recorded_trial_is_reproduced_bitwise(golden, recorder):
    """Fails if any code shared with Phase 6 (tracker, metrics, simulation, RNG) changes a bit."""
    current = recorder.compute_golden()
    assert current["metric_fields"] == golden["metric_fields"]
    assert set(current["trials"]) == set(golden["trials"])
    for key, expected in golden["trials"].items():
        assert current["trials"][key]["metrics"] == expected["metrics"], key
        assert current["trials"][key]["history_sha256"] == expected["history_sha256"], key


@pytest.mark.parametrize("workers", [1, 2])
def test_sweep_and_fusion_comparison_are_reproduced_bitwise_with_any_worker_count(
    golden, recorder, workers
):
    """Fails if the sweep machinery (_run_points, sweep, compare_fusion) changes a result."""
    current = recorder.compute_sweeps(workers)
    assert current["sweep_clutter_rate"] == golden["sweep_clutter_rate"]
    assert current["compare_fusion"] == golden["compare_fusion"]


def test_fixture_covers_the_stressful_paths(golden):
    """Fails if the fixture is regenerated without clutter, low Pd or the radar-only tracker."""
    cases = {key.split("/")[1] for key in golden["trials"]}
    assert cases == {"clean", "clutter5", "clutter5_radar_only", "low_pd"}
    assert {key.split("/")[0] for key in golden["trials"]} == {"crossing", "separated"}


def test_phase_6_random_streams_keep_their_draws_when_streams_are_appended():
    """Fails if a new stream is inserted before the Phase 6 ones instead of appended.

    SeedSequence(seed).spawn(n) gives child i the same key for any n, so the first nine
    generators must equal the ones spawned directly for nine streams.
    """
    assert tuple(RNG_STREAMS[: len(PHASE6_STREAMS)]) == PHASE6_STREAMS
    for seed in (0, 1, 12345):
        reference = [
            np.random.default_rng(child)
            for child in np.random.SeedSequence(seed).spawn(len(PHASE6_STREAMS))
        ]
        rngs = make_mtt_rngs(seed)
        for name, expected in zip(PHASE6_STREAMS, reference, strict=True):
            np.testing.assert_array_equal(rngs[name].random(8), expected.random(8))
