"""Guard of the disturbed simulations: the same seed must keep generating the same world.

The fixture tests/data/phase8_simulation_digests.json was recorded by
scripts/record_phase8_simulation_digests.py right after the truth disturbances were wired in,
before any tracker change of Phase 8b. It holds simulation data only (truth and scans), so a
before / after comparison of a tracker change is guaranteed to run on the same data. The digests
are compared bitwise where the recording environment (numpy, scipy, platform) matches; the
checksums are compared with a tight tolerance elsewhere.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy

from tests.test_phase6_regression import assert_same

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / "tests" / "data" / "phase8_simulation_digests.json"
RECORDER_PATH = ROOT / "scripts" / "record_phase8_simulation_digests.py"


@pytest.fixture(scope="module")
def recorder():
    spec = importlib.util.spec_from_file_location("record_phase8_simulation_digests", RECORDER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def fixture() -> dict:
    data = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    recorded = (data["numpy"], data["scipy"], data["platform"])
    data["exact"] = recorded == (np.__version__, scipy.__version__, sys.platform)
    return data


@pytest.fixture(scope="module")
def current(recorder) -> dict:
    return recorder.compute_digests()


def test_the_disturbed_simulations_are_reproduced(fixture, current):
    """Fails if a change to the simulator or a disturbance alters the generated truth or scans.

    The digest covers the truth and every scan of the run; the checksums give the same guard
    a tolerance where the digest cannot be compared.
    """
    assert set(current["scenes"]) == set(fixture["scenes"])
    for key, expected in fixture["scenes"].items():
        actual = current["scenes"][key]
        assert_same(actual["truth_end"], expected["truth_end"], fixture["exact"], key)
        assert_same(actual["camera_sum"], expected["camera_sum"], fixture["exact"], key)
        if fixture["exact"]:
            assert actual["simulation_sha256"] == expected["simulation_sha256"], key


def test_every_disturbance_hook_is_in_the_fixture(fixture, recorder):
    """Fails if a hook (bias, time offset, lever arm, turn, acceleration, random, combined) is
    dropped from the recorded scenes, which would leave it unguarded."""
    names = " ".join(fixture["scenes"])
    for hook in ("bias", "offset", "lever", "turn", "acceleration", "random", "combined"):
        assert hook in names, hook
    assert {key.split("/")[0] for key in fixture["scenes"]} == {"crossing", "separated"}
    assert len({v["simulation_sha256"] for v in fixture["scenes"].values()}) == len(
        fixture["scenes"]
    )  # every scene and seed is a different world
    assert recorder.SEEDS == (0, 1, 2)
