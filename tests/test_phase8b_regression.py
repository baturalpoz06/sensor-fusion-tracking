"""Phase 8b regression guard: the IMM and bias features must keep producing the recorded results.

The fixture tests/data/phase8b_golden.json was recorded by scripts/record_phase8b_golden.py when
the features were complete. As for the earlier phases, the comparison is bitwise (digests
included) where the recording environment (numpy, scipy and platform) matches and within a tight
tolerance, without the digests, elsewhere; the test never skips.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy

from tests.test_phase6_regression import assert_same, to_floats

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = ROOT / "tests" / "data" / "phase8b_golden.json"
RECORDER_PATH = ROOT / "scripts" / "record_phase8b_golden.py"
ARM_NAMES = {
    "ekf", "ekf_high_q_2", "imm_a_3", "imm_b_10_1", "ekf_bias", "ekf_always", "ekf_oracle",
    "imm_a_bias", "imm_b_bias",
}  # fmt: skip


@pytest.fixture(scope="module")
def golden() -> dict:
    data = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    recorded = (data["numpy"], data["scipy"], data["platform"])
    data["exact"] = recorded == (np.__version__, scipy.__version__, sys.platform)
    return data


@pytest.fixture(scope="module")
def recorder():
    spec = importlib.util.spec_from_file_location("record_phase8b_golden", RECORDER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def current(recorder) -> dict:
    return recorder.compute_golden()


def test_every_recorded_trial_is_reproduced(golden, current):
    """Fails if the IMM, the sensitivity, the bias estimate or the code they share changes a
    result: metrics (bitwise where the environment matches), the digest of every step's reported
    tracks, the bias estimates over time and the mode probabilities."""
    assert current["metric_fields"] == golden["metric_fields"]
    assert set(current["trials"]) == set(golden["trials"])
    for key, expected in golden["trials"].items():
        actual = current["trials"][key]
        assert set(actual) == set(expected), key
        assert_same(actual["metrics"], expected["metrics"], golden["exact"], key)
        for name in ("bias_samples", "bias_p1_last"):
            if name in expected:
                assert_same(actual[name], expected[name], golden["exact"], f"{key}/{name}")
        if golden["exact"]:
            assert actual["history_sha256"] == expected["history_sha256"], key
            assert actual.get("mode_sha256") == expected.get("mode_sha256"), key


def test_the_simulated_data_are_unchanged(golden, current):
    """Fails if the simulator generates other truth or scans for the recorded scenes."""
    assert set(current["simulations"]) == set(golden["simulations"])
    if golden["exact"]:
        assert current["simulations"] == golden["simulations"]


def test_the_fixture_reaches_every_feature(golden):
    """Fails if the fixture is regenerated without a feature: every arm of both scenes and both
    seeds, a bias estimate that moved, the oracle at exactly its bias, and IMM mode digests."""
    keys = list(golden["trials"])
    assert {k.split("/")[1] for k in keys} == ARM_NAMES
    assert {k.split("/")[0] for k in keys} == {"separated", "crossing"}
    assert {k.split("/")[2] for k in keys} == {"2000", "2001"}
    for key, record in golden["trials"].items():
        arm = key.split("/")[1]
        if "bias" in arm or arm in ("ekf_always", "ekf_oracle"):
            samples = to_floats(record["bias_samples"])
            assert len(samples) >= 3, key
            if arm == "ekf_oracle":
                assert all(s == pytest.approx(0.5 * np.pi / 180.0) for s in samples), key
            else:
                assert max(abs(s) for s in samples) > 1e-4, key
        else:
            assert "bias_samples" not in record, key
        assert ("mode_sha256" in record) == arm.startswith("imm"), key
    digests = {r["history_sha256"] for r in golden["trials"].values()}
    assert len(digests) == len(golden["trials"])  # every arm, scene and seed differs


def test_the_default_arm_of_the_fixture_is_the_plain_tracker(golden, recorder):
    """Fails if the 'ekf' arm of the fixture stops being the Phase 8a tracker: its digest must
    equal that of a tracker run by hand with the default configuration (no motion, no bias)."""
    from fusion.mtt_experiment import SCENARIOS
    from fusion.mtt_simulation import make_mtt_rngs, simulate_mtt
    from fusion.tracker.multi_target import run_multi_target_tracking

    config = recorder.scene_config("separated")
    sim = simulate_mtt(SCENARIOS["separated"], config, make_mtt_rngs(2000), config.truth)
    run = run_multi_target_tracking(sim, *config.tracker_setup())
    digest = recorder.PHASE7.PHASE6.history_digest(run.history)
    recorded = golden["trials"]["separated/ekf/2000"]["history_sha256"]
    if golden["exact"]:
        assert digest == recorded
    else:
        assert len(digest) == len(recorded)
