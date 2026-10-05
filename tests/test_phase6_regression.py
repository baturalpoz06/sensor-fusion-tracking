"""Phase 6 regression guard: results must stay identical to the recorded fixture.

The fixture tests/data/phase6_golden.json was recorded at commit 0067cac by
scripts/record_phase6_golden.py, before any Phase 7 change. Where the recording environment
(numpy, scipy and platform) matches, results must be bitwise identical, including a digest
of every step's tracks. Elsewhere the floating point operations can differ in the last
bits, so the metrics are compared with a tight tolerance instead and the digests are not
compared; the test never skips, so the guard stays active on other platforms.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy

from fusion.mtt_simulation import RNG_STREAMS, make_mtt_rngs

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = ROOT / "tests" / "data" / "phase6_golden.json"
RECORDER_PATH = ROOT / "scripts" / "record_phase6_golden.py"

# Tolerance of the comparison outside the recording environment: rounding differences are
# around 1e-15 relative; a changed algorithm moves the scores by orders of magnitude more.
RTOL = 1e-9
ATOL = 1e-12

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


def to_floats(value):
    """Nested lists of float hex strings as an array (NaN as nan)."""
    if isinstance(value, str):
        return float.fromhex(value)
    return [to_floats(item) for item in value]


def assert_same(actual, expected, exact: bool, where: str = "") -> None:
    """Identical text where exact, otherwise equal up to RTOL / ATOL (NaN equals NaN)."""
    if exact:
        assert actual == expected, where
    else:
        np.testing.assert_allclose(
            np.array(to_floats(actual)),
            np.array(to_floats(expected)),
            rtol=RTOL,
            atol=ATOL,
            equal_nan=True,
            err_msg=where,
        )


@pytest.fixture(scope="module")
def golden() -> dict:
    data = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    recorded = (data["numpy"], data["scipy"], data["platform"])
    data["exact"] = recorded == (np.__version__, scipy.__version__, sys.platform)
    return data


@pytest.fixture(scope="module")
def recorder():
    return load_recorder()


def test_the_comparison_helper_accepts_rounding_and_rejects_real_changes():
    """Fails if the tolerance path would accept a changed score, or the exact path a rounded one."""
    one, next_up = "0x1.0000000000000p+0", "0x1.0000000000001p+0"
    changed = float.hex(1.0 + 1e-6)
    assert_same([one, "nan"], [next_up, "nan"], exact=False)
    with pytest.raises(AssertionError):
        assert_same([one], [next_up], exact=True)
    with pytest.raises(AssertionError):
        assert_same([one], [changed], exact=False)
    with pytest.raises(AssertionError):
        assert_same([one], ["nan"], exact=False)


def test_every_recorded_trial_is_reproduced(golden, recorder):
    """Fails if any code shared with Phase 6 (tracker, metrics, simulation, RNG) changes a result.

    Bitwise, with the digest of every step's tracks, where the recording environment matches.
    """
    current = recorder.compute_golden()
    assert current["metric_fields"] == golden["metric_fields"]
    assert set(current["trials"]) == set(golden["trials"])
    for key, expected in golden["trials"].items():
        assert_same(current["trials"][key]["metrics"], expected["metrics"], golden["exact"], key)
        if golden["exact"]:
            assert current["trials"][key]["history_sha256"] == expected["history_sha256"], key


@pytest.mark.parametrize("workers", [1, 2])
def test_sweep_and_fusion_comparison_are_reproduced_with_any_worker_count(
    golden, recorder, workers
):
    """Fails if the sweep machinery (_run_points, sweep, compare_fusion) changes a result."""
    current = recorder.compute_sweeps(workers)
    for name in ("sweep_clutter_rate", "compare_fusion"):
        for metric, rows in golden[name].items():
            assert_same(current[name][metric], rows, golden["exact"], f"{name}/{metric}")


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


def test_the_robustness_pipeline_at_neutral_defaults_reproduces_the_phase_6_golden(
    golden, recorder
):
    """Fails if the truth / belief separation (disturbance hooks, tracker setup, diagnostics)
    changes a Phase 6 result at neutral defaults.

    Runs the whole Phase 6 grid through run_robustness_trial and simulate_and_track, not
    through the recorder's own loop, so the new code path itself is what is compared: the
    metrics and, where the environment matches, the digest of every step's tracks.
    """
    from fusion.mtt_experiment import SCENARIOS
    from fusion.robustness_experiment import (
        RobustnessConfig,
        run_robustness_trial,
        simulate_and_track,
    )

    for key, expected in golden["trials"].items():
        scenario, case, seed = key.split("/")
        config = RobustnessConfig.from_mtt(recorder.case_config(case))
        states = SCENARIOS[scenario]
        metrics = run_robustness_trial(states, config, int(seed))
        actual = [float(getattr(metrics, f"run_{name}")).hex() for name in golden["metric_fields"]]
        assert_same(actual, expected["metrics"], golden["exact"], key)
        if golden["exact"]:
            _, run = simulate_and_track(states, config, int(seed))
            assert recorder.history_digest(run.history) == expected["history_sha256"], key
