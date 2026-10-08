"""Tests of scripts/diag_outage_limitations.py.

The row builders, the stretch counter and the docstring check are tested on hand-built inputs; the
whole script on a 2-seed run, which says nothing about the tracker. Each test docstring names the
defect that makes it fail.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from fusion.mtt_experiment import SweepResult
from fusion.tracker.multi_target import CameraStepLog

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "diag_outage_limitations.py"


def load_script():
    spec = importlib.util.spec_from_file_location("diag_outage_limitations", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


diag = load_script()


def test_l1_rows_put_every_max_coast_below_the_outage_and_the_control_above_it():
    """Fails if a row cannot trigger a coast deletion or the control row could trigger one."""
    rows = diag.l1_rows()
    assert [label for label, _ in rows] == [
        "max coast 2 s", "max coast 5 s", "max coast 10 s", "max coast 15 s",
        "max coast 25 s (control)",
    ]  # fmt: skip
    for label, config in rows:
        start, end = config.dropout.span
        assert (start, end) == (20.0, 40.0)
        assert config.dropout.sensors == ("radar",)
        assert config.outage_policy == "aware"
        assert config.radar_clutter_rate == 0.0 and config.camera_clutter_rate == 0.0
        below = config.max_coast_time < end - start
        assert below == ("control" not in label)


def test_l2_rows_cover_every_clutter_pair_at_both_max_coasts_with_the_target_vanishing_inside():
    """Fails if a clutter pair or its clutter-free reference is missing, or the target vanishes
    outside the outage (then the coast clock would not decide its ghost life)."""
    rows = diag.l2_rows()
    assert len(rows) == 8
    pairs = {(coast, clutter) for _, coast, clutter, _ in rows}
    assert pairs == {(c, p) for c in (5.0, 10.0) for p in diag.L2_CLUTTER}
    for label, coast, (radar, camera), config in rows:
        start, end = config.dropout.span
        assert (start, end) == (20.0, 44.0)
        assert config.vanish == ((0, 22.0),)
        assert start <= config.vanish[0][1] < end
        assert config.max_coast_time == coast
        assert (config.radar_clutter_rate, config.camera_clutter_rate) == (radar, camera)
        assert label == f"coast {coast:g} s, clutter {radar:g}/{camera:g}"


def test_longest_stretches_count_only_radar_down_steps_without_a_camera_update():
    """Fails if an update does not reset the count, steps with the radar up are counted, or
    ambiguity skips are not counted separately."""
    log = [
        CameraStepLog(usable=(1, 2), assigned=((1, 0), (2, 1))),  # radar up: ignored
        CameraStepLog(usable=(1,), skipped=(2,), assigned=((1, 0),)),
        CameraStepLog(skipped=(1, 2)),
        CameraStepLog(skipped=(1, 2)),
        CameraStepLog(usable=(1, 2), assigned=((2, 0),)),
        CameraStepLog(usable=(1, 2), assigned=((1, 0), (2, 1))),
    ]
    down = np.array([False, True, True, True, True, True])
    # track 2: no update at steps 1, 2, 3 (3 steps); track 1: steps 2, 3, 4 (3 steps)
    # skipped as ambiguous: track 2 at steps 1, 2, 3 (3 steps), track 1 at steps 2, 3 (2 steps)
    assert diag.longest_stretches(log, down) == (3, 3)


def test_longest_stretches_restart_after_the_radar_comes_back():
    """Fails if a stretch of an earlier outage is joined to one of a later outage."""
    log = [CameraStepLog(skipped=(1,))] * 2 + [CameraStepLog(usable=(1,))] + [
        CameraStepLog(skipped=(1,))
    ] * 2
    down = np.array([True, True, False, True, True])
    assert diag.longest_stretches(log, down) == (2, 2)


def test_differences_are_per_seed_and_undefined_where_either_value_is():
    """Fails if a difference mixes seeds or turns a missing value into a number."""
    values = np.array([[1.0, 2.0, np.nan], [4.0, 6.0, 8.0]])
    d = diag.differences(values, values[0])
    np.testing.assert_array_equal(d[0], [0.0, 0.0, np.nan])
    np.testing.assert_array_equal(d[1], [3.0, 4.0, np.nan])


def test_stated_check_compares_the_camera_clutter_rows_at_one_decimal():
    """Fails if a row is judged against the wrong stated value or rounded differently."""
    rows = diag.l2_rows()
    life = np.zeros((len(rows), 2))
    for i, (_, coast, clutter, _) in enumerate(rows):
        life[i] = {5.0: 7.04, 10.0: 20.84}[coast] if clutter[1] > 0 else coast
    result = SweepResult("row", np.arange(len(rows), dtype=float), {"ghost_lifetime": life})
    lines = diag.stated_check(rows, result)
    checked = [line for line in lines if "measured" in line and "stated" in line]
    assert len(checked) == 4
    assert all(line.endswith("stated 7.0 s: same") for line in checked if "coast 5 s" in line)
    assert all(line.endswith("stated 20.9 s: differs") for line in checked if "coast 10" in line)


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    out = tmp_path_factory.mktemp("limitations") / "limitations.txt"
    run = subprocess.run(
        [sys.executable, str(SCRIPT), "--seeds", "2", "--workers", "1", "--out", str(out)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    return out.read_text(encoding="utf-8").splitlines()


def test_the_report_starts_with_provenance_and_holds_every_table(smoke):
    """Fails if the provenance header or a table or check of the report is missing."""
    assert smoke[0].startswith("# command: python ") and "diag_outage_limitations.py" in smoke[0]
    assert smoke[1].startswith("# git HEAD: ")
    assert smoke[2].startswith("# date (UTC): ")
    assert smoke[4].startswith("OUTAGE LIMITATIONS (Phase 7): 2 seeds (0..1)")
    assert any(line.startswith("L1, scene 'crossing'") for line in smoke)
    assert any(line.startswith("L1, scene 'separated'") for line in smoke)
    assert sum(line.startswith("  control run, longest stretch") for line in smoke) == 2
    assert any(line.startswith("closest approach of targets 0 and 1") for line in smoke)
    assert any(line.startswith("L2, scene 'vanishing'") for line in smoke)
    assert sum(line.strip().startswith("coast ") and "stated" in line for line in smoke) == 4


def test_a_single_seed_is_refused(tmp_path):
    """Fails if the script writes intervals from one seed."""
    run = subprocess.run(
        [sys.executable, str(SCRIPT), "--seeds", "1", "--out", str(tmp_path / "x.txt")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode != 0
    assert not (tmp_path / "x.txt").exists()
