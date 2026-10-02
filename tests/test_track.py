"""Tests for the track lifecycle (M-of-N confirmation, K-miss deletion)."""

import itertools

import numpy as np
import pytest

from fusion.filters.ekf import ExtendedKalmanFilter
from fusion.tracker.track import Lifecycle, LifecycleConfig, Track, TrackStatus, advance

TENTATIVE, CONFIRMED, DELETED = TrackStatus.TENTATIVE, TrackStatus.CONFIRMED, TrackStatus.DELETED
B, H, M = "B", "H", "M"  # birth (counts as a hit), hit, miss


def run(sequence: str, config: LifecycleConfig) -> list[Lifecycle]:
    """Lifecycle after every scan of a sequence such as "BHM" (B = birth, H = hit, M = miss)."""
    assert sequence[0] == B
    states = [Lifecycle.born(config)]
    for outcome in sequence[1:]:
        states.append(advance(states[-1], outcome == H, config))
    return states


def test_config_defaults_are_3_of_5_and_k5():
    config = LifecycleConfig()
    assert (config.confirm_hits, config.confirm_window, config.max_misses) == (3, 5, 5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"confirm_hits": 0},
        {"confirm_hits": 4, "confirm_window": 3},
        {"max_misses": -1},
    ],
)
def test_invalid_config_raises(kwargs):
    with pytest.raises(ValueError):
        LifecycleConfig(**kwargs)


def test_birth_counts_as_a_hit_and_starts_tentative():
    assert Lifecycle.born(LifecycleConfig()) == Lifecycle(TENTATIVE, 1, 1, 0)


def test_single_hit_confirmation_confirms_at_birth():
    config = LifecycleConfig(confirm_hits=1, confirm_window=1)
    assert Lifecycle.born(config).status is CONFIRMED


def test_a_single_tentative_miss_does_not_delete():
    states = run("BM", LifecycleConfig())
    assert states[-1] == Lifecycle(TENTATIVE, 1, 2, 1)


@pytest.mark.parametrize(
    ("sequence", "status", "scans"),
    [
        ("BHH", CONFIRMED, 3),
        ("BMHH", CONFIRMED, 4),
        ("BMHMH", CONFIRMED, 5),
        ("BMM", TENTATIVE, 3),
        ("BMMM", DELETED, 4),
        ("BHMMM", DELETED, 5),
        ("BMHMM", DELETED, 5),
    ],
)
def test_three_of_five_examples(sequence, status, scans):
    states = run(sequence, LifecycleConfig())
    assert len(states) == scans
    assert states[-1].status is status


def expected_outcome(outcomes: str, m: int, n: int) -> tuple[TrackStatus, int]:
    """Closed-form M-of-N result for the N - 1 outcomes after the birth hit.

    Within N scans either M hits or N - M + 1 misses occur, never both. The track
    is confirmed at the scan of the M-th hit or deleted at the scan of the
    (N - M + 1)-th miss, whichever comes first.
    """
    full = H + outcomes
    hit_scans = [i + 1 for i, o in enumerate(full) if o == H]
    miss_scans = [i + 1 for i, o in enumerate(full) if o == M]
    confirm_at = hit_scans[m - 1] if len(hit_scans) >= m else None
    delete_at = miss_scans[n - m] if len(miss_scans) >= n - m + 1 else None
    assert (confirm_at is None) != (delete_at is None)
    if confirm_at is not None:
        return CONFIRMED, confirm_at
    return DELETED, delete_at


@pytest.mark.parametrize(
    ("m", "n"), [(2, 2), (2, 3), (3, 3), (3, 5), (4, 6), (1, 4)], ids=lambda v: str(v)
)
@pytest.mark.parametrize("k", [0, 5])
def test_every_outcome_sequence_matches_the_closed_form(m, n, k):
    config = LifecycleConfig(confirm_hits=m, confirm_window=n, max_misses=k)
    for outcomes in itertools.product([H, M], repeat=n - 1):
        outcomes = "".join(outcomes)
        state = Lifecycle.born(config)
        decided_at = None
        if state.status is TENTATIVE:
            for outcome in outcomes:
                state = advance(state, outcome == H, config)
                if state.status is not TENTATIVE:
                    decided_at = state.scans
                    break
        else:
            decided_at = 1
        # A tentative track never lives past its N-th scan.
        assert decided_at is not None
        assert decided_at <= n
        assert (state.status, decided_at) == expected_outcome(outcomes, m, n)


def test_m_equal_n_deletes_on_the_first_miss():
    config = LifecycleConfig(confirm_hits=3, confirm_window=3)
    assert run("BM", config)[-1].status is DELETED
    assert run("BHM", config)[-1].status is DELETED
    assert run("BHH", config)[-1].status is CONFIRMED


def test_window_size_n_changes_tentative_outcomes():
    sequence = "BMHH"
    narrow = run(sequence[:2], LifecycleConfig(confirm_hits=3, confirm_window=3))
    assert narrow[-1].status is DELETED and narrow[-1].scans == 2
    wide = run(sequence, LifecycleConfig(confirm_hits=3, confirm_window=5))
    assert wide[-1].status is CONFIRMED and wide[-1].scans == 4


def confirmed_state(config: LifecycleConfig) -> Lifecycle:
    hits = config.confirm_hits
    return Lifecycle(CONFIRMED, hits=hits, scans=hits, consecutive_misses=0)


@pytest.mark.parametrize("k", [0, 2, 5])
def test_confirmed_track_survives_k_misses_and_dies_on_the_next(k):
    config = LifecycleConfig(max_misses=k)
    state = confirmed_state(config)
    for _ in range(k):
        state = advance(state, False, config)
        assert state.status is CONFIRMED
    assert advance(state, False, config).status is DELETED


def test_a_hit_resets_the_consecutive_miss_counter():
    config = LifecycleConfig(max_misses=2)
    state = confirmed_state(config)
    for outcome in (False, False, True, False, False):
        state = advance(state, outcome, config)
        assert state.status is CONFIRMED
    assert state.consecutive_misses == 2
    assert advance(state, False, config).status is DELETED


@pytest.mark.parametrize("n", [3, 5, 9])
def test_confirmed_deletion_does_not_depend_on_the_window(n):
    config = LifecycleConfig(confirm_hits=3, confirm_window=n, max_misses=2)
    state = confirmed_state(config)
    statuses = []
    for _ in range(3):
        state = advance(state, False, config)
        statuses.append(state.status)
    assert statuses == [CONFIRMED, CONFIRMED, DELETED]


@pytest.mark.parametrize("k", [0, 2, 9])
def test_tentative_behavior_does_not_depend_on_k(k):
    config = LifecycleConfig(confirm_hits=3, confirm_window=5, max_misses=k)
    assert run("BMMM", config)[-1].status is DELETED
    assert run("BMHMH", config)[-1].status is CONFIRMED
    assert run("BMM", config)[-1].status is TENTATIVE


def test_confirmation_happens_on_a_hit_so_misses_are_reset():
    state = run("BMHMH", LifecycleConfig())[-1]
    assert state.status is CONFIRMED and state.consecutive_misses == 0


def test_a_deleted_track_cannot_be_advanced():
    config = LifecycleConfig()
    deleted = run("BMMM", config)[-1]
    assert deleted.status is DELETED
    with pytest.raises(ValueError, match="deleted"):
        advance(deleted, True, config)


def test_track_exposes_its_status():
    config = LifecycleConfig()
    ekf = ExtendedKalmanFilter(dt=0.1, accel_std=0.5, x0=np.zeros(4), P0=np.eye(4))
    track = Track(track_id=7, filter=ekf, lifecycle=Lifecycle.born(config))
    assert track.track_id == 7 and track.status is TENTATIVE
    track.lifecycle = advance(track.lifecycle, True, config)
    track.lifecycle = advance(track.lifecycle, True, config)
    assert track.status is CONFIRMED
