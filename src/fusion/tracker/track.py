"""Track lifecycle: tentative, confirmed and deleted states with M-of-N logic."""

from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

from fusion.filters.base import TrackFilter


class TrackStatus(Enum):
    """Lifecycle state of a track."""

    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    DELETED = "deleted"


@dataclass(frozen=True)
class LifecycleConfig:
    """Parameters of the track lifecycle.

    A tentative track is confirmed as soon as it has M hits, and deleted as
    soon as more than N - M of its scans were misses, so it is never tentative
    after its N-th scan (M-of-N logic). A confirmed track is deleted when it
    has more than K consecutive misses.

    Attributes:
        confirm_hits: M, hits needed to confirm a tentative track (birth counts as a hit).
        confirm_window: N, number of scans (birth included) in which M hits must occur.
        max_misses: K, consecutive misses a confirmed track survives.
    """

    confirm_hits: int = 3
    confirm_window: int = 5
    max_misses: int = 5

    def __post_init__(self) -> None:
        if self.confirm_hits < 1:
            raise ValueError(f"confirm_hits must be >= 1, got {self.confirm_hits}")
        if self.confirm_window < self.confirm_hits:
            raise ValueError(
                f"confirm_window must be >= confirm_hits, got {self.confirm_window} "
                f"< {self.confirm_hits}"
            )
        if self.max_misses < 0:
            raise ValueError(f"max_misses must be >= 0, got {self.max_misses}")


class Lifecycle(NamedTuple):
    """Lifecycle counters of one track.

    Attributes:
        status: Current state.
        hits: Radar scans in which the track was assigned a measurement, birth included.
        scans: Radar scans the track has seen, birth included.
        consecutive_misses: Misses since the last hit.
    """

    status: TrackStatus
    hits: int
    scans: int
    consecutive_misses: int

    @classmethod
    def born(cls, config: LifecycleConfig) -> "Lifecycle":
        """State of a track right after birth; the birth measurement counts as a hit."""
        confirmed = config.confirm_hits <= 1
        status = TrackStatus.CONFIRMED if confirmed else TrackStatus.TENTATIVE
        return cls(status, hits=1, scans=1, consecutive_misses=0)


def advance(state: Lifecycle, hit: bool, config: LifecycleConfig) -> Lifecycle:
    """Lifecycle after one more radar scan.

    Rules, with misses = scans - hits:
        TENTATIVE: hits >= M -> CONFIRMED; otherwise misses > N - M -> DELETED.
            A single miss does not delete (3-of-5: the third miss does).
        CONFIRMED: consecutive_misses > K -> DELETED.

    Args:
        state: Counters before the scan.
        hit: Whether the track was assigned a measurement in this scan.
        config: Lifecycle parameters.

    Raises:
        ValueError: If the track is already deleted (a terminal state).
    """
    if state.status is TrackStatus.DELETED:
        raise ValueError("a deleted track cannot be advanced")
    scans = state.scans + 1
    hits = state.hits + int(hit)
    misses = 0 if hit else state.consecutive_misses + 1

    if state.status is TrackStatus.TENTATIVE:
        if hits >= config.confirm_hits:
            status = TrackStatus.CONFIRMED
        elif scans - hits > config.confirm_window - config.confirm_hits:
            status = TrackStatus.DELETED
        else:
            status = TrackStatus.TENTATIVE
    else:
        status = TrackStatus.DELETED if misses > config.max_misses else TrackStatus.CONFIRMED
    return Lifecycle(status, hits, scans, misses)


@dataclass
class Track:
    """A tracked target: its filter and its lifecycle.

    Attributes:
        track_id: Unique identifier, never reused.
        filter: Filter (EKF or IMM) holding the state estimate and covariance.
        lifecycle: Current lifecycle counters.
        coast_steps: Steps since the last measurement update of any sensor (zero at birth).
            Only the aware outage policy acts on it.
    """

    track_id: int
    filter: TrackFilter
    lifecycle: Lifecycle
    coast_steps: int = 0

    @property
    def status(self) -> TrackStatus:
        """Current lifecycle state."""
        return self.lifecycle.status
