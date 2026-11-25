"""Protocol schedule of the two-stage resolution and the storage time of every qubit."""

import math
from collections.abc import Mapping
from dataclasses import dataclass

from .rolling import SupportSequence, outputs, stage2


@dataclass(frozen=True)
class Schedule:
    """Distances and signal speed between the Tier2 node and the Tier1 nodes of the peers.

    Attributes:
        distance: Distance in m, one value for every peer or a mapping per peer.
        speed: Signal speed in m/s; 2e8 m/s is the 5 us/km of optical fiber.
        processing: Time delta in s of one local round.

    Raises:
        ValueError: If a distance is negative or not finite, the speed is not positive, or the processing time is
            negative or not finite.
    """

    distance: float | Mapping[int, float]
    speed: float = 2e8
    processing: float = 1e-6

    def __post_init__(self) -> None:
        distances = self._distances()
        if not distances or not all(0 <= d < math.inf for d in distances):
            raise ValueError("Supply finite, nonnegative distances for the peers.")
        if not (self.speed > 0 and 0 <= self.processing < math.inf):
            raise ValueError("Speed must be positive and processing time finite and nonnegative.")

    def _distances(self) -> list[float]:
        """Returns every distance of the schedule."""
        return list(self.distance.values()) if isinstance(self.distance, Mapping) else [self.distance]

    @property
    def tau(self) -> float:
        """Propagation delay between Tier2 and the farthest Tier1 node."""
        return max(self._distances()) / self.speed

    @property
    def end(self) -> float:
        """Time 5 tau + 3 delta at which the endpoints receive the correction information.

        With Tier1 nodes at different distances, it is an upper bound.
        """
        return 5 * self.tau + 3 * self.processing

    def delay(self, peer: int) -> float:
        """Returns the propagation delay between Tier2 and the Tier1 node of a peer.

        Args:
            peer: The peer qubit.

        Raises:
            ValueError: If the distances are given per peer and the peer has none.
        """
        if not isinstance(self.distance, Mapping):
            return self.distance / self.speed
        if peer not in self.distance:
            raise ValueError(f"No distance for peer {peer}.")
        return self.distance[peer] / self.speed


def storage_times(seq: SupportSequence, k: int, schedule: Schedule) -> dict[int, float]:
    """Returns the storage time in s of every qubit in the two-stage resolution.

    The orchestration qubits are stored for 2 tau + delta, the peers measured in Stage 2 for 2 tau + 2 delta, and the
    output qubits for 2 tau + 2 tau' + 3 delta, with tau and tau' the delays of the farthest peer and of the farthest
    peer measured in Stage 2.

    Args:
        seq: Support sequence of Stage 1.
        k: Number of qubits of each output star, 2 <= k <= kappa_b.
        schedule: Distances, signal speed and processing time.

    Raises:
        ValueError: If k is out of range or a peer has no distance.
    """
    delta, measured = schedule.processing, [c for _, c in stage2(seq, k)]
    setup = 2 * max(schedule.delay(c) for c in seq.gtl.V_c)
    collected = 2 * max(schedule.delay(c) for c in measured)
    times = {o: setup + delta for o in seq.gtl.V_o}
    times |= {c: setup + 2 * delta for c in measured}
    times |= {v: setup + collected + 3 * delta for center, leaves in outputs(seq, k) for v in (center, *leaves)}
    return times
