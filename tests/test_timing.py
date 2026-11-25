"""The storage time of every qubit, against a discrete-event simulation of the protocol."""

import heapq
import random

import pytest

from gtl_rolling import GTL, Schedule, SupportSequence, outputs, stage2, storage_times


def simulate(seq, k, schedule):
    """Storage time of every qubit and time of the last receipt of the corrections, message by message.

    Each message is an event at its arrival: the peers, their acknowledgments, the Stage-2 directives, the Stage-2
    outcomes and the correction information. Tier2 measures when every acknowledgment has arrived, the Tier1 nodes
    measure on receipt, and Tier2 relays the outcomes when every one has arrived, each in one local round.
    """
    gtl, delta = seq.gtl, schedule.processing
    measured = {c for _, c in stage2(seq, k)}
    retained = {v for center, leaves in outputs(seq, k) for v in (center, *leaves)}
    start, stop, last = dict.fromkeys(gtl.V_o, 0.0), {}, 0.0
    acknowledged, reported = set(), set()
    events = [(schedule.delay(c), "peer", c) for c in gtl.V_c]
    heapq.heapify(events)
    while events:
        time, kind, c = heapq.heappop(events)
        if kind == "peer":
            start[c] = time
            heapq.heappush(events, (time + schedule.delay(c), "acknowledgment", c))
        elif kind == "acknowledgment":
            acknowledged.add(c)
            if acknowledged == set(gtl.V_c):
                stop |= dict.fromkeys(gtl.V_o, time + delta)
                for v in gtl.V_c:
                    heapq.heappush(events, (time + delta + schedule.delay(v), "directive", v))
        elif kind == "directive" and c in measured:
            stop[c] = time + delta
            heapq.heappush(events, (time + delta + schedule.delay(c), "outcome", c))
        elif kind == "outcome":
            reported.add(c)
            if reported == measured:
                for v in retained:
                    heapq.heappush(events, (time + delta + schedule.delay(v), "correction", v))
        elif kind == "correction":
            stop[c], last = time, max(last, time)
    return {v: stop[v] - start[v] for v in gtl.graph}, last


def test_storage_times():
    """At 40 km, 0.401 ms from dispatch, 0.402 and 0.803 ms from arrival, and the end at 1.003 ms, as simulated."""
    gtl, schedule = GTL(3, 2), Schedule(distance=40e3)
    assert schedule.end == pytest.approx(1.003e-3)
    for seq in SupportSequence.all(gtl):
        t, (simulated, last) = storage_times(seq, 2, schedule), simulate(seq, 2, schedule)
        assert t == pytest.approx(simulated)
        assert last == pytest.approx(schedule.end)
        assert all(t[o] == pytest.approx(0.401e-3) for o in gtl.V_o)
        assert all(t[c] == pytest.approx(0.402e-3) for _, c in stage2(seq, 2))
        assert all(t[v] == pytest.approx(0.803e-3) for center, (leaf,) in outputs(seq, 2) for v in (center, leaf))


@pytest.mark.parametrize("k", [2, 3])
def test_storage_times_unequal_distances(k):
    """With a random distance per Tier1 node, the storage times are those of the simulation.

    No qubit is stored longer than with every node at the farthest distance, and the last endpoint receives the
    correction information by Schedule.end.
    """
    rng, gtl = random.Random(43), GTL(4, 3)
    for seq in SupportSequence.all(gtl):
        schedule = Schedule(distance={c: rng.uniform(10e3, 100e3) for c in gtl.V_c})
        t, (simulated, last) = storage_times(seq, k, schedule), simulate(seq, k, schedule)
        farthest = storage_times(seq, k, Schedule(distance=max(schedule.distance.values())))
        assert t == pytest.approx(simulated, abs=1e-15)
        assert last <= schedule.end + 1e-15
        assert all(t[v] <= farthest[v] + 1e-15 for v in gtl.graph)
