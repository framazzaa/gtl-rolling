"""Noise channels, and how single errors move through the two-stage resolution with the NSF update rules."""

import math

import networkx as nx
import pytest

from gtl_rolling import (
    GTL,
    SupportSequence,
    dephasing,
    depolarizing,
    nbar,
    noise_operator,
    outputs,
    propagate,
    proximity,
    stage1,
    stage2,
)

CASES = [(3, 2), (2, 3), (4, 2)]
EMPTY = frozenset()


def operators(graph, sequence, v):
    """The noise operators of qubit v, as (alpha, beta) -> its set of Z qubits after the measurements of sequence."""
    return {
        (alpha, beta): propagate(graph, sequence, noise_operator(graph, v, alpha, beta))
        for alpha, beta in ((1, 0), (0, 1), (1, 1))
    }


@pytest.mark.parametrize("n_o, kb", CASES)
def test_maps_after_stage1(n_o, kb):
    """After Stage 1 the noise operators of o_i act on nbar(o_i), and those of a peer c on c and N~(c).

    The noise operator (alpha, beta) of o_i is Z^alpha on nbar(o_i), and that of a peer c is
    Z_c^alpha prod_{k in N~} Z_k^beta, with alpha and beta swapped on the supports. N~ is the non-support of o_m and
    the rolled set for the support b0^(o_m), every support for a peer of the final rolled set, and {b0^(o_m)} for a
    non-support peer of o_m.
    """
    gtl = GTL(n_o, kb)
    for seq in SupportSequence.all(gtl):
        for i in range(1, n_o + 1):
            for (alpha, _), noise in operators(gtl.graph, stage1(seq), gtl.o(i)).items():
                assert noise == (nbar(seq, i) if alpha else EMPTY)
        gamma = frozenset(seq.rolled_set(n_o))
        n_tilde = {c: frozenset(seq.supports) for c in gamma}
        for m, b0 in enumerate(seq.supports, start=1):
            n_tilde[b0] = frozenset(seq.non_support(m)) | gamma
            n_tilde |= {c: frozenset({b0}) for c in seq.non_support(m)}
        assert sorted(n_tilde) == list(gtl.V_c)
        for c, neighbors in n_tilde.items():
            for (alpha, beta), noise in operators(gtl.graph, stage1(seq), c).items():
                if c in seq.supports:
                    alpha, beta = beta, alpha
                assert noise == (frozenset({c}) if alpha else EMPTY) | (neighbors if beta else EMPTY)


@pytest.mark.parametrize("n_o, kb", CASES)
def test_maps_after_stage2(n_o, kb):
    """After Stage 2 with Bell pairs, each pair qubit keeps the depolarizing form on its pair.

    The operators of o_i are those after Stage 1, and a measured peer loses its Z error while its X and Y errors act
    as Z on its support, or on every support for a peer of the final rolled set.
    """
    gtl = GTL(n_o, kb)
    for seq in SupportSequence.all(gtl):
        sequence = stage1(seq) + stage2(seq, 2)
        for i in range(1, n_o + 1):
            for (alpha, _), noise in operators(gtl.graph, sequence, gtl.o(i)).items():
                assert noise == (nbar(seq, i) if alpha else EMPTY)
        partner = {}
        for center, (leaf,) in outputs(seq, 2):
            partner |= {center: leaf, leaf: center}
        for v, w in partner.items():
            noises = set(operators(gtl.graph, sequence, v).values())
            assert noises == {frozenset({v}), frozenset({w}), frozenset({v, w})}
        gamma = frozenset(seq.rolled_set(n_o))
        support = {c: b0 for m, b0 in enumerate(seq.supports, start=1) for c in seq.non_support(m)}
        for c in set(gtl.V_c) - set(partner):
            target = frozenset(seq.supports) if c in gamma else frozenset({support[c]})
            assert operators(gtl.graph, sequence, c) == {(1, 0): EMPTY, (0, 1): target, (1, 1): target}


@pytest.mark.parametrize("n_o, kb", CASES)
def test_light_cone_of_bell_pairs(n_o, kb):
    """Only the qubits behind the exponents 2 and 2 kappa_b + pi of its fidelity send noise to a Bell pair (b0, a).

    Noise reaches a only from a and b0, and b0 only from the kappa_b peers of its group, the kappa_b peers of the final
    rolled set and the pi(b0, gamma) orchestration qubits on a shortest path between them.
    """
    gtl = GTL(n_o, kb)
    for seq in SupportSequence.all(gtl):
        sequence = stage1(seq) + stage2(seq, 2)
        reach = {v: set().union(*operators(gtl.graph, sequence, v).values()) for v in gtl.graph}
        gamma = seq.rolled_set(n_o)
        for m, (b0, (a,)) in enumerate(outputs(seq, 2), start=1):
            assert {v for v in gtl.graph if a in reach[v]} == {a, b0}
            reaching_b0 = {v for v in gtl.graph if b0 in reach[v]}
            for c in gamma:
                on_path = set(nx.shortest_path(gtl.graph, b0, c)) & set(gtl.V_o)
                assert reaching_b0 == {b0, *seq.non_support(m), *gamma} | on_path
                assert len(reaching_b0) == 2 * kb + proximity(gtl.graph, gtl.V_o, b0, c)


def test_channels():
    """The depolarizing weights sum to one and the dephasing probability is (1 - exp(-t/T)) / 2, also for tiny t."""
    assert sum(depolarizing(0.7)) == pytest.approx(1.0)
    assert depolarizing(1.0) == (1.0, 0.0, 0.0, 0.0)
    assert dephasing(0.0, 1e-3) == 0.0 and dephasing(1.0, math.inf) == 0.0
    assert dephasing(2e-3, 1e-3) == pytest.approx((1 - math.exp(-2)) / 2)
    assert dephasing(1e-18, 1.0) == pytest.approx(5e-19, rel=1e-15, abs=0)
