"""Fidelity of the output stars: the NSF simulation against the closed forms."""

import itertools
import math
import random

import networkx as nx
import noisy_graph_states as ngs
import noisy_graph_states.libs.graph as gt
import noisy_graph_states.libs.matrix as mat
import noisy_graph_states.tools.density_matrix as dm
import numpy as np
import pytest

from gtl_rolling import (
    GTL,
    SupportSequence,
    depolarizing,
    fidelity,
    joint_fidelity,
    noisy_state,
    outputs,
    proximity,
    run,
    stage1,
    stage2,
    star_fidelity,
)

CASES = [(3, 2), (2, 3)]


def overlap(ket, rho):
    """Fidelity <ket| rho |ket> of a density matrix with a pure state."""
    return float(np.real(ket.conj().T @ rho @ ket)[0, 0])


@pytest.mark.parametrize("pattern", ["".join(sides) for sides in itertools.product("RL", repeat=2)])
def test_nsf_matches_density_matrix(pattern):
    """The NSF gives the output state and fidelities of the library's full density-matrix engine, on GTL(2, 2).

    The density-matrix engine applies the noise and the measurements of both stages to the full state, independently of
    the noise maps of the NSF.
    """
    gtl = GTL(2, 2)
    seq = SupportSequence(gtl, pattern)
    rng = np.random.default_rng(0)
    p = {v: rng.uniform(0.7, 1.0) for v in gtl.graph}
    q = {v: rng.uniform(0.0, 0.3) for v in gtl.graph}
    stars = outputs(seq, 2)
    pairs = sorted(v for center, leaves in stars for v in (center, *leaves))

    nsf = run(seq, 2, noisy_state(gtl.graph, p, q))
    target = gt.graph_state(nx.convert_node_labels_to_integers(nsf.graph.subgraph(pairs), ordering="sorted"))
    rho_nsf = ngs.apply_nsf_maps_to_dm(ngs.reduce_maps(nsf, pairs), target @ target.conj().T, pairs)

    ket = gt.graph_state(gtl.graph)
    state = dm.State(gtl.graph, ket @ ket.conj().T)
    for v in gtl.graph:
        state = dm.z_noise(dm.pauli_noise(state, [v], depolarizing(p[v])), [v], q[v])
    for instruction in stage1(seq) + stage2(seq, 2):
        if instruction[0] == "x":
            state = dm.x_measurement(state, instruction[1], instruction[2])
        else:
            state = dm.z_measurement(state, instruction[1])
    rho_dm = mat.ptrace(state.density_matrix, [v for v in gtl.graph if v not in pairs])

    assert np.allclose(rho_nsf, rho_dm, rtol=0, atol=1e-12)
    assert fidelity(nsf, stars) == pytest.approx(overlap(target, rho_dm), abs=1e-12)
    bell = gt.graph_state(nx.Graph([(0, 1)]))
    for center, leaves in stars:
        others = [pairs.index(v) for v in pairs if v not in (center, *leaves)]
        assert fidelity(nsf, [(center, leaves)]) == pytest.approx(overlap(bell, mat.ptrace(rho_dm, others)), abs=1e-12)


@pytest.mark.parametrize("n_o, kb", CASES)
def test_bell_pair_fidelity(n_o, kb):
    """Under depolarizing noise, every Bell pair (b0, a) has fidelity (1 + p^2 + 2 p^(2 kappa_b + pi)) / 4.

    pi is the proximity of b0 to the final rolled set in the initial resource state. The NSF gives the fidelity for the
    leaf kept by stage2, and the closed form for any leaf of the group, since they are twins.
    """
    gtl, p = GTL(n_o, kb), 0.9
    for seq in SupportSequence.all(gtl):
        gamma = seq.rolled_set(n_o)[0]
        out = run(seq, 2, noisy_state(gtl.graph, p))
        for pair, (center, leaves) in zip(outputs(seq, 2), outputs(seq, kb), strict=True):
            expected = (1 + p**2 + 2 * p ** (2 * kb + proximity(gtl.graph, gtl.V_o, center, gamma))) / 4
            assert fidelity(out, [pair]) == pytest.approx(expected, abs=1e-12)
            for leaf in leaves:
                assert star_fidelity(seq, 2, (center, (leaf,)), p) == pytest.approx(expected, abs=1e-15)


@pytest.mark.parametrize("n_o, kb", CASES)
def test_star_fidelity_closed_form(n_o, kb):
    """Under depolarizing noise, the closed-form fidelity of every k-qubit output star equals the NSF one."""
    rng, gtl = random.Random(n_o * 10 + kb), GTL(n_o, kb)
    for seq in SupportSequence.all(gtl):
        for k in range(2, kb + 1):
            p = rng.uniform(0.7, 1.0)
            out = run(seq, k, noisy_state(gtl.graph, p))
            for star in outputs(seq, k):
                assert fidelity(out, [star]) == pytest.approx(star_fidelity(seq, k, star, p), abs=1e-12)


@pytest.mark.parametrize("n_o, kb", CASES)
def test_joint_fidelity_closed_form(n_o, kb):
    """The closed-form joint fidelity of every set of output stars with at most 6 qubits equals the NSF one.

    It is one without noise, and the NSF value does not depend on the order in which the stars are given.
    """
    rng, gtl = random.Random(n_o * 10 + kb), GTL(n_o, kb)
    for seq in SupportSequence.all(gtl):
        for k in range(2, kb + 1):
            stars = outputs(seq, k)
            assert fidelity(run(seq, k, noisy_state(gtl.graph)), stars) == pytest.approx(1.0, abs=1e-12)
            p = rng.uniform(0.7, 1.0)
            out = run(seq, k, noisy_state(gtl.graph, p))
            for size in range(2, 6 // k + 1):
                for subset in itertools.combinations(stars, size):
                    expected = joint_fidelity(seq, k, subset, p)
                    assert fidelity(out, subset) == pytest.approx(expected, abs=1e-12)
                    assert fidelity(out, subset[::-1]) == pytest.approx(expected, abs=1e-12)


def test_joint_fidelity_bounds():
    """On GTL(10, 2) the joint fidelity of the Bell pairs lies above their product and not above their minimum."""
    gtl = GTL(10, 2)
    for seq in (SupportSequence.right(gtl), SupportSequence(gtl, "RLRLRLRLRL")):
        stars = outputs(seq, 2)
        for p in (0.8, 0.9, 0.99):
            single = [star_fidelity(seq, 2, star, p) for star in stars]
            assert math.prod(single) < joint_fidelity(seq, 2, stars, p) <= min(single)


def test_fidelity_grid_matches_simulation():
    """On an array of p, including 0 and 1, the broadcast closed forms equal the NSF point by point."""
    seq = SupportSequence(GTL(2, 2), "LR")
    stars = outputs(seq, 2)
    ps = np.array([[0.0, 0.5], [0.9, 1.0]])
    single = star_fidelity(seq, 2, stars[0], ps)
    joint = joint_fidelity(seq, 2, stars, ps)
    assert single.shape == joint.shape == ps.shape
    for index, p in np.ndenumerate(ps):
        state = run(seq, 2, noisy_state(seq.gtl.graph, p))
        assert single[index] == pytest.approx(fidelity(state, stars[:1]), abs=1e-12)
        assert joint[index] == pytest.approx(fidelity(state, stars), abs=1e-12)
