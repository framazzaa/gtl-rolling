"""Entanglement Rolling and the two-stage resolution, on the graphs alone."""

import itertools
import math

import networkx as nx
import noisy_graph_states.libs.graph as gt
import numpy as np
import pytest
import stim

from gtl_rolling import (
    GTL,
    SupportSequence,
    apply,
    local_complementation,
    measure_x,
    measure_y,
    measure_z,
    outputs,
    proximity,
    stage1,
    stage2,
)

CASES = [(2, 1), (3, 2), (4, 3)]
RANDOM_GRAPHS = [nx.gnp_random_graph(7, 0.5, seed=seed) for seed in range(4)]
PAULI = {"X": np.array([[0, 1], [1, 0]]), "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.diag([1.0, -1.0])}


def sqrt_i(pauli, sign):
    """The unitary sqrt(sign i P) = (1 + sign i P)/sqrt(2)."""
    return (np.eye(2) + sign * 1j * PAULI[pauli]) / np.sqrt(2)


def gate(simulator, unitary, qubit):
    """Applies a single-qubit unitary to a qubit of the simulator."""
    simulator.do_tableau(stim.Tableau.from_unitary_matrix(unitary, endian="little"), [qubit])


def graph_state(graph):
    """Stim simulator in the graph state of graph."""
    simulator = stim.TableauSimulator()
    simulator.h(*graph)
    for u, v in graph.edges:
        simulator.cz(u, v)
    return simulator


def correction(graph, kind, a, b0, minus):
    """Local unitary U, as {qubit: matrix}, such that the unmeasured qubits are in U |G'> after the measurement."""
    if kind == "z":
        return {b: PAULI["Z"] for b in graph[a]} if minus else {}
    if kind == "y":
        return {b: sqrt_i("Z", 1 if minus else -1) for b in graph[a]}
    flipped = set(graph[b0]) - set(graph[a]) - {a} if minus else set(graph[a]) - set(graph[b0]) - {b0}
    return {b0: sqrt_i("Y", -1 if minus else 1)} | {b: PAULI["Z"] for b in flipped}


@pytest.mark.parametrize("graph", RANDOM_GRAPHS)
def test_graph_rules_match_nsf(graph):
    """The graph rules give the same graphs as those of the NSF library, which its simulations use."""
    for v in graph:
        assert nx.utils.graphs_equal(local_complementation(graph, v), gt.local_complementation(graph, v))
        assert nx.utils.graphs_equal(measure_y(graph, v), gt.measure_y(graph, v))
        assert nx.utils.graphs_equal(measure_z(graph, v), gt.measure_z(graph, v))
        for b0 in graph[v]:
            assert nx.utils.graphs_equal(measure_x(graph, v, b0), gt.measure_x(graph, v, b0))


@pytest.mark.parametrize("graph", RANDOM_GRAPHS)
def test_graph_rules_physical(graph):
    """Each graph rule gives the state left by its operation, checked on the stabilizer tableau with Stim.

    Local complementation is sqrt(-i X_v) prod_{b in N_v} sqrt(i Z_b). For both outcomes of the Pauli-X (every
    support), Pauli-Y and Pauli-Z measurements, the unmeasured qubits are in the graph state of the rule up to the
    corrections of `correction`, and the measured qubit in the eigenstate of its outcome.
    """
    for v in graph:
        simulator = graph_state(graph)
        gate(simulator, sqrt_i("X", -1), v)
        for b in graph[v]:
            gate(simulator, sqrt_i("Z", 1), b)
        assert simulator.canonical_stabilizers() == graph_state(local_complementation(graph, v)).canonical_stabilizers()
        measurements = [("z", None, measure_z(graph, v)), ("y", None, measure_y(graph, v))]
        measurements += [("x", b0, measure_x(graph, v, b0)) for b0 in graph[v]]
        for (kind, b0, rule), minus in itertools.product(measurements, (False, True)):
            measured = graph_state(graph)
            getattr(measured, f"postselect_{kind}")(v, desired_value=minus)
            for q, u in correction(graph, kind, v, b0, minus).items():
                gate(measured, u.conj().T, q)
            expected = graph_state(rule)
            getattr(expected, f"reset_{kind}")(v)
            if minus:
                expected.x(v) if kind == "z" else expected.z(v)
            assert measured.canonical_stabilizers() == expected.canonical_stabilizers(), (kind, v, b0, minus)


@pytest.mark.parametrize("n_o, kb", CASES)
def test_rolling(n_o, kb):
    """Every step rolls the entanglement onto its support, for every support pattern.

    The final rolled set is R_B(o_j), with o_j the last orchestration qubit with a left support, or L_B(o_1) if there
    is none.
    """
    gtl = GTL(n_o, kb)
    group_of = {c: m for m in range(n_o + 1) for c in gtl.group(m)}
    for seq in SupportSequence.all(gtl):
        graphs = apply(gtl.graph, stage1(seq))
        used = []
        for i, side in enumerate(seq.pattern, start=1):
            before, after, o, b0 = graphs[i - 1], graphs[i], gtl.o(i), seq.supports[i - 1]
            N_o = set(before[o])
            assert N_o == set(seq.rolled_set(i - 1)) | set(gtl.R_B(i))
            assert {b0, *seq.non_support(i)} == set(gtl.R_B(i) if side == "R" else seq.rolled_set(i - 1))
            assert seq.rolled_set(i) == (seq.rolled_set(i - 1) if side == "R" else gtl.R_B(i))
            assert set(after[b0]) == N_o - {b0}
            assert not any(after.has_edge(u, v) for u, v in itertools.combinations(N_o - {b0}, 2))
            if i < n_o:
                assert set(seq.rolled_set(i)) <= set(after[gtl.o(i + 1)])
            assert {group_of[c] for c in seq.non_support(i)} <= {group_of[b0]}
            used += [b0, *seq.non_support(i)]
        j = max((i for i, side in enumerate(seq.pattern, start=1) if side == "L"), default=0)
        assert seq.rolled_set(n_o) == (gtl.R_B(j) if j else gtl.L_B(1))
        assert sorted(group_of[b0] for b0 in seq.supports) == [m for m in range(n_o + 1) if m != j]
        assert sorted(used + list(seq.rolled_set(n_o))) == list(gtl.V_c)


@pytest.mark.parametrize("n_o, kb", CASES)
def test_proximity_reduction(n_o, kb):
    """pi(c_i, c_j) Pauli-X measurements along a shortest path make c_i and c_j adjacent.

    Each measurement takes the next peer of the path as support and lowers the proximity by one.
    """
    gtl = GTL(n_o, kb)
    for c_i, c_j in itertools.permutations(gtl.V_c, 2):
        path = nx.shortest_path(gtl.graph, c_i, c_j)
        on_path = [v for v in path if v in gtl.V_o]
        assert proximity(gtl.graph, gtl.V_o, c_i, c_j) == len(on_path)
        graphs = apply(gtl.graph, [("x", o, path[path.index(o) + 1]) for o in on_path])
        for step in range(len(on_path) - 1):
            assert proximity(graphs[step + 1], gtl.V_o, c_i, c_j) == len(on_path) - step - 1
        assert graphs[-1].has_edge(c_i, c_j)


@pytest.mark.parametrize("n_o, kb", CASES)
def test_two_stage_outputs(n_o, kb):
    """kappa_b + n_o (kappa_b - k) Pauli-Z measurements after Stage 1 leave n_o disjoint k-qubit stars.

    After Stage 1 every peer of the final rolled set is adjacent exactly to the n_o supports, and the stars are
    centered on the supports.
    """
    gtl = GTL(n_o, kb)
    for seq in SupportSequence.all(gtl):
        after_stage1 = apply(gtl.graph, stage1(seq))[-1]
        assert all(set(after_stage1[c]) == set(seq.supports) for c in seq.rolled_set(n_o))
        for k in range(2, kb + 1):
            z = stage2(seq, k)
            assert len(z) == kb + n_o * (kb - k)
            final = apply(gtl.graph, stage1(seq) + z)[-1]
            stars = {frozenset((center, leaf)) for center, leaves in outputs(seq, k) for leaf in leaves}
            assert {frozenset(e) for e in final.edges} == stars
            assert [center for center, _ in outputs(seq, k)] == list(seq.supports)


@pytest.mark.parametrize("n_o, kb", [(2, 2), (3, 2), (2, 3)])
def test_reachable_configurations(n_o, kb):
    """Any supports reach (n_o + 1) kappa_b^n_o labeled graphs, with (n_o + 1) C(kappa_b, 2)^n_o Bell allocations.

    Every support at every step is tried. Each graph is fixed by the final rolled group and one center in each other
    group, adjacent to the other peers of its group and to the rolled group, and the Pauli-Z measurements of the
    other peers leave one pair per center.
    """
    gtl = GTL(n_o, kb)
    graphs = [gtl.graph]
    for o in gtl.V_o:
        graphs = [apply(graph, [("x", o, b0)])[-1] for graph in graphs for b0 in graph[o]]
    reached = {frozenset(frozenset(e) for e in graph.edges) for graph in graphs}

    groups = [set(gtl.group(m)) for m in range(n_o + 1)]
    predicted, allocations = set(), set()
    for rolled in groups:
        others = [group for group in groups if group is not rolled]
        for centers in itertools.product(*others):
            pairs_of = list(zip(others, centers, strict=True))
            edges = frozenset(frozenset((c, v)) for group, c in pairs_of for v in (group | rolled) - {c})
            predicted.add(edges)
            for leaves in itertools.product(*(group - {c} for group, c in pairs_of)):
                pairs = {frozenset(pair) for pair in zip(centers, leaves, strict=True)}
                assert {e for e in edges if e <= {*centers, *leaves}} == pairs
                allocations.add(frozenset(pairs))
    assert reached == predicted
    assert len(reached) == (n_o + 1) * kb**n_o
    assert len(allocations) == (n_o + 1) * math.comb(kb, 2) ** n_o
