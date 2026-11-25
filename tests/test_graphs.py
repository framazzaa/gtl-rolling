"""The structure of the GTL graphs."""

import networkx as nx
import pytest

from gtl_rolling import GTL

CASES = [(2, 1), (3, 2), (4, 3)]


@pytest.mark.parametrize("n_o, kb", CASES)
def test_gtl_structure(n_o, kb):
    """The graph meets every condition of a GTL graph, checked from the graph alone.

    The conditions:
        - the vertices split into n_o orchestration qubits V_o and kappa peers V_c, with n = n_o + kappa;
        - the graph is connected and two-colorable, with color classes V_o and V_c;
        - the bridges, i.e., the peers adjacent to more than one orchestration qubit, are adjacent to exactly two;
        - every orchestration qubit has kappa_c = 2 kappa_b peers;
        - every bridge is adjacent to exactly one consecutive pair (o_j, o_{j+1}), and each pair shares kappa_b bridges;
        - o_1 and o_{n_o} have kappa_b bridges, the others 2 kappa_b, and the other peers have degree one;
        - n_o = (kappa - kappa_b) / (kappa_c - kappa_b);
        - R_B(o_i) and L_B(o_i) are the kappa_b bridges shared with o_{i+1} and o_{i-1}, or at the two ends the peers
          that are not bridges, and they partition N(o_i);
        - one peer of each of the first n_o groups gives a unit bidiagonal n_o x n_o minor of the adjacency between
          V_o and V_c, so the adjacency matrix has rank 2 n_o over GF(2).
    """
    gtl = GTL(n_o, kb)
    G, V_o, V_c, B = gtl.graph, set(gtl.V_o), set(gtl.V_c), set(gtl.B)

    def orchestration_neighbors(v):
        return set(G[v]) & V_o

    def bridges_of(i):
        return set(G[gtl.o(i)]) & B

    assert set(G) == V_o | V_c and not V_o & V_c
    assert len(V_o) == n_o and gtl.n == n_o + gtl.kappa
    assert nx.is_connected(G)
    assert all((u in V_o) != (v in V_o) for u, v in G.edges)
    assert B == {c for c in V_c if len(orchestration_neighbors(c)) > 1}
    assert all(len(orchestration_neighbors(b)) == 2 for b in B)
    assert gtl.kappa_c == 2 * kb
    assert all(len(set(G[o]) & V_c) == gtl.kappa_c for o in V_o)
    pairs = [{gtl.o(j), gtl.o(j + 1)} for j in range(1, n_o)]
    assert all(orchestration_neighbors(b) in pairs for b in B)
    assert all(len(bridges_of(j) & bridges_of(j + 1)) == kb for j in range(1, n_o))
    assert [len(bridges_of(i)) for i in range(1, n_o + 1)] == [kb] + [2 * kb] * (n_o - 2) + [kb]
    assert all(G.degree(c) == 1 for c in V_c - B)
    assert gtl.kappa - kb == n_o * (gtl.kappa_c - kb)
    for i in range(1, n_o + 1):
        R, L, N = set(gtl.R_B(i)), set(gtl.L_B(i)), set(G[gtl.o(i)])
        assert R == (bridges_of(i) & bridges_of(i + 1) if i < n_o else N - B)
        assert L == (bridges_of(i) & bridges_of(i - 1) if i > 1 else N - B)
        assert len(R) == len(L) == kb and not R & L and R | L == N
    minor = [[G.has_edge(gtl.o(i), gtl.group(m)[0]) for m in range(n_o)] for i in range(1, n_o + 1)]
    assert minor == [[m in (i - 1, i) for m in range(n_o)] for i in range(1, n_o + 1)]
