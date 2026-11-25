"""Support sequences, the measurements of the two-stage resolution, and the graph rules of Pauli measurements."""

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property

import networkx as nx

from .graphs import GTL

type Instruction = tuple[str, int] | tuple[str, int, int]


@dataclass(frozen=True)
class SupportSequence:
    """Support vertices b0^(o_1), ..., b0^(o_{n_o}) of Stage 1, given by a pattern of "R" and "L", one for each o_i.

    Attributes:
        gtl: The GTL graph state.
        pattern: The side, "L" or "R", of the support of o_1, ..., o_{n_o}.

    Raises:
        ValueError: If pattern is not a string of n_o characters "L" or "R".
    """

    gtl: GTL
    pattern: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.pattern, str)
            or len(self.pattern) != self.gtl.n_o
            or not set(self.pattern) <= {"L", "R"}
        ):
            raise ValueError(f"pattern must be a string of {self.gtl.n_o} characters 'L' or 'R'.")

    @classmethod
    def right(cls, gtl: GTL) -> "SupportSequence":
        """Returns the support sequence of gtl with every support on the right."""
        return cls(gtl, "R" * gtl.n_o)

    @classmethod
    def all(cls, gtl: GTL) -> list["SupportSequence"]:
        """Returns every support sequence of gtl, one for each of the 2^n_o patterns."""
        return [cls(gtl, "".join(sides)) for sides in itertools.product("RL", repeat=gtl.n_o)]

    @cached_property
    def _steps(self) -> tuple[tuple[int, tuple[int, ...], tuple[int, ...]], ...]:
        """(support, other peers of its side, rolled set after the step) for o_1, ..., o_{n_o}."""
        steps, gamma = [], self.gtl.L_B(1)
        for i, side in enumerate(self.pattern, start=1):
            side_set = self.gtl.R_B(i) if side == "R" else gamma
            if side == "L":
                gamma = self.gtl.R_B(i)
            steps.append((side_set[0], side_set[1:], gamma))
        return tuple(steps)

    @property
    def supports(self) -> tuple[int, ...]:
        """Support vertices b0^(o_1), ..., b0^(o_{n_o})."""
        return tuple(step[0] for step in self._steps)

    def non_support(self, i: int) -> tuple[int, ...]:
        """Returns bar(S)_b0^(o_i), the other peers of the side of b0^(o_i), which give the leaves of the i-th star.

        Args:
            i: Index of the orchestration qubit, 1 <= i <= n_o.

        Raises:
            IndexError: If o_i does not exist.
        """
        self.gtl.o(i)
        return self._steps[i - 1][1]

    def rolled_set(self, i: int) -> tuple[int, ...]:
        """Returns the rolled set Gamma^(o_i) after the measurement of o_i, with Gamma^(o_0) = L_B(o_1).

        Args:
            i: Index of the orchestration qubit, 0 <= i <= n_o.

        Raises:
            IndexError: If i is out of range.
        """
        if not 0 <= i <= self.gtl.n_o:
            raise IndexError(f"Gamma^(o_{i}) does not exist for n_o = {self.gtl.n_o}.")
        return self.gtl.L_B(1) if i == 0 else self._steps[i - 1][2]


def stage1(seq: SupportSequence) -> list[Instruction]:
    """Returns the Pauli-X measurements ("x", o_i, b0^(o_i)) of Stage 1, for o_1, ..., o_{n_o} in this order."""
    return [("x", seq.gtl.o(i), b0) for i, b0 in enumerate(seq.supports, start=1)]


def outputs(seq: SupportSequence, k: int) -> list[tuple[int, tuple[int, ...]]]:
    """Returns the output stars of k qubits as (center, leaves); k = 2 gives Bell pairs.

    The star of o_i is centered on b0^(o_i), with the first k - 1 peers of bar(S)_b0^(o_i) as leaves.

    Args:
        seq: Support sequence of Stage 1.
        k: Number of qubits of each star, 2 <= k <= kappa_b.

    Raises:
        ValueError: If k is out of range.
    """
    if not 2 <= k <= seq.gtl.kappa_b:
        raise ValueError("2 <= k <= kappa_b is required.")
    return [(b0, seq.non_support(i)[: k - 1]) for i, b0 in enumerate(seq.supports, start=1)]


def stage2(seq: SupportSequence, k: int) -> list[Instruction]:
    """Returns the Pauli-Z measurements of Stage 2, on the kappa_b + n_o (kappa_b - k) peers outside the output stars.

    Args:
        seq: Support sequence of Stage 1.
        k: Number of qubits of each output star, 2 <= k <= kappa_b.

    Raises:
        ValueError: If k is out of range.
    """
    kept = {v for center, leaves in outputs(seq, k) for v in (center, *leaves)}
    return [("z", c) for c in seq.gtl.V_c if c not in kept]


# As in the NSF library, a measured vertex stays in the graph, isolated.


def local_complementation(graph: nx.Graph, v: int) -> nx.Graph:
    """Returns the local complementation tau_v(G), in which every edge between two neighbors of v is toggled.

    Args:
        graph: The graph G.
        v: The vertex.
    """
    result = nx.Graph(graph)
    for a, b in itertools.combinations(graph[v], 2):
        if result.has_edge(a, b):
            result.remove_edge(a, b)
        else:
            result.add_edge(a, b)
    return result


def measure_z(graph: nx.Graph, a: int) -> nx.Graph:
    """Returns G - a, the graph left by a Pauli-Z measurement of a.

    Args:
        graph: The graph G.
        a: The measured vertex.
    """
    result = nx.Graph(graph)
    result.remove_edges_from(list(graph.edges(a)))
    return result


def measure_y(graph: nx.Graph, a: int) -> nx.Graph:
    """Returns tau_a(G) - a, the graph left by a Pauli-Y measurement of a.

    Args:
        graph: The graph G.
        a: The measured vertex.
    """
    return measure_z(local_complementation(graph, a), a)


def measure_x(graph: nx.Graph, a: int, b0: int) -> nx.Graph:
    """Returns tau_b0(tau_a(tau_b0(G)) - a), the graph left by a Pauli-X measurement of a with support b0.

    Args:
        graph: The graph G.
        a: The measured vertex.
        b0: The support, a neighbor of a.

    Raises:
        ValueError: If b0 is not a neighbor of a.
    """
    if b0 not in graph[a]:
        raise ValueError(f"The support {b0} is not a neighbor of {a}.")
    return local_complementation(measure_z(local_complementation(local_complementation(graph, b0), a), a), b0)


_RULES = {"x": measure_x, "y": measure_y, "z": measure_z}


def apply(graph: nx.Graph, sequence: Sequence[Instruction]) -> list[nx.Graph]:
    """Returns the input graph and the graph after each measurement, by the graph rules of the Pauli measurements.

    Args:
        graph: Graph of the input graph state.
        sequence: Measurement instructions, applied in order: ("x", a, b0), a Pauli-X measurement of a with support
            b0, ("y", a) or ("z", a), a Pauli-Y or Pauli-Z measurement of a.

    Raises:
        ValueError: If an instruction is not a Pauli measurement, or the support of a Pauli-X measurement is not a
            neighbor of the measured vertex.
    """
    graphs = [graph]
    for kind, *args in sequence:
        if kind not in _RULES:
            raise ValueError(f"Unknown instruction {(kind, *args)}.")
        graphs.append(_RULES[kind](graphs[-1], *args))
    return graphs


def proximity(graph: nx.Graph, V_o: Sequence[int], c_i: int, c_j: int) -> int:
    """Returns the peer proximity pi(c_i, c_j), one plus the number of bridges strictly inside a shortest path.

    A bridge is a peer adjacent to more than one orchestration qubit in graph.

    Args:
        graph: Graph of the state.
        V_o: Orchestration qubits.
        c_i: First peer.
        c_j: Second peer, other than c_i.

    Raises:
        ValueError: If c_i and c_j are not two distinct peers.
    """
    V_o = set(V_o)
    if c_i == c_j or c_i in V_o or c_j in V_o:
        raise ValueError(f"{c_i} and {c_j} must be two distinct peers.")
    path = nx.shortest_path(graph, c_i, c_j)
    return 1 + sum(1 for v in path[1:-1] if v not in V_o and len(V_o.intersection(graph[v])) > 1)
