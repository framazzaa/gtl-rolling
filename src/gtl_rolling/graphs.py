"""Generalized tree-like (GTL) graph states."""

import numbers
from dataclasses import dataclass
from functools import cached_property

import networkx as nx


@dataclass(frozen=True)
class GTL:
    """GTL graph state (G, kappa_b) with n_o >= 2 orchestration qubits and n_o + 1 groups of kappa_b peers.

    o_i is adjacent to the peers of groups L_B(o_i) = i - 1 and R_B(o_i) = i. Qubits are numbered peers first, group by
    group, then o_1, ..., o_{n_o}.

    Attributes:
        n_o: Number of orchestration qubits.
        kappa_b: Minimum bridge degree hat(kappa)_b, i.e., the number of peers in each group.

    Raises:
        TypeError: If n_o or kappa_b is not an integer.
        ValueError: If n_o < 2 or kappa_b < 1.
    """

    n_o: int
    kappa_b: int

    def __post_init__(self) -> None:
        if not isinstance(self.n_o, numbers.Integral) or not isinstance(self.kappa_b, numbers.Integral):
            raise TypeError("n_o and kappa_b must be integers.")
        if self.n_o < 2:
            raise ValueError("A GTL graph state needs n_o >= 2 orchestration qubits.")
        if self.kappa_b < 1:
            raise ValueError("kappa_b must be positive and nonzero.")

    @property
    def kappa(self) -> int:
        """Number of peer qubits, kappa = (n_o + 1) kappa_b."""
        return (self.n_o + 1) * self.kappa_b

    @property
    def kappa_c(self) -> int:
        """Peer degree: number of peers adjacent to each orchestration qubit."""
        return 2 * self.kappa_b

    @property
    def n(self) -> int:
        """Total number of qubits, n = n_o + kappa."""
        return self.kappa + self.n_o

    @property
    def V_c(self) -> tuple[int, ...]:
        """Peer qubits c_1, ..., c_kappa."""
        return tuple(range(self.kappa))

    @property
    def V_o(self) -> tuple[int, ...]:
        """Orchestration qubits o_1, ..., o_{n_o}."""
        return tuple(range(self.kappa, self.n))

    def o(self, i: int) -> int:
        """Returns the vertex of o_i.

        Args:
            i: Index of the orchestration qubit, 1 <= i <= n_o.

        Raises:
            IndexError: If o_i does not exist.
        """
        if not 1 <= i <= self.n_o:
            raise IndexError(f"o_{i} does not exist for n_o = {self.n_o}.")
        return self.kappa + i - 1

    def group(self, m: int) -> tuple[int, ...]:
        """Returns the peers of group m.

        Args:
            m: Index of the group, 0 <= m <= n_o.

        Raises:
            IndexError: If the group does not exist.
        """
        if not 0 <= m <= self.n_o:
            raise IndexError(f"Group {m} does not exist for n_o = {self.n_o}.")
        return tuple(range(m * self.kappa_b, (m + 1) * self.kappa_b))

    def R_B(self, i: int) -> tuple[int, ...]:
        """Returns the right bridge neighborhood of o_i, i.e., group i.

        Args:
            i: Index of the orchestration qubit, 1 <= i <= n_o.

        Raises:
            IndexError: If o_i does not exist.
        """
        self.o(i)
        return self.group(i)

    def L_B(self, i: int) -> tuple[int, ...]:
        """Returns the left bridge neighborhood of o_i, i.e., group i - 1.

        Args:
            i: Index of the orchestration qubit, 1 <= i <= n_o.

        Raises:
            IndexError: If o_i does not exist.
        """
        self.o(i)
        return self.group(i - 1)

    @property
    def B(self) -> tuple[int, ...]:
        """Bridges: the peers adjacent to two orchestration qubits (groups 1, ..., n_o - 1)."""
        return tuple(c for m in range(1, self.n_o) for c in self.group(m))

    @cached_property
    def graph(self) -> nx.Graph:
        """Graph G, frozen since every computation shares it."""
        G = nx.Graph()
        G.add_nodes_from(range(self.n))
        for i in range(1, self.n_o + 1):
            G.add_edges_from((self.o(i), c) for c in self.L_B(i) + self.R_B(i))
        return nx.freeze(G)

    def label(self, v: int) -> str:
        """Returns the name of vertex v: c_j for the peer c_j, o_i for o_i."""
        return f"c_{v + 1}" if v < self.kappa else f"o_{v - self.kappa + 1}"
