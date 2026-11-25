"""Noise channels and the propagation of Pauli errors through the measurements."""

import math
from collections.abc import Iterable, Mapping, Sequence

import networkx as nx
import noisy_graph_states as ngs

from .rolling import Instruction, SupportSequence


def nbar(seq: SupportSequence, i: int) -> frozenset[int]:
    """Returns bar(N), the support vertices on which a Z or Y error of o_i acts after Stage 1.

    The supports are {b0^(o_l): l >= i} if every support from o_i on is on the right; otherwise, with o_j the first of
    them on the left, {b0^(o_j)} | {b0^(o_l): l < i}.

    Args:
        seq: Support sequence of Stage 1.
        i: Index of the orchestration qubit, 1 <= i <= n_o.

    Raises:
        IndexError: If o_i does not exist.
    """
    seq.gtl.o(i)  # The slices below would silently accept an index out of range.
    later = seq.pattern[i - 1 :]
    if "L" not in later:
        return frozenset(seq.supports[i - 1 :])
    j = i + later.index("L")
    return frozenset((seq.supports[j - 1],) + seq.supports[: i - 1])


def depolarizing(p: float) -> tuple[float, float, float, float]:
    """Returns the probabilities of I, X, Y and Z of the depolarizing channel.

    The channel is p rho + (1 - p)/4 sum_P P rho P, P in {I, X, Y, Z}.

    Args:
        p: Depolarizing parameter, 1 for no noise and 0 for the completely depolarizing channel.

    Raises:
        ValueError: If p is outside [0, 1].
    """
    if not 0 <= p <= 1:
        raise ValueError(f"p = {p} is outside [0, 1].")
    return (p + (1 - p) / 4, (1 - p) / 4, (1 - p) / 4, (1 - p) / 4)


def dephasing(t: float, T: float) -> float:
    """Returns the phase-flip probability q(t) = (1 - exp(-t/T))/2 of a memory with dephasing time T.

    Args:
        t: Storage time, finite and nonnegative.
        T: Dephasing time of the memory, positive; inf for a perfect memory.

    Raises:
        ValueError: If t < 0 or T <= 0, or either is nan.
    """
    if not (t >= 0 and T > 0):
        raise ValueError(f"t = {t} and T = {T}: t >= 0 and T > 0 are required.")
    return 0.0 if math.isinf(T) else -math.expm1(-t / T) / 2


def noisy_state(
    graph: nx.Graph, p: float | Mapping[int, float] = 1.0, q: float | Mapping[int, float] = 0.0
) -> ngs.State:
    """Returns the graph state with depolarizing noise p and a phase flip of probability q on each qubit.

    Args:
        graph: Graph of the state.
        p: Depolarizing parameter, one value for every qubit or a mapping per qubit; qubits missing from a mapping
            have no depolarizing noise.
        q: Phase-flip probability, one value for every qubit or a mapping per qubit; qubits missing from a mapping
            have no phase flip.

    Raises:
        ValueError: If a value of p or q is outside [0, 1], or a mapping has a qubit outside the graph.
    """
    for values in (p, q):
        if isinstance(values, Mapping) and not set(values) <= set(graph):
            raise ValueError(f"Noise for qubits outside the graph: {sorted(set(values) - set(graph))}.")
    state = ngs.State(graph=graph, maps=[])
    for v in graph.nodes:
        p_v = p.get(v, 1.0) if isinstance(p, Mapping) else p
        q_v = q.get(v, 0.0) if isinstance(q, Mapping) else q
        weights = depolarizing(p_v)
        if not 0 <= q_v <= 1:
            raise ValueError(f"q = {q_v} on qubit {v} is outside [0, 1].")
        if p_v < 1:
            state = ngs.pauli_noise(state, [v], weights)
        if q_v > 0:
            state = ngs.z_noise(state, [v], q_v)
    return state


def noise_operator(graph: nx.Graph, a: int, alpha: int, beta: int) -> frozenset[int]:
    """Returns the qubits of the noise operator Lambda_{a, alpha beta} = Z_a^alpha prod_{b in N_a} Z_b^beta of qubit a.

    (alpha, beta) = (1, 0), (0, 1) and (1, 1) give a Z, an X and a Y error on a.

    Args:
        graph: Graph of the state.
        a: The qubit.
        alpha: Exponent of Z_a, 0 or 1.
        beta: Exponent of the Z operators on the neighbors of a, 0 or 1.

    Raises:
        ValueError: If alpha or beta is not 0 or 1.
    """
    if alpha not in (0, 1) or beta not in (0, 1):
        raise ValueError("alpha and beta must be 0 or 1.")
    qubits = {a} if alpha else set()
    if beta:
        qubits |= set(graph[a])
    return frozenset(qubits)


def propagate(graph: nx.Graph, sequence: Sequence[Instruction], noise: Iterable[int]) -> frozenset[int]:
    """Returns the qubits on which a noise operator acts as Z after a sequence of measurements, by the NSF library.

    Args:
        graph: Graph of the state before the measurements.
        sequence: Measurement instructions, applied in order.
        noise: Qubits on which the operator acts as Z before the measurements.

    Raises:
        ValueError: If a qubit of noise is outside the graph.
    """
    noise = tuple(sorted(noise))
    if not set(noise) <= set(graph):
        raise ValueError(f"Noise on qubits outside the graph: {sorted(set(noise) - set(graph))}.")
    state = ngs.State(graph=graph, maps=[ngs.Map(weights=[1.0], noises=[noise])])
    state = ngs.Strategy(graph=graph, sequence=tuple(sequence), autoload=False)(state)
    return frozenset(state.maps[0].noises[0])
