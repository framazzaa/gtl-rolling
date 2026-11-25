"""Two-stage resolution with the noisy stabilizer formalism (NSF), and the fidelity of its outputs."""

from collections.abc import Sequence

import networkx as nx
import noisy_graph_states as ngs
import noisy_graph_states.libs.graph as gt
import numpy as np

from .rolling import SupportSequence, stage1, stage2

Star = tuple[int, tuple[int, ...]]


def run(seq: SupportSequence, k: int, state: ngs.State) -> ngs.State:
    """Returns the state after the two-stage resolution of a noisy GTL state with the NSF.

    Args:
        seq: Support sequence of Stage 1.
        k: Number of qubits of each output star, 2 <= k <= kappa_b.
        state: Noisy GTL state, e.g. from noisy_state(seq.gtl.graph, p, q).
    """
    sequence = tuple(stage1(seq) + stage2(seq, k))
    # autoload=False: the library never reads a cached strategy from disk.
    return ngs.Strategy(graph=seq.gtl.graph, sequence=sequence, autoload=False)(state)


def fidelity(state: ngs.State, stars: Sequence[Star]) -> float:
    """Returns the fidelity of output stars with the ideal star graph states, the measurement corrections applied.

    For several stars this is their joint fidelity.

    Args:
        state: NSF state after the measurements, e.g. from run.
        stars: Stars (center, leaves); each must be an isolated component of the graph of state, so that the other
            qubits can be traced out.

    Raises:
        ValueError: If there is no star, a star is not an isolated component of the graph, or a qubit occurs twice.
    """
    if not stars:
        raise ValueError("Give at least one star.")
    qubits = [q for center, leaves in stars for q in (center, *leaves)]
    if len(set(qubits)) != len(qubits):
        raise ValueError("Each qubit may occur only once in the fidelity target.")
    for center, leaves in stars:
        if set(state.graph[center]) != set(leaves) or any(set(state.graph[leaf]) != {center} for leaf in leaves):
            raise ValueError(f"Qubits {(center, *leaves)} do not form an isolated star.")
    ket = gt.graph_state(nx.relabel_nodes(state.graph.subgraph(qubits), {q: i for i, q in enumerate(qubits)}))
    rho = ngs.apply_nsf_maps_to_dm(ngs.reduce_maps(state, qubits), ket @ ket.conj().T, qubits)
    return float(np.real(ket.conj().T @ rho @ ket)[0, 0])
