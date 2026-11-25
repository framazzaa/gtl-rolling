"""Checks that every output of the two-stage resolution is fixed by the support sequence up to one bit."""

import itertools

import stim

from gtl_rolling import GTL, SupportSequence, nbar, outputs, stage1, stage2

INSTANCES = [(2, 2), (3, 2), (4, 2), (2, 3), (3, 3)]


def graph_state(gtl: GTL) -> stim.Circuit:
    """Circuit that prepares the GTL graph state."""
    circuit = stim.Circuit()
    circuit.append("RX", range(gtl.n))
    circuit.append("CZ", [q for edge in gtl.graph.edges for q in edge])
    return circuit


def random_outcomes(gtl: GTL, x_measured: list[int], z_measured: list[int]) -> bool:
    """Whether every measurement has a random outcome, given the outcomes of the measurements before it."""
    simulator = stim.TableauSimulator()
    simulator.do(graph_state(gtl))
    for q in x_measured:
        if simulator.peek_x(q):
            return False
        simulator.postselect_x(q, desired_value=False)
    for q in z_measured:
        if simulator.peek_z(q):
            return False
        simulator.postselect_z(q, desired_value=False)
    return True


def pauli(n: int, qubits: tuple[int, ...], basis: str) -> stim.PauliString:
    """Pauli string on n qubits with `basis` on `qubits` and the identity elsewhere."""
    string = stim.PauliString(n)
    for q in qubits:
        string[q] = basis
    return string


def expected_parity_set(seq: SupportSequence, k: int, star: tuple[int, ...]) -> set[int]:
    """Final rolled set, measured peers of the group of the star, and orchestration qubits whose noise reaches it."""
    gtl = seq.gtl
    center = star[0]
    rolled = set(seq.rolled_set(gtl.n_o))
    surplus = set(gtl.group(center // gtl.kappa_b)) - set(star)
    between = {gtl.o(i) for i in range(1, gtl.n_o + 1) if center in nbar(seq, i)}
    return rolled | surplus | between


def check(seq: SupportSequence, k: int) -> None:
    """Stops unless every k-qubit output of seq is fixed by the support sequence, up to one parity of outcomes.

    The resolution runs as a Stim circuit, with no correction between the measurements, and is checked exactly on the
    stabilizer tableau. Every outcome is random given the earlier ones, so no parity of outcomes is fixed in advance.
    X_u X_v, for every two qubits u, v of the star, stabilizes the output with sign +1, and Z...Z on the star with sign
    (-1)^beta, beta being the parity of the outcomes of expected_parity_set; as no parity is fixed, no other set of
    outcomes gives the sign. These k stabilizers act on the star only, so the output is the pure state
    (|+...+> + (-1)^beta |-...->)/sqrt(2): Phi+ if beta = 0 and Psi+ if beta = 1 for k = 2, corrected by one X.
    """
    gtl = seq.gtl
    x_measured = [o for _, o, _ in stage1(seq)]
    z_measured = [v for _, v in stage2(seq, k)]
    if not random_outcomes(gtl, x_measured, z_measured):
        raise RuntimeError(f"{seq.pattern}: a parity of the outcomes is fixed in advance.")
    circuit = graph_state(gtl)
    circuit.append("MX", x_measured)
    circuit.append("M", z_measured)
    record = {q: i for i, q in enumerate(x_measured + z_measured)}
    for center, leaves in outputs(seq, k):
        star = (center, *leaves)
        for u, v in itertools.combinations(star, 2):
            if not circuit.has_flow(stim.Flow(output=pauli(gtl.n, (u, v), "X"))):
                raise RuntimeError(f"{seq.pattern}, star {star}: X_u X_v depends on the outcomes.")
        parity = sorted(expected_parity_set(seq, k, star))
        if not set(parity) <= record.keys():
            raise RuntimeError(f"{seq.pattern}, star {star}: the parity of {parity} includes unmeasured qubits.")
        flow = stim.Flow(output=pauli(gtl.n, star, "Z"), measurements=[record[q] for q in parity])
        if not circuit.has_flow(flow):
            raise RuntimeError(f"{seq.pattern}, star {star}: Z...Z is not fixed by the parity of {parity}.")


def main():
    """Checks every instance, support sequence and output size."""
    for n_o, kappa_b in INSTANCES:
        gtl = GTL(n_o, kappa_b)
        for k in range(2, kappa_b + 1):
            sequences = SupportSequence.all(gtl)
            for seq in sequences:
                check(seq, k)
            seq = SupportSequence.right(gtl)
            bits = [
                " + ".join(gtl.label(q) for q in sorted(expected_parity_set(seq, k, (center, *leaves))))
                for center, leaves in outputs(seq, k)
            ]
            print(
                f"GTL({n_o}, {kappa_b}), {k}-qubit outputs: checked for {len(sequences)} support sequences; "
                f"supports on the right, beta = " + "; ".join(bits)
            )


if __name__ == "__main__":
    main()
