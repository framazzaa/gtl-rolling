"""Checks that the outputs of the two-stage resolution do not depend on the order of the measurements."""

import argparse
import itertools

import noisy_graph_states as ngs

from gtl_rolling import (
    GTL,
    SupportSequence,
    fidelity,
    measure_x,
    measure_z,
    noisy_state,
    outputs,
    stage1,
    stage2,
    star_fidelity,
)

P = 0.9
TOLERANCE = 1e-12


def with_supports(graph, order):
    """The order with a valid support for each Pauli-X measurement, and the number of replaced supports.

    The support is adjacent to its orchestration qubit when the latter is measured, and never measured afterwards: the
    support of the sequence or, if it is not adjacent yet, another neighbor. The non-Pauli corrections act on the
    supports only, and hence leave the bases of the later measurements unchanged.
    """
    resolved, replaced = [], 0
    for i, instruction in enumerate(order):
        if instruction[0] == "z":
            graph = measure_z(graph, instruction[1])
            resolved.append(instruction)
            continue
        _, o, b0 = instruction
        later = {step[1] for step in order[i + 1 :]}
        if not graph.has_edge(o, b0):
            b0 = next((v for v in graph[o] if v not in later), None)
            replaced += 1
        if b0 is None or b0 in later:
            raise RuntimeError(f"No support of {o} is left unmeasured in {order}.")
        graph = measure_x(graph, o, b0)
        resolved.append(("x", o, b0))
    return resolved, replaced


def run_order(gtl, order):
    """NSF state after the measurements of order, under depolarizing noise P on every qubit."""
    return ngs.Strategy(graph=gtl.graph, sequence=tuple(order), autoload=False)(noisy_state(gtl.graph, P))


def signature(state):
    """Graph and noise maps, in a form that does not depend on how the library lists them."""
    maps = tuple(
        tuple(sorted((round(w, 12), tuple(sorted(n))) for w, n in zip(m.weights, m.noises, strict=True)))
        for m in state.maps
    )
    return frozenset(frozenset(e) for e in state.graph.edges()), maps


def main():
    """Compares every order of the measurements with the left-to-right one.

    For every support pattern and every order of the measurements of both stages (Bell pairs), interleaved orders
    included, the NSF simulation under depolarizing noise must give the graph and the noise maps of the left-to-right
    order, whose fidelities agree with the closed form.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", nargs="+", default=["2,2", "3,2", "2,3"], help="n_o,kappa_b pairs")
    args = parser.parse_args()

    total = 0
    for n_o, kb in (map(int, instance.split(",")) for instance in args.instances):
        gtl, count, replaced = GTL(n_o, kb), 0, 0
        for seq in SupportSequence.all(gtl):
            reference = stage1(seq) + stage2(seq, 2)
            out = run_order(gtl, reference)
            for star in outputs(seq, 2):
                if not abs(fidelity(out, [star]) - star_fidelity(seq, 2, star, P)) < TOLERANCE:
                    raise RuntimeError(f"GTL({n_o}, {kb}), {seq.pattern}: NSF and closed form differ on {star}.")
            expected = signature(out)
            for order in itertools.permutations(reference):
                resolved, changed = with_supports(gtl.graph, order)
                if signature(run_order(gtl, resolved)) != expected:
                    raise RuntimeError(f"GTL({n_o}, {kb}), {seq.pattern}: {order} gives another state.")
                count += 1
                replaced += changed > 0
        print(
            f"GTL({n_o}, {kb}): {count} orders ({replaced} with another support), every support pattern, same graph "
            "and noise maps after Stage 2"
        )
        total += count
    print(f"{total} orders in total; the fidelities of the left-to-right order agree with the closed form")


if __name__ == "__main__":
    main()
