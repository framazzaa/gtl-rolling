"""Noise maps of every qubit after Stage 1 against their closed form, on larger instances than the tests."""

import argparse
import itertools
import random

import noisy_graph_states as ngs

from gtl_rolling import GTL, SupportSequence, nbar, noise_operator, stage1

N_O = range(2, 9)
KAPPA_B = range(2, 6)
MAX_PATTERNS = 64
ERRORS = ((1, 0), (0, 1), (1, 1))


def patterns(n_o, rng):
    """Every left/right support pattern of n_o supports, or MAX_PATTERNS random ones when there are more."""
    every = ["".join(p) for p in itertools.product("RL", repeat=n_o)]
    return every if len(every) <= MAX_PATTERNS else rng.sample(every, MAX_PATTERNS)


def propagated(gtl, seq):
    """Every noise operator of every qubit after Stage 1, propagated in a single run of the library."""
    keys = [(v, e) for v in gtl.graph for e in ERRORS]
    maps = [ngs.Map(weights=[1.0], noises=[tuple(sorted(noise_operator(gtl.graph, v, *e)))]) for v, e in keys]
    state = ngs.State(graph=gtl.graph, maps=maps)
    for _, o, b0 in stage1(seq):
        state = ngs.x_measurement(state, o, b0)
    return {key: frozenset(m.noises[0]) for key, m in zip(keys, state.maps, strict=True)}


def closed_form(gtl, seq):
    """The same operators from the closed form."""
    gamma = frozenset(seq.rolled_set(gtl.n_o))
    n_tilde = {c: frozenset(seq.supports) for c in gamma}
    for m, b0 in enumerate(seq.supports, start=1):
        n_tilde[b0] = frozenset(seq.non_support(m)) | gamma
        n_tilde |= {c: frozenset({b0}) for c in seq.non_support(m)}
    expected = {}
    for c, neighbors in n_tilde.items():
        for alpha, beta in ERRORS:
            a, b = (beta, alpha) if c in seq.supports else (alpha, beta)
            expected[(c, (alpha, beta))] = (frozenset({c}) if a else frozenset()) | (neighbors if b else frozenset())
    for i in range(1, gtl.n_o + 1):
        for alpha, beta in ERRORS:
            expected[(gtl.o(i), (alpha, beta))] = nbar(seq, i) if alpha else frozenset()
    return expected


def main():
    """Compares the propagated noise maps with the closed form.

    For every qubit, its three noise operators, the Z, X and Y errors with (alpha, beta) = (1, 0), (0, 1) and (1, 1),
    are propagated through Stage 1 with the update rules of the library and compared with the closed form:
    Z_c^alpha prod_{k in N~} Z_k^beta for a peer c (alpha and beta swapped for a support), and Z^alpha on nbar(o_i) for
    an orchestration qubit o_i. Instances: n_o = 2..8, kappa_b = 2..5, every pattern of left and right supports, or 64
    random ones (seed 0) when there are more.
    """
    argparse.ArgumentParser(description=__doc__).parse_args()
    rng = random.Random(0)
    peers = orchestration = 0
    for n_o, kb in itertools.product(N_O, KAPPA_B):
        gtl = GTL(n_o, kb)
        for pattern in patterns(n_o, rng):
            seq = SupportSequence(gtl, pattern)
            if propagated(gtl, seq) != closed_form(gtl, seq):
                raise RuntimeError(f"GTL({n_o}, {kb}), supports {pattern}: maps differ from the closed form.")
            peers += len(gtl.V_c)
            orchestration += n_o
    print(f"{peers} peer maps and {orchestration} orchestration-qubit maps coincide with the closed form")


if __name__ == "__main__":
    main()
