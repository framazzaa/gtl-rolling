"""Entanglement Rolling against the Y/Z resolution, for the same number of Bell pairs and under the same noise."""

import argparse
import csv
import os

import noisy_graph_states as ngs

from gtl_rolling import GTL, SupportSequence, fidelity, noisy_state, outputs, stage1, stage2, star_fidelity

OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
# Final rolled set in the middle of the backbone, with a left support at step ceil(N/2) and right supports elsewhere
# (proximities of the pairs 1, 1, 2, 2, ...), or L_B(o_1) at its left end, with right supports only (1, 2, ..., N).
SUPPORTS = {
    "middle": lambda n_o: "R" * ((n_o + 1) // 2 - 1) + "L" + "R" * (n_o - (n_o + 1) // 2),
    "right": lambda n_o: "R" * n_o,
}
TOLERANCE = 1e-10


def simulate(gtl, sequence, pairs, p):
    """Fidelity of every pair after the measurements of `sequence`, under depolarizing noise p on every qubit."""
    out = ngs.Strategy(graph=gtl.graph, sequence=tuple(sequence), autoload=False)(noisy_state(gtl.graph, p))
    return [fidelity(out, [pair]) for pair in pairs]


def rolling(gtl, pattern, p):
    """Measurements at Tier2 and at Tier1, and fidelity of every Bell pair, of Entanglement Rolling.

    Tier2 measures o_1, ..., o_{n_o} in the X basis, and the Tier1 nodes measure every peer outside the outputs in the Z
    basis, leaving n_o Bell pairs placed by the support sequence.
    """
    seq = SupportSequence(gtl, pattern)
    tier2, tier1, pairs = stage1(seq), stage2(seq, 2), outputs(seq, 2)
    single = simulate(gtl, tier2 + tier1, pairs, p)
    deviation = max(abs(f - star_fidelity(seq, 2, pair, p)) for f, pair in zip(single, pairs, strict=True))
    if not deviation < TOLERANCE:
        raise RuntimeError(f"NSF and closed form differ by {deviation:.1e}.")
    return len(tier2), len(tier1), single


def yz(gtl, p):
    """Measurements at Tier2 and at Tier1, and fidelity of every Bell pair, of the Y/Z resolution.

    Tier2 measures o_1, o_3, ... in the Y basis and o_2, o_4, ... in the Z basis. Each Y measurement joins the two peer
    groups of its orchestration qubit into one entangled state of 2 kappa_b peers, from which the Tier1 nodes keep the
    first peers of the two groups as a Bell pair by measuring the others in the Z basis: ceil(n_o/2) Bell pairs. On
    the chain (kappa_b = 1) these states are already Bell pairs, and Tier1 measures nothing.
    """
    tier2 = [("y" if i % 2 else "z", gtl.o(i)) for i in range(1, gtl.n_o + 1)]
    pairs = [(gtl.L_B(i)[0], (gtl.R_B(i)[0],)) for i in range(1, gtl.n_o + 1, 2)]
    kept = {v for center, leaves in pairs for v in (center, *leaves)}
    tier1 = [("z", c) for c in gtl.V_c if c not in kept]
    return len(tier2), len(tier1), simulate(gtl, tier2 + tier1, pairs, p)


def compare(n_pairs, kappa_b, supports, ps):
    """One row for each scheme and each p, for N = n_pairs Bell pairs.

    Rolling runs on GTL(N, kappa_b), and the Y/Z resolution on GTL(2N - 1, kappa_b) and on the chain GTL(2N - 1, 1).
    The fidelities are computed with the NSF simulation, and those of Rolling are checked against the closed form. The
    latency runs from the end of the setup, at 2 tau with the acknowledgments of the peers at Tier2, to the receipt of
    the correction information at the endpoints, with the schedule of gtl_rolling.timing, in units of the one-way delay
    tau and neglecting the local rounds: 3 tau when the Tier1 nodes measure, and tau otherwise, when the outcomes of
    Tier2 go to the endpoints directly.
    """
    pattern = SUPPORTS[supports](n_pairs)
    schemes = (
        ("rolling", GTL(n_pairs, kappa_b), lambda gtl, p: rolling(gtl, pattern, p)),
        ("yz", GTL(2 * n_pairs - 1, kappa_b), yz),
        ("yz", GTL(2 * n_pairs - 1, 1), yz),
    )
    rows = []
    for method, gtl, scheme in schemes:
        tier2, tier1, noiseless = scheme(gtl, 1.0)
        label = f"{method}, GTL({gtl.n_o}, {gtl.kappa_b})"
        if len(noiseless) != n_pairs or not all(abs(f - 1) < TOLERANCE for f in noiseless):
            raise RuntimeError(f"{label}: the noiseless resolution does not give {n_pairs} Bell pairs.")
        for p in ps:
            single = scheme(gtl, p)[2]
            rows.append(
                {
                    "pairs": n_pairs,
                    "method": method,
                    "n_o": gtl.n_o,
                    "kappa_b": gtl.kappa_b,
                    "qubits": gtl.n,
                    "tier2_measurements": tier2,
                    "tier1_measurements": tier1,
                    "latency_tau": 3 if tier1 else 1,
                    "p": p,
                    "fidelity_min": round(min(single), 6),
                    "fidelity_max": round(max(single), 6),
                }
            )
    return rows


def main():
    """Compares the two resolutions and writes the table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, nargs="+", default=[3, 6, 9], help="numbers of Bell pairs (default 3 6 9)")
    parser.add_argument("--kappa-b", type=int, default=2, help="minimum bridge degree, at least 2 (default 2)")
    parser.add_argument("--supports", choices=SUPPORTS, default="middle", help="supports of Rolling (default middle)")
    parser.add_argument("--p", type=float, nargs="+", default=[0.9, 0.98], help="depolarizing p (default 0.9 0.98)")
    args = parser.parse_args()
    if args.kappa_b < 2 or min(args.pairs) < 2:
        parser.error("kappa_b >= 2 and at least 2 Bell pairs are required.")

    rows = [row for n_pairs in args.pairs for row in compare(n_pairs, args.kappa_b, args.supports, args.p)]
    custom = args.pairs != parser.get_default("pairs") or args.p != parser.get_default("p")
    suffix = "_custom" if custom else ""
    path = os.path.join(OUTPUT, f"yz_resolution_kb{args.kappa_b}_{args.supports}{suffix}.csv")
    os.makedirs(OUTPUT, exist_ok=True)
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(
            f"{r['pairs']} pairs, {r['method']} on GTL({r['n_o']}, {r['kappa_b']}): {r['qubits']} qubits, "
            f"{r['tier2_measurements']} + {r['tier1_measurements']} measurements, {r['latency_tau']} tau; "
            f"p = {r['p']}: F = {r['fidelity_min']:.3f}-{r['fidelity_max']:.3f}"
        )
    print(f"Rolling agrees with the closed form; table: {path}")


if __name__ == "__main__":
    main()
