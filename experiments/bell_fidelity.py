"""Single and joint fidelity of the Bell pairs of the two-stage resolution under depolarizing noise."""

import argparse
import math
import os

import numpy as np

from gtl_rolling import (
    GTL,
    SupportSequence,
    fidelity,
    joint_fidelity,
    noisy_state,
    outputs,
    proximity,
    run,
    star_fidelity,
)

OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
P_SINGLE = np.linspace(0.0, 1.0, 100)
P_JOINT = np.linspace(0.6, 1.0, 41)
NSF_MAX_PAIRS = 3
TOLERANCE = 1e-10


def fidelities(seq, p):
    """Single fidelities by the NSF, joint fidelity by its closed form, and the product of the single fidelities, at p.

    The single fidelities are checked against their closed form and, for at most NSF_MAX_PAIRS pairs, the joint
    fidelity against the NSF, whose density matrix grows exponentially with the number of pairs.
    """
    stars = outputs(seq, 2)
    out = run(seq, 2, noisy_state(seq.gtl.graph, p))
    single = [fidelity(out, [star]) for star in stars]
    joint = joint_fidelity(seq, 2, stars, p)
    checks = [(f, star_fidelity(seq, 2, star, p)) for f, star in zip(single, stars, strict=True)]
    if len(stars) <= NSF_MAX_PAIRS:
        checks.append((fidelity(out, stars), joint))
    deviation = max(abs(a - b) for a, b in checks)
    if not deviation < TOLERANCE:
        raise RuntimeError(f"NSF and closed form differ by {deviation:.1e} at p = {p}.")
    return single, joint, math.prod(single)


def pgfplots(rows):
    """The pgfplots figure of the single fidelities, the joint fidelity and their product against P_JOINT."""

    def coordinates(values):
        """The pgfplots coordinates of the values against P_JOINT."""
        return " ".join(f"({p:.3f},{v:.5f})" for p, v in zip(P_JOINT, values, strict=True))

    singles = []
    for m in range(len(rows[0][0])):
        style = "gray, opacity=0.7" + (", forget plot" if m else "")
        singles.append(rf"\addplot[{style}] coordinates {{{coordinates(r[0][m] for r in rows)}}};")
    singles.insert(1, r"\addlegendentry{single pairs}")
    lines = "\n".join(singles)
    return rf"""\begin{{tikzpicture}}
\begin{{axis}}[
    width=8.5cm, height=5.6cm,
    tick align=outside, tick pos=left,
    label style={{font=\small}}, tick label style={{font=\footnotesize}},
    xlabel={{Depolarizing parameter $p$}}, ylabel={{Fidelity}},
    xmin=0.6, xmax=1.0, ymin=0, ymax=1.02,
    xtick={{0.6,0.7,0.8,0.9,1.0}}, ytick={{0,0.25,0.5,0.75,1}},
    legend pos=north west, legend cell align=left, legend style={{font=\footnotesize, draw=none, fill=none}},
    every axis plot/.append style={{line width=0.8pt}},
]
{lines}
\addplot[black, line width=1.4pt] coordinates {{{coordinates(r[1] for r in rows)}}};
\addlegendentry{{all pairs at once (joint)}}
\addplot[black, dashed] coordinates {{{coordinates(r[2] for r in rows)}}};
\addlegendentry{{product of the single fidelities}}
\end{{axis}}
\end{{tikzpicture}}
"""


def main():
    """Computes, checks and writes the fidelities of the Bell pairs.

    The single fidelities are computed for p from 0 to 1, and the joint fidelity F_J, the probability that all the pairs
    are correct at once, for p from 0.6 to 1. The pairs are disjoint but correlated, since the noise of the peers of the
    final rolled set and of the orchestration qubits acts on several centers: prod_m F_m < F_J <= min_m F_m is checked
    at every point.

    Column F_i of the table is the pair of o_i, centered on b0^(o_i).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-o", type=int, default=3, help="orchestration qubits (default 3)")
    parser.add_argument("--kappa-b", type=int, default=2, help="minimum bridge degree (default 2)")
    parser.add_argument("--pattern", help="support sides, e.g. RLR (default: every support on the right)")
    args = parser.parse_args()
    gtl = GTL(args.n_o, args.kappa_b)
    seq = SupportSequence(gtl, args.pattern) if args.pattern else SupportSequence.right(gtl)
    default = seq == SupportSequence.right(GTL(3, 2))
    suffix = "" if default else f"_no{gtl.n_o}_kb{gtl.kappa_b}_{seq.pattern}"

    curves = np.array([fidelities(seq, p)[0] for p in P_SINGLE]).T
    rows = [fidelities(seq, p) for p in P_JOINT]
    for p, (single, joint, product) in zip(P_JOINT, rows, strict=True):
        if not (p == 1.0 or product < joint <= min(single)):
            raise RuntimeError(f"prod F_m < F_J <= min F_m is violated at p = {p}.")

    os.makedirs(OUTPUT, exist_ok=True)
    table = os.path.join(OUTPUT, f"bell_fidelity{suffix}.dat")
    with open(table, "w") as file:
        file.write(" ".join(["p", *(f"F_{i}" for i in range(1, gtl.n_o + 1))]) + "\n")
        file.writelines(" ".join(f"{x:.8e}" for x in (p, *f)) + "\n" for p, f in zip(P_SINGLE, curves.T, strict=True))
    figure = os.path.join(OUTPUT, f"joint_fidelity{suffix}.tex")
    with open(figure, "w") as file:
        file.write(pgfplots(rows))

    gamma = seq.rolled_set(gtl.n_o)[0]
    for i, ((center, _), curve) in enumerate(zip(outputs(seq, 2), curves, strict=True), start=1):
        pi = proximity(gtl.graph, gtl.V_o, center, gamma)
        print(f"pair of o_{i}: proximity {pi}, F = 1/2 at p = {np.interp(0.5, curve, P_SINGLE):.3f}")
    for p in (0.90, 0.98):
        single, joint, product = fidelities(seq, p)
        print(f"p = {p}: single {', '.join(f'{f:.3f}' for f in single)}; joint {joint:.3f}; product {product:.3f}")
    joint_check = ", the joint one equal to the NSF" if gtl.n_o <= NSF_MAX_PAIRS else ""
    print(
        f"single fidelities by the NSF, equal to the closed form{joint_check}; prod F_m < F_J <= min F_m at every "
        f"point; table: {table}; figure: {figure}"
    )


if __name__ == "__main__":
    main()
