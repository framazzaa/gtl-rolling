"""Exhaustive search for the fewest Pauli measurements that instantiate n_o Bell pairs from a GTL graph state."""

import argparse
import collections
import csv
import itertools
import math
import multiprocessing
import os
import subprocess
import tempfile

import networkx as nx

from gtl_rolling import GTL, measure_x, measure_y, measure_z

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, "output")
INSTANCES = (
    [(2, kappa_b) for kappa_b in range(2, 10)]
    + [(3, kappa_b) for kappa_b in range(2, 8)]
    + [(4, kappa_b) for kappa_b in range(2, 6)]
    + [(5, 2), (5, 3), (6, 2)]
)
MAX_QUBITS = 32
REFERENCE_MAX_QUBITS = 11
OFFSETS = (-12, 6)


def compile_search(directory):
    """Compile the C program into `directory` and return the path of the binary."""
    binary = os.path.join(directory, "minimum_measurements")
    subprocess.run(["cc", "-O3", "-o", binary, os.path.join(HERE, "minimum_measurements.c")], check=True)
    return binary


def search(binary, gtl):
    """Runs the C search on gtl, one process per CPU.

    Every qubit is left unmeasured or measured in the X, Y or Z basis. The two outcomes of a Pauli measurement leave
    states that differ by a Pauli operator, so adaptive protocols add no other case. A Bell pair is left by two
    unmeasured qubits decoupled from every other qubit. Peers with the same neighbors are interchangeable, so one
    pattern is evaluated for each class of patterns that differ by a permutation of such peers: 4^n_o
    C(kappa_b + 3, 3)^(n_o + 1) classes instead of 4^n patterns.

    Returns the number of classes; for every number of measurements, the largest number of decoupled Bell pairs, over
    the patterns that measure every orchestration qubit and over all the patterns; and the number of patterns for
    every number of measurements and of Bell pairs.
    """
    if gtl.n > MAX_QUBITS:
        raise ValueError(f"The search handles at most {MAX_QUBITS} qubits.")
    parts = os.cpu_count()
    command = [binary, str(gtl.n_o), str(gtl.kappa_b)]
    processes = [
        subprocess.Popen([*command, str(part), str(parts)], stdout=subprocess.PIPE, text=True) for part in range(parts)
    ]
    classes, best = 0, {"pairs_pure": [0] * (gtl.n + 1), "pairs_any": [0] * (gtl.n + 1)}
    distribution = collections.Counter()
    for process in processes:
        out, _ = process.communicate()
        if process.returncode:
            raise RuntimeError(f"The search failed on GTL({gtl.n_o}, {gtl.kappa_b}).")
        for line in out.splitlines():
            key, *values = line.split()
            if key == "patterns":
                classes += int(values[0])
            elif key == "distribution":
                m, pairs, count = map(int, values)
                distribution[m, pairs] += count
            elif key in best:
                best[key] = [max(a, int(b)) for a, b in zip(best[key], values, strict=True)]
    return classes, best["pairs_pure"], best["pairs_any"], distribution


def _measure_x(graph, v):
    """Graph after a Pauli-X measurement of v with its smallest neighbor as support; unchanged if v is isolated."""
    return measure_x(graph, v, min(graph[v])) if graph[v] else graph


RULES = (_measure_x, measure_y, measure_z)


def _reference_branch(job):
    """The results of reference over the patterns that start with the given choices for the first qubits."""
    n_o, kappa_b, prefix = job
    gtl = GTL(n_o, kappa_b)
    orchestration = set(gtl.V_o)
    pure, every, distribution = [0] * (gtl.n + 1), [0] * (gtl.n + 1), collections.Counter()

    def visit(v, graph, measured):
        """Every choice for the qubits from v on, recording the Bell pairs left by each complete pattern."""
        if v == gtl.n:
            rest = [u for u in graph if u not in measured]
            pairs = sum(len(c) == 2 for c in nx.connected_components(graph.subgraph(rest)))
            m = len(measured)
            every[m] = max(every[m], pairs)
            distribution[m, pairs] += 1
            if orchestration <= measured:
                pure[m] = max(pure[m], pairs)
            return
        visit(v + 1, graph, measured)
        for rule in RULES:
            visit(v + 1, rule(graph, v), measured | {v})

    graph, measured = gtl.graph, frozenset()
    for v, choice in enumerate(prefix):
        if choice:
            graph, measured = RULES[choice - 1](graph, v), measured | {v}
    visit(len(prefix), graph, measured)
    return pure, every, distribution


def reference(gtl):
    """The maxima and the distribution of `search` with the graph rules of gtl_rolling, on all the 4^n patterns.

    The qubits are visited in a fixed order. After each measurement the library gives the graph of the corrected
    state, and the basis of a later measurement refers to that state: for a fixed order, the bases chosen this way
    are in one-to-one correspondence with the physical ones, so every pattern is covered once.
    """
    jobs = [(gtl.n_o, gtl.kappa_b, prefix) for prefix in itertools.product(range(4), repeat=2)]
    with multiprocessing.Pool() as pool:
        branches = pool.map(_reference_branch, jobs)
    pure = [max(column) for column in zip(*(b[0] for b in branches), strict=True)]
    every = [max(column) for column in zip(*(b[1] for b in branches), strict=True)]
    return pure, every, sum((b[2] for b in branches), collections.Counter())


def first(pairs, n_o):
    """Smallest number of measurements that leaves n_o Bell pairs (None if they are never obtained)."""
    return pairs.index(n_o) if n_o in pairs else None


def pgfplots(results):
    """The pgfplots figure of the largest number of Bell pairs divided by n_o, against m - (n - 2 n_o)."""
    curves = []
    for r in results:
        offset = r["n"] - 2 * r["n_o"]
        points = " ".join(
            f"({m - offset},{pairs / r['n_o']:.4f})"
            for m, pairs in enumerate(r["pairs_any"])
            if OFFSETS[0] <= m - offset <= OFFSETS[1]
        )
        curves.append(f"\\addplot[gray, opacity=0.55] coordinates {{{points}}};")
    lines = "\n".join(curves)
    return rf"""\begin{{tikzpicture}}
\begin{{axis}}[
    width=8.6cm, height=5.6cm,
    tick align=outside, tick pos=left,
    xtick style={{color=black}}, ytick style={{color=black}},
    label style={{font=\small}}, tick label style={{font=\footnotesize}},
    every axis plot/.append style={{line width=0.6pt}},
    xlabel={{$m - (n - 2n_o)$}}, ylabel={{max.\ Bell pairs$/n_o$}},
    xmin={OFFSETS[0] - 0.5}, xmax={OFFSETS[1] + 0.5}, ymin=-0.05, ymax=1.1,
    xtick={{{OFFSETS[0]},-8,-4,0,4}}, ytick={{0,0.5,1}},
]
\draw[densely dotted, gray] (axis cs:0,-0.05) -- (axis cs:0,1.1);
{lines}
\end{{axis}}
\end{{tikzpicture}}
"""


def write(results, suffix):
    """Write the table of the instances and the figure into the output directory, with suffix in their names."""
    os.makedirs(OUTPUT, exist_ok=True)
    with open(os.path.join(OUTPUT, f"minimum_measurements{suffix}.csv"), "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["n_o", "kappa_b", "n", "patterns", "classes", "min_measurements"])
        for r in results:
            needed = first(r["pairs_any"], r["n_o"])
            writer.writerow([r["n_o"], r["kappa_b"], r["n"], 4 ** r["n"], r["classes"], needed])
    path = os.path.join(OUTPUT, f"minimum_measurements{suffix}.tex")
    with open(path, "w") as file:
        file.write(pgfplots(results))
    sizes = [r["n"] for r in results]
    print(f"{len(results)} instances, n = {min(sizes)} to {max(sizes)}; figure: {path}")


def main():
    """Searches every instance, checks the minimum and writes the table and the figure.

    For every instance, n_o Bell pairs must first appear with n - 2 n_o measurements, i.e., when every qubit outside the
    pairs is measured, and the classes must cover the 4^n patterns exactly once. Up to REFERENCE_MAX_QUBITS qubits, the
    graph rules of gtl_rolling must give the same distribution on all the 4^n patterns.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", nargs="+", help="n_o,kappa_b pairs (default: 21 instances, n = 8 to 31)")
    args = parser.parse_args()
    instances = sorted(tuple(map(int, i.split(","))) for i in args.instances) if args.instances else INSTANCES

    results = []
    with tempfile.TemporaryDirectory() as directory:
        binary = compile_search(directory)
        for n_o, kappa_b in instances:
            gtl = GTL(n_o, kappa_b)
            classes, pure, every, distribution = search(binary, gtl)
            label = f"GTL({n_o}, {kappa_b})"
            if classes != 4**n_o * math.comb(kappa_b + 3, 3) ** (n_o + 1):
                raise RuntimeError(f"{label}: unexpected number of classes.")
            if distribution.total() != 4**gtl.n:
                raise RuntimeError(f"{label}: the classes do not cover the 4^n patterns exactly once.")
            needed = gtl.n - 2 * n_o
            if not first(pure, n_o) == first(every, n_o) == needed:
                raise RuntimeError(f"{label}: n_o Bell pairs do not first appear with n - 2 n_o measurements.")
            checked = gtl.n <= REFERENCE_MAX_QUBITS
            if checked and reference(gtl) != (pure, every, distribution):
                raise RuntimeError(f"{label}: the graph rules of gtl_rolling give another distribution.")
            results.append(
                {"n_o": n_o, "kappa_b": kappa_b, "n": gtl.n, "classes": classes, "pairs_pure": pure, "pairs_any": every}
            )
            print(
                f"GTL({n_o}, {kappa_b}), n = {gtl.n}: {classes:.2e} classes; {n_o} Bell pairs first with {needed} "
                f"measurements (n - 2 n_o), at most {max(every[:needed])} with fewer"
                + ("; same distribution with the graph rules of gtl_rolling on all 4^n patterns" if checked else ""),
                flush=True,
            )

    # Other instances go to their own files, so that they never overwrite the results of the 21 instances.
    write(results, "" if instances == sorted(INSTANCES) else "_custom")


if __name__ == "__main__":
    main()
