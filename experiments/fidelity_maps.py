"""Fidelity of Bell pairs and GHZ states under depolarizing noise and memory dephasing, and the F = 1/2 thresholds."""

import argparse
import functools
import multiprocessing
import os

import noisy_graph_states as ngs
import numpy as np
from matplotlib.figure import Figure

from gtl_rolling import (
    GTL,
    Schedule,
    SupportSequence,
    dephasing,
    depolarizing,
    fidelity,
    noisy_state,
    outputs,
    proximity,
    run,
    stage2,
    star_fidelity,
    storage_times,
)

OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
DISTANCE = 40e3
RANDOM_DISTANCES = (10e3, 100e3)
RESOURCES = {
    "bell": {"kappa_b": 2, "k": 2, "p_min": 0.70, "fidelity_min": 0.25},
    "ghz": {"kappa_b": 3, "k": 3, "p_min": 0.75, "fidelity_min": 0.125},
}
HEAT_MAP_N_O = 4
THRESHOLD_N_O = (2, 3, 4, 5)
LOG10_T = (-3.0, -1.0)
HEAT_MAP_PIXELS = (744, 554)
CHECK_POINTS_PER_AXIS = 21
TOLERANCE = 1e-10


def schedule(gtl, seed):
    """Every peer at DISTANCE or, with a seed, at its own random distance in RANDOM_DISTANCES.

    The random distances are drawn for every resource from the seed, n_o and kappa_b, so that a run is reproducible.
    Since Tier2 waits for the farthest Tier1 node, the storage times depend only on the delays of the farthest peer and
    of the farthest peer measured in Stage 2: the resources of different n_o are evaluated on different networks, and
    their thresholds need not be ordered in n_o.
    """
    if seed is None:
        return Schedule(distance=DISTANCE)
    rng = np.random.default_rng((seed, gtl.n_o, gtl.kappa_b))
    return Schedule(distance=dict(zip(gtl.V_c, rng.uniform(*RANDOM_DISTANCES, len(gtl.V_c)).tolist(), strict=True)))


def suffix(seed):
    """File-name suffix of the runs with random distances."""
    return "" if seed is None else f"_random{seed}"


def instance(kind, n_o, seed):
    """Resource, support sequence (right supports), storage times, and output stars with their proximity, best first.

    The storage times follow the protocol schedule of gtl_rolling.timing, and the proximity is that of the center of
    each star to the final rolled set.
    """
    r = RESOURCES[kind]
    gtl = GTL(n_o, r["kappa_b"])
    seq = SupportSequence.right(gtl)
    gamma = seq.rolled_set(n_o)[0]
    stars = sorted(
        (proximity(gtl.graph, gtl.V_o, center, gamma), (center, leaves)) for center, leaves in outputs(seq, r["k"])
    )
    return gtl, seq, storage_times(seq, r["k"], schedule(gtl, seed)), stars


def simulate(kind, gtl, seq, t, stars, p, T):
    """NSF fidelities of the given stars, from a full run of both stages."""
    q = {v: dephasing(t[v], T) for v in gtl.graph}
    out = run(seq, RESOURCES[kind]["k"], noisy_state(gtl.graph, p, q))
    return [fidelity(out, [star]) for star in stars]


@functools.cache
def propagated(kind, n_o, seed):
    """The instance, and its NSF state after both stages, with the noise maps propagated once.

    The update rules of the NSF change the noise operators of the maps and leave their weights unchanged, so the maps
    are propagated once, with placeholder weights, and nsf_fidelities gives them the weights of every p and T.
    """
    gtl, seq, t, stars = instance(kind, n_o, seed)
    return gtl, seq, t, stars, run(seq, RESOURCES[kind]["k"], noisy_state(gtl.graph, 0.5, 0.5))


def nsf_fidelities(kind, n_o, seed, picks, p, T):
    """NSF fidelities, at p and T, of the outputs of the instance at the positions picks of its list, best first.

    noisy_state gives every qubit, in the order of the graph, a depolarizing map with the weights of X, Y and Z, and
    then a dephasing map, whose propagated noise operators are reweighted here.
    """
    gtl, _, t, stars, out = propagated(kind, n_o, seed)
    weights = []
    for v in gtl.graph:
        weights += [list(depolarizing(p)[1:]), [dephasing(t[v], T)]]
    maps = [ngs.Map(weights=w, noises=m.noises) for w, m in zip(weights, out.maps, strict=True)]
    state = ngs.State(graph=out.graph, maps=maps)
    return [fidelity(state, [stars[i][1]]) for i in picks]


def compare(job):
    """Fidelities of the outputs at picks, at p and T, from a full NSF run and from nsf_fidelities."""
    kind, n_o, seed, picks, p, T = job
    gtl, seq, t, stars, _ = propagated(kind, n_o, seed)
    return simulate(kind, gtl, seq, t, [stars[i][1] for i in picks], p, T), nsf_fidelities(*job)


def heat_map_row(job):
    """NSF fidelities of the best and the worst output of the heat maps for every p, at one T."""
    kind, seed, ps, T = job
    return [nsf_fidelities(kind, HEAT_MAP_N_O, seed, (0, -1), p, T) for p in ps]


def pixel_centers(lo, hi, count):
    """Coordinates of the centers of `count` equal pixels between lo and hi."""
    return lo + (np.arange(count) + 0.5) * (hi - lo) / count


def heat_maps(kind, seed, pool):
    """Heat maps of the best and the worst output against p and T, with the dashed contour F = 1/2, by the NSF.

    The maps are written once the reweighted maps agree with full NSF runs on a coarser grid and, without dephasing,
    with the closed form.
    """
    r = RESOURCES[kind]
    gtl, seq, t, stars, _ = propagated(kind, HEAT_MAP_N_O, seed)
    (pi_best, best), (pi_worst, worst) = stars[0], stars[-1]
    width, height = HEAT_MAP_PIXELS
    p = pixel_centers(r["p_min"], 1.0, width)
    log_t = pixel_centers(*LOG10_T, height)

    grid = [
        (kind, HEAT_MAP_N_O, seed, (0, -1), p_c, T_c)
        for p_c in np.linspace(r["p_min"], 1.0, CHECK_POINTS_PER_AXIS)
        for T_c in np.logspace(*LOG10_T, CHECK_POINTS_PER_AXIS)
    ]
    deviation = max(abs(a - b) for full, fast in pool.map(compare, grid) for a, b in zip(full, fast, strict=True))
    closed = 0.0
    for p_c in np.linspace(r["p_min"], 1.0, CHECK_POINTS_PER_AXIS):
        fast = nsf_fidelities(kind, HEAT_MAP_N_O, seed, (0, -1), p_c, np.inf)
        exact = [star_fidelity(seq, r["k"], star, p_c) for star in (best, worst)]
        closed = max(closed, *(abs(a - b) for a, b in zip(fast, exact, strict=True)))
    if not (deviation < TOLERANCE and closed < TOLERANCE):
        raise RuntimeError(f"{kind}: NSF runs differ by {deviation:.1e}, and from the closed form by {closed:.1e}.")

    f = np.array(pool.map(heat_map_row, [(kind, seed, p, 10.0**x) for x in log_t]))
    for name, column in (("best", 0), ("worst", 1)):
        fig = Figure(figsize=(width / 100, height / 100), dpi=100)
        ax = fig.add_axes((0, 0, 1, 1))
        ax.set_axis_off()
        extent = (r["p_min"], 1.0, *LOG10_T)
        ax.imshow(
            f[:, :, column],
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap="YlOrRd_r",
            vmin=r["fidelity_min"],
            vmax=1,
        )
        ax.contour(p, log_t, f[:, :, column], levels=[0.5], colors="black", linestyles="dashed", linewidths=3.0)
        ax.set_xlim(extent[:2])
        ax.set_ylim(extent[2:])
        fig.savefig(os.path.join(OUTPUT, f"fidelity_{kind}_{name}{suffix(seed)}.png"), dpi=100)
    print(
        f"{kind}, n_o = {HEAT_MAP_N_O}: best and worst outputs at proximity {pi_best} and {pi_worst}; NSF on "
        f"{width}x{height} points; reweighted maps vs full NSF runs on {CHECK_POINTS_PER_AXIS}x{CHECK_POINTS_PER_AXIS} "
        f"points: max deviation {deviation:.1e}; without dephasing, NSF vs closed form: max deviation {closed:.1e}"
    )


def threshold_curve(job):
    """For every T, the p at which the NSF fidelity of the worst output is 1/2.

    The p is found by bisection, since the fidelity increases with p; a T at which the fidelity stays below 1/2 at
    p = 1 has none.
    """
    kind, n_o, seed, T_values = job
    curve = []
    for T in T_values:
        if nsf_fidelities(kind, n_o, seed, (-1,), 1.0, T)[0] <= 0.5:
            continue
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if nsf_fidelities(kind, n_o, seed, (-1,), mid, T)[0] < 0.5 else (lo, mid)
        curve.append(((lo + hi) / 2, T))
    return curve


def thresholds(kind, n_o, curve, seed, pool):
    """Checks a threshold curve against full NSF runs and writes it to a table."""
    checked = curve[::8]
    errors = [abs(full[0] - 0.5) for full, _ in pool.map(compare, [(kind, n_o, seed, (-1,), p, T) for p, T in checked])]
    if not all(e < TOLERANCE for e in errors):
        raise RuntimeError(f"{kind}, n_o = {n_o}: the NSF fidelity at the threshold differs from 1/2.")
    with open(os.path.join(OUTPUT, f"threshold_{kind}_no{n_o}{suffix(seed)}.dat"), "w") as file:
        file.write("p T\n")
        file.writelines(f"{p:.8e} {T:.8e}\n" for p, T in curve)
    if not curve:
        print(f"{kind}, n_o = {n_o}: no sampled coherence time gives F > 1/2 at p = 1.")
        return
    print(
        f"{kind}, n_o = {n_o}: threshold p from {min(p for p, _ in curve):.4f} to {max(p for p, _ in curve):.4f}; "
        f"full NSF runs at {len(checked)} of its points: |F - 1/2| <= {max(errors):.1e}"
    )


def main():
    """Writes the heat maps and the threshold curves, computed with the NSF and checked."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--random-distances",
        type=int,
        nargs="?",
        const=0,
        metavar="SEED",
        help="every peer at its own random distance between 10 and 100 km, drawn with SEED (default 0); the files end "
        "in _random<SEED>",
    )
    seed = parser.parse_args().random_distances

    os.makedirs(OUTPUT, exist_ok=True)
    if seed is None:
        fixed = Schedule(distance=DISTANCE)
        print(f"timing: distance {DISTANCE / 1e3:.0f} km, tau = {fixed.tau * 1e3:.3f} ms, end {fixed.end * 1e3:.3f} ms")
        gtl, seq, t, stars = instance("bell", HEAT_MAP_N_O, None)
        stored = {
            "orchestration qubits": t[gtl.o(1)],
            "measured peers": t[stage2(seq, RESOURCES["bell"]["k"])[0][1]],
            "retained peers": t[stars[0][1][0]],
        }
        print("storage: " + ", ".join(f"{name} {time * 1e3:.3f} ms" for name, time in stored.items()))
    else:
        print(f"timing: every peer at a random distance between 10 and 100 km, seed {seed}")
    with multiprocessing.Pool() as pool:
        for kind in RESOURCES:
            heat_maps(kind, seed, pool)
        jobs = [(kind, n_o, seed, np.logspace(*LOG10_T, 81)) for kind in RESOURCES for n_o in THRESHOLD_N_O]
        for (kind, n_o, _, _), curve in zip(jobs, pool.map(threshold_curve, jobs), strict=True):
            thresholds(kind, n_o, curve, seed, pool)


if __name__ == "__main__":
    main()
