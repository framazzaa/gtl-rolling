"""Closed-form fidelity of the output stars of the two-stage resolution under depolarizing noise."""

import itertools
from collections.abc import Sequence

import numpy as np

from .noise import nbar
from .rolling import SupportSequence, outputs
from .simulate import Star


def star_fidelity(seq: SupportSequence, k: int, star: Star, p: float | np.ndarray) -> float | np.ndarray:
    """Returns the closed-form fidelity of a k-qubit output star of the two-stage resolution of a noisy GTL state.

    Every qubit undergoes depolarizing noise p. Then

        F = [(1 + p)^k + (1 - p)^k + 2^k p^(2 kappa_b + pi)] / 2^(k+1),

    where pi = pi(b0, gamma) is the peer proximity, in the initial resource state, of the center b0 to any peer gamma of
    the final rolled set Gamma^(o_{n_o}). For a Bell pair, this is F = (1 + p^2 + 2 p^(2 kappa_b + pi)) / 4.

    Args:
        seq: Support sequence of the resolution.
        k: Number of qubits of each output star, 2 <= k <= kappa_b.
        star: Output star (center, leaves): the center is a support of seq, the k - 1 leaves peers of its group.
        p: Depolarizing parameter in [0, 1], a float or a numpy array.

    Returns:
        The fidelity, a float or an array of the shape of p.

    Raises:
        ValueError: If the star is not an output of seq, or a parameter is out of range.
    """
    return joint_fidelity(seq, k, [star], p)


def joint_fidelity(seq: SupportSequence, k: int, stars: Sequence[Star], p: float | np.ndarray) -> float | np.ndarray:
    """Returns the closed-form joint fidelity of distinct output stars, the probability that all are correct at once.

    Under the noise of star_fidelity, with s stars,

        F_J = 2^-(k s) sum_beta p^(kappa_b [|beta| odd]) prod_m f(beta_m) prod_i p^[n_i(beta) odd],

    over beta in {0, 1}^s, with f(0) = [(1 + p)^k + (1 - p)^k]/2, f(1) = 2^(k-1) p^kappa_b, and n_i(beta) the number of
    stars m with beta_m = 1 whose center is in nbar(o_i). The sum has 2^s terms.

    Args:
        seq: Support sequence of the resolution.
        k: Number of qubits of each output star, 2 <= k <= kappa_b.
        stars: Output stars (center, leaves) with distinct centers, each as in star_fidelity.
        p: Depolarizing parameter in [0, 1], a float or a numpy array.

    Returns:
        The joint fidelity, a float or an array of the shape of p.

    Raises:
        ValueError: If there is no star, a star is not an output of seq, two stars share a center, or a parameter is out
            of range.
    """
    _validate_inputs(seq, k, stars, p)
    gtl = seq.gtl
    f = (((1 + p) ** k + (1 - p) ** k) / 2, 2 ** (k - 1) * p**gtl.kappa_b)
    reached = [[m for m, (center, _) in enumerate(stars) if center in nbar(seq, i)] for i in range(1, gtl.n_o + 1)]
    total = 0.0
    for beta in itertools.product((0, 1), repeat=len(stars)):
        term = p ** (gtl.kappa_b * (sum(beta) % 2))
        for b in beta:
            term = term * f[b]
        for stars_of_o in reached:
            if sum(beta[m] for m in stars_of_o) % 2:
                term = term * p
        total = total + term
    return total / 2 ** (k * len(stars))


def _validate_inputs(seq: SupportSequence, k: int, stars: Sequence[Star], p: float | np.ndarray) -> None:
    """Raises ValueError unless the stars are distinct outputs of seq and p is a depolarizing parameter."""
    if not 2 <= k <= seq.gtl.kappa_b:
        raise ValueError("2 <= k <= kappa_b is required.")
    if not stars:
        raise ValueError("Give at least one star.")
    if not np.all((p >= 0) & (p <= 1)):
        raise ValueError("The depolarizing parameter p must be in [0, 1].")

    # Each support owns a disjoint group. Removing it also rules out a repeated star.
    available = dict(outputs(seq, seq.gtl.kappa_b))
    for center, leaves in stars:
        group = available.pop(center, None)
        if group is None:
            raise ValueError(f"Center {center} must be a distinct support of this sequence.")
        selected = set(leaves)
        if len(leaves) != k - 1 or len(selected) != k - 1 or not selected.issubset(group):
            raise ValueError(f"Choose {k - 1} distinct leaves from {group} for center {center}.")
