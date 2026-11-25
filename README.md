# gtl-rolling

Entanglement Rolling on generalized tree-like (GTL) graph states: build the resource, choose the supports of the rolling, and compute the fidelity of the Bell pairs and star states it instantiates under depolarizing noise and memory dephasing, in closed form and with the [noisy stabilizer formalism](https://github.com/jwallnoefer/noisy_graph_states).

## Installation

```bash
pip install "gtl-rolling @ git+https://github.com/framazzaa/gtl-rolling@v0.1.0"
```

Requires Python 3.12 or later.

## Usage

```python
from gtl_rolling import GTL, SupportSequence, outputs, star_fidelity

seq = SupportSequence(GTL(3, 2), "LRR")
for pair in outputs(seq, 2):
    center, (leaf,) = pair
    print(center, leaf, f"{star_fidelity(seq, 2, pair, p=0.98):.6f}")
```

```text
0 1 0.942060
4 5 0.942060
6 7 0.933021
```

`GTL(3, 2)` has three orchestration qubits and two peers per group, and `LRR` sets the side of the support of each orchestration measurement. Each line gives the two peers of a Bell pair and its fidelity under depolarizing noise `p`.

## Experiments and tests

The scripts in `experiments/` reproduce the numerical results of the related research article and write to `experiments/output/`; the exhaustive search also needs a C compiler. Install a checkout with `pip install -e ".[experiments,test]"` and run the tests with `python -m pytest`.

## Related research articles

[[1]](https://arxiv.org/abs/2605.15029) _A Resource-Driven Framework for Configurable Entanglement in Quantum Networks_. Francesco Mazza, Claudio Pellitteri, Angela Sara Cacciapuoti, Marcello Caleffi -- arXiv:2605.15029, 2026.

## License

MIT, see [LICENSE](LICENSE).
