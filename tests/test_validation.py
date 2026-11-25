"""The public entry points reject invalid inputs instead of returning a number."""

import math

import numpy as np
import pytest

from gtl_rolling import (
    GTL,
    Schedule,
    SupportSequence,
    apply,
    dephasing,
    fidelity,
    joint_fidelity,
    measure_x,
    nbar,
    noisy_state,
    outputs,
    propagate,
    proximity,
    run,
    star_fidelity,
)

SEQ = SupportSequence.right(GTL(2, 3))
PAIRS = outputs(SEQ, 2)
CENTER, LEAVES = outputs(SEQ, 3)[0]
SMALL = SupportSequence.right(GTL(2, 2))
PAIR = outputs(SMALL, 2)[0]


def small_output():
    """NSF output state of GTL(2, 2) under depolarizing noise."""
    return run(SMALL, 2, noisy_state(SMALL.gtl.graph, 0.9))


INVALID = {
    "n_o not an integer": (TypeError, lambda: GTL(2.5, 2)),
    "pattern not a string": (ValueError, lambda: SupportSequence(SMALL.gtl, ["R", "R"])),
    "support not a neighbor": (ValueError, lambda: measure_x(SMALL.gtl.graph, SMALL.gtl.o(1), 5)),
    "unknown instruction": (ValueError, lambda: apply(SMALL.gtl.graph, [("w", 0)])),
    "proximity of a peer to itself": (ValueError, lambda: proximity(SMALL.gtl.graph, SMALL.gtl.V_o, 0, 0)),
    "proximity of an orchestration qubit": (ValueError, lambda: proximity(SMALL.gtl.graph, SMALL.gtl.V_o, 6, 0)),
    "noise on an unknown qubit": (ValueError, lambda: noisy_state(SMALL.gtl.graph, {999: 0.5})),
    "propagated noise on an unknown qubit": (ValueError, lambda: propagate(SMALL.gtl.graph, [], [999])),
    "no star": (ValueError, lambda: joint_fidelity(SEQ, 2, [], 0.9)),
    "fidelity of no star": (ValueError, lambda: fidelity(small_output(), [])),
    "star not isolated": (
        ValueError,
        lambda: fidelity(run(SEQ, 3, noisy_state(SEQ.gtl.graph, 0.9)), [(LEAVES[0], (CENTER,))]),
    ),
    "star of 3 qubits for k = 2": (ValueError, lambda: star_fidelity(SEQ, 2, (CENTER, LEAVES), 0.9)),
    "center not a support": (ValueError, lambda: star_fidelity(SEQ, 2, (999, (1000,)), 0.98)),
    "repeated leaf": (ValueError, lambda: star_fidelity(SEQ, 3, (CENTER, (LEAVES[0], LEAVES[0])), 0.98)),
    "p above one": (ValueError, lambda: star_fidelity(SEQ, 2, (CENTER, LEAVES[:1]), 1.1)),
    "repeated star": (ValueError, lambda: joint_fidelity(SEQ, 2, [PAIRS[0], PAIRS[0]], 0.9)),
    "leaf of another group": (ValueError, lambda: joint_fidelity(SEQ, 2, [(PAIRS[0][0], PAIRS[1][1])], 0.9)),
    "nan in p": (ValueError, lambda: joint_fidelity(SEQ, 2, PAIRS, np.array([0.9, np.nan]))),
    "repeated pair in fidelity": (ValueError, lambda: fidelity(small_output(), [PAIR, PAIR])),
    "repeated leaf in fidelity": (ValueError, lambda: fidelity(small_output(), [(PAIR[0], PAIR[1] * 2)])),
    "nbar of o_0": (IndexError, lambda: nbar(SMALL, 0)),
    "nbar of o_3": (IndexError, lambda: nbar(SMALL, 3)),
    "nan time": (ValueError, lambda: dephasing(math.nan, 1.0)),
    "nan dephasing time": (ValueError, lambda: dephasing(1.0, math.nan)),
    "nan distance": (ValueError, lambda: Schedule(distance=math.nan)),
    "infinite distance": (ValueError, lambda: Schedule(distance=math.inf)),
    "nan distance of one peer": (ValueError, lambda: Schedule(distance={0: 40e3, 1: math.nan})),
    "nan speed": (ValueError, lambda: Schedule(distance=40e3, speed=math.nan)),
    "nan processing time": (ValueError, lambda: Schedule(distance=40e3, processing=math.nan)),
    "infinite processing time": (ValueError, lambda: Schedule(distance=40e3, processing=math.inf)),
}


@pytest.mark.parametrize("error, call", INVALID.values(), ids=INVALID.keys())
def test_invalid_inputs(error, call):
    """Each invalid input raises its error."""
    with pytest.raises(error):
        call()
