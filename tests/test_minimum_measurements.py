"""The exhaustive search of the measurement patterns, against the graph rules of gtl_rolling."""

import importlib
import math
import shutil
from pathlib import Path

import pytest

from gtl_rolling import GTL

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"


@pytest.mark.skipif(shutil.which("cc") is None, reason="the search needs a C compiler")
def test_search_matches_graph_rules(tmp_path, monkeypatch):
    """On GTL(2, 2) the C search over the classes of patterns agrees with the graph rules on all the 4^8 patterns.

    Both give the same maxima and the same number of patterns for every number of measurements and of Bell pairs, and
    n_o Bell pairs first appear with n - 2 n_o measurements.
    """
    monkeypatch.syspath_prepend(str(EXPERIMENTS))
    mm = importlib.import_module("minimum_measurements")
    gtl = GTL(2, 2)
    classes, pure, every, distribution = mm.search(mm.compile_search(str(tmp_path)), gtl)
    assert classes == 4**gtl.n_o * math.comb(gtl.kappa_b + 3, 3) ** (gtl.n_o + 1)
    assert distribution.total() == 4**gtl.n
    assert mm.reference(gtl) == (pure, every, distribution)
    assert mm.first(pure, gtl.n_o) == mm.first(every, gtl.n_o) == gtl.n - 2 * gtl.n_o
