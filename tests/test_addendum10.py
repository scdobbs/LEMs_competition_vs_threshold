"""Addendum 10: the section 1.4 clip rule and the flux-form part runner on a synthetic tile."""

from __future__ import annotations

import numpy as np
from tests.synthetic_tile import hydro_from_padded, synthetic_dem

from lemcvt.analyses.gm_flux import _row, adjacency_to_fixed, classify_a10, d_grid
from lemcvt.models.partitioned_nonlinear import partitioned_threshold_model_nonlinear


def test_adjacency_to_fixed_four_neighbours() -> None:
    hs = np.ones((5, 5), dtype=bool)
    hs[0, :] = hs[-1, :] = hs[:, 0] = hs[:, -1] = False     # boundary ring
    hs[2, 2] = False                                       # a channel cell in the middle
    adj = adjacency_to_fixed(hs)
    # every interior cell touches the ring or the centre: only cells with no fixed 4-neighbour are interior
    assert not adj[~hs].any()
    assert adj[1, 1] and adj[1, 2] and adj[2, 1]
    big = np.ones((7, 7), dtype=bool)
    big[0, :] = big[-1, :] = big[:, 0] = big[:, -1] = False
    a = adjacency_to_fixed(big)
    assert not a[3, 3] and a[1, 3] and not a[2, 2]         # (2, 2) touches the ring only diagonally


def test_classify_a10() -> None:
    assert classify_a10(True, 0.0099) == "converged"
    assert classify_a10(True, 0.01) == "no_steady_solution"
    assert classify_a10(False, 0.0) == "not_converged"
    assert classify_a10(False, 0.5) == "no_steady_solution"


def test_d_grid() -> None:
    g = d_grid()
    assert g.size == 14 and np.isclose(g[0], 3e-3) and np.isclose(g[-1], 5e-2)


def test_row_counts_both_ways() -> None:
    z = synthetic_dem(40, 40, 1.0, seed=3)
    zf, fd, area = hydro_from_padded(z, 1.0)
    f = partitioned_threshold_model_nonlinear(zf, area, fd, 1.0, 0.4, 1e-4, 2e-4)
    S_c = 0.05                                             # low: many cells clipped
    Z, _ = f(200.0, 5e-3, S_c, solver="newton", form="flux")
    r = _row(f, Z, zf, 200.0, 5e-3, 2e-4, S_c, 0.0)
    assert r["n_clipped"] >= r["n_clipped_adjacent"]
    assert r["outcome_a9"] in ("converged", "no_steady_solution", "not_converged")
    assert 0.0 <= r["frac_clipped_interior"] <= 1.0
    # with a high S_c nothing is clipped and both rules agree
    Z2, _ = f(200.0, 5e-3, 50.0, solver="newton", form="flux")
    r2 = _row(f, Z2, zf, 200.0, 5e-3, 2e-4, 50.0, 0.0)
    assert r2["n_clipped"] == 0 and r2["outcome"] == r2["outcome_a9"] == "converged"
