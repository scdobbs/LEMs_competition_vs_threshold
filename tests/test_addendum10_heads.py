"""Addendum 10, 4: synthetic checks of the Test 2 analyse-phase helpers (lemcvt.analyses.heads.analyse)."""

import numpy as np

from lemcvt.analyses.heads.analyse import (
    bootstrap_fit,
    ci,
    fit_loglog,
    fit_pooled,
    iqr_overlap,
    match_fraction,
    partition_heads_rc,
)
from lemcvt.analyses.heads.paths import d8_accumulate
from lemcvt.analyses.width.network import FlowGraph
from lemcvt.analyses.width.tiles import Tile


def test_ols_rma_recover_planted_slope():
    rng = np.random.default_rng(1)
    x = rng.uniform(-1.0, 0.0, 400)
    y = 3.0 - 1.65 * x + rng.normal(0, 0.05, x.size)
    f = fit_loglog(x, y)
    assert abs(f["b_ols"] + 1.65) < 0.03
    assert abs(f["b_rma"] + 1.65) < 0.03
    assert f["r2"] > 0.95
    # RMA = OLS / |r| always; with noise in y the RMA magnitude is larger
    y2 = 3.0 - 1.0 * x + rng.normal(0, 0.3, x.size)
    f2 = fit_loglog(x, y2)
    assert abs(f2["b_rma"]) > abs(f2["b_ols"])
    assert np.isclose(f2["b_rma"], f2["b_ols"] / abs(f2["r"]))


def test_pooled_fit_removes_tile_intercepts_and_bootstrap_brackets_truth():
    rng = np.random.default_rng(2)
    xs, ys, gs = [], [], []
    for g, a in enumerate([1.0, 2.0, 3.5]):        # very different intercepts per tile
        x = rng.uniform(-1, 0, 60)
        xs.append(x)
        ys.append(a - 2.0 * x + rng.normal(0, 0.1, x.size))
        gs.append(np.full(x.size, g))
    x, y, g = np.concatenate(xs), np.concatenate(ys), np.concatenate(gs)
    fp = fit_pooled(x, y, g)
    assert abs(fp["b_ols"] + 2.0) < 0.1
    assert len(fp["means"]) == 3
    bt = bootstrap_fit(x, y, g, 300, np.random.default_rng(3))
    lo, hi = ci(bt["b_ols"])
    assert lo < -2.0 < hi
    assert bt["mean_x"].shape == (300, 3)


def test_null_relation_interval_includes_zero():
    rng = np.random.default_rng(4)
    x = rng.uniform(-1, 0, 50)
    y = rng.normal(3, 0.5, 50)
    bt = bootstrap_fit(x, y, np.zeros(50), 1000, np.random.default_rng(5))
    lo, hi = ci(bt["b_ols"])
    assert lo < 0.0 < hi


def test_iqr_overlap_rule():
    assert iqr_overlap(300, 3000, 1000, 5000)       # bracket straddles q1
    assert iqr_overlap(3000, 300, 1000, 5000)       # order does not matter
    assert iqr_overlap(2000, 2500, 1000, 5000)      # bracket inside the IQR
    assert iqr_overlap(100, 9000, 1000, 5000)       # IQR inside the bracket
    assert iqr_overlap(100, 1000, 1000, 5000)       # touching counts
    assert not iqr_overlap(100, 900, 1000, 5000)    # below
    assert not iqr_overlap(6000, 9000, 1000, 5000)  # above


def _tile(fd, dx=1.0):
    ny, nx = fd.shape
    area = d8_accumulate(fd, dx * dx)
    z = np.zeros(fd.shape)
    return Tile("GM", 9, z, area, fd.astype(np.uint8), dx, dx, 0.0, float(ny), "")


def test_partition_head_finder_tiny_grid():
    # 3 rows x 5 cols; rows 0-1 flow south (code 4) into row 2, which flows east (code 1) to the outlet at col 4
    fd = np.full((3, 5), 4, dtype=int)
    fd[2, :] = 1
    t = _tile(fd)
    # areas: rows 0, 1 = 1, 2; row 2 = 3, 6, 9, 12, 15 (cumulative along the bottom row)
    assert np.allclose(t.area[2], [3, 6, 9, 12, 15])
    g = FlowGraph(t, 1e9, 0.0)
    r, c = partition_heads_rc(t.area, g.ui, g.uj, 2.0)
    # A >= 2 first reached in row 1 of every column (donor row 0 has A = 1); the bottom row cells have donors >= 2
    assert sorted(zip(r.tolist(), c.tolist(), strict=False)) == [(1, j) for j in range(5)]
    r, c = partition_heads_rc(t.area, g.ui, g.uj, 10.0)
    # A >= 10 first at (2, 3) (A = 12), whose largest donor (2, 2) has A = 9 < 10
    assert list(zip(r.tolist(), c.tolist(), strict=False)) == [(2, 3)]


def test_match_fraction():
    a = np.array([[0.0, 0.0], [100.0, 0.0]])
    b = np.array([[10.0, 0.0]])
    f, d = match_fraction(a, b, 20.0)
    assert f == 0.5 and np.allclose(d, [10.0, 90.0])
    f0, _ = match_fraction(a, np.empty((0, 2)), 20.0)
    assert f0 == 0.0
