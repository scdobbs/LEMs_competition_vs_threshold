"""Synthetic-surface tests for acrit_slope (no raster data needed)."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from acrit_slope.config import CONFIG  # noqa: E402
from acrit_slope.fits import fit_common_slope, fit_ols, fit_rma, two_level_bootstrap  # noqa: E402
from acrit_slope.heads import (block_mean, d8_accumulate, detect_heads, measure_heads, path_slopes,  # noqa: E402
                               plan_curvature, walk_path)
from width_analysis.network import FlowGraph  # noqa: E402
from width_analysis.tiles import Tile  # noqa: E402

DX = 1.0


def make_tile(z, fd, dx=DX):
    area = d8_accumulate(fd, dx * dx)
    return Tile("SY", 1, z.astype(float), area, fd.astype(np.uint8), dx, dx, 0.0, z.shape[0] * dx, "")


def south_plane(n=60, slope=0.3):
    """Plane dipping south (rows increase southward); every cell flows S (code 4), bottom row flows E."""
    ii, jj = np.indices((n, n))
    z = slope * (n - 1 - ii) * DX
    fd = np.full((n, n), 4, dtype=np.uint8)
    fd[-1, :] = 1
    return make_tile(z, fd)


def diagonal_plane(n=60, slope=0.3):
    """Plane dipping SE along the diagonal; every cell flows SE (code 2). Drop per step = slope * sqrt(2) * dx."""
    ii, jj = np.indices((n, n))
    z = slope * np.sqrt(2) * DX * ((n - 1 - ii) + (n - 1 - jj)) / 2.0  # z along the diagonal direction
    # along a SE step, (n-1-i)+(n-1-j) falls by 2 -> dz = slope*sqrt(2)*dx: slope along the sqrt(2) dx path = slope
    fd = np.full((n, n), 2, dtype=np.uint8)
    return make_tile(z, fd)


# ---------------------------------------------------------------------------
# path-averaged slope
# ---------------------------------------------------------------------------

def test_planar_surface_S_up_equals_plane_slope():
    tile = south_plane(slope=0.3)
    g = FlowGraph(tile, 10.0, 0.0)
    s = path_slopes(g, 40, 30, CONFIG["L_values_m"])
    for L in CONFIG["L_values_m"]:
        assert s[f"S_up_{L:g}"] == pytest.approx(0.3, abs=1e-12)
        assert s[f"S_centred_{L:g}"] == pytest.approx(0.3, abs=1e-12)
    assert s["S_pixel"] == pytest.approx(0.3, abs=1e-12)
    assert s["upslope_path_m"] >= max(CONFIG["L_values_m"])


def test_diagonal_path_length_has_sqrt2_factor():
    tile = diagonal_plane(slope=0.3)
    g = FlowGraph(tile, 10.0, 0.0)
    dist, z, rc = walk_path(g, 40, 40, "up", 10.0)
    steps = np.diff(dist)
    assert np.allclose(steps, np.sqrt(2) * DX)
    assert dist[-1] >= 10.0
    assert rc[1] == (39, 39)                      # NW donor
    s = path_slopes(g, 40, 40, CONFIG["L_values_m"])
    for L in CONFIG["L_values_m"]:
        assert s[f"S_up_{L:g}"] == pytest.approx(0.3, abs=1e-12)   # exact path length, not the chord


def test_short_upslope_path_is_flagged():
    tile = south_plane(n=60, slope=0.3)
    g = FlowGraph(tile, 10.0, 0.0)
    s = path_slopes(g, 5, 30, CONFIG["L_values_m"])  # only 5 m of path above row 5
    assert s["upslope_path_m"] == pytest.approx(5.0)
    assert np.isnan(s["S_up_20"]) and np.isnan(s["S_up_10"]) and np.isfinite(s["S_up_5"])


# ---------------------------------------------------------------------------
# head detection on a synthetic network
# ---------------------------------------------------------------------------

def test_detected_heads_are_exactly_the_expected_cells():
    # 7 columns flow south, bottom row flows east: column cell at row r has A = (r+1) dx^2
    n = 7
    ii, jj = np.indices((n, n))
    z = 1.0 * (n - 1 - ii) + 0.01 * (n - 1 - jj)
    fd = np.full((n, n), 4, dtype=np.uint8)
    fd[-1, :] = 1
    tile = make_tile(z, fd)
    g = FlowGraph(tile, 1.0, 0.0)
    rows, cols = detect_heads(tile.area, g.ui, g.uj, 3.5)
    expected = {(3, j) for j in range(n)}             # A = 4 >= 3.5, donor A = 3 < 3.5
    assert set(zip(rows.tolist(), cols.tolist())) == expected
    rows, cols = detect_heads(tile.area, g.ui, g.uj, 6.5)
    assert set(zip(rows.tolist(), cols.tolist())) == {(6, 0)}   # bottom-left: A = 7, donor 6; the rest have a donor >= 7


def test_measure_heads_exclusions_and_reasons():
    tile = south_plane(n=120, slope=0.3)
    g = FlowGraph(tile, 1.0, 0.0)
    cfg = dict(CONFIG)
    cfg["edge_buffer_m"] = 30.0
    recs = measure_heads(tile, g, 50.5, cfg)        # heads on row 50: 51 m from the top edge
    assert len(recs) == 120
    kept = [r for r in recs if r["kept"]]
    edge = [r for r in recs if r["reason"] == "tile_edge"]
    assert len(edge) == 60 and len(kept) == 60       # 30 columns at each side are within 30 m of an edge
    assert all(r["S_up_10"] == pytest.approx(0.3) for r in kept)
    assert all(r["AS2"] == pytest.approx(51.0 * 0.09) for r in kept)
    cfg2 = dict(cfg)
    cfg2["edge_buffer_m"] = 5.0
    recs2 = measure_heads(tile, g, 10.5, cfg2)       # heads on row 10: only 10 m of path above
    assert all(r["reason"] in ("tile_edge", "path_leaves_tile") for r in recs2)
    assert sum(r["reason"] == "path_leaves_tile" for r in recs2) == 110


def test_flat_fill_proxy_fires_on_an_epsilon_flat():
    tile = south_plane(n=120, slope=0.3)
    tile.z[40:80, :] = tile.z[80, :] + 1e-5 * (80 - np.arange(40, 80))[:, None]   # near-flat band
    g = FlowGraph(tile, 1.0, 0.0)
    recs = measure_heads(tile, g, 60.5, CONFIG)      # heads on row 60, inside the flat band
    kept_cols = [r for r in recs if r["edge_distance_m"] >= 30]
    assert all(r["reason"] == "flat_fill" for r in kept_cols)


# ---------------------------------------------------------------------------
# plan curvature sign and block mean
# ---------------------------------------------------------------------------

def test_plan_curvature_sign_convention():
    x, y = np.meshgrid(np.arange(-10, 11.0), np.arange(-10, 11.0))
    trough = 0.05 * x**2 + 0.2 * y            # a valley running along y: convergent
    nose = -0.05 * x**2 + 0.2 * y             # a ridge along y: divergent
    kt = plan_curvature(trough, 1.0)
    kn = plan_curvature(nose, 1.0)
    assert np.nanmean(kt[5:-5, 5:-5]) > 0
    assert np.nanmean(kn[5:-5, 5:-5]) < 0
    plane = 0.3 * y
    assert np.allclose(np.nan_to_num(plan_curvature(plane, 1.0)), 0.0)


def test_block_mean():
    z = np.arange(36.0).reshape(6, 6)
    b = block_mean(z, 3)
    assert b.shape == (2, 2)
    assert b[0, 0] == pytest.approx(z[:3, :3].mean())


# ---------------------------------------------------------------------------
# fits
# ---------------------------------------------------------------------------

def test_fits_recover_a_known_exponent():
    x = np.log10(np.array([0.1, 0.15, 0.2, 0.3, 0.5, 0.7]))
    y = 3.0 - 2.0 * x
    assert fit_ols(x, y)["b"] == pytest.approx(-2.0)
    assert fit_rma(x, y)["b"] == pytest.approx(-2.0)
    g = np.array(["AP", "AP", "AP", "GM", "GM", "GM"])
    y2 = y + np.where(g == "GM", 0.5, 0.0)
    f = fit_common_slope(x, y2, g)
    assert f["b"] == pytest.approx(-2.0)
    assert f["intercepts"]["GM"] - f["intercepts"]["AP"] == pytest.approx(0.5)


def test_two_level_bootstrap_brackets_the_truth():
    rng = np.random.default_rng(1)
    keys = [("AP", i) for i in range(1, 6)] + [("GM", i) for i in range(1, 6)]
    true_s = np.logspace(-1, -0.2, 10)
    logA = {k: float(3.5 - 2.0 * np.log10(s)) for k, s in zip(keys, true_s)}
    site = {k: k[0] for k in keys}
    hs = {k: s * np.exp(rng.normal(0, 0.1, 200)) for k, s in zip(keys, true_s)}
    boot = two_level_bootstrap(keys, logA, site, hs, "RMA", 300, rng)
    lo, hi = np.percentile(boot["b"], [2.5, 97.5])
    assert lo <= -2.0 <= hi
    assert boot["n_ok"] > 250
