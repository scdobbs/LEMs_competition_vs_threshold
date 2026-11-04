"""Synthetic-surface tests for head_extraction (directive section 9)."""

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from acrit_slope.heads import d8_accumulate  # noqa: E402
from head_extraction import classify as cl  # noqa: E402
from head_extraction import config as hcfg  # noqa: E402
from head_extraction.pelletier import component_top_heads, contour_curvature, pelletier_heads, wiener_filter  # noqa: E402
from width_analysis.network import FlowGraph  # noqa: E402
from width_analysis.tiles import Tile  # noqa: E402

DX = 1.0


def hollow_surface(n=200, S0=0.3, k0=0.4, W=15, L=None):
    """Planar slope rising northward (row 0 highest) with a parabolic hollow of half-width W whose
    cross-slope curvature k(y) = k0 (1 - y/L) decays to zero at the top (y = L, row 0).
    y is distance upslope from the bottom row; contour curvature on the axis is k(y)/sqrt(1+S0^2)."""
    L = L or (n - 1)
    ii, jj = np.indices((n, n))
    y = (n - 1 - ii) * DX
    x = (jj - n // 2) * DX
    k = k0 * (1 - y / L)
    inside = np.abs(x) < W
    z = S0 * y + np.where(inside, 0.5 * k * x**2, 0.5 * k * W**2)
    return z, y, x


def test_hollow_head_within_2m_of_analytic_crossing():
    S0, k0, kt = 0.3, 0.4, 0.1
    z, y, x = hollow_surface(S0=S0, k0=k0)
    L = z.shape[0] - 1
    y_star = L * (1 - kt * np.sqrt(1 + S0**2) / k0)          # k(y*) / sqrt(1+S0^2) = kt
    kappa = contour_curvature(z, DX)
    r, c = pelletier_heads(z, kappa, kt)                      # no noise: no filter, no conditioning needed
    assert len(r) >= 1
    assert np.all(np.abs(y[r, c] - y_star) <= 2.0), (y[r, c], y_star)
    r2, c2 = component_top_heads(z, kappa, kt)
    assert len(r2) == 1 and abs(y[r2[0], c2[0]] - y_star) <= 2.0


def test_wiener_filter_reduces_rms_error():
    rng = np.random.default_rng(0)
    n = 256
    ii, jj = np.indices((n, n))
    truth = 8.0 * np.sin(2 * np.pi * jj / 120.0) + 0.25 * ii + 3.0 * np.cos(2 * np.pi * ii / 90.0)
    noisy = truth + rng.normal(0, 0.3, truth.shape)
    filt, m = wiener_filter(noisy, DX, hcfg.EXTRACTION)
    rms_noisy = np.sqrt(np.mean((noisy - truth) ** 2))
    rms_filt = np.sqrt(np.mean((filt - truth) ** 2))
    assert rms_filt < rms_noisy
    assert m["beta"] < 0 and m["N2"] > 0


def test_heads_bit_identical_when_area_config_changes():
    rng = np.random.default_rng(1)
    z, _, _ = hollow_surface(n=120)
    z = z + rng.normal(0, 0.05, z.shape)
    ext = copy.deepcopy(hcfg.EXTRACTION)
    kappa = contour_curvature(wiener_filter(z, DX, ext)[0], DX)
    r1, c1 = pelletier_heads(z, kappa, 0.1)
    # change every area-related value anywhere in the config and the A_crit tables
    qc2 = copy.deepcopy(hcfg.QC); qc2["trunk_area_m2"] = 7.0; qc2["area_overlay_min_m2"] = 1.0
    lsd2 = copy.deepcopy(hcfg.LSDTT); lsd2["threshold_contributing_pixels"] = 5; lsd2["pruning_drainage_area"] = 1.0
    acrit2 = {k: v * 100 for k, v in hcfg.A_CRIT_M2.items()}
    assert "area" not in " ".join(ext.keys()).lower() and "acrit" not in " ".join(ext.keys()).lower()
    kappa2 = contour_curvature(wiener_filter(z, DX, ext)[0], DX)
    r2, c2 = pelletier_heads(z, kappa2, 0.1)
    assert np.array_equal(r1, r2) and np.array_equal(c1, c2)
    assert acrit2 and qc2 and lsd2   # the changed values are simply never consulted by the extractor


def _flow_south_tile(z):
    fd = np.full(z.shape, 4, dtype=np.uint8); fd[-1, :] = 1
    area = d8_accumulate(fd, DX * DX)
    return Tile("SY", 1, z.astype(float), area, fd, DX, DX, 0.0, z.shape[0] * DX, "")


def _records(tile, graph, rc):
    curv = (np.full((tile.shape[0] // 10, tile.shape[1] // 10), 0.5), 10, 10.0)   # convergent everywhere (QC grid)
    r = np.array([p[0] for p in rc]); c = np.array([p[1] for p in rc])
    df = cl.measure(tile, graph, r, c, hcfg.EXTRACTION, hcfg.QC, curv)
    return cl.classify(df, hcfg.QC)


def test_path_leaves_tile_only_at_the_grid_edge():
    z = 0.3 * (120 - 1 - np.indices((120, 120))[0]) * DX
    tile = _flow_south_tile(z)
    g = FlowGraph(tile, 1.0, 0.0)
    curv = (np.full((12, 12), 0.5), 10, 10.0)
    df = cl.measure(tile, g, np.array([10, 50]), np.array([60, 60]), hcfg.EXTRACTION, hcfg.QC, curv)
    assert bool(df.flag_path_leaves_tile.iloc[0]) and not bool(df.flag_short_upslope_path.iloc[0])   # path hits row 0
    assert not bool(df.flag_path_leaves_tile.iloc[1])
    z2 = z.copy(); z2[:45, :] = z[45, :] - 1.0          # an interior divide: rows above 45 are lower
    fd = np.full(z2.shape, 4, dtype=np.uint8); fd[-1, :] = 1; fd[:45, :] = 64   # upper block flows north
    from acrit_slope.heads import d8_accumulate
    tile2 = Tile("SY", 1, z2, d8_accumulate(fd, 1.0), fd, DX, DX, 0.0, 120.0, "")
    g2 = FlowGraph(tile2, 1.0, 0.0)
    df2 = cl.measure(tile2, g2, np.array([50]), np.array([60]), hcfg.EXTRACTION, hcfg.QC, curv)
    assert bool(df2.flag_short_upslope_path.iloc[0]) and not bool(df2.flag_path_leaves_tile.iloc[0]) and bool(df2.retained.iloc[0])


def test_rill_rule_fires_on_parallel_gullies_not_on_single_hollow():
    n = 160
    ii, jj = np.indices((n, n))
    z = 0.70 * (n - 1 - ii) * DX                         # 35 degree plane dipping south
    tile = _flow_south_tile(z)
    g = FlowGraph(tile, 1.0, 0.0)
    gullies = [(60, 40 + 8 * k) for k in range(8)]       # 8 heads 8 m apart on one row, all flowing south
    df = _records(tile, g, gullies)
    assert (df.head_class == "rill").all(), df[["head_class", "class_detail"]]
    single = _records(tile, g, [(60, 80)])
    assert single.head_class.iloc[0] != "rill"


def test_road_rule_fires_on_collinear_heads_across_the_fall_line():
    n = 160
    ii, jj = np.indices((n, n))
    z = 0.20 * (n - 1 - ii) * DX                         # gentle plane (not a rill slope)
    tile = _flow_south_tile(z)
    g = FlowGraph(tile, 1.0, 0.0)
    heads = [(70, 40 + 12 * k) for k in range(6)]        # a row across the fall line, 12 m apart (6 within 60 m)
    df = _records(tile, g, heads)
    assert (df.head_class == "road_ditch").all(), df[["head_class", "class_detail"]]
    along = [(40 + 12 * k, 80) for k in range(6)]        # a line down the fall line: not a road
    df2 = _records(tile, g, along)
    assert not (df2.head_class == "road_ditch").any()
