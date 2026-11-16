"""Partition channel heads, their exclusion rules, path-averaged slopes and the
plan-curvature QC field.

Everything is computed from the repo's existing filled DEM, D8 area and D8
flow-direction grids through ``width_analysis.network.FlowGraph`` (receiver and
largest-area-donor links). No routing is recomputed.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from lemcvt.analyses.width.network import D8_OFFSETS, FlowGraph
from lemcvt.analyses.width.tiles import Tile

EXCLUSION_REASONS = ["tile_edge", "path_leaves_tile", "flat_fill"]


# ---------------------------------------------------------------------------
# head detection
# ---------------------------------------------------------------------------

def donor_max_area(area: np.ndarray, ui: np.ndarray, uj: np.ndarray) -> np.ndarray:
    """Drainage area of each pixel's largest donor (0 where it has none)."""
    out = np.zeros_like(area, dtype=float)
    ok = ui >= 0
    out[ok] = area[ui[ok], uj[ok]]
    return out


def detect_heads(area: np.ndarray, ui: np.ndarray, uj: np.ndarray, a_crit: float) -> tuple[np.ndarray, np.ndarray]:
    """Rows/cols of the partition channel heads for threshold ``a_crit``.

    A head has A >= a_crit and no D8 donor with A >= a_crit; the second
    condition is ``max donor area < a_crit``.
    """
    mask = (area >= a_crit) & (donor_max_area(area, ui, uj) < a_crit)
    rows, cols = np.nonzero(mask)
    return rows, cols


def d8_accumulate(fd: np.ndarray, cell_area: float) -> np.ndarray:
    """D8 drainage area by topological accumulation (synthetic tests only)."""
    ny, nx = fd.shape
    area = np.full(fd.shape, cell_area, dtype=float)
    n_in = np.zeros(fd.shape, dtype=int)
    rec = {}
    for code, (a, b) in D8_OFFSETS.items():
        for i, j in zip(*np.nonzero(fd == code), strict=False):
            i2, j2 = i + a, j + b
            if 0 <= i2 < ny and 0 <= j2 < nx:
                rec[(i, j)] = (i2, j2)
                n_in[i2, j2] += 1
    stack = [(i, j) for i in range(ny) for j in range(nx) if n_in[i, j] == 0]
    while stack:
        i, j = stack.pop()
        if (i, j) in rec:
            i2, j2 = rec[(i, j)]
            area[i2, j2] += area[i, j]
            n_in[i2, j2] -= 1
            if n_in[i2, j2] == 0:
                stack.append((i2, j2))
    return area


# ---------------------------------------------------------------------------
# paths and slopes
# ---------------------------------------------------------------------------

def walk_path(graph: FlowGraph, i: int, j: int, direction: str, max_dist: float):
    """Main-stem path from (i, j): up the largest-area donors or down the receivers.

    Returns (dist, z, rc): cumulative exact path distance (diagonal steps are
    sqrt(2) * pixel), elevation, and (row, col) of each pixel, starting at the
    pixel itself. The walk stops at the first pixel whose distance reaches
    ``max_dist`` or where the path ends (no donor / receiver inside the tile).
    """
    nxt = graph.up if direction == "up" else graph.down
    z = graph.tile.z
    dist, zz, rc = [0.0], [float(z[i, j])], [(i, j)]
    d = 0.0
    while d < max_dist:
        i2, j2 = nxt(i, j)
        if i2 < 0:
            break
        d += graph.step_length(i, j, i2, j2)
        dist.append(d)
        zz.append(float(z[i2, j2]))
        rc.append((i2, j2))
        i, j = i2, j2
    return np.array(dist), np.array(zz), rc


def elevation_at(dist: np.ndarray, z: np.ndarray, L: float) -> float:
    """Elevation at exact path distance L by linear interpolation; NaN if the path is shorter."""
    if len(dist) == 0 or dist[-1] < L:
        return np.nan
    return float(np.interp(L, dist, z))


def path_slopes(graph: FlowGraph, i: int, j: int, L_values) -> dict:
    """S_up_L, S_centred_L for every L, plus S_pixel and the path lengths reached.

    S_up_L = (z(L upslope) - z(head)) / L along the largest-area-donor path.
    S_centred_L = (z(L/2 upslope) - z(L/2 downslope)) / L.
    S_pixel = D8 slope from the head to its receiver (diagnostic only).
    """
    L_max = float(max(L_values))
    du, zu, _ = walk_path(graph, i, j, "up", L_max)
    dd, zd, _ = walk_path(graph, i, j, "down", L_max)
    z0 = float(graph.tile.z[i, j])
    out = {"upslope_path_m": float(du[-1]), "downslope_path_m": float(dd[-1])}
    for L in L_values:
        zL = elevation_at(du, zu, L)
        out[f"S_up_{L:g}"] = (zL - z0) / L if np.isfinite(zL) else np.nan
        zh_up = elevation_at(du, zu, L / 2)
        zh_dn = elevation_at(dd, zd, L / 2)
        out[f"S_centred_{L:g}"] = (zh_up - zh_dn) / L if np.isfinite(zh_up) and np.isfinite(zh_dn) else np.nan
    ri, rj = graph.down(i, j)
    out["S_pixel"] = (z0 - float(graph.tile.z[ri, rj])) / graph.step_length(i, j, ri, rj) if ri >= 0 else np.nan
    return out


def trace_slope(graph: FlowGraph, i: int, j: int, half_length: float) -> float:
    """Fall over the main-stem trace +/- half_length around the pixel divided by its length
    (the width_analysis filled-flat proxy). NaN where either half is shorter."""
    du, zu, _ = walk_path(graph, i, j, "up", half_length)
    dd, zd, _ = walk_path(graph, i, j, "down", half_length)
    z_up, z_dn = elevation_at(du, zu, half_length), elevation_at(dd, zd, half_length)
    if not (np.isfinite(z_up) and np.isfinite(z_dn)):
        return np.nan
    return float((z_up - z_dn) / (2.0 * half_length))


# ---------------------------------------------------------------------------
# plan curvature on block-mean subsampled topography
# ---------------------------------------------------------------------------

def block_mean(z: np.ndarray, n: int) -> np.ndarray:
    """Block mean over n x n pixels (the trailing partial blocks are dropped)."""
    ny, nx = (z.shape[0] // n) * n, (z.shape[1] // n) * n
    return z[:ny, :nx].reshape(ny // n, n, nx // n, n).mean(axis=(1, 3))


def plan_curvature(zb: np.ndarray, L: float) -> np.ndarray:
    """Plan (contour) curvature from the 9-point Zevenbergen & Thorne (1987) stencil.

    Sign convention here: positive = convergent (a trough, contours concave
    upslope), negative = divergent (a nose). Checked in the tests on
    z = a x^2 + b y. NaN on the one-cell border and where the gradient is zero.
    """
    z = zb
    out = np.full(z.shape, np.nan)
    c = z[1:-1, 1:-1]
    zn, zs = z[:-2, 1:-1], z[2:, 1:-1]
    zw, ze = z[1:-1, :-2], z[1:-1, 2:]
    znw, zne, zsw, zse = z[:-2, :-2], z[:-2, 2:], z[2:, :-2], z[2:, 2:]
    D = ((zw + ze) / 2 - c) / L**2
    E = ((zn + zs) / 2 - c) / L**2
    F = (-znw + zne + zsw - zse) / (4 * L**2)
    G = (-zw + ze) / (2 * L)
    H = (zn - zs) / (2 * L)
    denom = G**2 + H**2
    with np.errstate(divide="ignore", invalid="ignore"):
        k = 2 * (D * H**2 + E * G**2 - F * G * H) / denom
    k[denom == 0] = np.nan
    out[1:-1, 1:-1] = k
    return out


def curvature_grid(tile: Tile, block_m: float) -> tuple[np.ndarray, int, float]:
    """Plan curvature of the tile on a block-mean grid of about ``block_m``. Returns (grid, n, L)."""
    n = max(1, int(round(block_m / tile.pixel_size)))
    L = n * tile.pixel_size
    return plan_curvature(block_mean(tile.z, n), L), n, L


# ---------------------------------------------------------------------------
# per-tile measurement
# ---------------------------------------------------------------------------

def measure_heads(tile: Tile, graph: FlowGraph, a_crit: float, cfg: dict, curv=None) -> list[dict]:
    """One record per detected head with slopes, QC fields and the exclusion verdict.

    Exclusion reasons are tested in the order tile_edge, path_leaves_tile,
    flat_fill; ``reason`` is the first that applies (empty = kept) and the
    boolean columns record every rule that fired.
    """
    rows, cols = detect_heads(tile.area, graph.ui, graph.uj, a_crit)
    L_values = [float(v) for v in cfg["L_values_m"]]
    L_max = max(L_values)
    ny, nx = tile.shape
    if curv is None:
        curv = curvature_grid(tile, cfg["curvature_block_m"])
    cgrid, nblk, _ = curv
    out = []
    for i, j in zip(rows, cols, strict=False):
        i, j = int(i), int(j)
        edge = float(min(i, j, ny - 1 - i, nx - 1 - j)) * tile.pixel_size
        s = path_slopes(graph, i, j, L_values)
        ts = trace_slope(graph, i, j, cfg["flat_fill_half_length_m"])
        x, y = tile.rc_to_xy(i, j)
        bi, bj = i // nblk, j // nblk
        pc = float(cgrid[bi, bj]) if (bi < cgrid.shape[0] and bj < cgrid.shape[1]) else np.nan
        flags = {
            "tile_edge": edge < cfg["edge_buffer_m"],
            "path_leaves_tile": s["upslope_path_m"] < L_max,
            "flat_fill": bool(np.isfinite(ts) and ts < cfg["flat_fill_slope_max"]),
        }
        reason = next((r for r in EXCLUSION_REASONS if flags[r]), "")
        rec: dict[str, Any] = dict(site=tile.site, tile=tile.tile, row=i, col=j, x=float(x), y=float(y),
                   area_m2=float(tile.area[i, j]), a_crit_m2=float(a_crit),
                   edge_distance_m=edge, trace_slope_20m=ts, plan_curvature=pc,
                   plan_curvature_sign=int(np.sign(pc)) if np.isfinite(pc) else 0,
                   convergent=bool(np.isfinite(pc) and pc > 0),
                   reason=reason, kept=(reason == ""))
        rec.update(s)
        rec.update({f"flag_{k}": bool(v) for k, v in flags.items()})
        # QC only (not an exclusion): the fit window itself is flat, i.e. the head sits at the
        # downstream lip of a filled flat that the +/- 10 m trace proxy does not catch
        rec["flag_upslope_flat"] = bool(np.isfinite(s[f"S_up_{cfg['L_default_m']:g}"]) and s[f"S_up_{cfg['L_default_m']:g}"] < cfg["flat_fill_slope_max"])
        rec["AS2"] = rec["area_m2"] * rec[f"S_up_{cfg['L_default_m']:g}"] ** 2
        out.append(rec)
    return out
