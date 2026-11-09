"""The v1 measurement method, re-implemented from ``legacy/measurement_notebooks`` for
comparison only (addendum 1, item 6). Not a specification.

theta, k_s (``updated_best_fit_tile_finder.ipynb`` cell 11)
    Outlets = every tile pixel returned by TopoAnalysis ``outlets_at_coastlines``
    (the ring where a 3-iteration dilation and erosion of the data mask differ: the
    tile edge). For each outlet a depth-first upstream walk over channel pixels
    (``A >= A_MIN``, 5.04e4 m^2 on AP and 1e4 m^2 on GM) accumulates chi by the
    trapezoid rule with A0 = 1; z is relative to the outlet. theta: coarse grid
    0.10-0.80 (29 steps) then a 31-step refinement +-0.06 around the best, minimising
    the total SSR of per-basin through-origin fits. k_s: pooled through-origin fit at
    theta*. Only outlets with >= 3 points contribute.

hilltop curvature (cell 12)
    Block-mean the filled grid to 30 m (AP) or 10 m (GM), recompute D8 and area,
    take curvature at "first-order divides" with a 9-point stencil through
    ``topo_analysis.extract_hilltop_curvature_first_order`` (module not available),
    then the median magnitude. Here the missing step is approximated by the divides
    of the coarse D8 grid between basins of the coarse channel network at the Taylor
    threshold and a 3x3 Laplacian; flagged as an approximation in the output.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion

from lemcvt.hydro.d8 import d8_area, d8_divides, d8_flow_direction, d8_upstream_lists
from lemcvt.measure.hilltop_curvature import basin_labels

V1_A_MIN = {"AP": 5.04e4, "GM": 1.0e4}
V1_SUBSAMPLE = {"AP": 30, "GM": 10}
V1_U_M_PER_MYR = {"AP": 55.0, "GM": 147.0}


@dataclass
class LegacyChiResult:
    A_min: float
    theta_coarse: float
    theta_star: float
    ks: float
    r2: float
    n_points: int
    n_basins: int
    n_outlets: int

    def as_dict(self):
        return asdict(self)


def coastline_outlets(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Port of ``Elevation.outlets_at_coastlines``: ring cells of the data mask."""
    m1 = (z > 0).astype(np.uint8)
    m2 = binary_dilation(m1, iterations=3).astype(np.uint8)
    m3 = binary_erosion(m1, iterations=3).astype(np.uint8)
    m3 = (m3.astype(float) + m2.astype(float)).astype(np.uint8)
    return np.where(m3 == 1)


def _chi_traverse(theta, root, A, Z, ups, dx, a_min):
    ny, nx = Z.shape
    ri, rj = root
    if not np.isfinite(Z[ri, rj]):
        return np.empty(0), np.empty(0)
    z0 = Z[ri, rj]
    chi_vals, z_vals = [0.0], [0.0]
    visited = np.zeros_like(Z, dtype=bool)
    visited[ri, rj] = True
    stack = [((ri, rj), 0.0)]
    while stack:
        (pi, pj), parent_chi = stack.pop()
        for ui, uj in ups[pi * nx + pj]:
            if visited[ui, uj] or not np.isfinite(Z[ui, uj]):
                continue
            if not (np.isfinite(A[ui, uj]) and A[ui, uj] >= a_min):
                continue
            a1 = A[pi, pj] ** (-theta) if (A[pi, pj] > 0 and np.isfinite(A[pi, pj])) else np.nan
            a2 = A[ui, uj] ** (-theta)
            next_chi = parent_chi
            if np.isfinite(a1) and np.isfinite(a2):
                seg = dx * (np.sqrt(2.0) if (abs(ui - pi) == 1 and abs(uj - pj) == 1) else 1.0)
                next_chi = parent_chi + 0.5 * (a1 + a2) * seg
            chi_vals.append(next_chi)
            z_vals.append(Z[ui, uj] - z0)
            visited[ui, uj] = True
            stack.append(((ui, uj), next_chi))
    x = np.asarray(chi_vals, float)
    y = np.asarray(z_vals, float)
    m = np.isfinite(x) & np.isfinite(y) & (x > 0.0)
    return x[m], y[m]


def _ols_origin(x, y):
    if x.size < 3:
        return np.nan, np.nan
    den = float(np.dot(x, x))
    if den <= 0 or not np.isfinite(den):
        return np.nan, np.nan
    k = float(np.dot(x, y) / den)
    r = y - k * x
    return k, float(np.dot(r, r))


def legacy_chi(z: np.ndarray, area: np.ndarray, fd: np.ndarray, dx: float, site: str,
               a_min: float | None = None) -> LegacyChiResult:
    a_min = V1_A_MIN[site] if a_min is None else a_min
    rows, cols = coastline_outlets(z)
    outlets = list(zip(rows.tolist(), cols.tolist(), strict=True))
    ups = d8_upstream_lists(fd)
    # only outlets with channel pixels upstream matter; pre-filter to speed up (same result)
    outlets = [o for o in outlets if np.isfinite(area[o]) and area[o] >= a_min]

    def total_ssr(theta):
        tot = 0.0
        for o in outlets:
            x, y = _chi_traverse(theta, o, area, z, ups, dx, a_min)
            _k, ssr = _ols_origin(x, y)
            if np.isfinite(ssr):
                tot += ssr
        return tot

    coarse = np.linspace(0.10, 0.80, 29)
    c = [(th, total_ssr(th)) for th in coarse]
    th_c, _ = min(c, key=lambda kv: kv[1])
    lo, hi = max(0.10, th_c - 0.06), min(0.80, th_c + 0.06)
    refine = np.linspace(lo, hi, 31)
    r = [(th, total_ssr(th)) for th in refine]
    th_star, _ = min(r, key=lambda kv: kv[1])
    xs, ys, nb = [], [], 0
    for o in outlets:
        x, y = _chi_traverse(th_star, o, area, z, ups, dx, a_min)
        if x.size >= 3:
            xs.append(x)
            ys.append(y)
            nb += 1
    if nb == 0:
        return LegacyChiResult(a_min, float(th_c), float(th_star), np.nan, np.nan, 0, 0, len(outlets))
    X = np.concatenate(xs)
    Y = np.concatenate(ys)
    ks, ssr = _ols_origin(X, Y)
    denom = float(np.dot(Y, Y))
    r2 = 1.0 - ssr / denom if denom > 0 else np.nan
    return LegacyChiResult(a_min, float(th_c), float(th_star), float(ks), float(r2), int(X.size), nb, len(outlets))


def block_mean(arr: np.ndarray, k: int) -> np.ndarray:
    ny, nx = arr.shape
    nyk, nxk = ny // k, nx // k
    return arr[:nyk * k, :nxk * k].reshape(nyk, k, nxk, k).mean(axis=(1, 3))


def legacy_curvature(z: np.ndarray, dx: float, site: str, U: float) -> dict:
    """Approximation of v1 cell 12 (block-mean, coarse D8, divides, 3x3 Laplacian)."""
    k = V1_SUBSAMPLE[site]
    zs = block_mean(z, k)
    dxs = dx * k
    zf = d8_flow_direction(zs)
    area_s = d8_area(zf, zs, dxs)
    channel = area_s >= V1_A_MIN[site]
    lab = basin_labels(area_s, zf, channel)
    div = d8_divides(zf).astype(bool)
    lap = np.full(zs.shape, np.nan)
    lap[1:-1, 1:-1] = ((zs[1:-1, 2:] - 2 * zs[1:-1, 1:-1] + zs[1:-1, :-2]) + (zs[2:, 1:-1] - 2 * zs[1:-1, 1:-1] + zs[:-2, 1:-1])) / dxs**2
    neighbours_differ = np.zeros_like(div)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            sh = np.roll(np.roll(lab, dr, axis=0), dc, axis=1)
            neighbours_differ |= (sh != lab) & (sh > 0) & (lab > 0)
    sel = div & neighbours_differ & ~channel & np.isfinite(lap)
    vals = lap[sel]
    neg = vals[vals < 0]
    if neg.size >= 5:
        mag = float(np.median(-neg))
    elif vals.size >= 10:
        mag = float(np.median(np.abs(vals)))
    else:
        mag = np.nan
    return {"approximation": "topo_analysis.extract_hilltop_curvature_first_order unavailable; divides between coarse "
                             "basins + 3x3 Laplacian", "subsample_m": k, "dx_coarse": dxs, "n_hilltops": int(vals.size),
            "n_negative": int(neg.size), "C_ht": mag, "D0": float(U / mag) if np.isfinite(mag) and mag > 0 else np.nan}
