#!/usr/bin/env python3
"""Addendum 20, decision 2: Perron's hilltop estimator (Perron, Kirchner & Dietrich 2009, SI section 2.1), with the plateau range set by
the rule written in the directive instead of by eye.

Curvature: at every cell, the least-squares quadratic z = a + b x + c y + d x^2 + e x y + f y^2 over all cells whose centres lie within
radius r of the cell centre; |grad z| = sqrt(b^2 + c^2), lap z = 2 d + 2 f. For a full disk this is a linear filter, so every coefficient is
a correlation of z with a fixed kernel (rows of the pseudo-inverse of the design matrix). Cells within r + 2 cells of the tile edge or of
nodata are excluded, so every kept cell has a full disk and the filter is exact there.

Bins: A |grad z| (m^2; A the registered D8 area in m^2), log-spaced, 4 per decade, edges 1e-3 ... 1e5 m^2; cells below 1e-3 m^2 (including
A |grad z| = 0) go to the first bin; cells at or above 1e5 m^2 are counted and dropped. Per bin: n, fraction convex (lap < -1e-9 m^-1: smaller magnitudes are numerical zeros of the FFT on exactly planar cells),
L_bin = -10^<log10(-lap)> over the convex cells, and the signed arithmetic mean of lap.

Plateau rule (directive decision 2, as written): start at the lowest bin with >= 200 cells; add bins in increasing A |grad z| while the bin
has >= 200 cells, its convex fraction is >= 0.6 and |L_bin / L_plat - 1| <= 0.20 (L_plat the cell-weighted geometric mean of the bins already
included); stop at the first bin that fails; lap_h = the cell-weighted geometric mean over the included bins; fewer than 3 bins = no
plateau. "Cell-weighted" is read as weighted by the convex cells over which each bin's geometric mean is taken, so lap_h is the geometric
mean over every convex cell of the included bins (the start bin is not required to meet the convex-fraction condition; whether it does is
recorded).

Uncertainty: block bootstrap over 100 m x 100 m blocks (2,000 draws, seed 20261120); per-block bin tables are summed with multinomial
block weights and the plateau rule is re-applied in every draw.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from a20_common import (
    BIN_EDGES,
    BLOCK_M,
    FIXED_RANGES_M2,
    LAP_ZERO,
    N_BOOT_BLOCK,
    PLATEAU_MIN_BINS,
    PLATEAU_MIN_CELLS,
    PLATEAU_MIN_FRAC_CONVEX,
    PLATEAU_TOL,
    SEED,
)
from scipy.ndimage import distance_transform_edt
from scipy.signal import fftconvolve

N_BINS = BIN_EDGES.size - 1


# ----------------------------------------------------------------------------- the disk quadratic

def disk_offsets(r: float, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    kr, kc = int(math.floor(r / dy + 1e-9)), int(math.floor(r / dx + 1e-9))
    ii, jj = np.mgrid[-kr:kr + 1, -kc:kc + 1]
    inside = np.hypot(ii * dy, jj * dx) <= r + 1e-9
    return ii[inside], jj[inside]


def disk_kernels(r: float, dx: float, dy: float) -> dict[str, np.ndarray]:
    """Correlation kernels (on the (2k+1)^2 lattice, zero outside the disk) for b, c and lap = 2d + 2f; x east, y north."""
    ii, jj = disk_offsets(r, dx, dy)
    X = jj * dx                 # column offset -> east
    Y = -ii * dy                # row offset down -> south, so y north = -row
    G = np.column_stack([np.ones_like(X, dtype=float), X, Y, X * X, X * Y, Y * Y]).astype(float)
    P = np.linalg.pinv(G)
    kr, kc = int(ii.max()), int(jj.max())
    out = {}
    for name, w in (("b", P[1]), ("c", P[2]), ("lap", 2.0 * (P[3] + P[5]))):
        k = np.zeros((2 * kr + 1, 2 * kc + 1))
        k[ii + kr, jj + kc] = w
        out[name] = k
    out["n_cells"] = np.array(ii.size)
    return out


def correlate(z: np.ndarray, k: np.ndarray) -> np.ndarray:
    return fftconvolve(z, k[::-1, ::-1], mode="same")


def valid_interior(Z: np.ndarray, r: float, dx: float) -> np.ndarray:
    """Cells farther than r + 2 cells from the tile edge and from nodata."""
    bad = np.pad(~np.isfinite(Z), 1, constant_values=True)
    d = distance_transform_edt(~bad)[1:-1, 1:-1]          # distance in cells to the nearest bad cell or the frame
    return d > (r / dx + 2.0)


def disk_quadratic(Z: np.ndarray, r: float, dx: float, dy: float) -> dict[str, np.ndarray]:
    """|grad z| and lap z from the disk quadratic of radius r; NaN outside the valid interior."""
    k = disk_kernels(r, dx, dy)
    ok = valid_interior(Z, r, dx)
    zz = np.where(np.isfinite(Z), Z, np.nanmean(Z)) - np.nanmean(Z)
    b = correlate(zz, k["b"])
    c = correlate(zz, k["c"])
    lap = correlate(zz, k["lap"])
    grad = np.hypot(b, c)
    grad[~ok] = np.nan
    lap[~ok] = np.nan
    return {"grad": grad, "lap": lap, "valid": ok, "n_disk": int(k["n_cells"])}


def direct_quadratic(Z: np.ndarray, r: float, dx: float, dy: float, row: int, col: int) -> tuple[float, float]:
    """Independent check: the least-squares fit at one cell by numpy lstsq on the disk's points (|grad z|, lap)."""
    ii, jj = disk_offsets(r, dx, dy)
    z = Z[row + ii, col + jj].astype(float)
    X, Y = jj * dx, -ii * dy
    G = np.column_stack([np.ones_like(X, dtype=float), X, Y, X * X, X * Y, Y * Y]).astype(float)
    coef, *_ = np.linalg.lstsq(G, z, rcond=None)
    return float(math.hypot(coef[1], coef[2])), float(2 * coef[3] + 2 * coef[5])


# ----------------------------------------------------------------------------- bins and block tables

def bin_index(X: np.ndarray) -> np.ndarray:
    """Bin of A |grad z| (0 ... N_BINS - 1); below the first edge -> 0; at or above the last edge -> -1 (dropped)."""
    j = np.searchsorted(BIN_EDGES, X, side="right") - 1
    j = np.where(X < BIN_EDGES[0], 0, j)
    return np.where(X >= BIN_EDGES[-1], -1, j)


def block_index(shape: tuple[int, int], dx: float, dy: float) -> tuple[np.ndarray, int]:
    ny, nx = shape
    br = (np.arange(ny) * dy // BLOCK_M).astype(np.int64)
    bc = (np.arange(nx) * dx // BLOCK_M).astype(np.int64)
    nbc = int(bc.max()) + 1
    return br[:, None] * nbc + bc[None, :], int((int(br.max()) + 1) * nbc)


def block_tables(lap: np.ndarray, X: np.ndarray, blocks: np.ndarray, n_blocks: int, extra: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    """Per (block, bin): n, n_convex, sum log10(-lap) over convex, sum lap; per block the fixed ranges (convex cells with X <= 1 and 3 m^2).
    ``extra``: other curvature grids (same sign convention) whose convex count and log sums are tabulated on the same bins."""
    ok = np.isfinite(lap) & np.isfinite(X)
    b = bin_index(X[ok])
    keep = b >= 0
    blk = blocks[ok][keep]
    bb = b[keep]
    lv = lap[ok][keep]
    flat = blk * N_BINS + bb
    size = n_blocks * N_BINS
    conv = lv < -LAP_ZERO
    out = {"n": np.bincount(flat, minlength=size).reshape(n_blocks, N_BINS).astype(float),
           "n_convex": np.bincount(flat[conv], minlength=size).reshape(n_blocks, N_BINS).astype(float),
           "slog": np.bincount(flat[conv], weights=np.log10(-lv[conv]), minlength=size).reshape(n_blocks, N_BINS),
           "ssum": np.bincount(flat, weights=lv, minlength=size).reshape(n_blocks, N_BINS),
           "n_zero": np.bincount(flat[np.abs(lv) <= LAP_ZERO], minlength=size).reshape(n_blocks, N_BINS).astype(float)}
    Xk = X[ok][keep]
    for R in FIXED_RANGES_M2:
        m = conv & (Xk <= R)
        out[f"fixed_{R:g}_n"] = np.bincount(blk[m], minlength=n_blocks).astype(float)
        out[f"fixed_{R:g}_slog"] = np.bincount(blk[m], weights=np.log10(-lv[m]), minlength=n_blocks)
    for name, g in (extra or {}).items():
        gv = g[ok][keep]
        cg = np.isfinite(gv) & (gv < -LAP_ZERO)
        out[f"{name}_n_convex"] = np.bincount(flat[cg], minlength=size).reshape(n_blocks, N_BINS).astype(float)
        out[f"{name}_slog"] = np.bincount(flat[cg], weights=np.log10(-gv[cg]), minlength=size).reshape(n_blocks, N_BINS)
    out["n_dropped_above_last_edge"] = np.array(int((b < 0).sum()))
    out["n_below_first_edge"] = np.array(int((X[ok] < BIN_EDGES[0]).sum()))
    return out


# ----------------------------------------------------------------------------- the plateau rule

def plateau(n: np.ndarray, nconv: np.ndarray, slog: np.ndarray) -> dict[str, Any]:
    """The rule of decision 2 on one set of bin totals. Returns the included bins and lap_h (negative), or no plateau."""
    with np.errstate(divide="ignore", invalid="ignore"):
        L = -(10.0 ** (slog / nconv))                  # L_bin (negative); NaN where no convex cell
        fc = nconv / n
    start = next((k for k in range(N_BINS) if n[k] >= PLATEAU_MIN_CELLS), None)
    if start is None:
        return {"plateau": False, "reason": "no bin with >= 200 cells", "bins": []}
    inc = [start]
    stop_reason = "last bin"
    for k in range(start + 1, N_BINS):
        sl, sn = slog[inc].sum(), nconv[inc].sum()
        L_plat = -(10.0 ** (sl / sn)) if sn > 0 else np.nan
        if n[k] < PLATEAU_MIN_CELLS:
            stop_reason = f"bin {k}: n {int(n[k])} < 200"
            break
        if not fc[k] >= PLATEAU_MIN_FRAC_CONVEX:
            stop_reason = f"bin {k}: fraction convex {fc[k]:.3f} < 0.6"
            break
        if not (np.isfinite(L[k]) and np.isfinite(L_plat) and abs(L[k] / L_plat - 1.0) <= PLATEAU_TOL):
            stop_reason = f"bin {k}: L_bin/L_plat - 1 = {L[k] / L_plat - 1.0:+.3f}"
            break
        inc.append(k)
    sl, sn = slog[inc].sum(), nconv[inc].sum()
    lap_h = -(10.0 ** (sl / sn)) if sn > 0 else np.nan
    rec = {"plateau": len(inc) >= PLATEAU_MIN_BINS and np.isfinite(lap_h), "bins": inc, "n_bins": len(inc), "start_bin": start, "stop": stop_reason,
           "start_bin_frac_convex": float(fc[start]) if n[start] > 0 else None, "start_bin_meets_convex_condition": bool(fc[start] >= PLATEAU_MIN_FRAC_CONVEX),
           "range_m2": [float(BIN_EDGES[inc[0]]), float(BIN_EDGES[inc[-1] + 1])], "n_cells": int(n[inc].sum()), "n_convex_cells": int(sn), "lap_h": float(lap_h)}
    if not rec["plateau"]:
        rec["reason"] = f"{len(inc)} bins included (< 3)"
    return rec


def bin_curve(n: np.ndarray, nconv: np.ndarray, slog: np.ndarray, ssum: np.ndarray) -> list[dict[str, Any]]:
    out = []
    for k in range(N_BINS):
        out.append({"bin": k, "lo_m2": float(BIN_EDGES[k]), "hi_m2": float(BIN_EDGES[k + 1]), "n": int(n[k]),
                    "frac_convex": float(nconv[k] / n[k]) if n[k] > 0 else None,
                    "L_bin": float(-(10.0 ** (slog[k] / nconv[k]))) if nconv[k] > 0 else None,
                    "mean_signed": float(ssum[k] / n[k]) if n[k] > 0 else None})
    return out


def estimate(tab: dict[str, np.ndarray], n_boot: int = N_BOOT_BLOCK, seed: int = SEED, extra: tuple[str, ...] = (), _noflat: bool = True) -> dict[str, Any]:
    """Point estimate, bin curve, fixed ranges and the block bootstrap from per-block tables (one tile, or several tiles stacked).
    Also (labelled sensitivity, not the rule as written) the same with the numerical-zero cells (|lap| <= 1e-9 m^-1: pit-filled flats)
    removed from the bin counts, under "no_flat"."""
    n, nc, sl, ss = (tab[k].sum(axis=0) for k in ("n", "n_convex", "slog", "ssum"))
    pt = plateau(n, nc, sl)
    rec: dict[str, Any] = {"point": pt, "curve": bin_curve(n, nc, sl, ss), "n_blocks": int(tab["n"].shape[0]),
                           "n_blocks_with_cells": int((tab["n"].sum(axis=1) > 0).sum())}
    for R in FIXED_RANGES_M2:
        fn, fs = tab[f"fixed_{R:g}_n"].sum(), tab[f"fixed_{R:g}_slog"].sum()
        rec[f"fixed_{R:g}"] = {"n_convex": int(fn), "lap": float(-(10.0 ** (fs / fn))) if fn > 0 else None}
    for name in extra:
        en, es = tab[f"{name}_n_convex"].sum(axis=0), tab[f"{name}_slog"].sum(axis=0)
        if pt["plateau"]:
            b = pt["bins"]
            rec[f"{name}_on_plateau_bins"] = {"n_convex": int(en[b].sum()), "lap": float(-(10.0 ** (es[b].sum() / en[b].sum()))) if en[b].sum() > 0 else None}
    if n_boot > 0:
        nb = tab["n"].shape[0]
        rng = np.random.default_rng(seed)
        W = np.stack([np.bincount(rng.integers(0, nb, nb), minlength=nb) for _ in range(n_boot)]).astype(float)
        Nb, NCb, SLb = W @ tab["n"], W @ tab["n_convex"], W @ tab["slog"]
        laps, nbins, lo_e, hi_e = np.full(n_boot, np.nan), np.zeros(n_boot, int), np.full(n_boot, np.nan), np.full(n_boot, np.nan)
        for i in range(n_boot):
            p = plateau(Nb[i], NCb[i], SLb[i])
            nbins[i] = p.get("n_bins", 0)
            if p["plateau"]:
                laps[i] = p["lap_h"]
                lo_e[i], hi_e[i] = p["range_m2"]
        okb = np.isfinite(laps)
        rec["boot"] = {"n_boot": n_boot, "seed": seed, "share_with_plateau": float(okb.mean()), "n_bins_median": float(np.median(nbins)),
                       "lap_ci68": [float(np.percentile(laps[okb], 16)), float(np.percentile(laps[okb], 84))] if okb.sum() >= 10 else None,
                       "lap_ci95": [float(np.percentile(laps[okb], 2.5)), float(np.percentile(laps[okb], 97.5))] if okb.sum() >= 10 else None,
                       "range_lo_median_m2": float(np.nanmedian(lo_e)) if okb.any() else None, "range_hi_median_m2": float(np.nanmedian(hi_e)) if okb.any() else None}
        for R in FIXED_RANGES_M2:
            fnb, fsb = W @ tab[f"fixed_{R:g}_n"], W @ tab[f"fixed_{R:g}_slog"]
            with np.errstate(divide="ignore", invalid="ignore"):
                v = -(10.0 ** (fsb / fnb))
            v = v[np.isfinite(v)]
            rec[f"fixed_{R:g}"]["lap_ci68"] = [float(np.percentile(v, 16)), float(np.percentile(v, 84))] if v.size >= 10 else None
    if _noflat and "n_zero" in tab:
        t2 = dict(tab)
        t2["n"] = tab["n"] - tab["n_zero"]
        nf = estimate(t2, n_boot=n_boot, seed=seed, _noflat=False)
        rec["no_flat"] = {"point": nf["point"], "boot": nf.get("boot"), "n_zero_cells": int(tab["n_zero"].sum())}
    return rec


def d0_from_lap(E: float, rec: dict[str, Any]) -> dict[str, Any]:
    """D0_P = E / |lap_h| with the interval (the more negative lap bound -> the lower D)."""
    out: dict[str, Any] = {"D0_P": None, "plateau": bool(rec["point"]["plateau"])}
    if rec["point"]["plateau"]:
        out["D0_P"] = float(E / abs(rec["point"]["lap_h"]))
        out["lap_h"] = rec["point"]["lap_h"]
    ci = rec.get("boot", {}).get("lap_ci68")
    if ci:
        out["D0_P_ci68"] = [float(E / abs(ci[0])), float(E / abs(ci[1]))]
        out["lap_ci68"] = ci
    for R in FIXED_RANGES_M2:
        v = rec[f"fixed_{R:g}"]["lap"]
        out[f"D0_fixed_{R:g}"] = float(E / abs(v)) if v else None
    nf = rec.get("no_flat")
    if nf:
        out["no_flat_plateau"] = bool(nf["point"]["plateau"])
        out["D0_P_no_flat"] = float(E / abs(nf["point"]["lap_h"])) if nf["point"]["plateau"] else None
        ci = (nf.get("boot") or {}).get("lap_ci68")
        out["D0_P_no_flat_ci68"] = [float(E / abs(ci[0])), float(E / abs(ci[1]))] if ci else None
        out["no_flat_range_m2"] = nf["point"].get("range_m2")
    return out
