#!/usr/bin/env python3
"""Addendum 20 supplement: Perron's hilltop Laplacian restricted to first-order drainage basins, as his Fig. 3a caption and SI 2.1 state
("Plot of the Laplacian of elevation against the product of drainage area and slope for first-order drainage basins in Gabilan Mesa").
Decision 2 of the directive pooled every cell of each tile; this run applies the same binning only to the cells of first-order basins
(network from A_t, steepest descent, basins draining to the point just upslope of the junction with a higher-order stream; `a20_sstar_perron2`).

GM (reproduction check, NCALM tiles = Perron's survey): A_t = 180^2 m2 (Perron et al. 2008). TTV (M&D polygon, full-site grid window):
A_t = 7,400 m2 (M&D Table 1 median, reproduced at map scale) and 180^2 m2. Per bin of A|grad z| (4 per decade): n, the signed mean of lap z,
the geometric mean of |lap z| over convex cells, the fraction convex; the decision-2 plateau rule on these bins; and the signed mean over
A|grad z| <= 1 m2 and <= 3 m2 (Perron's by-eye range). Block bootstrap over basins (500 draws).
Writes <run>/ttv_cal/lap_h_basins.json.
usage: a20_lap_h_basins.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a20_common import GM_NCALM, LAP_ZERO, SEED, dump, metre_tile, out_path, slurm_record  # noqa: E402
from a20_perron import N_BINS, bin_index, disk_quadratic, plateau  # noqa: E402
from a20_sstar_perron import strahler_first_order  # noqa: E402
from a20_sstar_perron2 import basin_cells  # noqa: E402
from a20_ttv_cal import md_polygon_on, read_window  # noqa: E402

PERRON = {"GM": -11.8e-3, "AP": -5.5e-3}


def tables(lap, X, lab):
    ok = np.isfinite(lap) & np.isfinite(X) & (lab >= 0)
    b = bin_index(X[ok])
    keep = b >= 0
    lv, bb, bl, xv = lap[ok][keep], b[keep], lab[ok][keep], X[ok][keep]
    return {"lap": lv, "bin": bb, "basin": bl, "X": xv}


def summarize(T, n_boot=500) -> dict:
    lv, bb, bl, xv = T["lap"], T["bin"], T["basin"], T["X"]
    conv = lv < -LAP_ZERO

    def bins(sel):
        n = np.bincount(bb[sel], minlength=N_BINS).astype(float)
        nc = np.bincount(bb[sel & conv], minlength=N_BINS).astype(float)
        sl = np.bincount(bb[sel & conv], weights=np.log10(-lv[sel & conv]), minlength=N_BINS)
        ss = np.bincount(bb[sel], weights=lv[sel], minlength=N_BINS)
        return n, nc, sl, ss

    allm = np.ones(lv.size, dtype=bool)
    n, nc, sl, ss = bins(allm)
    p = plateau(n, nc, sl)
    rec = {"n_cells": int(lv.size), "n_basins": int(np.unique(bl).size), "plateau_rule": p,
           "curve": [{"bin": k, "n": int(n[k]), "signed_mean": float(ss[k] / n[k]) if n[k] else None, "geomean_convex": float(-(10 ** (sl[k] / nc[k]))) if nc[k] else None,
                      "frac_convex": float(nc[k] / n[k]) if n[k] else None} for k in range(N_BINS)]}
    for R in (1.0, 3.0):
        m = xv <= R
        rec[f"signed_mean_X_le_{R:g}"] = float(lv[m].mean()) if m.any() else None
        rec[f"geomean_convex_X_le_{R:g}"] = float(-(10 ** np.mean(np.log10(-lv[m & conv])))) if (m & conv).any() else None
        rec[f"frac_convex_X_le_{R:g}"] = float(conv[m].mean()) if m.any() else None
        rec[f"n_X_le_{R:g}"] = int(m.sum())
    uniq, inv = np.unique(bl, return_inverse=True)
    rng = np.random.default_rng(SEED)
    bs = {"signed_mean_X_le_1": [], "signed_mean_X_le_3": [], "plateau_lap_h": []}
    groups = [np.flatnonzero(inv == i) for i in range(uniq.size)]
    for _ in range(n_boot):
        ii = np.concatenate([groups[j] for j in rng.integers(0, uniq.size, uniq.size)])
        for R in (1.0, 3.0):
            m = xv[ii] <= R
            bs[f"signed_mean_X_le_{R:g}"].append(lv[ii][m].mean() if m.any() else np.nan)
        b2, l2, c2 = bb[ii], lv[ii], conv[ii]
        n_ = np.bincount(b2, minlength=N_BINS).astype(float)
        nc_ = np.bincount(b2[c2], minlength=N_BINS).astype(float)
        sl_ = np.bincount(b2[c2], weights=np.log10(-l2[c2]), minlength=N_BINS)
        pp = plateau(n_, nc_, sl_)
        bs["plateau_lap_h"].append(pp["lap_h"] if pp["plateau"] else np.nan)
    for k, v in bs.items():
        v = np.array(v, dtype=float)
        rec[f"{k}_ci68"] = [float(np.nanpercentile(v, 16)), float(np.nanpercentile(v, 84))] if np.isfinite(v).sum() >= 10 else None
        if k == "plateau_lap_h":
            rec["plateau_share_boot"] = float(np.isfinite(v).mean())
    return rec


def run(tiles_or_window, A_t, mask=None) -> dict:
    parts = []
    off = 0
    for Z, A, FD, dx in tiles_or_window:
        q = disk_quadratic(Z, 7.0, dx, dx)
        _first, basin, proper = strahler_first_order(A, FD, A_t)
        lab = basin_cells(A, FD, basin, proper)
        if mask is not None:
            lab = np.where(mask, lab, -1)
        T = tables(q["lap"], A * q["grad"], np.where(lab >= 0, lab + off, -1))
        parts.append(T)
        off += 10_000_000_000
    T = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    return summarize(T)


def main() -> None:
    t0 = time.time()
    slurm_record(os.path.dirname(out_path("ttv_cal", "lap_h_basins.json")))
    out: dict = {"perron_lap_h": PERRON}
    gm = []
    for t in GM_NCALM:
        td = metre_tile(t)
        gm.append((td.Z, td.A, td.FD, td.dx))
    r = run(gm, 180.0 ** 2)
    r["ratio_signed_le1_to_perron"] = r["signed_mean_X_le_1"] / PERRON["GM"]
    r["ratio_signed_le3_to_perron"] = r["signed_mean_X_le_3"] / PERRON["GM"]
    r["ratio_plateau_to_perron"] = (r["plateau_rule"]["lap_h"] / PERRON["GM"]) if r["plateau_rule"]["plateau"] else None
    out["GM_ncalm_At180m"] = r
    print(f"GM first-order basins (A_t 180^2): plateau {r['plateau_rule'].get('plateau')} {r['plateau_rule'].get('range_m2')} lap_h {r['plateau_rule'].get('lap_h')}; "
          f"signed mean <=1 {r['signed_mean_X_le_1']} <=3 {r['signed_mean_X_le_3']} (Perron -0.0118); basins {r['n_basins']} [{time.time() - t0:.0f} s]", flush=True)
    W = read_window()
    poly = md_polygon_on(W["Z"].shape, W["tr"])
    for tag, A_t in (("TTV_At7400", 7400.0), ("TTV_At180m", 180.0 ** 2)):
        r = run([(W["Z"], W["A"], W["FD"], W["dx"])], A_t, mask=poly)
        out[tag] = r
        print(f"{tag}: plateau {r['plateau_rule'].get('plateau')} {r['plateau_rule'].get('range_m2')} lap_h {r['plateau_rule'].get('lap_h')}; signed mean <=1 {r['signed_mean_X_le_1']} "
              f"<=3 {r['signed_mean_X_le_3']}; geomean convex <=1 {r['geomean_convex_X_le_1']}; basins {r['n_basins']} [{time.time() - t0:.0f} s]", flush=True)
    out["wall_s"] = time.time() - t0
    dump(out, out_path("ttv_cal", "lap_h_basins.json"))


if __name__ == "__main__":
    main()
