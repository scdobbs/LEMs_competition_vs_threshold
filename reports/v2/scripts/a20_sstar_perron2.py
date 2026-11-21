#!/usr/bin/env python3
"""Addendum 20 supplement, T3 as published: Perron et al. (2009) SI 2.1 with the network threshold of Perron et al. (2008, JGR 113, F04003).

Perron et al. (2008) section 3.1.2: the spectral roll-off is found by fitting regression lines to the log-transformed, binned 1-D (radially
collapsed) power spectrum, "beginning with the lowest frequency point, and subsequently moving through all the points, the line was fit to all
points at frequencies greater than or equal to the present point. The roll-off frequency was identified as the point at which the magnitude of
the regression line slope reached a maximum, and above which the goodness of fit (r2) reached a plateau". Gabilan Mesa: 5.6e-3 1/m
(wavelength 180 m). Perron et al. (2009) SI 2.1: the network starts at A_t = (roll-off wavelength)^2 (here 32,400 m2 at GM), steepest
descent; first-order links; basins drain to the point just upslope of the junction with a second-order stream; D/K and m by iteratively
reweighted least squares to the mean log10 S* in log bins of A; points from basins showing power-law S*-A relations, pooled.

Here: (1) our implementation of the 2008 roll-off procedure on each NCALM GM tile (check: about 180 m) and on the TTV window; (2) S* points
either the main-stem stream profile of each first-order basin (outlet upward along the largest-area donor to A = 100 m2; Perron 2009 Fig. 3b:
"stream profiles in the same basins") or every cell of the basin with A >= 100 m2; basins all, or selected objectively (own log S*-log A slope
< 0, R2 >= 0.5, >= 10 points); A_t = Perron's 32,400 m2 at GM, and our roll-off^2 at GM and TTV. GM is the reproduction check against
D/K = 124 +- 3, m = 0.35 +- 0.003. Writes <run>/ttv_cal/sstar_perron2.json.
usage: a20_sstar_perron2.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a20_common import GM_NCALM, dump, load_json, metre_tile, out_path, slurm_record  # noqa: E402
from a20_perron import disk_quadratic  # noqa: E402
from a20_sstar_perron import fit_all, strahler_first_order  # noqa: E402
from a20_ttv_cal import md_polygon_on, read_window  # noqa: E402

from lemcvt.hydro.d8 import d8_receiver_arrays  # noqa: E402

PERRON_GM = {"D_over_K": 124.0, "D_over_K_se": 3.0, "m": 0.35, "m_se": 0.003, "lap_h": -11.8e-3, "rolloff_wavelength_m": 180.0}
A_PROFILE_MIN = 100.0


def rolloff(Z, dx) -> dict:
    z = np.where(np.isfinite(Z), Z, np.nanmean(Z))
    ny, nx = z.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    G = np.column_stack([np.ones(z.size), xx.ravel(), yy.ravel()])
    c, *_ = np.linalg.lstsq(G, z.ravel(), rcond=None)
    z = z - (G @ c).reshape(z.shape)
    w = np.outer(np.hanning(ny), np.hanning(nx))
    P = np.abs(np.fft.fft2(z * w)) ** 2
    fy, fx = np.fft.fftfreq(ny, dx), np.fft.fftfreq(nx, dx)
    f = np.hypot(*np.meshgrid(fy, fx, indexing="ij"))
    edges = np.logspace(np.log10(1.0 / (min(ny, nx) * dx)), np.log10(0.5 / dx), 41)
    lf, lp = [], []
    for i in range(edges.size - 1):
        s = (f >= edges[i]) & (f < edges[i + 1])
        if s.sum() >= 5:
            lf.append(np.log10(np.sqrt(edges[i] * edges[i + 1])))
            lp.append(np.log10(np.mean(P[s])))
    lf, lp = np.array(lf), np.array(lp)
    rows = []
    for i in range(lf.size - 3):
        x, y = lf[i:], lp[i:]
        sl, ic = np.polyfit(x, y, 1)
        r2 = 1 - np.sum((y - (ic + sl * x)) ** 2) / np.sum((y - y.mean()) ** 2)
        rows.append((10 ** lf[i], -sl, r2))
    j = int(np.argmax([r[1] for r in rows]))
    return {"rolloff_frequency": float(rows[j][0]), "rolloff_wavelength_m": float(1.0 / rows[j][0]), "beta_max": float(rows[j][1]), "r2_at_max": float(rows[j][2]),
            "curve": [{"f": float(a), "beta": float(b), "r2": float(c_)} for a, b, c_ in rows]}


def basin_cells(A, FD, basin, proper):
    """Every cell draining to a proper first-order outlet: the outlet id per cell (or -1)."""
    r_to, c_to, good = d8_receiver_arrays(FD)
    ny, nx = A.shape
    lab = np.full(ny * nx, -1, dtype=np.int64)
    for o, ok in proper.items():
        if ok:
            lab[o] = o
    rec = np.where(good, r_to * nx + c_to, -1).ravel()
    order = np.argsort(-np.nan_to_num(A.ravel(), nan=-1.0), kind="stable")       # receivers before donors
    for f in order:
        if lab[f] < 0:
            r = rec[f]
            if r >= 0 and lab[r] >= 0:
                lab[f] = lab[r]
    return lab.reshape(ny, nx)


def main_stems(A, FD, proper):
    r_to, c_to, good = d8_receiver_arrays(FD)
    ny, nx = A.shape
    Af = np.nan_to_num(A.ravel(), nan=-1.0)
    rec = np.where(good, r_to * nx + c_to, -1).ravel()
    best = np.full(ny * nx, -1, dtype=np.int64)
    src = np.flatnonzero(rec >= 0)
    for f in src[np.argsort(Af[src], kind="stable")]:
        best[rec[f]] = f
    lab = np.full(ny * nx, -1, dtype=np.int64)
    for o, ok in proper.items():
        if not ok:
            continue
        cur = o
        while cur >= 0 and Af[cur] >= A_PROFILE_MIN:
            lab[cur] = o
            cur = best[cur]
    return lab.reshape(ny, nx)


def points(Z, A, FD, dx, lap_h, A_t, mask=None):
    q = disk_quadratic(Z, 7.0, dx, dx)
    _first, _basin, proper = strahler_first_order(A, FD, A_t)
    d = q["lap"] - lap_h
    good = np.isfinite(d) & (d > 0) & np.isfinite(q["grad"]) & (q["grad"] > 0) & np.isfinite(A) & (A >= A_PROFILE_MIN)
    if mask is not None:
        good &= mask
    out = {}
    for name, lab in (("main_stem_profiles", main_stems(A, FD, proper)), ("basin_cells", basin_cells(A, FD, _basin, proper))):
        k = good & (lab >= 0)
        out[name] = {"la": np.log10(A[k]), "ls": np.log10(q["grad"][k] / d[k]), "bb": lab[k]}
    return out, {"A_t_m2": A_t, "n_first_order_basins_proper": int(sum(proper.values()))}


def pool(parts):
    off = 0
    la, ls, bb = [], [], []
    for p in parts:
        la.append(p["la"])
        ls.append(p["ls"])
        bb.append(p["bb"] + off)
        off += 10_000_000_000
    return {"la": np.concatenate(la), "ls": np.concatenate(ls), "bb": np.concatenate(bb)}


def main() -> None:
    t0 = time.time()
    slurm_record(os.path.dirname(out_path("ttv_cal", "sstar_perron2.json")))
    cal = load_json(out_path("ttv_cal", "ttv_cal.json"))
    out: dict = {"published_gm": PERRON_GM, "gm": {"rolloff": {}}, "ttv": {}}
    tiles = {}
    for t in GM_NCALM:
        td = metre_tile(t)
        tiles[t] = td
        out["gm"]["rolloff"][t] = rolloff(td.Z, td.dx)
        print(f"GM {t}: roll-off {out['gm']['rolloff'][t]['rolloff_wavelength_m']:.0f} m (Perron 180 m)", flush=True)
    our_ro = float(np.median([v["rolloff_wavelength_m"] for v in out["gm"]["rolloff"].values()]))
    gm_crest = cal["T3"]["gm_check"]["our_crest_disk7_lap_h"]
    for at_tag, A_t in (("perron_180m", PERRON_GM["rolloff_wavelength_m"] ** 2), ("our_rolloff", our_ro ** 2)):
        for lh_tag, lh in (("perron_lap_h", PERRON_GM["lap_h"]), ("our_crest", gm_crest)):
            parts = {"main_stem_profiles": [], "basin_cells": []}
            info = {}
            for t, td in tiles.items():
                p, inf = points(td.Z, td.A, td.FD, td.dx, lh, A_t)
                info[t] = inf
                for k in parts:
                    parts[k].append(p[k])
            for k, plist in parts.items():
                C = pool(plist)
                for sel in (True, False):
                    key = f"{at_tag}:{lh_tag}:{k}:{'selected' if sel else 'all'}"
                    r = fit_all(C, sel) | {"A_t_m2": A_t, "tiles": info}
                    out["gm"][key] = r
                    print(f"GM {key}: D/K {r.get('D_over_K')} {r.get('D_over_K_ci68')} m {r.get('m')} {r.get('m_ci68')} basins {r.get('n_basins')}/{r.get('n_basins_all')} "
                          f"(Perron 124, 0.35) [{time.time() - t0:.0f} s]", flush=True)
    W = read_window()
    poly = md_polygon_on(W["Z"].shape, W["tr"])
    ro = rolloff(W["Z"], W["dx"])
    out["ttv"]["rolloff_window"] = ro
    print(f"TTV window roll-off {ro['rolloff_wavelength_m']:.0f} m", flush=True)
    lh = cal["T3"]["lap_h_crest"]["disk7_median"]
    # the TTV window roll-off (863 m on the 2.8 x 3.6 km window) gives A_t larger than the M&D basins, so the field-mapped heads set the
    # threshold: M&D Table 1 median source area 7,400 m2 (reproduced at map scale, T2); Perron's GM 32,400 m2 and the roll-off beside it
    for at_tag, A_t in (("md_table1_median", 7400.0), ("gm_perron_180m", PERRON_GM["rolloff_wavelength_m"] ** 2), ("our_rolloff", ro["rolloff_wavelength_m"] ** 2)):
        p, inf = points(W["Z"], W["A"], W["FD"], W["dx"], lh, A_t, mask=poly)
        for k, C in p.items():
            for sel in (False, True):
                key = f"{at_tag}:crest:{k}:{'selected' if sel else 'all'}"
                r = fit_all(C, sel) | {"A_t_m2": A_t, "info": inf}
                out["ttv"][key] = r
                print(f"TTV {key}: D/K {r.get('D_over_K')} {r.get('D_over_K_ci68')} m {r.get('m')} {r.get('m_ci68')} basins {r.get('n_basins')}/{r.get('n_basins_all')} [{time.time() - t0:.0f} s]", flush=True)
    out["wall_s"] = time.time() - t0
    dump(out, out_path("ttv_cal", "sstar_perron2.json"))


if __name__ == "__main__":
    main()
