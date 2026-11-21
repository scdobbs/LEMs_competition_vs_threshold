#!/usr/bin/env python3
"""Addendum 20 supplement, T3 as Perron et al. (2009) Supplementary Information section 2.1 describes it, on Perron's own survey (NCALM
Gabilan Mesa, GM1-GM7: the reproduction check against his D/K = 124 +- 3 m^(2m+1), m = 0.35 +- 0.003) and on TTV1-TTV3 (M&D polygon).

Procedure (SI 2.1, quoted in the report):
  * 1 m grid; |grad z| and lap z from the least-squares quadratic within 7 m (`a20_perron.disk_quadratic`, verified);
  * network: steepest descent (D8, the registered grids) from a threshold area A_t = (wavelength of the kink in the power spectrum)^2.
    The kink is found here per tile by a two-segment fit to the radially averaged 2-D power spectrum (detrended, Hann-windowed) in
    log-log over wavelengths 2-200 m (Perron's own kink values are in a reference not available here); fixed A_t = 100, 400, 1,000 m2
    are reported beside it;
  * first-order streams: network links with no tributaries (Strahler order 1); basin outlet: the point just upslope of the junction with a
    higher-order stream; points: the first-order stream cells (the stream profiles of Fig. 3b) with lap z > lap_h;
  * basins "showing power-law relationships between S* and A" (Perron's judgement) are selected here objectively: the basin's own OLS of
    log S* on log A has a negative slope and R^2 >= 0.5 (>= 10 points); all basins reported beside it;
  * D/K and m from an iteratively reweighted least-squares fit (Tukey bisquare, c = 4.685, MAD scale) to the mean log10 S* in 0.1-decade
    bins of A (bins with >= 5 points); uncertainty by bootstrap over basins (500 draws, seed 20261120).
lap_h: GM Perron's -11.8e-3 and our crest value (7 m disk); TTV our crest value (7 m disk) in the M&D polygon.
Writes <run>/ttv_cal/sstar_perron.json.
usage: a20_sstar_perron.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a19_hilltops import md_polygon_mask  # noqa: E402
from a20_common import GM_NCALM, SEED, dump, load_json, metre_tile, out_path, slurm_record  # noqa: E402
from a20_perron import disk_quadratic  # noqa: E402

from lemcvt.hydro.d8 import d8_receiver_arrays  # noqa: E402

PERRON_GM = {"D_over_K": 124.0, "D_over_K_se": 3.0, "m": 0.35, "m_se": 0.003, "lap_h": -11.8e-3}
FIXED_AT = (100.0, 400.0, 1000.0)


def spectrum_kink(Z, dx) -> dict:
    z = np.where(np.isfinite(Z), Z, np.nanmean(Z))
    ny, nx = z.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    G = np.column_stack([np.ones(z.size), xx.ravel(), yy.ravel()])
    c, *_ = np.linalg.lstsq(G, z.ravel(), rcond=None)
    z = z - (G @ c).reshape(z.shape)
    w = np.outer(np.hanning(ny), np.hanning(nx))
    P = np.abs(np.fft.fft2(z * w)) ** 2
    ky = np.fft.fftfreq(ny, dx)
    kx = np.fft.fftfreq(nx, dx)
    k = np.hypot(*np.meshgrid(ky, kx, indexing="ij"))
    lam = np.where(k > 0, 1.0 / np.where(k > 0, k, 1), np.inf)
    edges = np.logspace(np.log10(2.0), np.log10(200.0), 41)
    lc, lp = [], []
    for i in range(edges.size - 1):
        s = (lam >= edges[i]) & (lam < edges[i + 1])
        if s.sum() >= 10:
            lc.append(np.log10(np.sqrt(edges[i] * edges[i + 1])))
            lp.append(np.log10(np.mean(P[s])))
    x, y = np.array(lc), np.array(lp)
    best = None
    for xb in np.linspace(x[2], x[-3], 200):
        X = np.column_stack([np.ones_like(x), np.where(x <= xb, x - xb, 0), np.where(x > xb, x - xb, 0)])
        cc, *_ = np.linalg.lstsq(X, y, rcond=None)
        sse = float(np.sum((y - X @ cc) ** 2))
        if best is None or sse < best[0]:
            best = (sse, xb, cc)
    return {"kink_wavelength_m": float(10 ** best[1]), "slope_short": float(best[2][1]), "slope_long": float(best[2][2])}


def strahler_first_order(A, FD, A_t):
    """Order-1 network cells, their basin (outlet) id, and whether the outlet is a proper junction with a higher-order stream."""
    r_to, c_to, good = d8_receiver_arrays(FD)
    ny, nx = A.shape
    net = np.isfinite(A) & (A >= A_t)
    idx = np.flatnonzero(net)
    order_ = np.argsort(A.ravel()[idx], kind="stable")       # donors before receivers
    rec_flat = np.where(good, r_to * nx + c_to, -1).ravel()
    max_o = np.zeros(ny * nx, dtype=np.int32)
    n_max = np.zeros(ny * nx, dtype=np.int32)
    strahler = np.zeros(ny * nx, dtype=np.int32)
    for f in idx[order_]:
        o = 1 if max_o[f] == 0 else (max_o[f] + 1 if n_max[f] >= 2 else max_o[f])
        strahler[f] = o
        r = rec_flat[f]
        if r >= 0 and net.ravel()[r]:
            if o > max_o[r]:
                max_o[r], n_max[r] = o, 1
            elif o == max_o[r]:
                n_max[r] += 1
    S = strahler.reshape(ny, nx)
    first = S == 1
    # outlet of each first-order link: the order-1 cell whose receiver is a higher-order network cell
    basin = np.full(ny * nx, -1, dtype=np.int64)
    fidx = np.flatnonzero(first.ravel())
    proper = {}
    for f in fidx[np.argsort(-A.ravel()[fidx], kind="stable")]:      # receivers first
        r = rec_flat[f]
        if r >= 0 and first.ravel()[r]:
            basin[f] = basin[r]
        else:
            basin[f] = f
            proper[f] = bool(r >= 0 and S.ravel()[r] >= 2)
    return first, basin.reshape(ny, nx), proper


def irls_bisquare(x, y, c=4.685, iters=50):
    X = np.column_stack([np.ones_like(x), x])
    w = np.ones_like(y)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(iters):
        r = y - X @ beta
        s = np.median(np.abs(r - np.median(r))) / 0.6745
        if s <= 0:
            break
        u = r / (c * s)
        w = np.where(np.abs(u) < 1, (1 - u ** 2) ** 2, 0.0)
        Wx = X * w[:, None]
        beta_new = np.linalg.lstsq(Wx.T @ X, Wx.T @ y, rcond=None)[0]
        if np.allclose(beta_new, beta, rtol=1e-10, atol=1e-12):
            beta = beta_new
            break
        beta = beta_new
    return beta


def binned_fit(la, ls):
    if la.size < 10:
        return None
    edges = np.arange(np.floor(la.min() * 10) / 10, la.max() + 0.1, 0.1)
    idx = np.digitize(la, edges) - 1
    xs, ys = [], []
    for k in range(edges.size - 1):
        s = ls[idx == k]
        if s.size >= 5:
            xs.append(0.5 * (edges[k] + edges[k + 1]))
            ys.append(float(np.mean(s)))
    if len(xs) < 3:
        return None
    b0, b1 = irls_bisquare(np.array(xs), np.array(ys))
    return {"D_over_K": float(10 ** b0), "m": float(-b1), "n_bins": len(xs), "A_range_m2": [float(10 ** min(xs)), float(10 ** max(xs))]}


def collect(tiles, lap_h_fn, A_t_mode, mask_fn=None) -> dict:
    pts = []           # (log A, log S*, basin global id)
    info = {}
    for ti, t in enumerate(tiles):
        td = metre_tile(t)
        q = disk_quadratic(td.Z, 7.0, td.dx, td.filled.dy)
        kink = spectrum_kink(td.Z, td.dx)
        A_t = kink["kink_wavelength_m"] ** 2 if A_t_mode == "kink" else float(A_t_mode)
        first, basin, proper = strahler_first_order(td.A, td.FD, A_t)
        lh = lap_h_fn(t)
        d = q["lap"] - lh
        keep = first & np.isfinite(d) & (d > 0) & np.isfinite(q["grad"]) & (q["grad"] > 0)
        if mask_fn is not None:
            keep &= mask_fn(td)
        b = basin[keep]
        ok_b = np.array([proper.get(int(x), False) for x in b])
        la = np.log10(td.A[keep])[ok_b]
        ls = np.log10(q["grad"][keep] / d[keep])[ok_b]
        bb = b[ok_b] + ti * 10_000_000_000
        pts.append((la, ls, bb))
        info[t] = {"kink": kink, "A_t_m2": A_t, "n_first_order_basins": int(sum(proper.values())), "n_points": int(la.size), "lap_h": lh}
    la = np.concatenate([p[0] for p in pts])
    ls = np.concatenate([p[1] for p in pts])
    bb = np.concatenate([p[2] for p in pts])
    return {"la": la, "ls": ls, "bb": bb, "info": info}


def fit_all(C, select: bool, n_boot=500) -> dict:
    la, ls, bb = C["la"], C["ls"], C["bb"]
    if la.size == 0:
        return {"n_basins": 0, "n_points": 0}
    uniq, inv = np.unique(bb, return_inverse=True)
    sel = np.ones(uniq.size, dtype=bool)
    if select:
        for i in range(uniq.size):
            s = inv == i
            if s.sum() < 10 or np.ptp(la[s]) < 0.3:
                sel[i] = False
                continue
            sl, ic = np.polyfit(la[s], ls[s], 1)
            pred = ic + sl * la[s]
            r2 = 1 - np.sum((ls[s] - pred) ** 2) / np.sum((ls[s] - ls[s].mean()) ** 2)
            sel[i] = bool(sl < 0 and r2 >= 0.5)
    keep = sel[inv]
    pt = binned_fit(la[keep], ls[keep])
    if pt is None:
        return {"n_basins": int(sel.sum())}
    rng = np.random.default_rng(SEED)
    ids = np.flatnonzero(sel)
    pos = {i: np.flatnonzero(inv == i) for i in ids}
    bd, bm = [], []
    for _ in range(n_boot):
        draw = rng.choice(ids, ids.size)
        ii = np.concatenate([pos[i] for i in draw])
        f = binned_fit(la[ii], ls[ii])
        if f:
            bd.append(f["D_over_K"])
            bm.append(f["m"])
    pt.update({"n_basins": int(sel.sum()), "n_basins_all": int(uniq.size), "n_points": int(keep.sum()), "selection": "power-law basins (slope < 0, R2 >= 0.5)" if select else "all first-order basins",
               "D_over_K_ci68": [float(np.percentile(bd, 16)), float(np.percentile(bd, 84))], "m_ci68": [float(np.percentile(bm, 16)), float(np.percentile(bm, 84))]})
    return pt


def main() -> None:
    t0 = time.time()
    slurm_record(os.path.dirname(out_path("ttv_cal", "sstar_perron.json")))
    cal = load_json(out_path("ttv_cal", "ttv_cal.json"))
    out: dict = {"published_gm": PERRON_GM, "gm": {}, "ttv": {}}
    gm_crest = cal["T3"]["gm_check"]["our_crest_disk7_lap_h"]
    for mode in ("kink",) + FIXED_AT:
        for lh_tag, lh in (("perron_lap_h", PERRON_GM["lap_h"]), ("our_crest", gm_crest)):
            C = collect(GM_NCALM, lambda _t, lh=lh: lh, mode)
            for sel in (True, False):
                k = f"A_t={mode}:{lh_tag}:{'selected' if sel else 'all'}"
                out["gm"][k] = fit_all(C, sel) | {"tiles": C["info"]}
                r = out["gm"][k]
                print(f"GM {k}: D/K {r.get('D_over_K')} {r.get('D_over_K_ci68')} m {r.get('m')} {r.get('m_ci68')} basins {r.get('n_basins')}/{r.get('n_basins_all')} "
                      f"(Perron 124 +- 3, 0.35 +- 0.003) [{time.time() - t0:.0f} s]", flush=True)
    ttv_lh = cal["T3"]["lap_h_crest"]["disk7_median"]
    for mode in ("kink",) + FIXED_AT:
        C = collect(["TTV1", "TTV2", "TTV3"], lambda _t: ttv_lh, mode, mask_fn=md_polygon_mask)
        for sel in (True, False):
            k = f"A_t={mode}:crest:{'selected' if sel else 'all'}"
            out["ttv"][k] = fit_all(C, sel) | {"tiles": C["info"]}
            r = out["ttv"][k]
            print(f"TTV {k}: D/K {r.get('D_over_K')} {r.get('D_over_K_ci68')} m {r.get('m')} {r.get('m_ci68')} basins {r.get('n_basins')}/{r.get('n_basins_all')} [{time.time() - t0:.0f} s]", flush=True)
    out["wall_s"] = time.time() - t0
    dump(out, out_path("ttv_cal", "sstar_perron.json"))


if __name__ == "__main__":
    main()
