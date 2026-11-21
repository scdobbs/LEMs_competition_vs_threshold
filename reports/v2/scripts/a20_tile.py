#!/usr/bin/env python3
"""Addendum 20, Part A per tile (sections 2.2-2.6, 3.1-3.2; decisions 1-4): D0_P (Perron's estimator) and the revised D0_N on one tile.

Grid: the tile's registered *filled* 1 m elevation grid in metres (`_metre_tile`, the grid `total_curvature` used in addendum 19); area:
the registered D8 area grid in m^2 (cells x cell area, as loaded).

D0_N (decision 1): the addendum 19 geometric set on the 10^3 network (divides between outlet basins, edge margin, |grad z| <= 0.4, farther
than L_f/2 from the network, the addendum 19 masks reused), then the 10 m TIGER road/trail buffer, then C_ht(L_f) <= 0 dropped; no segment
filter. The 20-px fragment filter beside it. C_ht(L) at L = 3 ... 51 m on the same pixels (decision 4), median with the segment bootstrap
(2,000, seed 20261120). Labelled sensitivity (not in the directive): the same median with the 100 m block bootstrap, because the segments
are now 2-4 px fragments and the segment bootstrap then approaches a pixel bootstrap.

D0_P (decision 2): `a20_perron` at r = 3, 5, 7, 10, 15 m; bins of A|grad z|; the plateau rule; fixed ranges <= 1 and <= 3 m^2; the 100 m
block bootstrap with the rule re-applied in every draw; per-block tables written for the site pooling. At r = 7 m also the 11 m square
curvature tabulated on the same bins (four-way comparison), and D-infinity area in place of D8 on the GM and AP primary tiles (sensitivity).

Four-way (decision 2, last item): divide-line pixels with the 11 m square and with the 7 m disk; plateau cells (this tile's r = 7 m plateau
bins) with the 7 m disk and with the 11 m square; each with both statistics (median over the set, and the geometric mean over its convex
cells), so that pixel selection, window shape and statistic are separated.

D_R (decision 3): per D0_N pixel, L_H and R along D8 to the 10^3 network, at TTV1-TTV3 also to the snapped-head network (inside the M&D
polygon), the Euclidean-length sensitivity, the addendum 18 P4 network (model-network D_R); E* at S_c = 1.27 with C_ht(L_f).
Outputs <run>/tiles/<tile>.json, <tile>_tables.npz, <tile>_divide_pixels.parquet, <tile>_plateau_cells.parquet; figure
reports/v2/figures/a20/plateau/<tile>_plateau.png.
usage: a20_tile.py --tile TILE
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a19_hilltops import d_r_block, md_polygon_mask  # noqa: E402
from a20_common import (  # noqa: E402
    A18_SUMMARY,
    AP_PRIMARY,
    D0_MIN_PIXELS,
    D0N_ALL_HALF_WIDTHS,
    E_SITE,
    FIG_DIR,
    FRAGMENT_SENS_PX,
    GM_PRIMARY,
    LAP_ZERO,
    MD_TILES,
    N_BOOT,
    PERRON_R_PRIMARY,
    PERRON_RADII_M,
    SEED,
    W_F,
    a19_path,
    dump,
    hillshade,
    load_json,
    metre_tile,
    out_path,
    ratio_interval,
    segment_bootstrap_median,
    segments_of,
    site_of,
    slurm_record,
    trace_to_mask,
)
from a20_perron import BIN_EDGES, N_BINS, bin_index, block_index, block_tables, d0_from_lap, disk_quadratic, estimate  # noqa: E402

from lemcvt.measure.hilltop_curvature import total_curvature  # noqa: E402

PLATEAU_CELL_X_MAX = 10.0       # m^2: per-cell rows kept for the site four-way (the pooled plateau must end below; checked at site level)


def block_bootstrap_median(vals: np.ndarray, blocks: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    ok = np.isfinite(vals)
    v, b = vals[ok], blocks[ok]
    if v.size == 0:
        return {"median": None, "n": 0}
    order = np.argsort(v, kind="stable")
    vs = v[order]
    uniq, inv = np.unique(b[order], return_inverse=True)
    rng = np.random.default_rng(seed)
    bs = np.empty(n_boot)
    for i in range(n_boot):
        w = np.bincount(rng.integers(0, uniq.size, uniq.size), minlength=uniq.size).astype(float)[inv]
        cw = np.cumsum(w)
        bs[i] = vs[np.searchsorted(cw, 0.5 * cw[-1])]
    return {"median": float(np.median(v)), "n": int(v.size), "n_blocks": int(uniq.size), "ci68": [float(np.percentile(bs, 16)), float(np.percentile(bs, 84))]}


def geo_mean_convex(lap: np.ndarray) -> dict:
    """Geometric mean of |lap| over the convex cells (lap < 0), and the median of -lap over all cells of the set."""
    v = lap[np.isfinite(lap)]
    c = v[v < -LAP_ZERO]
    return {"n": int(v.size), "n_convex": int(c.size), "frac_convex": float(c.size / v.size) if v.size else None,
            "geomean_abs": float(10 ** np.mean(np.log10(-c))) if c.size else None, "median_neg": float(np.median(-v)) if v.size else None}


def snapped_network_on_tile(td) -> np.ndarray | None:
    p = out_path("heads", "snapped_network.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p)
    t_full = z["transform"]
    x = t_full[2] + (z["cols"].astype(float) + 0.5) * t_full[0]
    y = t_full[5] + (z["rows"].astype(float) + 0.5) * t_full[4]
    tr = td.filled.transform
    c = np.floor((x - tr.c) / tr.a).astype(int)
    r = np.floor((y - tr.f) / tr.e).astype(int)
    ny, nx = td.Z.shape
    ok = (r >= 0) & (r < ny) & (c >= 0) & (c < nx)
    m = np.zeros((ny, nx), dtype=bool)
    m[r[ok], c[ok]] = True
    return m


def plateau_figure(tile, Z, dx, lap7, inc_mask, counted, rec, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from scipy.ndimage import binary_dilation
    hs = hillshade(Z, dx)
    fig, ax = plt.subplots(figsize=(10.5, 10))
    ax.imshow(np.nan_to_num(hs, nan=1.0), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    v = np.where(inc_mask, lap7, np.nan)
    lim = float(np.nanpercentile(np.abs(v), 95)) if np.isfinite(v).any() else 0.02
    im = ax.imshow(v, cmap="RdBu", norm=TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim), interpolation="nearest", alpha=0.9)
    ring = binary_dilation(counted, iterations=2) & ~binary_dilation(counted, iterations=1)
    rr, cc = np.nonzero(ring)
    ax.plot(cc, rr, ",", color="k", alpha=0.9)
    p = rec["perron"]["r7"]
    txt = (f"plateau bins {p['point']['range_m2'][0]:.3g}-{p['point']['range_m2'][1]:.3g} m² ({p['point']['n_bins']} bins, {p['point']['n_cells']:,} cells)"
           if p["point"].get("plateau") else f"no plateau ({p['point'].get('reason')})")
    d = rec["D0_P"]["r7"]
    ax.set_title(f"{tile}: plateau cells at r = 7 m coloured by ∇²z (m⁻¹; blue convex); black outline = D₀,N pixels ({int(counted.sum()):,})\n{txt}; "
                 f"D₀,P {d['D0_P'] if d['D0_P'] is None else round(d['D0_P'], 5)}, D₀,N {rec['D0_N']['A'].get('D0') if rec['D0_N'].get('A') else None}", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, shrink=0.6, label="∇²z, 7 m disk (m⁻¹)")
    for sc in (1.0, 0.8, 0.6):
        fig.savefig(path, dpi=int(150 * sc), bbox_inches="tight", facecolor="white")
        if os.path.getsize(path) <= 5_000_000:
            break
    plt.close(fig)
    return {"path": os.path.relpath(path, os.path.dirname(os.path.dirname(FIG_DIR))), "bytes": os.path.getsize(path)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tile", required=True)
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    p.add_argument("--no-dinf", action="store_true")
    a = p.parse_args()
    tile = a.tile
    t0 = time.time()
    slurm_record(os.path.dirname(out_path("tiles", f"{tile}.json")))
    site = site_of(tile)
    E = E_SITE[site]
    td = metre_tile(tile)
    Z, A, FD, dx, dy = td.Z, td.A, td.FD, td.dx, td.filled.dy
    rec: dict = {"tile": tile, "site": site, "E": E, "dx": dx, "dy": dy, "shape": list(Z.shape), "inputs": td.provenance(),
                 "grid": "registered filled grid (td.Z via _metre_tile), metres; area = registered D8 area grid in m^2", "n_nodata": int((~np.isfinite(Z)).sum())}
    m19 = np.load(a19_path("hilltops", f"{tile}_masks.npz"))
    blocks, n_blocks = block_index(Z.shape, dx, dy)
    # ------------------------------------------------------------------ D0_N (decision 1)
    C = {w: -total_curvature(Z, w, dx, dy) for w in D0N_ALL_HALF_WIDTHS}         # positive = convex
    convex = np.isfinite(C[W_F]) & (C[W_F] > LAP_ZERO)
    rec_numzero = int((np.isfinite(C[W_F]) & (C[W_F] > 0) & (C[W_F] <= LAP_ZERO) & m19["roadfree"]).sum())
    roadfree = m19["roadfree"]
    counted = roadfree & convex
    seg_grid = segments_of(counted)
    frag = segments_of(roadfree)
    sizes = np.bincount(frag.ravel())
    small = sizes < FRAGMENT_SENS_PX
    small[0] = False
    counted20 = roadfree & ~small[frag] & convex
    n_c = int(counted.sum())
    rec["numerical_zero_divide_px_dropped"] = rec_numzero
    rec["set"] = {"n_geom": int(m19["geom"].sum()), "n_removed_road": int(m19["geom"].sum() - roadfree.sum()), "n_removed_nonconvex": int(roadfree.sum() - n_c), "n_counted": n_c,
                  "n_segments": int(seg_grid.max()), "segment_size_median_px": float(np.median(np.bincount(seg_grid.ravel())[1:])) if n_c else None,
                  "n_counted_fragment20": int(counted20.sum()), "a19_n_counted_segment50": int(m19["counted"].sum()), "same_as_a19_fragment0": None}
    a19 = load_json(a19_path("hilltops", f"{tile}.json"))
    rec["set"]["same_as_a19_fragment0"] = bool(a19["set"]["segment_filter_sensitivity"]["fragment_0"]["n"] == n_c)
    seg = seg_grid[counted]
    curve = {}
    for w in D0N_ALL_HALF_WIDTHS:
        r_ = segment_bootstrap_median(C[w][counted], seg, a.n_boot)
        r_["L_nominal_m"] = 2 * w + 1
        curve[str(w)] = r_ | {"D0": ratio_interval(E, r_)}
    rec["D0_N"] = {"n_counted": n_c, "usable": bool(n_c >= D0_MIN_PIXELS), "curve": curve}
    if n_c:
        cf = curve[str(W_F)]
        rec["D0_N"]["A"] = {"C_ht": cf["median"], "C_ht_ci68": cf.get("ci68")} | ratio_interval(E, cf)
        bb = block_bootstrap_median(C[W_F][counted], blocks[counted], a.n_boot)
        rec["D0_N"]["A_block_bootstrap"] = bb | ratio_interval(E, bb)
    if counted20.any():
        s20 = segments_of(counted20)[counted20]
        r20 = segment_bootstrap_median(C[W_F][counted20], s20, a.n_boot)
        rec["D0_N_fragment20"] = {"n": int(counted20.sum()), "usable": bool(counted20.sum() >= D0_MIN_PIXELS), "C_ht": r20["median"], "C_ht_ci68": r20.get("ci68")} | ratio_interval(E, r20)
    print(f"{tile}: D0_N counted {n_c} (a19 segment-50 set {rec['set']['a19_n_counted_segment50']}, fragment-20 {int(counted20.sum())}); D0_N_A {rec['D0_N'].get('A', {}).get('D0')} [{time.time() - t0:.0f} s]", flush=True)
    # ------------------------------------------------------------------ D0_P (decision 2)
    rec["perron"], rec["D0_P"] = {}, {}
    tabs_out: dict[str, np.ndarray] = {"blocks_shape": np.array(Z.shape), "n_blocks": np.array(n_blocks)}
    q7 = None
    for r in PERRON_RADII_M:
        q = disk_quadratic(Z, r, dx, dy)
        X = A * q["grad"]
        extra = {"sq11": -C[W_F]} if r == PERRON_R_PRIMARY else None
        tab = block_tables(q["lap"], X, blocks, n_blocks, extra)
        est = estimate(tab, extra=("sq11",) if extra else ())
        key = f"r{r:g}"
        rec["perron"][key] = est | {"n_valid": int(q["valid"].sum()), "n_disk": q["n_disk"]}
        rec["D0_P"][key] = d0_from_lap(E, est)
        for k, v in tab.items():
            tabs_out[f"{key}__{k}"] = v
        if r == PERRON_R_PRIMARY:
            q7 = q | {"X": X}
        print(f"{tile}: r {r:g} m plateau {est['point'].get('plateau')} bins {est['point'].get('range_m2')} lap_h {est['point'].get('lap_h')} D0_P {rec['D0_P'][key]['D0_P']} "
              f"(boot share {est['boot']['share_with_plateau']:.2f}) [{time.time() - t0:.0f} s]", flush=True)
    lap7, X7 = q7["lap"], q7["X"]
    b7 = np.where(np.isfinite(X7), bin_index(np.nan_to_num(X7, nan=-1.0)), -1)
    b7 = np.where(np.isfinite(lap7), b7, -1)
    pt7 = rec["perron"]["r7"]["point"]
    inc_mask = np.isin(b7, pt7["bins"]) if pt7.get("plateau") else np.zeros(Z.shape, dtype=bool)
    # D-infinity area (sensitivity) on the GM and AP primary tiles
    if not a.no_dinf and tile in (GM_PRIMARY + AP_PRIMARY):
        from lemcvt.hydro.dinf import compute_dinf_area, compute_dinf_flow_direction
        t1 = time.perf_counter()
        ang, rf, fi = compute_dinf_flow_direction(Z, dx, FD_d8=FD)
        Ad = compute_dinf_area(Z.astype(float), dx, ang, rf, fi)
        tab = block_tables(lap7, Ad * q7["grad"], blocks, n_blocks)
        est = estimate(tab)
        rec["perron"]["r7_dinf"] = est | {"wall_s": time.perf_counter() - t1}
        rec["D0_P"]["r7_dinf"] = d0_from_lap(E, est)
        for k, v in tab.items():
            tabs_out[f"r7_dinf__{k}"] = v
        del ang, rf, fi, Ad
        print(f"{tile}: D-infinity r 7 plateau {est['point'].get('range_m2')} D0_P {rec['D0_P']['r7_dinf']['D0_P']} [{time.perf_counter() - t1:.0f} s]", flush=True)
    # ------------------------------------------------------------------ four-way (decision 2, last item), this tile
    sq = -C[W_F]                                              # 11 m square curvature, negative = convex
    fw = {"divide_square11": geo_mean_convex(sq[counted]), "divide_disk7": geo_mean_convex(lap7[counted]),
          "plateau_disk7": geo_mean_convex(lap7[inc_mask]) if inc_mask.any() else None, "plateau_square11": geo_mean_convex(sq[inc_mask]) if inc_mask.any() else None,
          "n_divide_pixels_without_disk7": int((counted & ~np.isfinite(lap7)).sum()), "overlap_divide_plateau_px": int((counted & inc_mask).sum())}
    rec["four_way"] = fw
    # ------------------------------------------------------------------ D_R and E* (decision 3)
    rows, cols = np.nonzero(counted)
    Cf = C[W_F][rows, cols]
    pix = pd.DataFrame({"tile": tile, "row": rows, "col": cols, "segment": seg, "block": blocks[rows, cols], "C_Lf": Cf, "lap7": lap7[rows, cols], "X7": X7[rows, cols],
                        "in_fragment20": counted20[rows, cols]})
    for w in D0N_ALL_HALF_WIDTHS:
        pix[f"C_w{w}"] = C[w][rows, cols]
    nets: dict[str, np.ndarray] = {"net_1e3": m19["channel_1e3"]}
    md_mask = None
    if tile in MD_TILES:
        md_mask = md_polygon_mask(td)
        sn = snapped_network_on_tile(td)
        if sn is not None:
            nets["snapped_heads"] = sn
            rec["snapped_network_px_on_tile"] = int(sn.sum())
        nets["picked_heads_a19_A3"] = m19["picked_network"]
        pix["in_md_polygon"] = md_mask[rows, cols]
        rec["counted_inside_md_polygon"] = int((counted & md_mask).sum())
    fits = load_json(os.path.join(A18_SUMMARY, "per_tile_fits.json"))
    spec_key = next((k for k in (f"{tile}:primary", f"{tile}:offmesa", f"{tile}:ncalm", f"{tile}:pamap", f"{tile}:ap1_3dep") if k in fits), None)
    if spec_key and fits[spec_key].get("P4"):
        A_crit = float(fits[spec_key]["P4"]["A_crit"])
        nets["model_P4"] = np.isfinite(A) & (A >= A_crit)
        rec["model_P4"] = {"spec": spec_key, "A_crit": A_crit, "D": fits[spec_key]["P4"]["D"]}
    rec["D_R"] = {}
    for name, target in nets.items():
        tr = trace_to_mask(Z, FD, dx, dy, rows, cols, target)
        if name in ("snapped_heads", "picked_heads_a19_A3"):
            inside = md_mask[rows, cols]
            tr = {k: (np.where(inside, v, False) if v.dtype == bool else np.where(inside, v, np.nan)) for k, v in tr.items()}
            tr["reached"] = tr["reached"].astype(bool)
            tr["left_tile"] = tr["left_tile"].astype(bool)
            blk = d_r_block(E, Cf, tr, seg, a.n_boot, name)
            blk["n_inside_md_polygon"] = int(inside.sum())
        else:
            blk = d_r_block(E, Cf, tr, seg, a.n_boot, name)
        pp = blk.pop("per_pixel", None)
        if pp is not None:
            blk["D_R_block_bootstrap"] = block_bootstrap_median(np.where(pp["used"], pp["D_R"], np.nan), blocks[rows, cols], a.n_boot)
            pix[f"D_R_{name}"] = pp["D_R"]
            pix[f"D_R_euclid_{name}"] = pp["D_R_euclid"]
            pix[f"used_{name}"] = pp["used"]
        pix[f"L_H_{name}"] = tr["L_H"]
        pix[f"R_{name}"] = tr["R"]
        pix[f"reached_{name}"] = tr["reached"]
        pix[f"left_{name}"] = tr["left_tile"]
        rec["D_R"][name] = blk
        print(f"{tile}: D_R on {name}: used {blk.get('n_used')} of {blk['n_pixels']} (left {blk['n_left_tile']}); D_R median {blk.get('D_R', {}).get('median')} [{time.time() - t0:.0f} s]", flush=True)
    # ------------------------------------------------------------------ write
    pix.to_parquet(out_path("tiles", f"{tile}_divide_pixels.parquet"), index=False)
    keep = np.isfinite(lap7) & np.isfinite(X7) & (X7 < PLATEAU_CELL_X_MAX)
    kr, kc = np.nonzero(keep)
    pd.DataFrame({"row": kr.astype(np.int32), "col": kc.astype(np.int32), "block": blocks[kr, kc].astype(np.int32), "bin7": b7[kr, kc].astype(np.int16),
                  "lap7": lap7[kr, kc].astype(np.float32), "sq11": sq[kr, kc].astype(np.float32), "X7": X7[kr, kc].astype(np.float32)}).to_parquet(out_path("tiles", f"{tile}_plateau_cells.parquet"), index=False)
    rec["plateau_cells_rows"] = int(keep.sum())
    rec["plateau_cells_x_max_m2"] = PLATEAU_CELL_X_MAX
    rec["tile_plateau_hi_m2"] = pt7.get("range_m2", [None, None])[1] if pt7.get("plateau") else None
    np.savez_compressed(out_path("tiles", f"{tile}_tables.npz"), **tabs_out)
    np.savez_compressed(out_path("tiles", f"{tile}_masks.npz"), counted=counted, counted20=counted20, plateau_r7=inc_mask)
    os.makedirs(os.path.join(FIG_DIR, "plateau"), exist_ok=True)
    rec["figure"] = plateau_figure(tile, Z, dx, lap7, inc_mask, counted, rec, os.path.join(FIG_DIR, "plateau", f"{tile}_plateau.png"))
    rec["bin_edges_m2"] = BIN_EDGES.tolist()
    rec["n_bins"] = N_BINS
    rec["wall_s"] = time.time() - t0
    dump(rec, out_path("tiles", f"{tile}.json"))
    print(f"{tile}: done; D0_P(r7) {rec['D0_P']['r7']['D0_P']} D0_N_A {rec['D0_N'].get('A', {}).get('D0')} [{rec['wall_s']:.0f} s]", flush=True)


if __name__ == "__main__":
    main()
