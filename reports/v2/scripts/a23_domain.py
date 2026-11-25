#!/usr/bin/env python3
"""Addendum 23, Part A item 1 (decision 2.1): one domain around the M&D study area, cut from the full-site 1 m grid.

  - window around the polygon, outlets (polygon exits with A >= 1e4 m2), the domain as the union of their upstream basins, the
    fixed outlets (receiver outside the domain) and the nested ones, the hillslope fringe, the box cut;
  - the polygon and the TIGER roads on the box lattice (distance grids), the picks, the tile footprints;
  - the mapped-head network and the observed 12 m criterion networks (area x slope^2 >= 876 m2; area >= 6,120 m2) on the domain;
  - the re-measured inputs: K0 = U / k_s (chi-z at m = 0.33 above 1e3 m2), the crest curvature (D_form) and the relief D on the
    mapped-head network, each beside the addendum 21/22 per-tile values;
  - the domain check: the TTV2 tile with its edge ring as the outlets through the domain route reproduces the addendum 22 fits.
Outputs <run>/domain/domain.npz, domain.json. usage: a23_domain.py [--no-check] [--margin-m 1500]
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a19_common import hilltop_set, segment_bootstrap_median, segments_of, tile_bounds  # noqa: E402
from a19_hilltops import md_polygon_mask  # noqa: E402
from a21_common import picks  # noqa: E402
from a23_common import (  # noqa: E402
    A_F,
    BLOCK_M,
    C_F,
    DOMAIN_CHECK_TOL,
    FULL_ORIGIN,
    HILLTOP_A_C,
    KS_A_C,
    N_BOOT_SEG,
    OUTLET_A_MIN_M2,
    ROAD_BUFFER_HILLTOP_M,
    SEED,
    TILES,
    TTV_FULL,
    W_F,
    WINDOW_MARGIN_M,
    M,
    U,
    a21_path,
    a22_path,
    channel_heads,
    criterion_sets,
    dump,
    fit_terms,
    full_grid_meta,
    json_ready,
    label_upstream,
    load_json,
    mask_from_entries,
    md_layer_mask,
    md_polygon_area_and_length,
    metre_tile,
    mse_at,
    out_path,
    read_window,
    receiver_rc,
    reduce_mask,
    slurm_record,
    terms_of,
    tiger_roads_in_box,
)

from lemcvt.analyses.er_star import trace_to_channel  # noqa: E402
from lemcvt.measure.hilltop_curvature import total_curvature  # noqa: E402
from lemcvt.measure.phase6_inputs import ks_at_theta  # noqa: E402


def polygon_extent() -> tuple[float, float, float, float]:
    from a23_common import MD_GPKG
    from osgeo import ogr
    ogr.UseExceptions()
    src = ogr.Open(MD_GPKG)                      # keep the datasource alive while the layer is used
    lyr = src.GetLayerByName("study_area")
    xmin, xmax, ymin, ymax = lyr.GetExtent()
    del lyr, src
    return xmin, xmax, ymin, ymax


def unpack_bits(bits: np.ndarray, shape) -> np.ndarray:
    return np.unpackbits(bits)[: shape[0] * shape[1]].reshape(shape).astype(bool)


def build_domain(margin_m: float, t0: float) -> dict[str, Any]:
    meta = full_grid_meta()
    x0f, y1f = FULL_ORIGIN
    assert abs(meta["transform"][2] - x0f) < 1e-6 and abs(meta["transform"][5] - y1f) < 1e-6, meta["transform"]
    nyf, nxf = meta["shape"]
    xmin, xmax, ymin, ymax = polygon_extent()
    c0 = max(int(math.floor(xmin - x0f - margin_m)), 0)
    c1 = min(int(math.ceil(xmax - x0f + margin_m)), nxf)
    r0 = max(int(math.floor(y1f - ymax - margin_m)), 0)
    r1 = min(int(math.ceil(y1f - ymin + margin_m)), nyf)
    Zw = read_window(TTV_FULL["filled"], r0, r1, c0, c1).astype(np.float64)
    nod = meta["nodata"]
    if nod is not None:
        Zw[Zw == nod] = np.nan
    Aw = read_window(TTV_FULL["area_cells"], r0, r1, c0, c1).astype(np.float64)       # 1 m cells: cells = m2
    FDw = read_window(TTV_FULL["fd"], r0, r1, c0, c1).astype(np.uint8)
    valid_w = np.isfinite(Zw) & (Aw > 0)
    Aw[~valid_w] = np.nan
    x0w, y1w = x0f + c0, y1f - r0
    print(f"window rows {r0}-{r1} cols {c0}-{c1} ({r1 - r0} x {c1 - c0}); valid {valid_w.sum():,} [{time.time() - t0:.0f} s]", flush=True)
    poly_w = md_layer_mask("study_area", Zw.shape, x0w, y1w, False)
    nr, nc, good = receiver_rc(FDw)
    nr_c, nc_c = np.clip(nr, 0, Zw.shape[0] - 1), np.clip(nc, 0, Zw.shape[1] - 1)
    recv_in_poly = good & poly_w[nr_c, nc_c]
    exits = poly_w & valid_w & (Aw >= OUTLET_A_MIN_M2) & ~recv_in_poly
    exits_all_sizes = poly_w & valid_w & ~recv_in_poly
    seeds = np.flatnonzero(exits.ravel())
    lab = label_upstream(FDw, seeds)
    domain_w = lab >= 0
    nested = exits & good & domain_w[nr_c, nc_c]
    fixed = exits & ~nested
    edge_touch = bool(domain_w[0, :].any() or domain_w[-1, :].any() or domain_w[:, 0].any() or domain_w[:, -1].any())
    if edge_touch:
        raise RuntimeError("the domain touches the window edge; enlarge --margin-m")
    # every domain cell's receiver is a domain cell unless the cell is a fixed outlet
    leak = domain_w & good & ~domain_w[nr_c, nc_c] & ~fixed
    assert not leak.any(), int(leak.sum())
    no_recv_inside = domain_w & ~good & ~fixed
    rows, cols = np.nonzero(domain_w)
    br0, br1, bc0, bc1 = int(rows.min()) - 1, int(rows.max()) + 2, int(cols.min()) - 1, int(cols.max()) + 2
    box = (slice(br0, br1), slice(bc0, bc1))
    Z = np.where(domain_w, Zw, np.nan)[box]
    A = np.where(domain_w, Aw, np.nan)[box]
    FD = np.where(domain_w, FDw, 0).astype(np.uint8)[box]
    FD[fixed[box]] = 0
    domain = domain_w[box]
    poly = poly_w[box]
    outlets = fixed[box]
    x0, y1 = x0w + bc0, y1w - br0
    md = md_polygon_area_and_length(domain, x0, y1)
    rec: dict[str, Any] = {
        "window": {"full_rows": [r0, r1], "full_cols": [c0, c1], "margin_m": margin_m, "x0": x0w, "y1": y1w},
        "box": {"full_rows": [r0 + br0, r0 + br1], "full_cols": [c0 + bc0, c0 + bc1], "x0": x0, "y1": y1, "shape": list(Z.shape), "epsg": 6339},
        "polygon": {"n_cells": int(poly_w.sum()), "area_m2_raster": float(poly_w.sum()), "area_m2_vector": md["polygon_area_m2"], "n_valid_cells": int((poly_w & valid_w).sum()),
                    "md_traced_length_in_polygon_m": md["md_traced_length_in_polygon_m"]},
        "outlets": {"A_min_m2": OUTLET_A_MIN_M2, "n_exits_at_or_above_A_min": int(exits.sum()), "n_fixed": int(fixed.sum()), "n_nested": int(nested.sum()),
                    "n_polygon_exit_cells_all_sizes": int(exits_all_sizes.sum()), "n_fixed_without_receiver": int((fixed & ~good).sum()),
                    "largest_outlet_area_m2": float(np.nanmax(Aw[fixed])) if fixed.any() else None, "outlet_areas_m2": sorted(Aw[fixed].tolist(), reverse=True)},
        "domain": {"n_cells": int(domain_w.sum()), "area_km2": float(domain_w.sum()) / 1e6, "n_cells_in_polygon": int((domain_w & poly_w).sum()),
                   "share_of_polygon_covered": float((domain_w & poly_w).sum() / (poly_w & valid_w).sum()), "share_of_domain_outside_polygon": float((domain_w & ~poly_w).sum() / domain_w.sum()),
                   "hillslope_fringe_cells": int((poly_w & valid_w & ~domain_w).sum()), "hillslope_fringe_share_of_polygon": float((poly_w & valid_w & ~domain_w).sum() / (poly_w & valid_w).sum()),
                   "n_cells_without_receiver_not_fixed": int(no_recv_inside.sum()), "largest_area_cell_is_fixed": bool(outlets.ravel()[int(np.nanargmax(np.where(np.isfinite(A), A, -1)))])},
    }
    print(f"domain: {rec['domain']['n_cells']:,} cells ({rec['domain']['area_km2']:.3f} km2), {rec['outlets']['n_fixed']} fixed outlets ({rec['outlets']['n_nested']} nested), covers "
          f"{rec['domain']['share_of_polygon_covered']:.3f} of the polygon; fringe {rec['domain']['hillslope_fringe_share_of_polygon']:.3f}; {rec['domain']['share_of_domain_outside_polygon']:.3f} of the domain "
          f"outside the polygon; box {Z.shape} [{time.time() - t0:.0f} s]", flush=True)
    return {"Z": Z, "A": A, "FD": FD, "domain": domain, "poly": poly, "outlets": outlets, "x0": x0, "y1": y1, "rec": rec, "full_box_r0": r0 + br0, "full_box_c0": c0 + bc0}


def roads_and_picks(D: dict[str, Any], t0: float) -> None:
    from scipy.ndimage import distance_transform_edt
    ny, nx = D["Z"].shape
    rd = tiger_roads_in_box(D["x0"], D["y1"], nx, ny)
    D["road_cells"] = rd["cells"]
    D["road_dist"] = distance_transform_edt(~rd["cells"]).astype(np.float32) if rd["cells"].any() else np.full((ny, nx), np.inf, dtype=np.float32)
    D["rec"]["roads"] = {k: rd[k] for k in ("features", "n_features", "mtfcc", "source")}
    D["rec"]["roads"]["n_line_cells_in_box"] = int(rd["cells"].sum())
    D["rec"]["roads"]["n_line_cells_in_domain"] = int((rd["cells"] & D["domain"]).sum())
    D["rec"]["roads"]["trail_classes_present"] = sorted(k for k in rd["mtfcc"] if k in ("S1500", "S1710", "S1820", "S1830"))
    pk = picks()
    xs, ys = np.array([q["x"] for q in pk]), np.array([q["y"] for q in pk])
    cols = np.floor(xs - D["x0"]).astype(int)
    rows = np.floor(D["y1"] - ys).astype(int)
    on = (rows >= 0) & (rows < ny) & (cols >= 0) & (cols < nx)
    rr, cc = np.clip(rows, 0, ny - 1), np.clip(cols, 0, nx - 1)
    in_dom = on & D["domain"][rr, cc]
    in_poly = on & D["poly"][rr, cc]
    tile_of = np.array(["" for _ in pk], dtype="U8")
    D["rec"]["tile_boxes"] = {}
    for t in TILES:
        b = tile_bounds(t)
        tr0, tc0 = int(round(D["y1"] - b["top"])), int(round(b["left"] - D["x0"]))
        D["rec"]["tile_boxes"][t] = {"r0": tr0, "r1": tr0 + int(b["height"]), "c0": tc0, "c1": tc0 + int(b["width"]), "left": b["left"], "top": b["top"]}
        inside = (xs >= b["left"]) & (xs < b["right"]) & (ys >= b["bottom"]) & (ys < b["top"])
        tile_of[inside] = t
    D["pick_id"] = np.array([q["pick_id"] for q in pk], dtype=np.int32)
    D["pick_row"], D["pick_col"] = rows.astype(np.int32), cols.astype(np.int32)
    D["pick_in_domain"], D["pick_in_poly"], D["pick_tile"] = in_dom, in_poly, tile_of
    D["rec"]["picks"] = {"n": len(pk), "n_on_box": int(on.sum()), "n_in_domain": int(in_dom.sum()), "n_in_polygon": int(in_poly.sum()),
                         "n_in_domain_and_polygon": int((in_dom & in_poly).sum()), "per_tile": {t: int(((tile_of == t) & in_dom).sum()) for t in TILES},
                         "per_tile_in_polygon": {t: int(((tile_of == t) & in_poly).sum()) for t in TILES}, "outside_domain_ids": D["pick_id"][~in_dom].tolist()}
    print(f"roads: {rd['n_features']} TIGER features, classes {rd['mtfcc']}, {D['rec']['roads']['n_line_cells_in_domain']:,} line cells in the domain; picks in domain "
          f"{in_dom.sum()} (per tile {D['rec']['picks']['per_tile']}) [{time.time() - t0:.0f} s]", flush=True)


def networks(D: dict[str, Any], t0: float) -> None:
    """The mapped-head network (a21 pick seeds) and the 12 m criterion networks from the a21 full-site statistics, on the domain."""
    Z, A, FD = D["Z"], D["A"], D["FD"]
    ny, nx = Z.shape
    fr0, fc0 = D["full_box_r0"], D["full_box_c0"]
    sd = np.load(a21_path("grid12", "TTV2_seeds.npz"))
    pr, pc = sd["pick_seed_r"].astype(int) - fr0, sd["pick_seed_c"].astype(int) - fc0
    on = (pr >= 0) & (pr < ny) & (pc >= 0) & (pc < nx)
    on &= D["domain"][np.clip(pr, 0, ny - 1), np.clip(pc, 0, nx - 1)]
    D["net_mapped"] = mask_from_entries(pr[on], pc[on], A, FD)
    D["rec"]["networks"] = {"mapped": {"n_pick_seeds_in_domain": int(on.sum()), "n_channel_1m": int(D["net_mapped"].sum()), "seed_ids": sd["pick_id"][on].tolist()}}
    g12 = np.load(a21_path("grid12", "grid12_arrays.npz"))
    for key, alpha_key, C in (("net_C", "a2", C_F), ("net_A", "a0", A_F)):
        _ch, hd = criterion_sets(g12[f"M_{alpha_key}"], g12[f"Mdon_{alpha_key}"], C)
        R, Cc = np.nonzero(hd)
        rows_s, cols_s, n_blocks_in = [], [], 0
        for rb, cb in zip(R.tolist(), Cc.tolist(), strict=True):
            fr, fc = rb * BLOCK_M, cb * BLOCK_M                       # full-grid 1 m rows/cols of the block
            r_lo, r_hi = fr - fr0, fr - fr0 + BLOCK_M
            c_lo, c_hi = fc - fc0, fc - fc0 + BLOCK_M
            if r_hi <= 0 or c_hi <= 0 or r_lo >= ny or c_lo >= nx:
                continue
            r_lo, r_hi, c_lo, c_hi = max(r_lo, 0), min(r_hi, ny), max(c_lo, 0), min(c_hi, nx)
            sub = A[r_lo:r_hi, c_lo:c_hi]
            if not np.isfinite(sub).any():
                continue
            n_blocks_in += 1
            k = int(np.nanargmax(sub))
            rows_s.append(r_lo + k // sub.shape[1])
            cols_s.append(c_lo + k % sub.shape[1])
        rows_a, cols_a = np.array(rows_s, dtype=int), np.array(cols_s, dtype=int)
        D[key] = mask_from_entries(rows_a, cols_a, A, FD)
        D["rec"]["networks"][key] = {"threshold": C, "alpha": 2.0 if alpha_key == "a2" else 0.0, "n_12m_head_cells_touching_domain": n_blocks_in, "n_channel_1m": int(D[key].sum()),
                                     "n_heads_1m": int(channel_heads(D[key], FD).sum()), "n_12m_heads_full_site": int(hd.sum())}
    print("networks: " + "; ".join(f"{k}: {v['n_channel_1m']:,} cells" for k, v in D["rec"]["networks"].items()) + f" [{time.time() - t0:.0f} s]", flush=True)


def measured_inputs(D: dict[str, Any], t0: float) -> None:
    Z, A, FD, poly = D["Z"], D["A"], D["FD"], D["poly"]
    # ---- K0 = U / k_s: basins labelled by their fixed outlet
    labels = label_upstream(FD, np.flatnonzero(D["outlets"].ravel()))
    out: dict[str, Any] = {"U": U, "m": M}
    for a_c in (KS_A_C, 1e4):
        r = ks_at_theta(Z, A, FD, 1.0, 1.0, M, a_c, labels, n_boot=200, seed=SEED)
        r.pop("_moments", None)
        out[f"ks_{a_c:g}"] = r | {"K0": (U / r["ks"]) if r.get("ks") else None}
    out["K0"] = out[f"ks_{KS_A_C:g}"]["K0"]
    out["ks"] = out[f"ks_{KS_A_C:g}"]["ks"]
    print(f"k_s {out['ks']:.3f} (n_basins {out[f'ks_{KS_A_C:g}']['n_basins']}) K0 {out['K0']:.4g} [{time.time() - t0:.0f} s]", flush=True)
    # ---- crest curvature on the 1e3 network (addendum 19/20 set without the segment filter), D_form
    road_mask = D["road_dist"] <= ROAD_BUFFER_HILLTOP_M
    C_Lf = -total_curvature(Z, W_F, 1.0, 1.0)                                      # positive = convex
    hs = hilltop_set(Z, A, FD, 1.0, 1.0, HILLTOP_A_C, w_f=W_F, road_mask=road_mask, segment_min_px=50, C_at_Lf=C_Lf)
    counted = hs["masks"]["roadfree"] & np.isfinite(C_Lf) & (C_Lf > 1e-9)       # the addendum 20 D0_N set (numerical zeros dropped)
    seg = segments_of(counted)
    cb = segment_bootstrap_median(C_Lf[counted], seg[counted], N_BOOT_SEG, seed=SEED)
    out["crest"] = {"set": {k: v for k, v in hs.items() if k.startswith("n_")}, "n_counted": int(counted.sum()), "n_counted_in_polygon": int((counted & poly).sum()),
                    "C_ht_Lf": cb, "L_f_m": 2 * W_F + 1, "D_form": U / cb["median"] if cb["median"] > 0 else None,
                    "D_form_ci68": [U / cb["ci68"][1], U / cb["ci68"][0]] if cb.get("ci68") else None,
                    "physical_D_check": {"D_phys": 0.0025, "model_crest_curvature_at_D_phys": U / 0.0025, "measured_over_model_at_D_phys": cb["median"] / (U / 0.0025)}}
    cbp = segment_bootstrap_median(C_Lf[counted & poly], seg[counted & poly], N_BOOT_SEG, seed=SEED)
    out["crest"]["C_ht_Lf_in_polygon"] = cbp
    out["crest"]["D_form_in_polygon"] = U / cbp["median"] if cbp.get("median") and cbp["median"] > 0 else None
    print(f"crest: {int(counted.sum())} pixels, C_ht {cb['median']:.4g} -> D_form {out['crest']['D_form']:.4g} [{time.time() - t0:.0f} s]", flush=True)
    # ---- relief D: crest pixels inside the polygon traced to the mapped-head network
    rows, cols = np.nonzero(counted & poly)
    trc = trace_to_channel(Z, D["net_mapped"].astype(float), FD, 1.0, 1.0, rows, cols, 0.5)
    used = trc["reached"] & (trc["L_H"] > 0) & (trc["R"] > 0)
    Dr = np.where(used, U * trc["L_H"] ** 2 / (2 * np.where(trc["R"] > 0, trc["R"], np.nan)), np.nan)
    rb = segment_bootstrap_median(Dr, seg[rows, cols], N_BOOT_SEG, seed=SEED)
    out["relief"] = {"network": "mapped", "n_pixels": int(rows.size), "n_reached": int(trc["reached"].sum()), "n_used": int(used.sum()), "n_left": int(trc["left_tile"].sum()),
                     "L_H_median_m": float(np.median(trc["L_H"][used])) if used.any() else None, "R_median_m": float(np.median(trc["R"][used])) if used.any() else None,
                     "D_relief": rb}
    out["D_relief"] = rb["median"]
    for key in ("net_C", "net_A"):
        tr2 = trace_to_channel(Z, D[key].astype(float), FD, 1.0, 1.0, rows, cols, 0.5)
        u2 = tr2["reached"] & (tr2["L_H"] > 0) & (tr2["R"] > 0)
        D2 = np.where(u2, U * tr2["L_H"] ** 2 / (2 * np.where(tr2["R"] > 0, tr2["R"], np.nan)), np.nan)
        out["relief"][f"sensitivity_{key}"] = segment_bootstrap_median(D2, seg[rows, cols], 0) | {"n_used": int(u2.sum())}
    # ---- the addendum 21/22 per-tile values beside them
    per = {}
    for t in TILES:
        a21 = load_json(a21_path("tiles", f"{t}.json"))
        per[t] = {"K0": a21["K0"], "ks": U / a21["K0"], "D_form": a21["D"]["D_form"], "C_crest": a21["D"]["C_crest"], "D_relief": a21["D"]["D_relief"],
                  "D_relief_ci68": a21["D"].get("D_relief_ci68"), "L_H_median_m": a21["D"]["relief"]["mapped"].get("L_H_median_m")}
    out["per_tile_a21"] = per
    D["rec"]["measured"] = out
    D["crest_counted"] = counted
    print(f"relief D {rb['median']:.4g} (n {rb['n']}, L_H {out['relief']['L_H_median_m']:.0f} m); tiles D_relief {[round(per[t]['D_relief'], 4) for t in TILES]} K0 {[per[t]['K0'] for t in TILES]} [{time.time() - t0:.0f} s]", flush=True)


def domain_check(t0: float) -> dict[str, Any]:
    """TTV2 cut as before; its edge ring as the outlets through the domain route (noflux_outside on) reproduces the addendum 22 fits."""
    from a23_common import domain_ad_model

    from lemcvt.models.common import make_boundary_mask
    from lemcvt.optimize.v2 import ADProfiler, PartitionedProfiler
    td = metre_tile("TTV2")
    Z, A, FD = td.Z, td.A, td.FD
    a22 = load_json(a22_path("tiles", "TTV2.json"))
    K0 = float(a22["K0"])
    Kb, Db = a22["bounds_rescaled"]["K"], a22["bounds_rescaled"]["D"]
    lo, hi = np.array([U / Kb[1], U / Db[1]]), np.array([U / Kb[0], U / Db[0]])
    ring = make_boundary_mask(np.isfinite(A))
    prof = PartitionedProfiler(Z, A, FD, 1.0, U, m_values=[M], outlets=ring, noflux_outside=True)
    poly = md_polygon_mask(td)
    keep = poly[prof.gd["rev_j"], prof.gd["rev_i"]]
    hz = np.load(a21_path("tiles", "TTV2_heads.npz"))
    out: dict[str, Any] = {"tile": "TTV2", "tolerance": DOMAIN_CHECK_TOL, "checks": {}}
    for net, name in (("A", "PF_A"), ("C", "PF_C")):
        mask = unpack_bits(hz[f"net_{net}_1m"], Z.shape)
        red = reduce_mask(prof, mask, M)
        T = terms_of(red, prof.Z_obs_flat, keep)
        f = fit_terms(T, U, K0, lo, hi)["free_KD"]
        st = a22["models"][name]
        chk = {"a22": {"mse_poly": st["mse_poly"], "K": st["K"], "D": st["D"]}, "domain_route": {"mse_poly": f["mse"], "K": f["K"], "D": f["D"]},
               "rel_mse": abs(f["mse"] / st["mse_poly"] - 1), "rel_K": abs(f["K"] / st["K"] - 1), "rel_D": abs(f["D"] / st["D"] - 1)}
        chk["pass"] = bool(chk["rel_mse"] <= DOMAIN_CHECK_TOL and chk["rel_K"] <= 1e-6 and chk["rel_D"] <= 1e-6)
        out["checks"][name] = chk
        if net == "C":
            st2 = a22["models"]["F2_relief"]
            m2 = mse_at(T, U, K0, st2["D"])
            out["checks"]["F2_relief"] = {"a22_mse_poly": st2["mse_poly"], "domain_route": m2, "rel_mse": abs(m2 / st2["mse_poly"] - 1), "pass": bool(abs(m2 / st2["mse_poly"] - 1) <= DOMAIN_CHECK_TOL)}
    ad22 = load_json(a22_path("profiles", "AD_TTV2.json"))["optimum_poly"]
    adm = domain_ad_model({"Z": Z, "A": A, "FD": FD, "outlets": ring})
    gd = prof.gd
    obs_poly = np.where(poly[gd["rev_j"], gd["rev_i"]], Z[gd["rev_j"], gd["rev_i"]].astype(float), np.nan)
    pp = ADProfiler(adm, obs_poly, U)
    red = pp.reduce(ad22["K"] / ad22["D"])
    m_ad = pp.mse_at(red, U / ad22["D"])
    out["checks"]["AD_at_a22_K_D"] = {"a22_mse_poly": ad22["mse"], "domain_route": m_ad, "rel_mse": abs(m_ad / ad22["mse"] - 1), "pass": bool(abs(m_ad / ad22["mse"] - 1) <= DOMAIN_CHECK_TOL)}
    out["pass"] = all(c["pass"] for c in out["checks"].values())
    print("domain check TTV2: " + ", ".join(f"{k} {'PASS' if v['pass'] else 'FAIL'} ({v['rel_mse']:.1e})" for k, v in out["checks"].items()) + f" [{time.time() - t0:.0f} s]", flush=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--no-check", action="store_true")
    p.add_argument("--margin-m", type=float, default=WINDOW_MARGIN_M)
    a = p.parse_args()
    t0 = time.time()
    slurm_record(os.path.dirname(out_path("domain", "domain.json")))
    D = build_domain(a.margin_m, t0)
    roads_and_picks(D, t0)
    networks(D, t0)
    measured_inputs(D, t0)
    if not a.no_check:
        D["rec"]["domain_check"] = domain_check(t0)
    # the TTV2 config bounds x 65/77 (addenda 21-22) for the fits
    a22 = load_json(a22_path("tiles", "TTV2.json"))
    D["rec"]["bounds_rescaled"] = a22["bounds_rescaled"]
    D["rec"]["bounds_source"] = "TTV2 config K/D bounds x 65/77 (addendum 22 tiles/TTV2.json bounds_rescaled)"
    arrays = {"Z": D["Z"].astype(np.float32), "A": D["A"].astype(np.float32), "FD": D["FD"], "domain_bits": np.packbits(D["domain"]), "poly_bits": np.packbits(D["poly"]),
              "outlets_bits": np.packbits(D["outlets"]), "road_dist": D["road_dist"], "road_cells_bits": np.packbits(D["road_cells"]), "crest_counted_bits": np.packbits(D["crest_counted"]),
              "net_C_bits": np.packbits(D["net_C"]), "net_A_bits": np.packbits(D["net_A"]), "net_mapped_bits": np.packbits(D["net_mapped"]),
              "pick_id": D["pick_id"], "pick_row": D["pick_row"], "pick_col": D["pick_col"], "pick_in_domain": D["pick_in_domain"], "pick_in_poly": D["pick_in_poly"], "pick_tile": D["pick_tile"]}
    np.savez_compressed(out_path("domain", "domain.npz"), **arrays)
    D["rec"]["wall_s"] = time.time() - t0
    dump(json_ready(D["rec"]), out_path("domain", "domain.json"))
    print(f"domain: done [{time.time() - t0:.0f} s]", flush=True)


if __name__ == "__main__":
    main()
