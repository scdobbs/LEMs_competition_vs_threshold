#!/usr/bin/env python3
"""Addendum 19, Part A per tile (sections 2.1, 2.6, 2.7; decisions 1, 11, 12, 13): hilltop curvature on the fixed A >= 10^3 m^2 network.

Per tile: the decision-1 hilltop set with every filter stage kept (geometric filters as Part B on the 10^3 network, the road/trail buffer,
the 50-px segment filter, the C_ht(L_f) >= 0 drop); C_ht(L) on the counted set at every window of the Part B test (7-51 m) and of the
decision-11 scan (3-31 m, odd) with the 2,000-draw segment bootstrap (seed 20261120); D0_N_A = E/C_ht(L_f); L_w as written (crest width to
the 10^3 network) and L_w' (crest width to the rollover network, the Part B meaning), D0_N_B at each, both window tests; D0_N_unf (no
segment or road filter) and D0_N without the convex drop; the rollover (Part B) set rebuilt for the same-pixel diagnostic and the QC
comparison; sigma_C(L) over all cells inside the margin (decision 11); D_R = E L_H^2 / (2 R) per counted pixel traced on D8 to the 10^3
network, to the picked-head network inside the M&D polygon (TTV1-TTV3), and to the addendum 18 P4 channel set (model-network D_R);
E* = 2 C_ht L_H / S_c at S_c = 1.27 (decision 13); the legacy (manuscript, approximate) block-mean route on GM and AP (decision 10).
Outputs <run>/hilltops/<tile>.json, <tile>_pixels.parquet, <tile>_masks.npz.
usage: a19_hilltops.py --tile TILE [--n-boot 2000]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a19_common import (  # noqa: E402
    A18_SUMMARY,
    A_NET_M2,
    ALL_HALF_WIDTHS,
    E_SITE,
    E_STAR_BANDS,
    MD_GPKG,
    MD_TILES,
    N_BOOT,
    SCAN_HALF_WIDTHS,
    TEST_HALF_WIDTHS,
    W_F,
    WINDOW_TOL,
    crest_width_stats,
    dump,
    fnum,
    hilltop_set,
    largest_window_not_above,
    load_json,
    metre_tile,
    out_path,
    partb_measurement,
    partb_per_tile,
    ratio_interval,
    road_mask_for_tile,
    segment_bootstrap_median,
    segments_of,
    site_of,
    slurm_record,
    trace_to_mask,
)

from lemcvt.analyses.er_star import S_C_REFERENCE  # noqa: E402
from lemcvt.measure.hilltop_curvature import total_curvature  # noqa: E402
from lemcvt.measure.phase6_inputs import option_iii_mask  # noqa: E402


def curve_on(C_grids: dict[int, np.ndarray], mask: np.ndarray, seg: np.ndarray, dx: float, n_boot: int, windows) -> dict[str, dict]:
    out = {}
    s = seg[mask]
    for w in windows:
        v = C_grids[w][mask]
        rec = segment_bootstrap_median(v, s, n_boot)
        rec["L_m"] = float((2 * w + 1) * dx)
        rec["L_nominal_m"] = int(2 * w + 1)
        rec["frac_convex"] = float(np.mean(v[np.isfinite(v)] > 0)) if np.isfinite(v).any() else None
        out[str(w)] = rec
    return out


def window_test(curve: dict[str, dict], width_m: float, dx: float) -> dict:
    ref = curve[str(W_F)]["median"]
    tested = [w for w in TEST_HALF_WIDTHS if w >= W_F and (2 * w + 1) <= width_m + 1e-9]
    ratios = {str(w): curve[str(w)]["median"] / ref for w in tested}
    worst = max((abs(r - 1) for r in ratios.values()), default=0.0)
    return {"median_crest_width_m": width_m, "windows_tested_nominal_m": [2 * w + 1 for w in tested], "ratios_to_L_f": ratios, "max_abs_deviation": worst,
            "pass": bool(worst <= WINDOW_TOL), "only_L_f_in_range": bool(len(tested) <= 1), "tol": WINDOW_TOL}


def d_r_block(E: float, C: np.ndarray, tr: dict, seg: np.ndarray, n_boot: int, label: str) -> dict:
    L, R, eu = tr["L_H"], tr["R"], tr["L_euclid"]
    used = tr["reached"] & (L > 0) & (R > 0) & np.isfinite(C) & (C > 0)
    rec = {"network": label, "n_pixels": int(C.size), "n_reached": int(tr["reached"].sum()), "n_left_tile": int(tr["left_tile"].sum()),
           "n_R_nonpositive": int(np.sum(tr["reached"] & ~(R > 0))), "n_used": int(used.sum())}
    if not used.any():
        return rec
    D_R = np.where(used, E * L ** 2 / (2 * np.where(R > 0, R, np.nan)), np.nan)
    D_R_eu = np.where(used, E * eu ** 2 / (2 * np.where(R > 0, R, np.nan)), np.nan)
    D_C = np.where(used, E / C, np.nan)
    S = np.where(used, R / np.where(L > 0, L, np.nan), np.nan)
    ratio_norm = np.where(used, 2 * S / (C * L), np.nan)          # 1 on a linear hillslope (addendum 8)
    E_star = np.where(used, 2 * C * L / S_C_REFERENCE, np.nan)
    rec.update({"L_H_median_m": float(np.median(L[used])), "L_H_q16_q84_m": [float(np.percentile(L[used], 16)), float(np.percentile(L[used], 84))],
                "L_euclid_median_m": float(np.median(eu[used])), "R_median_m": float(np.median(R[used])), "Sbar_median": float(np.median(S[used])),
                "D_R": segment_bootstrap_median(D_R, seg, n_boot), "D_R_euclid": segment_bootstrap_median(D_R_eu, seg, n_boot),
                "D_C_median_on_used": float(np.median(D_C[used])), "D_R_over_D_C_median": float(np.median((D_R / D_C)[used])),
                "ratio_normalised": segment_bootstrap_median(ratio_norm, seg, n_boot), "E_star_at_1.27": segment_bootstrap_median(E_star, seg, 0),
                "E_star_bands": {"lt_10": float(np.mean(E_star[used] < E_STAR_BANDS[0])), "10_30": float(np.mean((E_star[used] >= E_STAR_BANDS[0]) & (E_star[used] <= E_STAR_BANDS[1]))),
                                 "gt_30": float(np.mean(E_star[used] > E_STAR_BANDS[1]))}})
    rec["per_pixel"] = {"D_R": D_R, "D_R_euclid": D_R_eu, "used": used}
    return rec


def md_polygon_mask(td) -> np.ndarray:
    from osgeo import gdal, ogr
    gdal.UseExceptions()
    ogr.UseExceptions()
    ny, nx = td.Z.shape
    tr = td.filled.transform
    drv = gdal.GetDriverByName("MEM")
    ds = drv.Create("", nx, ny, 1, gdal.GDT_Byte)
    ds.SetGeoTransform((tr.c, tr.a, tr.b, tr.f, tr.d, tr.e))
    ds.SetProjection(td.filled.crs.to_wkt())
    src = ogr.Open(MD_GPKG)
    gdal.RasterizeLayer(ds, [1], src.GetLayerByName("study_area"), burn_values=[1], options=["ALL_TOUCHED=FALSE"])
    return ds.GetRasterBand(1).ReadAsArray().astype(bool)


def picked_network_on_tile(td) -> np.ndarray | None:
    p = out_path("heads", "picked_network.npz")
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tile", required=True)
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    a = p.parse_args()
    tile = a.tile
    t0 = time.time()
    out_dir = os.path.dirname(out_path("hilltops", f"{tile}.json"))
    slurm_record(out_dir)
    site = site_of(tile)
    E = E_SITE[site]
    td = metre_tile(tile)
    Z, A, FD, dx, dy = td.Z, td.A, td.FD, td.dx, td.filled.dy
    meas = partb_measurement()
    per = partb_per_tile()[tile]
    A_c_roll = float(meas["tiles"][tile]["measure_run"]["A_c_rollover"])
    rec: dict = {"tile": tile, "site": site, "E": E, "dx": dx, "dy": dy, "shape": list(Z.shape), "inputs": td.provenance(),
                 "part_b": {"A_c_rollover": A_c_roll, "C_ht_Lf": fnum(per["C_ht_Lf"]), "D0_A": fnum(per["D0_A"]), "L_w_m": fnum(per["L_w_m"]), "D0_B": fnum(per["D0_B"]),
                            "n_hilltop_px": int(float(per["n_hilltop_px"])), "median_crest_width_m": fnum(per["median_crest_width_m"]), "window_pass": per["window_pass"]}}
    # curvature grids (positive = convex), every window of the test and of the scan
    C = {w: -total_curvature(Z, w, dx, dy) for w in ALL_HALF_WIDTHS}
    print(f"{tile}: curvature at {len(C)} windows [{time.time() - t0:.0f} s]", flush=True)
    # decision 11: sigma_C(L) on all cells inside the margin (the curvature grid is NaN within w of the edge)
    rec["sigma_C"] = {str(w): {"L_m": float((2 * w + 1) * dx), "sigma": float(np.nanstd(-C[w])), "n": int(np.isfinite(C[w]).sum()), "median_abs": float(np.nanmedian(np.abs(C[w])))} for w in SCAN_HALF_WIDTHS}
    # roads
    road_mask, road_info = road_mask_for_tile(td)
    rec["roads"] = road_info
    # the decision-1 set
    hs = hilltop_set(Z, A, FD, dx, dy, A_NET_M2, W_F, road_mask=road_mask, C_at_Lf=C[W_F])
    masks = hs["masks"]
    dist_net = hs["dist"]
    rec["set"] = {k: v for k, v in hs.items() if k not in ("masks", "dist", "channel", "net")}
    counted = masks["counted"]
    seg_grid = segments_of(counted)
    seg = seg_grid[counted]
    rec["set"]["n_segments_counted"] = int(np.unique(seg).size)
    print(f"{tile}: counted {counted.sum()} px in {rec['set']['n_segments_counted']} segments (geom {masks['geom'].sum()}, road -{hs['n_removed_road']}, segment -{hs['n_removed_segment']}, nonconvex -{hs['n_removed_nonconvex']}) [{time.time() - t0:.0f} s]", flush=True)
    # crest widths: to the 10^3 network (as written) and to the rollover network (Part B meaning)
    channel_roll = np.isfinite(A) & (A >= A_c_roll)
    from scipy.ndimage import distance_transform_edt
    dist_roll = distance_transform_edt(~channel_roll, sampling=(dy, dx))
    width_net = crest_width_stats(dist_net, counted)
    width_roll = crest_width_stats(dist_roll, counted)
    rec["crest_width_to_1e3_m"] = width_net
    rec["crest_width_to_rollover_m"] = width_roll
    # C_ht(L) on the counted set, with the bootstrap at every window
    curve = curve_on(C, counted, seg_grid, dx, a.n_boot, ALL_HALF_WIDTHS)
    rec["curve_counted"] = curve
    n_counted = int(counted.sum())
    rec["D0_N"] = {"n_counted": n_counted, "usable": bool(n_counted >= 500)}
    if n_counted > 0:
        cf = curve[str(W_F)]
        rec["D0_N"]["A"] = {"C_ht": cf["median"], "C_ht_ci68": cf.get("ci68"), "C_ht_ci95": cf.get("ci95")} | ratio_interval(E, cf)
        for tag, width in (("B", width_net["50"] if width_net else 0.0), ("B_prime_rollover_width", width_roll["50"] if width_roll else 0.0)):
            w_w = largest_window_not_above(width, TEST_HALF_WIDTHS, dx, W_F)
            cw = curve[str(w_w)]
            rec["D0_N"][tag] = {"L_w_m": int(2 * w_w + 1), "w": w_w, "crest_width_m": width, "C_ht": cw["median"], "C_ht_ci68": cw.get("ci68")} | ratio_interval(E, cw)
            rec["D0_N"][f"window_test_{tag}"] = window_test(curve, width, dx)
        # the 11 -> L_w curve as D0_N(L) for the gallery (2.6.3): D0 at every window
        rec["D0_N"]["D0_by_window"] = {k: ratio_interval(E, v) | {"L_nominal_m": v["L_nominal_m"]} for k, v in curve.items()}
    # the labelled sensitivities: unf (no segment/road filter) and segfilt without the convex drop
    for name, m in (("unf", masks["unf"]), ("segfilt_no_convex_drop", masks["segfilt"]), ("geom_no_filters_no_convex_drop", masks["geom"]), ("parent50", masks["parent50"])):
        sg = segments_of(m)
        cv = curve_on(C, m, sg, dx, a.n_boot, (W_F,) + tuple(w for w in {rec["D0_N"].get("B", {}).get("w"), rec["D0_N"].get("B_prime_rollover_width", {}).get("w")} if w and w != W_F))
        rec[name] = {"n": int(m.sum()), "n_segments": int(np.unique(sg[m]).size), "curve": cv, "D0_A": ratio_interval(E, cv[str(W_F)])}
        for tag in ("B", "B_prime_rollover_width"):
            w_w = rec["D0_N"].get(tag, {}).get("w")
            if w_w is not None:
                rec[name][f"D0_{tag}"] = ratio_interval(E, cv[str(w_w)]) | {"L_w_m": int(2 * w_w + 1)}
    # the Part B rollover set rebuilt (same-pixel diagnostic and QC comparison)
    roll_mask, _m0, _d, roll_diag = option_iii_mask(Z, A, FD, dx, dy, A_c_roll, W_F)
    v = C[W_F][roll_mask]
    rec["rollover_set"] = {"n": int(roll_mask.sum()), "C_ht_Lf_recomputed": float(np.nanmedian(v)) if v.size else None, "part_b_C_ht_Lf": fnum(per["C_ht_Lf"]),
                           "overlap_with_counted_px": int((roll_mask & counted).sum()), "diag": {k: v_ for k, v_ in roll_diag.items() if not hasattr(v_, "shape")}}
    rec["rollover_set"]["check_rel"] = abs(rec["rollover_set"]["C_ht_Lf_recomputed"] / fnum(per["C_ht_Lf"]) - 1) if rec["rollover_set"]["C_ht_Lf_recomputed"] and fnum(per["C_ht_Lf"]) else None
    print(f"{tile}: C_ht(L_f) counted {rec['D0_N'].get('A', {}).get('C_ht')} (rollover set recomputed {rec['rollover_set']['C_ht_Lf_recomputed']} vs Part B {fnum(per['C_ht_Lf'])}) [{time.time() - t0:.0f} s]", flush=True)
    # D_R traces (decision 12) and E* (decision 13)
    rows, cols = np.nonzero(counted)
    Cf = C[W_F][rows, cols]
    pix = pd.DataFrame({"tile": tile, "row": rows, "col": cols, "segment": seg, "C_Lf": Cf, "dist_1e3_m": dist_net[rows, cols], "dist_rollover_m": dist_roll[rows, cols],
                        "road_buffer": (road_mask[rows, cols] if road_mask is not None else False)})
    for w in ALL_HALF_WIDTHS:
        pix[f"C_w{w}"] = C[w][rows, cols]
    nets: dict[str, np.ndarray] = {"net_1e3": hs["channel"]}
    md_mask = None
    if tile in MD_TILES:
        md_mask = md_polygon_mask(td)
        pk = picked_network_on_tile(td)
        rec["md_polygon_px"] = int(md_mask.sum())
        rec["counted_inside_md_polygon"] = int((counted & md_mask).sum())
        if pk is not None:
            nets["picked_heads"] = pk
            rec["picked_network_px_on_tile"] = int(pk.sum())
        else:
            rec["picked_network_px_on_tile"] = None
        pix["in_md_polygon"] = md_mask[rows, cols]
    # model network: the addendum 18 P4 channel set at the site m (primary group; off-mesa, ncalm, pamap, ap1_3dep groups as registered)
    fits = load_json(os.path.join(A18_SUMMARY, "per_tile_fits.json"))
    spec_key = next((k for k in (f"{tile}:primary", f"{tile}:offmesa", f"{tile}:ncalm", f"{tile}:pamap", f"{tile}:ap1_3dep") if k in fits), None)
    if spec_key and fits[spec_key].get("P4"):
        A_crit = float(fits[spec_key]["P4"]["A_crit"])
        nets["model_P4"] = np.isfinite(A) & (A >= A_crit)
        rec["model_P4"] = {"spec": spec_key, "A_crit": A_crit, "D": fits[spec_key]["P4"]["D"]}
    rec["D_R"] = {}
    for name, target in nets.items():
        tr = trace_to_mask(Z, FD, dx, dy, rows, cols, target)
        if name == "picked_heads":
            inside = md_mask[rows, cols]
            tr = {k: (np.where(inside, v, False) if v.dtype == bool else np.where(inside, v, np.nan)) for k, v in tr.items()}
            tr["reached"] = tr["reached"].astype(bool)
            tr["left_tile"] = tr["left_tile"].astype(bool)
        blk = d_r_block(E, Cf, tr, seg, a.n_boot, name)
        pp = blk.pop("per_pixel", None)
        rec["D_R"][name] = blk
        pix[f"L_H_{name}"] = tr["L_H"]
        pix[f"R_{name}"] = tr["R"]
        pix[f"L_eu_{name}"] = tr["L_euclid"]
        pix[f"reached_{name}"] = tr["reached"]
        pix[f"left_{name}"] = tr["left_tile"]
        if pp is not None:
            pix[f"D_R_{name}"] = pp["D_R"]
            pix[f"used_{name}"] = pp["used"]
        print(f"{tile}: D_R on {name}: used {blk.get('n_used')} of {blk['n_pixels']} (left {blk['n_left_tile']}); L_H median {blk.get('L_H_median_m')}; D_R median {blk.get('D_R', {}).get('median')} [{time.time() - t0:.0f} s]", flush=True)
    # decision 10: the manuscript's route as the package holds it (approximate, as labelled there)
    if site in ("GM", "AP"):
        from lemcvt.measure.legacy_v1 import legacy_curvature
        try:
            rec["legacy_v1_curvature_approx"] = legacy_curvature(Z, dx, site, E)
        except Exception as exc:  # noqa: BLE001
            rec["legacy_v1_curvature_approx"] = {"error": repr(exc)}
    # write
    pix.to_parquet(out_path("hilltops", f"{tile}_pixels.parquet"), index=False)
    np.savez_compressed(out_path("hilltops", f"{tile}_masks.npz"), counted=counted, unf=masks["unf"], geom=masks["geom"], segfilt=masks["segfilt"], roadfree=masks["roadfree"], parent50=masks["parent50"],
                        divide_between_basins=masks["divide_between_basins"], rollover_set=roll_mask, road_mask=(road_mask if road_mask is not None else np.zeros_like(counted)),
                        channel_1e3=hs["channel"], channel_rollover=channel_roll, md_polygon=(md_mask if md_mask is not None else np.zeros_like(counted)),
                        picked_network=(nets.get("picked_heads", np.zeros_like(counted))), transform=np.array(list(td.filled.transform)[:6]))
    rec["wall_s"] = time.time() - t0
    dump(rec, out_path("hilltops", f"{tile}.json"))
    print(f"{tile}: done, D0_N_A {rec['D0_N'].get('A', {}).get('D0')} (Part B D0_A {fnum(per['D0_A'])}) [{rec['wall_s']:.0f} s]", flush=True)


if __name__ == "__main__":
    main()
