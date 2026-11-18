#!/usr/bin/env python3
"""Addendum 14 stage A: machine-generated appendix tables (markdown) from the run directory.

usage: make_a14_tables.py --run DIR [--thresholds JSON] [--jobs ID,ID,...] > appendix.md
Every table names the file it was built from. Missing products are reported as "pending".
"""
import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a14_common import LEGACY_REF, PERRON_S2  # noqa: E402

SITES = ("GM", "AP", "TTV")


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def fmt(v, d=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if isinstance(v, float):
        if abs(v) >= 1e5:
            return f"{v:,.0f}"
        if abs(v) >= 100:
            return f"{v:.1f}"
        if abs(v) < 0.01 and v != 0:
            return f"{v:.3e}"
        return f"{v:.{d}f}"
    return str(v)


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for r in rows:
        out.append("| " + " | ".join(fmt(x) for x in r) + " |")
    return "\n".join(out) + "\n"


def pct(df, col, qs=(0.05, 0.25, 0.5, 0.75, 0.95)):
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        return [0] + ["-"] * (len(qs) + 2)
    return [int(s.size), float(s.min())] + [float(s.quantile(q)) for q in qs] + [float(s.max())]


def screen_masks(df, T, site):
    m = {}
    m["trunk: A_max_inside <= %.0e m2" % T["trunk_A_max_inside_m2_max"]] = df["A_max_inside_m2"] <= T["trunk_A_max_inside_m2_max"]
    m["trunk: largest single edge inflow <= %.0e m2" % T["trunk_A_inflow_max_single_m2_max"]] = df["A_inflow_max_single_m2"] <= T["trunk_A_inflow_max_single_m2_max"]
    m["transient: chi-z R2 (area-weighted) >= %.2f" % T["transient_chi_r2_area_weighted_min"]] = pd.to_numeric(df["chi_r2_area_weighted"], errors="coerce") >= T["transient_chi_r2_area_weighted_min"]
    m["transient: slope-area dBIC (area-weighted) <= %.0f" % T["transient_sa_dbic_area_weighted_max"]] = ~(pd.to_numeric(df["sa_dbic_area_weighted"], errors="coerce") > T["transient_sa_dbic_area_weighted_max"])
    m["disturbance: NLCD disturbed fraction <= %.2f" % T["disturbance_nlcd_disturbed_frac_max"]] = ~(pd.to_numeric(df["nlcd_disturbed_frac"], errors="coerce") > T["disturbance_nlcd_disturbed_frac_max"])
    if site == "AP":
        m["disturbance (AP): mining fraction <= %.2f" % T["disturbance_mining_frac_max"]] = ~(pd.to_numeric(df["mining_frac"], errors="coerce") > T["disturbance_mining_frac_max"])
    if T["upstream_fully_valid_required"].get(site):
        m["upstream area fully in valid data (decision 7)"] = df["upstream_fully_valid"].astype(str).str.lower() == "true"
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--thresholds", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "a14_thresholds.json"))
    p.add_argument("--jobs", default="")
    a = p.parse_args()
    R = a.run
    T = json.load(open(a.thresholds))
    out = []
    W = out.append

    fp = load(os.path.join(R, "footprints", "footprints.json"))
    ti = load(os.path.join(R, "ttv_inventory", "ttv_inventory.json"))
    pools = {s: load(os.path.join(R, "pools", f"{s}_pools.json")) for s in SITES}
    wins = {s: pd.read_csv(os.path.join(R, "pools", f"{s}_windows.csv")) if os.path.exists(os.path.join(R, "pools", f"{s}_windows.csv")) else None for s in SITES}
    conf = {s: load(os.path.join(R, "pools", f"{s}_conflicts.json")) for s in SITES}
    cmp_files = glob.glob(os.path.join(R, "usgs_vs_pasda", "usgs_vs_pasda_*.json"))
    cmp = load(cmp_files[0]) if cmp_files else None
    marin = load(os.path.join(R, "marin_select", "marin_select.json"))
    grids = {s: (load(os.path.join(R, f"grid_{s}", f"{s}_grid.json")), load(os.path.join(R, f"grid_{s}", f"{s}_route.json")), load(os.path.join(R, f"grid_{s}", f"{s}_checks.json"))) for s in ("AP", "TTV")}

    # ---------------------------------------------------------------- A1 counts
    W("### A1. Windows per lattice offset (section 3.1), from `pools/<site>_pools.json`\n")
    rows = []
    for s in SITES:
        pj = pools[s]
        if not pj:
            rows.append([s, "pending"] + [""] * 6)
            continue
        for k, c in pj["counts"].items():
            extra = c.get("n_in_3dep_index_and_coarse_valid", c.get("n_also_in_ncalm_2006_raster", "-"))
            extra2 = c.get("n_in_pamap_watershed", "-")
            rows.append([s, k, f'{pj["offsets_cells"][int(k[-1])]} cells ({pj["offsets_cells"][int(k[-1])][0] * pj["dx"]:.0f}, {pj["offsets_cells"][int(k[-1])][1] * pj["dx"]:.0f} m)',
                         c["n_windows"], c["n_coarse_valid_only"], c["n_candidates"], extra, extra2])
    W(table(["site", "offset", "offset (cells, m)", "lattice windows", "coarse grid all valid", "candidates (footprint 100 %)", "AP: in 3DEP index / TTV: also in NCALM raster", "AP: inside PAMAP watershed"], rows))
    rows = []
    for s in SITES:
        pj = pools[s]
        if pj:
            rows.append([s, pj["coarse"].split("/")[-1], f'{pj["shape"][0]} x {pj["shape"][1]}', f'{pj["dx"]:.4f}', pj["n_valid"], pj["W_cells"], f'{pj["W_m"]:.0f}', pj["footprint"], pj["A_c"], pj.get("routing_seconds")])
    W(table(["site", "coarse grid", "shape", "cell (m)", "valid cells", "W (cells)", "W (m)", "footprint mask", "A_c (m2)", "routing s"], rows))

    # ---------------------------------------------------------------- A2 screen distributions
    W("\n### A2. Screen statistics of the candidate windows (section 3.2), percentiles over all four offsets, from `pools/<site>_windows.csv`\n")
    cols = [("A_max_inside_m2", "largest drainage area inside (m2)"), ("A_inflow_edge_m2", "area entering across the edge, total (m2)"),
            ("A_inflow_max_single_m2", "largest single edge inflow (m2)"), ("edge_touched_frac", "fraction of cells with upstream area touching the footprint edge"),
            ("n_basins", "edge-outlet basins"), ("chi_r2_area_weighted", "chi-z R2 at theta_ref, area-weighted"), ("chi_r2_worst_big_basin", "chi-z R2, worst basin >= 10 %"),
            ("sa_dbic_area_weighted", "slope-area dBIC (one - two segments), area-weighted"), ("sa_dbic_worst_big_basin", "dBIC, worst basin >= 10 %"),
            ("frac_draining_cells_break_preferred", "fraction of draining cells in basins where a break is preferred"),
            ("nlcd_disturbed_frac", "NLCD 2021 disturbed fraction (11, 21-24, 31)"), ("nlcd_developed_frac", "NLCD developed (21-24)"), ("nlcd_water_frac", "NLCD open water (11)"),
            ("mining_frac", "AP: PA DEP mining fraction (longwall, mined-out, surface permits)"), ("longwall_panels_frac", "AP: longwall panels"), ("mined_out_underground_frac", "AP: mined-out underground"),
            ("surface_mining_permits_historic_frac", "AP: historic surface-mining permits"), ("conflict_degree_direct", "conflict degree (direct drains-through edges)"),
            ("conflict_degree_closure", "conflict degree (transitive closure)"), ("relief_m", "relief (m)"), ("mean_slope", "mean gradient")]
    for s in SITES:
        df = wins[s]
        if df is None:
            W(f"{s}: pending\n")
            continue
        rows = []
        for c, name in cols:
            if c in df.columns:
                rows.append([name] + pct(df, c))
        W(f"**{s}** ({len(df)} candidate windows)\n\n" + table(["statistic", "n", "min", "p05", "p25", "median", "p75", "p95", "max"], rows))

    # ---------------------------------------------------------------- A3 thresholds
    W("\n### A3. Proposed thresholds and the windows each removes (section 3.2), from `scripts/a14_thresholds.json`\n")
    W("Thresholds: " + "; ".join(f"{k} = {v}" for k, v in T.items() if not k.startswith("_")) + "\n")
    for s in SITES:
        df = wins[s]
        if df is None:
            continue
        M = screen_masks(df, T, s)
        allm = np.ones(len(df), bool)
        rows = []
        for name, m in M.items():
            m = m.fillna(False).to_numpy(bool)
            rows.append([name, int((~m).sum()), f"{100 * (~m).mean():.1f} %", int(m.sum())])
            allm &= m
        rows.append(["**all screens jointly**", int((~allm).sum()), f"{100 * (~allm).mean():.1f} %", int(allm.sum())])
        per_off = df.assign(pass_all=allm).groupby("offset")["pass_all"].sum().to_dict()
        W(f"**{s}** ({len(df)} candidates; passing all screens per offset: {per_off})\n\n" + table(["screen", "windows removed (alone)", "share", "windows kept (alone)"], rows))

    # ---------------------------------------------------------------- A4 metrics and tolerances
    W("\n### A4. Selection metrics against the reference (section 3.3): distributions, tolerance counts alone and jointly with the screens\n")
    mcols = [("theta_legacy", "theta by chi (legacy method)"), ("ks_ref_median", "k_s at theta_ref, median of basins"), ("ks_ref_pooled", "k_s at theta_ref, pooled"),
             ("C_ht", "|C_ht| (3x3 Laplacian, convex first-order divides)"), ("n_basins_chi", "basins in the chi fit"), ("C_ht_n_convex", "convex divide pixels")]
    for s in SITES:
        df = wins[s]
        pj = pools[s]
        if df is None or pj is None:
            continue
        ref = pj["reference"]
        rows = [[name] + pct(df, c) for c, name in mcols if c in df.columns]
        W(f"**{s}** reference: theta {fmt(ref.get('theta'))}, k_s {fmt(ref.get('ks'))}, |C_ht| {fmt(ref.get('C_ht'))} ({ref.get('source')})\n\n"
          + table(["metric", "n", "min", "p05", "p25", "median", "p75", "p95", "max"], rows))
        M = screen_masks(df, T, s)
        allm = np.ones(len(df), bool)
        for m in M.values():
            allm &= m.fillna(False).to_numpy(bool)
        tt = df["tol_theta"].astype(str).str.lower() == "true"
        tk = df["tol_ks"].astype(str).str.lower() == "true"
        tc = df["tol_C_ht"].astype(str).str.lower() == "true"
        ta = tt & tk & tc
        rows = [["theta within +-%.2f" % T["tolerance_theta_abs"], int(tt.sum()), int((tt & allm).sum())],
                ["k_s within +-%.0f %%" % (100 * T["tolerance_ks_rel"]), int(tk.sum()), int((tk & allm).sum())],
                ["|C_ht| within +-%.0f %%" % (100 * T["tolerance_C_ht_rel"]), int(tc.sum()), int((tc & allm).sum())],
                ["k_s and |C_ht| (theta free)", int((tk & tc).sum()), int((tk & tc & allm).sum())],
                ["**all three**", int(ta.sum()), int((ta & allm).sum())]]
        per_off = df.assign(e=(ta & allm)).groupby("offset")["e"].sum().to_dict()
        W(table(["tolerance", "windows within (alone)", "within and passing all screens"], rows) + f"\nEligible (all screens and all tolerances) per offset: {per_off}; total candidates {len(df)}.\n")

    # ---------------------------------------------------------------- A5 original tiles
    W("\n### A5. The original ten tiles against the new rules (section 3.4), from `pools/<site>_pools.json` (`original_tiles`)\n")
    rows = []
    for s in ("AP", "GM"):
        pj = pools[s]
        if not pj:
            continue
        for t in pj["original_tiles"]:
            if "error" in t:
                rows.append([t["tile"], t["error"]] + [""] * 16)
                continue
            df = wins[s]
            M = screen_masks(pd.DataFrame([t]), T, s) if df is not None else {}
            passes = all(bool(m.fillna(False).iloc[0]) for m in M.values()) if M else None
            rows.append([t["tile"], f'{t["row0"]}, {t["col0"]}', t["coarse_all_valid"], t["footprint_full"], t["A_max_inside_m2"], t["A_inflow_max_single_m2"], t["upstream_fully_valid"],
                         t.get("chi_r2_area_weighted"), t.get("sa_dbic_area_weighted"), t.get("nlcd_disturbed_frac"), t.get("mining_frac"), passes,
                         t.get("theta_legacy"), t.get("ks_ref_median"), t.get("C_ht"), t.get("tol_theta"), t.get("tol_ks"), t.get("tol_C_ht"), t.get("tol_all")])
    W(table(["tile", "row0, col0", "coarse valid", "footprint full", "A_max inside (m2)", "largest edge inflow (m2)", "upstream fully valid", "chi R2 (w)", "dBIC (w)", "NLCD disturbed",
             "mining", "passes all proposed screens", "theta", "k_s(theta_ref) median", "|C_ht|", "tol theta", "tol k_s", "tol C_ht", "tol all"], rows))

    # ---------------------------------------------------------------- A6 references
    W("\n### A6. References per site (decision 6)\n")
    rows = []
    for s in ("GM", "AP"):
        L = LEGACY_REF[s]
        P = PERRON_S2[s]
        wg = (pools[s] or {}).get("whole_grid", {})
        ks_perron = L["U"] / (P["K"] * 1e-6)
        rows.append([s, L["theta"], L["ks"], L["C_ht"], P["lap_h"], P["m"], f'{ks_perron:.2f}', wg.get("theta_legacy", "-"), wg.get("ks_ref_median", "-"), wg.get("ks_ref_pooled", "-"), wg.get("C_ht", "-"),
                     wg.get("theta_fit_chi_vectorised", "-"), wg.get("n_basins", "-")])
    W(table(["site", "legacy theta", "legacy k_s (median of basins)", "legacy |C_ht|", "Perron S2 |lap z_h|", "Perron S2 m", "E/K from Perron S2 (k_s, m^(2m)... units of m=Perron)",
             "this code, whole coarse grid: theta (legacy chi)", "k_s(theta_ref) median", "k_s pooled", "|C_ht|", "theta (vectorised chi fit, >= 10 px basins)", "basins"], rows))
    pj = pools["TTV"]
    if pj and "md_proxy" in pj:
        md = pj["md_proxy"]
        W("\n**TTV M&D proxy** (`pools/TTV_pools.json` `md_proxy`): " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in md.items() if not isinstance(v, (list, dict))}) + "\n")
        wg = pj.get("whole_grid", {})
        W("TTV whole 10 m grid (18 x 18 km domain), this code: " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in wg.items()}) + "\n")

    # ---------------------------------------------------------------- A7 conflicts
    W("\n### A7. Conflict graphs (section 3.2, screen 4), from `pools/<site>_conflicts.json`\n")
    rows = []
    for s in SITES:
        pj = pools[s]
        if not pj:
            continue
        for k, c in pj["conflicts"].items():
            rows.append([s, k, pj["counts"][k]["n_candidates"], c.get("n_direct_edges"), c.get("n_direct_pairs_undirected"), c.get("max_degree_direct"), c.get("n_conflict_pairs_closure"), c.get("n_windows_with_conflict_closure"), c.get("max_degree_closure")])
    W(table(["site", "offset", "candidates", "direct edges (A drains into B)", "direct pairs", "max direct degree", "closure pairs", "windows in closure", "max closure degree"], rows))

    # ---------------------------------------------------------------- A8 unit checks
    W("\n### A8. Unit and datum checks (section 5)\n")
    rows = []
    if fp:
        for s in ("GM", "AP"):
            r = fp[s]["rule_5_3"]
            d = r["dz_coarse_minus_lidar"]
            rows.append([s, "5.3 coarse grid vs lidar block-mean", r["n_cells"], d["median"], d["p05"], d["p95"], r["fit_coarse_on_lidar"]["theil_sen_slope"], f'{r["offset_expected_m"]}', r["pass"]])
    if ti:
        r = ti["rule_5_3_TTV"]
        d = r["dz_ned_minus_lidar"]
        rows.append(["TTV", "5.3 NED (10 m, EPSG:6339) vs NCALM 2006 block-mean", r["n_cells_fully_covered_land"], d["median"], d["p05"], d["p95"], r["fit_ned_on_lidar"]["theil_sen_slope"], "[-5, 5]", r["pass"]])
    W(table(["site", "rule", "cells", "median dz (m)", "p05", "p95", "Theil-Sen slope", "expected offset (m)", "pass"], rows))
    g, rt, ck = grids["AP"]
    if ck:
        rows = []
        for t in ("AP1", "AP2", "AP3", "AP4", "AP5"):
            r = ck[t]
            d = r["dz_new_minus_pamap"]
            rows.append([t, r["n"], d["median"], r["a13_median"], d["p05"], d["p95"], r["fit_new_on_pamap"]["theil_sen_slope"], r["median_within_0.1_of_a13"], r["slope_pass_0.99_1.01"], r["pass"]])
        W("\n**Rule 5.1, AP** (`grid_AP/AP_checks.json`): new 1 m watershed grid (class-2 mean, unfilled cells) against the PAMAP tiles in metres\n\n"
          + table(["tile", "cells", "median dz new - PAMAP (m)", "addendum 13 median", "p05", "p95", "Theil-Sen slope", "median within 0.1 of a13", "slope 0.99-1.01", "pass"], rows))
    else:
        W("\n**Rule 5.1, AP**: pending (needs the watershed grid).\n")
    g2, rt2, ck2 = grids["TTV"]
    if ck2 and "rule_5_2" in ck2:
        r = ck2["rule_5_2"]
        d = r["dz_marin_minus_ncalm"]
        W("\n**Rule 5.2, TTV** (`grid_TTV/TTV_checks.json`): Marin 2019 1 m grid against the NCALM 2006 raster on ground cells\n\n"
          + table(["cells", "median dz (m)", "p05", "p95", "Theil-Sen slope", "median within 1 m", "slope 0.98-1.02", "units flag", "ellipsoid flag", "pass", "stop"],
                  [[r["n"], d["median"], d["p05"], d["p95"], r["fit_marin_on_ncalm"]["theil_sen_slope"], r["median_pass_abs_le_1m"], r["slope_pass_0.98_1.02"],
                    r["units_error_flag_slope_near_3.28_or_0.305"], r["ellipsoid_flag_offset_near_-32m"], r["pass"], r["stop"]]]))
    else:
        W("\n**Rule 5.2, TTV**: pending (needs the Marin grid).\n")

    # ---------------------------------------------------------------- A9 TTV raster
    if ti:
        W("\n### A9. The TTV NCALM raster and the NED (section 2), from `ttv_inventory/ttv_inventory.json`\n")
        b = ti["be_record"]
        n = ti["ned_record"]
        rows = [["output_be.tif", ti["copy_be"]["bytes"], ti["copy_be"]["sha256_dst"][:16], b["format"] + " " + b["dtype"] + " (COG, LZW)", b["crs"]["name"], b["crs"]["horizontal_units"], b["crs"]["vertical"],
                 f'{b["res"][0]} x {b["res"][1]}', f'{b["bounds"][0]:.1f}, {b["bounds"][1]:.1f}, {b["bounds"][2]:.1f}, {b["bounds"][3]:.1f}', fmt(b["nodata"]), f'{b["stats"]["min"]:.2f}-{b["stats"]["max"]:.2f}', f'{100 * b["stats"]["frac_valid"]:.1f} %'],
                ["USGS_13_n38w123_20120801.tif", ti["copy_ned"]["bytes"], ti["copy_ned"]["sha256_dst"][:16], n["format"] + " " + n["dtype"] + " (COG, LZW)", n["crs"]["name"], n["crs"]["horizontal_units"], n["crs"]["vertical"],
                 f'{n["res"][0]:.6g} deg', f'{n["bounds"][0]:.4f}, {n["bounds"][1]:.4f}, {n["bounds"][2]:.4f}, {n["bounds"][3]:.4f}', fmt(n["nodata"]), f'{n["stats"]["min"]:.1f}-{n["stats"]["max"]:.1f}', f'{100 * n["stats"]["frac_valid"]:.3f} %']]
        W(table(["file", "bytes", "sha256 (16)", "format", "CRS (header)", "horizontal units", "vertical CRS", "resolution", "extent", "nodata", "value range", "valid"], rows))
        v = ti["be_vs_las_class2"]
        W("\n**output_be.tif against the NCALM LAS class-2 points** (" + f'{v["n_class2"]:,} points): nearest-cell dz median {v["nearest_cell_dz_raster_minus_point"]["median"]:.4f} m, p05-p95 '
          f'{v["nearest_cell_dz_raster_minus_point"]["p05"]:.3f} to {v["nearest_cell_dz_raster_minus_point"]["p95"]:.3f}; bilinear p05-p95 {v["bilinear_dz_raster_minus_point"]["p05"]:.3f} to {v["bilinear_dz_raster_minus_point"]["p95"]:.3f} m; '
          f'cells equal to the mean of their points within 1 mm: {100 * v["frac_cells_with_points_where_raster_equals_point_mean_within_1mm"]:.1f} %, within 1 cm {100 * v["frac_within_1cm_of_mean"]:.1f} %; '
          f'equal to the min {100 * v["frac_within_1mm_of_min"]:.1f} %, max {100 * v["frac_within_1mm_of_max"]:.1f} %; raster between the cell\'s min and max {100 * v["frac_raster_between_min_and_max_of_cell_points"]:.1f} %; '
          f'valid cells without a class-2 point {100 * v["frac_valid_cells_without_class2_point_(interpolated)"]:.1f} %.\n')
        W("Coverage: " + json.dumps(ti["be_coverage"]) + "\nLow surface: " + json.dumps(ti["be_low_surface"]) + "\n")

    # ---------------------------------------------------------------- A10 file lists
    W("\n### A10. Full-grid data: file lists (section 4)\n")
    if fp:
        w = fp["AP"]["watershed"]
        W(f'AP watershed footprint: valid {w["valid_area_km2"]:.1f} km2, buffered (500 m) {w["buffered_area_km2"]:.1f} km2, envelope UTM 17N {[round(v) for v in w["envelope_utm17"]]}.\n')
        pz = w["pasda"]
        W(f'\n**PASDA quadrants** (`footprints/footprints.json`): {pz["n_quadrants"]} files, {pz["n_listed"]} listed, {len(pz["absent"])} absent, {len(pz["not_in_index"])} not in index, '
          f'{len(pz["index_rule_mismatch"])} index/rule mismatches, {pz["n_already_downloaded"]} already on Sherlock; total {pz["total_GB"]:.2f} GB zipped (gate 60 GB: {pz["gate_60GB_total"]}), LAS estimate {pz["las_bytes_estimate_GB"]:.1f} GB.\n')
        W(table(["quadrant zip", "bytes", "listed", "in index", "index = rule", "already downloaded"], [[r["file"], r["bytes"], r["listed"], r["in_index"], r.get("index_matches_rule_within_1ft"), r["already_downloaded"]] for r in pz["files"]]))
        if "usgs_laz" in w:
            u = w["usgs_laz"]
            W(f'\n**USGS LAZ tiles** (rockyweb `PA_WesternPA_2019_D20/PA_WesternPA_1_2019/LAZ/`): {u["n_files"]} files, {u["total_GB"]:.2f} GB, {u["points"]:,} points (Entwine source list); the project lists {u["n_listing_files"]} LAZ, {u["listing_total_GB"]:.0f} GB.\n')
            man = {}
            mp = os.path.join(os.environ.get("SCRATCH", ""), "a14", "ap_usgs_raw", "MANIFEST.sha256")
            if os.path.exists(mp):
                for line in open(mp):
                    parts = line.split()
                    if len(parts) == 2:
                        man[parts[1]] = parts[0]
            W(table(["LAZ", "bytes (listing)", "points", "sha256 (16)"], [[r["file"], r["bytes"], r["points"], man.get(r["file"], "pending")[:16]] for r in u["files"]]))
    if marin:
        W(f'\n**Marin 2019 tiles** (`marin_select/marin_select.json`): pool watersheds {marin["watershed_km2"]:.1f} km2 ({marin["n_terminal_basins"]} terminal basins from {marin["n_candidate_windows"]} candidate windows), buffered {marin["buffered_km2"]:.1f} km2; '
          f'{marin["n_tiles"]} tiles, {marin["n_in_ept_sources"]} in the Entwine source list, {marin["points_total"]:,} points, listed bytes {marin["bytes_listed_total"] / 1e9:.2f} GB, estimate {marin["bytes_estimate_from_points_at_3.5B"] / 1e9:.2f} GB.\n')
        W(table(["tile", "file", "in source list", "points", "bytes (listing)"], [[r["tile"], r["file"], r["in_ept_sources"], r["points"], r["bytes_listed"]] for r in marin["tiles"]]))
    else:
        W("\nMarin 2019 tile list: pending.\n")

    # ---------------------------------------------------------------- A11 comparison
    W("\n### A11. USGS copy against PASDA (section 4.1.2)\n")
    if cmp:
        W(json.dumps({k: v for k, v in cmp.items() if k not in ("usgs_tiles", "usgs_headers", "download")}, indent=1, default=str) + "\n")
        W(table(["USGS tile", "points", "version/pf", "CRS horizontal", "CRS vertical", "units", "x range", "z range"],
                [[h.get("file"), h.get("count"), f'{h.get("version")}/{h.get("point_format")}', h.get("srs_horizontal", "")[:60], h.get("srs_vertical", "")[:60], str(h.get("srs_units")), f'{h.get("minx")}-{h.get("maxx")}', f'{h.get("minz")}-{h.get("maxz")}'] for h in cmp["usgs_headers"]]))
        W(table(["file", "curl (code bytes B/s)", "wall s"], [[d["file"], d["curl"], round(d["seconds"])] for d in cmp["download"]]))
    else:
        W("pending\n")

    # ---------------------------------------------------------------- A12 gridding and routing
    W("\n### A12. Gridding and routing of the full grids (sections 4.1.5, 4.3.3)\n")
    for s in ("AP", "TTV"):
        g, rt, ck = grids[s]
        if g:
            W(f"**{s} grid** (`grid_{s}/{s}_grid.json`): " + json.dumps(g["stats"], default=str) + "\n")
            bad = [f for f in g["files"] if "error" in f]
            W(f"files: {len(g['files'])}, skipped: {len(bad)}; vertical units seen: {sorted({str(f.get('vertical_unit')) for f in g['files']})}; horizontal: {sorted({str(f.get('crs_name')) for f in g['files']})}\n")
        if rt:
            W(f"**{s} routing** (`grid_{s}/{s}_route.json`): " + json.dumps({k: v for k, v in rt.items() if k != "largest_roots"}, default=str) + "\nlargest outlets: " + json.dumps(rt["largest_roots"][:5], default=str) + "\n")
        if not g and not rt:
            W(f"{s}: pending\n")

    # ---------------------------------------------------------------- A13 jobs
    if a.jobs:
        W("\n### A13. Slurm accounting of the stage A jobs\n")
        r = subprocess.run(["sacct", "-j", a.jobs, "-X", "-o", "JobID,JobName%20,Partition,State,Elapsed,Timelimit,AllocCPUS,MaxRSS,TotalCPU", "-P"], capture_output=True, text=True)
        lines = [line.split("|") for line in r.stdout.strip().splitlines()]
        if lines:
            W(table(lines[0], lines[1:]))
    print("\n".join(out))


if __name__ == "__main__":
    main()
