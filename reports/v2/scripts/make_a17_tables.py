#!/usr/bin/env python3
"""Addendum 17 Part A: appendix tables of the register-and-figures report, generated from the run directory.
usage: make_a17_tables.py --run DIR [--jobs id,id,...]"""
import argparse
import csv
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a17_common import A16, E_M_PER_YR, E_SOURCE, REPO, SURVEY, TRUNK_SENSITIVITY_KM2, Z_RANGE, Z_RANGE_A15  # noqa: E402


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def fmt(v, d=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    if isinstance(v, (float, np.floating)):
        if abs(v) >= 1e5:
            return f"{v:,.0f}"
        if abs(v) >= 100:
            return f"{v:.1f}"
        if abs(v) < 0.01 and v != 0:
            return f"{v:.3e}"
        return f"{v:.{d}f}"
    return str(v)


def pct(v):
    return "-" if v is None or not np.isfinite(v) else f"{v:+.1%}"


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for r in rows:
        out.append("| " + " | ".join(fmt(v) for v in r) + " |")
    return "\n".join(out) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--jobs", default="")
    a = p.parse_args()
    R = a.run
    out = []
    W = out.append
    import yaml
    cfg = yaml.safe_load(open(os.path.join(REPO, "configs", "data.yaml")))
    man = list(csv.DictReader(open(os.path.join(REPO, "data", "manifest.csv"))))

    # ---------------- A1 unit checks under the decision-1 windows
    reg = load(os.path.join(R, "register", "register_check.json"))
    W("### A1. Unit checks of the parked tiles under the decision-1 windows (section 2.1), from register/register_check.json\n")
    if reg:
        for site, lab in (("TTV", "TTV-x"), ("GMF", "GM-FEMA2018-x")):
            r = reg[site]
            W(f"**{lab}**: window {Z_RANGE[site][0]:.0f}-{Z_RANGE[site][1]:.0f} m (addendum 15/16 window {Z_RANGE_A15[site][0]:.0f}-{Z_RANGE_A15[site][1]:.0f} m); {r['n']} tiles, "
              f"all pass: {fmt(r['all_pass'])}; failed the old window: {r['n_failed_a16']}; files identical to the addendum 16 parked cut: {fmt(r['all_identical_to_a16'])}.\n")
            W(table(["tile", "kind", "SW corner (E, N)", "z min", "z median", "z max", "pass (new window)", "pass (old window)", "identical to a16", "A max (km2)"],
                    [[t["tile"], t["kind"], f"{t['fp'][0]:.0f}, {t['fp'][1]:.0f}", t["z_min"], t["z_median"], t["z_max"], t["unit_check_pass"], t["a16_unit_check_pass"], t["identical_to_a16_parked"], t["A_max_m2"] / 1e6] for t in r["tiles"]]))
        W("Manifest rows against the files on disk: " + "; ".join(f"{k}: {v['rows']} rows, all ok {fmt(v['all_ok'])}" for k, v in reg["manifest"].items()) + "\n")

    # ---------------- A2 area_model
    am = load(os.path.join(R, "area_model", "area_model.json"))
    W("### A2. GM1-GM5 modelling area grids (decision 2), from area_model/area_model.json\n")
    if am:
        W(table(["tile", "area_model = old grid: outlet A (km2)", "rule grid outlet A (km2)", "lidar-only outlet A (km2)", "rule == old", "cells where rule < old", "max old - rule (km2)", "area_model sha256 (16)", "area_model_rule sha256 (16)"],
                [[t, v["outlet_old_m2"] / 1e6, v["outlet_rule_m2"] / 1e6, v["outlet_new_lidar_only_m2"] / 1e6, v["area_model_rule"]["equals_old"], v["cells_rule_lt_old"], v["max_old_minus_rule_m2"] / 1e6, v["area_model"]["sha256"][:16] + "...", v["area_model_rule"]["sha256"][:16] + "..."]
                 for t, v in am.items() if not t.startswith("_")]))
        W(f"Manifest: {am['_manifest']['rows_updated']} area_model rows updated, {am['_manifest']['rows_added']} area_model_rule rows added.\n")

    # ---------------- A3 control footprint, fetch, grid, route, check
    fp = load(os.path.join(R, "gm_ctrl", "GMC_fetch_footprint.json"))
    W("### A3. GM survey-control grid: footprint, fetch, grid, routing and checks (section 2.2.1-2.2.2)\n")
    if fp:
        W(table(["tile", "NCALM SW corner (E, N, EPSG:32610)", "catchment on the NED box (km2, tile included)", "upstream outside the tile (km2)", "upstream cells outside the FEMA survey", "largest NED area in the tile (km2)"],
                [[t, f"{v['fp_sw_utm10n'][0]:.0f}, {v['fp_sw_utm10n'][1]:.0f}", v["catchment_km2_total"], v["upstream_outside_tile_km2"], v["upstream_cells_outside_survey"], v["max_area_in_tile_km2"]] for t, v in fp["tiles"].items()]))
        b = fp["boxes"]
        W(table(["union tiles + catchments (km2)", "footprint with 500 m buffer, in the survey (km2)", "bounding box (EPSG:32610)", "1 m lattice cells (M)", "1 km boxes", "in the copy of record", "fetched now", "NED routing (s)"],
                [[fp["union_tiles_catchments_km2"], fp["footprint_km2"], ", ".join(f"{x:.0f}" for x in fp["bbox_utm10n"]), fp["lattice_cells_1m_M"], b["n"], b["n_existing_record"], b["n_missing"], fp["ned_routing_seconds"]]]))
    ef = load(os.path.join(R, "ept_GMC", "ept_fetch_6339.json"))
    if ef:
        s = ef["summary"]
        W(f"Fetch of the missing boxes (a14_ept_fetch.py): {s['n_ok']} / {s['n_boxes']} boxes, {s['bytes'] / 1e9:.2f} GB in {s['seconds']:.0f} s ({s['MB_per_s']:.1f} MB s-1); CRS written EPSG:{s['epsg']}+{s['vertical_epsg']}.\n")
    g = load(os.path.join(R, "grid_GMC", "GMC_grid.json"))
    if g:
        s = g["stats"]
        W("Grid (a14_grid_route.py --site GMC):\n")
        W(table(["lattice", "cells inside footprint", "cells with class 2 (inside)", "class-2 points", "density (pts m-2)", "points per cell 5-50-95 %", "NN-filled within 10 m", "left nodata beyond 10 m", "NN fill distance 50-95-99-max (m)", "seconds", "peak GB"],
                [[f"{s['lattice_shape'][0]} x {s['lattice_shape'][1]}", s["n_inside_footprint"], s["n_cells_with_class2_inside"], s["total_class2"], s["class2_density_inside_pts_per_m2"], str(s["points_per_cell_percentiles_5_50_95"]),
                  f"{s['n_filled_within_max']:,} ({100 * s['frac_inside_filled']:.1f}%)", s["n_gap_beyond_max_left_nodata"], str(s["nn_fill_distance_percentiles_50_95_99_max_m"]), s["seconds_total"], s["peak_rss_GB"]]]))
        dates = sorted({tuple(f["utc_dates_class2"][:2]) for f in g["files"] if f.get("utc_dates_class2")})
        if dates:
            W(f"Class-2 flight dates over the {len(g['files'])} boxes: {min(d[0] for d in dates)} to {max(d[1] for d in dates)} UTC ({g['files'][0].get('utc_dates_class2', ['', '', ''])[2]}).\n")
    gp = load(os.path.join(R, "grid_GMC", "GMC_gaps.json"))
    rt = load(os.path.join(R, "grid_GMC", "GMC_route.json"))
    if gp and rt:
        W("Re-routing with the gap rule (a15_reroute.py --site GMC):\n")
        W(table(["nodata components", "enclosed (filled)", "enclosed cells / km2", "enclosed >= 1e4 cells", "valid after", "data-edge outlets", "largest basin km2", "budget closed", "pad", "seconds", "peak GB"],
                [[gp["n_nodata_components"], gp["n_enclosed_components"], f"{gp['n_enclosed_cells']:,} / {gp['enclosed_area_km2']:.3f}", gp.get("n_enclosed_ge_1e4_cells", 0), rt["n_valid"], rt["n_edge_cells"], rt["max_area_km2"], rt["root_area_sum_equals_valid"], rt["pad_cells"], rt["seconds_total"], rt["peak_rss_GB"]]]))
        W("Largest outlets: " + "; ".join(f"({r['x']:,.0f}, {r['y']:,.0f}) {r['area_cells'] / 1e6:.2f} km2 ({100 * r['share_of_valid']:.1f}%)" for r in rt["largest_roots"][:5]) + "\n")
    ck = load(os.path.join(R, "grid_GMC", "gmc_check.json"))
    if ck:
        W("Units and datum (a16_gmf_check.py --site GMC):\n")
        r1 = ck["rule_5_3_ned_minus_gmf"]
        r2 = ck.get("ncalm_overlap", {})
        rows = [["rule 5.3: NED 10 m - grid block mean (ground cells)", r1["n_cells_10m"], r1["dz"]["median"], f"{r1['dz']['p05']:.2f}, {r1['dz']['p95']:.2f}", r1["dz"]["rms"], f"{r1['fit_ned_on_gmf']['theil_sen_slope']:.5f} ({r1['fit_ned_on_gmf']['theil_sen_slope_range'][0]:.5f}-{r1['fit_ned_on_gmf']['theil_sen_slope_range'][1]:.5f})", "|median| < 5 m, slope 0.98-1.02", "pass" if r1["pass"] else "FAIL"]]
        if r2.get("n"):
            rows.append(["NCALM 2003 - grid (every 5th NCALM cell, bilinear)", r2["n"], r2["dz_ncalm_minus_gmf"]["median"], f"{r2['dz_ncalm_minus_gmf']['p05']:.2f}, {r2['dz_ncalm_minus_gmf']['p95']:.2f}", r2["dz_ncalm_minus_gmf"]["rms"],
                         f"{r2['fit_ncalm_on_gmf']['theil_sen_slope']:.4f} ({r2['fit_ncalm_on_gmf']['theil_sen_slope_range'][0]:.4f}-{r2['fit_ncalm_on_gmf']['theil_sen_slope_range'][1]:.4f})", "|offset| 30-34 m (NCALM ellipsoidal), slope 0.98-1.02; stop near 3.28 / 0.305",
                         ("offset in window, " if r2["offset_magnitude_in_30_34m"] else "offset OUTSIDE window, ") + ("slope pass" if r2["slope_pass_0.98_1.02"] else "slope FAIL") + ("; STOP" if r2["stop_slope_near_3.28_or_0.305"] else "; no stop")])
        W(table(["comparison", "cells", "median dz (m)", "5-95 %", "RMS", "Theil-Sen slope (range)", "criterion", "result"], rows))

    # ---------------- A4 control tiles
    cf = load(os.path.join(R, "draw_gmc", "GMC_footprints.json"))
    ct = load(os.path.join(R, "tiles_gmc", "tiles.json"))
    W("### A4. The control tiles GM1_fema-GM7_fema (section 2.2.3), from draw_gmc/GMC_footprints.json and tiles_gmc/tiles.json\n")
    if cf:
        W(f"Transformation: {cf['transformation']}; control grid bounds (EPSG:6339): {', '.join(f'{x:,.0f}' for x in cf['grid_bounds_6339'])}.\n")
        W(table(["tile", "NCALM SW corner (EPSG:32610)", "reprojected SW corner (EPSG:6339)", "shift at SW corner (dx, dy m)", "shift at NE corner (dx, dy m)", "snapped SW corner", "snap residual (m)"],
                [[t, f"{v['ncalm_sw_32610'][0]:.3f}, {v['ncalm_sw_32610'][1]:.3f}", f"{v['sw_6339'][0]:.3f}, {v['sw_6339'][1]:.3f}", f"{v['reprojection_shift_m_at_corners']['sw'][0]:.4f}, {v['reprojection_shift_m_at_corners']['sw'][1]:.4f}",
                  f"{v['reprojection_shift_m_at_corners']['ne'][0]:.4f}, {v['reprojection_shift_m_at_corners']['ne'][1]:.4f}", f"{v['fp_x0']}, {v['fp_y0']}", f"{v['snap_residual_m'][0]:.4f}, {v['snap_residual_m'][1]:.4f}"] for t, v in cf["tiles"].items()]))
    if ct:
        W(table(["tile", "SW corner (E, N)", "z median", "z min", "z max", f"unit check ({Z_RANGE['GMC'][0]:.0f}-{Z_RANGE['GMC'][1]:.0f} m)", "A max (km2)", "outlet cells", "filled-gap channel cells", "class-2 pts/m2", "NN-filled frac"],
                [[t["tile"], f"{t['fp_x0']}, {t['fp_y0']}", t["z_median"], t["z_min"], t["z_max"], t["unit_check_pass"], t["A_max_m2"] / 1e6, t["n_outlets_fd0"], t["gapfill_channel_cells_A_ge_1e3"], t.get("class2_density_pts_m2"), t.get("frac_nn_filled_le10m")] for t in ct["tiles"]]))
        W("Checksums (sha256) of the control tiles' products:\n")
        W(table(["tile", "role", "path", "sha256 (first 16)", "bytes"], [[t["tile"], role, f["path"], f["sha256"][:16] + "...", f["bytes"]] for t in ct["tiles"] for role, f in t["files"].items()]))
        if ct["unit_check_failures"]:
            W(f"Unit-check failures: {[r['tile'] for r in ct['unit_check_failures']]} -> stop, nothing registered.\n")

    # ---------------- A5 the area check
    ac = load(os.path.join(R, "gm_ctrl_area", "gm_ctrl_area_check.json"))
    W("### A5. The FEMA full-grid area against the old grids at the addendum 16 entries (section 2.2.4, report only), from gm_ctrl_area/gm_ctrl_area_check.json\n")
    if ac:
        rows = []
        for t, ents in ac["entries"].items():
            for e in ents:
                rows.append([t, f"{e['tile_cell_EN'][0]:,.1f}, {e['tile_cell_EN'][1]:,.1f}", e["old_m2"] / 1e6, e["new_lidar_only_m2"] / 1e6, e["D_old_minus_new_m2"] / 1e6, e["perron_strip_inflow_m2"] / 1e6, e["fema_at_cell_m2"] / 1e6, e["fema_max_3x3_m2"] / 1e6, e["fema_max_11x11_m2"] / 1e6,
                             str(e["fema_max_11x11_offset_rc"]), pct(e["fema_3x3_vs_old_rel"]) if e["fema_3x3_vs_old_rel"] is not None else "-", pct(e["fema_11x11_vs_old_rel"]) if e["fema_11x11_vs_old_rel"] is not None else "-", "trunk" if e["trunk_ge_1e5"] else "minor"])
        W(table(["tile", "entry (E, N)", "old (km2)", "lidar-only (km2)", "D = old - new", "strip inflow (a16)", "FEMA at cell", "FEMA max 3x3", "FEMA max 11x11", "11x11 argmax offset (dr, dc)", "FEMA 3x3 vs old", "FEMA 11x11 vs old", "kind"], rows))
        W("Largest area inside each footprint, FEMA full grid against the registered grid:\n")
        W(table(["tile", "registered grid", "registered A max (km2)", "argmax (r, c)", "FEMA A max (km2)", "argmax (r, c)", "rel. diff", "cells >= 0.1 km2 registered / FEMA"],
                [[t, v["registered_role"], v["registered_max_m2"] / 1e6, str(v["registered_argmax_rc"]), v["fema_max_m2"] / 1e6, str(v["fema_argmax_rc"]), pct(v["rel_diff"]), f"{v['registered_cells_ge_1e5']:,} / {v['fema_cells_ge_1e5']:,}"] for t, v in ac["tiles"].items()]))

    # ---------------- A6 the registered tile list
    W("### A6. The registered tile list per site (section 2.1), from configs/data.yaml and the cut records\n")
    kinds = {}
    for path in (os.path.join(os.path.dirname(A16), "20261118T160326Z_a15_stage_b", "tiles", "tiles.json"), os.path.join(R, "tiles_ttv", "tiles.json"), os.path.join(R, "tiles_gmf", "tiles.json"), os.path.join(R, "tiles_gmc", "tiles.json")):
        j = load(path)
        if j:
            for t in j["tiles"]:
                kinds[(t["version"], t["tile"])] = t
    roles = {}
    for r in man:
        roles.setdefault((r["site"], r["tile"]), set()).add(r["role"])
    rows = []
    n_site = {}
    for v, blk in cfg["versions"].items():
        site = blk.get("site_label", blk["site"])
        for t in blk["tiles"]:
            k = kinds.get((v, t), {})
            rl = roles.get((v, t), set())
            src = "area_model (old tile grid, decision 2)" if "area_model" in rl else "full-grid routing (role area)"
            kind = k.get("kind", "?")
            a_max = (am[t]["outlet_old_m2"] if ("area_model" in rl and am and t in am) else k.get("A_max_m2", np.nan)) / 1e6 if k else np.nan   # GM1-GM5: the area_model grid, not the lidar-only cut
            rows.append([site, t, v, SURVEY.get(v, "?"), f"{k.get('fp_x0', '?')}, {k.get('fp_y0', '?')} (EPSG:{blk['crs_epsg']})", src, "M&D by design" if kind == "M&D" else kind, a_max])
            n_site[site] = n_site.get(site, 0) + 1
    W(table(["site", "tile", "version", "survey", "footprint SW corner (E, N)", "area source for modelling", "kind", "largest A inside (km2, on the area source)"], rows))
    W("Registered per site: " + ", ".join(f"{s} {n}" for s, n in n_site.items()) + f"; in all {sum(n_site.values())} (the {n_site.get('GM', 0) and 7} control tiles included in GM).\n")
    big = [(r[1], r[2], r[7]) for r in rows if isinstance(r[7], float) and np.isfinite(r[7]) and r[7] > TRUNK_SENSITIVITY_KM2]
    W(f"Decision 7 (for addendum 18): tiles whose largest drainage area inside exceeds {TRUNK_SENSITIVITY_KM2:.0f} km2 on the registered grid: " + (", ".join(f"{t} ({v}, {a:.2f} km2)" for t, v, a in big) if big else "none") + ".\n")
    W("Decision 6: E carried over to every tile of the site: " + "; ".join(f"{s} {E_M_PER_YR[s]:.2e} m/yr ({E_SOURCE[s]})" for s in E_M_PER_YR) + ".\n")

    # ---------------- A7 figures
    fg = load(os.path.join(R, "figures.json"))
    W("### A7. The figures (section 3), from figures.json\n")
    if fg:
        W(table(["figure", "PNG bytes", "PDF bytes", "size (in)", "scale applied", "dpi", "notes"],
                [[k, v["png_bytes"], v["pdf_bytes"], f"{v['size_in'][0]} x {v['size_in'][1]}", v["scale"], v["dpi"], (f"block {v['block_m']} m; eligible windows {v['eligible_counts']}; tiles {v['n_tiles']}" if "block_m" in v else "")] for k, v in fg["figures"].items()]))
        W("Panel statistics (title values):\n")
        W(table(["panel", "kind", "survey", "area grid", "z min", "z max", "largest A inside (km2)", "NN-filled frac", "network cells (A >= 1e3)"],
                [[k, v["kind"], v["survey"], v["area_role"], v["z_min"], v["z_max"], v["A_max_m2"] / 1e6, v["frac_nn_filled"], v["n_net_cells"]] for k, v in fg["panels"].items()]))

    # ---------------- A8 slurm
    W("### A8. Slurm accounting of the addendum 17 Part A jobs\n")
    if a.jobs:
        try:
            txt = subprocess.run(["sacct", "-j", a.jobs, "-X", "-o", "JobID,JobName%22,Partition,State,Elapsed,Timelimit,AllocCPUS,MaxRSS,TotalCPU", "-P"], capture_output=True, text=True, check=True).stdout
            lines = [line.split("|") for line in txt.strip().splitlines()]
            W(table(lines[0], lines[1:]))
            txt2 = subprocess.run(["sacct", "-j", a.jobs, "-o", "JobID,MaxRSS,Elapsed,AllocCPUS", "-P"], capture_output=True, text=True, check=True).stdout
            peak = {}
            for line in txt2.strip().splitlines()[1:]:
                jid, rss, el, cpus = line.split("|")
                base = jid.split(".")[0]
                if rss.endswith("K"):
                    peak[base] = max(peak.get(base, 0), float(rss[:-1]) / 1e6)
                elif rss.endswith("M"):
                    peak[base] = max(peak.get(base, 0), float(rss[:-1]) / 1e3)
                elif rss.endswith("G"):
                    peak[base] = max(peak.get(base, 0), float(rss[:-1]))
            W("Peak memory per job (GB, from the step records): " + "; ".join(f"{k}: {v:.1f}" for k, v in peak.items()) + "\n")

            def secs(s):
                d, _, hms = s.rpartition("-")
                h, m, sec = (int(x) for x in hms.split(":"))
                return (int(d) * 86400 if d else 0) + h * 3600 + m * 60 + sec
            used = sum(secs(line[4]) * int(line[6]) for line in lines[1:]) / 3600
            lim = sum(secs(line[5]) * int(line[6]) for line in lines[1:]) / 3600
            W(f"Core-hours used (elapsed x CPUs): {used:.2f}; sum of CPU-weighted limits: {lim:.1f}.\n")
        except Exception as e:  # pragma: no cover
            W(f"sacct failed: {e!r}\n")
    txt = "\n".join(out)
    with open(os.path.join(R, "appendix_tables.md"), "w") as fh:
        fh.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
