#!/usr/bin/env python3
"""Addendum 15: appendix tables of the stage B report, generated from the run directory. usage: make_a15_tables.py --run DIR [--jobs id,id,...]"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a15_common import STAGE_A, THRESHOLDS  # noqa: E402

SITES = ["GM", "AP", "TTV"]


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
    T = json.load(open(THRESHOLDS))

    # ---------------- B1 georeference
    ex = load(os.path.join(R, "md_georef", "extract.json"))
    loc = load(os.path.join(R, "md_georef", "locate.json"))
    gcp = load(os.path.join(R, "md_georef", "gcp.json"))
    dig = load(os.path.join(R, "md_georef", "digitize.json"))
    r22 = load(os.path.join(R, "md_georef", "rule_2_2.json"))
    W("### B1. M&D (1989) Fig. 3: extraction, placement and ground control (section 2)\n")
    if ex:
        W(table(["item", "value"], [["PDF page", ex["page"]], ["page pixels", f"{ex['page_pixels'][0]} x {ex['page_pixels'][1]} at {ex['dpi']} dpi (1-bit CCITT scan)"],
                                    ["crop (page px: left, top, right, bottom)", ex["crop_box_page_px"]], ["crop pixels", f"{ex['crop_pixels'][0]} x {ex['crop_pixels'][1]}"],
                                    ["scale bar", f"500 m = {ex['scale_bar_px']} px -> {ex['m_per_px']:.4f} m/px (1:{ex['map_scale_1_to']:,.0f})"],
                                    ["thick-line components / network components", f"{ex['n_thick_components']} / {ex['n_network_components']}"]]))
    if loc:
        lr = loc["rec"]
        W("\nCoarse placement (channel-network cross-correlation at 5 m):\n")
        W(table(["rank", "ncc", "scale", "rotation (deg)", "top-left of crop (E, N)", "second peak ncc", "second peak (E, N)"],
                [[i + 1, c["ncc"], c["scale"], c["rot_deg"], f"{c['top_left_utm'][0]:.0f}, {c['top_left_utm'][1]:.0f}", c["second_ncc"], f"{c['second_top_left_utm'][0]:.0f}, {c['second_top_left_utm'][1]:.0f}"] for i, c in enumerate(lr["coarse"][:5])]))
        W(f"\nLocal refinement: {lr['n_patches']} patches, {lr['n_good']} good matches, {lr['n_used']} used; patch affine RMS {lr['patch_affine_rms_m']:.1f} m, max {lr['patch_affine_max_m']:.1f} m; pixel size {lr['pixel_size_m'][0]:.4f} x {lr['pixel_size_m'][1]:.4f} m; rotation {lr['rotation_deg'][0]:.2f} / {lr['rotation_deg'][1]:.2f} deg.\n")
    if gcp:
        gr = gcp["rec"]
        W("\nGround control points (figure pixel -> EPSG:6339) and residuals of the affine fit:\n")
        W(table(["id", "feature", "figure (row, col)", "E", "N", "residual (m)"], [[g["id"], g["feature"], f"{g['fig_rc'][0]:.0f}, {g['fig_rc'][1]:.0f}", g["utm"][0], g["utm"][1], g["residual_m"]] for g in gr["gcps"]]))
        A = np.array(gr["affine_col_row_to_xy"])
        W(f"\nAffine (col, row, 1) -> (E, N): E = {A[0,0]:.6f} col + {A[1,0]:.6f} row + {A[2,0]:.3f}; N = {A[0,1]:.6f} col + {A[1,1]:.6f} row + {A[2,1]:.3f}. Pixel size {gr['pixel_size_m'][0]:.4f} x {gr['pixel_size_m'][1]:.4f} m (scale bar {gr['scale_bar_m_per_px']:.4f}); rotation {gr['rotation_deg'][0]:.2f} / {gr['rotation_deg'][1]:.2f} deg. Rule 2.1: RMS {gr['rule_2_1']['rms_m']:.1f} m, max {gr['rule_2_1']['max_m']:.1f} m, n = {gr['rule_2_1']['n']}: **{'pass' if gr['rule_2_1']['pass'] else 'fail'}**. Figure junctions {gr['n_figure_junctions']}, D8 confluences {gr['n_dem_confluences']}, unambiguous pairs {gr['n_unambiguous_pairs']}.\n")
    if dig:
        W("\n### B2. Digitised layers (section 2.4) and rule 2.2\n")
        sa, ch, hd, co = dig["study_area"], dig["channels"], dig["heads"], dig["colluvium"]
        W(table(["layer", "content"], [["study_area", f"area {sa['area_km2']:.3f} km2 (raster {sa['raster_area_km2']:.3f}), {sa['n_vertices']} vertices ({sa.get('n_vertices_snapped_to_dashes', '-')} snapped to dashes); z on the 1 m grid min {sa['z_min_1m']:.1f}, p01 {sa['z_p01_1m']:.1f}, p99 {sa['z_p99_1m']:.1f}, max {sa['z_max_1m']:.1f} m"],
                                        ["channels", f"{ch['n_lines']} node-to-node lines, {ch['total_length_km']:.1f} km in {ch['n_components']} connected trees (paper: 9.1 km)"],
                                        ["channel_heads", f"{hd['n_endpoints']} skeleton end points; {hd['n_exits_at_boundary']} exits at the boundary; {hd['n_heads_all_upstream_endpoints']} upstream end points; {hd['n_heads_excluding_continuations']} excluding discontinuous-reach continuations (paper: 68 + 1 culvert; 5 trail heads excluded)"],
                                        ["labels placed", ", ".join(f"{k} ({v[0]:.0f}, {v[1]:.0f})" for k, v in hd["labels"].items()) + f"; culvert triangle at {hd['culvert_triangle_utm'][0]:.0f}, {hd['culvert_triangle_utm'][1]:.0f}"],
                                        ["colluvial_deposits", f"{co['n_polygons']} polygons, {co['total_area_km2']:.3f} km2 (stipple-dot density > {co['dot_density_threshold']} in a {co['density_window_px']} px window)"],
                                        ["ground_control", "the GCP table above"], ["file", dig["gpkg"]]]))
    if r22:
        W(f"\nRule 2.2: area {r22['area_km2']:.3f} km2 in [{r22['area_window'][0]:.3f}, {r22['area_window'][1]:.3f}]: {'pass' if r22['area_pass'] else 'fail'}; z_min {r22['z_min_1m']:.1f} m in [46, 76]: {'pass' if r22['z_min_pass'] else 'fail'}; z_max {r22['z_max_1m']:.1f} m in [299, 329]: {'pass' if r22['z_max_pass'] else 'fail'}. **{'PASS' if r22['pass'] else 'FAIL'}**\n")

    # ---------------- B3 re-routing
    W("\n### B3. Re-routing with the gap rule (section 3), from grid_<site>/<site>_gaps.json and <site>_route.json\n")
    rows = []
    for s in SITES:
        g = load(os.path.join(R, f"grid_{s}", f"{s}_gaps.json"))
        r = load(os.path.join(R, f"grid_{s}", f"{s}_route.json"))
        if not g or not r:
            continue
        pc = g.get("enclosed_size_cells_percentiles_50_90_99_max") or ["-"] * 4
        fd = g.get("enclosed_fill_distance_m_percentiles_50_90_99_max") or ["-"] * 4
        rows.append([s, f"{g['shape'][0]} x {g['shape'][1]}", g.get("pad_cells", 0), g["n_valid_before"], g["n_nodata_components"], g["n_enclosed_components"], g["n_enclosed_cells"], g["enclosed_area_km2"],
                     f"{fmt(pc[0])} / {fmt(pc[1])} / {fmt(pc[2])} / {fmt(pc[3])}", f"{fmt(fd[0])} / {fmt(fd[1])} / {fmt(fd[2])} / {fmt(fd[3])}", g.get("n_enclosed_ge_1e4_cells"),
                     r["n_valid"], r["n_roots"], r["max_area_km2"], r["root_area_sum_equals_valid"], r["seconds_total"], r["peak_rss_GB"]])
    W(table(["site", "grid", "pad", "valid before", "nodata comps", "enclosed comps", "enclosed cells", "enclosed km2", "enclosed size p50/p90/p99/max (cells)", "fill distance p50/p90/p99/max (m)", "enclosed >= 1e4 cells",
             "valid after", "outlets", "largest basin km2", "budget closed", "s", "peak GB"], rows))
    for s in SITES:
        r = load(os.path.join(R, f"grid_{s}", f"{s}_route.json"))
        if r:
            W(f"\n{s} largest outlets: " + "; ".join(f"({o['x']:.0f}, {o['y']:.0f}) {o['area_cells'] / 1e6:.2f} km2 ({o['share_of_valid'] * 100:.1f} %)" for o in r["largest_roots"][:5]) + "\n")
    bd = load(os.path.join(R, "grid_GM", "GM_budget_diag.json"))
    if bd:
        W(f"\nGM budget diagnostic (padded run): receivers higher {bd['n_receiver_higher']}, two-cycles {bd['n_two_cycles']}, receivers equal in the float32 product {bd['n_receiver_equal']}, root sum {bd['root_area_sum']:,} of {bd['n_valid']:,}.\n")
    bd0 = load(os.path.join(R, "grid_GM", "first_pass_unpadded", "GM_budget_diag.json"))
    if bd0:
        W(f"GM first pass (unpadded, superseded): root sum {bd0['root_area_sum']:,} of {bd0['n_valid']:,}; receivers higher {bd0['n_receiver_higher']} (all in the last two columns), two-cycles {bd0['n_two_cycles']}.\n")

    # ---------------- B4 TTV reference
    tr = load(os.path.join(R, "ttv_reference", "TTV_reference.json"))
    if tr:
        W("\n### B4. TTV reference from the M&D polygon on the 10 m NED (decision 4), against the stage A proxy\n")
        pg = tr["polygon"]
        px = tr["stage_a_proxy"]
        W(table(["quantity", "M&D polygon (a15)", "proxy catchment (stage A)"],
                [["theta (legacy chi)", pg["theta_legacy"], px["theta"]], ["k_s at theta, median of basins", pg["ks_ref_median"], px["ks"]], ["k_s pooled", pg["ks_ref_pooled"], "-"],
                 ["basins in the k_s fit", pg["ks_ref_n_basins"], "-"], ["|C_ht| (1/m)", pg["C_ht"], px["C_ht"]], ["convex divide pixels", pg["C_ht_n_convex"], "-"],
                 ["cells / km2", f"{pg['polygon_cells']} / {pg['polygon_km2']:.3f}", "6,729 / 0.673"], ["z range (m)", f"{pg['z_min']:.1f}-{pg['z_max']:.1f}", "61-317"],
                 ["largest A inside (km2)", pg["A_max_inside_m2"] / 1e6, "0.67"], ["chi R2 (area-weighted) at theta", pg["basin_stats_at_theta"].get("chi_r2_area_weighted"), "-"],
                 ["dBIC (area-weighted)", pg["basin_stats_at_theta"].get("sa_dbic_area_weighted"), "-"]]))
        W(f"\nWhole-grid k_s at the polygon theta: median {tr['whole_grid_ks_at_theta']['ks_ref_median']:.2f}, pooled {tr['whole_grid_ks_at_theta']['ks_ref_pooled']:.2f} ({tr['whole_grid_ks_at_theta']['n_basins']} basins).\n")

    # ---------------- B5 eligibility
    W("\n### B5. Eligibility under the settled rules (section 4), from draw/<site>_draw.json\n")
    rows = []
    for s in SITES:
        d = load(os.path.join(R, "draw", f"{s}_draw.json"))
        if not d:
            continue
        e = d["eligibility"]
        rows.append([s, e["n_candidates"], e["n_screens_pass"], e["n_tol25"], e["n_screens_and_tol25"], e["n_gap_excluded_of_screens_tol25"], e["n_eligible_tol25_after_gap"],
                     e["n_tol50"], e["n_screens_and_tol50"], e["n_gap_excluded_of_screens_tol50"], e["n_eligible_tol50_after_gap"], e["min_eligible"], f"+-{int(e['tolerance_used'] * 100)} %", e["n_eligible"],
                     ", ".join(f"{k}: {v}" for k, v in e["per_offset"].items())])
    W(table(["site", "candidates", "pass set B screens", "within +-25 %", "screens and +-25 %", "gap-excluded (25 %)", "eligible at +-25 %", "within +-50 %", "screens and +-50 %", "gap-excluded (50 %)", "eligible at +-50 %", "widening threshold", "tolerance used", "eligible", "per offset"], rows))
    for s in SITES:
        d = load(os.path.join(R, "draw", f"{s}_draw.json"))
        if d:
            W(f"\n{s} windows removed by each set B screen (alone): " + "; ".join(f"{k}: {v}" for k, v in d["eligibility"]["screens_each"].items()) + ".\n")

    # original tiles
    W("\n### B6. The original ten tiles under the settled rules (sections 1.9 and 4.3), from stage A pools/<site>_pools.json\n")
    rows = []
    import rasterio
    for s in ("GM", "AP"):
        pj = load(os.path.join(STAGE_A, "pools", f"{s}_pools.json"))
        d = load(os.path.join(R, "draw", f"{s}_draw.json"))
        ref = d["reference"] if d else None
        with rasterio.open(os.path.join(STAGE_A, "pools", f"{s}_nlcd2021.tif")) as ds:
            nlcd = ds.read(1)
        lw = None
        if s == "AP":
            with rasterio.open(os.path.join(STAGE_A, "pools", "AP_padep_longwall_panels.tif")) as ds:
                lw = ds.read(1)
        for t in pj["original_tiles"]:
            if "error" in t:
                rows.append([t["tile"], t["error"]] + ["-"] * 12)
                continue
            r0, c0, Wc = t["row0"], t["col0"], t["W"]
            t["nlcd_builtup_frac"] = float(np.isin(nlcd[r0:r0 + Wc, c0:c0 + Wc], T["disturbance_nlcd_classes"]).mean())
            if lw is not None:
                t["longwall_panels_frac"] = float(lw[r0:r0 + Wc, c0:c0 + Wc].mean())
            sc = {"trunk <= 5 km2": t["A_max_inside_m2"] <= T["trunk_A_max_inside_m2_max"], "inflow <= 2 km2": t["A_inflow_max_single_m2"] <= T["trunk_A_inflow_max_single_m2_max"],
                  "R2 >= 0.90": t["chi_r2_area_weighted"] >= T["transient_chi_r2_area_weighted_min"], "dBIC <= 20": not (t["sa_dbic_area_weighted"] > T["transient_sa_dbic_area_weighted_max"]),
                  "NLCD (11, 22-24, 31) <= 10 %": not (t.get("nlcd_builtup_frac", t.get("nlcd_disturbed_frac", 0)) > 0.10),
                  "longwall = 0": (not (t.get("longwall_panels_frac", 0) > 0)) if s == "AP" else None, "upstream valid": t["upstream_fully_valid"] if s == "GM" else None}
            ks_dev = t["ks_ref_median"] / ref["ks"] - 1 if ref else np.nan
            ch_dev = t["C_ht"] / ref["C_ht"] - 1 if ref else np.nan
            rows.append([t["tile"], t["A_max_inside_m2"] / 1e6, t["A_inflow_max_single_m2"] / 1e6, t["chi_r2_area_weighted"], t["sa_dbic_area_weighted"], t["nlcd_builtup_frac"], t.get("longwall_panels_frac", "-") if s == "AP" else "-",
                         t["upstream_fully_valid"] if s == "GM" else "-", ", ".join(k for k, v in sc.items() if v is False) or "none", t["theta_legacy"], f"{ks_dev * 100:+.0f} %", f"{ch_dev * 100:+.0f} %",
                         abs(ks_dev) <= 0.25 and abs(ch_dev) <= 0.25, abs(ks_dev) <= 0.5 and abs(ch_dev) <= 0.5])
    W(table(["tile", "trunk inside (km2)", "largest inflow (km2)", "chi R2 (w)", "dBIC (w)", "NLCD (11, 22-24, 31)", "longwall", "upstream valid", "set B screens failed", "theta (reported)", "k_s dev", "|C_ht| dev", "within +-25 %", "within +-50 %"], rows))
    W("\nNLCD and longwall fractions recomputed over the tiles' coarse cells from the stage A rasters (pools/<site>_nlcd2021.tif, AP_padep_longwall_panels.tif).\n")

    # ---------------- B7 M&D windows and the draw
    for s in SITES:
        d = load(os.path.join(R, "draw", f"{s}_draw.json"))
        if not d:
            continue
        W(f"\n### B7-{s}. The draw at {s} (section 5): seed {d['seed']}, {d['n_eligible']} eligible windows, target {d['target']}, accepted {d['n_accepted']}, skipped {d['n_skipped']}\n")
        if d.get("md_windows"):
            m = d["md_windows"]
            W(f"M&D windows (5.2): polygon {m['polygon_km2']:.3f} km2 on the 10 m lattice, {m['n_candidate_windows_intersecting']} candidate windows intersect it; chosen {m['n_windows']} non-overlapping windows covering {m['coverage'] * 100:.1f} % ({'>= 80 %: pass' if m['coverage_pass_80'] else 'below 80 %: best coverage reported'}).\n")
            W(table(["window (offset, id)", "footprint SW (E, N)", "set B screens pass", "screens failed", "within +-25 %", "within +-50 %", "k_s dev", "|C_ht| dev", "theta dev", "gap-excluded (channel cells in filled gaps)"],
                    [[f"{w['offset']}, {w['window']}", f"{w['fp'][0]}, {w['fp'][1]}", w["screens_pass"], ", ".join(k for k, v in w["screens"].items() if not v) or "none", w["tol25"], w["tol50"], f"{w['ks_rel_dev'] * 100:+.0f} %", f"{w['C_ht_rel_dev'] * 100:+.0f} %", f"{w['theta_dev']:+.2f}", f"{w['gap_excluded']} ({w.get('gap_channel_cells_in_filled_gap')})"] for w in m["windows"]]))
        W("\nPre-placed: " + "; ".join(f"{q['id']} ({q['kind']}) SW {q['fp'][0]}, {q['fp'][1]}" for q in d["pre_placed"]) + "\n")
        W("\nWithin-offset direct-edge check against stage A: " + "; ".join(f"offset {k}: stage A {v['stage_a']}, here {v['here']}, equal {v['equal']}" for k, v in d["within_offset_edge_check"].items()) + "\n")
        W("\nOrdered list (random permutation of the eligible windows sorted by offset and window id):\n")
        W(table(["rank", "offset", "window", "footprint SW (E, N)", "k_s dev", "|C_ht| dev", "theta", "result"],
                [[o["rank"], o["offset"], o["window"], f"{o['fp'][0]}, {o['fp'][1]}", f"{o['ks_rel_dev'] * 100:+.0f} %", f"{o['C_ht_rel_dev'] * 100:+.0f} %", o["theta"], o["result"]] for o in d["ordered"]]))
        W("\nSkip reasons: " + "; ".join(f"{k}: {v}" for k, v in d["skip_reasons"].items()) + "\n")

    # ---------------- B8 tiles
    tj = load(os.path.join(R, "tiles", "tiles.json"))
    if tj:
        W("\n### B8. Tiles cut (section 6): per-tile unit check and products, from tiles/tiles.json\n")
        rows = []
        for t in tj["tiles"]:
            rows.append([t["tile"], t["kind"], t["version"], f"{t['fp_x0']}, {t['fp_y0']}", t["epsg"], t["frac_valid"], t["z_median"], t["z_min"], t["z_max"], t["unit_check_pass"], t["A_max_m2"] / 1e6, t["n_outlets_fd0"],
                         t["gapfill_cells"], t["gapfill_channel_cells_A_ge_1e3"], t.get("class2_density_pts_m2", "-"), t.get("frac_nn_filled_le10m", "-"), t.get("frac_gap_nodata_stage_a", "-")])
        W(table(["tile", "kind", "version", "SW corner (E, N)", "EPSG", "valid frac", "z median", "z min", "z max", "unit check", "A max (km2)", "outlet cells", "filled-gap cells", "filled-gap channel cells", "class-2 pts/m2", "NN-filled frac", "stage A gap frac"], rows))
        if tj.get("gm_comparison"):
            W("\nGM originals: the re-cut grids against the existing tiles, cell by cell:\n")
            rows = [[k] + [v[c] for c in ("same_footprint", "area_frac_equal", "area_frac_rel_diff_gt_1pct", "area_frac_rel_diff_gt_1pct_interior", "area_max_abs_diff_m2", "area_old_max", "area_new_max", "area_frac_new_gt_old", "area_frac_new_lt_old", "fd_frac_equal", "fd_frac_equal_interior", "fd_old_zero_cells", "fd_new_zero_cells", "z_frac_equal_1mm", "z_max_abs_diff", "z_median_diff")] for k, v in tj["gm_comparison"].items()]
            W(table(["tile", "same footprint", "area equal frac", "|dA|/A > 1 % frac", "same, interior", "max |dA| (m2)", "old A max", "new A max", "new > old frac", "new < old frac", "fd equal frac", "fd equal interior", "fd 0 old", "fd 0 new", "z equal (1 mm) frac", "max |dz|", "median dz"], rows))
        W("\nChecksums (sha256) of the products:\n")
        rows = []
        for t in tj["tiles"]:
            for role, f in t["files"].items():
                rows.append([t["tile"], role, f["path"], f["sha256"][:16] + "...", f["bytes"]])
        W(table(["tile", "role", "path (under LEMCVT_DATA)", "sha256 (first 16)", "bytes"], rows))

    # ---------------- B9 configs and seeds
    cfg = load(os.path.join(R, "draw", "draw_config.json"))
    if cfg:
        W("\n### B9. Draw configuration (written before the draw)\n")
        W("```json\n" + json.dumps({k: v for k, v in cfg.items() if k != "thresholds_values"}, indent=1) + "\n```\n")
        W("Threshold set B: " + json.dumps(cfg["thresholds_values"]) + "\n")

    # ---------------- B10 slurm
    if a.jobs:
        W("\n### B10. Slurm accounting of the stage B jobs\n")
        try:
            txt = subprocess.run(["sacct", "-j", a.jobs, "-X", "-o", "JobID,JobName%22,Partition,State,Elapsed,Timelimit,AllocCPUS,MaxRSS,TotalCPU", "-P"], capture_output=True, text=True, check=True).stdout
            lines = [ln.split("|") for ln in txt.strip().splitlines()]
            W(table(lines[0], lines[1:]))
        except Exception as e:  # pragma: no cover
            W(f"sacct failed: {e!r}\n")
    with open(os.path.join(R, "appendix_tables.md"), "w") as f:
        f.write("\n".join(out))
    print("wrote", os.path.join(R, "appendix_tables.md"))


if __name__ == "__main__":
    main()
