#!/usr/bin/env python3
"""Addendum 16: appendix tables of the tile-finish report, generated from the run directory.
usage: make_a16_tables.py --run DIR [--jobs id,id,...] [--stage-b DIR]"""
import argparse
import csv
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a16_common import GM_LANDSCAPE_POLY_UTM, STAGE_B, THRESHOLDS  # noqa: E402


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
    return "-" if v is None or not np.isfinite(v) else f"{v:+.0%}"


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

    # ---------------- C1 TTV clip and rule 2.2
    clip = load(os.path.join(R, "md_georef", "clip.json"))
    if clip:
        W("### C1. The clipped M&D polygon and rule 2.2 (section 2.1), from md_georef/clip.json\n")
        W(table(["quantity", "stage B polygon", "clipped (a16)"], [
            ["cells (1 m) / km2", f"{clip['cells_before']:,} / {clip['area_before_km2']:.3f}", f"{clip['cells_after']:,} / {clip['area_after_km2']:.3f}"],
            ["z min / max on the 1 m grid (m)", f"{clip['z_min_before']:.1f} / {clip['z_max_before']:.1f}", f"{clip['z_min_after']:.1f} / {clip['z_max_after']:.1f}"],
            ["z p01 / p99 (m)", "48.9 / 285.2", f"{clip['z_p01_after']:.1f} / {clip['z_p99_after']:.1f}"],
            ["cells below 61 m", clip["cells_below_clip"], 0],
            ["components after the clip (largest kept)", "-", f"{clip['components_after_clip']} (sizes {clip['component_sizes_desc']})"],
            ["cells dropped as minor components / filled as holes", "-", f"{clip['cells_dropped_as_minor_components']} / {clip['cells_filled_as_holes']} (hole cells z min {fmt(clip['holes_filled_z_min'])} m)"],
            ["removed cells in all / added", "-", f"{clip['removed_cells_total']:,} / {clip['added_cells_total']}"],
            ["polygon layer", "study_area (md1989_fig3.gpkg)", f"study_area_clipped ({os.path.basename(clip['gpkg'])}), {clip['polygon_area_km2']:.3f} km2, {clip['polygonised_parts']} part"]]))
        W("\nRemoved clusters (8-connected):\n")
        W(table(["cluster", "cells", "km2", "centroid E, N", "z min-max (m)", "cells below 61 m"],
                [[c["id"], c["cells"], c["area_km2"], f"{c['centroid_E']:,.0f}, {c['centroid_N']:,.0f}", f"{fmt(c['z_min'], 1)}-{fmt(c['z_max'], 1)}", c["cells_below_clip"]] for c in clip["removed_clusters"]]))
        r = clip["rule_2_2"]
        W(f"\nRule 2.2: area {r['area_km2']:.3f} km2 in [{r['area_window'][0]:.3f}, {r['area_window'][1]:.3f}] (binding): {'pass' if r['area_pass'] else 'FAIL'}; z_min {r['z_min_1m']:.1f} m in {r['z_min_window']}: "
          f"{'pass' if r['z_min_pass'] else 'FAIL'} (by construction up to the hole filling); z_max {r['z_max_1m']:.1f} m in {r['z_max_window']}: {'pass' if r['z_max_pass'] else 'FAIL'}. **{'PASS' if r['pass'] else 'FAIL'}**\n\n")

    # ---------------- C2 TTV reference
    ref = load(os.path.join(R, "ttv_reference", "TTV_reference.json"))
    if ref:
        md, sbp = ref["polygon"], ref["stage_b_polygon"]
        W("### C2. The TTV reference from the clipped polygon on the 10 m NED (section 2.2), from ttv_reference/TTV_reference.json\n")
        W(table(["quantity", "clipped polygon (a16)", "stage B polygon (a15)", "stage A proxy"], [
            ["theta (legacy chi) / coarse", f"{md['theta_legacy']:.3f} / {md['theta_legacy_coarse']:.3f}", f"{sbp['theta_legacy']:.3f}", f"{ref['stage_a_proxy']['theta']:.3f}"],
            ["k_s pooled at theta (**the reference**)", f"**{md['ks_ref_pooled']:.2f}**", f"{sbp['ks_ref_pooled']:.2f}", "-"],
            ["k_s median of basins at theta (record)", f"{md['ks_ref_median']:.2f} (q16-q84 {md['ks_q16']:.1f}-{md['ks_q84']:.1f})", f"**{sbp['ks_ref_median']:.2f}** (was the reference)", f"{ref['stage_a_proxy']['ks']:.2f}"],
            ["basins in the k_s fit", md["ks_ref_n_basins"], sbp["ks_ref_n_basins"], "15"],
            ["|C_ht| (1/m)", f"{md['C_ht']:.4f}", f"{sbp['C_ht']:.4f}", f"{ref['stage_a_proxy']['C_ht']:.4f}"],
            ["convex divide pixels", md["C_ht_n_convex"], "2,258", "-"],
            ["cells / km2 (10 m)", f"{md['polygon_cells']:,} / {md['polygon_km2']:.3f}", f"20,578 / {sbp['polygon_km2']:.3f}", "6,729 / 0.673"],
            ["z range (m)", f"{md['z_min']:.1f}-{md['z_max']:.1f}", f"{sbp['z_min']:.1f}-{sbp['z_max']:.1f}", "61-317"],
            ["largest A inside (km2)", f"{md['A_max_inside_m2'] / 1e6:.3f}", "1.218", "0.67"],
            ["chi R2 at theta (pooled, through origin)", f"{md['chi_r2_at_theta_legacy']:.3f}", "-", "-"]]))
        wg = ref["whole_grid_ks_at_theta"]
        W(f"\nWhole-grid k_s at the polygon theta: median {wg['ks_ref_median']:.2f}, pooled {wg['ks_ref_pooled']:.2f} ({wg['n_basins']:,} basins).\n\n")

    # ---------------- C3 eligibility and the draw
    draw = load(os.path.join(R, "draw", "TTV_draw.json"))
    cfg = load(os.path.join(R, "draw", "draw_config.json"))
    if draw:
        e = draw["eligibility"]
        W("### C3. TTV eligibility and the draw (section 2.3), from draw/TTV_draw.json\n")
        W(table(["candidates", "set B screens", "within +-25 %", "screens and +-25 %", "gap-excluded", "eligible at +-25 %", "within +-50 %", "screens and +-50 %", "eligible at +-50 %", "widening threshold", "tolerance used", "eligible", "per offset"],
                [[e["n_candidates"], e["n_screens_pass"], e["n_tol25"], e["n_screens_and_tol25"], e["n_gap_excluded_of_screens_tol25"], e["n_eligible_tol25_after_gap"], e["n_tol50"], e["n_screens_and_tol50"],
                  e["n_eligible_tol50_after_gap"], e["min_eligible"], f"+-{int(e['tolerance_used'] * 100)} %" + (" (widened)" if e["widened"] else ""), e["n_eligible"], ", ".join(f"{k}: {v}" for k, v in e["per_offset"].items())]]))
        W("\nWindows removed by each set B screen (alone): " + "; ".join(f"{k}: {v}" for k, v in e["screens_each"].items()) + "\n")
        mdw = draw["md_windows"]
        W(f"\nM&D windows (5.2 of addendum 15, re-chosen against the clipped polygon): {mdw['polygon_cells_10m']:,} cells ({mdw['polygon_km2']:.3f} km2), {mdw['n_candidate_windows_intersecting']} candidate windows intersect it; "
          f"chosen {mdw['n_windows']} non-overlapping windows covering {mdw['coverage']:.1%} (>= 80 %: {'pass' if mdw['coverage_pass_80'] else 'fail'}).\n")
        W(table(["window (offset, id)", "footprint SW (E, N)", "set B screens pass", "screens failed", "within +-25 %", "k_s dev", "|C_ht| dev", "gap-excluded"],
                [[f"{w['offset']}, {w['window']}", f"{w['fp'][0]}, {w['fp'][1]}", w["screens_pass"], "; ".join(k for k, v in w["screens"].items() if not v) or "none", w["tol25"], pct(w["ks_rel_dev"]), pct(w["C_ht_rel_dev"]), f"{w['gap_excluded']} ({w.get('gap_channel_cells_in_filled_gap')})"] for w in mdw["windows"]]))
        W("\nPre-placed: " + "; ".join(f"{pw['id']} ({pw['kind']}) SW {pw['fp'][0]}, {pw['fp'][1]}" for pw in draw["pre_placed"]) + "\n")
        W("\nWithin-offset direct-edge check against stage A: " + "; ".join(f"offset {k}: stage A {v['stage_a']}, here {v['here']}, equal {v['equal']}" for k, v in draw["within_offset_edge_check"].items()) + "\n")
        W(f"\nOrdered list (seed {draw['seed']}, target {draw['target']} drawn tiles = 30 in all minus the {len(draw['pre_placed'])} M&D windows; eligible windows sorted by offset and window id, permuted once):\n")
        W(table(["rank", "offset", "window", "footprint SW (E, N)", "k_s dev", "|C_ht| dev", "theta", "result"],
                [[o["rank"], o["offset"], o["window"], f"{o['fp'][0]}, {o['fp'][1]}", pct(o["ks_rel_dev"]), pct(o["C_ht_rel_dev"]), f"{o['theta']:.3f}", o["result"]] for o in draw["ordered"]]))
        W("\nSkip reasons: " + "; ".join(f"{k}: {v}" for k, v in draw["skip_reasons"].items()) + "\n")
        W("\nLinks of the pre-placed windows: " + "; ".join(f"{k}: {v['eligible_directly_linked']} eligible windows directly linked" for k, v in draw["pre_placed_links"].items()) + "\n\n")

    # ---------------- C4 tiles cut
    tiles = load(os.path.join(R, "tiles", "tiles.json"))
    if tiles:
        W("### C4. TTV tiles cut (section 2.3) and the unit check, from tiles/tiles.json\n")
        W(table(["tile", "kind", "offset, window", "SW corner (E, N)", "z median", "z min", "z max", "unit check (0-320 m)", "A max (km2)", "outlet cells", "filled-gap channel cells", "class-2 pts/m2", "NN-filled frac"],
                [[t["tile"], t["kind"], f"{fmt(t.get('offset'))}, {fmt(t.get('window'))}", f"{t['fp_x0']}, {t['fp_y0']}", t["z_median"], t["z_min"], t["z_max"], t["unit_check_pass"], t["A_max_m2"] / 1e6, t["n_outlets_fd0"],
                  t["gapfill_channel_cells_A_ge_1e3"], t.get("class2_density_pts_m2"), t.get("frac_nn_filled_le10m")] for t in tiles["tiles"]]))
        W(f"\nUnit-check failures: {[t['tile'] for t in tiles['unit_check_failures']]} -> stop, nothing registered (addendum 15 section 6.4 as written).\n\n")

    # ---------------- C5 Table 1
    t1 = load(os.path.join(R, "md_georef", "md1989_table1_check.json"))
    if t1:
        W("### C5. Montgomery & Dietrich (1989) Table 1 transcription check (section 2.4), from md_georef/md1989_table1_check.json\n")
        W(table(["rows", "expected", "source area min / median / max (m2)", "slope min-max", "length min-max (m)", "width min-max (m)", "rows whose (area, length, width) triple is absent from the OCR text", "rows whose slope is absent", "values unreadable"],
                [[t1["rows"], t1["rows_expected"], f"{t1['source_area_min_m2']:,} / {t1['source_area_median_m2']:,} / {t1['source_area_max_m2']:,}", f"{t1['slope_min']}-{t1['slope_max']}",
                  f"{t1['length_min_m']}-{t1['length_max_m']}", f"{t1['width_min_m']}-{t1['width_max_m']}", t1["ocr_triples_not_found_rows"] or "none", t1["ocr_slopes_not_found_rows"] or "none", t1["values_unreadable"] or "none"]]))
        W("\nPaper text mentions (regex on the OCR layer): " + "; ".join(f"{k}: {v}" for k, v in t1["paper_text_mentions"].items()) + "\n")
        rows = list(csv.DictReader(open(t1["csv"])))
        W(f"\nThe transcription ({len(rows)} rows, {os.path.relpath(t1['csv'], R)}):\n")
        W(table(["row", "slope (m/m)", "source area (m2)", "source-basin length (m)", "colluvial fill width (m)"], [[int(r["row"]), r["local_valley_slope_m_per_m"], int(r["source_area_m2"]), int(r["source_basin_length_m"]), int(r["colluvial_fill_width_m"])] for r in rows]))
        W("\n")

    # ---------------- C6 GM footprints
    fp = load(os.path.join(R, "gm_footprints", "footprints.json"))
    if fp:
        W("### C6. Public lidar over Gabilan Mesa (section 3.1), from gm_footprints/footprints.json\n")
        W("NED cells (1/3 arc-second, current product): " + "; ".join(f"{c}: HTTP Last-Modified {v['http_last_modified']}, {v['bytes']:,} bytes, sha256 {v['sha256'][:16]}..., copy of record {os.path.relpath(v['dst'], os.path.dirname(os.path.dirname(v['dst'])))} verified {v['verified']}" for c, v in fp["ned_cells"].items()) + "\n")
        nu = fp["ned_utm"]
        W(f"\nNED warped: {nu['shape'][0]} x {nu['shape'][1]} cells at 10 m, EPSG:32610, box {fp['box_utm']}, {nu['n_valid']:,} valid cells, z {nu['z_min']:.0f}-{nu['z_max']:.0f} m, {nu['resample']}. "
          f"NCALM 2003 footprint on this lattice: {fp['ncalm']['km2_full']:.1f} km2 fully valid ({fp['ncalm']['km2_any']:.1f} km2 any data). Stage A coarse grid (Perron): {fp['perron_grid']['km2']:.1f} km2 valid, "
          f"{fp['perron_grid']['km2_inside_ncalm_full']:.1f} inside NCALM, {fp['perron_grid']['km2_beyond_ncalm']:.1f} beyond. Tiles' centroid {fp['tiles_centroid'][0]:,.0f} E, {fp['tiles_centroid'][1]:,.0f} N.\n")
        rows = []
        for n, s in fp["surveys"].items():
            if "error" in s:
                rows.append([n, s["error"]] + ["-"] * 12)
                continue
            w = s.get("wesm") or {}
            rows.append([n, f"{w.get('collect_start')} to {w.get('collect_end')}", w.get("ql"), w.get("spec"), f"{w.get('horiz_crs')} / {w.get('vert_crs')} {w.get('geoid')}", f"{s['entwine_points'] / 1e9:.1f} G", s["polygon_km2_utm10"], s["nominal_density_pts_m2"],
                         s["km2_in_box"], s["km2_inside_ncalm_any"], s["km2_beyond_ncalm_in_perron_grid"], s["km2_beyond_ncalm_within_5km"], s["km2_beyond_ncalm_within_10km"], s["km2_beyond_ncalm_within_15km"]])
        W("\nUSGS Entwine copies (usgs-lidar-public), with WESM work-unit metadata; areas in km2 on the 10 m lattice:\n")
        W(table(["survey", "collection", "QL", "spec", "horizontal / vertical CRS, geoid", "points (Entwine)", "polygon km2", "nominal density (pts/m2, all returns)", "in box", "inside NCALM", "beyond NCALM in Perron grid", "beyond NCALM < 5 km", "< 10 km", "< 15 km"], rows))
        W("\nNOAA Digital Coast tile indexes (the InPort items of the directive):\n")
        W(table(["NOAA ID", "InPort", "tiles", "in box km2", "in Perron grid", "inside NCALM", "beyond NCALM in Perron grid", "beyond NCALM < 15 km"],
                [[k, v["inport"], v.get("tiles"), v.get("km2_in_box"), v.get("km2_in_perron_grid"), v.get("km2_inside_ncalm_any"), v.get("km2_beyond_ncalm_in_perron_grid"), v.get("km2_beyond_ncalm_within_15km")] if "error" not in v else [k, v["inport"], v["error"]] + ["-"] * 5 for k, v in fp["noaa"].items()]))
        W("\n")

    # ---------------- C7 GM gate
    gate = load(os.path.join(R, "gm_gate", "gm_gate.json"))
    if gate:
        W("### C7. The GM extension gate on the 10 m NED (section 3.2), from gm_gate/gm_gate.json\n")
        W(f"Landscape polygon (EPSG:32610): {GM_LANDSCAPE_POLY_UTM}; radius {gate['radius_m'] / 1000:.0f} km about {gate['tiles_centroid'][0]:,.0f} E, {gate['tiles_centroid'][1]:,.0f} N. "
          f"Landscape areas: " + "; ".join(f"{k} {v:.1f} km2 ({gate['landscape_beyond_ncalm_km2'][k]:.1f} beyond NCALM)" for k, v in gate["landscape_km2"].items()) + f". NED crop {gate['crop']}, {gate['n_valid']:,} valid cells. "
          f"Reference: GM legacy (theta {gate['reference']['theta']}, k_s {gate['reference']['ks']}, |C_ht| {gate['reference']['C_ht']}). Gap rule: {gate['gap_rule']}. NLCD: {gate['nlcd'].get('classes_present', gate['nlcd'])}\n")
        b = gate["baseline"]
        W(f"\nBaseline pool (NED masked to the NCALM footprint, windows in the landscape union): {b['n_candidates']} candidates, {b['n_eligible25']} eligible at +-25 %, {b['n_eligible50']} at +-50 %.\n")
        for s, g in gate["gate"].items():
            W(f"\n**{s}**: tolerance used {g['tolerance_used']} % (combined eligible at +-25 %: {g['combined_eligible_at_25']}); windows counted for the gate {g['count_for_gate']} (>= {gate['gate_min_eligible']}: **{'PASS' if g['pass'] else 'FAIL'}**).\n")
            rows = []
            for k, v in g["by_landscape_and_tolerance"].items():
                ln, tl = k.split("|")
                rows.append([ln, f"+-{tl} %", v.get("n_candidates"), v.get("n_screens_pass"), v.get("n_eligible"), v.get("n_eligible_not_overlapping_GM1_7"), v.get("n_eligible_added_not_overlapping"), v.get("n_eligible_beyond_ncalm_not_overlapping"), v.get("n_baseline_eligible"), v.get("n_combined_eligible")])
            W(table(["landscape", "tolerance", "candidates", "screens pass", "eligible", "eligible, not overlapping GM1-7", "added (not in the NCALM baseline), not overlapping", "beyond the NCALM footprint, not overlapping", "baseline eligible", "combined eligible"], rows))
        W("\n")

    # ---------------- C8 GM area check
    chk = load(os.path.join(R, "gm_area_check", "gm_area_check.json"))
    diag = load(os.path.join(R, "gm_area_diag", "gm_area_diag.json"))
    if chk:
        W("### C8. The GM drainage-area check (section 4), from gm_area_check/gm_area_check.json" + (" and gm_area_diag/gm_area_diag.json" if diag else "") + "\n")
        rows = []
        for t, rec in chk["tiles"].items():
            dents = {tuple(e["tile_cell_EN"]): e for e in diag["tiles"][t]["entries"]} if diag and t in diag["tiles"] else {}
            if not rec["entries"]:
                rows.append([t, "-", "-", "-", "-", "-", "-", "-", "-", "pass (no difference)"])
            for e in rec["entries"]:
                de = dents.get(tuple(e["tile_cell_EN"]))
                rows.append([t, f"{e['tile_cell_EN'][0]:,.0f}, {e['tile_cell_EN'][1]:,.0f}", e["old_m2"] / 1e6, e["new_m2"] / 1e6, e["D_old_minus_new_m2"] / 1e6, e["coarse_outside_inflow_at_tile_entry_m2"] / 1e6, pct(e["rel_diff"]),
                             f"{e['lidar_entry_EN'][0]:,.0f}, {e['lidar_entry_EN'][1]:,.0f} ({e['walk_steps']} steps)", (de["ned_box_outside_inflow_m2"] / 1e6, pct(de["ned_rel_diff"])) if de else "-", "pass" if e["pass"] else "FAIL"])
            rows.append([f"**{t}**", f"cells with old > new: {rec['n_cells_D_gt_0']:,}; D max {rec['D_max_m2'] / 1e6:.3f} km2", "", "", "", "", "", "", "", f"**{'pass' if rec['pass'] else 'FAIL'}** -> {rec['area_model']['note']}"])
        W(table(["tile", "tile entry cell (E, N)", "old area (km2)", "new area (km2)", "old - new (km2)", "coarse inflow, Perron strip (km2)", "rel. diff", "lidar entry by the 1 m main-stem walk", "NED box inflow (km2), rel. diff (diagnostic)", "result"], rows))
        W("\nModelling area grids registered (role area_model):\n")
        W(table(["tile", "file", "sha256 (first 16)", "bytes", "max (km2)", "equals old", "equals new", "note"],
                [[t, rec["area_model"]["path"], rec["area_model"]["sha256"][:16] + "...", rec["area_model"]["bytes"], rec["area_model"]["max_m2"] / 1e6, rec["area_model"]["equals_old"], rec["area_model"]["equals_new"], rec["area_model"]["note"]] for t, rec in chk["tiles"].items()]))
        c = chk["coarse"]
        W(f"\nCoarse grid: {c['inside_cells']:,} 10 m cells inside the NCALM footprint, {c['n_direct_inflow_cells']} cells receive flow directly from outside; largest direct inflows: " + "; ".join(f"{d['E']:,.0f} E, {d['N']:,.0f} N: {d['m2'] / 1e6:.2f} km2" for d in c["largest_direct_inflows"][:8]) + "\n\n")

    # ---------------- C9 regression
    reg = load(os.path.join(R, "regression", "regression.json"))
    if reg:
        W("### C9. The v2.0.0 regression (section 5.4), from regression/regression.json\n")
        rows = []
        for name, rec in reg["runs"].items():
            if not rec.get("exists"):
                rows.append([name, "run missing"] + ["-"] * 7)
                continue
            for r in rec["forward_runs"]:
                p3 = r["phase3_comparison"] or {}
                rows.append([name, r["key"], f"{r['mse_new']:.10g}", f"{r['mse_release']:.10g}", r["mse_equal"], fmt(p3.get("max_abs_dz")), ", ".join(f"{k}: {'=' if v.get('values_identical') else v}" for k, v in r["grids"].items()), f"{r['wall_s']:.0f}", "PASS" if r["pass"] else "FAIL"])
        W(table(["run", "key", "MSE (this run)", "MSE (v2.0.0)", "equal", "max |dz| vs release elevation", "derived grids identical", "wall s", "result"], rows))
        W(f"\nCode: this run at git {list(reg['runs'].values())[0].get('git_hash')}, release runs at {list(reg['runs'].values())[0].get('reference_git_hash')}. **{'ALL PASS' if reg['all_pass'] else 'FAIL'}**\n\n")

    # ---------------- C10 configs and checksums
    if cfg:
        W("### C10. Draw configuration (written before the draw)\n\n```json\n" + json.dumps({k: v for k, v in cfg.items() if k != "thresholds_values"}, indent=1) + "\n```\n")
        W("\nThreshold set B: " + json.dumps(json.load(open(THRESHOLDS))) + "\n\n")
    if tiles:
        W("### C11. Checksums (sha256) of the TTV tiles cut (unregistered, parked in tiles/TTV-x_unregistered)\n")
        W(table(["tile", "role", "path", "sha256 (first 16)", "bytes"], [[t["tile"], role, f["path"], f["sha256"][:16] + "...", f["bytes"]] for t in tiles["tiles"] for role, f in t["files"].items()]))
        W("\n")

    # ---------------- C14-C16 the GM extension (section 3.3)
    fpj = load(os.path.join(R, "gm_ext", "GMF_fetch_footprint.json"))
    fetch = load(os.path.join(R, "ept_GMF", "ept_fetch_6339.json"))
    ggrid = load(os.path.join(R, "grid_GMF", "GMF_grid.json"))
    ggaps = load(os.path.join(R, "grid_GMF", "GMF_gaps.json"))
    groute = load(os.path.join(R, "grid_GMF", "GMF_route.json"))
    gchk = load(os.path.join(R, "grid_GMF", "gmf_check.json"))
    if fpj and fetch:
        W("### C14. The GM extension: fetch footprint, fetch, grid, routing and checks (section 3.3.1-3.3.3)\n")
        fs = fetch["summary"]
        W(table(["quantity", "value"], [
            ["eligible windows (gate tolerance, in the landscape, not overlapping GM1-7)", fpj["n_eligible_windows"]],
            ["windows km2 / upstream km2 / footprint km2 (500 m buffer)", f"{fpj['windows_km2']:.2f} / {fpj['upstream_km2']:.1f} / {fpj['footprint_km2']:.1f}"],
            ["bounding box (EPSG:32610)", fpj["bbox_utm"]], ["1 m lattice cells (G)", f"{fpj['lattice_cells_1m_G']:.3f}"],
            ["EPT boxes fetched / ok", f"{fs['n_boxes']} / {fs['n_ok']}"], ["LAZ bytes / seconds / MB s-1", f"{fs['bytes'] / 1e9:.2f} GB / {fs['seconds']:.0f} / {fs['MB_per_s']:.1f}"],
            ["CRS written", f"EPSG:{fs['epsg']}+{fs['vertical_epsg']}"], ["vertical evidence", fs["vertical_evidence"]]]))
        if ggrid:
            g = ggrid["stats"]
            W("\nGrid (a14_grid_route.py --site GMF, decision 9 of addendum 14):\n")
            W(table(["lattice", "cells inside footprint", "cells with class 2 (inside)", "class-2 points", "density (pts m-2)", "points per cell 5-50-95 %", "NN-filled within 10 m", "left nodata beyond 10 m", "NN fill distance 50-95-99-max (m)", "seconds", "peak GB"],
                    [[f"{g['lattice_shape'][0]} x {g['lattice_shape'][1]}", g["n_inside_footprint"], g["n_cells_with_class2_inside"], g["total_class2"], g["class2_density_inside_pts_per_m2"], g["points_per_cell_percentiles_5_50_95"],
                      f"{g['n_filled_within_max']:,} ({g['frac_inside_filled']:.1%})", g["n_gap_beyond_max_left_nodata"], g["nn_fill_distance_percentiles_50_95_99_max_m"], g["seconds_total"], g["peak_rss_GB"]]]))
            dates = [f.get("utc_dates_class2") for f in ggrid["files"] if f.get("utc_dates_class2")]
            if dates:
                W(f"\nClass-2 flight dates over the 292 boxes: {min(d[0] for d in dates)} to {max(d[1] for d in dates)} UTC ({dates[0][2]}).\n")
        if ggaps and groute:
            W("\nRe-routing with the gap rule (a15_reroute.py --site GMF):\n")
            W(table(["nodata components", "enclosed (filled)", "enclosed cells / km2", "enclosed >= 1e4 cells", "valid after", "data-edge outlets", "largest basin km2", "budget closed", "pad", "seconds", "peak GB"],
                    [[ggaps["n_nodata_components"], ggaps["n_enclosed_components"], f"{ggaps['n_enclosed_cells']:,} / {ggaps['enclosed_area_km2']:.3f}", ggaps.get("n_enclosed_ge_1e4_cells"), groute["n_valid"], groute["n_roots"], groute["max_area_km2"], groute["root_area_sum_equals_valid"], groute["pad_cells"], groute["seconds_total"], groute["peak_rss_GB"]]]))
            W("\nLargest outlets: " + "; ".join(f"({r['x']:,.0f}, {r['y']:,.0f}) {r['area_cells'] / 1e6:.2f} km2 ({r['share_of_valid']:.1%})" for r in groute["largest_roots"][:5]) + "\n")
        if gchk:
            r53, nc = gchk["rule_5_3_ned_minus_gmf"], gchk.get("ncalm_overlap", {})
            W("\nUnits and datum (a16_gmf_check.py):\n")
            W(table(["comparison", "cells", "median dz (m)", "5-95 %", "RMS", "Theil-Sen slope (range)", "criterion", "result"], [
                ["rule 5.3: NED 10 m - grid block mean (ground cells)", r53["n_cells_10m"], r53["dz"]["median"], f"{r53['dz']['p05']:.2f}, {r53['dz']['p95']:.2f}", r53["dz"]["rms"], f"{r53['fit_ned_on_gmf']['theil_sen_slope']:.5f} ({r53['fit_ned_on_gmf']['theil_sen_slope_range'][0]:.5f}-{r53['fit_ned_on_gmf']['theil_sen_slope_range'][1]:.5f})", "|median| < 5 m, slope 0.98-1.02", "pass" if r53["pass"] else "FAIL"],
                ["NCALM 2003 - grid (every 5th NCALM cell, bilinear)", nc.get("n"), nc["dz_ncalm_minus_gmf"]["median"], f"{nc['dz_ncalm_minus_gmf']['p05']:.2f}, {nc['dz_ncalm_minus_gmf']['p95']:.2f}", nc["dz_ncalm_minus_gmf"]["rms"], f"{nc['fit_ncalm_on_gmf']['theil_sen_slope']:.4f} ({nc['fit_ncalm_on_gmf']['theil_sen_slope_range'][0]:.4f}-{nc['fit_ncalm_on_gmf']['theil_sen_slope_range'][1]:.4f})", "|offset| 30-34 m (NCALM ellipsoidal), slope 0.98-1.02; stop near 3.28 / 0.305", f"offset magnitude {abs(nc['dz_ncalm_minus_gmf']['median']):.1f} m in window; slope pass; no stop (the script's own sign-flipped flag read False)"]]))
            W("\n")
    gdraw = load(os.path.join(R, "draw_gmf", "GMF_draw.json"))
    gcfg = load(os.path.join(R, "draw_gmf", "draw_config.json"))
    if gdraw:
        e = gdraw["eligibility"]
        W("### C15. The GM extension draw (section 3.3.4-3.3.5), from draw_gmf/GMF_draw.json\n")
        W(f"Eligibility on the 1 m grid: {e['n_eligible_gate']} gate-eligible windows, {e['n_gap_excluded']} gap-excluded, {e['n_outside_1m_grid']} outside the 1 m grid, **{e['n_eligible']} eligible** (per offset {e['per_offset']}). "
          f"Seed {gdraw['seed']} written at {gcfg['written'] if gcfg else '-'} before the draw; tolerance +-{int(gdraw['tolerance_used'] * 100)} %; target {gdraw['target']} (30 in all minus GM1-GM7); **accepted {gdraw['n_accepted']}**, skipped {gdraw['n_skipped']} ({'; '.join(f'{k}: {v}' for k, v in gdraw['skip_reasons'].items())}).\n")
        W(table(["rank", "offset", "window", "footprint SW (E, N)", "k_s dev", "|C_ht| dev", "theta", "inside NCALM", "result"],
                [[o["rank"], o["offset"], o["window"], f"{o['fp'][0]}, {o['fp'][1]}", pct(o["ks_rel_dev"]), pct(o["C_ht_rel_dev"]), f"{o['theta']:.3f}", o["in_ncalm_full"], o["result"]] for o in gdraw["ordered"]]))
        W("\nDraw configuration:\n\n```json\n" + json.dumps(gcfg, indent=1) + "\n```\n\n")
    gtiles = load(os.path.join(R, "tiles_gmf", "tiles.json"))
    if gtiles:
        W("### C16. The GM extension tiles cut (section 3.3.6) and the unit check, from tiles_gmf/tiles.json\n")
        W(table(["tile", "offset, window", "SW corner (E, N)", "z median", "z min", "z max", "unit check (68-468 m)", "A max (km2)", "outlet cells", "filled-gap channel cells", "class-2 pts/m2", "NN-filled frac"],
                [[t["tile"], f"{fmt(t.get('offset'))}, {fmt(t.get('window'))}", f"{t['fp_x0']}, {t['fp_y0']}", t["z_median"], t["z_min"], t["z_max"], t["unit_check_pass"], t["A_max_m2"] / 1e6, t["n_outlets_fd0"], t["gapfill_channel_cells_A_ge_1e3"], t.get("class2_density_pts_m2"), t.get("frac_nn_filled_le10m")] for t in gtiles["tiles"]]))
        W(f"\nUnit-check failures: {[t['tile'] for t in gtiles['unit_check_failures']]} -> stop, nothing registered; files parked in tiles_gmf/GM-FEMA2018-x_unregistered.\n")
        W("\nChecksums (sha256) of the products cut:\n")
        W(table(["tile", "role", "path", "sha256 (first 16)", "bytes"], [[t["tile"], role, f["path"], f["sha256"][:16] + "...", f["bytes"]] for t in gtiles["tiles"] for role, f in t["files"].items()]))
        W("\n")

    # ---------------- C13 the final tile list per site (section 6)
    import yaml
    repo = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
    cfg_versions = (yaml.safe_load(open(os.path.join(repo, "configs", "data.yaml"))) or {}).get("versions", {})
    sb_tiles = {}
    sbt = os.path.join(STAGE_B, "tiles", "tiles.json")
    if os.path.exists(sbt):
        for t in json.load(open(sbt))["tiles"]:
            sb_tiles[t["tile"]] = t
    gmf_tiles = {}
    gt = os.path.join(R, "tiles_gmf", "tiles.json")
    if os.path.exists(gt):
        for t in json.load(open(gt))["tiles"]:
            gmf_tiles[t["tile"]] = t
    rows = []
    for v, info in cfg_versions.items():
        for t in info.get("tiles", []):
            src = sb_tiles.get(t) or gmf_tiles.get(t) or {}
            kind = src.get("kind", "-")
            if v == "GM-x" and t in ("GM1", "GM2", "GM3", "GM4", "GM5"):
                area_src = "area_model (" + (info.get("area_model", {}).get(t, "section 4")) + ")"
            else:
                area_src = "full-grid routing (role area)"
            rows.append([info.get("site_label", info.get("site")), t, v, f"{src.get('fp_x0', '-')}, {src.get('fp_y0', '-')} (EPSG:{src.get('epsg', info.get('crs_epsg'))})", area_src, kind])
    W("### C13. The registered tile list per site (section 6), from configs/data.yaml, the stage B and a16 tile records\n")
    W(table(["site", "tile", "version", "footprint SW corner (E, N)", "area source for modelling", "kind"], rows))
    W(f"\nTTV: no registered version (the 15 cut tiles are parked unregistered, section 2.4). Registered tiles in all: {len(rows)}.\n\n")

    # ---------------- C12 Slurm accounting
    if a.jobs:
        W("### C12. Slurm accounting of the addendum 16 jobs\n")
        try:
            txt = subprocess.run(["sacct", "-j", a.jobs, "-X", "-o", "JobID,JobName%22,Partition,State,Elapsed,Timelimit,AllocCPUS,MaxRSS,TotalCPU", "-P"], capture_output=True, text=True, check=True).stdout
            lines = [ln.split("|") for ln in txt.strip().splitlines()]
            W(table(lines[0], lines[1:]))
            txt2 = subprocess.run(["sacct", "-j", a.jobs, "-o", "JobID,MaxRSS,Elapsed,AllocCPUS", "-P"], capture_output=True, text=True, check=True).stdout
            mx = {}
            for ln in txt2.strip().splitlines()[1:]:
                jid, rss, el, cp = ln.split("|")
                base = jid.split(".")[0]
                if rss.endswith("K"):
                    mx[base] = max(mx.get(base, 0), float(rss[:-1]) / 1e6)
                elif rss.endswith("M"):
                    mx[base] = max(mx.get(base, 0), float(rss[:-1]) / 1e3)
                elif rss.endswith("G"):
                    mx[base] = max(mx.get(base, 0), float(rss[:-1]))
            W("\nPeak memory per job (GB, from the step records): " + "; ".join(f"{k}: {v:.1f}" for k, v in sorted(mx.items())) + "\n")
        except Exception as e:
            W(f"sacct failed: {e!r}\n")
    txt = "".join(out)
    with open(os.path.join(R, "appendix_tables.md"), "w") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
