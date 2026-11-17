#!/usr/bin/env python3
"""Addendum 13 report appendices: markdown tables from the a13 run outputs.

usage: make_a13_tables.py --inventory-run DIR --las-run DIR --pasda-check JSON --source DIR > appendix.md
"""
import argparse, glob, json, os


def fmt(x, nd=3):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}" if abs(x) < 1e6 else f"{x:.4g}"
    return str(x)


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(fmt(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def main():
    p = argparse.ArgumentParser(); p.add_argument("--inventory-run", required=True); p.add_argument("--las-run"); p.add_argument("--pasda-check", required=True); p.add_argument("--source", required=True)
    a = p.parse_args()
    inv = json.load(open(os.path.join(a.inventory_run, "inventory", "inventory.json")))
    print("## C1. Inventory of `lem_topography` (section 2.1), from run `%s`\n" % os.path.basename(a.inventory_run))
    rows = []
    for rel, r in inv.items():
        crs = r.get("crs", {}); st = r.get("stats", {})
        if r.get("format", "").startswith("LAS"):
            rows.append([rel, r["bytes"], r["sha256"][:16], r["format"] + f" pf{r['point_format']}", "not stated (no VLR)", "not stated", "not stated", "-",
                         f"x {r['mins'][0]:.2f}-{r['maxs'][0]:.2f}, y {r['mins'][1]:.2f}-{r['maxs'][1]:.2f}", "-", f"z {r['mins'][2]:.2f}-{r['maxs'][2]:.2f}"])
        elif "res" in r:
            rows.append([rel, r["bytes"], r["sha256"][:16], r["format"] + " " + r.get("dtype", ""), f"{crs.get('name')} (EPSG:{crs.get('epsg')}); datum {crs.get('datum')}", crs.get("horizontal_units"),
                         crs.get("vertical"), f"{r['res'][0]:.6g} x {r['res'][1]:.6g}", f"{r['bounds'][0]:.3f}, {r['bounds'][1]:.3f}, {r['bounds'][2]:.3f}, {r['bounds'][3]:.3f}",
                         fmt(r.get("nodata")), f"{st.get('min'):.2f}-{st.get('max'):.2f} (valid {100*st.get('frac_valid', 0):.1f} %)" if st else "-"])
        else:
            rows.append([rel, r["bytes"], r["sha256"][:16], r.get("format"), "-", "-", "-", "-", "-", "-", "-"])
    print(table(["file", "bytes", "sha256 (16)", "format", "CRS (header)", "horizontal units", "vertical datum/units", "resolution", "extent (xmin, ymin, xmax, ymax)", "nodata", "value range"], rows))
    for rel, r in inv.items():
        if r.get("format", "").startswith("LAS"):
            print(f"**LAS detail, `{rel}`:** points {r['point_count']:,}; density {r['density_per_header_unit2']:.3f} per header unit²; classes {r['classes']}; return numbers {r['return_number']}; "
                  f"numbers of returns {r['number_of_returns']}; GPS time: {r['gps_time_range']}; generating software {r['generating_software']}; VLRs {r['n_vlrs']}; class z ranges {r['class_z_range']}.\n")
    gm = json.load(open(os.path.join(a.inventory_run, "grid_match", "grid_match.json")))
    print("## C2. Tiles against the parent grids (section 2.2.1-2.2.2)\n")
    rows = []
    for t, r in gm.items():
        o = r["origin_offset_parent_pixels"]; n = r["nearest"]; b = r["bilinear"]; w = n.get("where_abs_dz_gt_0.1m", {})
        rows.append([t, r["parent"].split("/")[-1], "yes" if r["tile_inside_parent"] else "no", f"{o['col_frac']:.3f}, {o['row_frac']:.3f}", f"{r['res_ratio_tile_over_parent'][0]:.6f}",
                     r.get("parent_units_inferred", "").split(" (")[0], n["n_compared"], n["n_exact_equal"], n["n_abs_dz_gt"]["0.01"], n["n_abs_dz_gt"]["0.1"], n["n_abs_dz_gt"]["1.0"], fmt(n["max_abs_dz_m"]),
                     f"{w.get('sign', {}).get('tile_higher', 0)} / {w.get('sign', {}).get('parent_higher', 0)}", fmt(b["median_dz_m"], 4), f"{b['p05_dz_m']:.3f} / {b['p95_dz_m']:.3f}", fmt(b["max_abs_dz_m"])])
    print(table(["tile", "parent", "inside", "origin offset (frac. parent px: col, row)", "res ratio tile/parent", "parent z units (inferred)", "n cells", "exact equal (nearest)", "n abs(dz)>0.01 m", "n abs(dz)>0.1 m", "n abs(dz)>1 m",
                 "max abs(dz) nearest (m)", "abs(dz)>0.1: tile higher / parent higher", "median dz bilinear", "p05 / p95 bilinear", "max abs(dz) bilinear"], rows))
    rows = [[t, fmt(r["margin_m"]["west"], 0), fmt(r["margin_m"]["east"], 0), fmt(r["margin_m"]["south"], 0), fmt(r["margin_m"]["north"], 0), r.get("parent_nodata_within_tile_window"),
             f"{r.get('parent_nodata_within_1km_buffer')} / {r.get('parent_cells_within_1km_buffer')}"] for t, r in gm.items()]
    print(table(["tile", "margin W (m)", "margin E", "margin S", "margin N", "parent nodata in tile window", "parent nodata / cells within 1 km"], rows))
    po = os.path.join(a.inventory_run, "parent_overview", "parent_nodata_around_tiles.json")
    if os.path.exists(po):
        pn = json.load(open(po)); rows = []
        for t, r in pn.items():
            if not t.startswith("AP"):
                continue
            rows.append([t] + [fmt(r[f"buffer_{b}m"]["frac_nodata"], 3) for b in (500, 1000, 2000, 5000)] + [", ".join(f"{k.split('_')[0]} {v:.2f}" for k, v in r["buffer_2000m"]["nodata_by_side"].items())])
        print(table(["tile", "nodata frac within 500 m", "1 km", "2 km", "5 km", "2 km window, nodata frac by half (N, S, W, E)"], rows))
    ttv = json.load(open(os.path.join(a.inventory_run, "ttv", "ttv.json")))
    print("## C3. TTV LAS (section 2.2.3)\n")
    rows = [["header bounds (x, y, z)", f"{ttv['header']['mins']} to {ttv['header']['maxs']}"], ["name = floor(header min x, y)", ttv["name_check"]["name_equals_floor_of_header_min"]],
            ["lon/lat under EPSG:32610", f"lon {ttv['geographic_bounds']['32610']['lon']}, lat {ttv['geographic_bounds']['32610']['lat']}; inside Point Bonita quad: {ttv['geographic_bounds']['32610']['inside_point_bonita_quad']}"],
            ["lon/lat under EPSG:26910", f"lon {ttv['geographic_bounds']['26910']['lon']}, lat {ttv['geographic_bounds']['26910']['lat']}"],
            ["classes", ttv["classes"]], ["return number / number of returns", f"{ttv['return_number']} / {ttv['number_of_returns']}"], ["intensity range, scan angle range", f"{ttv['intensity_range']}, {ttv['scan_angle_range']}"],
            ["density, all points (per m² of bbox)", fmt(ttv["density_all_per_m2_if_metres"])], ["density, class 2 (per m² of bbox)", fmt(ttv["density_ground_per_m2_if_metres"])],
            ["class-2 z percentiles (0.01, 0.1, 1, 5, 50, 95, 99, 99.9 %)", ", ".join(f"{v:.2f}" for v in ttv["z_percentiles_used_points"].values())],
            ["class-2 points below 2 m: n, location", f"{ttv['n_points_below_2m']}; x {ttv.get('low_points_location_utm', {}).get('x')}, y {ttv.get('low_points_location_utm', {}).get('y')}"],
            ["fraction of class-2 points in 61-314 m", fmt(ttv["frac_used_points_in_61_314m"])],
            ["1 m cells with >= 1 class-2 point", fmt(ttv["grid"]["frac_1m_cells_with_ground_point"])], ["2 m cells with >= 1 class-2 point", fmt(ttv["grid"]["frac_cells_with_used_point"])]]
    for k in ("density_used_points_all", "density_used_points_hollow_lap_gt_0.02", "density_used_points_divergent_lap_lt_-0.02"):
        v = ttv[k]; rows.append([k.replace("density_used_points_", "class-2 density, ") + " (pts/m², occupied 2 m cells)", f"{fmt(v['points_per_m2'])} over {v['n_cells']} cells ({100*v['frac_of_cells']:.1f} %); empty before fill {fmt(v['frac_cells_empty_before_fill'])}"])
    print(table(["quantity", "value"], rows))
    chk = json.load(open(a.pasda_check))
    print("## C4. PASDA candidate files (section 3)\n")
    rows = [[t, v["epsg"], ", ".join(f"{x:.1f}" for x in v["bounds_src"]), ", ".join(f"{x:.0f}" for x in v["sp_south_ftus_buffered"]), len(v["files"]), ", ".join(f.replace("_LAS.zip", "") for f in v["files"])] for t, v in chk["per_tile"].items()]
    print(table(["tile", "EPSG", "bounds in tile CRS (m)", "buffered footprint, SP South ftUS (E0, N0, E1, N1)", "n files", "files"], rows))
    shas = {}
    for mf in glob.glob(os.path.join(a.source, "AP-3DEP", "las", "MANIFEST*.sha256")) + glob.glob(os.path.join(a.source, "AP-3DEP", "las", "*.sha256")):
        for line in open(mf):
            parts = line.split()
            if len(parts) == 2:
                shas[os.path.basename(parts[1])] = parts[0]
    raw = {}
    rp = os.path.join(a.source, "AP-3DEP", "las", "MANIFEST_zips.sha256")
    if os.path.exists(rp):
        for line in open(rp):
            parts = line.split()
            if len(parts) == 2:
                raw[os.path.basename(parts[1])] = parts[0]
    dl = {}
    lp = os.path.join(a.source, "AP-3DEP", "las", "download_log.txt")
    if os.path.exists(lp):
        for line in open(lp):
            parts = line.split()
            if len(parts) >= 4:
                dl[parts[0]] = (parts[1], parts[2], parts[3])
    rows = []
    for r in chk["files"]:
        f = r["file"]; d = dl.get(f, ("-", "-", "-")); las = f.replace("_LAS.zip", "")
        lasname = next((k for k in shas if k.startswith(las) and k.lower().endswith((".las", ".laz"))), None)
        rows.append([f, ",".join(r["tiles"]), r["bytes"], "yes" if r["in_index"] else "no", "yes" if r.get("index_matches_rule_within_1ft") else "no", d[0], d[1], d[2], raw.get(f, "-")[:16], (shas.get(lasname, "-") if lasname else "-")[:16]])
    print(table(["zip", "tiles", "listed bytes", "in index", "index = rule", "HTTP", "received bytes", "wall", "sha256 zip (16)", "sha256 LAS (16)"], rows))
    print(json.dumps(chk["summary"], indent=1))
    if a.las_run and os.path.exists(os.path.join(a.las_run, "stage1_files.json")):
        s1 = json.load(open(os.path.join(a.las_run, "stage1_files.json")))
        print("\n## C5. Per-file header checks (section 5, rules 5.1-5.2), run `%s`\n" % os.path.basename(a.las_run))
        rows = []
        for k, r in s1.items():
            h = r["header"]; c = r.get("crs", {}); l = r["laspy"]; pu = (r.get("pdal", {}).get("srs_units") or {}); ex = r["rule_5_1"]["excess_ft"]
            rows.append([k, h["version"], h["point_format"], f"{h['point_count']:,}", c.get("name"), c.get("horizontal_units"), f"{c.get('vertical_name')} / {c.get('vertical_units')}", f"{pu.get('horizontal')} / {pu.get('vertical')}",
                         f"{h['mins'][0]:.1f}-{h['maxs'][0]:.1f}, {h['mins'][1]:.1f}-{h['maxs'][1]:.1f}", f"{h['mins'][2]:.1f}-{h['maxs'][2]:.1f}", ", ".join(f"{v:.1f}" for v in ex.values()),
                         "pass" if r["rule_5_1"]["pass"] else "FAIL", "pass" if r["rule_5_2"]["pass"] else "FAIL", l["classes"], f"{l['n_class2']:,}", fmt(l["density_class2_per_header_unit2"] / (1200 / 3937) ** 2 if l["density_class2_per_header_unit2"] else None, 2),
                         " to ".join(x[:10] for x in l.get("utc_range", ["-", "-"]))])
        print(table(["file", "LAS", "pf", "points", "CRS name (header)", "horiz. units", "vertical CRS / units", "pdal units h / v", "header x, y bounds (ftUS)", "z (ftUS)", "W, S, E, N excess over quadrant (ft)", "5.1", "5.2",
                     "classes", "class 2", "class-2 density (pts/m²)", "UTC dates"], rows))
    if a.las_run:
        rows = []
        for f in sorted(glob.glob(os.path.join(a.las_run, "*_pdal_stats_class2.json"))):
            d = json.load(open(f)); st = d.get("stages", d.get("metadata", {})).get("filters.stats", {}).get("statistic", [])
            m = {x["name"]: x for x in st}
            if m:
                rows.append([os.path.basename(f).replace("_pdal_stats_class2.json", ""), f"{m['Z']['count']:,}", f"{m['X']['minimum']:.1f}-{m['X']['maximum']:.1f}", f"{m['Y']['minimum']:.1f}-{m['Y']['maximum']:.1f}",
                             f"{m['Z']['minimum']:.1f}-{m['Z']['maximum']:.1f}", fmt(m['Z']['average'], 1), f"{m['Intensity']['minimum']:.0f}-{m['Intensity']['maximum']:.0f}", f"{m['ReturnNumber']['minimum']:.0f}-{m['ReturnNumber']['maximum']:.0f}",
                             f"{m['GpsTime']['minimum']:.1f}-{m['GpsTime']['maximum']:.1f}"])
        if rows:
            print("\n## C5b. PDAL class-2 statistics per file (`pdal pipeline`: readers.las, filters.range Classification[2:2], filters.stats; header units, ftUS)\n")
            print(table(["file", "class-2 count", "X", "Y", "Z (ftUS)", "mean Z", "intensity", "return number", "GpsTime (adjusted standard GPS s)"], rows))
    if a.las_run and os.path.exists(os.path.join(a.las_run, "stage2_tiles.json")):
        s2 = json.load(open(os.path.join(a.las_run, "stage2_tiles.json")))
        print("\n## C6. Per-tile agreement and coverage (rules 5.3-5.5)\n")
        print(f"Transformer: {s2['transformer']}; buffer {s2['buffer_m']} m; 1 ftUS = {s2['ft_us_in_m']:.12f} m.\n")
        rows = []
        for t, r in s2["tiles"].items():
            x = r["rule_5_3"]
            rows.append([t, f"{x['n_points']:,}", fmt(x["median_dz_m"]), f"{x['p05_dz_m']:.3f} / {x['p95_dz_m']:.3f}", fmt(x["rms_dz_m"]), fmt(x["theil_sen_slope_median_of_7x4000"], 4),
                         f"{min(x['theil_sen_slopes']):.4f}-{max(x['theil_sen_slopes']):.4f}", fmt(x["ols_slope"], 4), fmt(x["frac_abs_dz_gt_2m"], 4), fmt(x["frac_abs_dz_gt_1m"], 4), fmt(x["frac_cells_with_abs_dz_gt_2m"], 4),
                         "pass" if x["pass"] else "FAIL"])
        print(table(["tile", "class-2 points in tile", "median dz (m)", "p05 / p95 dz", "rms dz", "Theil-Sen slope (median of 7 x 4000)", "slope range", "OLS slope", "frac abs(dz)>2 m", "frac abs(dz)>1 m", "frac tile cells with abs(dz)>2 m", "5.3"], rows))
        rows = []
        for t, r in s2["tiles"].items():
            bc = r["rule_5_3"].get("by_class", {})
            rows.append([t] + [f"{bc[k]['n']:,} / {fmt(bc[k]['median_dz'])} / {fmt(bc[k]['frac_abs_dz_gt_2m'], 4)} / {fmt(bc[k]['share_of_gt2m_points'], 3)}" for k in ("divergent", "hollow", "hilltop", "channel", "edge_or_unclassed")])
        print(table(["tile", "divergent: n / median dz / frac >2 m / share of >2 m points", "hollow", "hilltop (option iii)", "channel", "edge or unclassed"], rows))
        rows = []
        for t, r in s2["tiles"].items():
            c = r["rule_5_4"]
            rows.append([t, fmt(c["frac_1m_cells_ge1_class2"], 4), fmt(c["frac_1m_cells_zero_class2"], 4), fmt(c["density_class2_overall_per_m2"], 3), fmt(c["density_class2_in_tile_per_m2"], 3),
                         fmt(c.get("density_class2_hilltop_option_iii_per_m2"), 3), c.get("n_cells_hilltop_option_iii"), fmt(c.get("frac_cells_zero_hilltop_option_iii"), 3), fmt(c.get("density_class2_hollow_per_m2"), 3), c.get("n_cells_hollow"),
                         fmt(c.get("frac_cells_zero_hollow"), 3), fmt(c.get("density_class2_divergent_per_m2"), 3), fmt(c.get("density_class2_channel_per_m2"), 3), fmt(c["frac_tile_cells_zero_class2"], 4)])
        print(table(["tile", "1 m cells with >=1 class-2 (buffered)", "with 0", "class-2 density, buffered (pts/m²)", "in tile", "on hilltop (option iii)", "n hilltop cells", "hilltop cells with 0", "in hollows", "n hollow cells", "hollow cells with 0", "divergent", "channel", "tile cells with 0"], rows))
        rows = [[t, r.get("rule_5_5", {}).get("utc_min", "-")[:19], r.get("rule_5_5", {}).get("utc_max", "-")[:19], r.get("rule_5_5", {}).get("n_distinct_days", "-"), ", ".join(f"{d} ({n:,})" for d, n in r.get("rule_5_5", {}).get("days", {}).items())] for t, r in s2["tiles"].items()]
        print(table(["tile", "first class-2 point (UTC)", "last", "distinct days", "days (points)"], rows))


if __name__ == "__main__":
    main()
