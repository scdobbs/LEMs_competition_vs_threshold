#!/usr/bin/env python3
"""Addendum 13, section 5: header, unit, agreement and coverage checks of the USGS 3DEP (PASDA LiDAR2020 PA_South) LAS files.

Stage 1, per file: `pdal info --metadata` (saved), a PDAL pipeline readers.las -> filters.range Classification[2:2] -> filters.stats
(saved; the class-2 stats), and one laspy pass that records the header, the CRS (WKT, horizontal and vertical units), the class
and return counts, the GPS-time range and caches the class-2 points (x, y, z in header units, gps_time).
  Rule 5.1 (naming): header bounds inside the quadrant that the name implies, within 50 ft on each side.
  Rule 5.2 (units): horizontal and vertical units must both be US survey feet, as the metadata say; nothing is converted on an
  assumption; the conversion used is 1 ftUS = 1200/3937 m (the international foot 0.3048 differs by 2 ppm).
Stage 2, per AP tile:
  Rule 5.3 (agreement): class-2 points inside the tile, transformed to the tile CRS (EPSG:26917, metres) with pyproj (the datum
  step that pyproj applies is recorded), z to metres, compared with the PAMAP-based filled_m grid by bilinear lookup: median dz,
  5th-95th percentile, Theil-Sen slope (median over random subsamples); pass = |median| <= 1.0 m and slope in 0.98-1.02; a map
  of |dz| > 2 m and its distribution over the hillslope classes (Addendum 7 class raster: 1 divergent, 2 hollow, 3 hilltop
  (option iii), 4 channel).
  Rule 5.4 (coverage): inside the 500 m-buffered footprint the fraction of 1 m cells with >= 1 and with 0 class-2 points; the
  class-2 density overall, on the option (iii) hilltop pixels (class 3) and in the hollows (class 2), on the tile's own cells.
  Rule 5.5 (dates): GPS-time range per tile converted to UTC dates (Adjusted Standard GPS Time + 1e9, GPS-UTC = 18 s).

usage: a13_las_checks.py --las-dir DIR --check-json pasda_check.json --data LEMCVT_DATA --results-root DIR --out DIR [--cache DIR]
       [--buffer 500] [--stage 1|2|both]
"""
import argparse, datetime as dt, glob, json, os, re, subprocess, sys, time
import numpy as np
import rasterio
from scipy.ndimage import map_coordinates
from scipy.stats import theilslopes
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a13_pasda_check import quadrant_bounds

FT_US = 1200.0 / 3937.0
TILE_FILLED_M = "AP/tile{n}/AP_1m_best_tile_filled_m.tif"
GPS_EPOCH = dt.datetime(1980, 1, 6, tzinfo=dt.timezone.utc)
LEAP = 18  # GPS - UTC seconds in 2019-2020


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr


def file_key(path):
    m = re.search(r"(\d{8}PAS_(?:NE|NW|SE|SW))", os.path.basename(path))
    return m.group(1) if m else os.path.basename(path)


def stage1(las_dir, out, cache):
    import laspy
    os.makedirs(out, exist_ok=True); os.makedirs(cache, exist_ok=True)
    files = sorted(glob.glob(os.path.join(las_dir, "*.las")) + glob.glob(os.path.join(las_dir, "*.laz")))
    recs = {}
    for path in files:
        key = file_key(path); t0 = time.time(); rec = {"path": path, "bytes": os.path.getsize(path)}
        code, so, se = run(["pdal", "info", "--metadata", path])
        rec["pdal_info_metadata_exit"] = code
        if code == 0:
            md = json.loads(so)["metadata"]; json.dump(md, open(os.path.join(out, f"{key}_pdal_metadata.json"), "w"), indent=1)
            srs = md.get("srs", {})
            rec["pdal"] = {"count": md.get("count"), "point_format": md.get("dataformat_id"), "version": f'{md.get("major_version")}.{md.get("minor_version")}',
                           "global_encoding": md.get("global_encoding"), "software_id": md.get("software_id"), "system_id": md.get("system_id"),
                           "creation": [md.get("creation_year"), md.get("creation_doy")], "bounds": [md.get("minx"), md.get("miny"), md.get("minz"), md.get("maxx"), md.get("maxy"), md.get("maxz")],
                           "scale": [md.get("scale_x"), md.get("scale_y"), md.get("scale_z")], "offset": [md.get("offset_x"), md.get("offset_y"), md.get("offset_z")],
                           "srs_units": srs.get("units"), "srs_horizontal": srs.get("horizontal"), "srs_vertical": srs.get("vertical"), "compoundwkt": srs.get("compoundwkt"), "wkt": srs.get("wkt"),
                           "vlr_count": len(md.get("vlrs", [])) if isinstance(md.get("vlrs"), list) else None}
        else:
            rec["pdal_info_error"] = se[-2000:]
        pipe = {"pipeline": [path, {"type": "filters.range", "limits": "Classification[2:2]"}, {"type": "filters.stats", "dimensions": "X,Y,Z,Intensity,ReturnNumber,NumberOfReturns,GpsTime", "enumerate": "ReturnNumber"}]}
        pj = os.path.join(cache, f"{key}_pipe.json"); json.dump(pipe, open(pj, "w"))
        mj = os.path.join(out, f"{key}_pdal_stats_class2.json")
        code, so, se = run(["pdal", "pipeline", pj, "--metadata", mj])
        rec["pdal_stats_class2_exit"] = code
        if code == 0 and os.path.exists(mj):
            mj_d = json.load(open(mj))
            st = mj_d.get("stages", mj_d.get("metadata", {})).get("filters.stats", {})
            stats = st.get("statistic", st) if isinstance(st, dict) else st
            rec["pdal_stats_class2"] = {s["name"]: {k: s.get(k) for k in ("count", "minimum", "maximum", "average")} for s in stats} if isinstance(stats, list) else stats
        else:
            rec["pdal_stats_error"] = se[-2000:]
        # laspy pass
        with laspy.open(path) as f:
            h = f.header
            rec["header"] = {"version": str(h.version), "point_format": int(h.point_format.id), "point_count": int(h.point_count), "mins": [float(v) for v in h.mins], "maxs": [float(v) for v in h.maxs],
                             "scales": [float(v) for v in h.scales], "offsets": [float(v) for v in h.offsets], "global_encoding": int(h.global_encoding.value) if hasattr(h.global_encoding, "value") else str(h.global_encoding),
                             "gps_time_type": str(getattr(h.global_encoding, "gps_time_type", "")), "generating_software": h.generating_software, "system_identifier": h.system_identifier,
                             "creation_date": str(h.creation_date), "n_vlrs": len(h.vlrs), "n_evlrs": len(getattr(h, "evlrs", []) or []),
                             "vlrs": [{"user_id": v.user_id, "record_id": int(v.record_id), "description": v.description} for v in h.vlrs],
                             "number_of_points_by_return": [int(x) for x in h.number_of_points_by_return][:8]}
            crs = None
            try:
                crs = h.parse_crs()
            except Exception as e:
                rec["crs_parse_error"] = repr(e)
            if crs is not None:
                hu = crs.axis_info[0].unit_name if crs.axis_info else None
                vu = None; vname = None
                if crs.is_compound and len(crs.sub_crs_list) > 1:
                    v = crs.sub_crs_list[1]; vname = v.name; vu = v.axis_info[0].unit_name if v.axis_info else None
                    hu = crs.sub_crs_list[0].axis_info[0].unit_name
                rec["crs"] = {"name": crs.name, "epsg": crs.to_epsg(), "is_compound": crs.is_compound, "horizontal_units": hu, "vertical_name": vname, "vertical_units": vu,
                              "datum": crs.sub_crs_list[0].datum.name if crs.is_compound else (crs.datum.name if crs.datum else None), "wkt": crs.to_wkt()}
            else:
                rec["crs"] = {"name": None, "horizontal_units": None, "vertical_units": None}
            cls = np.zeros(256, np.int64); ret = np.zeros(16, np.int64); gps = [np.inf, -np.inf]; n = 0
            xs, ys, zs, ts = [], [], [], []
            has_gps = "gps_time" in h.point_format.dimension_names
            for pts in f.chunk_iterator(4_000_000):
                c = np.asarray(pts.classification); cls += np.bincount(c, minlength=256)[:256]; ret += np.bincount(np.asarray(pts.return_number), minlength=16)[:16]; n += c.size
                g = np.asarray(pts.gps_time) if has_gps else None
                if has_gps:
                    gps = [min(gps[0], float(g.min())), max(gps[1], float(g.max()))]
                k = c == 2
                if k.any():
                    xs.append(np.asarray(pts.x)[k]); ys.append(np.asarray(pts.y)[k]); zs.append(np.asarray(pts.z)[k]); ts.append(g[k] if has_gps else np.zeros(int(k.sum())))
            X = np.concatenate(xs) if xs else np.zeros(0); Y = np.concatenate(ys) if ys else np.zeros(0); Z = np.concatenate(zs) if zs else np.zeros(0); T = np.concatenate(ts) if ts else np.zeros(0)
            np.savez(os.path.join(cache, f"{key}_class2.npz"), x=X, y=Y, z=Z, t=T)
            rec["laspy"] = {"n_read": n, "classes": {int(i): int(v) for i, v in enumerate(cls) if v}, "return_number": {int(i): int(v) for i, v in enumerate(ret) if v},
                            "gps_time_range": gps if has_gps else None, "n_class2": int(X.size), "class2_z_range_header_units": [float(Z.min()), float(Z.max())] if Z.size else None,
                            "density_all_per_header_unit2": n / ((h.maxs[0] - h.mins[0]) * (h.maxs[1] - h.mins[1])), "density_class2_per_header_unit2": X.size / ((h.maxs[0] - h.mins[0]) * (h.maxs[1] - h.mins[1]))}
            if has_gps:
                adj = 1e9 if (rec["header"]["global_encoding"] & 1) else 0.0
                rec["laspy"]["utc_range"] = [(GPS_EPOCH + dt.timedelta(seconds=g + adj - LEAP)).isoformat() for g in gps]
        # Rule 5.1
        qb = quadrant_bounds(key + "_LAS.zip"); mn, mx = rec["header"]["mins"], rec["header"]["maxs"]
        tol = 50.0
        rec["rule_5_1"] = {"quadrant_bounds_ftus": qb, "header_xy_bounds": [mn[0], mn[1], mx[0], mx[1]],
                           "excess_ft": {"west": qb[0] - mn[0], "south": qb[1] - mn[1], "east": mx[0] - qb[2], "north": mx[1] - qb[3]},
                           "pass": bool(mn[0] >= qb[0] - tol and mn[1] >= qb[1] - tol and mx[0] <= qb[2] + tol and mx[1] <= qb[3] + tol)}
        # Rule 5.2
        hu = (rec["crs"].get("horizontal_units") or "") ; vu = (rec["crs"].get("vertical_units") or "")
        pu = (rec.get("pdal", {}).get("srs_units") or {})
        def is_ftus(s):
            s = (s or "").lower().replace("_", " "); return ("us survey foot" in s) or ("foot us" in s) or ("us foot" in s) or ("ftus" in s)
        rec["rule_5_2"] = {"laspy_horizontal": hu, "laspy_vertical": vu, "pdal_horizontal": pu.get("horizontal"), "pdal_vertical": pu.get("vertical"),
                           "pass": bool(is_ftus(hu) and is_ftus(vu) and is_ftus(pu.get("horizontal")) and is_ftus(pu.get("vertical"))),
                           "note": "pass requires US survey feet stated for both axes by both readers; 1 ftUS = 1200/3937 m is the only conversion used"}
        rec["seconds"] = time.time() - t0
        recs[key] = rec
        print(key, "5.1", rec["rule_5_1"]["pass"], "5.2", rec["rule_5_2"]["pass"], "n", n, "class2", X.size, "%.0fs" % rec["seconds"], file=sys.stderr, flush=True)
        json.dump(recs, open(os.path.join(out, "stage1_files.json"), "w"), indent=1, default=str)
    return recs


def stage2(check_json, data, results_root, out, cache, buffer_m, recs):
    from pyproj import Transformer
    os.makedirs(out, exist_ok=True)
    chk = json.load(open(check_json))
    tr = Transformer.from_crs(6565, 26917, always_xy=True)
    # z: ftUS -> m is a unit scaling only; the vertical datum (NAVD88) is the same as the PAMAP grids state (checked in the report)
    res = {"transformer": tr.description, "buffer_m": buffer_m, "ft_us_in_m": FT_US, "tiles": {}}
    for tkey, v in chk["per_tile"].items():
        n = re.search(r"\d", tkey).group(0); tile = f"AP{n}"; t0 = time.time()
        tp = os.path.join(data, TILE_FILLED_M.format(n=n))
        with rasterio.open(tp) as ds:
            Zt = ds.read(1).astype(np.float64); T = ds.transform; tb = ds.bounds; nod = ds.nodata; epsg = ds.crs.to_epsg()
        okt = np.isfinite(Zt) & ((Zt != nod) if nod is not None and not np.isnan(nod) else True)
        cls_paths = sorted(glob.glob(os.path.join(results_root, f"*_d_diagnostic_{tile}", f"{tile}_hillslope_classes.tif")))
        cls = None
        if cls_paths:
            with rasterio.open(cls_paths[-1]) as cd:
                cls = cd.read(1).astype(np.int8); assert cd.transform == T and cd.shape == Zt.shape
        rec = {"tile": tile, "tile_path": tp, "tile_epsg": epsg, "tile_bounds": list(tb), "tile_res": [T.a, -T.e], "class_raster": cls_paths[-1] if cls_paths else None, "files": v["files"]}
        xs, ys, zs, ts = [], [], [], []
        for f in v["files"]:
            key = f.replace("_LAS.zip", ""); cp = os.path.join(cache, f"{key}_class2.npz")
            if not os.path.exists(cp):
                rec.setdefault("missing_cache", []).append(key); continue
            d = np.load(cp); xs.append(d["x"]); ys.append(d["y"]); zs.append(d["z"]); ts.append(d["t"])
        X = np.concatenate(xs); Y = np.concatenate(ys); Z = np.concatenate(zs); Tg = np.concatenate(ts)
        xm, ym = tr.transform(X, Y); zm = Z * FT_US
        b = buffer_m
        inb = (xm >= tb.left - b) & (xm <= tb.right + b) & (ym >= tb.bottom - b) & (ym <= tb.top + b)
        xm, ym, zm, Tg = xm[inb], ym[inb], zm[inb], Tg[inb]
        rec["n_class2_in_buffered_footprint"] = int(inb.sum())
        # Rule 5.4 on 1 m cells of the buffered footprint
        W = int(np.ceil(tb.right - tb.left + 2 * b)); H = int(np.ceil(tb.top - tb.bottom + 2 * b))
        ci = np.clip(((xm - (tb.left - b))).astype(int), 0, W - 1); ri = np.clip(((tb.top + b) - ym).astype(int), 0, H - 1)
        occ = np.zeros((H, W), np.int32); np.add.at(occ, (ri, ci), 1)
        area_buf = float((tb.right - tb.left + 2 * b) * (tb.top - tb.bottom + 2 * b))
        rec["rule_5_4"] = {"buffered_footprint_m2": area_buf, "frac_1m_cells_ge1_class2": float((occ > 0).mean()), "frac_1m_cells_zero_class2": float((occ == 0).mean()),
                           "density_class2_overall_per_m2": float(inb.sum() / area_buf),
                           "cells_1m_count_percentiles": {str(q): float(np.percentile(occ, q)) for q in (5, 25, 50, 75, 95)}}
        # inside the tile, on the tile's own cells
        col = (xm - T.c) / T.a; row = (ym - T.f) / T.e
        intile = (col >= 0) & (col < Zt.shape[1]) & (row >= 0) & (row < Zt.shape[0])
        c_i = col[intile].astype(int); r_i = row[intile].astype(int)
        cnt = np.zeros(Zt.shape, np.int32); np.add.at(cnt, (r_i, c_i), 1)
        cell_area = T.a * (-T.e)
        rec["rule_5_4"]["tile_cell_area_m2"] = cell_area
        rec["rule_5_4"]["density_class2_in_tile_per_m2"] = float(intile.sum() / ((tb.right - tb.left) * (tb.top - tb.bottom)))
        rec["rule_5_4"]["frac_tile_cells_zero_class2"] = float((cnt == 0).mean())
        if cls is not None:
            for code, name in ((3, "hilltop_option_iii"), (2, "hollow"), (1, "divergent"), (4, "channel")):
                m = cls == code
                rec["rule_5_4"][f"density_class2_{name}_per_m2"] = float(cnt[m].sum() / (m.sum() * cell_area)) if m.any() else None
                rec["rule_5_4"][f"n_cells_{name}"] = int(m.sum()); rec["rule_5_4"][f"frac_cells_zero_{name}"] = float((cnt[m] == 0).mean()) if m.any() else None
        # Rule 5.3: bilinear lookup
        Zf = np.where(okt, Zt, 0.0); wt = okt.astype(np.float64)
        rr = row[intile] - 0.5; cc = col[intile] - 0.5   # cell centres at integer indices
        zp = map_coordinates(Zf, [rr, cc], order=1, mode="nearest"); ww = map_coordinates(wt, [rr, cc], order=1, mode="nearest")
        good = ww > 0.999
        z3 = zm[intile][good]; zp = zp[good] / ww[good]; dz = z3 - zp
        rng = np.random.default_rng(13)
        slopes = []
        for _ in range(7):
            idx = rng.choice(z3.size, min(4000, z3.size), replace=False)
            slopes.append(float(theilslopes(z3[idx], zp[idx])[0]))
        ols = float(np.polyfit(zp, z3, 1)[0])
        med = float(np.median(dz)); slope = float(np.median(slopes))
        rec["rule_5_3"] = {"n_points": int(z3.size), "median_dz_m": med, "p05_dz_m": float(np.percentile(dz, 5)), "p95_dz_m": float(np.percentile(dz, 95)), "mean_dz_m": float(dz.mean()),
                           "rms_dz_m": float(np.sqrt(np.mean(dz ** 2))), "theil_sen_slope_median_of_7x4000": slope, "theil_sen_slopes": slopes, "ols_slope": ols,
                           "frac_abs_dz_gt_2m": float(np.mean(np.abs(dz) > 2)), "frac_abs_dz_gt_1m": float(np.mean(np.abs(dz) > 1)), "frac_abs_dz_gt_0.5m": float(np.mean(np.abs(dz) > 0.5)),
                           "pass_median": bool(abs(med) <= 1.0), "pass_slope": bool(0.98 <= slope <= 1.02), "pass": bool(abs(med) <= 1.0 and 0.98 <= slope <= 1.02),
                           "z_pamap_range_m": [float(zp.min()), float(zp.max())], "z_3dep_range_m": [float(z3.min()), float(z3.max())]}
        big = np.abs(dz) > 2
        G = np.zeros(Zt.shape, np.int32); np.add.at(G, (r_i[good][big], c_i[good][big]), 1)
        Dm = np.full(Zt.shape, np.nan); S = np.zeros(Zt.shape); C = np.zeros(Zt.shape); np.add.at(S, (r_i[good], c_i[good]), dz); np.add.at(C, (r_i[good], c_i[good]), 1)
        Dm = np.where(C > 0, S / np.maximum(C, 1), np.nan)
        np.save(os.path.join(out, f"{tile}_mean_dz_per_cell.npy"), Dm.astype(np.float32)); np.save(os.path.join(out, f"{tile}_count_abs_dz_gt2.npy"), G)
        rec["rule_5_3"]["cells_with_abs_dz_gt_2m"] = int((G > 0).sum()); rec["rule_5_3"]["frac_cells_with_abs_dz_gt_2m"] = float((G > 0).mean())
        if cls is not None:
            cl = cls[r_i[good], c_i[good]]
            rec["rule_5_3"]["by_class"] = {name: {"n": int((cl == code).sum()), "median_dz": float(np.median(dz[cl == code])) if (cl == code).any() else None,
                                                  "frac_abs_dz_gt_2m": float(np.mean(np.abs(dz[cl == code]) > 2)) if (cl == code).any() else None,
                                                  "share_of_gt2m_points": float(np.sum(big & (cl == code)) / max(big.sum(), 1))}
                                           for code, name in ((1, "divergent"), (2, "hollow"), (3, "hilltop"), (4, "channel"), (0, "edge_or_unclassed"))}
        # Rule 5.5
        if Tg.size and np.isfinite(Tg).all() and Tg.max() > 0:
            adj = 1e9
            t_utc = [(GPS_EPOCH + dt.timedelta(seconds=float(g) + adj - LEAP)) for g in (Tg.min(), Tg.max())]
            days = ((Tg + adj - LEAP) // 86400).astype(np.int64)
            ud, uc = np.unique(days, return_counts=True)
            rec["rule_5_5"] = {"utc_min": t_utc[0].isoformat(), "utc_max": t_utc[1].isoformat(), "n_distinct_days": int(ud.size),
                               "days": {(GPS_EPOCH + dt.timedelta(days=int(d))).date().isoformat(): int(c) for d, c in zip(ud, uc)},
                               "leaf_off_inference": "dates fall between 18 Nov 2019 and 22 Mar 2020 (project report 1.4); in SW Pennsylvania that is the leaf-off season, inferred from the calendar, not from the data"}
        rec["seconds"] = time.time() - t0
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 3, figsize=(16, 5.2))
            im = ax[0].imshow(Dm, cmap="RdBu_r", vmin=-2, vmax=2); fig.colorbar(im, ax=ax[0], label="mean dz per cell, 3DEP - PAMAP (m)"); ax[0].set_title(f"{tile}: dz")
            ax[1].imshow(G > 0, cmap="gray_r"); ax[1].set_title("cells with |dz| > 2 m")
            im = ax[2].imshow(cnt, cmap="viridis", vmin=0, vmax=8); fig.colorbar(im, ax=ax[2], label="class-2 points per tile cell"); ax[2].set_title("class-2 count")
            fig.savefig(os.path.join(out, f"{tile}_dz_coverage.png"), dpi=110); plt.close(fig)
        except Exception as e:
            rec["figure_error"] = repr(e)
        res["tiles"][tile] = rec
        print(tile, "5.3 pass", rec["rule_5_3"]["pass"], "median %.3f slope %.4f" % (med, slope), "5.4 occ %.3f dens %.2f hilltop %s hollow %s" % (rec["rule_5_4"]["frac_1m_cells_ge1_class2"], rec["rule_5_4"]["density_class2_overall_per_m2"], rec["rule_5_4"].get("density_class2_hilltop_option_iii_per_m2"), rec["rule_5_4"].get("density_class2_hollow_per_m2")), file=sys.stderr, flush=True)
        json.dump(res, open(os.path.join(out, "stage2_tiles.json"), "w"), indent=1, default=str)
    return res


def main():
    p = argparse.ArgumentParser(); p.add_argument("--las-dir", required=True); p.add_argument("--check-json", required=True); p.add_argument("--data", required=True)
    p.add_argument("--results-root", required=True); p.add_argument("--out", required=True); p.add_argument("--cache"); p.add_argument("--buffer", type=float, default=500.0)
    p.add_argument("--stage", default="both")
    a = p.parse_args(); cache = a.cache or os.path.join(a.out, "cache")
    recs = {}
    if a.stage in ("1", "both"):
        recs = stage1(a.las_dir, a.out, cache)
    elif os.path.exists(os.path.join(a.out, "stage1_files.json")):
        recs = json.load(open(os.path.join(a.out, "stage1_files.json")))
    if a.stage in ("2", "both"):
        stage2(a.check_json, a.data, a.results_root, a.out, cache, a.buffer, recs)


if __name__ == "__main__":
    main()
