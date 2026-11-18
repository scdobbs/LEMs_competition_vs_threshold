#!/usr/bin/env python3
"""Addendum 14, decision 9 and sections 4.1.5, 4.3.3, 5.1, 5.2: grid a point-cloud collection at 1 m and route it.

Gridding rule (decision 9): 1 m cells on an integer-metre lattice in NAD83(2011) UTM of the site's zone (AP EPSG:6346,
TTV EPSG:6339); each cell = the mean of the class-2 (ground) points inside it; empty cells inside the footprint are
filled by nearest neighbour and flagged (per-cell `filled` mask; the per-cell class-2 count is kept as the density);
heights in metres, converted only when the file header states feet (1 ftUS = 1200/3937 m); no TIN.
Routing (decision 7): priority-flood fill and D8 on the whole grid (a14_bigroute), drainage area as exact cell counts.
Checks: rule 5.1 (AP: the new grid against the PAMAP tiles AP1-AP5, metres; median dz within 0.1 m of the addendum 13
3DEP-point values, Theil-Sen slope 0.99-1.01) and rule 5.2 (TTV: the new grid against the NCALM 2006 raster on ground
cells; median within 1.0 m, slope 0.98-1.02; a slope near 3.28 or 0.305 or an offset near -32 m stops the work).

usage: a14_grid_route.py --site AP|TTV --files LIST.txt --out DIR --footprint GEOJSON [--stage grid|route|check|all]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import rasterio
from rasterio.transform import Affine

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a14_common import FT_US, SOURCE, dump, dz_stats, slurm_record, theil_sen_sub, write_tif  # noqa: E402

EPSG = {"AP": 6346, "TTV": 6339, "GMF": 6339}   # GMF: the GM extension survey of addendum 16 (section 3.3.2), checked by a16_gmf_check.py
A13_AP_MEDIANS = {"AP1": -0.079, "AP2": -0.077, "AP3": -0.066, "AP4": -0.026, "AP5": -1.206}
NODATA = -9999.0


def rss_gb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM"):
                    return int(line.split()[1]) / 1e6
    except OSError:
        pass
    return None


def lattice_from_footprint(geojson, epsg_out, pad=5.0):
    from osgeo import ogr, osr
    ogr.UseExceptions()
    ds = ogr.Open(geojson)
    lyr = ds.GetLayer()
    f = lyr.GetNextFeature()
    g = f.GetGeometryRef().Clone()
    src = lyr.GetSpatialRef()
    src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    dst = osr.SpatialReference()
    dst.ImportFromEPSG(epsg_out)
    dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    g.Transform(osr.CoordinateTransformation(src, dst))
    x0, x1, y0, y1 = g.GetEnvelope()
    x0, y0 = np.floor(x0 - pad), np.floor(y0 - pad)
    x1, y1 = np.ceil(x1 + pad), np.ceil(y1 + pad)
    tr = Affine(1.0, 0, x0, 0, -1.0, y1)
    shape = (int(y1 - y0), int(x1 - x0))
    return tr, shape, g, dst


def rasterize_geom(g, srs, tr, shape):
    from osgeo import gdal, ogr
    ny, nx = shape
    ds = gdal.GetDriverByName("MEM").Create("", nx, ny, 1, gdal.GDT_Byte)
    ds.SetGeoTransform((tr.c, tr.a, 0, tr.f, 0, tr.e))
    ds.SetProjection(srs.ExportToWkt())
    vds = ogr.GetDriverByName("MEM").CreateDataSource("v")
    lyr = vds.CreateLayer("p", srs=srs, geom_type=ogr.wkbMultiPolygon)
    f = ogr.Feature(lyr.GetLayerDefn())
    f.SetGeometry(g)
    lyr.CreateFeature(f)
    gdal.RasterizeLayer(ds, [1], lyr, burn_values=[1], options=["ALL_TOUCHED=TRUE"])
    return ds.GetRasterBand(1).ReadAsArray().astype(bool)


def file_crs(h):
    """Horizontal CRS, horizontal unit and vertical unit as the LAS header states them (None if absent)."""
    crs = h.parse_crs()
    if crs is None:
        return None, None, None, None
    horiz, vert = (crs.sub_crs_list[0], crs.sub_crs_list[1]) if crs.is_compound and len(crs.sub_crs_list) > 1 else (crs, None)
    hu = horiz.axis_info[0].unit_name if horiz.axis_info else None
    vu = vert.axis_info[0].unit_name if (vert is not None and vert.axis_info) else None
    return crs, horiz, hu, vu


def z_scale_for(vu, hu):
    """Metres per stated vertical unit; None when the vertical unit is not stated (decision 10: no assumption)."""
    u = vu or None
    if u is None:
        return None, "vertical unit not stated"
    ul = u.lower()
    if "us survey foot" in ul or ul in ("us survey foot", "foot_us", "ftus"):
        return FT_US, "US survey foot -> 1200/3937 m"
    if ul in ("foot", "international foot", "ft"):
        return 0.3048, "international foot -> 0.3048 m"
    if ul.startswith("met"):
        return 1.0, "metre"
    return None, f"unrecognised vertical unit {u!r}"


def stage_grid(a, tr, shape, inside, epsg_out):
    import laspy
    from pyproj import CRS, Transformer
    ny, nx = shape
    n = ny * nx
    ssum = np.zeros(n, np.float64)
    cnt = np.zeros(n, np.uint16)
    files = [line.strip() for line in open(a.files) if line.strip()]
    recs = []
    t_all = time.time()
    import zipfile
    tmpdir = os.path.join(a.work or os.environ.get("L_SCRATCH_JOB", "/tmp"), "unzip")
    os.makedirs(tmpdir, exist_ok=True)
    for k, path in enumerate(files):
        t0 = time.time()
        rec = {"file": os.path.basename(path), "bytes": os.path.getsize(path)}
        las_path = path
        if path.lower().endswith(".zip"):
            # PASDA delivery: one LAS (+ one XML) per zip; extract the LAS to node-local scratch, delete after use
            with zipfile.ZipFile(path) as zf:
                names = [n for n in zf.namelist() if n.lower().endswith((".las", ".laz"))]
                rec["zip_members"] = zf.namelist()
                las_path = zf.extract(names[0], tmpdir) if len(names) == 1 else None
            if las_path is None:
                rec["error"] = f"zip holds {len(names)} point files"
                recs.append(rec)
                continue
            rec["las_bytes"] = os.path.getsize(las_path)
        with laspy.open(las_path) as f:
            h = f.header
            crs, horiz, hu, vu = file_crs(h)
            rec.update({"las_version": str(h.version), "point_format": int(h.point_format.id), "points": int(h.point_count),
                        "crs_name": crs.name if crs is not None else None, "horizontal_unit": hu, "vertical_unit": vu,
                        "horizontal_epsg": horiz.to_epsg() if horiz is not None else None, "software": h.generating_software})
            zs, how = z_scale_for(vu, hu)
            rec["z_scale"] = zs
            rec["z_conversion"] = how
            if zs is None or horiz is None:
                rec["error"] = "CRS or vertical unit not stated in the header; file skipped (decision 10)"
                recs.append(rec)
                print(rec, flush=True)
                continue
            tf = Transformer.from_crs(horiz, CRS.from_epsg(epsg_out), always_xy=True)
            rec["transformer"] = tf.description[:200]
            n2 = n_in = 0
            has_gps = "gps_time" in h.point_format.dimension_names
            gmin, gmax = np.inf, -np.inf
            for pts in f.chunk_iterator(4_000_000):
                c = np.asarray(pts.classification)
                g = c == 2
                if has_gps and g.any():
                    gt = np.asarray(pts.gps_time)[g]
                    gmin, gmax = min(gmin, float(gt.min())), max(gmax, float(gt.max()))
                if not g.any():
                    continue
                x, y = tf.transform(np.asarray(pts.x)[g], np.asarray(pts.y)[g])
                z = np.asarray(pts.z)[g] * zs
                col = np.floor((x - tr.c) / tr.a).astype(np.int64)
                row = np.floor((y - tr.f) / tr.e).astype(np.int64)
                ok = (row >= 0) & (row < ny) & (col >= 0) & (col < nx)
                n2 += int(g.sum())
                n_in += int(ok.sum())
                idx = row[ok] * nx + col[ok]
                u, inv = np.unique(idx, return_inverse=True)
                ssum[u] += np.bincount(inv, weights=z[ok], minlength=u.size)
                cc = np.bincount(inv, minlength=u.size)
                cnt[u] = np.minimum(cnt[u].astype(np.int64) + cc, 65535).astype(np.uint16)
            rec.update({"class2": n2, "class2_in_lattice": n_in, "seconds": time.time() - t0})
            if has_gps and np.isfinite(gmin):
                import datetime as dt
                flag = bool(int(h.global_encoding.value) & 1) if hasattr(h.global_encoding, "value") else True
                adjusted = flag or gmax > 7e5          # GPS week seconds never exceed 604,800; larger values are adjusted standard GPS time
                off = 1e9 if adjusted else 0.0
                epoch = dt.datetime(1980, 1, 6, tzinfo=dt.UTC)
                d0 = epoch + dt.timedelta(seconds=gmin + off - 18)
                d1 = epoch + dt.timedelta(seconds=gmax + off - 18)
                rec["gps_time_range"] = [gmin, gmax]
                rec["utc_dates_class2"] = [d0.strftime("%Y-%m-%d %H:%M"), d1.strftime("%Y-%m-%d %H:%M"), ("adjusted standard GPS time (header flag)" if flag else "adjusted standard GPS time (inferred from magnitude; header flag unset)") if adjusted else "GPS week time (dates not derivable)"]
        if las_path != path:
            os.remove(las_path)
        recs.append(rec)
        if k % 10 == 0:
            print(k, rec["file"], rec.get("class2"), round(rec.get("seconds", 0), 1), "s; rss GB", rss_gb(), flush=True)
    has = cnt > 0
    mean = np.full(n, NODATA, np.float32)
    mean[has] = (ssum[has] / cnt[has]).astype(np.float32)
    del ssum
    mean = mean.reshape(ny, nx)
    has = has.reshape(ny, nx)
    cnt = cnt.reshape(ny, nx)
    # nearest-neighbour fill inside the footprint
    from scipy.ndimage import distance_transform_edt
    t0 = time.time()
    empty = inside & ~has
    dist, (ir, ic) = distance_transform_edt(~has, return_distances=True, return_indices=True)
    # fill gaps only within MAX_FILL_M of a ground cell; farther cells (water, no lidar) stay nodata and are counted
    fill_ok = empty & (dist <= a.max_fill)
    gap = empty & ~fill_ok
    filled = np.zeros((ny, nx), np.uint8)
    filled[fill_ok] = 1
    filled[gap] = 2
    mean[fill_ok] = mean[ir[fill_ok], ic[fill_ok]]
    mean[gap] = NODATA
    nn_dist = np.minimum(dist, 255).astype(np.uint8)
    del dist, ir, ic
    mean[~inside] = NODATA
    inside_data = inside & ~gap
    stats = {"n_files": len(files), "n_files_ok": sum("error" not in r for r in recs), "lattice_shape": [ny, nx], "transform": list(tr)[:6], "epsg": epsg_out,
             "n_inside_footprint": int(inside.sum()), "n_cells_with_class2": int(has.sum()), "n_cells_with_class2_inside": int((has & inside).sum()),
             "max_fill_distance_m": a.max_fill, "n_filled_within_max": int(fill_ok.sum()), "n_gap_beyond_max_left_nodata": int(gap.sum()),
             "frac_inside_filled": float(fill_ok.sum() / max(1, inside_data.sum())), "frac_inside_gap_nodata": float(gap.sum() / max(1, inside.sum())),
             "n_filled": int(fill_ok.sum()), "n_valid_cells_in_grid": int(inside_data.sum()),
             "nn_fill_distance_percentiles_50_95_99_max_m": [float(v) for v in np.percentile(nn_dist[fill_ok], [50, 95, 99, 100])] if fill_ok.any() else None,
             "class2_density_inside_pts_per_m2": float(cnt[inside_data].astype(np.float64).sum() / max(1, inside_data.sum())),
             "points_per_cell_percentiles_5_50_95": [int(v) for v in np.percentile(cnt[has & inside], [5, 50, 95])] if (has & inside).any() else None,
             "frac_data_cells_with_ge1_point": float((has & inside).sum() / max(1, inside_data.sum())),
             "total_class2": int(sum(r.get("class2", 0) for r in recs)), "seconds_total": time.time() - t_all, "seconds_fill": time.time() - t0, "peak_rss_GB": rss_gb()}
    crs = rasterio.crs.CRS.from_epsg(epsg_out)
    write_tif(os.path.join(a.out, f"{a.site}_1m_mean.tif"), mean, tr, crs, nodata=NODATA)
    write_tif(os.path.join(a.out, f"{a.site}_1m_class2_count.tif"), cnt, tr, crs)
    write_tif(os.path.join(a.out, f"{a.site}_1m_filled_mask.tif"), filled, tr, crs)
    write_tif(os.path.join(a.out, f"{a.site}_1m_nn_fill_dist.tif"), nn_dist, tr, crs)
    dump({"files": recs, "stats": stats}, os.path.join(a.out, f"{a.site}_grid.json"))
    print("grid stats", json.dumps(stats, indent=1), flush=True)


def stage_route(a, tr, shape, epsg_out):
    from a14_bigroute import d8_area_big, d8_chunked, priority_flood
    t_all = time.time()
    with rasterio.open(os.path.join(a.out, f"{a.site}_1m_mean.tif")) as ds:
        Z = ds.read(1).astype(np.float64)
        crs = ds.crs
    valid = Z != NODATA
    Z[~valid] = np.nan
    Z = priority_flood(Z, valid, 1.0)
    write_tif(os.path.join(a.out, f"{a.site}_1m_filled.tif"), np.where(valid, Z, NODATA).astype(np.float32), tr, crs, nodata=NODATA)
    t1 = rss_gb()
    fd = d8_chunked(Z)
    fd[~valid] = 0
    # data-edge cells (bordering nodata or the grid edge) are the outlets: no receiver (see a14_pools.route_full_grid)
    from scipy.ndimage import binary_erosion
    edge = valid & ~binary_erosion(valid, structure=np.ones((3, 3), bool), border_value=0)
    fd[edge] = 0
    n_edge = int(edge.sum())
    del edge
    write_tif(os.path.join(a.out, f"{a.site}_1m_fd.tif"), fd, tr, crs)
    count, target = d8_area_big(fd, Z, valid)
    write_tif(os.path.join(a.out, f"{a.site}_1m_area_cells.tif"), count, tr, crs)
    # outlets: valid cells whose receiver is outside the valid data (roots); the largest is the watershed outlet
    roots = (target < 0) & valid.ravel()
    ra = count.ravel()[roots]
    order = np.argsort(-ra)[:20]
    ri = np.nonzero(roots)[0][order]
    ny, nx = shape
    rec = {"n_valid": int(valid.sum()), "n_edge_cells": n_edge, "max_area_cells": int(count.max()), "max_area_km2": float(count.max() / 1e6),
           "n_roots": int(roots.sum()), "root_area_sum_cells": int(ra.sum()), "root_area_sum_equals_valid": bool(ra.sum() == valid.sum()),
           "largest_roots": [{"row": int(i // nx), "col": int(i % nx), "x": tr.c + (i % nx + 0.5) * tr.a, "y": tr.f + (i // nx + 0.5) * tr.e,
                              "area_cells": int(count.ravel()[i]), "share_of_valid": float(count.ravel()[i] / valid.sum())} for i in ri],
           "seconds_total": time.time() - t_all, "peak_rss_GB_after_fill": t1, "peak_rss_GB": rss_gb()}
    dump(rec, os.path.join(a.out, f"{a.site}_route.json"))
    print("route", json.dumps({k: v for k, v in rec.items() if k != "largest_roots"}, indent=1), rec["largest_roots"][:3], flush=True)


def stage_check(a, tr, shape, epsg_out):
    from pyproj import Transformer
    from scipy.ndimage import map_coordinates
    rec = {}
    with rasterio.open(os.path.join(a.out, f"{a.site}_1m_mean.tif")) as ds:
        Zm = ds.read(1)
    with rasterio.open(os.path.join(a.out, f"{a.site}_1m_filled_mask.tif")) as ds:
        filled = ds.read(1).astype(bool)
    Zg = np.where((Zm != NODATA) & ~filled, Zm, np.nan).astype(np.float64)      # ground cells only (no NN fill)
    ny, nx = shape

    def sample(xs, ys):
        fc = (xs - tr.c) / tr.a - 0.5
        fr = (ys - tr.f) / tr.e - 0.5
        v = map_coordinates(Zg, [fr, fc], order=1, mode="constant", cval=np.nan)
        return v

    if a.site == "AP":
        data = a.data or os.environ.get("LEMCVT_DATA")
        for t in ("AP1", "AP2", "AP3", "AP4", "AP5"):
            path = os.path.join(data, "AP", f"tile{t[-1]}", "AP_1m_best_tile_filled_m.tif")
            with rasterio.open(path) as ds:
                zt = ds.read(1).astype(np.float64)
                ttr, tepsg, tnod = ds.transform, ds.crs.to_epsg(), ds.nodata
            tf = Transformer.from_crs(tepsg, epsg_out, always_xy=True)
            rr, cc = np.mgrid[0:zt.shape[0], 0:zt.shape[1]]
            xs = ttr.c + (cc.ravel() + 0.5) * ttr.a
            ys = ttr.f + (rr.ravel() + 0.5) * ttr.e
            X, Y = tf.transform(xs, ys)
            v = sample(X, Y)
            zz = zt.ravel()
            ok = np.isfinite(v) & np.isfinite(zz) & (zz != tnod if tnod is not None else True)
            if ok.sum() < 1000:
                rec[t] = {"tile": t, "n": int(ok.sum()), "pass": "not covered by this grid"}
                continue
            dz = v[ok] - zz[ok]
            fit = theil_sen_sub(zz[ok], v[ok])
            med = float(np.median(dz))
            r = {"tile": t, "n": int(ok.sum()), "frac_tile_cells_compared": float(ok.sum() / zz.size), "dz_new_minus_pamap": dz_stats(dz), "fit_new_on_pamap": fit,
                 "a13_median": A13_AP_MEDIANS[t], "median_within_0.1_of_a13": bool(abs(med - A13_AP_MEDIANS[t]) <= 0.1),
                 "slope_pass_0.99_1.01": bool(0.99 <= fit["theil_sen_slope"] <= 1.01), "transformer": tf.description[:160],
                 "shift_at_centre_m": [float(X.mean() - xs.mean()), float(Y.mean() - ys.mean())]}
            r["pass"] = (r["median_within_0.1_of_a13"] and r["slope_pass_0.99_1.01"]) if t != "AP5" else "excluded (addendum 13, 6.2)"
            rec[t] = r
            print(t, med, fit["theil_sen_slope"], r["pass"], flush=True)
    else:
        ncalm = os.path.join(SOURCE, "TTV", "NCALM_2006_be", "output_be.tif")
        with rasterio.open(ncalm) as ds:
            zn = ds.read(1).astype(np.float64)
            ntr, nepsg, nnod = ds.transform, ds.crs.to_epsg(), ds.nodata
        tf = Transformer.from_crs(nepsg, epsg_out, always_xy=True)
        rr, cc = np.mgrid[0:zn.shape[0], 0:zn.shape[1]]
        xs = ntr.c + (cc.ravel() + 0.5) * ntr.a
        ys = ntr.f + (rr.ravel() + 0.5) * ntr.e
        X, Y = tf.transform(xs, ys)
        v = sample(X, Y)
        zz = zn.ravel()
        ok = np.isfinite(v) & np.isfinite(zz) & (zz != nnod) & (zz > 3.0)
        dz = v[ok] - zz[ok]
        fit = theil_sen_sub(zz[ok], v[ok])
        med = float(np.median(dz))
        r = {"n": int(ok.sum()), "dz_marin_minus_ncalm": dz_stats(dz), "fit_marin_on_ncalm": fit, "median_pass_abs_le_1m": bool(abs(med) <= 1.0),
             "slope_pass_0.98_1.02": bool(0.98 <= fit["theil_sen_slope"] <= 1.02),
             "units_error_flag_slope_near_3.28_or_0.305": bool(abs(fit["theil_sen_slope"] - 3.28) < 0.2 or abs(fit["theil_sen_slope"] - 0.305) < 0.05),
             "ellipsoid_flag_offset_near_-32m": bool(abs(med + 32) < 5), "transformer": tf.description[:160], "shift_at_centre_m": [float(X.mean() - xs.mean()), float(Y.mean() - ys.mean())]}
        r["pass"] = r["median_pass_abs_le_1m"] and r["slope_pass_0.98_1.02"]
        r["stop"] = r["units_error_flag_slope_near_3.28_or_0.305"] or r["ellipsoid_flag_offset_near_-32m"]
        rec["rule_5_2"] = r
        # the block pattern of dz on a 10 x 10 grid of the NCALM raster
        D = np.full(zz.shape, np.nan)
        D[ok] = dz
        D = D.reshape(zn.shape)
        blocks = []
        for bi in range(10):
            row = []
            for bj in range(10):
                d = D[bi * zn.shape[0] // 10:(bi + 1) * zn.shape[0] // 10, bj * zn.shape[1] // 10:(bj + 1) * zn.shape[1] // 10]
                d = d[np.isfinite(d)]
                row.append(float(np.median(d)) if d.size >= 100 else None)
            blocks.append(row)
        rec["rule_5_2"]["block_medians_10x10"] = blocks
        print("rule 5.2", med, fit["theil_sen_slope"], r["pass"], flush=True)
        # class-2 density over the TTV candidate pool and in hollows (Laplacian at 11 m > +0.02 1/m on the filled 1 m grid)
        pools_csv = os.path.join(a.out, "..", "pools", "TTV_windows.csv")
        if os.path.exists(pools_csv):
            import pandas as pd

            from lemcvt.measure.hilltop_curvature import total_curvature
            with rasterio.open(os.path.join(a.out, f"{a.site}_1m_class2_count.tif")) as ds:
                cnt = ds.read(1)
            with rasterio.open(os.path.join(a.out, f"{a.site}_1m_filled.tif")) as ds:
                zf = ds.read(1).astype(np.float64)
            zf[zf == NODATA] = np.nan
            dfw = pd.read_csv(pools_csv)
            dens = []
            for _, w in dfw.iterrows():
                c0 = int(round(w.x0 - tr.c))
                r0 = int(round(tr.f - w.y1))
                Wc = int(round(w.x1 - w.x0))
                if r0 < 0 or c0 < 0 or r0 + Wc > ny or c0 + Wc > nx:
                    continue
                cw = cnt[r0:r0 + Wc, c0:c0 + Wc]
                zw = zf[r0:r0 + Wc, c0:c0 + Wc]
                if not np.isfinite(zw).all():
                    dens.append({"offset": int(w.offset), "window": int(w.window), "complete": False})
                    continue
                lap = -total_curvature(zw, 5, 1.0)          # -C: positive = convex; hollows are concave: lap_grid = C > 0.02
                hollow = (-lap) > 0.02
                dens.append({"offset": int(w.offset), "window": int(w.window), "complete": True, "density_pts_m2": float(cw.mean()), "frac_empty_cells": float((cw == 0).mean()),
                             "hollow_frac": float(np.nanmean(hollow)), "density_hollows": float(cw[hollow].mean()) if hollow.any() else None,
                             "frac_empty_hollows": float((cw[hollow] == 0).mean()) if hollow.any() else None,
                             "density_divergent": float(cw[(-lap) < -0.02].mean()) if ((-lap) < -0.02).any() else None})
            dd = pd.DataFrame(dens)
            dd.to_csv(os.path.join(a.out, "TTV_pool_density.csv"), index=False)
            ok_ = dd[dd.complete] if len(dd) else dd
            rec["pool_density"] = {"n_windows": int(len(dd)), "n_complete": int(len(ok_)),
                                   "density_pts_m2_percentiles_5_50_95": [float(v) for v in ok_.density_pts_m2.quantile([0.05, 0.5, 0.95])] if len(ok_) else None,
                                   "density_hollows_percentiles_5_50_95": [float(v) for v in ok_.density_hollows.dropna().quantile([0.05, 0.5, 0.95])] if len(ok_) else None,
                                   "frac_empty_cells_median": float(ok_.frac_empty_cells.median()) if len(ok_) else None,
                                   "frac_empty_hollows_median": float(ok_.frac_empty_hollows.dropna().median()) if len(ok_) else None,
                                   "hollow_definition": "cells with Laplacian (least-squares quadratic, 11 m window) > +0.02 1/m on the filled 1 m grid (the addendum 13 stand-in)"}
            print("pool density", rec["pool_density"], flush=True)
    dump(rec, os.path.join(a.out, f"{a.site}_checks.json"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--site", required=True, choices=["AP", "TTV", "GMF"])
    p.add_argument("--files", help="text file: one LAS/LAZ path per line")
    p.add_argument("--out", required=True)
    p.add_argument("--footprint", required=True, help="GeoJSON polygon (any CRS) that bounds the lattice and masks the fill")
    p.add_argument("--stage", default="all", choices=["grid", "route", "check", "all"])
    p.add_argument("--data", default=None)
    p.add_argument("--work", default=None, help="node-local scratch for unzipping (default $L_SCRATCH_JOB)")
    p.add_argument("--max-fill", type=float, default=10.0, help="nearest-neighbour fill only within this distance (m); farther empty cells stay nodata")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    slurm_record(a.out)
    epsg_out = EPSG[a.site]
    tr, shape, g, srs = lattice_from_footprint(a.footprint, epsg_out)
    print(a.site, "lattice", shape, list(tr)[:6], "EPSG", epsg_out, flush=True)
    if a.stage in ("grid", "all"):
        inside = rasterize_geom(g, srs, tr, shape)
        stage_grid(a, tr, shape, inside, epsg_out)
        del inside
    if a.stage in ("route", "all"):
        stage_route(a, tr, shape, epsg_out)
    if a.stage in ("check", "all") and a.site != "GMF":
        stage_check(a, tr, shape, epsg_out)


if __name__ == "__main__":
    main()
