#!/usr/bin/env python3
"""Addendum 14, sections 3.1, 4.1.1-4.1.3, 4.2 and 5.3: lidar footprints on the coarse lattices, the AP watershed footprint
and its 3DEP file list, and rule 5.3 (coarse grid against the lidar) for GM and AP.

For each site the fine lidar grid is block-averaged onto the coarse (selection) lattice by binning cell centres
(a14_common.bin_to_lattice), giving per coarse cell the number of fine cells, the number of valid fine cells and the
mean height. These grids are written to the run directory and are the footprint masks the pool script uses
("100 % valid" = every fine cell inside the coarse cell is valid and the coarse cell is fully inside the fine raster).

AP watershed footprint (4.1.1): the valid area of PAMAP_DEM_FIXED_REPROJECT_12-2025.tif at 30 m, polygonised, unioned and
buffered by 500 m; in NAD83(2011) / PA South ftUS (EPSG:6565) the 5,000 ft quadrants intersecting it are listed with the
`pasda_tiles_for_ap.py` naming rule, checked against the saved PASDA listing (sizes) and the 5K tile index (addendum 13
checks), and summed for the 60 GB gate. The same footprint in EPSG:3857 selects the USGS 1 km LAZ tiles of
PA_WesternPA_1_2019 from the Entwine source list (bounds per file) with sizes from the rockyweb LAZ listing (4.1.2).

usage: a14_footprints.py --out DIR [--ancillary DIR] [--listing-usgs HTML] [--ept-sources JSON] [--marin-index SHP]
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np
import rasterio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a13_pasda_check import quadrant_bounds  # noqa: E402
from a14_common import (  # noqa: E402
    SITES,
    SOURCE,
    TTV_LATTICE,
    bin_to_lattice,
    dump,
    dz_stats,
    slurm_record,
    theil_sen_sub,
    write_tif,
)
from pasda_tiles_for_ap import BLOCK, HALF, SP_SOUTH_FTUS  # noqa: E402

PASDA_BASE = "https://pasda.psu.edu/download/usgs/LiDAR2020/LAS/PA_South/"
ROCKY_BASE = "https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/LPC/Projects/PA_WesternPA_2019_D20/PA_WesternPA_1_2019/LAZ/"


def site_footprint(site, out, coarse_path=None):
    """Bin the lidar onto the coarse lattice; write fraction/mean grids; return the arrays and the rule 5.3 record."""
    S = SITES[site]
    coarse_path = coarse_path or S["coarse"]
    with rasterio.open(coarse_path) as ds:
        cz = ds.read(1).astype(np.float64)
        ctr, ccrs, cnod = ds.transform, ds.crs, ds.nodata
    cvalid = np.isfinite(cz)
    if cnod is not None and np.isfinite(cnod):
        cvalid &= cz != cnod
    with rasterio.open(S["lidar"]) as ls:
        ltr = ls.transform
        lidar_bounds = list(ls.bounds)
    nominal = (ctr.a * -ctr.e) / (ltr.a * -ltr.e)
    t0 = time.time()
    b = bin_to_lattice(S["lidar"], ctr, cz.shape, S["lidar_z_scale"], block_rows=512)
    n_tot, n_val = b["n_total"], b["n_valid"]
    frac_valid = np.where(n_tot > 0, n_val / np.maximum(n_tot, 1), 0.0)
    coverage = n_tot / nominal
    full = (n_val == n_tot) & (coverage >= 0.98)
    lmean = np.where(n_val > 0, b["sum_z"] / np.maximum(n_val, 1), np.nan)
    write_tif(os.path.join(out, f"{site}_lidar_valid_frac.tif"), frac_valid.astype(np.float32), ctr, ccrs)
    write_tif(os.path.join(out, f"{site}_lidar_coverage.tif"), coverage.astype(np.float32), ctr, ccrs)
    write_tif(os.path.join(out, f"{site}_lidar_full.tif"), full.astype(np.uint8), ctr, ccrs)
    write_tif(os.path.join(out, f"{site}_lidar_mean.tif"), np.where(np.isfinite(lmean), lmean, -9999).astype(np.float32), ctr, ccrs, nodata=-9999.0)
    rec = {"coarse": coarse_path, "coarse_shape": list(cz.shape), "coarse_transform": list(ctr)[:6], "coarse_epsg": ccrs.to_epsg(),
           "coarse_cell_m": [ctr.a, -ctr.e], "lidar": S["lidar"], "lidar_record_copy": S["lidar_record"], "lidar_bounds": lidar_bounds,
           "lidar_cell_m": [ltr.a, -ltr.e], "lidar_z_scale": S["lidar_z_scale"], "lidar_z_units": S["lidar_z_units"],
           "fine_cells_per_coarse_cell_nominal": nominal, "n_coarse_valid": int(cvalid.sum()), "n_coarse_cells": int(cz.size),
           "n_lidar_full_cells": int(full.sum()), "n_lidar_any_cells": int((n_val > 0).sum()),
           "lidar_full_area_km2": float(full.sum() * ctr.a * -ctr.e / 1e6), "lidar_any_area_km2": float((n_val > 0).sum() * ctr.a * -ctr.e / 1e6),
           "coarse_valid_area_km2": float(cvalid.sum() * ctr.a * -ctr.e / 1e6),
           "n_lidar_full_but_coarse_invalid": int((full & ~cvalid).sum()), "n_coarse_valid_but_no_lidar": int((cvalid & (n_val == 0)).sum()),
           "seconds_binning": time.time() - t0}
    sel = full & cvalid
    dz = cz[sel] - lmean[sel]
    fit = theil_sen_sub(lmean[sel], cz[sel]) if sel.sum() > 10 else {"theil_sen_slope": np.nan}
    lo, hi = S["ned_offset_expect_m"]
    med = float(np.median(dz)) if dz.size else np.nan
    rec["rule_5_3"] = {"n_cells": int(sel.sum()), "dz_coarse_minus_lidar": dz_stats(dz), "fit_coarse_on_lidar": fit,
                       "slope_pass_0.98_1.02": bool(0.98 <= fit["theil_sen_slope"] <= 1.02), "offset_expected_m": [lo, hi],
                       "offset_pass": bool(lo <= med <= hi), "note": "coarse grid against the lidar block-averaged onto the coarse cells; cells fully inside valid lidar"}
    rec["rule_5_3"]["pass"] = rec["rule_5_3"]["slope_pass_0.98_1.02"] and rec["rule_5_3"]["offset_pass"]
    # block medians of dz on a 10 x 10 grid of the coarse domain (spatial pattern of the offset)
    ny, nx = cz.shape
    blocks = []
    for bi in range(10):
        row = []
        for bj in range(10):
            m = np.zeros_like(sel)
            m[bi * ny // 10:(bi + 1) * ny // 10, bj * nx // 10:(bj + 1) * nx // 10] = True
            d = (cz - lmean)[sel & m]
            row.append(float(np.median(d)) if d.size >= 20 else None)
        blocks.append(row)
    rec["rule_5_3"]["block_medians_10x10"] = blocks
    print(site, "footprint:", rec["n_lidar_full_cells"], "full cells;", "rule 5.3 median", med, "slope", fit["theil_sen_slope"], flush=True)
    return rec, {"cz": cz, "cvalid": cvalid, "ctr": ctr, "ccrs": ccrs, "full": full, "any": n_val > 0}


def polygon_from_mask(mask, transform, crs_wkt):
    from osgeo import gdal, ogr, osr
    gdal.UseExceptions()
    ny, nx = mask.shape
    drv = gdal.GetDriverByName("MEM")
    ds = drv.Create("", nx, ny, 1, gdal.GDT_Byte)
    ds.SetGeoTransform((transform.c, transform.a, 0, transform.f, 0, transform.e))
    ds.SetProjection(crs_wkt)
    band = ds.GetRasterBand(1)
    band.WriteArray(mask.astype(np.uint8))
    srs = osr.SpatialReference()
    srs.ImportFromWkt(crs_wkt)
    vds = ogr.GetDriverByName("Memory").CreateDataSource("m")
    lyr = vds.CreateLayer("p", srs=srs, geom_type=ogr.wkbPolygon)
    lyr.CreateField(ogr.FieldDefn("v", ogr.OFTInteger))
    gdal.Polygonize(band, band, lyr, 0)
    geoms = ogr.Geometry(ogr.wkbMultiPolygon)
    n = 0
    for f in lyr:
        if f.GetField("v") == 1:
            geoms.AddGeometry(f.GetGeometryRef().Clone())
            n += 1
    u = geoms.UnionCascaded()
    return u, n


def ap_watershed_files(arrs, out, ancillary, listing_usgs, ept_sources):
    from osgeo import ogr, osr
    ogr.UseExceptions()
    ctr, ccrs = arrs["ctr"], arrs["ccrs"]
    poly, n_polys = polygon_from_mask(arrs["any"], ctr, ccrs.to_wkt())
    buf = poly.Buffer(500.0)
    rec = {"n_polygons_raw": n_polys, "valid_area_km2": poly.GetArea() / 1e6, "buffered_area_km2": buf.GetArea() / 1e6,
           "buffer_m": 500.0, "n_parts_buffered": buf.GetGeometryCount() if buf.GetGeometryName() == "MULTIPOLYGON" else 1,
           "envelope_utm17": buf.GetEnvelope()}
    src = osr.SpatialReference()
    src.ImportFromWkt(ccrs.to_wkt())
    src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    # PASDA quadrants in EPSG:6565 (ftUS)
    sp = osr.SpatialReference()
    sp.ImportFromEPSG(SP_SOUTH_FTUS)
    sp.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    g_sp = buf.Clone()
    g_sp.Transform(osr.CoordinateTransformation(src, sp))
    E0, E1, N0, N1 = g_sp.GetEnvelope()
    quads = []
    for qe in range(int(np.floor(E0 / HALF)), int(np.floor(E1 / HALF)) + 1):
        for qn in range(int(np.floor(N0 / HALF)), int(np.floor(N1 / HALF)) + 1):
            e_sw, n_sw = qe * HALF, qn * HALF
            ring = ogr.Geometry(ogr.wkbLinearRing)
            for px, py in [(e_sw, n_sw), (e_sw + HALF, n_sw), (e_sw + HALF, n_sw + HALF), (e_sw, n_sw + HALF), (e_sw, n_sw)]:
                ring.AddPoint(px, py)
            q = ogr.Geometry(ogr.wkbPolygon)
            q.AddGeometry(ring)
            if not q.Intersects(g_sp):
                continue
            be = np.floor(e_sw / BLOCK) * BLOCK
            bn = np.floor(n_sw / BLOCK) * BLOCK + BLOCK
            ew = "E" if e_sw - be >= HALF else "W"
            ns = "N" if n_sw >= bn - HALF else "S"
            name = f"{int(bn):06d}"[:4] + f"{int(be):07d}"[:4]
            quads.append(f"{name}PAS_{ns}{ew}_LAS.zip")
    quads = sorted(set(quads))
    # listing and index checks (addendum 13)
    html = open(os.path.join(ancillary, "PA_South_listing_20261117.html")).read()
    listed = {n: int(s) for _d, s, _h, n in re.findall(r'(\d+/\d+/\d+\s+\d+:\d+ [AP]M)\s+(\d+)\s+<A HREF="([^"]+)">([^<]+)</A>', html)}
    drv = ogr.GetDriverByName("ESRI Shapefile")
    idx_ds = drv.Open(os.path.join(ancillary, "tileindex", "QL2_5K_SPSouth_TileIndex", "QL2_5K_SPSouth_TileIndex.shp"))
    index = {}
    for f in idx_ds.GetLayer():
        g = f.GetGeometryRef()
        index[f.GetField("TILEID")] = g.GetEnvelope()
    # the 28 addendum 13 files are stored as <quadrant>_LAS.las
    have = {os.path.basename(p)[:-4] + ".zip" for p in os.listdir(os.path.join(SOURCE, "AP-3DEP", "las")) if p.endswith(".las")}
    rows = []
    for q in quads:
        tid = q.replace("_LAS.zip", "")
        qb = quadrant_bounds(q)
        r = {"file": q, "listed": q in listed, "bytes": listed.get(q), "in_index": tid in index, "already_downloaded": q in have}
        if tid in index:
            ix = index[tid]
            r["index_matches_rule_within_1ft"] = bool(all(abs(u - v) < 1.0 for u, v in zip((ix[0], ix[2], ix[1], ix[3]), qb, strict=False)))
        rows.append(r)
    total = sum(r["bytes"] or 0 for r in rows)
    remaining = sum(r["bytes"] or 0 for r in rows if not r["already_downloaded"])
    rec["pasda"] = {"n_quadrants": len(rows), "n_listed": sum(r["listed"] for r in rows), "absent": [r["file"] for r in rows if not r["listed"]],
                    "not_in_index": [r["file"] for r in rows if not r["in_index"]], "index_rule_mismatch": [r["file"] for r in rows if r.get("index_matches_rule_within_1ft") is False],
                    "n_already_downloaded": sum(r["already_downloaded"] for r in rows), "total_bytes": total, "total_GB": total / 1e9,
                    "remaining_bytes": remaining, "remaining_GB": remaining / 1e9, "gate_60GB_total": total <= 60e9,
                    "las_bytes_estimate_GB": total / 1e9 * 8.1 / 5.37, "files": rows}
    with open(os.path.join(out, "AP_watershed_pasda_urls.txt"), "w") as fh:
        for r in rows:
            if r["listed"] and not r["already_downloaded"]:
                fh.write(PASDA_BASE + r["file"] + "\n")
    # USGS LAZ tiles (EPSG:3857 bounds from the EPT source list; sizes from the rockyweb listing)
    if ept_sources and os.path.exists(ept_sources):
        web = osr.SpatialReference()
        web.ImportFromEPSG(3857)
        web.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        g_web = buf.Clone()
        g_web.Transform(osr.CoordinateTransformation(src, web))
        sizes = {}
        if listing_usgs and os.path.exists(listing_usgs):
            h2 = open(listing_usgs).read()
            for name, _d, s in re.findall(r'<a href="([^"]+\.laz)">[^<]*</a>\s*(\d{1,2}-\w{3}-\d{4} \d\d:\d\d)\s+(\d+)', h2):
                sizes[name] = int(s)
        srcs = json.load(open(ept_sources))
        usgs = []
        for s in srcs:
            b = s["bounds"]
            ring = ogr.Geometry(ogr.wkbLinearRing)
            for px, py in [(b[0], b[1]), (b[3], b[1]), (b[3], b[4]), (b[0], b[4]), (b[0], b[1])]:
                ring.AddPoint(px, py)
            q = ogr.Geometry(ogr.wkbPolygon)
            q.AddGeometry(ring)
            if q.Intersects(g_web):
                name = os.path.basename(s["path"])
                usgs.append({"file": name, "points": s.get("points"), "bytes": sizes.get(name)})
        tot_u = sum(u["bytes"] or 0 for u in usgs)
        rec["usgs_laz"] = {"n_files": len(usgs), "n_with_size": sum(u["bytes"] is not None for u in usgs), "total_bytes": tot_u, "total_GB": tot_u / 1e9,
                           "points": sum(u["points"] or 0 for u in usgs), "n_listing_files": len(sizes), "listing_total_GB": sum(sizes.values()) / 1e9,
                           "files": usgs, "base": ROCKY_BASE,
                           "note": "USGS 1 km tiles named by UTM 17N km (e.g. 17SND538398); the Entwine copy on AWS is EPSG:3857 and cannot equal PASDA's coordinates"}
        with open(os.path.join(out, "AP_watershed_usgs_laz_urls.txt"), "w") as fh:
            for u in usgs:
                fh.write(ROCKY_BASE + u["file"] + "\n")
    # save the footprint polygons
    gj = ogr.GetDriverByName("GeoJSON")
    for nm, g in (("AP_watershed_valid", poly), ("AP_watershed_buffer500", buf)):
        pth = os.path.join(out, nm + ".geojson")
        if os.path.exists(pth):
            os.remove(pth)
        ds = gj.CreateDataSource(pth)
        lyr = ds.CreateLayer(nm, srs=src, geom_type=ogr.wkbMultiPolygon)
        f = ogr.Feature(lyr.GetLayerDefn())
        f.SetGeometry(g)
        lyr.CreateFeature(f)
        ds = None
    return rec


def rasterize_index(shp, transform, shape, crs_wkt, out_path, where=None):
    """Byte raster: 1 where the coarse cell centre lies inside a polygon of the shapefile (reprojected on the fly)."""
    from osgeo import gdal, ogr
    gdal.UseExceptions()
    ny, nx = shape
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(out_path, nx, ny, 1, gdal.GDT_Byte, options=["COMPRESS=DEFLATE", "TILED=YES"])
    ds.SetGeoTransform((transform.c, transform.a, 0, transform.f, 0, transform.e))
    ds.SetProjection(crs_wkt)
    ds.GetRasterBand(1).Fill(0)
    vds = ogr.Open(shp)
    lyr = vds.GetLayer()
    if where:
        lyr.SetAttributeFilter(where)
    n = lyr.GetFeatureCount()
    gdal.RasterizeLayer(ds, [1], lyr, burn_values=[1], options=["ALL_TOUCHED=FALSE"])
    arr = ds.GetRasterBand(1).ReadAsArray()
    ds = None
    return arr.astype(bool), n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--ancillary", default=os.path.join(SOURCE, "AP-3DEP", "ancillary"))
    p.add_argument("--listing-usgs", default=None)
    p.add_argument("--ept-sources", default=None)
    p.add_argument("--marin-index", default=None)
    p.add_argument("--marin-boundary", default=None)
    p.add_argument("--ttv-ned", default=None, help="TTV_NED_10m_utm10n.tif from a14_ttv_inventory.py")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    slurm_record(a.out)
    rec = {}
    for site in ("GM", "AP"):
        r, arrs = site_footprint(site, a.out)
        rec[site] = r
        if site == "AP":
            # 3DEP tile index coverage of the AP coarse domain
            shp = os.path.join(a.ancillary, "tileindex", "QL2_5K_SPSouth_TileIndex_Clipped", "QL2_5K_SPSouth_TileIndex_Clipped.shp")
            cov, n = rasterize_index(shp, arrs["ctr"], arrs["cz"].shape, arrs["ccrs"].to_wkt(), os.path.join(a.out, "AP_3dep_index_cover.tif"))
            rec[site]["threedep_index"] = {"shapefile": shp, "n_features": n, "frac_coarse_valid_covered": float((cov & arrs["cvalid"]).sum() / max(1, arrs["cvalid"].sum())),
                                           "frac_lidar_full_covered": float((cov & arrs["full"]).sum() / max(1, arrs["full"].sum()))}
            rec[site]["watershed"] = ap_watershed_files(arrs, a.out, a.ancillary, a.listing_usgs, a.ept_sources)
            print("AP watershed:", {k: v for k, v in rec[site]["watershed"]["pasda"].items() if k != "files"}, flush=True)
            if "usgs_laz" in rec[site]["watershed"]:
                print("AP USGS LAZ:", {k: v for k, v in rec[site]["watershed"]["usgs_laz"].items() if k != "files"}, flush=True)
    # TTV: Marin 2019 (USGS NoCal B5b QL1, Lot 12) footprint on the TTV lattice
    if a.marin_index:
        from rasterio.transform import Affine
        L = TTV_LATTICE
        tr = Affine(L["dx"], 0, L["x0"], 0, -L["dx"], L["y1"])
        shape = (int((L["y1"] - L["y0"]) / L["dx"]), int((L["x1"] - L["x0"]) / L["dx"]))
        from rasterio.crs import CRS
        wkt = CRS.from_epsg(L["epsg"]).to_wkt()
        cov, n = rasterize_index(a.marin_index, tr, shape, wkt, os.path.join(a.out, "TTV_marin_index_cover.tif"))
        rec["TTV"] = {"marin_index": a.marin_index, "n_tiles_index": n, "n_cells_covered": int(cov.sum()), "covered_area_km2": float(cov.sum() * L["dx"] ** 2 / 1e6),
                      "lattice": L}
        if a.marin_boundary:
            covb, nb = rasterize_index(a.marin_boundary, tr, shape, wkt, os.path.join(a.out, "TTV_marin_boundary_cover.tif"))
            rec["TTV"]["boundary_cells_covered"] = int(covb.sum())
        if a.ttv_ned and os.path.exists(a.ttv_ned):
            with rasterio.open(a.ttv_ned) as ds:
                nz = ds.read(1)
            land = nz > 0
            rec["TTV"]["ned_land_cells_gt_0"] = int(land.sum())
            rec["TTV"]["ned_land_covered_by_marin_index"] = float((land & cov).sum() / max(1, land.sum()))
    dump(rec, os.path.join(a.out, "footprints.json"))
    print("wrote", os.path.join(a.out, "footprints.json"))


if __name__ == "__main__":
    main()
