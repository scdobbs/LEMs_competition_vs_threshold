#!/usr/bin/env python3
"""Addendum 14, 4.3.1: the Marin 2019 QL1 tiles (USGS CA_NoCal_Wildfires_B5b_QL1_2018, Lot 12) that cover the Marin
Headlands watersheds containing the TTV candidate pool, plus 500 m.

From the TTV coarse-grid routing (a14_pools.py: TTV_fd_full.tif, TTV_area_full.tif) every cell's terminal outlet is
labelled; the watersheds = the union of terminal basins that contain any cell of a TTV candidate window (TTV_windows.csv).
The union is polygonised on the 10 m lattice, buffered by 500 m and intersected with the Lot 12 tile index (Albers,
NAD83(2011)); the file names follow the Entwine source list (USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_<TileID>.laz) and the
size gate (30 GB) is checked from the rockyweb LAZ listing when available, else estimated from the Entwine point counts.

usage: a14_marin_select.py --pools DIR --out DIR --index SHP --ept-sources JSON [--listing HTML] [--ncalm-only]
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a14_common import dump, slurm_record, write_tif  # noqa: E402
from a14_pools import _terminal  # noqa: E402

from lemcvt.hydro.d8 import d8_receiver_arrays  # noqa: E402

ROCKY_B5B = "https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/LPC/Projects/CA_NoCAL_3DEP_Supp_Funding_2018_D18/CA_NoCal_Wildfires_B5b_QL1_2018/LAZ/"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pools", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--index", required=True)
    p.add_argument("--ept-sources", required=True)
    p.add_argument("--listing", default=None)
    p.add_argument("--buffer", type=float, default=500.0)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    slurm_record(a.out)
    from osgeo import gdal, ogr, osr
    gdal.UseExceptions()
    with rasterio.open(os.path.join(a.pools, "TTV_fd_full.tif")) as ds:
        fd = ds.read(1)
        tr, crs = ds.transform, ds.crs
    with rasterio.open(os.path.join(a.pools, "TTV_area_full.tif")) as ds:
        area = ds.read(1).astype(np.float64)
        area[area == ds.nodata] = np.nan
    ny, nx = fd.shape
    valid = np.isfinite(area)
    r_to, c_to, good = d8_receiver_arrays(fd)
    target = np.where(good.ravel() & valid.ravel(), r_to.ravel() * nx + c_to.ravel(), -1).astype(np.int64)
    bad = (target >= 0) & ~valid.ravel()[np.maximum(target, 0)]
    target[bad] = -1
    # receivers before donors: ascending area is not safe with ties; use area descending reversed = ascending area, ties by index
    af = np.where(valid, area, -np.inf).ravel()
    order_asc = np.lexsort((np.arange(af.size), af))      # ascending area: receivers (larger area) come later... invert
    order_asc = order_asc[::-1].copy()                      # descending area = receivers first
    lab = np.arange(ny * nx, dtype=np.int64)
    lab = np.asarray(_terminal(order_asc, target, lab)).reshape(ny, nx)
    lab[~valid] = -1
    df = pd.read_csv(os.path.join(a.pools, "TTV_windows.csv"))
    W = int(json.load(open(os.path.join(a.pools, "TTV_pools.json")))["W_cells"])
    basins = set()
    for _, w in df.iterrows():
        sub = lab[int(w.row0):int(w.row0) + W, int(w.col0):int(w.col0) + W]
        basins.update(np.unique(sub[sub >= 0]).tolist())
    mask = np.isin(lab, list(basins))
    write_tif(os.path.join(a.out, "TTV_pool_watersheds_10m.tif"), mask.astype(np.uint8), tr, crs)
    rec = {"n_candidate_windows": int(len(df)), "n_terminal_basins": len(basins), "watershed_cells": int(mask.sum()), "watershed_km2": float(mask.sum() * tr.a * -tr.e / 1e6)}
    # polygonise, buffer
    drv = gdal.GetDriverByName("MEM")
    ds = drv.Create("", nx, ny, 1, gdal.GDT_Byte)
    ds.SetGeoTransform((tr.c, tr.a, 0, tr.f, 0, tr.e))
    ds.SetProjection(crs.to_wkt())
    band = ds.GetRasterBand(1)
    band.WriteArray(mask.astype(np.uint8))
    srs = osr.SpatialReference()
    srs.ImportFromWkt(crs.to_wkt())
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    vds = ogr.GetDriverByName("MEM").CreateDataSource("m")
    lyr = vds.CreateLayer("p", srs=srs, geom_type=ogr.wkbPolygon)
    lyr.CreateField(ogr.FieldDefn("v", ogr.OFTInteger))
    gdal.Polygonize(band, band, lyr, 0)
    geoms = ogr.Geometry(ogr.wkbMultiPolygon)
    for f in lyr:
        if f.GetField("v") == 1:
            geoms.AddGeometry(f.GetGeometryRef().Clone())
    poly = geoms.UnionCascaded()
    buf = poly.Buffer(a.buffer)
    rec["buffered_km2"] = buf.GetArea() / 1e6
    gj = ogr.GetDriverByName("GeoJSON")
    for nm, g in (("TTV_pool_watersheds", poly), ("TTV_pool_watersheds_buffer500", buf)):
        pth = os.path.join(a.out, nm + ".geojson")
        if os.path.exists(pth):
            os.remove(pth)
        d2 = gj.CreateDataSource(pth)
        l2 = d2.CreateLayer(nm, srs=srs, geom_type=ogr.wkbMultiPolygon)
        f = ogr.Feature(l2.GetLayerDefn())
        f.SetGeometry(g)
        l2.CreateFeature(f)
        d2 = None
    # tiles
    ids = ogr.Open(a.index)
    il = ids.GetLayer()
    isrs = il.GetSpatialRef()
    isrs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    g_alb = buf.Clone()
    g_alb.Transform(osr.CoordinateTransformation(srs, isrs))
    il.SetSpatialFilter(g_alb)
    tiles = []
    for f in il:
        if f.GetGeometryRef().Intersects(g_alb):
            tiles.append(f.GetField("Tile_ID") or f.GetField("TileID"))
    tiles = sorted(set(tiles))
    srcs = {os.path.basename(s["path"]): s for s in json.load(open(a.ept_sources))}
    sizes = {}
    if a.listing and os.path.exists(a.listing):
        for name, _d, s in re.findall(r'<a href="([^"]+\.laz)">[^<]*</a>\s*(\d{1,2}-\w{3}-\d{4} \d\d:\d\d)\s+(\d+)', open(a.listing).read()):
            sizes[name] = int(s)
    rows = []
    for t in tiles:
        name = f"USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_{t}.laz"
        s = srcs.get(name)
        rows.append({"tile": t, "file": name, "in_ept_sources": s is not None, "points": s["points"] if s else None, "bytes_listed": sizes.get(name)})
    pts = sum(r["points"] or 0 for r in rows)
    byt = sum(r["bytes_listed"] or 0 for r in rows)
    rec.update({"n_tiles": len(rows), "n_in_ept_sources": sum(r["in_ept_sources"] for r in rows), "points_total": pts, "bytes_listed_total": byt,
                "bytes_estimate_from_points_at_3.5B": pts * 3.5, "gate_30GB_listed": byt <= 30e9 if byt else None, "gate_30GB_estimate": pts * 3.5 <= 30e9, "tiles": rows})
    with open(os.path.join(a.out, "marin_urls.txt"), "w") as fh:
        for r in rows:
            fh.write(ROCKY_B5B + r["file"] + "\n")
    dump(rec, os.path.join(a.out, "marin_select.json"))
    print(json.dumps({k: v for k, v in rec.items() if k != "tiles"}, indent=1))


if __name__ == "__main__":
    main()
