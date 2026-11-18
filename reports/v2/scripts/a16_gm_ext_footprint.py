#!/usr/bin/env python3
"""Addendum 16, section 3.3.1: the point-cloud fetch footprint of the GM extension survey.

The survey that passed the gate (a16_common.GM_GATE_SURVEYS[0]) is fetched over the union, for every eligible window of the gate
(eligible at the tolerance the widening rule selected, not overlapping GM1-GM7, in the landscape), of the window and its upstream
catchment on the 10 m routing (every cell whose D8 path enters the window), buffered by 500 m, so that every eligible window's
upstream area is in the fetched data and the 1 m full-grid routing can apply the "upstream area fully in valid data" rule and the
gap rule. (A first version took the terminal basins, as stage A did at TTV where the basins drain to the sea; here every window drains
to the Salinas and that union was the whole crop, 1,362 km2.) The routing is the gate's (a14_pools.route_full_grid on the NED crop
masked to the survey footprint).
Outputs (run dir gm_ext/): GMF_fetch_footprint.geojson (EPSG:32610; the fetch reprojects to EPSG:6339), GMF_fetch_footprint.json,
GMF_eligible_windows.csv.
usage: a16_gm_ext_footprint.py --out DIR --gate DIR --footprints DIR [--survey NAME]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a14_pools import crop, route_full_grid  # noqa: E402

try:
    import numba as _nb

    @_nb.njit(cache=True)
    def _propagate_up(order_asc, target, flag):
        # receivers before donors (ascending elevation): a cell is upstream of the windows if its receiver is
        for k in order_asc:
            t = target[k]
            if t >= 0 and flag[t]:
                flag[k] = True
        return flag
except Exception:  # pragma: no cover
    def _propagate_up(order_asc, target, flag):
        for k in order_asc:
            t = target[k]
            if t >= 0 and flag[t]:
                flag[k] = True
        return flag
from a16_common import GM_GATE_CROP_UTM, GM_GATE_SURVEYS, dump, slurm_record, write_tif  # noqa: E402

BUFFER_M = 500.0


def read_crop(path, box):
    with rasterio.open(path) as ds:
        b = ds.bounds
        assert b.left <= box[0] and b.bottom <= box[1] and box[2] <= b.right and box[3] <= b.top, (path, box, tuple(b))
        w = from_bounds(box[0], box[1], box[2], box[3], ds.transform)
        return ds.read(1, window=w), ds.window_transform(w), ds.crs, ds.nodata


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--gate", required=True)
    p.add_argument("--footprints", required=True)
    p.add_argument("--survey", default=GM_GATE_SURVEYS[0])
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    slurm_record(a.out)
    t0 = time.time()
    gate = json.load(open(os.path.join(a.gate, "gm_gate.json")))
    g = gate["gate"][a.survey]
    tol = g["tolerance_used"]
    df = pd.read_csv(os.path.join(a.gate, f"{a.survey}_windows.csv"))
    box = GM_GATE_CROP_UTM
    z, tr, crs, nod = read_crop(os.path.join(a.footprints, "GM_NED_10m_utm10n.tif"), box)
    z = z.astype(np.float64)
    z[z == nod] = np.nan
    land, _, _, _ = read_crop(os.path.join(a.gate, "GM_landscape_10m.tif"), box)
    m, _, _, _ = read_crop(os.path.join(a.footprints, f"GM_survey_{a.survey}_10m.tif"), box)
    W = 100
    in_land = np.array([crop(land.astype(bool), int(r.row0), int(r.col0), W).all() for r in df.itertuples()], bool)
    sel = df[f"eligible{tol}"].values & ~df["overlaps_GM1_7"].values & in_land
    E = df[sel].reset_index(drop=True)
    E.to_csv(os.path.join(a.out, "GMF_eligible_windows.csv"), index=False)
    print(a.survey, "tolerance", tol, "eligible windows for the extension:", len(E), flush=True)
    zs = np.where(m.astype(bool), z, np.nan)
    R = route_full_grid(zs, float(tr.a))
    ny, nx = z.shape
    wmask = np.zeros(z.shape, bool)
    for r in E.itertuples():
        r0, c0 = int(r.row0), int(r.col0)
        wmask[r0:r0 + W, c0:c0 + W] = True
    # upstream catchment of the windows: propagate the window flag from receivers to donors (ascending filled elevation)
    flag = wmask.ravel().copy()
    flag = np.asarray(_propagate_up(R["order_asc"], R["target"], flag)).reshape(ny, nx)
    bm = flag & np.isfinite(zs) & ~wmask
    from scipy.ndimage import binary_dilation
    nb = int(round(BUFFER_M / tr.a))
    yy, xx = np.ogrid[-nb:nb + 1, -nb:nb + 1]
    disk = (xx * xx + yy * yy) <= nb * nb
    fp = binary_dilation(bm | wmask, structure=disk) & m.astype(bool)
    write_tif(os.path.join(a.out, "GMF_fetch_footprint_10m.tif"), fp.astype(np.uint8), tr, crs)
    # polygonise
    from osgeo import gdal, ogr, osr
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32610)
    mem = gdal.GetDriverByName("MEM").Create("", nx, ny, 1, gdal.GDT_Byte)
    mem.SetGeoTransform((tr.c, tr.a, 0.0, tr.f, 0.0, tr.e))
    mem.SetProjection(srs.ExportToWkt())
    mem.GetRasterBand(1).WriteArray(fp.astype(np.uint8))
    tmp_ds = ogr.GetDriverByName("MEM").CreateDataSource("t")
    tmp = tmp_ds.CreateLayer("t", srs, ogr.wkbPolygon)
    tmp.CreateField(ogr.FieldDefn("v", ogr.OFTInteger))
    gdal.Polygonize(mem.GetRasterBand(1), mem.GetRasterBand(1), tmp, 0, ["8CONNECTED=8"])
    geom = ogr.Geometry(ogr.wkbMultiPolygon)
    for f in tmp:
        if f.GetField("v") == 1:
            geom.AddGeometry(f.GetGeometryRef().Clone())
    geom = geom.UnionCascaded() if hasattr(geom, "UnionCascaded") else geom.Buffer(0)
    path = os.path.join(a.out, "GMF_fetch_footprint.geojson")
    if os.path.exists(path):
        os.remove(path)
    ds = ogr.GetDriverByName("GeoJSON").CreateDataSource(path)
    lyr = ds.CreateLayer("footprint", srs, ogr.wkbMultiPolygon)
    f = ogr.Feature(lyr.GetLayerDefn())
    f.SetGeometry(geom)
    lyr.CreateFeature(f)
    ds = None
    rr_, cc_ = np.nonzero(fp)
    bbox = [float(tr.c + cc_.min() * tr.a), float(tr.f + (rr_.max() + 1) * tr.e), float(tr.c + (cc_.max() + 1) * tr.a), float(tr.f + rr_.min() * tr.e)]
    rec = {"survey": a.survey, "bbox_utm": bbox, "lattice_cells_1m_G": float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / 1e9), "tolerance": tol, "n_eligible_windows": int(len(E)), "upstream_km2": float(bm.sum() * tr.a * tr.a / 1e6),
           "windows_km2": float(wmask.sum() * tr.a * tr.a / 1e6), "buffer_m": BUFFER_M, "footprint_km2": float(fp.sum() * tr.a * tr.a / 1e6), "polygon_km2": geom.GetArea() / 1e6,
           "expected_points_all_returns_G": float(fp.sum() * tr.a * tr.a * 5.56 / 1e9), "geojson": path, "seconds": time.time() - t0}
    dump(rec, os.path.join(a.out, "GMF_fetch_footprint.json"))
    print(json.dumps(rec, indent=1), flush=True)


if __name__ == "__main__":
    main()
