#!/usr/bin/env python3
"""Addendum 17, section 2.2.1: the fetch footprint of the GM survey-control grid (GMC).

GM1-GM7 (their registered 1 km footprints, EPSG:32610) and their upstream catchments on the 10 m NED routed over the whole
addendum 16 box (a16_gm_area_diag conventions: GM_NED_10m_utm10n.tif over GM_GATE_CROP_UTM, every valid cell, data-edge
outlets), plus 500 m, intersected with the FEMA 2018 survey footprint. The 1 km fetch boxes (EPSG:6339 lattice, as
a14_ept_fetch.boxes_for_footprint cuts them) are split into those already in the copy of record (addendum 16, 292 boxes) and the
missing ones; the missing boxes' union (shrunk by 1 m so that touching neighbours are not re-fetched) is the fetch footprint of
the next job. Outputs (run dir gm_ctrl/): GMC_fetch_footprint_10m.tif, GMC_fetch_footprint.geojson (EPSG:32610),
GMC_fetch_missing.geojson (EPSG:6339; absent if nothing is missing), GMC_existing_files.txt (record paths), GMC_boxes.json,
GMC_fetch_footprint.json, GMC_catchments_10m.tif (tile id per upstream cell).
usage: a17_gm_ctrl_footprint.py --out DIR --footprints DIR [--data $LEMCVT_DATA]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import rasterio
from rasterio.windows import from_bounds

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a14_ept_fetch import boxes_for_footprint  # noqa: E402
from a14_pools import route_full_grid  # noqa: E402
from a16_gm_ext_footprint import _propagate_up  # noqa: E402
from a17_common import (  # noqa: E402
    CTRL_BUFFER_M,
    CTRL_TILES,
    FEMA_RECORD,
    FEMA_SURVEY,
    GM_GATE_CROP_UTM,
    dump,
    slurm_record,
    write_tif,
)


def read_crop(path, box):
    with rasterio.open(path) as ds:
        b = ds.bounds
        assert b.left <= box[0] and b.bottom <= box[1] and box[2] <= b.right and box[3] <= b.top, (path, box, tuple(b))
        w = from_bounds(box[0], box[1], box[2], box[3], ds.transform)
        return ds.read(1, window=w), ds.window_transform(w), ds.crs, ds.nodata


def polygonize(mask, tr, epsg, path, layer="footprint"):
    from osgeo import gdal, ogr, osr
    ogr.UseExceptions()
    ny, nx = mask.shape
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    mem = gdal.GetDriverByName("MEM").Create("", nx, ny, 1, gdal.GDT_Byte)
    mem.SetGeoTransform((tr.c, tr.a, 0.0, tr.f, 0.0, tr.e))
    mem.SetProjection(srs.ExportToWkt())
    mem.GetRasterBand(1).WriteArray(mask.astype(np.uint8))
    tmp_ds = ogr.GetDriverByName("MEM").CreateDataSource("t")
    tmp = tmp_ds.CreateLayer("t", srs, ogr.wkbPolygon)
    tmp.CreateField(ogr.FieldDefn("v", ogr.OFTInteger))
    gdal.Polygonize(mem.GetRasterBand(1), mem.GetRasterBand(1), tmp, 0, ["8CONNECTED=8"])
    geom = ogr.Geometry(ogr.wkbMultiPolygon)
    for f in tmp:
        if f.GetField("v") == 1:
            geom.AddGeometry(f.GetGeometryRef().Clone())
    geom = geom.UnionCascaded() if hasattr(geom, "UnionCascaded") else geom.Buffer(0)
    if os.path.exists(path):
        os.remove(path)
    ds = ogr.GetDriverByName("GeoJSON").CreateDataSource(path)
    lyr = ds.CreateLayer(layer, srs, ogr.wkbMultiPolygon)
    f = ogr.Feature(lyr.GetLayerDefn())
    f.SetGeometry(geom)
    lyr.CreateFeature(f)
    ds = None
    return geom.GetArea()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--footprints", required=True)
    p.add_argument("--data", default=os.environ.get("LEMCVT_DATA"))
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    slurm_record(a.out)
    t0 = time.time()
    # tile footprints from the registered GM-x grids
    fps = {}
    for t in CTRL_TILES:
        with rasterio.open(os.path.join(a.data, "GM-x", t, f"{t}_1m_filled.tif")) as ds:
            b = ds.bounds
            assert ds.crs.to_epsg() == 32610 and ds.width == ds.height == 1000, t
            fps[t] = [float(b.left), float(b.bottom)]
    box = GM_GATE_CROP_UTM
    z, tr, crs, nod = read_crop(os.path.join(a.footprints, "GM_NED_10m_utm10n.tif"), box)
    z = z.astype(np.float64)
    z[z == nod] = np.nan
    m, _, _, _ = read_crop(os.path.join(a.footprints, f"GM_survey_{FEMA_SURVEY}_10m.tif"), box)
    survey = m.astype(bool)
    dx = float(tr.a)
    ny, nx = z.shape
    R = route_full_grid(z, dx)
    print("NED box routed", z.shape, round(R["seconds"]), "s; max area km2", float(np.nanmax(R["area"])) / 1e6, flush=True)
    cx = tr.c + (np.arange(nx) + 0.5) * tr.a
    cy = tr.f + (np.arange(ny) + 0.5) * tr.e
    tiles_id = np.zeros(z.shape, np.int16)
    catch = np.zeros(z.shape, np.int16)
    per = {}
    for k, t in enumerate(CTRL_TILES, start=1):
        x0, y0 = fps[t]
        sq = ((cx >= x0) & (cx < x0 + 1000))[None, :] & ((cy >= y0) & (cy < y0 + 1000))[:, None]
        tiles_id[sq] = k
        flag = np.asarray(_propagate_up(R["order_asc"], R["target"], sq.ravel().copy())).reshape(ny, nx) & np.isfinite(z)
        up = flag & ~sq
        catch[up & (catch == 0)] = k
        per[t] = {"fp_sw_utm10n": [x0, y0], "tile_cells_10m": int(sq.sum()), "catchment_km2_total": float(flag.sum() * dx * dx / 1e6), "upstream_outside_tile_km2": float(up.sum() * dx * dx / 1e6),
                  "upstream_cells_outside_survey": int((up & ~survey).sum()), "max_area_in_tile_km2": float(np.nanmax(np.where(sq, R["area"], np.nan))) / 1e6}
        print(t, per[t], flush=True)
    union = (tiles_id > 0) | (catch > 0)
    from scipy.ndimage import binary_dilation
    nb = int(round(CTRL_BUFFER_M / dx))
    yy, xx = np.ogrid[-nb:nb + 1, -nb:nb + 1]
    disk = (xx * xx + yy * yy) <= nb * nb
    fp = binary_dilation(union, structure=disk) & survey
    write_tif(os.path.join(a.out, "GMC_fetch_footprint_10m.tif"), fp.astype(np.uint8), tr, crs)
    write_tif(os.path.join(a.out, "GMC_catchments_10m.tif"), np.where(tiles_id > 0, -tiles_id, catch).astype(np.int16), tr, crs)
    gj = os.path.join(a.out, "GMC_fetch_footprint.geojson")
    poly_area = polygonize(fp, tr, 32610, gj)
    # boxes on the EPSG:6339 lattice, split into existing (copy of record) and missing
    boxes, area_6339 = boxes_for_footprint(gj, 6339, 1000.0)
    names = [b["name"] for b in boxes]
    existing = [n for n in names if os.path.exists(os.path.join(FEMA_RECORD, f"ept_{n}.laz"))]
    missing = [n for n in names if n not in set(existing)]
    with open(os.path.join(a.out, "GMC_existing_files.txt"), "w") as fh:
        for n in existing:
            fh.write(os.path.join(FEMA_RECORD, f"ept_{n}.laz") + "\n")
    miss_path = os.path.join(a.out, "GMC_fetch_missing.geojson")
    if os.path.exists(miss_path):
        os.remove(miss_path)
    if missing:
        from osgeo import ogr, osr
        ogr.UseExceptions()
        u = ogr.Geometry(ogr.wkbMultiPolygon)
        for b in boxes:
            if b["name"] in set(missing):
                u.AddGeometry(ogr.CreateGeometryFromWkt(b["polygon_target"]))
        u = u.UnionCascaded() if hasattr(u, "UnionCascaded") else u.Buffer(0)
        u = u.Buffer(-1.0)                      # touching neighbours would otherwise "intersect" the union
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(6339)
        ds = ogr.GetDriverByName("GeoJSON").CreateDataSource(miss_path)
        lyr = ds.CreateLayer("missing", srs, ogr.wkbMultiPolygon)
        f = ogr.Feature(lyr.GetLayerDefn())
        f.SetGeometry(u)
        lyr.CreateFeature(f)
        ds = None
    rr_, cc_ = np.nonzero(fp)
    bbox = [float(tr.c + cc_.min() * tr.a), float(tr.f + (rr_.max() + 1) * tr.e), float(tr.c + (cc_.max() + 1) * tr.a), float(tr.f + rr_.min() * tr.e)]
    rec = {"tiles": per, "footprints_utm10n": fps, "buffer_m": CTRL_BUFFER_M, "survey": FEMA_SURVEY, "crop_utm": list(box), "ned_routing_seconds": R["seconds"],
           "union_tiles_catchments_km2": float(union.sum() * dx * dx / 1e6), "union_cells_outside_survey": int((union & ~survey).sum()),
           "footprint_km2": float(fp.sum() * dx * dx / 1e6), "polygon_km2": poly_area / 1e6, "polygon_km2_6339": area_6339 / 1e6, "bbox_utm10n": bbox,
           "lattice_cells_1m_M": float((bbox[2] - bbox[0] + 10) * (bbox[3] - bbox[1] + 10) / 1e6),
           "boxes": {"n": len(names), "n_existing_record": len(existing), "n_missing": len(missing), "missing": missing, "existing": existing},
           "expected_points_missing_M": float(len(missing) * 1e6 * 3.4 / 1e6), "seconds": time.time() - t0}
    dump({k: v for k, v in rec.items() if k != "boxes"} | {"boxes": rec["boxes"]}, os.path.join(a.out, "GMC_fetch_footprint.json"))
    dump(rec["boxes"], os.path.join(a.out, "GMC_boxes.json"))
    print(json.dumps({k: v for k, v in rec.items() if k not in ("boxes", "tiles")}, indent=1), "boxes", len(names), "existing", len(existing), "missing", len(missing), flush=True)


if __name__ == "__main__":
    main()
