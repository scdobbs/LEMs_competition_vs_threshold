#!/usr/bin/env python3
"""Addendum 16, section 2.1 (decision 1): clip the digitised M&D (1989) study-area polygon at the 61 m contour.

Polygon (stage B, md_georef/md1989_fig3.gpkg layer study_area, rasterised at 1 m as md_study_area_1m.tif over the working
window WIN of a15_md_georef.py)  ∩  {z >= 61 m on the TTV 1 m grid (stage A TTV_1m_mean.tif, the measured surface that
rule 2.2 was evaluated on)}; keep the largest 8-connected component; fill its holes. Then rule 2.2 of addendum 15 again:
area 2.1 km2 +- 15 % (binding), z_min 61 +- 15 m (passes by construction: every kept cell is >= 61 m), z_max 314 +- 15 m.
Products (run dir md_georef/): md1989_fig3_a16.gpkg (a copy of the stage B GeoPackage with the layer study_area_clipped
added; the stage B file is not modified), md_study_area_clipped_1m.tif, removed_cells_1m.tif (label of the removed
cluster), clip.json, rule_2_2_a16.json, clip_map.png.
usage: a16_ttv_clip.py --out DIR
"""
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import from_bounds
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a15_md_georef import TTV_GRID, WIN  # noqa: E402
from a16_common import STAGE_B, TTV_CLIP_Z, dump, slurm_record, write_tif  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    slurm_record(a.out)
    t0 = time.time()
    wx0, wx1, wy0, wy1 = WIN
    # the stage B 1 m polygon mask and the measured 1 m surface over the same window
    with rasterio.open(os.path.join(STAGE_B, "md_georef", "md_study_area_1m.tif")) as ds:
        inside = ds.read(1).astype(bool)
        tr = ds.transform
        assert abs(tr.c - wx0) < 1e-6 and abs(tr.f - wy1) < 1e-6, (tr, WIN)
    with rasterio.open(os.path.join(TTV_GRID, "TTV_1m_mean.tif")) as ds:
        w = from_bounds(wx0, wy0, wx1, wy1, ds.transform)
        z = ds.read(1, window=w).astype(np.float64)
        nod = ds.nodata
    z[z == nod] = np.nan
    assert z.shape == inside.shape, (z.shape, inside.shape)
    rec = {"source_polygon": os.path.join(STAGE_B, "md_georef", "md1989_fig3.gpkg"), "source_mask": "md_study_area_1m.tif", "grid": "stage A TTV_1m_mean.tif (measured surface)",
           "clip_z_m": TTV_CLIP_Z, "window": WIN, "cells_before": int(inside.sum()), "area_before_km2": float(inside.sum() / 1e6),
           "z_min_before": float(np.nanmin(z[inside])), "z_max_before": float(np.nanmax(z[inside])), "nodata_cells_inside_before": int(np.isnan(z[inside]).sum())}
    keep = inside & (z >= TTV_CLIP_Z)
    rec["cells_below_clip"] = int((inside & ~(z >= TTV_CLIP_Z)).sum())
    # largest 8-connected component
    lab, n = ndimage.label(keep, structure=np.ones((3, 3), bool))
    sizes = ndimage.sum(keep, lab, range(1, n + 1))
    big = int(np.argmax(sizes)) + 1
    rec["components_after_clip"] = int(n)
    rec["component_sizes_desc"] = sorted([int(s) for s in sizes], reverse=True)[:10]
    comp = lab == big
    # fill holes
    filled = ndimage.binary_fill_holes(comp)
    rec["cells_dropped_as_minor_components"] = int(keep.sum() - comp.sum())
    rec["cells_filled_as_holes"] = int(filled.sum() - comp.sum())
    rec["holes_filled_z_min"] = float(np.nanmin(z[filled & ~comp])) if (filled & ~comp).any() else None
    clipped = filled
    # removed cells (in the original polygon, not in the clipped one), clustered
    removed = inside & ~clipped
    rlab, rn = ndimage.label(removed, structure=np.ones((3, 3), bool))
    clusters = []
    for k in range(1, rn + 1):
        m = rlab == k
        rr, cc = np.nonzero(m)
        zz = z[m]
        clusters.append({"id": k, "cells": int(m.sum()), "area_km2": float(m.sum() / 1e6), "centroid_E": float(wx0 + cc.mean() + 0.5), "centroid_N": float(wy1 - rr.mean() - 0.5),
                         "z_min": float(np.nanmin(zz)) if np.isfinite(zz).any() else None, "z_max": float(np.nanmax(zz)) if np.isfinite(zz).any() else None,
                         "cells_below_clip": int((zz < TTV_CLIP_Z).sum()), "cells_nodata": int(np.isnan(zz).sum())})
    clusters.sort(key=lambda c: -c["cells"])
    rec["removed_clusters"] = clusters
    rec["removed_cells_total"] = int(removed.sum())
    rec["added_cells_total"] = int((clipped & ~inside).sum())
    # after
    zin = z[clipped]
    rec.update({"cells_after": int(clipped.sum()), "area_after_km2": float(clipped.sum() / 1e6), "z_min_after": float(np.nanmin(zin)), "z_max_after": float(np.nanmax(zin)),
                "z_p01_after": float(np.nanpercentile(zin, 1)), "z_p99_after": float(np.nanpercentile(zin, 99)), "nodata_cells_inside_after": int(np.isnan(zin).sum())})
    crs = rasterio.crs.CRS.from_epsg(6339)
    write_tif(os.path.join(a.out, "md_study_area_clipped_1m.tif"), clipped.astype(np.uint8), from_origin(wx0, wy1, 1.0, 1.0), crs)
    write_tif(os.path.join(a.out, "removed_cells_1m.tif"), rlab.astype(np.int32), from_origin(wx0, wy1, 1.0, 1.0), crs)
    # polygonise the clipped mask into a copy of the stage B GeoPackage
    from osgeo import gdal, ogr, osr
    gpkg = os.path.join(a.out, "md1989_fig3_a16.gpkg")
    shutil.copyfile(rec["source_polygon"], gpkg)
    ds = ogr.Open(gpkg, 1)
    if ds.GetLayerByName("study_area_clipped") is not None:
        ds.DeleteLayer("study_area_clipped")
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(6339)
    lyr = ds.CreateLayer("study_area_clipped", srs, ogr.wkbPolygon)
    lyr.CreateField(ogr.FieldDefn("area_km2", ogr.OFTReal))
    lyr.CreateField(ogr.FieldDefn("source", ogr.OFTString))
    mem = gdal.GetDriverByName("MEM").Create("", clipped.shape[1], clipped.shape[0], 1, gdal.GDT_Byte)
    mem.SetGeoTransform((wx0, 1.0, 0.0, wy1, 0.0, -1.0))
    mem.SetProjection(srs.ExportToWkt())
    mem.GetRasterBand(1).WriteArray(clipped.astype(np.uint8))
    tmp_ds = ogr.GetDriverByName("Memory").CreateDataSource("tmp")
    tmp = tmp_ds.CreateLayer("tmp", srs, ogr.wkbPolygon)
    tmp.CreateField(ogr.FieldDefn("v", ogr.OFTInteger))
    gdal.Polygonize(mem.GetRasterBand(1), mem.GetRasterBand(1), tmp, 0, ["8CONNECTED=8"])
    geoms = [f.GetGeometryRef().Clone() for f in tmp if f.GetField("v") == 1]
    rec["polygonised_parts"] = len(geoms)
    g = geoms[0]
    for h in geoms[1:]:
        g = g.Union(h)
    f = ogr.Feature(lyr.GetLayerDefn())
    f.SetGeometry(g)
    f.SetField("area_km2", g.GetArea() / 1e6)
    f.SetField("source", f"stage B study_area clipped at z >= {TTV_CLIP_Z} m on the TTV 1 m grid, largest component, holes filled (a16_ttv_clip.py)")
    lyr.CreateFeature(f)
    rec["polygon_area_km2"] = g.GetArea() / 1e6
    ds = None
    rec["gpkg"] = gpkg
    # rule 2.2 again (addendum 15, 2.5): area binding; z_min by construction
    area_ok = 2.1 * 0.85 <= rec["area_after_km2"] <= 2.1 * 1.15
    zmin_ok = 61 - 15 <= rec["z_min_after"] <= 61 + 15
    zmax_ok = 314 - 15 <= rec["z_max_after"] <= 314 + 15
    rule = {"area_km2": rec["area_after_km2"], "area_window": [2.1 * 0.85, 2.1 * 1.15], "area_pass": bool(area_ok), "area_binding": True,
            "z_min_1m": rec["z_min_after"], "z_min_window": [46, 76], "z_min_pass": bool(zmin_ok), "z_min_by_construction": True,
            "z_max_1m": rec["z_max_after"], "z_max_window": [299, 329], "z_max_pass": bool(zmax_ok), "pass": bool(area_ok and zmin_ok and zmax_ok),
            "stage_b": {"area_km2": rec["area_before_km2"], "z_min_1m": rec["z_min_before"], "z_max_1m": rec["z_max_before"]}}
    dump(rule, os.path.join(a.out, "rule_2_2_a16.json"))
    rec["rule_2_2"] = rule
    rec["seconds"] = time.time() - t0
    dump(rec, os.path.join(a.out, "clip.json"))
    print("clip", json.dumps({k: v for k, v in rec.items() if k not in ("removed_clusters",)}, indent=1), flush=True)
    print("removed clusters", json.dumps(clusters[:6], indent=1), flush=True)
    # map
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        gy, gx = np.gradient(np.nan_to_num(z, nan=np.nanmean(z)), 1.0, 1.0)
        sl = np.arctan(np.hypot(gx, gy))
        asp = np.arctan2(-gx, gy)
        hs = np.sin(np.radians(45)) * np.cos(sl) + np.cos(np.radians(45)) * np.sin(sl) * np.cos(np.radians(315) - asp)
        fig, ax = plt.subplots(figsize=(12, 12), dpi=110)
        ext = [wx0, wx1, wy0, wy1]
        ax.imshow(hs, cmap="gray", extent=ext, vmin=-0.3, vmax=1.3)
        xs = wx0 + np.arange(z.shape[1]) + 0.5
        ys = wy1 - np.arange(z.shape[0]) - 0.5
        ax.contour(xs, ys, inside.astype(float), levels=[0.5], colors="orange", linewidths=1.0)
        ax.contour(xs, ys, clipped.astype(float), levels=[0.5], colors="red", linewidths=1.0)
        ax.contour(xs, ys, np.nan_to_num(z, nan=0.0), levels=[61.0], colors="cyan", linewidths=0.6)
        rm = np.ma.masked_where(~removed, removed.astype(float))
        ax.imshow(rm, cmap="autumn", extent=ext, alpha=0.6, vmin=0, vmax=1)
        for c in clusters[:6]:
            ax.text(c["centroid_E"], c["centroid_N"], f"{c['cells']} cells\nz {c['z_min']:.0f}-{c['z_max']:.0f}", color="yellow", fontsize=7, ha="center")
        ax.set_title(f"M&D polygon (orange, {rec['area_before_km2']:.3f} km2) clipped at z >= {TTV_CLIP_Z:.0f} m (red, {rec['area_after_km2']:.3f} km2); 61 m contour cyan; removed cells shaded")
        fig.savefig(os.path.join(a.out, "clip_map.png"), bbox_inches="tight")
        plt.close(fig)
    except Exception as e:  # pragma: no cover
        print("map failed", repr(e), flush=True)
    if not area_ok:
        print("RULE 2.2 AREA FAILS: stop", flush=True)
        sys.exit(2)
    print("done", round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
