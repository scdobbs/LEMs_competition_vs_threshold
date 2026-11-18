#!/usr/bin/env python3
"""Addendum 16, section 3.1: public lidar footprints over Gabilan Mesa beyond the NCALM 2003 footprint.

Inputs (fetched by the job script into $SCRATCH/a16/footprints and $SCRATCH/a16/ned; the NED cells are copied to the
copies of record $GROUP_HOME/lemcvt_data/source/GM_NED_13arcsec/ with sha256 manifest entries):
  * the USGS 1/3 arc-second NED cells n36w121 and n37w121 (current product; the HTTP Last-Modified date is recorded);
  * the Entwine usgs-lidar-public boundaries (resources.geojson: name, point count, polygon) and each survey's ept.json;
  * the USGS WESM.csv work-unit rows (collection dates, QL, spec, CRS);
  * the NOAA Digital Coast tile indexes of InPort 64466 (ID 9269), 64465 (9270), 64364 (9263).
Products (run dir gm_footprints/): GM_NED_10m_utm10n.tif (the two cells mosaicked and warped to EPSG:32610, 10 m, bilinear,
over GM_LANDSCAPE_BOX_UTM), GM_ncalm_valid_frac_10m.tif (fraction of valid 1 m Gabilan_mesa.tif cells per 10 m cell),
GM_perron_valid_10m.tif (the stage A coarse grid's valid area on this lattice), survey masks GM_survey_<name>_10m.tif,
footprints.json (areas inside/beyond the NCALM footprint, within the Perron grid and within 5/10/15 km of the GM tiles'
centroid, density and dates), gm_footprints_map.png.
usage: a16_gm_footprints.py --out DIR --ned-dir DIR --fp-dir DIR
"""
import argparse
import csv
import json
import os
import shutil
import sys
import time

import numpy as np
import rasterio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a14_common import SITES, bin_to_lattice  # noqa: E402
from a16_common import (  # noqa: E402
    GM_LANDSCAPE_BOX_UTM,
    GM_SURVEYS,
    NED_CELLS_GM,
    NOAA_INPORT,
    NOAA_TILEINDEX,
    SOURCE,
    dump,
    sha256_file,
    slurm_record,
    write_tif,
)

GM_TILES = {"GM1": (700991, 3980451), "GM2": (698991, 3980451), "GM3": (695991, 3978451), "GM4": (694991, 3976451), "GM5": (699991, 3981451),
            "GM6": (697481, 3978961), "GM7": (697981, 3977961)}   # south-west corners, EPSG:32610 (stage B tiles.csv)
DX = 10.0


def copy_record(src, item_dir, manifest_item, manifest_all):
    os.makedirs(item_dir, exist_ok=True)
    dst = os.path.join(item_dir, os.path.basename(src))
    sha = sha256_file(src)
    if not (os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src)):
        shutil.copy2(src, dst)
    sha_dst = sha256_file(dst)
    for man in (manifest_item, manifest_all):
        rel = os.path.relpath(dst, os.path.dirname(man))
        lines = open(man).read().splitlines() if os.path.exists(man) else []
        if not any(line.endswith("  " + rel) for line in lines):
            with open(man, "a") as f:
                f.write(f"{sha}  {rel}\n")
    return {"src": src, "dst": dst, "bytes": os.path.getsize(dst), "sha256": sha, "verified": sha == sha_dst}


def rasterize_geoms(geoms_wkt, src_epsg, tr, shape, dst_epsg=32610):
    from osgeo import gdal, ogr, osr
    s = osr.SpatialReference()
    s.ImportFromEPSG(src_epsg)
    s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    d = osr.SpatialReference()
    d.ImportFromEPSG(dst_epsg)
    d.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    ct = osr.CoordinateTransformation(s, d)
    mem_ds = ogr.GetDriverByName("Memory").CreateDataSource("m")
    lyr = mem_ds.CreateLayer("g", d, ogr.wkbMultiPolygon)
    for wkt in geoms_wkt:
        g = ogr.CreateGeometryFromWkt(wkt)
        g.Transform(ct)
        f = ogr.Feature(lyr.GetLayerDefn())
        f.SetGeometry(g)
        lyr.CreateFeature(f)
    ny, nx = shape
    tgt = gdal.GetDriverByName("MEM").Create("", nx, ny, 1, gdal.GDT_Byte)
    tgt.SetGeoTransform((tr.c, tr.a, 0.0, tr.f, 0.0, tr.e))
    tgt.SetProjection(d.ExportToWkt())
    gdal.RasterizeLayer(tgt, [1], lyr, burn_values=[1], options=["ALL_TOUCHED=FALSE"])
    return tgt.ReadAsArray().astype(bool)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--ned-dir", required=True)
    p.add_argument("--fp-dir", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    slurm_record(a.out)
    t0 = time.time()
    from osgeo import gdal
    gdal.UseExceptions()
    rec = {"box_utm": GM_LANDSCAPE_BOX_UTM, "dx": DX, "epsg": 32610}
    # 1. NED cells: copies of record and the warp
    item_dir = os.path.join(SOURCE, "GM_NED_13arcsec")
    man_item = os.path.join(item_dir, "MANIFEST.sha256")
    man_all = os.path.join(SOURCE, "MANIFEST_all.sha256")
    rec["ned_cells"] = {}
    for cell, url in NED_CELLS_GM.items():
        src = os.path.join(a.ned_dir, os.path.basename(url))
        hdr = os.path.join(a.ned_dir, os.path.basename(url) + ".headers")
        lm = None
        if os.path.exists(hdr):
            for line in open(hdr, errors="ignore"):
                if line.lower().startswith("last-modified:"):
                    lm = line.split(":", 1)[1].strip()
        info = gdal.Info(src, format="json")
        rec["ned_cells"][cell] = {"url": url, "http_last_modified": lm, **copy_record(src, item_dir, man_item, man_all),
                                  "size": info["size"], "srs": info.get("coordinateSystem", {}).get("wkt", "")[:120], "metadata": info.get("metadata", {}).get("", {})}
    x0, y0, x1, y1 = GM_LANDSCAPE_BOX_UTM
    vrt = os.path.join(a.out, "GM_NED_mosaic.vrt")
    gdal.BuildVRT(vrt, [rec["ned_cells"][c]["dst"] for c in NED_CELLS_GM])
    ned_utm = os.path.join(a.out, "GM_NED_10m_utm10n.tif")
    gdal.Warp(ned_utm, vrt, dstSRS="EPSG:32610", outputBounds=(x0, y0, x1, y1), xRes=DX, yRes=DX, resampleAlg="bilinear", dstNodata=-9999.0,
              outputType=gdal.GDT_Float32, creationOptions=["COMPRESS=DEFLATE", "TILED=YES"], targetAlignedPixels=False)
    with rasterio.open(ned_utm) as ds:
        z = ds.read(1).astype(np.float64)
        tr, crs = ds.transform, ds.crs
    z[z == -9999.0] = np.nan
    ny, nx = z.shape
    rec["ned_utm"] = {"path": ned_utm, "shape": [ny, nx], "transform": list(tr)[:6], "n_valid": int(np.isfinite(z).sum()), "z_min": float(np.nanmin(z)), "z_max": float(np.nanmax(z)),
                      "resample": "bilinear", "datum_note": "source NAD83 (EPSG:4269) geographic, NAVD88 m; warped to EPSG:32610 (WGS 84 / UTM 10N, the NCALM grid's CRS) without a datum grid (shift ~1 m)"}
    print("NED warped", rec["ned_utm"], flush=True)
    # 2. NCALM 2003 footprint on this lattice: valid-fraction of the 1 m lidar per 10 m cell
    lidar = SITES["GM"]["lidar_record"]
    B = bin_to_lattice(lidar, tr, (ny, nx), valid_fn=lambda arr: arr > -1e30)
    frac = np.where(B["n_total"] > 0, B["n_valid"] / np.maximum(B["n_total"], 1), 0.0)
    write_tif(os.path.join(a.out, "GM_ncalm_valid_frac_10m.tif"), frac.astype(np.float32), tr, crs)
    ncalm_full = frac >= 0.999
    ncalm_any = frac > 0
    rec["ncalm"] = {"source": lidar, "cells_full": int(ncalm_full.sum()), "km2_full": float(ncalm_full.sum() * DX * DX / 1e6), "cells_any": int(ncalm_any.sum()), "km2_any": float(ncalm_any.sum() * DX * DX / 1e6)}
    # 3. the stage A coarse grid's valid area (Perron's GM_Lidar_overlap_10m) on this lattice (nearest)
    perron = SITES["GM"]["coarse"]
    pv_path = os.path.join(a.out, "GM_perron_valid_10m.tif")
    with rasterio.open(perron) as ds:
        pz = ds.read(1)
        pnod = ds.nodata
        ptr, pcrs = ds.transform, ds.crs
    pvalid = (pz != pnod).astype(np.uint8)
    tmp = os.path.join(a.out, "_perron_valid_src.tif")
    write_tif(tmp, pvalid, ptr, pcrs)
    gdal.Warp(pv_path, tmp, dstSRS="EPSG:32610", outputBounds=(x0, y0, x1, y1), xRes=DX, yRes=DX, resampleAlg="near", outputType=gdal.GDT_Byte, creationOptions=["COMPRESS=DEFLATE"])
    os.remove(tmp)
    with rasterio.open(pv_path) as ds:
        perron_valid = ds.read(1).astype(bool)
    rec["perron_grid"] = {"source": perron, "cells_valid_on_lattice": int(perron_valid.sum()), "km2": float(perron_valid.sum() * DX * DX / 1e6),
                          "km2_inside_ncalm_full": float((perron_valid & ncalm_full).sum() * DX * DX / 1e6), "km2_beyond_ncalm": float((perron_valid & ~ncalm_any).sum() * DX * DX / 1e6)}
    # 4. distance rings from the GM tiles' centroid
    cx = np.mean([v[0] + 500 for v in GM_TILES.values()])
    cy = np.mean([v[1] + 500 for v in GM_TILES.values()])
    xs = tr.c + (np.arange(nx) + 0.5) * tr.a
    ys = tr.f + (np.arange(ny) + 0.5) * tr.e
    dist = np.hypot(xs[None, :] - cx, ys[:, None] - cy)
    rings = {5: dist <= 5000, 10: dist <= 10000, 15: dist <= 15000}
    rec["tiles_centroid"] = [float(cx), float(cy)]
    # 5. surveys: Entwine polygons, ept.json, WESM rows
    ent = json.load(open(os.path.join(a.fp_dir, "entwine_resources.geojson")))
    feats = {f["properties"]["name"]: f for f in ent["features"]}
    wesm = {}
    with open(os.path.join(a.fp_dir, "WESM.csv"), newline="") as fh:
        for row in csv.DictReader(fh):
            wesm[row["workunit"]] = row
    from osgeo import ogr
    surveys = {}
    masks = {}
    for name in GM_SURVEYS:
        f = feats.get(name)
        if f is None:
            surveys[name] = {"error": "not in resources.geojson"}
            continue
        g = ogr.CreateGeometryFromJson(json.dumps(f["geometry"]))
        m = rasterize_geoms([g.ExportToWkt()], 4326, tr, (ny, nx))
        masks[name] = m
        write_tif(os.path.join(a.out, f"GM_survey_{name}_10m.tif"), m.astype(np.uint8), tr, crs)
        ept = {}
        ep = os.path.join(a.fp_dir, f"ept_{name}.json")
        if os.path.exists(ep):
            try:
                e = json.load(open(ep))
                ept = {"points": e.get("points"), "bounds": e.get("bounds"), "srs": e.get("srs"), "span": e.get("span"), "version": e.get("version")}
            except Exception as ex:
                ept = {"error": repr(ex)}
        # polygon area in UTM (whole survey) for the nominal density
        from osgeo import osr
        s = osr.SpatialReference()
        s.ImportFromEPSG(4326)
        s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        d = osr.SpatialReference()
        d.ImportFromEPSG(32610)
        d.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        gg = g.Clone()
        gg.Transform(osr.CoordinateTransformation(s, d))
        area_km2 = gg.GetArea() / 1e6
        w = wesm.get(name) or wesm.get(name.replace("-", "_").upper()) or {}
        cell = DX * DX / 1e6
        surveys[name] = {"entwine_points": f["properties"].get("count"), "entwine_url": f["properties"].get("url"), "polygon_km2_utm10": area_km2,
                         "nominal_density_pts_m2": (f["properties"].get("count") or 0) / max(area_km2 * 1e6, 1), "ept": ept,
                         "wesm": {k: w.get(k) for k in ("workunit", "project", "collect_start", "collect_end", "ql", "spec", "p_method", "horiz_crs", "vert_crs", "geoid", "lpc_pub_date", "lpc_category")} if w else None,
                         "km2_in_box": float(m.sum() * cell), "km2_in_perron_grid": float((m & perron_valid).sum() * cell),
                         "km2_inside_ncalm_full": float((m & ncalm_full).sum() * cell), "km2_inside_ncalm_any": float((m & ncalm_any).sum() * cell),
                         "km2_beyond_ncalm_in_perron_grid": float((m & perron_valid & ~ncalm_any).sum() * cell),
                         "km2_beyond_ncalm_within_5km": float((m & rings[5] & ~ncalm_any).sum() * cell), "km2_beyond_ncalm_within_10km": float((m & rings[10] & ~ncalm_any).sum() * cell),
                         "km2_beyond_ncalm_within_15km": float((m & rings[15] & ~ncalm_any).sum() * cell), "km2_within_15km": float((m & rings[15]).sum() * cell),
                         "covers_tiles": {t: bool(m[int((tr.f - (v[1] + 500)) / -tr.e), int((v[0] + 500 - tr.c) / tr.a)]) for t, v in GM_TILES.items()}}
        print(name, {k: v for k, v in surveys[name].items() if k not in ("ept", "wesm", "covers_tiles")}, flush=True)
    rec["surveys"] = surveys
    # 6. NOAA tile indexes (InPort items) on the lattice
    noaa = {}
    for nid, url in NOAA_TILEINDEX.items():
        path = os.path.join(a.fp_dir, os.path.basename(url))
        if not os.path.exists(path) or os.path.getsize(path) < 1000:
            noaa[nid] = {"inport": NOAA_INPORT[nid], "error": "tile index not downloaded"}
            continue
        try:
            ds = ogr.Open(path)
            lyr = ds.GetLayer(0)
            srs = lyr.GetSpatialRef()
            epsg = int(srs.GetAuthorityCode(None)) if srs is not None and srs.GetAuthorityCode(None) else 4326
            wkts = [ft.GetGeometryRef().ExportToWkt() for ft in lyr if ft.GetGeometryRef() is not None]
            m = rasterize_geoms(wkts, epsg, tr, (ny, nx))
            masks["NOAA_" + nid] = m
            write_tif(os.path.join(a.out, f"GM_noaa_{nid}_10m.tif"), m.astype(np.uint8), tr, crs)
            cell = DX * DX / 1e6
            noaa[nid] = {"inport": NOAA_INPORT[nid], "tiles": len(wkts), "layer_epsg": epsg, "km2_in_box": float(m.sum() * cell), "km2_in_perron_grid": float((m & perron_valid).sum() * cell),
                         "km2_inside_ncalm_any": float((m & ncalm_any).sum() * cell), "km2_beyond_ncalm_in_perron_grid": float((m & perron_valid & ~ncalm_any).sum() * cell),
                         "km2_beyond_ncalm_within_15km": float((m & rings[15] & ~ncalm_any).sum() * cell), "fields": [lyr.GetLayerDefn().GetFieldDefn(i).GetName() for i in range(lyr.GetLayerDefn().GetFieldCount())]}
            print("NOAA", nid, noaa[nid], flush=True)
        except Exception as ex:
            noaa[nid] = {"inport": NOAA_INPORT[nid], "error": repr(ex)}
    rec["noaa"] = noaa
    rec["seconds"] = time.time() - t0
    dump(rec, os.path.join(a.out, "footprints.json"))
    # 7. map
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        gy, gx = np.gradient(np.nan_to_num(z, nan=np.nanmean(z)), DX, DX)
        sl = np.arctan(np.hypot(gx, gy))
        asp = np.arctan2(-gx, gy)
        hs = np.sin(np.radians(45)) * np.cos(sl) + np.cos(np.radians(45)) * np.sin(sl) * np.cos(np.radians(315) - asp)
        fig, ax = plt.subplots(figsize=(16, 16), dpi=110)
        ext = [tr.c, tr.c + nx * tr.a, tr.f + ny * tr.e, tr.f]
        ax.imshow(hs, cmap="gray", extent=ext, vmin=-0.3, vmax=1.3)
        X, Y = np.meshgrid(xs, ys)
        ax.contour(X, Y, ncalm_any.astype(float), levels=[0.5], colors="lime", linewidths=1.5)
        ax.contour(X, Y, perron_valid.astype(float), levels=[0.5], colors="yellow", linewidths=1.2)
        cols = {"CA_FEMA_Z4_B2_2018": "red", "CA_FEMAR9Estrella_2019": "magenta", "CA_CaliforniaGaps_1_B23": "cyan", "CA_CaliforniaGaps_3_B23": "blue",
                "ARRA-CA_CentralCoast-Z4_2010": "orange", "USGS_LPC_CA_WestCoastElNinoUTM10_2016_LAS_2017": "brown", "NOAA_9269": "pink", "NOAA_9270": "white", "NOAA_9263": "purple"}
        for name, m in masks.items():
            if m.any():
                ax.contour(X, Y, m.astype(float), levels=[0.5], colors=cols.get(name, "black"), linewidths=1.0, linestyles="--" if name.startswith("NOAA") else "-")
        for t, (tx, ty) in GM_TILES.items():
            ax.add_patch(Rectangle((tx, ty), 1000, 1000, fill=False, ec="lime", lw=1.2))
            ax.text(tx + 500, ty + 500, t, color="lime", fontsize=7, ha="center", va="center")
        for r in (5000, 10000, 15000):
            ax.add_patch(plt.Circle((cx, cy), r, fill=False, ec="white", lw=0.5, ls=":"))
        ax.set_title("GM: NCALM 2003 footprint (lime), stage A coarse grid (yellow), surveys: FEMA Z4 B2 2018 red, Estrella 2019 magenta, CaliforniaGaps 1 cyan / 3 blue, ARRA 2010 orange, ElNino 2016 brown; NOAA indexes dashed")
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        fig.savefig(os.path.join(a.out, "gm_footprints_map.png"), bbox_inches="tight")
        plt.close(fig)
    except Exception as e:  # pragma: no cover
        print("map failed", repr(e), flush=True)
    print("done", round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
