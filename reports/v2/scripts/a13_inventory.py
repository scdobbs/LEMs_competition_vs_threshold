#!/usr/bin/env python3
"""Addendum 13, section 2.1: inventory of every file under $GROUP_HOME/lemcvt_data/source (rasters and LAS).

For each file: path, size, sha256 (from the item's MANIFEST.sha256), format, CRS (name, EPSG, horizontal units, vertical
CRS if the file carries one), resolution, extent, nodata, value range and valid-cell count; for LAS: header fields,
point count, density over the header bounding box, classification and return-number counts, GPS-time range if the
point format carries one. Units that the file does not state are reported as "not stated" and are never inferred here.

usage: a13_inventory.py --source DIR --out DIR [--skip-stats]
"""
import argparse, json, os, sys, time
import numpy as np
import rasterio
from rasterio.windows import Window

FT_US = 1200.0 / 3937.0


def manifest_shas(source):
    out = {}
    for root, _dirs, files in os.walk(source):
        for f in files:
            if f.endswith(".sha256"):
                for line in open(os.path.join(root, f)):
                    parts = line.strip().split(maxsplit=1)
                    if len(parts) == 2:
                        out[os.path.normpath(os.path.join(root, parts[1].lstrip("*")))] = parts[0]
    return out


def crs_record(crs):
    if crs is None:
        return {"stated": False, "wkt": None, "epsg": None, "name": None, "horizontal_units": "not stated", "vertical": "not stated"}
    rec = {"stated": True, "epsg": crs.to_epsg(), "name": None, "wkt": crs.to_wkt()}
    try:
        from pyproj import CRS
        c = CRS.from_wkt(crs.to_wkt())
        rec["name"] = c.name
        rec["datum"] = c.datum.name if c.datum else None
        rec["horizontal_units"] = c.axis_info[0].unit_name if c.axis_info else "not stated"
        rec["is_compound"] = c.is_compound
        rec["vertical"] = (c.sub_crs_list[1].name if c.is_compound and len(c.sub_crs_list) > 1 else "not stated (no vertical CRS in file)")
    except Exception as e:  # pyproj absent
        rec["name"] = str(crs)
        rec["horizontal_units"] = crs.linear_units or "not stated"
        rec["vertical"] = "not stated (no vertical CRS in file)"
        rec["pyproj_error"] = repr(e)
    return rec


def raster_record(path, skip_stats=False):
    t0 = time.time()
    with rasterio.open(path) as ds:
        rec = {"format": ds.driver, "dtype": ds.dtypes[0], "bands": ds.count, "width": ds.width, "height": ds.height,
               "res": [ds.res[0], ds.res[1]], "transform": list(ds.transform)[:6], "bounds": list(ds.bounds), "nodata": ds.nodata,
               "crs": crs_record(ds.crs), "tags": ds.tags(), "block_shape": list(ds.block_shapes[0]), "compression": ds.compression.value if ds.compression else None}
        if not skip_stats:
            vmin, vmax, nvalid, nnod, nnan, ssum = np.inf, -np.inf, 0, 0, 0, 0.0
            step = max(1, int(2e8 // (ds.width * 4)))  # ~200 MB of float32 per read
            for r0 in range(0, ds.height, step):
                a = ds.read(1, window=Window(0, r0, ds.width, min(step, ds.height - r0))).astype(np.float64)
                nan = np.isnan(a)
                nnan += int(nan.sum())
                if ds.nodata is not None and not np.isnan(ds.nodata):
                    nod = a == ds.nodata
                else:
                    nod = np.zeros_like(nan)
                nnod += int(nod.sum())
                ok = ~(nan | nod)
                if ok.any():
                    v = a[ok]
                    vmin, vmax = min(vmin, float(v.min())), max(vmax, float(v.max()))
                    nvalid += int(v.size); ssum += float(v.sum())
            rec["stats"] = {"min": vmin, "max": vmax, "mean": ssum / nvalid if nvalid else None, "n_valid": nvalid, "n_nodata": nnod, "n_nan": nnan,
                            "n_cells": ds.width * ds.height, "frac_valid": nvalid / (ds.width * ds.height)}
    rec["seconds"] = time.time() - t0
    return rec


def las_record(path, chunk=5_000_000):
    import laspy
    t0 = time.time()
    with laspy.open(path) as f:
        h = f.header
        rec = {"format": f"LAS {h.version}", "point_format": int(h.point_format.id), "point_record_length": int(h.point_format.size),
               "point_count": int(h.point_count), "system_identifier": h.system_identifier, "generating_software": h.generating_software,
               "file_creation": [h.creation_date.isoformat() if h.creation_date else None], "global_encoding": int(h.global_encoding.value) if hasattr(h.global_encoding, "value") else str(h.global_encoding),
               "scales": [float(s) for s in h.scales], "offsets": [float(o) for o in h.offsets],
               "mins": [float(v) for v in h.mins], "maxs": [float(v) for v in h.maxs], "n_vlrs": len(h.vlrs), "n_evlrs": len(getattr(h, "evlrs", []) or []),
               "vlrs": [{"user_id": v.user_id, "record_id": int(v.record_id), "description": v.description, "type": type(v).__name__} for v in h.vlrs],
               "dimensions": list(h.point_format.dimension_names), "number_of_points_by_return": [int(x) for x in h.number_of_points_by_return]}
        try:
            crs = h.parse_crs()
            rec["crs"] = ({"stated": True, "name": crs.name, "epsg": crs.to_epsg(), "horizontal_units": crs.axis_info[0].unit_name, "is_compound": crs.is_compound,
                           "vertical": (crs.sub_crs_list[1].name if crs.is_compound and len(crs.sub_crs_list) > 1 else "not stated (no vertical CRS in header)"), "wkt": crs.to_wkt()}
                          if crs is not None else {"stated": False, "horizontal_units": "not stated", "vertical": "not stated"})
        except Exception as e:
            rec["crs"] = {"stated": False, "error": repr(e), "horizontal_units": "not stated", "vertical": "not stated"}
        for v in h.vlrs:
            if v.user_id == "LASF_Projection" and v.record_id == 2112:
                rec["crs_wkt_vlr"] = v.string if hasattr(v, "string") else None
        cls = np.zeros(256, np.int64); ret = np.zeros(16, np.int64); nret = np.zeros(16, np.int64)
        zmin, zmax, n = np.inf, -np.inf, 0
        gps = [np.inf, -np.inf]; has_gps = "gps_time" in h.point_format.dimension_names
        cls_z = {}
        for pts in f.chunk_iterator(chunk):
            c = np.asarray(pts.classification); cls += np.bincount(c, minlength=256)[:256]
            ret += np.bincount(np.asarray(pts.return_number), minlength=16)[:16]
            nret += np.bincount(np.asarray(pts.number_of_returns), minlength=16)[:16]
            z = np.asarray(pts.z); zmin, zmax = min(zmin, float(z.min())), max(zmax, float(z.max())); n += z.size
            if has_gps:
                g = np.asarray(pts.gps_time); gps = [min(gps[0], float(g.min())), max(gps[1], float(g.max()))]
            for k in np.unique(c):
                zz = z[c == k]; d = cls_z.setdefault(int(k), [np.inf, -np.inf]); d[0], d[1] = min(d[0], float(zz.min())), max(d[1], float(zz.max()))
        area = (h.maxs[0] - h.mins[0]) * (h.maxs[1] - h.mins[1])
        rec.update({"n_read": n, "z_range_read": [zmin, zmax], "bbox_area_header_units2": float(area), "density_per_header_unit2": n / area,
                    "classes": {int(k): int(v) for k, v in enumerate(cls) if v}, "class_z_range": cls_z,
                    "return_number": {int(k): int(v) for k, v in enumerate(ret) if v}, "number_of_returns": {int(k): int(v) for k, v in enumerate(nret) if v},
                    "has_gps_time": has_gps, "gps_time_range": gps if has_gps else "not present (point format %d has no GPS time)" % h.point_format.id})
    rec["seconds"] = time.time() - t0
    return rec


def main():
    p = argparse.ArgumentParser(); p.add_argument("--source", required=True); p.add_argument("--out", required=True); p.add_argument("--skip-stats", action="store_true")
    p.add_argument("--only", nargs="*", help="substrings; process only matching paths")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    shas = manifest_shas(a.source)
    out = {}
    for root, _d, files in os.walk(a.source):
        if "ancillary" in root or "las" == os.path.basename(root):
            continue
        for f in sorted(files):
            path = os.path.normpath(os.path.join(root, f))
            if f.endswith(".sha256") or (a.only and not any(s in path for s in a.only)):
                continue
            rel = os.path.relpath(path, a.source)
            rec = {"path": path, "item": rel.split(os.sep)[0], "bytes": os.path.getsize(path), "sha256": shas.get(path, "not in manifest"),
                   "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(path)))}
            try:
                if f.lower().endswith((".tif", ".tiff")):
                    rec.update(raster_record(path, a.skip_stats))
                elif f.lower().endswith((".las", ".laz")):
                    rec.update(las_record(path))
                else:
                    rec["format"] = "text/other"
            except Exception as e:
                rec["error"] = repr(e)
            print(rel, "done", rec.get("seconds", 0), file=sys.stderr, flush=True)
            out[rel] = rec
    json.dump(out, open(os.path.join(a.out, "inventory.json"), "w"), indent=1, default=str)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("crs", "tags", "vlrs")} for k, v in out.items()}, indent=1, default=str)[:20000])


if __name__ == "__main__":
    main()
