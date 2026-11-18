#!/usr/bin/env python3
"""Addendum 14, 4.1.2 / 4.3.1: fetch class-2 points for a footprint from a USGS 3DEP Entwine (EPT) copy on AWS S3.

The EPT copies (usgs-lidar-public) are EPSG:3857 octrees with the source Z unchanged and no vertical CRS. For each 1 km
box of a lattice covering the footprint polygon, a PDAL pipeline reads the box (readers.ept, full resolution), keeps
Classification 2, reprojects horizontally to the target CRS and writes a LAZ whose header carries a compound CRS
(horizontal EPSG + the vertical CRS established by the independent comparison of a14_usgs_vs_pasda.py / rule 5.2; the
evidence string is recorded in the run metadata, decision 10). Boxes run in parallel; the per-box point counts, bytes and
timings are logged; a box that fails is retried once.

usage: a14_ept_fetch.py --ept URL --footprint GEOJSON --epsg 6346 --vertical-epsg 6360 --evidence TEXT --out DIR --work DIR [--procs 8] [--box 1000]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a14_common import dump, slurm_record  # noqa: E402


def boxes_for_footprint(geojson, epsg, box):
    from osgeo import ogr, osr
    ogr.UseExceptions()
    ds = ogr.Open(geojson)
    lyr = ds.GetLayer()
    feat = lyr.GetNextFeature()          # keep the feature alive while its geometry is cloned (OGR lifetime rule)
    g = feat.GetGeometryRef().Clone()
    src = lyr.GetSpatialRef()
    src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    dst = osr.SpatialReference()
    dst.ImportFromEPSG(epsg)
    dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    g.Transform(osr.CoordinateTransformation(src, dst))
    web = osr.SpatialReference()
    web.ImportFromEPSG(3857)
    web.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    to_web = osr.CoordinateTransformation(dst, web)
    x0, x1, y0, y1 = g.GetEnvelope()
    out = []
    for bx in np.arange(np.floor(x0 / box) * box, x1, box):
        for by in np.arange(np.floor(y0 / box) * box, y1, box):
            ring = ogr.Geometry(ogr.wkbLinearRing)
            for px, py in [(bx, by), (bx + box, by), (bx + box, by + box), (bx, by + box), (bx, by)]:
                ring.AddPoint(px, py)
            q = ogr.Geometry(ogr.wkbPolygon)
            q.AddGeometry(ring)
            if not q.Intersects(g):
                continue
            qw = q.Clone()
            qw.Transform(to_web)
            wx0, wx1, wy0, wy1 = qw.GetEnvelope()
            out.append({"name": f"{int(bx)}_{int(by)}", "box_target": [float(bx), float(by), float(bx + box), float(by + box)],
                        "bounds_3857": [wx0 - 2, wy0 - 2, wx1 + 2, wy1 + 2], "polygon_target": q.ExportToWkt()})
    return out, g.GetArea()


def fetch_one(args):
    b, ept, epsg, vepsg, work, out_dir = args
    laz = os.path.join(out_dir, f"ept_{b['name']}.laz")
    meta = os.path.join(work, f"ept_{b['name']}.meta.json")
    if os.path.exists(laz) and os.path.getsize(laz) > 1000 and os.path.exists(meta):
        return {"name": b["name"], "cached": True, "bytes": os.path.getsize(laz)}
    wb = b["bounds_3857"]
    pipe = {"pipeline": [{"type": "readers.ept", "filename": ept, "bounds": f"([{wb[0]},{wb[2]}],[{wb[1]},{wb[3]}])", "resolution": 0},
                         {"type": "filters.range", "limits": "Classification[2:2]"},
                         {"type": "filters.reprojection", "in_srs": "EPSG:3857", "out_srs": f"EPSG:{epsg}"},
                         {"type": "filters.crop", "polygon": b["polygon_target"]},
                         {"type": "writers.las", "filename": laz, "compression": "laszip", "scale_x": 0.001, "scale_y": 0.001, "scale_z": 0.001,
                          "offset_x": "auto", "offset_y": "auto", "offset_z": "auto", "a_srs": f"EPSG:{epsg}+{vepsg}", "minor_version": 4, "dataformat_id": 6}]}
    pp = os.path.join(work, f"ept_{b['name']}.json")
    json.dump(pipe, open(pp, "w"))
    t0 = time.time()
    rec = {"name": b["name"]}
    for attempt in (1, 2):
        r = subprocess.run(["pdal", "pipeline", pp, "--metadata", meta], capture_output=True, text=True)
        rec["exit"] = r.returncode
        if r.returncode == 0:
            break
        rec["stderr"] = r.stderr[-400:]
        time.sleep(5)
    rec["seconds"] = time.time() - t0
    rec["bytes"] = os.path.getsize(laz) if os.path.exists(laz) else 0
    try:
        em = json.load(open(meta))
        m = em.get("metadata", em.get("stages", em))
        wl = m.get("writers.las", {})
        rd = m.get("readers.ept", {})
        wl = wl[0] if isinstance(wl, list) and wl else wl
        rd = rd[0] if isinstance(rd, list) and rd else rd
        rec["points_written"] = wl.get("count") if isinstance(wl, dict) else None
        rec["points_read"] = rd.get("count") if isinstance(rd, dict) else None
    except Exception as e:
        rec["meta_error"] = repr(e)
    return rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ept", required=True)
    p.add_argument("--footprint", required=True)
    p.add_argument("--epsg", type=int, required=True)
    p.add_argument("--vertical-epsg", type=int, required=True)
    p.add_argument("--evidence", required=True, help="why the vertical CRS is what it is (recorded; decision 10)")
    p.add_argument("--out", required=True, help="LAZ output dir (scratch)")
    p.add_argument("--work", required=True)
    p.add_argument("--meta-out", required=True, help="run directory for the metadata json")
    p.add_argument("--procs", type=int, default=8)
    p.add_argument("--box", type=float, default=1000.0)
    a = p.parse_args()
    for d in (a.out, a.work, a.meta_out):
        os.makedirs(d, exist_ok=True)
    slurm_record(a.meta_out)
    boxes, area = boxes_for_footprint(a.footprint, a.epsg, a.box)
    print(len(boxes), "boxes over", round(area / 1e6, 1), "km2", flush=True)
    t0 = time.time()
    recs = []
    with ThreadPoolExecutor(max_workers=a.procs) as ex:
        for k, rec in enumerate(ex.map(fetch_one, [(b, a.ept, a.epsg, a.vertical_epsg, a.work, a.out) for b in boxes])):
            recs.append(rec)
            if k % 20 == 0:
                done = sum(r.get("bytes", 0) for r in recs)
                print(k, "boxes;", round(done / 1e9, 2), "GB;", round(time.time() - t0), "s", flush=True)
    tot = {"n_boxes": len(boxes), "n_ok": sum(r.get("exit", 0) == 0 for r in recs), "bytes": sum(r.get("bytes", 0) for r in recs),
           "points_written": sum(r.get("points_written") or 0 for r in recs), "seconds": time.time() - t0, "ept": a.ept, "epsg": a.epsg,
           "vertical_epsg": a.vertical_epsg, "vertical_evidence": a.evidence, "footprint": a.footprint, "box_m": a.box, "out": a.out}
    tot["MB_per_s"] = tot["bytes"] / 1e6 / max(1, tot["seconds"])
    dump({"summary": tot, "boxes": recs, "box_geometry": boxes}, os.path.join(a.meta_out, f"ept_fetch_{a.epsg}.json"))
    with open(os.path.join(a.meta_out, f"ept_files_{a.epsg}.txt"), "w") as fh:
        for r in recs:
            if r.get("bytes", 0) > 1000:
                fh.write(os.path.join(a.out, f"ept_{r['name']}.laz") + "\n")
    print(json.dumps(tot, indent=1))


if __name__ == "__main__":
    main()
