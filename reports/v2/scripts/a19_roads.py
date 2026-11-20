#!/usr/bin/env python3
"""Addendum 19, decision 1 (road and trail filter): fetch US Census TIGER/Line 2024 ROADS (all MTFCC classes) for every county that a
measured tile touches, store the zips under $GROUP_HOME/lemcvt_data/source/TIGER2024_roads/ with sha256, and write <run>/roads/roads.json.

No road/trail vector layer exists under source/ (checked 2026-11-20), so TIGER is the layer, as the directive says. Light work (a few MB
per county); runs on the login node. usage: a19_roads.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a19_common import (  # noqa: E402
    MEASURED_TILES,
    ROADS_DIR,
    TIGER_COUNTY_ZIP,
    TIGER_ROADS_URL,
    dump,
    out_path,
    sha256_file,
    site_of,
    tile_bounds,
)


def main() -> None:
    from osgeo import ogr, osr
    ogr.UseExceptions()
    osr.UseExceptions()
    t0 = time.time()
    os.makedirs(ROADS_DIR, exist_ok=True)
    county_ds = ogr.Open(f"/vsizip/{TIGER_COUNTY_ZIP}")
    lyr = county_ds.GetLayer(0)
    county_srs = lyr.GetSpatialRef()
    county_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    counties_by_site: dict[str, dict[str, dict]] = {}
    tiles_rec = {}
    for tile in MEASURED_TILES:
        b = tile_bounds(tile)
        src = osr.SpatialReference()
        src.ImportFromEPSG(b["crs_epsg"])
        src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        ct = osr.CoordinateTransformation(src, county_srs)
        ring = ogr.Geometry(ogr.wkbLinearRing)
        pad = 50.0   # the 10 m buffer plus margin
        for X, Y in ((b["left"] - pad, b["bottom"] - pad), (b["right"] + pad, b["bottom"] - pad), (b["right"] + pad, b["top"] + pad), (b["left"] - pad, b["top"] + pad), (b["left"] - pad, b["bottom"] - pad)):
            ring.AddPoint(X, Y)
        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)
        poly.Transform(ct)
        lyr.SetSpatialFilter(poly)
        hits = []
        for f in lyr:
            if f.GetGeometryRef().Intersects(poly):
                geoid = f.GetField("GEOID")
                hits.append(geoid)
                counties_by_site.setdefault(site_of(tile), {})[geoid] = {"geoid": geoid, "name": f.GetField("NAMELSAD"), "statefp": f.GetField("STATEFP")}
        lyr.SetSpatialFilter(None)
        tiles_rec[tile] = {"site": site_of(tile), "crs_epsg": b["crs_epsg"], "counties": sorted(hits)}
        print(tile, sorted(hits), flush=True)
    out = {"source": "US Census Bureau TIGER/Line Shapefiles 2024, ROADS (all MTFCC road classes incl. S1500 vehicular trail, S1710 walkway/pedestrian trail, S1820 bike path, S1830 bridle path)",
           "county_file": {"path": TIGER_COUNTY_ZIP, "sha256": sha256_file(TIGER_COUNTY_ZIP)}, "fetched": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "tiles": tiles_rec, "sites": {}}
    manifest = []
    for site, counties in counties_by_site.items():
        files = []
        for geoid, c in sorted(counties.items()):
            url = TIGER_ROADS_URL.format(fips=geoid)
            path = os.path.join(ROADS_DIR, os.path.basename(url))
            if not os.path.exists(path):
                subprocess.run(["curl", "-s", "--fail", "--max-time", "600", "-o", path, url], check=True)
            sha = sha256_file(path)
            ds = ogr.Open(f"/vsizip/{path}")
            n = ds.GetLayer(0).GetFeatureCount()
            mt: dict[str, int] = {}
            for f in ds.GetLayer(0):
                k = f.GetField("MTFCC")
                mt[k] = mt.get(k, 0) + 1
            files.append({"geoid": geoid, "county": c["name"], "state": c["statefp"], "url": url, "path": f"/vsizip/{path}", "zip": path, "sha256": sha,
                          "n_features": int(n), "mtfcc_counts": mt, "bytes": os.path.getsize(path)})
            manifest.append(f"{sha}  {os.path.basename(path)}")
            print(site, geoid, c["name"], n, "features", flush=True)
        out["sites"][site] = {"files": files}
    manifest.append(f"{out['county_file']['sha256']}  {os.path.basename(TIGER_COUNTY_ZIP)}")
    with open(os.path.join(ROADS_DIR, "MANIFEST.sha256"), "w") as f:
        f.write("\n".join(manifest) + "\n")
    out["manifest"] = os.path.join(ROADS_DIR, "MANIFEST.sha256")
    out["wall_s"] = time.time() - t0
    dump(out, out_path("roads", "roads.json"))
    print("roads.json written;", {s: [x["county"] for x in v["files"]] for s, v in out["sites"].items()})


if __name__ == "__main__":
    main()
