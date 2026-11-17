#!/usr/bin/env python3
"""Addendum 13, section 3.3.3-3.3.5: check the candidate PASDA files against the saved directory listing and the tile index.

  * candidates: the quadrant files of pasda_tiles_for_ap.py for each tile's footprint plus the buffer;
  * listing: the saved HTML listing of LiDAR2020/LAS/PA_South/ (IIS format); every candidate is marked present/absent with
    its listed size, and the total size is summed (the 40 GB gate);
  * tile index: QL2_5K_SPSouth_TileIndex.shp (TILEID = <block>PAS_<Q>); every candidate must exist there and its polygon
    must intersect its tile's buffered footprint in State Plane South ftUS; files the index puts elsewhere are reported;
  * absent candidates: whether the quadrant is in the clipped index (QL2_5K_SPSouth_TileIndex_Clipped.shp, clipped to the
    project boundary, which follows the PA border on the south), as the proxy for "inside the PA border".

usage: a13_pasda_check.py --buffer 500 --rasters <five AP filled_m.tif> --listing HTML --index DIR --out DIR
"""
import argparse, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rasterio
from pyproj import Transformer
from osgeo import ogr
from pasda_tiles_for_ap import sp_bounds, quadrant_files, SP_SOUTH_FTUS, HALF, BLOCK

BASE = "https://pasda.psu.edu/download/usgs/LiDAR2020/LAS/PA_South/"


def quadrant_bounds(name):
    """SP South ftUS bounds of <block>PAS_<Q>_LAS.zip from the naming rule (NW-corner northing/easting digits)."""
    m = re.match(r"(\d{4})(\d{4})PAS_(N|S)(E|W)", name)
    n4, e4, ns, ew = m.groups()
    bn = int(n4) * 100.0  # NW-corner northing (6 digits: first 4 + '00')
    be = int(e4) * 1000.0  # NW-corner easting (7 digits: first 4 + '000')
    e0 = be + (HALF if ew == "E" else 0.0); n0 = bn - (HALF if ns == "N" else BLOCK)
    return e0, n0, e0 + HALF, n0 + HALF


def main():
    p = argparse.ArgumentParser(); p.add_argument("--buffer", type=float, default=500.0); p.add_argument("--rasters", nargs="+", required=True)
    p.add_argument("--listing", required=True); p.add_argument("--index", required=True); p.add_argument("--out", required=True)
    a = p.parse_args(); os.makedirs(a.out, exist_ok=True)
    html = open(a.listing).read()
    listed = {n: int(s) for _d, s, _h, n in re.findall(r'(\d+/\d+/\d+\s+\d+:\d+ [AP]M)\s+(\d+)\s+<A HREF="([^"]+)">([^<]+)</A>', html)}
    drv = ogr.GetDriverByName("ESRI Shapefile")
    def load(shp):
        ds = drv.Open(shp); lyr = ds.GetLayer(); out = {}
        for f in lyr:
            g = f.GetGeometryRef().Clone(); g.FlattenTo2D(); out[f.GetField("TILEID")] = g.GetEnvelope()  # (minx, maxx, miny, maxy)
        return out
    index = load(os.path.join(a.index, "QL2_5K_SPSouth_TileIndex", "QL2_5K_SPSouth_TileIndex.shp"))
    clipped = load(os.path.join(a.index, "QL2_5K_SPSouth_TileIndex_Clipped", "QL2_5K_SPSouth_TileIndex_Clipped.shp"))
    tiles, allfiles, per_tile = [], {}, {}
    for path in a.rasters:
        with rasterio.open(path) as r:
            epsg = r.crs.to_epsg(); b = r.bounds
        name = re.search(r"(AP\d|GM\d|tile\d)", path).group(1)
        E0, N0, E1, N1 = sp_bounds(b.left, b.bottom, b.right, b.top, epsg, a.buffer)
        files = quadrant_files(E0, N0, E1, N1)
        per_tile[name] = {"raster": path, "epsg": epsg, "bounds_src": list(b), "sp_south_ftus_buffered": [E0, N0, E1, N1], "files": files}
        for f in files:
            allfiles.setdefault(f, []).append(name)
    rows = []
    for f in sorted(allfiles):
        tid = f.replace("_LAS.zip", ""); qb = quadrant_bounds(f)
        r = {"file": f, "tiles": allfiles[f], "listed": f in listed, "bytes": listed.get(f), "rule_bounds_ftus": qb, "in_index": tid in index, "in_clipped_index": tid in clipped}
        if tid in index:
            ix = index[tid]; r["index_envelope_ftus"] = ix
            r["index_matches_rule_within_1ft"] = all(abs(u - v) < 1.0 for u, v in zip((ix[0], ix[2], ix[1], ix[3]), qb))
            r["index_intersects_tile_footprint"] = {t: bool(ix[0] < per_tile[t]["sp_south_ftus_buffered"][2] and ix[1] > per_tile[t]["sp_south_ftus_buffered"][0]
                                                           and ix[2] < per_tile[t]["sp_south_ftus_buffered"][3] and ix[3] > per_tile[t]["sp_south_ftus_buffered"][1]) for t in allfiles[f]}
        rows.append(r)
    total = sum(r["bytes"] or 0 for r in rows)
    summary = {"n_candidates": len(rows), "n_listed": sum(r["listed"] for r in rows), "absent": [r["file"] for r in rows if not r["listed"]],
               "absent_in_clipped_index": [r["file"] for r in rows if not r["listed"] and r["in_clipped_index"]],
               "not_in_index": [r["file"] for r in rows if not r["in_index"]], "index_rule_mismatch": [r["file"] for r in rows if r.get("index_matches_rule_within_1ft") is False],
               "index_not_intersecting": [r["file"] for r in rows if r.get("index_intersects_tile_footprint") and not all(r["index_intersects_tile_footprint"].values())],
               "total_bytes_listed": total, "total_GB": total / 1e9, "gate_40GB": total <= 40e9, "listing_n_zips": len(listed), "listing_total_GB": sum(listed.values()) / 1e9}
    json.dump({"per_tile": per_tile, "files": rows, "summary": summary}, open(os.path.join(a.out, "pasda_check.json"), "w"), indent=1)
    with open(os.path.join(a.out, "urls_to_download.txt"), "w") as fh:
        for r in rows:
            if r["listed"]:
                fh.write(BASE + r["file"] + "\n")
    for t, v in per_tile.items():
        E0, N0, E1, N1 = v["sp_south_ftus_buffered"]; print(t, "EPSG", v["epsg"], "SP ftUS E %.0f-%.0f N %.0f-%.0f" % (E0, E1, N0, N1), len(v["files"]), "files")
    for r in rows:
        print("%-26s %-10s listed=%-5s %12s index=%-5s clipped=%-5s rule_ok=%s" % (r["file"], ",".join(r["tiles"]), r["listed"], r["bytes"], r["in_index"], r["in_clipped_index"], r.get("index_matches_rule_within_1ft")))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
