#!/usr/bin/env python3
"""List the PASDA LiDAR2020 PA_South LAS files that cover given tile footprints.

Usage:
  python pasda_tiles_for_ap.py --buffer 200 \
      --tile AP1 xmin ymin xmax ymax [--tile AP2 ...] [--src-epsg 6346]
  python pasda_tiles_for_ap.py --buffer 200 --from-rasters data/AP/tile*/*_filled_m.tif

Footprints are in the source CRS (default EPSG:6346, NAD83(2011) UTM 17N, metres).
With --from-rasters, bounds and CRS are read from each raster (needs rasterio).

Naming rule (PAMAP block scheme, reused by the PASDA LiDAR2020 delivery):
  block = 10,000 x 10,000 ft in NAD83 PA State Plane South (US survey ft);
  name  = first 4 digits of the NW-corner northing (6 digits) + first 4 digits of the
          NW-corner easting (7 digits) + 'PAS'; e.g. 18001230PAS = NW corner N 180,000, E 1,230,000.
  Each block is split into 5,000 ft quadrants: _NW, _NE, _SW, _SE.
  File: <block>PAS_<Q>_LAS.zip
Check the file list against the directory listing: border blocks may lack quadrants.
"""
import argparse, math, sys
from pyproj import Transformer

SP_SOUTH_FTUS = 6565  # NAD83(2011) / Pennsylvania South (ftUS)
BLOCK, HALF = 10_000.0, 5_000.0


def sp_bounds(xmin, ymin, xmax, ymax, src_epsg, buffer_m):
    t = Transformer.from_crs(src_epsg, SP_SOUTH_FTUS, always_xy=True)
    xmin, ymin, xmax, ymax = xmin - buffer_m, ymin - buffer_m, xmax + buffer_m, ymax + buffer_m
    # densify the edges: the projection change bends straight lines slightly
    pts = []
    for i in range(21):
        f = i / 20
        pts += [(xmin + f * (xmax - xmin), ymin), (xmin + f * (xmax - xmin), ymax),
                (xmin, ymin + f * (ymax - ymin)), (xmax, ymin + f * (ymax - ymin))]
    xs, ys = zip(*(t.transform(x, y) for x, y in pts))
    return min(xs), min(ys), max(xs), max(ys)


def quadrant_files(E0, N0, E1, N1):
    """All quadrant files whose 5,000 ft square intersects [E0,E1] x [N0,N1] (ftUS)."""
    out = set()
    for qe in range(math.floor(E0 / HALF), math.floor(E1 / HALF) + 1):
        for qn in range(math.floor(N0 / HALF), math.floor(N1 / HALF) + 1):
            e_sw, n_sw = qe * HALF, qn * HALF            # SW corner of the quadrant
            be = math.floor(e_sw / BLOCK) * BLOCK       # block W edge
            bn = math.floor(n_sw / BLOCK) * BLOCK + BLOCK  # block N edge (NW-corner northing)
            ew = "E" if e_sw - be >= HALF else "W"
            ns = "N" if n_sw >= bn - HALF else "S"
            name = f"{int(bn):06d}"[:4] + f"{int(be):07d}"[:4]
            out.add(f"{name}PAS_{ns}{ew}_LAS.zip")
    return sorted(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tile", nargs=5, action="append", metavar=("NAME", "XMIN", "YMIN", "XMAX", "YMAX"))
    p.add_argument("--from-rasters", nargs="+")
    p.add_argument("--src-epsg", type=int, default=6346)
    p.add_argument("--buffer", type=float, default=200.0, help="metres, in the source CRS")
    a = p.parse_args()
    tiles = []
    for name, *b in a.tile or []:
        tiles.append((name, *map(float, b), a.src_epsg))
    for path in a.from_rasters or []:
        import rasterio
        with rasterio.open(path) as r:
            epsg = r.crs.to_epsg()
            if epsg is None:
                sys.exit(f"{path}: CRS has no EPSG code; pass --tile instead")
            tiles.append((path, *r.bounds, epsg))
    if not tiles:
        p.error("give --tile or --from-rasters")
    allfiles = set()
    for name, x0, y0, x1, y1, epsg in tiles:
        E0, N0, E1, N1 = sp_bounds(x0, y0, x1, y1, epsg, a.buffer)
        files = quadrant_files(E0, N0, E1, N1)
        allfiles.update(files)
        print(f"# {name}: SP South ftUS E {E0:.0f}-{E1:.0f}, N {N0:.0f}-{N1:.0f}: {len(files)} files")
        for f in files:
            print(f"#   {f}")
    print(f"# total unique files: {len(allfiles)}")
    for f in sorted(allfiles):
        print(f"https://pasda.psu.edu/download/usgs/LiDAR2020/LAS/PA_South/{f}")


if __name__ == "__main__":
    main()
