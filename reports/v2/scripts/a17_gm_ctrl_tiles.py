#!/usr/bin/env python3
"""Addendum 17, section 2.2.3: the footprints of the GM survey-control tiles GM1_fema-GM7_fema on the FEMA lattice.

The registered GM-x footprints (EPSG:32610, NAD83 / UTM 10N) are reprojected to EPSG:6339 (NAD83(2011) / UTM 10N) with the
GDAL/PROJ of the lemcvt environment (offline: no NAD83 -> NAD83(2011) grid, so the transformation is the identity to 0.1 mm,
as in addenda 14-16) and the south-west corner is snapped to the nearest integer-metre lattice line; the offsets (reprojection
shift and snap residual) are reported. (A first run floored the corner, as a15_cut_tiles does for the AP originals whose corners
are not integers; the 0.1 mm negative northing shift then moved every control tile 1 m south of its NCALM footprint; those tiles
were unregistered and re-cut, addendum 17 report section 1.) Writes draw_gmc/GMC_tiles.csv for a15_cut_tiles.py --sites GMC and GMC_footprints.json.
usage: a17_gm_ctrl_tiles.py --out DIR --grid DIR [--data $LEMCVT_DATA]
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a17_common import CTRL_TILES, dump, slurm_record  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--grid", required=True, help="grid_GMC directory (the lattice the footprints must fall in)")
    p.add_argument("--data", default=os.environ.get("LEMCVT_DATA"))
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    slurm_record(a.out)
    from osgeo import osr
    src = osr.SpatialReference()
    src.ImportFromEPSG(32610)
    dst = osr.SpatialReference()
    dst.ImportFromEPSG(6339)
    for s in (src, dst):
        s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    ct = osr.CoordinateTransformation(src, dst)
    with rasterio.open(os.path.join(a.grid, "GMC_1m_filled.tif")) as ds:
        gb = ds.bounds
        gtr = ds.transform
    rows, rec = [], {"transformation": "EPSG:32610 -> EPSG:6339 by GDAL/PROJ offline", "grid_bounds_6339": [gb.left, gb.bottom, gb.right, gb.top], "tiles": {}}
    for t in CTRL_TILES:
        with rasterio.open(os.path.join(a.data, "GM-x", t, f"{t}_1m_filled.tif")) as ds:
            b = ds.bounds
        corners = {"sw": (b.left, b.bottom), "se": (b.right, b.bottom), "ne": (b.right, b.top), "nw": (b.left, b.top)}
        tc = {k: ct.TransformPoint(float(x), float(y))[:2] for k, (x, y) in corners.items()}
        shift = {k: [tc[k][0] - corners[k][0], tc[k][1] - corners[k][1]] for k in corners}
        x0, y0 = tc["sw"]
        fx, fy = int(np.round(x0)), int(np.round(y0))            # nearest lattice line (the shift is 1e-4 m)
        inside = gb.left <= fx and fx + 1000 <= gb.right and gb.bottom <= fy and fy + 1000 <= gb.top
        on_lattice = abs((fx - gtr.c) % 1.0) < 1e-6 and abs((gtr.f - fy) % 1.0) < 1e-6
        rec["tiles"][t] = {"ncalm_sw_32610": [b.left, b.bottom], "sw_6339": [x0, y0], "reprojection_shift_m_at_corners": shift, "fp_x0": fx, "fp_y0": fy,
                           "snap_residual_m": [x0 - fx, y0 - fy], "inside_control_grid": bool(inside), "on_lattice": bool(on_lattice)}
        assert inside and on_lattice, (t, rec["tiles"][t])
        rows.append({"tile": f"{t}_fema", "kind": "control", "fp_x0": fx, "fp_y0": fy, "epsg": 6339, "offset": "", "window": "", "rank": ""})
        print(t, rec["tiles"][t], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "GMC_tiles.csv"), index=False)
    rec["seconds"] = time.time()
    dump(rec, os.path.join(a.out, "GMC_footprints.json"))


if __name__ == "__main__":
    main()
