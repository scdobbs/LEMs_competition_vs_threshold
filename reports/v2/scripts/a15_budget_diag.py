#!/usr/bin/env python3
"""Addendum 15, section 3: diagnose a non-closed area budget (root area sum != valid cells) on a routed 1 m grid.
Reports cells whose D8 receiver is higher than themselves (processed out of order by the descending accumulation),
2-cycles, and cells whose receiver is nodata. usage: a15_budget_diag.py --dir RUN/grid_GM --site GM"""
import argparse
import json
import os
import sys

import numpy as np
import rasterio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a15_common import NODATA, dump

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "src"))
from lemcvt.hydro.d8 import d8_receiver_arrays

p = argparse.ArgumentParser()
p.add_argument("--dir", required=True)
p.add_argument("--site", required=True)
a = p.parse_args()
with rasterio.open(os.path.join(a.dir, f"{a.site}_1m_filled.tif")) as ds:
    Z = ds.read(1).astype(np.float64)
with rasterio.open(os.path.join(a.dir, f"{a.site}_1m_fd.tif")) as ds:
    fd = ds.read(1)
with rasterio.open(os.path.join(a.dir, f"{a.site}_1m_area_cells.tif")) as ds:
    cnt = ds.read(1)
valid = Z != NODATA
ny, nx = Z.shape
r_to, c_to, good = d8_receiver_arrays(fd)
target = np.where(good.ravel() & valid.ravel(), r_to.ravel().astype(np.int64) * nx + c_to.ravel(), -1)
del r_to, c_to
tv = target >= 0
to_nodata = tv & ~valid.ravel()[np.maximum(target, 0)]
zf = Z.ravel()
up = tv & ~to_nodata
higher = np.zeros(zf.size, bool)
higher[up] = zf[target[up]] > zf[up]
equal = np.zeros(zf.size, bool)
equal[up] = zf[target[up]] == zf[up]
cyc2 = np.zeros(zf.size, bool)
idx = np.nonzero(up)[0]
t = target[idx]
tt = target[t]
cyc2[idx] = tt == idx
roots = (target < 0) & valid.ravel()
rec = {"site": a.site, "n_valid": int(valid.sum()), "n_roots": int(roots.sum()), "root_area_sum": int(cnt.ravel()[roots].sum()),
       "n_receiver_higher": int(higher.sum()), "n_receiver_equal": int(equal.sum()), "n_two_cycles": int(cyc2.sum()), "n_receiver_nodata": int(to_nodata.sum()),
       "fd_codes_of_higher": {int(k): int(v) for k, v in zip(*np.unique(fd.ravel()[higher], return_counts=True), strict=True)},
       "z_range_higher": [float(zf[higher].min()), float(zf[higher].max())] if higher.any() else None,
       "example_higher_rc": [[int(i // nx), int(i % nx)] for i in np.nonzero(higher)[0][:10]],
       "area_in_higher_cells": int(cnt.ravel()[higher].sum())}
dump(rec, os.path.join(a.dir, f"{a.site}_budget_diag.json"))
print(json.dumps(rec, indent=1))
