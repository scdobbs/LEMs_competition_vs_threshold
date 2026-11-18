#!/usr/bin/env python3
"""Addendum 15, section 3 (decision 8): re-route the 1 m full grids with the water-gap rule.

Every nodata gap of the stage A 1 m grid (AP, TTV: `<site>_1m_mean.tif`, nearest-neighbour filled within 10 m, farther
cells nodata; GM: `Gabilan_mesa.tif` as delivered) is classified by 8-connected component labelling of the nodata mask:
  * edge-connected: the component touches the grid border (the area outside the footprint, the sea, the Bay, and every
    gap joined to them): stays nodata, its rim cells are outlets (data-edge outlets, the stage A convention);
  * enclosed: the component does not touch the border (ponds, pits, river reaches inside the footprint): filled by
    nearest neighbour (Euclidean, from the valid cells) so that flow passes through; recorded in the filled-gap mask.
Then priority-flood fill, D8 and drainage area on the whole grid as in stage A (a14_bigroute, lemcvt conventions).
Products (stage B run dir, grid_<site>/): <site>_1m_gapfill_mask.tif (1 = enclosed gap filled for routing),
<site>_1m_filled.tif, <site>_1m_fd.tif, <site>_1m_area_cells.tif, <site>_gaps.json, <site>_route.json.
usage: a15_reroute.py --site GM|AP|TTV --out DIR
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import rasterio
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a15_common import NODATA, SOURCE, STAGE_A, dump, rss_gb, slurm_record, write_tif  # noqa: E402

SRC = {"AP": os.path.join(STAGE_A, "grid_AP", "AP_1m_mean.tif"), "TTV": os.path.join(STAGE_A, "grid_TTV", "TTV_1m_mean.tif"),
       "GM": os.path.join(SOURCE, "GM_original", "Gabilan_mesa.tif")}


def classify_gaps(valid):
    """8-connected components of the nodata mask; enclosed = not touching the grid border. Returns (labels, enclosed_ids, n_comp)."""
    t0 = time.time()
    lab, n = ndimage.label(~valid, structure=np.ones((3, 3), bool))
    border = np.unique(np.concatenate([lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]]))
    is_edge = np.zeros(n + 1, bool)
    is_edge[border] = True
    is_edge[0] = True
    enclosed_ids = np.nonzero(~is_edge)[0]
    print("label:", n, "nodata components,", len(enclosed_ids), "enclosed, in", round(time.time() - t0, 1), "s", flush=True)
    return lab, enclosed_ids, n


def fill_enclosed(Z, valid, lab, enclosed_ids):
    """Nearest-neighbour fill of the enclosed components (sub-block EDT with a pad of min(h, w)/2 + 2 cells)."""
    objs = ndimage.find_objects(lab)
    ny, nx = Z.shape
    gapmask = np.zeros(Z.shape, np.uint8)
    sizes, dists = [], []
    t0 = time.time()
    for k in enclosed_ids:
        sl = objs[k - 1]
        h = sl[0].stop - sl[0].start
        w = sl[1].stop - sl[1].start
        pad = int(min(h, w) // 2 + 2)
        r0, r1 = max(0, sl[0].start - pad), min(ny, sl[0].stop + pad)
        c0, c1 = max(0, sl[1].start - pad), min(nx, sl[1].stop + pad)
        sub_valid = valid[r0:r1, c0:c1]
        comp = lab[r0:r1, c0:c1] == k
        if not sub_valid.any():
            continue
        dist, (ir, ic) = ndimage.distance_transform_edt(~sub_valid, return_distances=True, return_indices=True)
        zs = Z[r0:r1, c0:c1]
        zs[comp] = zs[ir[comp], ic[comp]]
        Z[r0:r1, c0:c1] = zs
        gapmask[r0:r1, c0:c1][comp] = 1
        sizes.append(int(comp.sum()))
        dists.append(float(dist[comp].max()))
    print("filled", len(sizes), "enclosed gaps,", int(sum(sizes)), "cells in", round(time.time() - t0, 1), "s", flush=True)
    return gapmask, np.array(sizes), np.array(dists)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--site", required=True, choices=["GM", "AP", "TTV", "GMF", "GMC"])
    p.add_argument("--out", required=True)
    p.add_argument("--src", default=None, help="source 1 m grid (default: the stage A/B grid of the site); GMF = the addendum 16 extension grid")
    a = p.parse_args()
    if a.src:
        SRC[a.site] = a.src
    os.makedirs(a.out, exist_ok=True)
    slurm_record(a.out)
    from a14_bigroute import d8_area_big, d8_chunked, priority_flood
    t_all = time.time()
    with rasterio.open(SRC[a.site]) as ds:
        Z = ds.read(1).astype(np.float64)
        tr, crs, nod = ds.transform, ds.crs, ds.nodata
    valid = np.isfinite(Z)
    if nod is not None and np.isfinite(nod):
        valid &= Z != nod
    else:
        valid &= Z > -1e30
    Z[~valid] = np.nan
    print(a.site, "grid", Z.shape, "valid", int(valid.sum()), "crs", crs.to_epsg(), flush=True)
    # The lemcvt D8 port (TopoAnalysis quirk) never compares the last two rows/columns eastward/southward, so valid cells
    # there get a least-uphill code and can form 2-cycles (found on GM: its data reach the grid edge; AP and TTV lattices
    # carry a nodata pad). Pad any grid whose valid cells touch the border by PAD nodata cells; products are cropped back.
    PAD = 2
    touches = valid[0, :].any() or valid[-1, :].any() or valid[:, 0].any() or valid[:, -1].any()
    if touches:
        Z = np.pad(Z, PAD, constant_values=np.nan)
        valid = np.pad(valid, PAD, constant_values=False)
        print("valid cells touch the grid border: padded by", PAD, "nodata cells on every side (products cropped back)", flush=True)
    else:
        PAD = 0
    ny, nx = Z.shape
    crop = (slice(PAD, ny - PAD), slice(PAD, nx - PAD)) if PAD else (slice(None), slice(None))
    lab, enclosed_ids, n_comp = classify_gaps(valid)
    comp_sizes = np.bincount(lab.ravel(), minlength=n_comp + 1)
    border_ids = np.setdiff1d(np.arange(1, n_comp + 1), enclosed_ids)
    gaps = {"site": a.site, "source": SRC[a.site], "shape": [ny - 2 * PAD, nx - 2 * PAD], "pad_cells": PAD, "transform": list(tr)[:6], "epsg": crs.to_epsg(), "n_valid_before": int(valid.sum()),
            "n_nodata_components": int(n_comp), "n_edge_connected_components": int(len(border_ids)), "n_edge_connected_cells": int(comp_sizes[border_ids].sum()),
            "n_enclosed_components": int(len(enclosed_ids)), "n_enclosed_cells": int(comp_sizes[enclosed_ids].sum()),
            "enclosed_area_km2": float(comp_sizes[enclosed_ids].sum() * tr.a * -tr.e / 1e6)}
    gapmask, sizes, dists = fill_enclosed(Z, valid, lab, enclosed_ids)
    del lab
    if sizes.size:
        order = np.argsort(-sizes)[:20]
        gaps.update({"enclosed_size_cells_percentiles_50_90_99_max": [float(v) for v in np.percentile(sizes, [50, 90, 99, 100])],
                     "enclosed_fill_distance_m_percentiles_50_90_99_max": [float(v) for v in np.percentile(dists, [50, 90, 99, 100])],
                     "largest_enclosed_cells": [int(sizes[i]) for i in order], "n_enclosed_ge_100_cells": int((sizes >= 100).sum()),
                     "n_enclosed_ge_1e4_cells": int((sizes >= 1e4).sum())})
    valid2 = valid | (gapmask == 1)
    gaps["n_valid_after"] = int(valid2.sum())
    gaps["seconds_gaps"] = time.time() - t_all
    dump(gaps, os.path.join(a.out, f"{a.site}_gaps.json"))
    print("gaps", json.dumps({k: v for k, v in gaps.items() if k != "largest_enclosed_cells"}, indent=1), flush=True)
    write_tif(os.path.join(a.out, f"{a.site}_1m_gapfill_mask.tif"), gapmask[crop], tr, crs)
    del gapmask, valid
    # routing (stage A conventions)
    Z = priority_flood(Z, valid2, float(tr.a))
    write_tif(os.path.join(a.out, f"{a.site}_1m_filled.tif"), np.where(valid2, Z, NODATA).astype(np.float32)[crop], tr, crs, nodata=NODATA)
    t1 = rss_gb()
    fd = d8_chunked(Z)
    fd[~valid2] = 0
    edge = valid2 & ~ndimage.binary_erosion(valid2, structure=np.ones((3, 3), bool), border_value=0)
    fd[edge] = 0
    n_edge = int(edge.sum())
    del edge
    write_tif(os.path.join(a.out, f"{a.site}_1m_fd.tif"), fd[crop], tr, crs)
    count, target = d8_area_big(fd, Z, valid2)
    write_tif(os.path.join(a.out, f"{a.site}_1m_area_cells.tif"), count[crop], tr, crs)
    roots = (target < 0) & valid2.ravel()
    ra = count.ravel()[roots]
    order = np.argsort(-ra)[:20]
    ri = np.nonzero(roots)[0][order]
    rec = {"site": a.site, "n_valid": int(valid2.sum()), "n_edge_cells": n_edge, "max_area_cells": int(count.max()), "max_area_km2": float(count.max() * tr.a * -tr.e / 1e6),
           "n_roots": int(roots.sum()), "root_area_sum_cells": int(ra.sum()), "root_area_sum_equals_valid": bool(ra.sum() == valid2.sum()),
           "largest_roots": [{"row": int(i // nx) - PAD, "col": int(i % nx) - PAD, "x": tr.c + (i % nx - PAD + 0.5) * tr.a, "y": tr.f + (i // nx - PAD + 0.5) * tr.e,
                              "area_cells": int(count.ravel()[i]), "share_of_valid": float(count.ravel()[i] / valid2.sum())} for i in ri],
           "seconds_total": time.time() - t_all, "peak_rss_GB_after_fill": t1, "peak_rss_GB": rss_gb(), "pad_cells": PAD}
    dump(rec, os.path.join(a.out, f"{a.site}_route.json"))
    print("route", json.dumps({k: v for k, v in rec.items() if k != "largest_roots"}, indent=1), rec["largest_roots"][:3], flush=True)


if __name__ == "__main__":
    main()
