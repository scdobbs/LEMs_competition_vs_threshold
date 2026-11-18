#!/usr/bin/env python3
"""Addendum 16, section 4 (decision 4): the GM drainage-area check for GM1-GM5 and the modelling area grids.

For each original GM tile: old = the existing tile area grid (GM/tile<n>/GM_1m_best_tile_area.tif, cut by the legacy
pipeline with the upstream area from beyond the lidar footprint), new = the stage B re-cut (GM-x/GM<n>/GM<n>_1m_area.tif,
clipped from the routing of Gabilan_mesa.tif alone, data-edge outlets). D = old - new is the area that enters the lidar
footprint from outside and reaches the cell; it is constant along a trunk between confluences with other outside-fed
tributaries, so its value where a trunk enters the tile is the total outside inflow of that trunk.
Entry cells of the tile: tile-ring cells with D > 0 whose D8 receiver lies inside the tile. For each, the 1 m main stem
is walked upstream on the full-grid routing (largest-area donor at every step) until it reaches a data-edge cell of the
lidar: the cell where the trunk enters the lidar footprint (section 4.1).
Coarse inflow (section 4.2): on the stage A 10 m routing of the Perron grid (pools/GM_area_full.tif, GM_fd_full.tif), the
area each coarse cell receives from outside the NCALM footprint = A_coarse - A_coarse_inside_only, where the inside-only
accumulation runs on the same D8 with the outside cells carrying no area. At the tile's entry cell the check compares
D with that coarse quantity at the matching coarse cell (maximum over the 3 x 3 coarse neighbourhood, since the coarse
channel can sit one cell off the 1 m trunk); the coarse inflow at the lidar-entry cell found by the walk is reported too.
Pass: |D - coarse| / D <= 10 % at every entry of the tile (section 4.3).
Modelling grid (section 4.4): <tile>_1m_area_model.tif = the old grid where the tile passes, else new + the coarse inflow
added at each entry cell and propagated along the tile's D8 downstream of it. Registered as role area_model in
data/manifest.csv (GM-x). Nothing is overwritten.
usage: a16_gm_area_check.py --out DIR [--data $LEMCVT_DATA] [--no-register]
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import rasterio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a16_common import GM_AREA_CHECK_TOL, STAGE_A, STAGE_B, dump, sha256_file, slurm_record  # noqa: E402

from lemcvt.hydro.d8 import d8_receiver_arrays  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
N = 1000
TILES = ["GM1", "GM2", "GM3", "GM4", "GM5"]
_CODES = np.array([1, 2, 4, 8, 16, 32, 64, 128])
_DR = np.array([0, 1, 1, 1, 0, -1, -1, -1])
_DC = np.array([1, 1, 0, -1, -1, -1, 0, 1])


def read_tile(path):
    with rasterio.open(path) as ds:
        return ds.read(1), ds.transform, ds.crs, ds.nodata


def donors_of(fd, r, c):
    """(row, col) of the neighbours whose D8 code points at (r, c)."""
    ny, nx = fd.shape
    out = []
    for code, dr, dc in zip(_CODES, _DR, _DC, strict=True):
        rr, cc = r - dr, c - dc          # the neighbour at (-dr, -dc) flows to (r, c) with code `code`
        if 0 <= rr < ny and 0 <= cc < nx and fd[rr, cc] == code:
            out.append((rr, cc))
    return out


def walk_main_stem_up(fd, area, valid, r, c, max_steps=200000):
    """Follow the largest-area donor upstream until a data-edge cell (valid cell bordering nodata/grid edge) or a source."""
    ny, nx = fd.shape
    path_len = 0
    while path_len < max_steps:
        # data-edge?
        edge = r == 0 or c == 0 or r == ny - 1 or c == nx - 1
        if not edge:
            edge = not valid[r - 1:r + 2, c - 1:c + 2].all()
        if edge:
            return (r, c), "data_edge", path_len
        d = donors_of(fd, r, c)
        if not d:
            return (r, c), "source", path_len
        r, c = max(d, key=lambda rc: area[rc])
        path_len += 1
    return (r, c), "max_steps", path_len


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--data", default=os.environ.get("LEMCVT_DATA"))
    p.add_argument("--no-register", action="store_true")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    slurm_record(a.out)
    t0 = time.time()
    # full 1 m grid (stage B)
    gb = os.path.join(STAGE_B, "grid_GM")
    with rasterio.open(os.path.join(gb, "GM_1m_fd.tif")) as ds:
        fd = ds.read(1)
        tr1 = ds.transform
    with rasterio.open(os.path.join(gb, "GM_1m_area_cells.tif")) as ds:
        area = ds.read(1)
    with rasterio.open(os.path.join(gb, "GM_1m_filled.tif")) as ds:
        zf = ds.read(1)
        nod = ds.nodata
    valid = zf != nod
    del zf
    print("full grid", fd.shape, "valid", int(valid.sum()), round(time.time() - t0), "s", flush=True)
    # coarse grid (stage A, Perron 10 m): outside-inflow per cell
    with rasterio.open(os.path.join(STAGE_A, "pools", "GM_area_full.tif")) as ds:
        A_c = ds.read(1).astype(np.float64)
        trc = ds.transform
        A_c[A_c == ds.nodata] = np.nan
    with rasterio.open(os.path.join(STAGE_A, "pools", "GM_fd_full.tif")) as ds:
        fd_c = ds.read(1)
    cyc, cxc = A_c.shape
    dxc = float(trc.a)
    # inside mask: coarse cells whose centre lies in a valid 1 m cell (the NCALM footprint)
    cx = trc.c + (np.arange(cxc) + 0.5) * trc.a
    cy = trc.f + (np.arange(cyc) + 0.5) * trc.e
    col1 = np.floor((cx - tr1.c) / tr1.a).astype(int)
    row1 = np.floor((tr1.f - cy) / -tr1.e).astype(int)
    inside = np.zeros(A_c.shape, bool)
    okc = (col1 >= 0) & (col1 < fd.shape[1])
    okr = (row1 >= 0) & (row1 < fd.shape[0])
    rr_, cc_ = np.meshgrid(np.where(okr, row1, 0), np.where(okc, col1, 0), indexing="ij")
    inside = valid[rr_, cc_] & okr[:, None] & okc[None, :] & np.isfinite(A_c)
    # inside-only accumulation on the coarse D8 (lemcvt order is irrelevant for sums; use topological accumulation via receivers)
    r_to, c_to, good = d8_receiver_arrays(fd_c)
    good &= np.isfinite(A_c)
    good[good] &= np.isfinite(A_c)[r_to[good], c_to[good]]
    tgt = np.where(good.ravel(), r_to.ravel() * cxc + c_to.ravel(), -1)
    # process cells in descending coarse area (donors have smaller area than receivers): a valid topological order for a tree
    order = np.argsort(-np.nan_to_num(A_c, nan=-1).ravel(), kind="stable")[::-1]   # ascending area: donors first
    w_in = np.where(inside, dxc * dxc, 0.0).ravel()
    acc = w_in.copy()
    for k in order:
        t = tgt[k]
        if t >= 0:
            acc[t] += acc[k]
    A_in = acc.reshape(A_c.shape)
    ext_c = np.where(np.isfinite(A_c), A_c - A_in, np.nan)      # area from outside the footprint reaching each coarse cell
    ext_c[~inside] = np.nan
    # direct inflow at a coarse cell: donors outside the footprint
    direct = np.zeros(A_c.shape)
    for k in np.nonzero((tgt >= 0) & ~inside.ravel())[0]:
        t = tgt[k]
        if inside.ravel()[t]:
            direct.ravel()[t] += A_c.ravel()[k]
    print("coarse: inside cells", int(inside.sum()), "max ext", float(np.nanmax(ext_c)) / 1e6, "km2; direct-inflow cells", int((direct > 0).sum()), flush=True)
    # per tile
    results = {}
    manifest_rows = []
    for t in TILES:
        n = t[-1]
        old, tro, crso, _ = read_tile(os.path.join(a.data, "GM", f"tile{n}", "GM_1m_best_tile_area.tif"))
        new, trn, crsn, _ = read_tile(os.path.join(a.data, "GM-x", t, f"{t}_1m_area.tif"))
        fdt, _, _, _ = read_tile(os.path.join(a.data, "GM-x", t, f"{t}_1m_fd.tif"))
        assert tro == trn, (t, tro, trn)
        old = old.astype(np.float64)
        new = new.astype(np.float64)
        D = old - new
        x0, y1 = trn.c, trn.f
        c0 = int(round((x0 - tr1.c) / tr1.a))
        r0 = int(round((tr1.f - y1) / -tr1.e))
        # entries: ring cells with D > 0 whose receiver is inside the tile
        ring = np.zeros((N, N), bool)
        ring[0, :] = ring[-1, :] = ring[:, 0] = ring[:, -1] = True
        rt, ct, gd = d8_receiver_arrays(fdt)
        inward = gd & ring & (D > 0.5)
        entries = []
        for r, c in zip(*np.nonzero(inward), strict=True):
            # skip ring cells that receive from another ring cell of the same trunk (keep the most upstream = first cell of the trunk in the tile):
            dons = donors_of(fdt, r, c)
            if any(ring[dr, dc] and D[dr, dc] > 0.5 for dr, dc in dons):
                continue
            R, C = r0 + r, c0 + c
            (er, ec), how, steps = walk_main_stem_up(fd, area, valid, R, C)
            # coarse cell of the tile entry and of the lidar entry
            def coarse_at(RR, CC, arr):
                X = tr1.c + (CC + 0.5) * tr1.a
                Y = tr1.f + (RR + 0.5) * tr1.e
                cc = int(np.floor((X - trc.c) / trc.a))
                rr = int(np.floor((trc.f - Y) / -trc.e))
                win = arr[max(rr - 1, 0):rr + 2, max(cc - 1, 0):cc + 2]
                return (float(np.nanmax(win)) if np.isfinite(win).any() else np.nan), (rr, cc), float(X), float(Y)
            ext_tile, rc_tile, X_t, Y_t = coarse_at(R, C, ext_c)
            dir_entry, rc_e, X_e, Y_e = coarse_at(er, ec, direct)
            Ac_entry, _, _, _ = coarse_at(er, ec, A_c)
            ext_entry, _, _, _ = coarse_at(er, ec, ext_c)
            d_val = float(D[r, c])
            rel = abs(d_val - ext_tile) / d_val if d_val > 0 and np.isfinite(ext_tile) else np.nan
            entries.append({"tile_cell_rc": [int(r), int(c)], "tile_cell_EN": [X_t, Y_t], "D_old_minus_new_m2": d_val, "old_m2": float(old[r, c]), "new_m2": float(new[r, c]),
                            "coarse_outside_inflow_at_tile_entry_m2": ext_tile, "coarse_rc_tile_entry": [int(rc_tile[0]), int(rc_tile[1])],
                            "lidar_entry_EN": [X_e, Y_e], "lidar_entry_how": how, "walk_steps": steps, "lidar_entry_new_area_m2": float(area[er, ec]),
                            "coarse_area_at_lidar_entry_m2": Ac_entry, "coarse_direct_inflow_at_lidar_entry_m2": dir_entry, "coarse_outside_inflow_at_lidar_entry_m2": ext_entry,
                            "rel_diff": rel, "pass": bool(np.isfinite(rel) and rel <= GM_AREA_CHECK_TOL)})
        n_diff = int((D > 0.5).sum())
        passed = all(e["pass"] for e in entries) if entries else True
        results[t] = {"n_cells_D_gt_0": n_diff, "D_max_m2": float(D.max()), "D_min_m2": float(D.min()), "n_cells_D_lt_0": int((D < -0.5).sum()), "entries": entries,
                      "n_entries": len(entries), "pass": passed, "fp": [float(x0), float(trn.f - N)]}
        print(t, json.dumps({k: v for k, v in results[t].items() if k != "entries"}), flush=True)
        for e in entries:
            print("   entry", json.dumps(e), flush=True)
        # modelling grid
        if passed:
            model = old.copy()
            note = "old tile area grid (check passed)"
        else:
            model = new.copy()
            for e in entries:
                add = e["coarse_outside_inflow_at_tile_entry_m2"]
                r, c = e["tile_cell_rc"]
                while True:
                    model[r, c] += add
                    if not gd[r, c]:
                        break
                    r, c = int(rt[r, c]), int(ct[r, c])
                    if not (0 <= r < N and 0 <= c < N):
                        break
            note = "new area + coarse inflow added at the entry cells and propagated downstream (check failed)"
        tdir = os.path.join(a.data, "GM-x", t)
        out_path = os.path.join(tdir, f"{t}_1m_area_model.tif")
        if os.path.exists(out_path):
            raise FileExistsError(out_path + ": exists; nothing is overwritten")
        prof = {"driver": "GTiff", "height": N, "width": N, "count": 1, "dtype": "float64", "crs": crsn, "transform": trn, "compress": "deflate", "tiled": True, "blockxsize": 256, "blockysize": 256}
        with rasterio.open(out_path, "w", **prof) as dst:
            dst.write(model, 1)
        results[t]["area_model"] = {"path": f"GM-x/{t}/{t}_1m_area_model.tif", "note": note, "sha256": sha256_file(out_path), "bytes": os.path.getsize(out_path),
                                    "max_m2": float(model.max()), "equals_old": bool(np.array_equal(model, old)), "equals_new": bool(np.array_equal(model, new))}
        manifest_rows.append({"path": results[t]["area_model"]["path"], "sha256": results[t]["area_model"]["sha256"], "bytes": results[t]["area_model"]["bytes"], "site": "GM-x", "tile": t, "kind": "input", "role": "area_model"})
    rec = {"tiles": results, "tolerance": GM_AREA_CHECK_TOL, "coarse": {"source": os.path.join(STAGE_A, "pools", "GM_area_full.tif"), "inside_cells": int(inside.sum()), "n_direct_inflow_cells": int((direct > 0).sum()),
                                                                        "largest_direct_inflows": sorted([{"rc": [int(i // cxc), int(i % cxc)], "E": float(trc.c + (i % cxc + 0.5) * trc.a), "N": float(trc.f + (i // cxc + 0.5) * trc.e), "m2": float(v)} for i, v in enumerate(direct.ravel()) if v > 1e5], key=lambda d: -d["m2"])[:15]},
           "seconds": time.time() - t0}
    dump(rec, os.path.join(a.out, "gm_area_check.json"))
    if not a.no_register:
        mpath = os.path.join(REPO, "data", "manifest.csv")
        rows = list(csv.DictReader(open(mpath)))
        have = {r["path"] for r in rows}
        added = [r for r in manifest_rows if r["path"] not in have]
        rows += added
        with open(mpath, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["path", "sha256", "bytes", "site", "tile", "kind", "role"])
            w.writeheader()
            w.writerows(rows)
        import yaml
        ypath = os.path.join(REPO, "configs", "data.yaml")
        txt = open(ypath).read()
        cfg = yaml.safe_load(txt)
        v = cfg["versions"]["GM-x"]
        if "area_model (m^2, float64; GM1-GM5 only, addendum 16 section 4)" not in v["products"]:
            v["products"].append("area_model (m^2, float64; GM1-GM5 only, addendum 16 section 4)")
        v["area_model"] = {t: results[t]["area_model"]["note"] for t in TILES}
        header = "".join(line + "\n" for line in txt.splitlines() if line.startswith("#"))
        with open(ypath, "w") as fh:
            fh.write(header)
            fh.write(yaml.safe_dump(cfg, sort_keys=False, width=120))
        print("registered", len(added), "manifest rows", flush=True)
    print("done", round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
