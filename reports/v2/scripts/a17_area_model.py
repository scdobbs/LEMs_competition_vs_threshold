#!/usr/bin/env python3
"""Addendum 17, section 2.1 (decision 2): GM1-GM5 modelling area = the old tile area grids.

For each of GM1-GM5 under $LEMCVT_DATA/GM-x/<tile>/: the addendum 16 rule grid <tile>_1m_area_model.tif is renamed
<tile>_1m_area_model_rule.tif (kept, unused), and a new <tile>_1m_area_model.tif is written as a copy of the old tile area grid
GM/tile<n>/GM_1m_best_tile_area.tif (float64, the GM-x tile's transform and CRS, which the old grid shares). data/manifest.csv:
the five area_model rows get the new checksums, five area_model_rule rows are added. configs/data.yaml: the area_model notes
are rewritten and an area_model_rule block added. Nothing else is touched; the old grids are not moved.
usage: a17_area_model.py --out DIR [--data $LEMCVT_DATA]
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import rasterio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a17_common import REPO, dump, sha256_file, slurm_record  # noqa: E402

TILES = ["GM1", "GM2", "GM3", "GM4", "GM5"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--data", default=os.environ.get("LEMCVT_DATA"))
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    slurm_record(a.out)
    t0 = time.time()
    rec = {}
    rows_new = []
    for t in TILES:
        n = t[-1]
        tdir = os.path.join(a.data, "GM-x", t)
        model = os.path.join(tdir, f"{t}_1m_area_model.tif")
        rule = os.path.join(tdir, f"{t}_1m_area_model_rule.tif")
        old = os.path.join(a.data, "GM", f"tile{n}", "GM_1m_best_tile_area.tif")
        if os.path.exists(rule):
            raise FileExistsError(rule + ": exists; this script is not re-runnable (nothing is overwritten)")
        with rasterio.open(old) as ds:
            a_old = ds.read(1).astype(np.float64)
            tr_old, crs_old = ds.transform, ds.crs
        with rasterio.open(model) as ds:
            a_rule = ds.read(1).astype(np.float64)
            tr, crs = ds.transform, ds.crs
        with rasterio.open(os.path.join(tdir, f"{t}_1m_area.tif")) as ds:
            a_new = ds.read(1).astype(np.float64)
            assert ds.transform == tr, (t, "area grid transform")
        assert tr_old == tr, (t, tr_old, tr)
        assert a_old.shape == a_rule.shape == (1000, 1000), (t, a_old.shape)
        sha_rule_before = sha256_file(model)
        os.rename(model, rule)                               # the addendum 16 rule grid, kept beside the new one
        prof = {"driver": "GTiff", "height": 1000, "width": 1000, "count": 1, "dtype": "float64", "crs": crs, "transform": tr, "compress": "deflate", "tiled": True, "blockxsize": 256, "blockysize": 256}
        with rasterio.open(model, "w", **prof) as dst:
            dst.write(a_old, 1)
        with rasterio.open(model) as ds:
            back = ds.read(1)
        assert np.array_equal(back, a_old), t
        rec[t] = {"old_grid": old, "old_crs": str(crs_old), "tile_crs": str(crs), "area_model": {"path": f"GM-x/{t}/{t}_1m_area_model.tif", "sha256": sha256_file(model), "bytes": os.path.getsize(model),
                                                                                                    "max_m2": float(a_old.max()), "equals_old": True, "equals_rule": bool(np.array_equal(a_old, a_rule))},
                  "area_model_rule": {"path": f"GM-x/{t}/{t}_1m_area_model_rule.tif", "sha256": sha256_file(rule), "bytes": os.path.getsize(rule), "sha256_before_rename": sha_rule_before,
                                      "max_m2": float(a_rule.max()), "equals_old": bool(np.array_equal(a_rule, a_old)), "equals_new": bool(np.array_equal(a_rule, a_new))},
                  "outlet_old_m2": float(a_old.max()), "outlet_rule_m2": float(a_rule.max()), "outlet_new_lidar_only_m2": float(a_new.max()),
                  "cells_rule_lt_old": int((a_rule < a_old - 0.5).sum()), "max_old_minus_rule_m2": float((a_old - a_rule).max())}
        assert rec[t]["area_model_rule"]["sha256"] == sha_rule_before
        rows_new.append({"path": rec[t]["area_model_rule"]["path"], "sha256": rec[t]["area_model_rule"]["sha256"], "bytes": rec[t]["area_model_rule"]["bytes"], "site": "GM-x", "tile": t, "kind": "input", "role": "area_model_rule"})
        print(t, {k: v for k, v in rec[t].items() if k.startswith(("outlet", "cells", "max"))}, flush=True)
    # manifest
    mpath = os.path.join(REPO, "data", "manifest.csv")
    rows = list(csv.DictReader(open(mpath)))
    n_upd = 0
    for r in rows:
        if r["site"] == "GM-x" and r["role"] == "area_model" and r["tile"] in rec:
            r["sha256"] = rec[r["tile"]]["area_model"]["sha256"]
            r["bytes"] = str(rec[r["tile"]]["area_model"]["bytes"])
            n_upd += 1
    have = {r["path"] for r in rows}
    added = [r for r in rows_new if r["path"] not in have]
    rows += added
    with open(mpath, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "sha256", "bytes", "site", "tile", "kind", "role"])
        w.writeheader()
        w.writerows(rows)
    # data.yaml
    import yaml
    ypath = os.path.join(REPO, "configs", "data.yaml")
    txt = open(ypath).read()
    cfg = yaml.safe_load(txt)
    v = cfg["versions"]["GM-x"]
    v["products"] = [x for x in v["products"] if not x.startswith("area_model")]
    v["products"] += ["area_model (m^2, float64; GM1-GM5 only: the old tile area grids, addendum 17 decision 2)",
                      "area_model_rule (m^2, float64; GM1-GM5 only: the addendum 16 section 4 rule grids, unused)"]
    v["area_model"] = {t: "old tile area grid GM/tile<n>/GM_1m_best_tile_area.tif (addendum 17 decision 2)" for t in TILES}
    v["area_model_rule"] = {t: ("old tile area grid (addendum 16 check passed)" if rec[t]["area_model_rule"]["equals_old"] else "new area + strip inflow at the entry cells (addendum 16 check failed); unused") for t in TILES}
    header = "".join(line + "\n" for line in txt.splitlines() if line.startswith("#"))
    with open(ypath, "w") as fh:
        fh.write(header)
        fh.write(yaml.safe_dump(cfg, sort_keys=False, width=120))
    rec["_manifest"] = {"rows_updated": n_upd, "rows_added": len(added)}
    rec["_seconds"] = time.time() - t0
    dump(rec, os.path.join(a.out, "area_model.json"))
    print("manifest: updated", n_upd, "added", len(added), "done", round(time.time() - t0, 1), "s", flush=True)


if __name__ == "__main__":
    main()
