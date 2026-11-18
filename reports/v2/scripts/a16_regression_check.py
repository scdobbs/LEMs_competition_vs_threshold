#!/usr/bin/env python3
"""Addendum 16, section 5.4: compare the regression forward runs (a16_regression_task.sh) with the v2.0.0 release runs.

For every forward run <A16_RUN_ID>_reg_<tile>_<ad|partitioned-refined> under $LEMCVT_RESULTS: the run key's MSE against the
release params.json entry (exact equality and relative difference), the Phase 3 comparison that the run itself recorded
(max |dz| against the release elevation grid), and a cell-by-cell comparison of every derived grid the release run holds
(elevation, filled, flow_direction, area, logarea, hillshade): identical bytes (sha256) and identical values.
Rule (section 5.4): every grid identical, every MSE equal to the release value; otherwise stop and report.
usage: a16_regression_check.py --run-id <A16_RUN_ID> --out DIR
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import rasterio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a16_common import RESULTS_ROOT, dump  # noqa: E402

REFS = {"GM1": {"ad": "20261111T012445Z_tile_run_GM1_v2_ad", "partitioned-refined": "20261111T050111Z_tile_run_GM1_v2_partitioned-refined"},
        "AP1": {"ad": "20261111T012439Z_tile_run_AP1_v2_ad", "partitioned-refined": "20261111T050726Z_tile_run_AP1_v2_partitioned-refined"}}
KEYS = {"ad_no_opt": "AD-no-opt", "ad_opt": "AD-opt", "acrit_opt": "Partitioned-model-a_crit-opt", "joint_opt": "Partitioned-model-a_crit-k-d-opt",
        "acrit_opt_D0B": "Partitioned-model-a_crit-opt_D0B"}
ROLES = ("elevation", "filled", "flow_direction", "area", "logarea", "hillshade")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    out = {"runs": {}, "all_pass": True}
    for tile, refs in REFS.items():
        site = tile[:2]
        for kind, ref in refs.items():
            new_dir = os.path.join(RESULTS_ROOT, f"{a.run_id}_reg_{tile}_{kind}")
            ref_dir = os.path.join(RESULTS_ROOT, ref)
            rec = {"new_run": new_dir, "reference_run": ref_dir, "exists": os.path.isdir(new_dir)}
            if not rec["exists"]:
                out["all_pass"] = False
                out["runs"][f"{tile}_{kind}"] = rec
                continue
            newp = json.load(open(os.path.join(new_dir, "params.json")))
            refp = json.load(open(os.path.join(ref_dir, "params.json")))
            ref_mse = {fr["key"]: fr["mse"] for fr in refp.get("forward_runs", [])}
            rec["forward_runs"] = []
            for fr in newp.get("forward_runs", []):
                k = fr["key"]
                r = {"key": k, "mse_new": fr["mse"], "mse_release": ref_mse.get(k), "mse_equal": fr["mse"] == ref_mse.get(k),
                     "mse_rel_diff": abs(fr["mse"] - ref_mse[k]) / ref_mse[k] if k in ref_mse else None, "wall_s": fr.get("wall_s"),
                     "phase3_comparison": {kk: fr["comparison_phase2"].get(kk) for kk in ("max_abs_dz", "max_abs_dz_over_relief", "rel_mse_diff", "pass_dz", "pass_mse")} if fr.get("comparison_phase2") else None,
                     "grids": {}}
                prefix = f"{site}_1m_best_tile_{KEYS.get(k, k)}"
                for role in ROLES:
                    fn = f"{prefix}_{role}.tif"
                    pn, pr = os.path.join(new_dir, fn), os.path.join(ref_dir, fn)
                    if not os.path.exists(pr):
                        r["grids"][role] = {"reference": "absent"}
                        continue
                    if not os.path.exists(pn):
                        r["grids"][role] = {"new": "absent"}
                        r["grids_pass"] = False
                        continue
                    with rasterio.open(pn) as d1, rasterio.open(pr) as d2:
                        g1, g2 = d1.read(1), d2.read(1)
                    same_vals = bool(np.array_equal(g1, g2, equal_nan=True)) if g1.shape == g2.shape else False
                    r["grids"][role] = {"sha256_equal": sha(pn) == sha(pr), "values_identical": same_vals, "dtype": str(g1.dtype),
                                        "max_abs_diff": float(np.nanmax(np.abs(g1.astype(np.float64) - g2.astype(np.float64)))) if g1.shape == g2.shape else None}
                r["grids_pass"] = all(v.get("values_identical", True) for v in r["grids"].values())
                r["pass"] = bool(r["mse_equal"] and r["grids_pass"])
                out["all_pass"] &= r["pass"]
                rec["forward_runs"].append(r)
            rec["git_hash"] = json.load(open(os.path.join(new_dir, "run.json"))).get("git_hash")
            rec["reference_git_hash"] = json.load(open(os.path.join(ref_dir, "run.json"))).get("git_hash")
            out["runs"][f"{tile}_{kind}"] = rec
    dump(out, os.path.join(a.out, "regression.json"))
    for name, rec in out["runs"].items():
        for r in rec.get("forward_runs", []):
            print(name, r["key"], "mse_equal", r["mse_equal"], f"rel {r['mse_rel_diff']:.2e}" if r["mse_rel_diff"] is not None else "", "grids", {k: v.get("values_identical", v) for k, v in r["grids"].items()}, "PASS" if r.get("pass") else "FAIL")
    print("ALL PASS" if out["all_pass"] else "REGRESSION FAILED")


if __name__ == "__main__":
    main()
