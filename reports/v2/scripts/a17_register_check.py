#!/usr/bin/env python3
"""Addendum 17, section 2.1: after the re-cut under the decision-1 windows, confirm that the registered TTV-x and GM-FEMA2018-x
files are byte-identical to the addendum 16 parked cut (same grids, same footprints), that every manifest row of the two versions
(and of the GM-x area_model rows) matches the file on disk, and summarise the unit checks. Output: register_check.json.
usage: a17_register_check.py --out DIR --tiles-ttv DIR --tiles-gmf DIR [--data $LEMCVT_DATA]
"""
import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a17_common import A16, REPO, Z_RANGE, dump, sha256_file, slurm_record  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--tiles-ttv", required=True)
    p.add_argument("--tiles-gmf", required=True)
    p.add_argument("--data", default=os.environ.get("LEMCVT_DATA"))
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    slurm_record(a.out)
    t0 = time.time()
    rec = {"windows": {k: list(v) for k, v in Z_RANGE.items()}}
    for site, new_dir, old_dir in (("TTV", a.tiles_ttv, os.path.join(A16, "tiles")), ("GMF", a.tiles_gmf, os.path.join(A16, "tiles_gmf"))):
        new = json.load(open(os.path.join(new_dir, "tiles.json")))
        old = json.load(open(os.path.join(old_dir, "tiles.json")))
        o = {t["tile"]: t for t in old["tiles"]}
        per = []
        for t in new["tiles"]:
            ot = o.get(t["tile"])
            same = ot is not None and all(ot["files"][k]["sha256"] == t["files"][k]["sha256"] for k in t["files"])
            per.append({"tile": t["tile"], "kind": t["kind"], "z_min": t["z_min"], "z_max": t["z_max"], "z_median": t["z_median"], "unit_check_pass": t["unit_check_pass"],
                        "a16_unit_check_pass": ot["unit_check_pass"] if ot else None, "identical_to_a16_parked": same, "fp": [t["fp_x0"], t["fp_y0"]], "A_max_m2": t["A_max_m2"]})
        rec[site] = {"n": len(per), "all_pass": all(x["unit_check_pass"] for x in per), "all_identical_to_a16": all(x["identical_to_a16_parked"] for x in per),
                     "n_failed_a16": sum(1 for x in per if x["a16_unit_check_pass"] is False), "tiles": per, "window_m": list(Z_RANGE[site]), "unit_check_failures": [x["tile"] for x in new["unit_check_failures"]]}
        print(site, {k: v for k, v in rec[site].items() if k != "tiles"}, flush=True)
    # manifest against disk
    rows = list(csv.DictReader(open(os.path.join(REPO, "data", "manifest.csv"))))
    chk = {"TTV-x": [], "GM-FEMA2018-x": [], "GM-x": []}
    for r in rows:
        if r["site"] in ("TTV-x", "GM-FEMA2018-x") or (r["site"] == "GM-x" and r["role"] in ("area_model", "area_model_rule")):
            path = os.path.join(a.data, r["path"])
            ok = os.path.exists(path) and sha256_file(path) == r["sha256"] and os.path.getsize(path) == int(r["bytes"])
            chk[r["site"]].append({"path": r["path"], "ok": bool(ok)})
    rec["manifest"] = {k: {"rows": len(v), "all_ok": all(x["ok"] for x in v), "bad": [x["path"] for x in v if not x["ok"]]} for k, v in chk.items()}
    import yaml
    cfg = yaml.safe_load(open(os.path.join(REPO, "configs", "data.yaml")))
    rec["data_yaml_versions"] = {k: {"n_tiles": v["n_tiles"], "tiles": v["tiles"], "unit_check_window_m": v.get("unit_check_window_m")} for k, v in cfg["versions"].items()}
    rec["seconds"] = time.time() - t0
    dump(rec, os.path.join(a.out, "register_check.json"))
    print("manifest", rec["manifest"], flush=True)
    bad = [s for s in ("TTV", "GMF") if not (rec[s]["all_pass"] and rec[s]["all_identical_to_a16"])] + [k for k, v in rec["manifest"].items() if not v["all_ok"]]
    if bad:
        print("CHECK FAILED", bad, flush=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
