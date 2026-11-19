#!/usr/bin/env python3
"""Addendum 18, section 4.1 (the regression gate): the P4 joint fit and the AD fit on GM1 (m = 0.31) and AP2 (PAMAP, m = 0.47) against
the v2.0.0 build, to a relative 1e-9 on the minimum MSE and the fitted K, D and A_crit (and r for AD).

Reads the a18_part / a18_ad runs of the specs GM1:v2m and AP2:pamap (their `phase6_check`: the union of this run's rows with the stored
Phase 6 rows, and the AD optimum against the Phase 6 optimum). Writes <run>/gate/gate.json; exits 1 when the gate fails.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a18_common import FIT_STEPS, GATE_REL_TOL, dump, run_dir  # noqa: E402

from lemcvt.results import find_runs  # noqa: E402

GATE = [("GM1", "v2m"), ("AP2", "pamap")]


def latest(step: str, label: str):
    runs = find_runs(step, label)
    if not runs:
        return None
    return runs[-1], json.load(open(runs[-1] / "params.json"))


def main():
    rd = run_dir()
    gd = os.path.join(rd, "gate")
    os.makedirs(gd, exist_ok=True)
    out = {"tolerance": GATE_REL_TOL, "tiles": {}, "pass": True}
    for tile, group in GATE:
        rec: dict = {}
        pr = latest(FIT_STEPS["part"], f"{tile}_{group}")
        ar = latest(FIT_STEPS["ad"], f"{tile}_{group}")
        if pr is None or ar is None:
            rec["missing"] = {"part": pr is None, "ad": ar is None}
            out["pass"] = False
            out["tiles"][f"{tile}:{group}"] = rec
            continue
        pc = pr[1]["result"]["phase6_check"]
        ac = ar[1]["result"]["phase6_check"]
        rec["part_run"], rec["ad_run"] = pr[0].name, ar[0].name
        rec["phase6_part_run"], rec["phase6_ad_run"] = pc.get("run"), ac.get("run")
        rec["m"] = pr[1]["model"]["m"]
        ok_p = bool(pc.get("available") and pc.get("same_model"))
        if ok_p:
            rec["part"] = {"phase6": pc["phase6_joint_optimum"], "union": pc["union_P4"], "rel_diff_union": pc["rel_diff_union_vs_phase6"],
                           "new_rows_only": pc["new_rows_only_P4"], "rel_diff_new_rows_only": pc["rel_diff_new_rows_only_vs_phase6"],
                           "max_rel_same_A_crit_rows": pc["max_rel_new_vs_phase6_P4_same_A_crit"], "n_common_rows": pc["n_common"]}
            rec["part_pass"] = bool(all(v <= GATE_REL_TOL for v in pc["rel_diff_union_vs_phase6"].values()))
        else:
            rec["part"] = pc
            rec["part_pass"] = False
        ok_a = bool(ac.get("available"))
        if ok_a:
            rec["ad"] = {"phase6": ac["phase6"], "new": ac["new"], "rel_diff": ac["rel_diff"]}
            rec["ad_pass"] = bool(all(v <= GATE_REL_TOL for v in ac["rel_diff"].values()))
        else:
            rec["ad"] = ac
            rec["ad_pass"] = False
        rec["pass"] = rec["part_pass"] and rec["ad_pass"]
        out["pass"] &= rec["pass"]
        out["tiles"][f"{tile}:{group}"] = rec
        print(f"{tile}:{group} m={rec['m']}: P4 {'PASS' if rec['part_pass'] else 'FAIL'} {rec['part'].get('rel_diff_union') if ok_p else rec['part']}; "
              f"AD {'PASS' if rec['ad_pass'] else 'FAIL'} {rec['ad'].get('rel_diff') if ok_a else rec['ad']}", flush=True)
    dump(out, os.path.join(gd, "gate.json"))
    print("GATE", "PASS" if out["pass"] else "FAIL", flush=True)
    sys.exit(0 if out["pass"] else 1)


if __name__ == "__main__":
    main()
