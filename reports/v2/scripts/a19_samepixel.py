#!/usr/bin/env python3
"""Addendum 19, section 2.2: the same-pixel curvature diagnostic of addendum 18 section 3.1, unchanged, on the D0_N (counted) pixel sets of the
12 survey pairs (GM1-GM7 NCALM/FEMA, AP1-AP5 PAMAP/3DEP). Four numbers per pair at L_f = 11 m, (a) old surface on old pixels ... (d) new on new;
S = median of (b)/(a), (d)/(c); P = median of (c)/(a), (d)/(b); reading by the rule of addendum 18 section 3.1.5 (rule R3).
Reads <run>/hilltops/<tile>_masks.npz; usage: a19_samepixel.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a18_common import SAME_PIXEL_RULES  # noqa: E402
from a18_prep import _curv, _map_set  # noqa: E402
from a19_common import L_F_M, SURVEY_PAIRS, dump, metre_tile, out_path, slurm_record  # noqa: E402


def reading(S, P):
    r = SAME_PIXEL_RULES
    if S <= r["surface"]["S_max"] and r["surface"]["P_range"][0] <= P <= r["surface"]["P_range"][1]:
        return "surface"
    if r["selection"]["S_range"][0] <= S <= r["selection"]["S_range"][1] and P <= r["selection"]["P_max"]:
        return "selection"
    return "mixed"


def main() -> None:
    t0 = time.time()
    out_dir = os.path.dirname(out_path("samepixel", "samepixel.json"))
    slurm_record(out_dir)
    out = {"pairs": {}, "sites": {}, "rule": SAME_PIXEL_RULES, "L_f_m": L_F_M, "pixel_set": "D0_N counted set (decision 1) of each survey's tile"}
    for _site, pairs in SURVEY_PAIRS.items():
        for old, new in pairs:
            t1 = time.time()
            td_o, td_n = metre_tile(old), metre_tile(new)
            sets, curvs = {}, {}
            for name, td, tile in (("old", td_o, old), ("new", td_n, new)):
                z = np.load(out_path("hilltops", f"{tile}_masks.npz"))
                sets[name] = z["counted"]
                curvs[name] = _curv(td.Z, td.dx, td.filled.dy)
            rec: dict = {"old": old, "new": new, "n_old": int(sets["old"].sum()), "n_new": int(sets["new"].sum())}
            va, vd = curvs["old"][sets["old"]], curvs["new"][sets["new"]]
            rec["a"] = float(np.nanmedian(va)) if va.size else None
            rec["d"] = float(np.nanmedian(vd)) if vd.size else None
            r_on, c_on, info_on = _map_set(sets["old"], td_o, td_n, L_F_M)
            r_no, c_no, info_no = _map_set(sets["new"], td_n, td_o, L_F_M)
            vb, vc = curvs["new"][r_on, c_on], curvs["old"][r_no, c_no]
            rec["b"] = float(np.nanmedian(vb)) if vb.size else None
            rec["c"] = float(np.nanmedian(vc)) if vc.size else None
            rec["mapping_old_to_new"], rec["mapping_new_to_old"] = info_on, info_no
            ok = all(rec[k] for k in ("a", "b", "c", "d"))
            rec["surface_ratios"] = [rec["b"] / rec["a"], rec["d"] / rec["c"]] if ok else None
            rec["selection_ratios"] = [rec["c"] / rec["a"], rec["d"] / rec["b"]] if ok else None
            rec["total_ratio_d_over_a"] = rec["d"] / rec["a"] if ok else None
            rec["usable_both_ge_500"] = bool(rec["n_old"] >= 500 and rec["n_new"] >= 500)
            rec["wall_s"] = time.time() - t1
            out["pairs"][f"{old}/{new}"] = rec
            print(f"{old}/{new}: n {rec['n_old']}/{rec['n_new']} a {rec['a']} b {rec['b']} c {rec['c']} d {rec['d']} ({rec['wall_s']:.0f} s)", flush=True)

    def site_rec(keys, label):
        keys = [k for k in keys if out["pairs"][k]["surface_ratios"]]
        sr = [x for k in keys for x in out["pairs"][k]["surface_ratios"]]
        pr = [x for k in keys for x in out["pairs"][k]["selection_ratios"]]
        tot = [out["pairs"][k]["total_ratio_d_over_a"] for k in keys]
        if not sr:
            return {"label": label, "pairs": keys, "reading": "no usable pair"}
        S, P = float(np.median(sr)), float(np.median(pr))
        return {"label": label, "pairs": keys, "S_surface": S, "P_selection": P, "S_values": sr, "P_values": pr, "S_range": [min(sr), max(sr)], "P_range": [min(pr), max(pr)],
                "total_ratio_median": float(np.median(tot)), "reading": reading(S, P)}

    gm = [f"{o}/{n}" for o, n in SURVEY_PAIRS["GM"]]
    ap = [f"{o}/{n}" for o, n in SURVEY_PAIRS["AP"]]
    out["sites"]["GM"] = site_rec(gm, "GM NCALM -> FEMA")
    out["sites"]["AP"] = site_rec(ap, "AP PAMAP -> 3DEP")
    out["sites"]["AP_without_AP2"] = site_rec([k for k in ap if not k.startswith("AP2/")], "AP PAMAP -> 3DEP without AP2")
    out["sites"]["GM_usable_pairs_only"] = site_rec([k for k in gm if out["pairs"][k]["usable_both_ge_500"]], "GM pairs with >= 500 px on both")
    out["sites"]["AP_usable_pairs_only"] = site_rec([k for k in ap if out["pairs"][k]["usable_both_ge_500"]], "AP pairs with >= 500 px on both")
    out["addendum_18_readings"] = {"GM": {"S": 0.974, "P": 0.797, "reading": "selection"}, "AP": {"S": 0.849, "P": 0.687, "reading": "mixed"}}
    out["wall_s"] = time.time() - t0
    dump(out, out_path("samepixel", "samepixel.json"))
    print("samepixel:", {k: (v.get("S_surface"), v.get("P_selection"), v["reading"]) for k, v in out["sites"].items()}, flush=True)


if __name__ == "__main__":
    main()
