#!/usr/bin/env python3
"""Addendum 18, section 4.2 (the cost gate): the expected core-hours per group and per section from the pilot timings.

The pilot runs (`a18_fits.py --pilot`: the thinned A_crit grid of 150 values with the Gram terms on the whole tile and both halves, no
refinement or fill; the AD r grid of 25 values without Brent) time one partitioned reduction and one AD reduction on one tile per site
(GM11, AP6, TTV2). A production partitioned run evaluates min(2000, n_unique) grid values + the staircase refinement (<= 1000) + the
1 % fill (<= 1500); an AD run 200 grid values + up to six bounded Brent refinements (<= 15 evaluations each in addendum 12). The
estimate takes the upper counts (3900 partitioned, 290 AD evaluations) times the pilot's mean seconds per evaluation of the site, and
the time limit of a task is twice its estimate (30 min floor; addendum 18, 0.4). Writes <run>/estimate/estimate.json and prints the
table in the priority order of section 7; exits 2 when the expected total (Part A used so far + Part B) exceeds 900 core-hours.
usage: a18_estimate.py [--part-evals 3900 --ad-evals 290]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a18_common import BUDGET_CORE_HOURS, FIT_STEPS, JOB_DIR, dump, run_dir  # noqa: E402

from lemcvt.results import find_runs  # noqa: E402

PRIORITY = [("5.1-5.3 primary groups (P4, AD, variants, held-out)", ["primary"]), ("5.7 and 5.6 TTV per-tile m", ["mt"]), ("5.4 trunk sensitivity", ["trunk"]),
            ("5.5 continuity (NCALM, PAMAP, AP1_3dep; GM1-GM5 at 0.31)", ["ncalm", "pamap", "ap1_3dep", "v2m"]), ("5.1 off-mesa group", ["offmesa"])]
PILOT_SITE = {"GM11": "GM", "AP6": "AP", "TTV2": "TTV"}


def used_core_hours(jobs: list[str]) -> float:
    """Elapsed x AllocCPUS over the finished tasks of the given job ids (sacct)."""
    if not jobs:
        return 0.0
    out = subprocess.check_output(["sacct", "-j", ",".join(jobs), "-X", "-P", "-n", "--format=JobID,ElapsedRaw,AllocCPUS,State"], text=True)
    tot = 0.0
    for line in out.strip().splitlines():
        jid, el, cpus, _st = line.split("|")[:4]
        tot += float(el or 0) * float(cpus or 0) / 3600.0
    return tot


def limit_str(hours: float) -> str:
    h = max(0.5, hours)
    m = int(math.ceil(h * 60 / 15.0) * 15)       # round up to 15 min
    return f"{m // 60:02d}:{m % 60:02d}:00"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--part-evals", type=int, default=3900)
    p.add_argument("--ad-evals", type=int, default=290)
    p.add_argument("--limit-factor", type=float, default=2.0)
    a = p.parse_args()
    rd = run_dir()
    ed = os.path.join(rd, "estimate")
    os.makedirs(ed, exist_ok=True)
    specs = json.load(open(os.path.join(rd, "prep", "fit_specs.json")))["specs"]
    timing: dict[str, dict] = {"part": {}, "ad": {}}
    pilots = {}
    for stage in ("part", "ad"):
        for r in find_runs(FIT_STEPS[stage]):
            if "_pilot_" not in r.name:
                continue
            pp = json.load(open(r / "params.json"))
            site = PILOT_SITE.get(pp["tile"], pp["tile"][:2])
            tm = pp["result"]["timing"]
            timing[stage][site] = {"s_per_eval_mean": tm["s_per_eval_mean"], "s_per_eval_median": tm["s_per_eval_median"], "n_timed": tm["n_timed"],
                                   "wall_total_s": tm["wall_total_s"], "n_unique_areas": pp["result"].get("n_unique_areas"), "run": r.name, "tile": pp["tile"]}
            pilots[r.name] = pp["tile"]
    missing = [(s, site) for s in ("part", "ad") for site in ("GM", "AP", "TTV") if site not in timing[s]]
    if missing:
        print("missing pilot timings:", missing)
        sys.exit(1)
    rows = []
    for key, s in specs.items():
        site = s["site"]
        ph = a.part_evals * timing["part"][site]["s_per_eval_mean"] / 3600.0
        ah = a.ad_evals * timing["ad"][site]["s_per_eval_mean"] / 3600.0
        rows.append({"spec": key, "tile": s["tile"], "group": s["group"], "site": site, "part_h": ph, "ad_h": ah, "total_h": ph + ah,
                     "part_limit": limit_str(a.limit_factor * ph), "ad_limit": limit_str(a.limit_factor * ah),
                     "part_limit_h": max(0.5, math.ceil(a.limit_factor * ph * 4) / 4), "ad_limit_h": max(0.5, math.ceil(a.limit_factor * ah * 4) / 4)})
    gate = {"GM1:v2m", "AP2:pamap"}
    by_group: dict[str, dict] = {}
    for r in rows:
        g = by_group.setdefault(r["group"], {"n": 0, "part_h": 0.0, "ad_h": 0.0, "total_h": 0.0, "limits_h": 0.0})
        already = r["spec"] in gate
        g["n"] += 1
        if not already:
            g["part_h"] += r["part_h"]
            g["ad_h"] += r["ad_h"]
            g["total_h"] += r["total_h"]
            g["limits_h"] += r["part_limit_h"] + r["ad_limit_h"]
    sections = []
    for label, groups in PRIORITY:
        sections.append({"section": label, "groups": groups, "n_specs": sum(by_group.get(g, {}).get("n", 0) for g in groups),
                         "expected_h": sum(by_group.get(g, {}).get("total_h", 0.0) for g in groups), "limits_h": sum(by_group.get(g, {}).get("limits_h", 0.0) for g in groups)})
    part_b_expected = sum(s["expected_h"] for s in sections)
    jobs = []
    jf = os.path.join(JOB_DIR, "JOBS_partA.txt")
    if os.path.exists(jf):
        jobs = [kv.split("=")[1] for kv in open(jf).read().split()]
    part_a_used = used_core_hours(jobs)
    gate_expected = sum(r["total_h"] for r in rows if r["spec"] in gate)
    total = part_a_used + gate_expected + part_b_expected
    out = {"pilot_timing": timing, "pilot_runs": pilots, "assumptions": {"part_evals": a.part_evals, "ad_evals": a.ad_evals, "limit_factor": a.limit_factor, "floor_h": 0.5},
           "specs": rows, "by_group": by_group, "sections_priority_order": sections, "part_b_expected_h": part_b_expected,
           "part_a_used_so_far_h": part_a_used, "part_a_jobs": jobs, "gate_expected_h_remaining": gate_expected, "expected_total_h": total,
           "budget_h": BUDGET_CORE_HOURS, "within_budget": bool(total <= BUDGET_CORE_HOURS)}
    dump(out, os.path.join(ed, "estimate.json"))
    print("pilot timings (s per evaluation, mean):", {s: {site: round(v["s_per_eval_mean"], 2) for site, v in d.items()} for s, d in timing.items()})
    print(f"{'section (priority order)':70s} {'specs':>5s} {'expected h':>11s} {'limits h':>9s}")
    for s in sections:
        print(f"{s['section']:70s} {s['n_specs']:5d} {s['expected_h']:11.1f} {s['limits_h']:9.1f}")
    print(f"Part B expected {part_b_expected:.1f} core-hours; Part A used so far {part_a_used:.1f}; gate remaining {gate_expected:.1f}; "
          f"EXPECTED TOTAL {total:.1f} of {BUDGET_CORE_HOURS:.0f} -> {'WITHIN BUDGET' if out['within_budget'] else 'OVER BUDGET: STOP'}")
    sys.exit(0 if out["within_budget"] else 2)


if __name__ == "__main__":
    main()
