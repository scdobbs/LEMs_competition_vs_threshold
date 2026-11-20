#!/usr/bin/env python3
"""Addendum 19, section 5.1: the SJ runs (slope-dependent threshold, C free) on the 11 TTV primary tiles at alpha = 0, b_hat, 2 (33 runs), through
`lemcvt phase8 slope-threshold` (the module unchanged: thin 2000, refine 400, the module's alternating-rank hold-out).

Rules (0.4): serc, 1 CPU, no --account, throttled; the estimate printed first; each task's limit twice the gate-3 estimate (30 min floor);
one submission at most 300 core-hours of limits (chunked otherwise). Run ids `<ts>_a19_sj_<tile>_a<alpha>` so that the summary finds them.
usage: a19_submit_sj.py [--dry-run] [--throttle 11]
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a19_common import JOB_DIR, MAX_SUBMISSION_LIMIT_CORE_H, TTV_PRIMARY, dump, load_json, out_path  # noqa: E402


def hms(h: float) -> str:
    s = int(round(h * 3600))
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--throttle", type=int, default=33)
    a = p.parse_args()
    fv = load_json(out_path("heads", "heads.json"))["field_values"]
    gate = load_json(out_path("fixed", "TTV2.json"))["timing"]["SJ_pilot"]
    alphas = fv["alphas"]
    est_h = gate["expected_h_full_run"]
    limit_h = max(2 * est_h, 0.5)
    specs = [(t, al) for al in alphas for t in TTV_PRIMARY]
    tot_est = est_h * len(specs)
    tot_lim = limit_h * len(specs)
    print(f"SJ: {len(specs)} runs x {est_h:.2f} h expected = {tot_est:.0f} core-hours; limits {hms(limit_h)} each = {tot_lim:.0f} core-hours of limits "
          f"(pilot {gate['s_per_evaluation']:.2f} s/evaluation x {gate['expected_evaluations_full']} evaluations)")
    per_chunk = max(1, int(math.floor(MAX_SUBMISSION_LIMIT_CORE_H / limit_h)))
    chunks = [specs[i:i + per_chunk] for i in range(0, len(specs), per_chunk)]
    plan = []
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    for j, ch in enumerate(chunks, 1):
        name = f"a19_sj_{j}"
        labels = " ".join(f"{t}:{al:g}" for t, al in ch)
        text = f"""#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --partition=serc
#SBATCH --time={hms(limit_h)}
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=0-{len(ch) - 1}%{a.throttle}
#SBATCH --output={JOB_DIR}/{name}_%A_%a.out
source {JOB_DIR}/header.sh
SPECS=({labels}); SPEC=${{SPECS[$SLURM_ARRAY_TASK_ID]}}; T=${{SPEC%%:*}}; AL=${{SPEC##*:}}
RID={ts}_a19_sj_${{T}}_a${{AL}}
$LEMCVT phase8 slope-threshold --tile $T --alpha $AL --run-id $RID; rc=$?
[ -d "$LEMCVT_RESULTS/$RID" ] && sacct -j $SLURM_JOB_ID -P --format=JobID,JobName,Partition,State,Elapsed,TotalCPU,MaxRSS,AllocCPUS,ReqMem,Timelimit > "$LEMCVT_RESULTS/$RID/slurm_sacct.txt" 2>/dev/null
echo "exit=$rc end=$(date -Is)"; exit $rc
"""
        plan.append({"name": name, "specs": [f"{t}:{al:g}" for t, al in ch], "limit": hms(limit_h), "limits_core_h": limit_h * len(ch), "expected_h": est_h * len(ch), "text": text})
        print(f"  {name}: {len(ch)} tasks, limit {hms(limit_h)}, {limit_h * len(ch):.0f} core-h of limits, expected {est_h * len(ch):.0f}")
        assert limit_h * len(ch) <= MAX_SUBMISSION_LIMIT_CORE_H + 1e-9
    rec = {"estimate": {"s_per_evaluation": gate["s_per_evaluation"], "expected_h_per_run": est_h, "limit_h": limit_h, "n_runs": len(specs), "expected_core_h": tot_est, "limits_core_h": tot_lim},
           "alphas": alphas, "plan": [{k: v for k, v in x.items() if k != "text"} for x in plan], "run_id_stamp": ts}
    if a.dry_run:
        dump(rec, out_path("sj", "submit_dry.json"))
        return
    jobs = {}
    for x in plan:
        path = os.path.join(JOB_DIR, x["name"] + ".sbatch")
        with open(path, "w") as f:
            f.write(x["text"])
        out = subprocess.check_output(["sbatch", "--parsable", path], text=True).strip()
        jobs[x["name"]] = out.split(";")[0]
        print("submitted", x["name"], out)
        with open(os.path.join(JOB_DIR, "JOBS.txt"), "a") as f:
            f.write(f"{out.split(';')[0]} {x['name']}\n")
    rec["jobs"] = jobs
    dump(rec, out_path("sj", "submit.json"))


if __name__ == "__main__":
    main()
