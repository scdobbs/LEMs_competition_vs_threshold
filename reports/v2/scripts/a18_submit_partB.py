#!/usr/bin/env python3
"""Addendum 18, Part B submissions: one Slurm array per (stage, site, chunk) over the fit specs, from <run>/estimate/estimate.json.

Rules (addendum 18, 0.4; addendum 2, 4): every job on serc, 1 CPU, no --account; every array throttled so that the concurrent CPUs of
all arrays stay well under 300; no submission over 300 core-hours of CPU-weighted limits (arrays are chunked by that); each task's limit
is twice its estimate (30 min floor); the estimate is printed first. The arrays depend (afterok) on the gate-check job when one is given.
Priority order of section 7: primary, then mt, trunk, continuity (ncalm, pamap, ap1_3dep, v2m), then off-mesa. The gate specs
(GM1:v2m, AP2:pamap) are not resubmitted.
usage: a18_submit_partB.py [--dependency afterok:<jobid>] [--dry-run] [--max-arrays 12]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a18_common import JOB_DIR, MAX_CONCURRENT_CPUS, MAX_SUBMISSION_LIMIT_CORE_HOURS, run_dir  # noqa: E402

ORDER = ["primary", "mt", "trunk", "ncalm", "pamap", "ap1_3dep", "v2m", "offmesa"]
GATE = {"GM1:v2m", "AP2:pamap"}
MEM = {"part": "8G", "ad": "8G"}


def sbatch_text(name: str, stage: str, specs: list[str], limit: str, throttle: int, dependency: str | None, rd: str, run_id: str) -> str:
    dep = f"#SBATCH --dependency={dependency}\n" if dependency else ""
    labels = " ".join(specs)
    return f"""#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --partition=serc
#SBATCH --time={limit}
#SBATCH --cpus-per-task=1
#SBATCH --mem={MEM[stage]}
#SBATCH --array=0-{len(specs) - 1}%{throttle}
#SBATCH --output={JOB_DIR}/{name}_%A_%a.out
{dep}source {JOB_DIR}/header.sh
SPECS=({labels}); SPEC=${{SPECS[$SLURM_ARRAY_TASK_ID]}}
LOG={JOB_DIR}/{name}.$SLURM_JOB_ID.$SLURM_ARRAY_TASK_ID.log
$PY reports/v2/scripts/a18_fits.py --spec $SPEC --stage {stage} 2>&1 | tee $LOG
rc=${{PIPESTATUS[0]}}
RUNDIR=$(grep -oP '^results: \\K.*' $LOG | tail -1); rm -f $LOG
[ -n "$RUNDIR" ] && [ -d "$RUNDIR" ] && sacct -j $SLURM_JOB_ID -P --format=JobID,JobName,Partition,State,Elapsed,TotalCPU,MaxRSS,AllocCPUS,ReqMem,Timelimit > "$RUNDIR/slurm_sacct.txt" 2>/dev/null
echo "exit=$rc end=$(date -Is)"; exit $rc
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dependency", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-arrays", type=int, default=12, help="the throttle is MAX_CONCURRENT_CPUS * 0.9 / this")
    p.add_argument("--groups", default=",".join(ORDER))
    p.add_argument("--throttle", type=int, default=None, help="per-array throttle override (the sum of the arrays' widths must stay under 300 CPUs)")
    a = p.parse_args()
    rd = run_dir()
    run_id = os.path.basename(rd)
    est = json.load(open(os.path.join(rd, "estimate", "estimate.json")))
    rows = {r["spec"]: r for r in est["specs"]}
    groups = a.groups.split(",")
    plan = []
    for stage in ("part", "ad"):
        for g in groups:
            for site in ("GM", "AP", "TTV"):
                specs = [k for k, r in rows.items() if r["group"] == g and r["site"] == site and k not in GATE]
                if not specs:
                    continue
                lim_h = max(rows[k][f"{stage}_limit_h"] for k in specs)
                lim = max(rows[k][f"{stage}_limit"] for k in specs)
                per_chunk = max(1, int(math.floor(MAX_SUBMISSION_LIMIT_CORE_HOURS / lim_h)))
                chunks = [specs[i:i + per_chunk] for i in range(0, len(specs), per_chunk)]
                for j, ch in enumerate(chunks):
                    plan.append({"name": f"a18_{stage}_{g}_{site}" + (f"_{j + 1}" if len(chunks) > 1 else ""), "stage": stage, "group": g, "site": site, "specs": ch,
                                 "limit": lim, "limit_h": lim_h, "limits_core_h": lim_h * len(ch), "expected_h": sum(rows[k][f"{stage}_h"] for k in ch)})
    throttle = a.throttle or max(1, int(MAX_CONCURRENT_CPUS * 0.9 / max(len(plan), a.max_arrays)))
    concurrent = sum(min(len(x["specs"]), throttle) for x in plan)
    assert concurrent <= MAX_CONCURRENT_CPUS * 0.9, concurrent
    tot_lim = sum(x["limits_core_h"] for x in plan)
    tot_exp = sum(x["expected_h"] for x in plan)
    print(f"{len(plan)} arrays, throttle %{throttle} each (<= {concurrent} concurrent CPUs), limits {tot_lim:.0f} core-hours in all (<= 300 per submission), expected {tot_exp:.0f}")
    for x in plan:
        print(f"  {x['name']:32s} tasks {len(x['specs']):3d} limit {x['limit']} ({x['limits_core_h']:.0f} core-h of limits, expected {x['expected_h']:.1f})")
        assert x["limits_core_h"] <= MAX_SUBMISSION_LIMIT_CORE_HOURS + 1e-9
    if a.dry_run:
        return
    os.makedirs(JOB_DIR, exist_ok=True)
    jobs = {}
    for x in plan:
        path = os.path.join(JOB_DIR, x["name"] + ".sbatch")
        with open(path, "w") as f:
            f.write(sbatch_text(x["name"], x["stage"], x["specs"], x["limit"], throttle, a.dependency, rd, run_id))
        out = subprocess.check_output(["sbatch", "--parsable", path], text=True).strip()
        jobs[x["name"]] = {"job_id": out.split(";")[0], "n_tasks": len(x["specs"]), "limit": x["limit"], "limits_core_h": x["limits_core_h"], "expected_h": x["expected_h"],
                           "specs": x["specs"], "throttle": throttle, "dependency": a.dependency}
        print("submitted", x["name"], out)
    jf = os.path.join(JOB_DIR, "JOBS_partB.json")
    old = json.load(open(jf)) if os.path.exists(jf) else {}
    old.update(jobs)
    json.dump(old, open(jf, "w"), indent=1)


if __name__ == "__main__":
    main()
