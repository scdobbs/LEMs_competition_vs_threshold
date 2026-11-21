#!/usr/bin/env python3
"""Addendum 20, section 5.1 (decision 6c): SJ (slope-dependent threshold, C free) on the 11 TTV primary tiles at alpha = 0, 1.65, 2 (33 runs;
b_hat in a second submission only if it differs from all three by more than 0.1), through `lemcvt phase8 slope-threshold` (the module
unchanged: thin 2000, refine <= 400, its alternating-rank hold-out).

Estimate per tile: seconds per evaluation from that tile's addendum 18 partitioned run (wall / evaluations; the same reduction per
evaluation), times 2,400 evaluations (the addendum 19 gate-3 pilot on TTV2 gave 6.09 s against addendum 18's 6.06 s). Addendum 18 shows
the TTV tiles differ by up to 25 % per evaluation (TTV11 7.6 s), so one uniform limit would either break the 2 x estimate rule or risk a
timeout; one 3-task array per tile, each with a limit of twice that tile's estimate (30 min floor). serc, 1 CPU, 8 GB (addendum 18 peak
2.6 GB), no --account. Run ids `<ts>_a20_sj_<tile>_a<alpha>`.
usage: a20_submit_sj.py [--alphas 0,1.65,2] [--tag main] [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a20_common import A18_SUMMARY, JOB_DIR, RESULTS_ROOT, TTV_PRIMARY, dump, hms, load_json, out_path  # noqa: E402

N_EVAL_FULL = 2400
MAX_SUBMISSION = 300.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--alphas", default="0,1.65,2")
    p.add_argument("--tag", default="main")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    alphas = [float(x) for x in a.alphas.split(",")]
    fits = load_json(os.path.join(A18_SUMMARY, "per_tile_fits.json"))
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    plan, tot_est, tot_lim = [], 0.0, 0.0
    for t in TTV_PRIMARY:
        pj = load_json(os.path.join(RESULTS_ROOT, fits[f"{t}:primary"]["run_part"], "params.json"))
        s_eval = float(pj["wall_s"]) / float(pj["result"]["n_evaluations"])
        est_h = s_eval * N_EVAL_FULL / 3600.0
        lim_h = max(2.0 * est_h, 0.5)
        n = len(alphas)
        tot_est += est_h * n
        tot_lim += lim_h * n
        assert lim_h * n <= MAX_SUBMISSION
        labels = " ".join(f"{al:g}" for al in alphas)
        name = f"a20_sj_{t}"
        text = f"""#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --partition=serc
#SBATCH --time={hms(lim_h)}
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=0-{n - 1}%{n}
#SBATCH --output={JOB_DIR}/{name}_%A_%a.out
source {JOB_DIR}/header.sh
AL=({labels}); AL=${{AL[$SLURM_ARRAY_TASK_ID]}}
RID={ts}_a20_sj_{t}_a${{AL}}
$LEMCVT phase8 slope-threshold --tile {t} --alpha $AL --run-id $RID; rc=$?
[ -d "$LEMCVT_RESULTS/$RID" ] && sacct -j ${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}} -P --format=JobID,JobName,Partition,State,Elapsed,TotalCPU,MaxRSS,AllocCPUS,ReqMem,Timelimit > "$LEMCVT_RESULTS/$RID/slurm_sacct.txt" 2>/dev/null
echo "exit=$rc end=$(date -Is)"; exit $rc
"""
        plan.append({"tile": t, "name": name, "s_per_eval_a18": s_eval, "expected_h": est_h, "limit": hms(lim_h), "limit_h": lim_h, "n_tasks": n, "text": text})
        print(f"  {t}: {s_eval:.2f} s/evaluation (a18) -> {est_h:.2f} h per run, limit {hms(lim_h)}; {n} tasks = {lim_h * n:.1f} core-h of limits")
    print(f"SJ ({a.tag}): {len(TTV_PRIMARY) * len(alphas)} runs at alpha {alphas}: expected {tot_est:.0f} core-hours, limits {tot_lim:.0f} core-hours over {len(plan)} submissions (largest {max(x['limit_h'] * x['n_tasks'] for x in plan):.1f})")
    rec = {"alphas": alphas, "tag": a.tag, "run_id_stamp": ts, "n_eval_full": N_EVAL_FULL, "expected_core_h": tot_est, "limits_core_h": tot_lim,
           "plan": [{k: v for k, v in x.items() if k != "text"} for x in plan]}
    if a.dry_run:
        dump(rec, out_path("sj", f"submit_{a.tag}_dry.json"))
        return
    jobs = {}
    for x in plan:
        path = os.path.join(JOB_DIR, x["name"] + ".sbatch")
        with open(path, "w") as f:
            f.write(x["text"])
        jid = subprocess.check_output(["sbatch", "--parsable", path], text=True).strip().split(";")[0]
        jobs[x["tile"]] = jid
        with open(os.path.join(JOB_DIR, "JOBS.txt"), "a") as f:
            f.write(f"{jid} {x['name']}\n")
        print("submitted", x["name"], jid)
    rec["jobs"] = jobs
    dump(rec, out_path("sj", f"submit_{a.tag}.json"))


if __name__ == "__main__":
    main()
