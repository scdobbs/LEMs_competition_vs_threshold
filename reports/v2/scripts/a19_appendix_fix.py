#!/usr/bin/env python3
"""Addendum 19, decision 8: fix the addendum 18 appendix C16 and write the corrected report as a new file; the original stays untouched.

The bug (make_a18_tables.py, C16 loop): `for _k, r in fits.items(): s = specs.get(k, {})` used the stale loop variable `k` from the previous
section, so the "spec" column and the "note" column repeated the last key ("AP5:pamap") on every row; the checksum table was empty because
`load(r["run_part"], "params.json")` looked under the addendum 18 run directory instead of the results root. Both tables are regenerated here
from the run records (params.json: spec, model, inputs.sha256, result.n_evaluations, timing, fill; run.json: git_hash, slurm_job_id).
usage: a19_appendix_fix.py
"""
from __future__ import annotations

import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a19_common import A18, A18_REPORT, REPO, RESULTS_ROOT, a18_specs, dump, load_json, out_path  # noqa: E402

LEMCVT_DATA = os.environ.get("LEMCVT_DATA", os.path.join(os.environ.get("GROUP_HOME", "/home/groups/hilley"), "lemcvt_data", "v1"))


def f(x, fmt=".4g"):
    if x is None:
        return "-"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, (int, np.integer)):
        return f"{x:,}"
    if isinstance(x, float):
        return format(x, fmt) if np.isfinite(x) else "-"
    return str(x)


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out + [""]


def main() -> None:
    specs = a18_specs()
    fits = load_json(os.path.join(A18, "summary", "per_tile_fits.json"))
    rows, rows_sha, checks = [], [], {"n_specs": len(fits), "n_part_runs_read": 0, "n_ad_runs_read": 0, "spec_mismatch": [], "missing": []}
    for key, r in fits.items():
        s = specs.get(key, {})
        rp = ra = rj_p = rj_a = None
        if r.get("run_part") and os.path.exists(os.path.join(RESULTS_ROOT, r["run_part"], "params.json")):
            rp = load_json(os.path.join(RESULTS_ROOT, r["run_part"], "params.json"))
            rj_p = load_json(os.path.join(RESULTS_ROOT, r["run_part"], "run.json"))
            checks["n_part_runs_read"] += 1
            if rp["spec"]["tile"] != s.get("tile") or rp["spec"]["group"] != s.get("group"):
                checks["spec_mismatch"].append(key)
        else:
            checks["missing"].append(r.get("run_part"))
        if r.get("run_ad") and os.path.exists(os.path.join(RESULTS_ROOT, r["run_ad"], "params.json")):
            ra = load_json(os.path.join(RESULTS_ROOT, r["run_ad"], "params.json"))
            rj_a = load_json(os.path.join(RESULTS_ROOT, r["run_ad"], "run.json"))
            checks["n_ad_runs_read"] += 1
        else:
            checks["missing"].append(r.get("run_ad"))
        m = rp["spec"]["m"] if rp else r["m"]
        K0 = rp["spec"]["K0"] if rp else r["K0"]
        D0A = rp["spec"].get("D0A") if rp else r.get("D0A")
        D0B = rp["spec"].get("D0B") if rp else r.get("D0B")
        trunk = rp["spec"].get("trunk") if rp else r.get("trunk")
        git_p = rj_p["git_hash"] if rj_p else r.get("git_part")
        git_a = rj_a["git_hash"] if rj_a else r.get("git_ad")
        git = git_p if git_p == git_a else f"{git_p} / {git_a}"
        job_p = rj_p["slurm_job_id"] + (f"_{rj_p['slurm_array_task_id']}" if rj_p and rj_p.get("slurm_array_task_id") not in (None, "") else "") if rj_p else r.get("job_part")
        job_a = rj_a["slurm_job_id"] + (f"_{rj_a['slurm_array_task_id']}" if rj_a and rj_a.get("slurm_array_task_id") not in (None, "") else "") if rj_a else r.get("job_ad")
        n_p = rp["result"]["n_evaluations"] if rp else r.get("n_evaluations_part")
        n_a = ra["result"]["n_evaluations"] if ra else r.get("n_evaluations_ad")
        w_p = (rp["result"]["timing"]["wall_total_s"] if rp else (r.get("timing_part") or {}).get("wall_total_s", 0)) / 3600
        w_a = (ra["result"]["timing"]["wall_total_s"] if ra else (r.get("timing_ad") or {}).get("wall_total_s", 0)) / 3600
        capped = rp["result"]["fill"].get("capped") if rp else r.get("fill", {}).get("capped")
        rows.append([key, f(m, ".2f"), f(K0, ".3e"), f(D0A), f(D0B), f(trunk), f(r.get("run_part")), f(r.get("run_ad")), f(git), f"{job_p} / {job_a}", f(n_p), f(n_a), f(w_p, ".2f"), f(w_a, ".2f"), f(capped),
                     (rp["spec"].get("note") if rp else s.get("note", ""))])
        if rp:
            sh = rp["inputs"]["sha256"]
            rows_sha.append([r["tile"], r["group"], sh["filled"][:16] + "...", sh["area"][:16] + "...", sh["fd"][:16] + "...", os.path.relpath(rp["inputs"]["grids"]["area"], LEMCVT_DATA)])
    c16 = ["### C16. Run ids, git hashes, Slurm jobs and input checksums of every fit", "",
           "*(Corrected by addendum 19, decision 8: regenerated from the run records; the original's \"spec\" and \"note\" columns repeated the last key of a previous loop, and its checksum table was empty.)*", ""]
    c16 += table(["spec", "m", "K0", "D0_A", "D0_B", "trunk", "partitioned run", "AD run", "git", "jobs part / ad", "evals part", "evals AD", "wall part (h)", "wall AD (h)", "fill capped", "note"], rows)
    c16 += table(["tile", "group", "sha256 filled (16)", "sha256 area (16)", "sha256 fd (16)", "area grid"], rows_sha)
    src = open(os.path.join(REPO, A18_REPORT), encoding="utf-8").read()
    i0 = src.index("### C16. Run ids")
    i1 = src.index("### C17.")
    header_note = ("> **Addendum 19 (decision 8), " + time.strftime("%Y-%m-%d") + ":** this file is the addendum 18 report with Appendix C16 corrected (the \"spec\" and \"note\" columns, "
                   "which repeated \"AP5:pamap\" and one string on every row, and the empty checksum table, regenerated from the run records). Nothing else is changed; the original file "
                   f"`{A18_REPORT}` is untouched.\n\n")
    fixed = src[:i0] + "\n".join(c16) + "\n" + src[i1:]
    first_nl = fixed.index("\n")
    fixed = fixed[: first_nl + 1] + "\n" + header_note + fixed[first_nl + 1:]
    stamp = time.strftime("%Y%m%dT%H%M") + "PST"
    out_name = f"reports/v2/{stamp}_expanded_fits_appendix_fix.md"
    with open(os.path.join(REPO, out_name), "w", encoding="utf-8") as fh:
        fh.write(fixed)
    # checks on the output
    spec_col = [ln.split("|")[1].strip() for ln in c16 if ln.startswith("| ") and ":" in ln.split("|")[1]]
    checks.update({"out": out_name, "n_rows_c16": len(rows), "n_rows_checksum": len(rows_sha), "unique_specs_in_column": len(set(spec_col)), "n_lines_original": src.count("\n"), "n_lines_fixed": fixed.count("\n"),
                   "original_spec_column_values": sorted(set(re.findall(r"^\| (AP5:pamap) \|", src[i0:i1], flags=re.M)))})
    dump(checks, out_path("appendix_fix", "checks.json"))
    print(checks)


if __name__ == "__main__":
    main()
