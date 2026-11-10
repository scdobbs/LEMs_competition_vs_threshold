"""Tables for reports/v2/03_performance.md from the profile, gate and optimizer runs.

Usage: PYTHONNOUSERSITE=1 LEMCVT_RESULTS=... python reports/v2/scripts/make_03_tables.py > reports/v2/03_tables.md
Reads: `*_profile_{GM1,AP5}` (step 1), the `tile_forward` runs made with `--reference-run` (one array per
Phase 3 step; identified by the git hash recorded at run start and the `_stepN` run-id suffix for the
snapshot gates), the section 3 baseline `tile_run_<tile>_<stage>` runs and the Phase 3 optimizer runs
(job ids in 03_optimizer_regression_jobs.json / 03_optimizer_phase3_jobs.json).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

root = Path(os.environ.get("LEMCVT_RESULTS", "/home/groups/hilley/lemcvt_results"))
repo = Path(__file__).resolve().parents[3]
TILES = ["AP1", "AP2", "AP3", "AP4", "AP5", "GM1", "GM2", "GM3", "GM4", "GM5"]
KEYS = ["ad_no_opt", "ad_opt", "acrit_opt", "nonlinear_opt", "joint_opt"]
STEP_COMMITS = {"phase2": "v1 loops (Phase 2 code, 840203e)", "step2": "9d046e3", "step3": "898467c", "step4": "db11fee",
                "step5": "a224750"}


def load(d: Path):
    try:
        return json.loads((d / "params.json").read_text()), json.loads((d / "run.json").read_text())
    except Exception:  # noqa: BLE001
        return None, None


def sacct_maxrss(d: Path) -> str:
    f = d / "slurm_sacct.txt"
    if not f.exists():
        return "-"
    for line in f.read_text().splitlines():
        if ".batch" in line:
            parts = line.split("|")
            for x in parts:
                if x.endswith("K") and x[:-1].isdigit():
                    return f"{int(x[:-1]) / 1e6:.1f} GB"
    return "-"


out = []
# ---- profile
out.append("## P1. Profile (step 1): wall time per operation on a full tile, cProfile on (adds overhead to Python loops)\n")
out.append("| operation | GM1 (s) | AP5 (s) | peak RSS after (GB) | what dominates (self time) |")
out.append("|---|---|---|---|---|")
prof = {}
for t in ("GM1", "AP5"):
    ds = sorted(d for d in root.iterdir() if d.name.endswith(f"_profile_{t}") and (d / "run.json").exists())
    if ds:
        prof[t] = load(ds[-1])[0]
if prof:
    steps = list(prof["GM1"]["steps"]) if "GM1" in prof else list(next(iter(prof.values()))["steps"])
    for st in steps:
        g = prof.get("GM1", {}).get("steps", {}).get(st, {})
        a = prof.get("AP5", {}).get("steps", {}).get(st, {})
        top = g.get("top", [])[:3]
        dom = "; ".join(f"{r['function']} {r['tottime_s']:.0f} s" for r in sorted(top, key=lambda r: -r["tottime_s"])[:2])
        out.append(f"| {st} | {g.get('wall_s', float('nan')):.1f} | {a.get('wall_s', float('nan')):.1f} | {g.get('peak_rss_gb_after', float('nan')):.1f} | {dom} |")
    out.append(f"\nProfile runs: " + ", ".join(f"{t}: `{sorted(d.name for d in root.iterdir() if d.name.endswith(f'_profile_{t}'))[-1]}`" for t in prof) + ".\n")

# ---- gates
out.append("## G1. Forward gates: Phase 3 code against the Phase 2 v2 outputs (max |dz| / relief; rel. MSE diff), tolerances 1e-8 / 1e-10\n")
runs = []
for d in sorted(root.iterdir()):
    if "_tile_forward_" not in d.name or d.name < "20261109T21":
        continue
    p, r = load(d)
    if not p or not p.get("reference_run"):
        continue
    m = re.search(r"_step(\d)$", d.name)
    step = f"step{m.group(1)}" if m else None
    if step is None:
        # gates run from the main tree: the step-2 array started 21:22 UTC (job 51698701), the step-5 array
        # 21:48 UTC (job 51700412); the git hash in run.json was taken at finish for runs before a6f8c64
        step = "step2" if d.name < "20261109T2140" else "step5"
    runs.append((step, p["tile"], d, p, r))
by_step: dict[str, list] = {}
for s, t, d, p, r in runs:
    by_step.setdefault(s, []).append((t, d, p, r))
for s in sorted(by_step):
    rows = by_step[s]
    n_cmp = sum(len([f for f in p["forward_runs"] if f.get("comparison_phase2")]) for _, _, p, _ in rows)
    n_pass = sum(sum(1 for f in p["forward_runs"] if f.get("comparison_phase2") and f["comparison_phase2"]["pass_dz"] and f["comparison_phase2"]["pass_mse"]) for _, _, p, _ in rows)
    mx_dz = max((f["comparison_phase2"]["max_abs_dz_over_relief"] for _, _, p, _ in rows for f in p["forward_runs"] if f.get("comparison_phase2")), default=float("nan"))
    mx_mse = max((f["comparison_phase2"]["rel_mse_diff"] for _, _, p, _ in rows for f in p["forward_runs"] if f.get("comparison_phase2")), default=float("nan"))
    out.append(f"\n### {s} ({STEP_COMMITS.get(s, '?')}): {len(rows)} tiles, {n_pass} of {n_cmp} comparisons pass; max |dz|/relief {mx_dz:.2e}, max rel. MSE diff {mx_mse:.2e}\n")
    out.append("| tile | run | " + " | ".join(KEYS) + " | peak RSS |")
    out.append("|---|---|" + "---|" * len(KEYS) + "---|")
    for t, d, p, r in sorted(rows, key=lambda x: TILES.index(x[0])):
        cells = []
        fr = {f["key"]: f for f in p["forward_runs"]}
        for k in KEYS:
            f = fr.get(k)
            if not f or not f.get("comparison_phase2"):
                cells.append("-")
            else:
                c = f["comparison_phase2"]
                cells.append(f"{c['max_abs_dz_over_relief']:.1e} / {c['rel_mse_diff']:.1e}{'' if c['pass_dz'] and c['pass_mse'] else ' FAIL'}")
        out.append(f"| {t} | {d.name} | " + " | ".join(cells) + f" | {sacct_maxrss(d)} |")

# ---- wall times per key per step
out.append("\n## G2. Wall time per forward evaluation (s) by step; ad_no_opt includes the AD build, nonlinear_opt runs up to 50 Picard iterations\n")
out.append("| step | tile | " + " | ".join(KEYS) + " |")
out.append("|---|---|" + "---|" * len(KEYS))
# Phase 2 runs (v1 loops) from run.log timestamps are not stored as wall_s; use the Phase 2 report instead
for s in sorted(by_step):
    for t, d, p, r in sorted(by_step[s], key=lambda x: TILES.index(x[0])):
        fr = {f["key"]: f for f in p["forward_runs"]}
        out.append(f"| {s} | {t} | " + " | ".join(f"{fr[k]['wall_s']:.0f}" if k in fr and "wall_s" in fr[k] else "-" for k in KEYS) + " |")

# ---- optimizer: baseline vs Phase 3
out.append("\n## O1. Optimizer stages: section 3 baseline (Phase 2 code) and Phase 3 code, against v1 (parameters within 1 %, MSE within 0.5 %)\n")
out.append("| tile | stage | code | run | params | rel. diff vs v1 | MSE | rel. diff vs v1 | wall (s) | v1 wall (s) | fev / solves | pass |")
out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
base_jobs = json.loads((repo / "reports/v2/03_optimizer_regression_jobs.json").read_text())
p3_jobs = json.loads((repo / "reports/v2/03_optimizer_phase3_jobs.json").read_text()) if (repo / "reports/v2/03_optimizer_phase3_jobs.json").exists() else []
p3_ids = {j["job_id"] for j in p3_jobs}
opt_runs = []
for d in sorted(root.iterdir()):
    if "_tile_run_" not in d.name:
        continue
    p, r = load(d)
    if not p or not p.get("stages_results") or p.get("optimizer") != "v1" or p.get("tile") not in ("GM1", "AP5"):
        continue   # the v2 (Phase 4) runs are compared in 04_optimizer_v2.md
    if (r.get("config") or {}).get("crop"):
        continue
    job = str(r.get("slurm_job_id") or (r.get("slurm_accounting") or {}).get("job_id") or "")
    code = "Phase 3" if job.split("_")[0] in p3_ids else "baseline"
    for st, res in p["stages_results"].items():
        ref = p["v1_reference"][st]
        rel = {k: (res["params"][k] - ref[k]) / ref[k] for k in res["params"]}
        dm = (res["mse"] - ref["MSE"]) / ref["MSE"]
        ok = all(abs(v) <= 0.01 for v in rel.values()) and abs(dm) <= 0.005
        opt_runs.append((p["tile"], st, code, d.name, res, ref, rel, dm, ok))
for t, st, code, name, res, ref, rel, dm, ok in sorted(opt_runs, key=lambda x: (x[0], ["ad", "acrit", "nonlinear", "joint"].index(x[1]), x[2])):
    out.append(f"| {t} | {st} | {code} | {name} | " + ", ".join(f"{k}={v:.6g}" for k, v in res["params"].items()) + " | "
               + ", ".join(f"{k} {v:+.1e}" for k, v in rel.items()) + f" | {res['mse']:.5g} | {dm:+.1e} | {res['wall_s']:.0f} | {ref['duration_s']:.0f} | {res['n_fev']} / {res['n_solves']} | {'yes' if ok else 'NO'} |")
# ---- O2. Phase 3 against the section 3 baseline: pass rule and trajectory comparison (addendum 3, section 2)
import subprocess


def _sacct(job_ids):
    try:
        txt = subprocess.check_output(["sacct", "-X", "-n", "-P", "-j", ",".join(job_ids),
                                       "--format=JobID,State,Elapsed,Timelimit,NodeList"], text=True, timeout=60)
    except Exception:  # noqa: BLE001
        return {}
    outd = {}
    for line in txt.strip().splitlines():
        jid, state, el, tl, node = (line.split("|") + [""] * 5)[:5]
        outd[jid.split("_")[0]] = {"state": state, "elapsed": el, "timelimit": tl, "node": node}
    return outd


def _same_mask(tile, a1, a2):
    """Cells with D8 area between two A_crit values (0 means both select the same channel mask)."""
    try:
        import numpy as np
        import rasterio

        data = Path(os.environ.get("LEMCVT_DATA", "/home/groups/hilley/lemcvt_data/v1"))
        site, n = tile[:2], tile[2:]
        with rasterio.open(data / site / f"tile{n}" / f"{site}_1m_best_tile_area.tif") as f:
            A = f.read(1)
        lo, hi = sorted((a1, a2))
        return int(np.sum((A >= lo) & (A < hi)))
    except Exception:  # noqa: BLE001
        return None


def _traj(d: Path):
    import pandas as pd

    f = d / "trajectory.parquet"
    return pd.read_parquet(f) if f.exists() else None


out.append("\n## O2. Phase 3 code against the section 3 baseline (addendum 3, section 2): pass rule (parameters within 1 %, MSE within 0.5 %; A_crit rule) and row-by-row trajectory comparison\n")
out.append("| tile | stage | baseline run | Phase 3 run | baseline params | Phase 3 params | rel. diff P3 vs baseline | MSE baseline / Phase 3 | rel. MSE diff | pass | evals (b / P3) | first row with different parameters | max per-eval abs dMSE | max per-eval rel dMSE | optimizer steps differ |")
out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
pairs = {}
for t, st, code, name, res, ref, rel, dm, ok in opt_runs:
    pairs.setdefault((t, st), {})[code] = (name, res)
import numpy as np

traj_rows = []
for (t, st), d in sorted(pairs.items(), key=lambda x: (x[0][0], ["ad", "acrit", "nonlinear", "joint"].index(x[0][1]))):
    if "baseline" not in d or "Phase 3" not in d:
        missing = "Phase 3" if "baseline" in d else "baseline"
        have = d.get("baseline") or d.get("Phase 3")
        out.append(f"| {t} | {st} | {have[0] if 'baseline' in d else '-'} | {have[0] if 'Phase 3' in d else '-'} | - | - | - | - | - | {missing} stage not finished | - | - | - | - | - |")
        continue
    (nb, rb), (np3, rp3) = d["baseline"], d["Phase 3"]
    rel = {k: (rp3["params"][k] - rb["params"][k]) / rb["params"][k] for k in rb["params"]}
    dmse = (rp3["mse"] - rb["mse"]) / rb["mse"]
    ok = all(abs(v) <= 0.01 for v in rel.values()) and abs(dmse) <= 0.005
    note = ""
    if not ok and abs(dmse) <= 0.005 and "A_crit" in rel and all(abs(v) <= 0.01 for k, v in rel.items() if k != "A_crit"):
        n_between = _same_mask(t, rb["params"]["A_crit"], rp3["params"]["A_crit"])
        if n_between == 0:
            ok, note = True, " (A_crit rule: same channel mask)"
        else:
            note = f" ({n_between} cells between the two A_crit values)"
    tb, tp = _traj(root / nb), _traj(root / np3)
    first_diff = "-"
    mx_abs = mx_rel = float("nan")
    steps_differ = "-"
    if tb is not None and tp is not None:
        pcols = [c for c in tb.columns if c.startswith("log10_")]
        n = min(len(tb), len(tp))
        pb, pp = tb[pcols].to_numpy()[:n], tp[pcols].to_numpy()[:n]
        diff_rows = np.where(np.any(pb != pp, axis=1))[0]
        first_diff = int(diff_rows[0]) if diff_rows.size else ("none" if len(tb) == len(tp) else f"none in first {n}; lengths {len(tb)} / {len(tp)}")
        same = np.all(pb == pp, axis=1)
        dm_ev = np.abs(tp["mse"].to_numpy()[:n] - tb["mse"].to_numpy()[:n])
        mx_abs = float(dm_ev[same].max()) if same.any() else float("nan")
        mx_rel = float((dm_ev[same] / np.abs(tb["mse"].to_numpy()[:n][same])).max()) if same.any() else float("nan")
        steps_differ = "no" if (not diff_rows.size and len(tb) == len(tp)) else "yes"
        traj_rows.append({"tile": t, "stage": st, "baseline": nb, "phase3": np3, "n_b": len(tb), "n_p3": len(tp),
                          "first_diff_row": first_diff, "max_abs_dmse": mx_abs, "max_rel_dmse": mx_rel})
    out.append(f"| {t} | {st} | {nb} | {np3} | " + ", ".join(f"{k}={v:.6g}" for k, v in rb["params"].items()) + " | "
               + ", ".join(f"{k}={v:.6g}" for k, v in rp3["params"].items()) + " | "
               + ", ".join(f"{k} {v:+.1e}" for k, v in rel.items()) + f" | {rb['mse']:.6g} / {rp3['mse']:.6g} | {dmse:+.1e} | "
               f"{'yes' if ok else 'NO'}{note} | {rb['n_fev']} / {rp3['n_fev']} | {first_diff} | {mx_abs:.2e} | {mx_rel:.2e} | {steps_differ} |")

# ---- O3. Job status of the baseline and Phase 3 stages (sacct)
out.append("\n## O3. Slurm status of the baseline (section 3) and Phase 3 optimizer jobs\n")
out.append("| set | tile | stage | job | state | elapsed | limit | node |")
out.append("|---|---|---|---|---|---|---|---|")
acct = _sacct([j["job_id"] for j in base_jobs + p3_jobs])
for label, jobs in (("baseline", base_jobs), ("Phase 3", p3_jobs)):
    for j in jobs:
        a = acct.get(str(j["job_id"]), {})
        out.append(f"| {label} | {j['tile']} | {j['stage']} | {j['job_id']} | {a.get('state', '?')}{' (' + j['status'] + ')' if j.get('status') else ''} | {a.get('elapsed', '?')} | {a.get('timelimit', j['time'])} | {a.get('node', '?')} |")

print("\n".join(out))
