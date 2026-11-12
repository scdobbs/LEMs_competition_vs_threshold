"""Appendix tables for the addendum 7 report (C1-C7): the slope-area units check, the slope-area fits after the fix,
the fig3 deltas, the curvature by hillslope class, the class-restricted refits, the rule per tile and site, and the
Slurm accounting.

Usage: PYTHONNOUSERSITE=1 LEMCVT_RESULTS=... LEMCVT_DATA=... python reports/v2/scripts/make_d_diagnostic_tables.py > appendix.md
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(os.environ.get("LEMCVT_RESULTS", "/home/groups/hilley/lemcvt_results"))
repo = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo / "src"))
TILES = ["AP1", "AP2", "AP3", "AP4", "AP5", "GM1", "GM2", "GM3", "GM4", "GM5"]
PRE_FIX_SLOPE_AREA = "20261112T175641Z_slope_area_10tiles"
out: list[str] = []


def fmt(v, f=".4g"):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    try:
        return format(v, f)
    except (TypeError, ValueError):
        return str(v)


def ivl(v, f=".0f"):
    if not v:
        return "-"
    try:
        return "; ".join(f"{fmt(a, f)}-{fmt(b, f)}" for a, b in v)
    except (TypeError, ValueError):
        return str(v)


def load(d: Path):
    try:
        return json.loads((d / "params.json").read_text()), json.loads((d / "run.json").read_text())
    except Exception:  # noqa: BLE001
        return None, None


def runs(step: str, pred=None):
    ds = []
    for d in sorted(root.iterdir()):
        if f"_{step}_" not in d.name or not (d / "run.json").exists():
            continue
        p, r = load(d)
        if p is None or r.get("step") != step or (r.get("config") or {}).get("crop"):
            continue
        if pred and not pred(d.name, p, r):
            continue
        ds.append((d, p, r))
    return ds


def latest(step: str, pred=None):
    ds = runs(step, pred)
    return ds[-1] if ds else (None, None, None)


# ======================================================================================= C1 slope-area units check
out.append("## C1. Section 2b.1-2b.3: the Phase 6 joint grid as the figure read it (old path) and as written (raw)\n")
d, p, r = latest("slope_area_check")
if p:
    out.append(f"Run `{d.name}` (job {r.get('slurm_job_id')}, git {r['git_hash'][:8]}). Old path = raw grid x the tile's stored-to-working factor (0.3048 on AP), what `_model_filled_grid` did before the fix.\n")
    out.append("| tile | run grids (unit) | factor | relief obs (m) | relief raw / obs | relief old path / obs | mean obs / raw / old (m) | MSE raw vs obs | joint MSE in run | MSE old path vs obs | MSE new path vs obs | channel bins | median S_raw / law | median S_old / law | raw fit k_s, theta | old-path fit k_s, theta |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for t in TILES:
        if t not in p["tiles"]:
            continue
        q = p["tiles"][t]
        cb = q["channel_bins"]
        out.append(f"| {t} | `{q['run'][:16]}...` ({q['run_z_units']}) | {q['z_factor_stored_to_working']:.4f} | {q['relief_obs']:.2f} | {q['relief_ratio_raw_over_obs']:.4f} | {q['relief_ratio_old_over_obs']:.4f} | "
                   f"{q['mean_obs']:.1f} / {q['mean_raw']:.1f} / {q['mean_old_path']:.1f} | {q['mse_raw_vs_obs']:.3f} | {q['joint_mse_in_run']:.3f} | {q['mse_old_path_vs_obs']:.5g} | {q['mse_new_path_vs_obs']:.3f} | "
                   f"{cb['n_bins']} | {cb['median_ratio_raw_over_law']:.3f} | {cb['median_ratio_old_over_law']:.3f} | {cb['raw_fit']['Ks']:.3g}, {cb['raw_fit']['theta']:.3f} | {cb['old_path_fit']['Ks']:.3g}, {cb['old_path_fit']['theta']:.3f} |")
    try:
        df = pd.read_parquet(d / "channel_bins.parquet")
        out.append("\nChannel bins above A_crit (raw model grid), binned median slope against (U/K) A^-m at the bin centre:\n")
        out.append("| tile | bin A (m^2) | S binned raw | S binned old path | (U/K) A^-m | raw / law | old / law |")
        out.append("|---|---|---|---|---|---|---|")
        for _, x in df.iterrows():
            out.append(f"| {x.tile} | {x.bin_A_m2:.3g} | {x.binned_S_raw:.4f} | {fmt(x.binned_S_old_path, '.4f')} | {x.law_U_over_K_A_minus_m:.4f} | {x.ratio_raw_over_law:.3f} | {fmt(x.ratio_old_over_law, '.3f')} |")
    except Exception as exc:  # noqa: BLE001
        out.append(f"(bins table not readable: {exc})")
else:
    out.append("pending")
out.append("")

# ======================================================================================= C2 slope-area fits after the fix
out.append("## C2. Section 2b.5: slope-area fits at the Phase 6 optima after the fix, against the pre-fix run\n")
d, p, r = latest("slope_area", lambda n, p, r: isinstance(next(iter(p["tiles"].values()))["model_source"], dict) and n != PRE_FIX_SLOPE_AREA)
pre = root / PRE_FIX_SLOPE_AREA
pp, _ = load(pre) if pre.exists() else (None, None)
if p:
    out.append(f"Run `{d.name}` (job {r.get('slurm_job_id')}); pre-fix run `{PRE_FIX_SLOPE_AREA}` (the Phase 6 report's figure).\n")
    out.append("| tile | series | k_s | theta | R^2 | points | pre-fix k_s | pre-fix theta | k_s ratio new / pre-fix |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for t in TILES:
        if t not in p["tiles"]:
            continue
        for key in ("orig", "AD_no_opt", "AD_opt", "part"):
            q = p["tiles"][t].get(key)
            q0 = (pp or {}).get("tiles", {}).get(t, {}).get(key) if pp else None
            if q:
                out.append(f"| {t} | {key} | {q['Ks']:.4g} | {q['theta']:.3f} | {q['R2']:.3f} | {q['n_points']} | {fmt(q0['Ks'] if q0 else None, '.4g')} | {fmt(q0['theta'] if q0 else None, '.3f')} | "
                           f"{fmt(q['Ks'] / q0['Ks'] if q0 else None, '.3f')} |")
else:
    out.append("pending")
out.append("")

# ======================================================================================= C3 fig3 deltas
out.append("## C3. Section 1.3: fig3_deltas from the per-stage Phase 6 runs\n")
try:
    from lemcvt.figures.fig3_deltas import collect_deltas

    df = collect_deltas()
    out.append("| tile | stage | D0 end | K0 (1/yr) | D0 (m^2/yr) | K_fit | D_fit | dK = log10(K/K0) | dD = log10(D/D0) | MSE | run |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for _, x in df.sort_values(["stage", "tile", "D0_end"]).iterrows():
        out.append(f"| {x.tile} | {x.stage} | {x.D0_end} | {x.K0:.3e} | {x.D0:.3e} | {x.K:.3e} | {x.D:.3e} | {x.dK:+.3f} | {x.dD:+.3f} | {x.mse:.4g} | `{x.run}` |")
except Exception as exc:  # noqa: BLE001
    out.append(f"pending ({exc})")
out.append("")

# ======================================================================================= C4-C6 the D diagnostic
dd = {}
for d, p, r in runs("d_diagnostic"):
    dd[p["tile"]] = (d, p, r)
out.append("## C4. Section 2.1: observed curvature by hillslope class at the Phase 6 joint optimum\n")
if dd:
    out.append("Classes on the model hillslope cells (A < A_crit, boundary and channel cells excluded): divergent A < A_roll, hollow A_roll <= A < A_crit, hilltop = the option (iii) set (subset of divergent). "
               "-lap3: observed -laplacian at the 3-cell window (the model's 5-point stencil); -lap11: the quadratic-surface estimator at L_f = 11 m. Concave = -lap3 < 0.\n")
    out.append("| tile | run (job) | A_roll (m^2) | A_crit | hillslope cells | channel cells | class | n | share | mean -lap3 (1/m) | median -lap3 | concave | mean -lap11 | median -lap11 | concave (11 m) | SSR share | class MSE (m^2) |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for t in TILES:
        if t not in dd:
            continue
        d, p, r = dd[t]
        c = p["curvature_by_class"]
        s = p["setup"]
        first = True
        for name in ("divergent", "hollow", "hilltop"):
            q = c["classes"][name]
            head = f"| {t} | `{d.name}` ({r.get('slurm_job_id')}) | {s['A_roll']:.3g} | {s['joint_optimum']['A_crit']:.0f} | {c['all_hillslope']['n']} | {c['n_channel']} |" if first else "| | | | | | |"
            first = False
            out.append(f"{head} {name} | {q['n']} | {100 * q['share_of_hillslope']:.1f} % | {q['lap3']['mean']:.4g} | {q['lap3']['median']:.4g} | {100 * q['lap3']['frac_concave']:.1f} % | "
                       f"{q['lap11']['mean']:.4g} | {q['lap11']['median']:.4g} | {100 * q['lap11']['frac_concave']:.1f} % | {100 * q['ssr_share_of_hillslope']:.1f} % | {q['mse']:.4g} |")
        a = c["all_hillslope"]
        out.append(f"| | | | | | | all hillslope | {a['n']} | 100 % | {a['lap3']['mean']:.4g} | {a['lap3']['median']:.4g} | {100 * a['lap3']['frac_concave']:.1f} % | {a['lap11']['mean']:.4g} | {a['lap11']['median']:.4g} | "
                   f"{100 * a['lap11']['frac_concave']:.1f} % | 100 % | {a['mse']:.4g} |")
    out.append("\nAll hillslope cells against the fitted convexity U/D_fit:\n")
    out.append("| tile | U/D_fit (1/m) | mean -lap3 | mean / (U/D_fit) | median / (U/D_fit) | mean -lap11 / (U/D_fit) | model -lap3 on hillslope cells (median) | channel MSE | hillslope MSE | MSE all (check) |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for t in TILES:
        if t not in dd:
            continue
        d, p, r = dd[t]
        c = p["curvature_by_class"]
        a = c["all_hillslope"]
        out.append(f"| {t} | {a['U_over_D_fit']:.4g} | {a['lap3']['mean']:.4g} | {a['mean_lap3_over_U_over_D_fit']:.3f} | {a['median_lap3_over_U_over_D_fit']:.3f} | {a['mean_lap11_over_U_over_D_fit']:.3f} | "
                   f"{a['model_lap3']['median']:.4g} | {c['channel']['mse']:.4g} | {a['mse']:.4g} | {c['mse_all_check']:.4g} |")
else:
    out.append("pending")
out.append("")

out.append("## C5. Section 2.2: the linear joint profile with the misfit restricted to each class (plus the channel cells of each A_crit)\n")
if dd:
    out.append("| tile | mask | cells in the misfit at the optimum | evaluations | A_crit (m^2) | K (1/yr) | D (m^2/yr) | MSE over the mask (m^2) | box | 1 % interval | K / K0 | D / D0_A | D / D0_B | Phase 6 joint (A_crit, K, D, MSE) |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for t in TILES:
        if t not in dd:
            continue
        d, p, r = dd[t]
        rf = p["refits"]
        ph = rf["all_vs_phase6"]["phase6"]
        for mk in ("all", "divergent", "hollow"):
            q = rf[mk]
            jo = q["joint_optimum"]
            out.append(f"| {t} | {mk} | {jo['n_misfit']} | {q['n_evaluations']} | {jo['A_crit']:.1f} | {jo['K']:.4e} | {jo['D']:.4e} | {jo['mse']:.5g} | {jo['where']} | {ivl(q['joint_identifiability']['level_0.01']['intervals'])} | "
                       f"{q['K_over_K0']:.3f} | {q['D_over_D0_A']:.2f} | {fmt(q['D_over_D0_B'], '.2f')} | {ph['A_crit']:.1f}, {ph['K']:.3e}, {ph['D']:.3e}, {ph['mse']:.4g} |")
    out.append("\nThe 'all' mask against the Phase 6 run (same reductions, coarser grid: thin 2000 + refinement, no 5 % fill): relative differences of A_crit, K, D, MSE:\n")
    out.append("| tile | A_crit | K | D | MSE |")
    out.append("|---|---|---|---|---|")
    for t in TILES:
        if t in dd:
            v = dd[t][1]["refits"]["all_vs_phase6"]
            out.append(f"| {t} | {v['A_crit_rel_diff']:+.2%} | {v['K_rel_diff']:+.2%} | {v['D_rel_diff']:+.2%} | {v['mse_rel_diff']:+.2%} |")
else:
    out.append("pending")
out.append("")

out.append("## C6. Section 2.3: the rule, per tile and per site\n")
if dd:
    out.append("(a) mean observed -lap3 over all hillslope cells within +-30 % of U/D_fit; (b) hollow mean -lap3 below half the divergent mean; (c) D from the divergent-only refit at least a factor 2 closer to D0_A than D_fit, in log ratio (|log10(D_div/D0_A)| <= |log10(D_fit/D0_A)| - log10 2); strict reading (log distance halved) beside it. Site: supported if all three hold on >= 4 of 5 tiles.\n")
    out.append("| tile | (a) mean -lap3 / (U/D_fit) | (a) | (b) hollow / divergent | (b) | D_fit | D_div | D_hollow | D0_A | D0_B | gap_fit (dec) | gap_div (dec) | reduction (dec) | (c) | (c) strict | all three |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    site_pass = {"AP": [], "GM": []}
    for t in TILES:
        if t not in dd:
            continue
        v = dd[t][1]["verdict"]
        site_pass[t[:2]].append((v["a_pass"], v["b_pass"], v["c_pass"], v["all_pass"]))
        out.append(f"| {t} | {v['a_mean_lap3_over_U_over_D_fit']:.3f} | {v['a_pass']} | {v['b_hollow_mean_over_divergent_mean']:.3f} | {v['b_pass']} | {v['D_fit']:.3e} | {v['D_divergent']:.3e} | {v['D_hollow']:.3e} | "
                   f"{v['D0_A']:.3e} | {fmt(v['D0_B'], '.3e')} | {v['c_log10_gap_fit']:.3f} | {v['c_log10_gap_divergent']:.3f} | {v['c_gap_reduction_log10']:+.3f} | {v['c_pass']} | {v['c_pass_strict_half_log_distance']} | **{v['all_pass']}** |")
    out.append("\n| site | tiles run | (a) passes | (b) passes | (c) passes | all three | supported (>= 4 of 5) |")
    out.append("|---|---|---|---|---|---|---|")
    for site, L in site_pass.items():
        if L:
            n = len(L)
            out.append(f"| {site} | {n} | {sum(x[0] for x in L)} | {sum(x[1] for x in L)} | {sum(x[2] for x in L)} | {sum(x[3] for x in L)} | **{sum(x[3] for x in L) >= 4}** |")
else:
    out.append("pending")
out.append("")

# ======================================================================================= C7 Slurm accounting
out.append("## C7. Slurm accounting of the addendum 7 jobs\n")
try:
    txt = subprocess.run(["sacct", "-S", "2026-11-12T11:40", "-P", "-X", "-n", "-o", "JobID,JobName,State,Elapsed,Timelimit,AllocCPUS,ReqMem"], capture_output=True, text=True, timeout=60).stdout
    bt = subprocess.run(["sacct", "-S", "2026-11-12T11:40", "-P", "-n", "-o", "JobID,JobName,MaxRSS"], capture_output=True, text=True, timeout=60).stdout

    def hrs(s):
        dd_ = 0
        if "-" in s:
            dd_, s = s.split("-")
        h, m, sec = (int(x) for x in s.split(":"))
        return int(dd_) * 24 + h + m / 60 + sec / 3600

    rss = {}
    for ln in bt.splitlines():
        jid, _, mr = (ln.split("|") + ["", ""])[:3]
        if jid.endswith(".batch") and mr:
            u = mr[-1]
            rss[jid[:-6]] = float(mr[:-1]) * {"K": 1 / 1048576, "M": 1 / 1024, "G": 1}.get(u, 1)
    out.append("| job | name | state | elapsed | limit | CPUs | mem | max RSS (GB) | core-h used |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    tot = 0.0
    for ln in txt.splitlines():
        jid, name, state, el, tl, cpus, mem = ln.split("|")[:7]
        if not (name.startswith("p7_") or name.startswith("lemcvt_a7")):
            continue
        used = hrs(el) * (int(cpus) if cpus else 1)
        tot += used
        out.append(f"| {jid} | {name} | {state} | {el} | {tl} | {cpus} | {mem} | {fmt(rss.get(jid), '.1f')} | {used:.1f} |")
    out.append(f"| total | | | | | | | | {tot:.1f} |")
except Exception as exc:  # noqa: BLE001
    out.append(f"(sacct not available here: {exc})")
out.append("")
print("\n".join(out))
