"""Appendix tables for the checkpoint report of addendum 4 (Phase 4b, Phase 5, sections 6.1-6.4).

Usage: PYTHONNOUSERSITE=1 LEMCVT_RESULTS=... LEMCVT_DATA=... python reports/v2/scripts/make_checkpoint_4b_5_tables.py > appendix.md
Reads the latest run of each step under $LEMCVT_RESULTS; sections whose runs do not exist yet are marked pending.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

root = Path(os.environ.get("LEMCVT_RESULTS", "/home/groups/hilley/lemcvt_results"))
repo = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo / "src"))
TILES = ["AP1", "AP2", "AP3", "AP4", "AP5", "GM1", "GM2", "GM3", "GM4", "GM5"]
C = 3937.0 / 1200.0
out: list[str] = []


def fmt(v, f=".4g"):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    return format(v, f)


def load(d: Path):
    try:
        return json.loads((d / "params.json").read_text()), json.loads((d / "run.json").read_text())
    except Exception:  # noqa: BLE001
        return None, None


def latest(step: str, label_filter=None):
    ds = []
    for d in sorted(root.iterdir()):
        if f"_{step}_" not in d.name or not (d / "run.json").exists():
            continue
        p, r = load(d)
        if p is None or r.get("step") != step:
            continue
        if label_filter and not label_filter(d.name, p, r):
            continue
        ds.append((d, p, r))
    return ds[-1] if ds else (None, None, None)


def v2_runs(stage: str, units: str | None = None, pred=None):
    """Latest v2 tile_run per tile holding `stage`, optionally restricted to a working z unit."""
    res = {}
    for d in sorted(root.iterdir()):
        if "_tile_run_" not in d.name or not (d / "params.json").exists():
            continue
        p, r = load(d)
        if not p or p.get("optimizer") != "v2" or stage not in p.get("v2", {}) or (r.get("config") or {}).get("crop"):
            continue
        if units and p.get("inputs", {}).get("z_units_working") != units:
            continue
        if pred and not pred(p):
            continue
        res[p["tile"]] = (d, p, r)
    return res


# ======================================================================================= A1 nonlinear rows
out.append("## A1. Nonlinear rows: GM v2 profiles (decision 2) and the AP5 baseline (03b, pending row)\n")
out.append("| tile | run (job) | z unit | grid + refine | evaluations hitting the Picard cap | v2 optimum A_crit (Picard it.) | MSE | v1 A_crit same code: MSE (it.) | v2 - v1 (rel.) | 1 % interval (m^2) | hillslope > 0.5 / > 0.9 S_c at the optimum | wall (h) |")
out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
nl = v2_runs("nonlinear")
for t in TILES:
    if t not in nl:
        continue
    d, p, r = nl[t]
    v = p["v2"]["nonlinear"]
    o = v["optimum"]
    c = v["certificate"].get("v1_optimum", {})
    i1 = v["identifiability"]["level_0.01"]
    rel = (o["mse"] - c["mse"]) / c["mse"] if c else float("nan")
    fr = f"{fmt(o.get('frac_hillslope_gt_0.5Sc'), '.3f')} / {fmt(o.get('frac_hillslope_gt_0.9Sc'), '.4f')}"
    out.append(f"| {t} | `{d.name}` ({r.get('slurm_job_id')}) | {p['inputs'].get('z_units_working')} | {v['n_grid']} + {v['n_refine']} | {v['n_hit_cap']} of {v['n_evaluations']} | "
               f"{o['A_crit']:.6g} ({o['picard_iterations']}) | {o['mse']:.6g} | {fmt(c.get('A_crit'), '.6g')}: {fmt(c.get('mse'), '.6g')} ({c.get('picard_iterations')}) | {rel:+.2e} | "
               f"{fmt(i1.get('A_crit_min'), '.4g')} - {fmt(i1.get('A_crit_max'), '.4g')} ({i1['n_grid_points']} pts) | {fr} | {v['wall_s'] / 3600:.1f} |")
# AP5 baseline (v1 optimizer, Phase 2 code) from the 03 jobs
base = None
for d in sorted(root.iterdir()):
    if "_tile_run_AP5" in d.name and (d / "run.json").exists():
        p, r = load(d)
        if p and p.get("optimizer") == "v1" and str(r.get("slurm_job_id")) == "51697696":
            base = (d, p, r)
if base:
    d, p, r = base
    s = p["stages_results"]["nonlinear"]
    ref = p["v1_reference"]["nonlinear"]
    fr_ = [f for f in p["forward_runs"] if f["key"] == "nonlinear_opt"]
    pic = fr_[0].get("picard_iterations") if fr_ else None
    out.append(f"\nAP5 nonlinear baseline (v1 optimizer, job 51697696, run `{d.name}`): A_crit {s['params']['A_crit']:.6g} against v1 {ref['A_crit']:.6g} "
               f"({(s['params']['A_crit'] - ref['A_crit']) / ref['A_crit']:+.1e}), MSE {s['mse']:.6g} against {ref['MSE']:.6g} ({(s['mse'] - ref['MSE']) / ref['MSE']:+.1e}), "
               f"{s['n_fev']} evaluations / {s['n_solves']} solves, {s['wall_s'] / 3600:.1f} h, Picard iterations at the optimum {pic} (the cap), message '{s['message']}'.")
out.append("")

# ======================================================================================= A2 Phase 4b
out.append("## A2. Phase 4b: D8 against D-infinity at the v1 `d8_vs_dinf` parameters (AP in feet), rebuilt and v1 partitioned D-infinity models\n")
d, p, r = latest("d8_vs_dinf", lambda n, p, r: p.get("pt_dinf_model") == "both")
if p:
    out.append(f"Run `{d.name}` (job {r.get('slurm_job_id')}, git {r['git_hash'][:8]}, units `{p.get('units')}`).\n")
    out.append("| tile | z unit | MSE AD D8 | MSE AD D-inf | MSE PT D8 | MSE PT D-inf rebuilt | MSE PT D-inf v1 | v1 printed (AD D8 / AD D-inf / PT D8 / PT D-inf) | channel cells D8 / D-inf rebuilt / D-inf v1 | channel RMS error D8 / rebuilt / v1 |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for t, res in p["tiles"].items():
        v1p = res.get("v1_printed") or {}
        out.append(f"| {t} | {res.get('z_units_working')} | {res['mse_ad_d8']:.5g} | {res['mse_ad_dinf']:.5g} | {res['mse_pt_d8']:.5g} | {fmt(res.get('mse_pt_dinf'), '.5g')} | {fmt(res.get('mse_pt_dinf_v1'), '.5g')} | "
                   f"{fmt(v1p.get('mse_ad_d8'), '.5g')} / {fmt(v1p.get('mse_ad_dinf'), '.5g')} / {fmt(v1p.get('mse_pt_d8'), '.5g')} / {fmt(v1p.get('mse_pt_dinf_v1'), '.5g')} | "
                   f"{res['n_channel_d8']} / {res.get('n_channel_dinf', '-')} / {res.get('n_channel_dinf_v1', '-')} | "
                   f"{fmt(res.get('pt_d8_channel_rms_error'), '.4g')} / {fmt(res.get('pt_dinf_channel_rms_error'), '.4g')} / {fmt(res.get('pt_dinf_v1_channel_rms_error'), '.4g')} |")
    out.append("\n### A2b. Rebuilt model: system composition and channel monotonicity (test 4)\n")
    out.append("| tile | boundary rows | channel rows (both / cardinal-only / diagonal-only receivers) | channel cells without a usable receiver (hillslope row) | hillslope rows | channel cells lower than every nonzero-weight receiver | lower than any | min / median drop to receiver |")
    out.append("|---|---|---|---|---|---|---|---|")
    for t, res in p["tiles"].items():
        s = res.get("pt_dinf_system", {})
        m = res.get("pt_dinf_monotonicity", {})
        out.append(f"| {t} | {s.get('n_boundary')} | {s.get('n_channel')} ({s.get('n_channel_both_receivers')} / {s.get('n_channel_cardinal_only')} / {s.get('n_channel_diagonal_only')}) | "
                   f"{s.get('n_channel_without_receiver_as_hillslope')} | {s.get('n_hillslope')} | **{m.get('n_lower_than_every_receiver')}** | {m.get('n_lower_than_any_receiver')} | "
                   f"{fmt(m.get('min_drop_to_receiver'), '.3g')} / {fmt(m.get('median_drop_to_receiver'), '.3g')} |")
    syn = p.get("synthetic")
    if syn:
        out.append(f"\n### A2c. Synthetic cases ({syn['n']} x {syn['n']} cells, dx {syn['dx']} m, A_crit {syn['A_crit']} m^2, GM-like m, U, K, D)\n")
        out.append("| case | relief | channel cells D8 / D-inf | mask differs | max dZ(rebuilt - D8)/relief | max dZ(v1 - D8)/relief | monotonicity violations (rebuilt) | MSE vs input D8 / rebuilt / v1 |")
        out.append("|---|---|---|---|---|---|---|---|")
        for name in ("plane_south", "v_equal_slopes", "v_rotated_20deg"):
            c = syn.get(name)
            if not c:
                continue
            out.append(f"| {name} | {c['relief']:.3g} | {c['n_channel_d8']} / {c['n_channel_dinf']} | {c['n_mask_differs']} | {c['max_abs_dZ_dinf_minus_d8_over_relief']:.2e} | "
                       f"{c['max_abs_dZ_dinf_v1_minus_d8_over_relief']:.2e} | {c['monotonicity']['n_lower_than_every_receiver']} | "
                       f"{c['mse_vs_input']['pt_d8']:.4g} / {c['mse_vs_input']['pt_dinf']:.4g} / {c['mse_vs_input']['pt_dinf_v1']:.4g} |")
        c = syn.get("v_rotated_20deg", {}).get("against_fine_d8")
        if c:
            out.append(f"\nRotated V against the D8 model at 4x finer resolution ({c['fine_n']} x {c['fine_n']}, dx {c['fine_dx']} m; {c['sampling']}): "
                       f"max |dZ|/relief rebuilt {c['max_abs_dZ_over_relief']['dinf_rebuilt']:.3g}, D8 coarse {c['max_abs_dZ_over_relief']['d8_coarse']:.3g}, v1 {c['max_abs_dZ_over_relief']['dinf_v1']:.3g}; "
                       f"rms/relief rebuilt {c['rms_dZ_over_relief']['dinf_rebuilt']:.3g}, D8 coarse {c['rms_dZ_over_relief']['d8_coarse']:.3g}, v1 {c['rms_dZ_over_relief']['dinf_v1']:.3g}.")
else:
    out.append("pending (no `d8_vs_dinf` run with both models yet)")
out.append("")

# ======================================================================================= A3 Phase 5.1
out.append("## A3. Phase 5.1: corrected AP grids written and added to the manifest\n")
d, p, r = latest("ap_units_convert")
if p:
    out.append(f"Run `{d.name}` (job {r.get('slurm_job_id')}, git {r['git_hash'][:8]}). Factor {p['tiles']['AP1']['factor_m_per_ftUS']!r} m per US survey foot.\n")
    out.append("| tile | written | sha256 (12) | bytes | z range stored (ft) | z range (m) | bit-identical to the loader's conversion | median slope (m/m) |")
    out.append("|---|---|---|---|---|---|---|---|")
    for t, v in p["tiles"].items():
        out.append(f"| {t} | `{Path(v['written']).name}` | {v['sha256'][:12]} | {v['bytes']} | {v['z_range_ft'][0]:.2f} - {v['z_range_ft'][1]:.2f} | "
                   f"{v['z_range_m'][0]:.2f} - {v['z_range_m'][1]:.2f} | {v['bit_identical_to_loader']} | {fmt(v['sanity'].get('median_slope'), '.3f')} |")
    out.append(f"\nManifest: {p['manifest']['n_entries']} entries; added `{'`, `'.join(p['manifest']['added'])}`.")
else:
    out.append("pending")
out.append("")

# ======================================================================================= A4 Phase 5.2
out.append("## A4. Phase 5.2: D8 directions recomputed from the corrected filled tile against the original `*_fd` (check only)\n")
d, p, r = latest("ap_d8_check")
if p:
    out.append(f"Run `{d.name}` (job {r.get('slurm_job_id')}). Tie tolerance {p['tiles']['AP1']['tie_tolerance_rel']:g} relative on the drop.\n")
    out.append("| tile | cells | differing cells (new metres vs original) | edge-adjacent (outer 2 rows/cols) | of which outer 1 | ties (equal steepest drop) | anything else | interior cells | recomputed metres vs recomputed feet | recomputed feet vs original (interior) | original code 0 / new code 0 |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for t, v in p["tiles"].items():
        out.append(f"| {t} | {v['n_cells']} | {v['n_diff_new_m_vs_orig']} | {v['n_edge_adjacent_margin2']} | {v['n_edge_adjacent_margin1_only']} | {v['n_tie_flats']} | **{v['n_other']}** | "
                   f"{v['n_interior_cells']} | {v['n_diff_new_m_vs_new_ft']} | {v['n_diff_new_ft_vs_orig_interior']} | {v['n_orig_code_zero']} / {v['n_new_code_zero']} |")
    ex = [(t, e) for t, v in p["tiles"].items() for e in v["examples_other"]]
    if ex:
        out.append("\nExamples of the 'anything else' class (first 20 per tile; drops in metres with TopoAnalysis's 1.41 diagonal divisor):\n")
        out.append("| tile | row, col | fd original | fd new (m) | fd new (ft) | drop to original receiver | best drop | NaN neighbours |")
        out.append("|---|---|---|---|---|---|---|---|")
        for t, e in ex[:40]:
            out.append(f"| {t} | {e['row']}, {e['col']} | {e['fd_orig']} | {e['fd_new_m']} | {e['fd_new_ft']} | {fmt(e['drop_to_orig_receiver'], '.6g')} | {e['best_drop']:.6g} | {e['neighbours_nan']} |")
else:
    out.append("pending")
out.append("")

# ======================================================================================= A5 Phase 5.3 bounds
out.append("## A5. Phase 5.3: AP parameter scaling by c = 3937/1200 (loader, `model.param_z_units: ftUS`)\n")
try:
    from lemcvt.config import load_tile_config

    out.append("| tile | K0_ad ft -> m | K0_partitioned ft -> m | D0 ft -> m | K bounds ft -> m | D bounds ft -> m | param_scale |")
    out.append("|---|---|---|---|---|---|---|")
    for t in TILES[:5]:
        f = load_tile_config(t, reproduce_v1_units=True).model
        m = load_tile_config(t, reproduce_v1_units=False).model
        out.append(f"| {t} | {f.K0_ad:.4g} -> {m.K0_ad:.4g} | {f.K0_partitioned:.4g} -> {m.K0_partitioned:.4g} | {f.D0:.4g} -> {m.D0:.4g} | "
                   f"[{f.K_bounds[0]:.3g}, {f.K_bounds[1]:.3g}] -> [{m.K_bounds[0]:.3g}, {m.K_bounds[1]:.3g}] | "
                   f"[{f.D_bounds[0]:.3g}, {f.D_bounds[1]:.3g}] -> [{m.D_bounds[0]:.3g}, {m.D_bounds[1]:.3g}] | {m.param_scale:.10f} |")
except Exception as exc:  # noqa: BLE001
    out.append(f"(configs not loadable here: {exc})")
out.append("")

# ======================================================================================= A6 Phase 5.5 invariance
out.append("## A6. Phase 5.5: invariance of the v2 profiles under the unit change (metre run at the feet runs' evaluation points; tolerance 1e-8 relative)\n")
inv: dict = {}
for d in sorted(root.iterdir()):
    if "_ap_invariance_" in d.name and (d / "params.json").exists():
        p, r = load(d)
        if not p:
            continue
        t = p["tile"]
        if t not in inv:
            inv[t] = (d, {"tile": t, "checks": {}, "feet_runs": {}, "runs": {}}, r)
        for st, c in p["checks"].items():
            if isinstance(c, dict):
                inv[t][1]["checks"][st] = c
                inv[t][1]["feet_runs"][st] = p["feet_runs"].get(st)
                inv[t][1]["runs"][st] = (d.name, r.get("slurm_job_id"))
if inv:
    out.append("| tile | stage | feet run | points | max rel. diff: MSE x c^2 vs feet | K vs c K_ft | D vs c D_ft | profile shape | optimum identical (A_crit / r / (A_crit, m)) | box position identical | pass |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for t, (d, p, r) in inv.items():
        ch = p["checks"]
        for st in ("partitioned", "ad", "m_profile"):
            c = ch.get(st)
            if not c:
                continue
            if st == "partitioned":
                mse_c = max(c["mse_stage2_times_c2_vs_feet"], c["mse_joint_times_c2_vs_feet"])
                shape = max(c["shape_stage2"], c["shape_joint"])
                opt = f"stage 2 {c['stage2_optimum_A_crit']['metres']:.6g} / {c['stage2_optimum_A_crit']['feet']:.6g}; joint {c['joint_optimum_A_crit']['metres']:.6g} / {c['joint_optimum_A_crit']['feet']:.6g} ({'same' if c['argmin_stage2_identical'] and c['argmin_joint_identical'] else 'DIFFERENT'})"
                K, D = c["K_opt_vs_c_K_feet"], c["D_opt_vs_c_D_feet"]
            elif st == "ad":
                mse_c, shape = c["mse_times_c2_vs_feet"], c["shape"]
                opt = f"r {c['r_optimum']['metres']:.6g} / {c['r_optimum']['feet']:.6g} ({'same' if c['argmin_identical'] else 'DIFFERENT'})"
                K, D = c["K_vs_c_K_feet"], c["D_vs_c_D_feet"]
            else:
                mse_c, shape = c["mse_joint_times_c2_vs_feet"], max(c["shape"], c["profile_m_shape"])
                om, of = c["optimum"]["metres"], c["optimum"]["feet"]
                opt = f"({om['A_crit']:.6g}, {om['m']:.2f}) / ({of['A_crit']:.6g}, {of['m']:.2f}) ({'same' if c['optimum_identical'] else 'DIFFERENT'})"
                K, D = c["K_opt_vs_c_K_feet"], c["D_opt_vs_c_D_feet"]
            out.append(f"| {t} | {st} (run `{p['runs'][st][0]}`, job {p['runs'][st][1]}) | `{p['feet_runs'][st]}` | {c['n_points']} | {mse_c:.1e} | {K:.1e} | {D:.1e} | {shape:.1e} | {opt} | {c.get('where_identical')} | {'yes' if c['pass'] else 'NO'} |")
    out.append("\nModel scaling check (loader values against c x the feet run's values, relative): " + "; ".join(
        f"{t}: " + ", ".join(f"{k} {v:.1e}" for k, v in p['checks']['partitioned']['model_scaling_rel_err'].items()) for t, (d, p, r) in inv.items() if 'partitioned' in p['checks']))
else:
    out.append("pending")
out.append("")

# ======================================================================================= A7 Phase 5.6 AP nonlinear metres
out.append("## A7. Phase 5.6: AP nonlinear v2 profile in metres (K0, D0 = c x v1 values, S_c = 1.27)\n")
nlm = v2_runs("nonlinear", units="m", pred=lambda p: p["tile"].startswith("AP"))
if nlm:
    out.append("| tile | run (job) | K0, D0 (m) | grid + refine | evaluations hitting the cap | optimum A_crit (Picard it.) | MSE (m^2) | 1 % interval (m^2) | hillslope > 0.5 S_c / > 0.9 S_c / max ratio at the optimum | v1 feet A_crit (same code in metres: MSE, it.) | wall (h) |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for t in TILES[:5]:
        if t not in nlm:
            out.append(f"| {t} | pending | | | | | | | | | |")
            continue
        d, p, r = nlm[t]
        v = p["v2"]["nonlinear"]
        o = v["optimum"]
        c = v["certificate"].get("v1_optimum", {})
        i1 = v["identifiability"]["level_0.01"]
        out.append(f"| {t} | `{d.name}` ({r.get('slurm_job_id')}) | {p['model']['K0_partitioned']:.4g}, {p['model']['D0']:.4g} | {v['n_grid']} + {v['n_refine']} | **{v['n_hit_cap']}** of {v['n_evaluations']} | "
                   f"{o['A_crit']:.6g} ({o['picard_iterations']}) | {o['mse']:.6g} | {fmt(i1.get('A_crit_min'), '.4g')} - {fmt(i1.get('A_crit_max'), '.4g')} ({i1['n_grid_points']} pts) | "
                   f"{fmt(o.get('frac_hillslope_gt_0.5Sc'), '.3f')} / {fmt(o.get('frac_hillslope_gt_0.9Sc'), '.4f')} / {fmt(o.get('max_slope_ratio'), '.2f')} | "
                   f"{fmt(c.get('A_crit'), '.6g')}: {fmt(c.get('mse'), '.6g')} ({c.get('picard_iterations')}) | {v['wall_s'] / 3600:.1f} |")
else:
    out.append("pending")
out.append("")

# ======================================================================================= A8 6.1 exponent profile on the wide grid
out.append("## A8. Section 6.1: exponent profile on m = 0.05-0.90, AP in metres, and the site profile S(m)\n")
try:
    from lemcvt.analyses.exponent_site import k_recovery, latest_audit, latest_m_profile_runs, site_summary, theta_band_at

    runs = latest_m_profile_runs()
    if len(runs) < 10:
        out.append(f"({len(runs)} of 10 tiles have a wide-grid m_profile run in metres; the rest are pending)")
    if runs:
        out.append("| tile | run (job) | z unit | m grid | fitted (A_crit, m) | K, D at the fit | MSE at fit / at config m | 1 % contour m | 5 % contour m | MSE(m)/min at m = 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90 |")
        out.append("|---|---|---|---|---|---|---|---|---|---|")
        ms = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
        for t in TILES:
            if t not in runs:
                out.append(f"| {t} | pending | | | | | | | | |")
                continue
            d, p = runs[t]
            v = p["v2"]["m_profile"]
            o = v["optimum"]
            c1, c5 = v["contour_0.01"], v["contour_0.05"]
            pm = {round(q["m"], 2): q["mse"] for q in v["profile_m"]}
            mn = min(pm.values())
            cm = v.get("config_m_row") or {}
            rj = json.loads((d / "run.json").read_text())
            out.append(f"| {t} | `{d.name}` ({rj.get('slurm_job_id')}) | {p['inputs'].get('z_units_working')} | {min(v['grid']['m_values']):.2f}-{max(v['grid']['m_values']):.2f} x {v['grid']['n_acrit']} A_crit | "
                       f"({o['A_crit']:.4g}, {o['m']:.2f}) | {o['K_opt']:.4g}, {o['D_opt']:.4g} ({o['where']}) | {o['mse_joint']:.5g} / {fmt(cm.get('mse_joint'), '.5g')} | "
                       f"{c1['m_min']:.2f}-{c1['m_max']:.2f} | {c5['m_min']:.2f}-{c5['m_max']:.2f} | " + ", ".join(f"{pm[m] / mn:.3f}" if m in pm else "-" for m in ms) + " |")
        summ = site_summary(runs=runs)
        aud = latest_audit()
        out.append("\n### A8b. Site profile S(m) = sum_t MSE*_t(m)/MSE*_t,min: optimum, 1 % and 5 % intervals, leave-one-out\n")
        out.append("| site | tiles | m_site | S_min | 1 % interval | 5 % interval | at grid edge | median fitted A_crit (tile optima) | median A_crit at m_site |")
        out.append("|---|---|---|---|---|---|---|---|---|")
        for site, rec in summ["sites"].items():
            a = rec["all"]
            out.append(f"| {site} | {', '.join(a['tiles'])} | **{a['m_opt']:.2f}** | {a['S_min']:.3f} | {a['interval_0.01']['m_min']:.2f}-{a['interval_0.01']['m_max']:.2f} ({a['interval_0.01']['n_grid_points']} pts, {len(a['interval_0.01']['intervals'])} int.) | "
                       f"{a['interval_0.05']['m_min']:.2f}-{a['interval_0.05']['m_max']:.2f} | {a['interval_0.01']['touches_grid_edge']} | {rec['median_fitted_A_crit']:.4g} | {rec['median_A_crit_at_site_m']:.4g} |")
        out.append("\n| site | left out | m_site | 1 % interval | 5 % interval |")
        out.append("|---|---|---|---|---|")
        for site, rec in summ["sites"].items():
            for t, a in rec["leave_one_out"].items():
                out.append(f"| {site} | {t} | {a['m_opt']:.2f} | {a['interval_0.01']['m_min']:.2f}-{a['interval_0.01']['m_max']:.2f} | {a['interval_0.05']['m_min']:.2f}-{a['interval_0.05']['m_max']:.2f} |")
        out.append("\n### A8c. Measured site-pooled theta(A_c) at the median fitted A_crit (interpolated in log A_c; 95 % basin-bootstrap band)\n")
        if aud:
            out.append(f"Audit run `{aud[0].name}`.\n")
            out.append("| site | A_c used | estimator | theta | 95 % band | basins at the nearest step | S(m) optimum m_site | m_site inside band | 1 % interval overlaps band |")
            out.append("|---|---|---|---|---|---|---|---|---|")
            for site, rec in summ["sites"].items():
                for which, a_c in (("median fitted A_crit", rec["median_fitted_A_crit"]), ("median A_crit at m_site", rec["median_A_crit_at_site_m"])):
                    for est in ("pooled", "equal"):
                        b = theta_band_at(aud[1], site, a_c, est)
                        if not b:
                            out.append(f"| {site} | {which} {a_c:.4g} | {est} | - | no band | | {rec['all']['m_opt']:.2f} | - | - |")
                            continue
                        i1 = rec["all"]["interval_0.01"]
                        ins = b["lo95"] <= rec["all"]["m_opt"] <= b["hi95"]
                        ov = not (i1["m_max"] < b["lo95"] or i1["m_min"] > b["hi95"])
                        out.append(f"| {site} | {which} {a_c:.4g} | {est} | {b['theta']:.2f} | [{b['lo95']:.2f}, {b['hi95']:.2f}] | {b['n_basins_nearest']} | {rec['all']['m_opt']:.2f} | {'yes' if ins else 'no'} | {'yes' if ov else 'no'} |")
        # ======================================================================================= A9 6.2
        out.append("\n## A9. Section 6.2: K recovery consistent in units, at the site-level m\n")
        if aud and aud[1].get("settings", {}).get("thetas") and min(aud[1]["settings"]["thetas"]) <= 0.0 + 1e-9:
            rows = k_recovery(summ, aud[1], runs=runs)
            out.append("| tile | m_site | K_fit (joint profile at m_site; A_crit, D) | k_s at theta = m_site, A_c = 1e3 (basins, n_eff) | U/k_s / K_fit (1e3) | k_s at A_c = 1e4 (basins, n_eff) | U/k_s / K_fit (1e4) | site-pooled k_s 1e3 / 1e4 |")
            out.append("|---|---|---|---|---|---|---|---|")
            for q in rows:
                out.append(f"| {q['tile']} | {q['m_site']:.2f} | {q['K_fit']:.4g} ({q['A_crit_fit']:.4g}, {q['D_fit']:.4g}) | {fmt(q['ks_1000'], '.4g')} ({q['n_basins_1000']}, {fmt(q['n_eff_1000'], '.1f')}) | "
                           f"**{fmt(q['ratio_1000'], '.3g')}** | {fmt(q['ks_10000'], '.4g')} ({q['n_basins_10000']}, {fmt(q['n_eff_10000'], '.1f')}) | **{fmt(q['ratio_10000'], '.3g')}** | "
                           f"{fmt(q['ks_site_pooled_1000'], '.4g')} / {fmt(q['ks_site_pooled_10000'], '.4g')} |")
        else:
            out.append("pending (needs the audit rerun with the theta grid from 0.00)")
except Exception as exc:  # noqa: BLE001
    out.append(f"pending ({exc!r})")
out.append("")

# ======================================================================================= A10 6.3 curvature
out.append("## A10. Section 6.3: curvature options side by side (every row carries its hilltop definition, window and pixel count)\n")
d, p, r = latest("measure_audit", lambda n, p, r: p.get("L_f_selection") is not None)
if p:
    lf = p["L_f_selection"]
    out.append(f"Audit run `{d.name}` (job {r.get('slurm_job_id')}, git {r['git_hash'][:8]}).\n")
    out.append(f"### A10a. L_f from the synthetic (crest width {lf['crest_width_m']} m, tolerance {100 * lf['tol']:.0f} %): facet-plus-noise bias of the recovered curvature against window length\n")
    L = lf["curves"]["AP"]["window_lengths_m"]
    out.append("| window L (m, synthetic dx = 1) | " + " | ".join(f"{x:g}" for x in L) + " |")
    out.append("|---|" + "---|" * len(L))
    for site in ("AP", "GM"):
        c = lf["curves"][site]
        sd = c.get("bias_std_across_seeds")
        out.append(f"| {site} ({c['variant']}; facet {c['facet_len_m']:.2f} m, noise {c['noise_sigma']:.3f} m, c_true {c['c_true']:.3g}; mean over {c.get('n_seeds', 1)} realisations) | "
                   + " | ".join(f"{100 * b:+.2f} %" + (f" (sd {100 * s_:.2f})" if sd else "") for b, s_ in zip(c["bias"], sd or [None] * len(c["bias"]), strict=True)) + " |")
    out.append(f"\n**L_f = {lf['L_f_synthetic_m']} m (w_f = {lf['w_f']} cells; on AP (2 w_f + 1) dx = {(2 * lf['w_f'] + 1) * 0.9752:.2f} m).** Rule: {lf['rule']}.\n")
    out.append("### A10b. C_ht and D0 by option\n")
    out.append("| tile | option | hilltop definition | network | window (m) | pixels | segments | C_ht (1/m) [68 %] | D0 (m^2/yr) [68 %] | width 2d 25 / 50 % (m) | synthetic bias at median / 25 % width |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    syn_old = p.get("synthetic_ridges_post_exclusion", {})
    syn3 = p.get("synthetic_ridges_option_iii", {})
    for t in TILES:
        h = p["tiles"][t]["hilltops"]
        site = t[:2]
        for key, label in (("1000", "network 1e3 m^2"), ("10000", "network 1e4 m^2"), ("rollover", "rollover network (measure set)")):
            net = h["networks"].get(key)
            if not net:
                continue
            for sub, opt in (("all", "fixed L*, no exclusion"), ("excluded", "fixed L*, exclusion L*/2 (option i)")):
                q = net[sub]["at_fixed_L"]
                w = net[sub].get("width_2x_distance_m") or {}
                bias = ""
                if key == "10000" and sub == "excluded":
                    b = syn_old.get(site, {}).get("bias_post_exclusion", {}).get(t)
                    if b:
                        bias = f"{100 * b['bias_at_median_width']:+.1f} % / {100 * b['bias_at_q25_width']:+.1f} %"
                ci = q.get("C_ht_ci68")
                dci = q.get("D0_ci68")
                out.append(f"| {t} | {opt} | divides between outlet basins of the network, 2 cells from the edge, slope <= 0.4{'; pixels within L*/2 of the network dropped' if sub == 'excluded' else ''} | {label} (A_c {net['A_c']:.3g}) | "
                           f"{q['L_m']:.1f} | {q['n_pixels']} | {net[sub]['n_segments']} | {fmt(q['C_ht'], '.4g')} {f'[{ci[0]:.3g}, {ci[1]:.3g}]' if ci else ''} | {fmt(q.get('D0'), '.4g')} {f'[{dci[0]:.3g}, {dci[1]:.3g}]' if dci else ''} | "
                           f"{fmt(w.get('25'), '.0f')} / {fmt(w.get('50'), '.0f')} | {bias} |")
        o3 = h.get("option_iii")
        if o3:
            q = o3["at_fixed_L"]
            ci = q.get("C_ht_ci68")
            dci = q.get("D0_ci68")
            w = o3.get("width_2x_distance_m") or {}
            b = syn3.get(site, {}).get("bias_option_iii", {}).get(t)
            bias = f"{100 * b['bias_at_median_width']:+.1f} % / {100 * b['bias_at_q25_width']:+.1f} %" if b else ""
            out.append(f"| {t} | **fixed L_f, exclusion L_f/2 (option iii)** | measure set (rollover network), pixels within L_f/2 of the network dropped | rollover (A_c {o3['A_c']:.3g}) | "
                       f"{q['L_m']:.1f} | {q['n_pixels']} (of {o3['n_before_exclusion']}) | {o3['n_segments']} | {fmt(q['C_ht'], '.4g')} {f'[{ci[0]:.3g}, {ci[1]:.3g}]' if ci else ''} | "
                       f"{fmt(q.get('D0'), '.4g')} {f'[{dci[0]:.3g}, {dci[1]:.3g}]' if dci else ''} | {fmt(w.get('25'), '.0f')} / {fmt(w.get('50'), '.0f')} | {bias} |")
else:
    out.append("pending")
out.append("")

# ======================================================================================= A11 6.4 ladder rerun
out.append("## A11. Section 6.4.2: free ladder rerun with the theta grid from 0.00 (pooled / equal-weight / median estimators; n_eff; fraction of kept basins with best theta at the lower edge)\n")
d, p, r = latest("measure_audit", lambda n, p, r: p.get("settings", {}).get("theta_grid_lower_edge") == 0.0)
if p:
    out.append(f"Audit run `{d.name}` (job {r.get('slurm_job_id')}); theta grid {min(p['settings']['thetas']):.2f}-{max(p['settings']['thetas']):.2f}; basin bootstrap, {p['settings']['n_boot']} draws.\n")
    steps_show = [100.0, 398.107, 1000.0, 1995.26, 3981.07, 10000.0, 39810.7, 100000.0]

    def cell(s, est):
        e = s.get(est)
        if not e:
            return "-"
        ci = e.get("theta_ci95")
        return f"{e['theta_star']:.2f}" + (f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "") + (f" ({s.get('n_basins')})" if (s.get("n_basins") or 0) < 10 else "")

    def row(name, ladder):
        st = {round(s["A_c"], 3): s for s in ladder}
        cells = []
        for a in steps_show:
            s = next((v for k, v in st.items() if abs(k - a) < 1e-3 * a), None)
            if not s or not s.get("pooled"):
                cells.append("-")
                continue
            cells.append(f"P {cell(s, 'pooled')}<br>E {cell(s, 'equal')}<br>M {cell(s, 'median')}<br>n_eff {fmt(s['pooled'].get('n_eff_centred'), '.1f')}, lower-edge {fmt(s.get('frac_basins_theta_at_lower_edge'), '.2f')}")
        return f"| {name} | " + " | ".join(cells) + " |"

    out.append("| tile | " + " | ".join(f"{a:g}" for a in steps_show) + " |")
    out.append("|---|" + "---|" * len(steps_show))
    for t in TILES:
        out.append(row(t, p["tiles"][t]["free_ladder"]))
    for site in ("AP", "GM"):
        out.append(row(f"**{site} pooled**", p["pooled"][site]["ladder"]))
    out.append("\nP = pooled through-origin (02c), E = equal-weight, M = median of per-basin best theta; [95 %] basin bootstrap; (n) basins when fewer than 10; "
               "lower-edge = fraction of kept basins whose own best theta is 0.00.")
else:
    out.append("pending")
out.append("")
print("\n".join(out))
