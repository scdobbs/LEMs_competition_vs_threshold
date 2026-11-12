"""Appendix tables for the Phase 6 results report (addendum 6, section 6; outline sections 2-8).

Usage: PYTHONNOUSERSITE=1 LEMCVT_RESULTS=... LEMCVT_DATA=... python reports/v2/scripts/make_phase6_results_tables.py > appendix.md
Reads the latest Phase 6 run of each step and tile under $LEMCVT_RESULTS (metres, uncropped); sections whose runs
do not exist yet are marked pending.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(os.environ.get("LEMCVT_RESULTS", "/home/groups/hilley/lemcvt_results"))
repo = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo / "src"))
TILES = ["AP1", "AP2", "AP3", "AP4", "AP5", "GM1", "GM2", "GM3", "GM4", "GM5"]
out: list[str] = []


def fmt(v, f=".4g"):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    try:
        return format(v, f)
    except (TypeError, ValueError):
        return str(v)


def ci(v, f=".3g"):
    return f"[{fmt(v[0], f)}, {fmt(v[1], f)}]" if v else "-"


def ivl(v, f=".0f"):
    """Identifiability intervals, a list of [lo, hi] pairs, as 'lo-hi; lo-hi'."""
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


def tile_runs(stage: str, variant: str = "A", k_factor: float | None = 1.0, suffix: str | None = None):
    """Latest v2 tile_run per tile holding `stage` in metres with the D0 variant (and K factor for the nonlinear joint)."""
    res = {}
    for d, p, r in runs("tile_run"):
        if p.get("optimizer") != "v2" or stage not in p.get("v2", {}) or p.get("inputs", {}).get("z_units_working") != "m":
            continue
        if p.get("model", {}).get("d0_variant", "A") != variant:
            continue
        if suffix and not d.name.endswith(suffix):
            continue
        if stage == "nonlinear_joint" and ("split" in p["v2"]["nonlinear_joint"]["setup"] or (k_factor is not None and p["v2"]["nonlinear_joint"]["setup"].get("k_factor", 1.0) != k_factor)):
            continue
        if p.get("model", {}).get("nonlinear_max_iter", 50) != 100 and stage in ("nonlinear", "nonlinear_joint"):
            continue        # Phase 6 runs only (cap 100)
        if "measured" not in (r.get("config") or {}).get("tile", {}):
            continue        # Phase 6 configs only
        res[p["tile"]] = (d, p, r)
    return res


def hours(p, key="wall_s"):
    v = p.get(key)
    return f"{v / 3600:.1f}" if isinstance(v, int | float) else "-"


# ======================================================================================= B1 configs
out.append("## B1. Section 1.3: the measured inputs written into the configs (D0 bracket)\n")
try:
    from lemcvt.config import load_tile_config

    out.append("| tile | m | k_s (m^(2m)) at 1e3 [68 %] (n_eff) | K0 = U/k_s (1/yr) | C_ht_A at L_f (1/m) | D0_A (m^2/yr) [68 %] | L_w (m) | C_ht_B (1/m) | D0_B (m^2/yr) | D0_B / D0_A | Picard cap |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for t in TILES:
        c = load_tile_config(t)
        m_ = c.measured
        pk = m_["provenance"]["ks"]
        pc = m_["provenance"]["C_ht"]
        out.append(f"| {t} | {m_['m']} | {m_['ks']:.3f} {ci(pk.get('ci68'))} ({fmt(pk.get('n_eff_centred'), '.1f')}) | {c.model.K0_partitioned:.3e} | {m_['C_ht']:.4g} | "
                   f"{c.model.D0:.3e} {ci(pc.get('D0_ci68'))} | {m_['L_w_m']} | {m_['C_ht_B']:.4g} | {fmt(c.model.D0_B, '.3e')} | {fmt(c.model.D0_B / c.model.D0 if c.model.D0_B else None, '.2f')} | {c.model.nonlinear_max_iter} |")
    c = load_tile_config("AP1")
    out.append(f"\nProvenance: measured-inputs run `{c.measured['provenance']['ks'].get('run')}` (git {str(c.measured['provenance']['ks'].get('git_hash'))[:8]}); "
               f"site m from the S(m) minima (AP 1 % interval {c.measured['provenance']['m'].get('S_interval_1pct')}).")
except Exception as exc:  # noqa: BLE001
    out.append(f"(configs not loadable here: {exc})")
out.append("")

# ======================================================================================= B2 seam test
out.append("## B2. Section 3: the AP1 seam test (pre-registered rule)\n")
d, p, r = latest("seam_test")
if p:
    a, v = p["affected_set"], p["verdict"]
    out.append(f"Run `{d.name}` (job {r.get('slurm_job_id')}, git {r['git_hash'][:8]}); misfit over all valid cells ({p['partitioned']['n_cells_misfit_all']}) "
               f"and over all valid cells except S ({p['partitioned']['n_cells_misfit_masked']}).\n")
    out.append(f"|S| = **{a['n_S']}** cells ({100 * a['frac_S_of_valid']:.2f} % of {a['n_valid']} valid): rows 385-395 {a['n_rows_band']}, downstream of the {a['n_seam_cells']} seam cells "
               f"{a['n_downstream_of_seam']}, inconsistent {a['n_inconsistent']} (union).\n")
    out.append("| profile | mask | A_crit (m^2) | K (1/yr) | D (m^2/yr) | MSE (m^2) | 1 % interval | 5 % interval |")
    out.append("|---|---|---|---|---|---|---|---|")
    for mk in ("all", "masked"):
        q = p["partitioned"][mk]
        jo, s2 = q["joint_optimum"], q["stage2_optimum"]
        i1, i5 = q["joint_identifiability"]["level_0.01"], q["joint_identifiability"]["level_0.05"]
        out.append(f"| linear joint | {mk} | {jo['A_crit']:.1f} | {jo['K']:.4e} | {jo['D']:.4e} | {jo['mse']:.5g} | {ivl(i1['intervals'])} | {fmt(i5['A_crit_min'], '.0f')}-{fmt(i5['A_crit_max'], '.0f')} |")
        i1s = q["stage2_identifiability"]["level_0.01"]
        out.append(f"| stage 2 at D0_A | {mk} | {s2['A_crit']:.1f} | - | - | {s2['mse']:.5g} | {ivl(i1s['intervals'])} | - |")
        ao = p["ad"][mk]["optimum"]
        out.append(f"| AD | {mk} | - | {ao['K']:.4e} | {ao['D']:.4e} | {ao['mse']:.5g} | r = {ao['r']:.4e} | - |")
    sh = p["partitioned"]["profile_shape"]
    out.append(f"\nProfile shapes (max relative difference of MSE/MSE_min between the masks): joint {sh['joint_profile_max_rel_shape_diff']:.3g}, stage 2 {sh['stage2_profile_max_rel_shape_diff']:.3g}.\n")
    out.append("| criterion | value | threshold | pass |")
    out.append("|---|---|---|---|")
    out.append(f"| masked joint A_crit inside the unmasked 1 % interval | {v['joint_A_crit']['masked']:.1f} in {ivl(v['unmasked_1pct_intervals'])} | - | {v['masked_joint_A_crit_inside_unmasked_1pct_interval']} |")
    out.append(f"| K relative change | {100 * v['K_rel_change']:.3f} % | < 2 % | {v['K_rel_change'] < 0.02} |")
    out.append(f"| D relative change | {100 * v['D_rel_change']:.3f} % | < 5 % | {v['D_rel_change'] < 0.05} |")
    out.append(f"| AD r change (decades) | {v['ad_r_log10_change']:.4f} | < 0.1 | {v['ad_r_log10_change'] < 0.1} |")
    out.append(f"| **seam negligible** | | | **{v['negligible']}** |")
else:
    out.append("pending")
out.append("")

# ======================================================================================= B3 AD
out.append("## B3. Section 5.1: AD profile (r = K/D, closed-form D; 200-point grid + Brent)\n")
ad = tile_runs("ad")
if ad:
    out.append("| tile | run (job) | r = K/D | K (1/yr) | D (m^2/yr) | MSE (m^2) | box | 1 % interval in r | 5 % interval in r | AD no-opt MSE at (K0, D0_A) | v1 K, D, MSE (converted) | wall (h) |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for t in TILES:
        if t not in ad:
            continue
        d, p, r = ad[t]
        v = p["v2"]["ad"]
        o = v["optimum"]
        i1, i5 = v["identifiability"]["level_0.01"], v["identifiability"]["level_0.05"]
        no = next((f for f in p["forward_runs"] if f["key"] == "ad_no_opt"), {})
        ref = p.get("v1_reference", {}).get("ad", {})
        c = 3937 / 1200 if t.startswith("AP") else 1.0
        out.append(f"| {t} | `{d.name}` ({r.get('slurm_job_id')}) | {o['r']:.4e} | {o['K']:.4e} | {o['D']:.4e} | {o['mse']:.5g} | {o.get('where')} | "
                   f"{fmt(i1.get('r_min'), '.3e')}-{fmt(i1.get('r_max'), '.3e')} | {fmt(i5.get('r_min'), '.3e')}-{fmt(i5.get('r_max'), '.3e')} | {fmt(no.get('mse'), '.5g')} | "
                   f"{fmt(ref.get('K', 0) * c, '.3e')}, {fmt(ref.get('D', 0) * c, '.3e')}, {fmt(ref.get('MSE', 0) / c**2, '.5g')} | {hours(v)} |")
else:
    out.append("pending")
out.append("")

# ======================================================================================= B4 linear partitioned
out.append("## B4. Section 5.2: linear partitioned model (stage 2 at D0_A and D0_B, joint; thin 2000 + refinement + every unique area in the 5 % intervals)\n")
pt = tile_runs("partitioned", suffix="partitioned-refined")
if pt:
    out.append("| tile | run (job) | evaluations (fill) | stage 2 A_crit at D0_A (MSE) | 1 % | stage 2 A_crit at D0_B (MSE) | 1 % | joint A_crit | K | D | MSE | joint 1 % | joint 5 % (points) | K, D range over the 1 % interval | direct check | wall (h) |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for t in TILES:
        if t not in pt:
            continue
        d, p, r = pt[t]
        v = p["v2"]["partitioned"]
        s2, jo = v["stage2_optimum"], v["joint_optimum"]
        sb = v.get("stage2_B_optimum")
        i1s, i1b = v["stage2_identifiability"]["level_0.01"], (v.get("stage2_B_identifiability") or {}).get("level_0.01", {})
        i1, i5 = v["joint_identifiability"]["level_0.01"], v["joint_identifiability"]["level_0.05"]
        fill = v.get("interval_fill", {})
        out.append(f"| {t} | `{d.name}` ({r.get('slurm_job_id')}) | {v['n_evaluations']} ({fill.get('n_unique_evaluated', '-')}) | {s2['A_crit']:.1f} ({s2['mse']:.5g}) | {ivl(i1s['intervals'])} | "
                   f"{fmt(sb['A_crit'] if sb else None, '.1f')} ({fmt(sb['mse'] if sb else None, '.5g')}) | {ivl(i1b.get('intervals'))} | {jo['A_crit']:.1f} | {jo['K']:.4e} | {jo['D']:.4e} | {jo['mse']:.5g} | "
                   f"{ivl(i1['intervals'])} | {fmt(i5['A_crit_min'], '.0f')}-{fmt(i5['A_crit_max'], '.0f')} ({i5['n_grid_points']}) | "
                   f"K {fmt(i1.get('K_min'), '.2e')}-{fmt(i1.get('K_max'), '.2e')}, D {fmt(i1.get('D_min'), '.2e')}-{fmt(i1.get('D_max'), '.2e')} | "
                   f"{fmt(v.get('direct_check', {}).get('joint', {}).get('rel_diff'), '.1e')} | {hours(v.get('refinement', {}))} + {hours(fill)} |")
else:
    out.append("pending")
out.append("")

# ======================================================================================= B5 nonlinear
out.append("## B5. Section 5.3: nonlinear A_crit profile at D0_A (all tiles) and D0_B (AP), S_c = 1.27, Picard cap 100\n")
for variant in ("A", "B"):
    nl = tile_runs("nonlinear", variant=variant)
    if not nl:
        out.append(f"D0_{variant}: pending\n")
        continue
    out.append(f"D0_{variant}:\n")
    out.append("| tile | run (job) | D0 used | grid + refine | cap hits | optimum A_crit (Picard it.) | MSE (m^2) | 1 % interval | hillslope > 0.5 / > 0.9 S_c | linear stage-2 A_crit, same D0 | wall (h) |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for t in TILES:
        if t not in nl:
            continue
        d, p, r = nl[t]
        v = p["v2"]["nonlinear"]
        o = v["optimum"]
        i1 = v["identifiability"]["level_0.01"]
        s2 = None
        if t in pt:
            vp = pt[t][1]["v2"]["partitioned"]
            s2 = (vp["stage2_optimum"] if variant == "A" else vp.get("stage2_B_optimum")) or {}
        out.append(f"| {t} | `{d.name}` ({r.get('slurm_job_id')}) | {p['model']['D0']:.3e} | {v['n_grid']} + {v['n_refine']} | {v['n_hit_cap']} of {v['n_evaluations']} | "
                   f"{o['A_crit']:.6g} ({o['picard_iterations']}) | {o['mse']:.6g} | {fmt(i1.get('A_crit_min'), '.4g')}-{fmt(i1.get('A_crit_max'), '.4g')} | "
                   f"{fmt(o.get('frac_hillslope_gt_0.5Sc'), '.3f')} / {fmt(o.get('frac_hillslope_gt_0.9Sc'), '.4f')} | {fmt((s2 or {}).get('A_crit'), '.1f')} | {hours(v)} |")
    out.append("")

# ======================================================================================= B6 nonlinear joint
out.append("## B6. Section 5.4: nonlinear joint profile over (A_crit, D) at K fixed to the linear joint optimum (30 x 20 grid), and the K check\n")
nlj = tile_runs("nonlinear_joint")
if nlj:
    out.append("| tile | run (job) | K (1/yr) | A_crit range (m^2) | optimum A_crit | optimum D (m^2/yr) | MSE (m^2) | on grid edge (A_crit / D) | 1 % contour: A_crit, D | 5 % contour: A_crit, D (points) | cap hits (in 5 %) | hillslope > 0.5 / > 0.9 S_c | Picard it. | linear joint D | wall (h) |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for t in TILES:
        if t not in nlj:
            continue
        d, p, r = nlj[t]
        v = p["v2"]["nonlinear_joint"]
        o, c = v["optimum"], v["contours"]
        c1, c5 = c["contour_0.01"], c["contour_0.05"]
        s = v["setup"]
        out.append(f"| {t} | `{d.name}` ({r.get('slurm_job_id')}) | {s['K']:.4e} | {s['A_crit_range'][0]:.0f}-{s['A_crit_range'][1]:.0f} | {o['A_crit']:.1f} | {o['D']:.4e} | {o['mse']:.5g} | "
                   f"{v['on_grid_edge']['A_crit']} / {v['on_grid_edge']['D']} | {c1['A_crit_min']:.0f}-{c1['A_crit_max']:.0f}, {c1['D_min']:.2e}-{c1['D_max']:.2e} | "
                   f"{c5['A_crit_min']:.0f}-{c5['A_crit_max']:.0f}, {c5['D_min']:.2e}-{c5['D_max']:.2e} ({c5['n_points']}) | {v['n_hit_cap']} ({v['n_hit_cap_in_5pct_contour']}) | "
                   f"{fmt(o.get('frac_hillslope_gt_0.5Sc'), '.3f')} / {fmt(o.get('frac_hillslope_gt_0.9Sc'), '.4f')} | {o['picard_iterations']} | {s['linear_joint_optimum']['D']:.4e} | {hours(v)} |")
    # K check
    out.append("\nK check (same grid at K x 0.95 and K x 1.05):\n")
    out.append("| tile | K factor | run (job) | optimum A_crit | optimum D | MSE | MSE change vs K x 1 | A_crit change | D change |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for t in ("AP5", "GM1"):
        base = nlj.get(t)
        for kf in (0.95, 1.05):
            kr = tile_runs("nonlinear_joint", k_factor=kf).get(t)
            if not kr:
                out.append(f"| {t} | {kf} | pending | | | | | | |")
                continue
            d, p, r = kr
            o = p["v2"]["nonlinear_joint"]["optimum"]
            ob = base[1]["v2"]["nonlinear_joint"]["optimum"] if base else None
            out.append(f"| {t} | {kf} | `{d.name}` ({r.get('slurm_job_id')}) | {o['A_crit']:.1f} | {o['D']:.4e} | {o['mse']:.5g} | "
                       f"{fmt((o['mse'] / ob['mse'] - 1) if ob else None, '+.3%')} | {fmt((o['A_crit'] / ob['A_crit'] - 1) if ob else None, '+.2%')} | {fmt((o['D'] / ob['D'] - 1) if ob else None, '+.2%')} |")
else:
    out.append("pending")
out.append("")

# ======================================================================================= B7 D-inf joint
out.append("## B7. Section 5.5: D-infinity joint profile (rebuilt partitioned model) beside the D8 joint optimum\n")
dj = tile_runs("dinf_joint")
if dj:
    out.append("| tile | run (job) | D-inf unique areas (used) | joint A_crit | K | D | MSE | box | 1 % interval | direct check | channel cells lower than every receiver | D8 joint A_crit, K, D, MSE | MSE ratio D-inf / D8 | wall (h) |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for t in ("AP1", "GM1"):
        if t not in dj:
            continue
        d, p, r = dj[t]
        v = p["v2"]["dinf_joint"]
        jo = v["joint_optimum"]
        i1 = v["joint_identifiability"]["level_0.01"]
        d8 = pt.get(t)
        j8 = d8[1]["v2"]["partitioned"]["joint_optimum"] if d8 else None
        out.append(f"| {t} | `{d.name}` ({r.get('slurm_job_id')}) | {v['grid']['n_unique']} ({v['grid']['n_used']}) | {jo['A_crit']:.1f} | {jo['K']:.4e} | {jo['D']:.4e} | {jo['mse']:.5g} | {jo['where']} | "
                   f"{ivl(i1['intervals'])} | {v['direct_check']['joint']['rel_diff']:.1e} | {v['channel_monotonicity_at_optimum']['n_lower_than_every_receiver']} of {v['channel_monotonicity_at_optimum']['n_channel']} | "
                   f"{fmt(j8['A_crit'] if j8 else None, '.1f')}, {fmt(j8['K'] if j8 else None, '.3e')}, {fmt(j8['D'] if j8 else None, '.3e')}, {fmt(j8['mse'] if j8 else None, '.5g')} | "
                   f"{fmt(jo['mse'] / j8['mse'] if j8 else None, '.2f')} | {hours(v)} |")
else:
    out.append("pending")
out.append("")

# ======================================================================================= B8 curvature comparison + D recovery
out.append("## B8. Sections 4 and 5.7: C_ht(L) of the fitted surfaces on the option (iii) set, and D recovery\n")
d, p, r = latest("curvature_compare")
if p:
    out.append(f"Run `{d.name}` (job {r.get('slurm_job_id')}). Ratio C_obs(L)/C_model(L) at every window; channel fraction = fraction of the option (iii) pixels that are channel cells of the fitted model.\n")
    out.append("| tile | surface | pixels | 7 | 9 | 11 | 15 | 21 | 25 | 31 | 41 | 51 m | channel fraction | median -laplacian5 (all / hillslope px) | U/D_fit |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for t in TILES:
        if t not in p["tiles"]:
            continue
        q = p["tiles"][t]
        for name in ("observed", "linear_joint", "nonlinear_joint", "ad"):
            s = q["surfaces"].get(name)
            if not s:
                continue
            if name == "observed":
                cells = " | ".join(f"{v:.3g}" for v in s["C_ht_L"].values())
            else:
                cells = " | ".join(f"{fmt(v, '.2f')}" for v in s["ratio_obs_over_model_L"].values())
            out.append(f"| {t} | {name}{' (C_ht, 1/m)' if name == 'observed' else ' (obs/model)'} | {q['n_pixels']} | {cells} | {fmt(s.get('frac_option_iii_pixels_channel'), '.3f')} | "
                       f"{fmt(s.get('median_minus_laplacian5'), '.3g')} / {fmt(s.get('median_minus_laplacian5_hillslope_pixels'), '.3g')} | {fmt(s.get('U_over_D'), '.3g')} |")
    out.append("\nD recovery (addendum 5, 3.7; addendum 6, 5.7):\n")
    out.append("| tile | D0_A | D0_B | D_fit linear joint | / D0_A | / D0_B | D_fit nonlinear joint | / D0_A | / D0_B | D_fit AD | / D0_A | C_ht obs at L_f | U/D0_A |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for t in TILES:
        if t not in p["tiles"]:
            continue
        q = p["tiles"][t]["d_recovery"]
        U = p["tiles"][t]["U"]
        out.append(f"| {t} | {fmt(q['D0_A'], '.3e')} | {fmt(q['D0_B'], '.3e')} | {fmt(q.get('D_fit_linear_joint'), '.3e')} | {fmt(q.get('D_fit_linear_joint_over_D0_A'), '.2f')} | {fmt(q.get('D_fit_linear_joint_over_D0_B'), '.2f')} | "
                   f"{fmt(q.get('D_fit_nonlinear_joint'), '.3e')} | {fmt(q.get('D_fit_nonlinear_joint_over_D0_A'), '.2f')} | {fmt(q.get('D_fit_nonlinear_joint_over_D0_B'), '.2f')} | "
                   f"{fmt(q.get('D_fit_ad'), '.3e')} | {fmt(q.get('D_fit_ad_over_D0_A'), '.2f')} | {fmt(q.get('C_ht_obs_at_L_f'), '.3g')} | {fmt(U / q['D0_A'], '.3g')} |")
else:
    out.append("pending")
out.append("")

# ======================================================================================= B9 figures' inputs: sweep, d8_vs_dinf, slope-area, supp_fig5
out.append("## B9. Section 5.6: figure inputs (resolution sweep at D0_A and D0_B, d8_vs_dinf at the config parameters, slope-area at the new optima, supp_fig5)\n")
for variant in ("A", "B"):
    sw = runs("sweep_resolution", lambda n, p, r: p.get("params_source") == "config" and p.get("d0_variant", "A") == variant)
    if not sw:
        out.append(f"sweep D0_{variant}: pending\n")
        continue
    frames = [pd.read_parquet(d / "table.parquet").assign(run=d.name) for d, p, r in sw if (d / "table.parquet").exists()]
    df = pd.concat(frames).sort_values("run").drop_duplicates(subset=["tile", "dx"], keep="last")
    out.append(f"Resolution sweep at D0_{variant} (runs {', '.join(sorted({d.name for d, _, _ in sw}))}): median Ks_mod/Ks_obs by site at selected dx\n")
    out.append("| site | routing | dx 5 | 10 | 20 | 50 | 100 m | rows |")
    out.append("|---|---|---|---|---|---|---|---|")
    for site in ("AP", "GM"):
        sub = df[df.site == site]
        for key in ("ratio_d8", "ratio_dinf"):
            if key not in sub.columns:
                continue
            cells = []
            for dxv in (5.0, 10.0, 20.0, 50.0, 100.0):
                m = sub[np.isclose(sub.dx, dxv, rtol=0.07)]
                cells.append(f"{np.nanmedian(m[key]):.2f}" if len(m) else "-")
            out.append(f"| {site} | {key[6:]} | " + " | ".join(cells) + f" | {len(sub)} |")
    out.append("")
d, p, r = latest("d8_vs_dinf", lambda n, p, r: p.get("params_source") == "config")
if p:
    out.append(f"d8_vs_dinf at the config parameters, metres (run `{d.name}`, job {r.get('slurm_job_id')}):\n")
    out.append("| tile | K | D | m | A_crit | MSE AD D8 | AD D-inf | PT D8 | PT D-inf rebuilt | PT D-inf v1 | channel cells D8 / D-inf |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for t, q in p["tiles"].items():
        c = q["params"]
        out.append(f"| {t} | {c['K']:.3e} | {c['D']:.3e} | {c['m']} | {c['A_crit']:.0f} | {q['mse_ad_d8']:.5g} | {q['mse_ad_dinf']:.5g} | {q['mse_pt_d8']:.5g} | {fmt(q.get('mse_pt_dinf'), '.5g')} | "
                   f"{fmt(q.get('mse_pt_dinf_v1'), '.5g')} | {q['n_channel_d8']} / {q.get('n_channel_dinf', '-')} |")
    out.append("")
else:
    out.append("d8_vs_dinf: pending\n")
d, p, r = latest("slope_area", lambda n, p, r: isinstance(next(iter(p["tiles"].values()))["model_source"], dict))
if p:
    out.append(f"Slope-area fits at the new optima (run `{d.name}`, job {r.get('slurm_job_id')}; log-binned, v1 thresholds):\n")
    out.append("| tile | series | k_s | theta | R^2 | points |")
    out.append("|---|---|---|---|---|---|")
    for t in TILES:
        if t not in p["tiles"]:
            continue
        for key in ("orig", "AD_no_opt", "AD_opt", "part"):
            q = p["tiles"][t].get(key)
            if q:
                out.append(f"| {t} | {key} | {q['Ks']:.4g} | {q['theta']:.3f} | {q['R2']:.3f} | {q['n_points']} |")
    out.append("")
else:
    out.append("slope-area: pending\n")
d, p, r = latest("synthetic_pe")
if p:
    out.append(f"supp_fig5 regenerated: run `{d.name}` (job {r.get('slurm_job_id')}); local k_s rows {len(p.get('summary_local_ks', []))}; D_bal {p['baseline']['D_bal']:.4g}, Pe {p['baseline']['Pe_primary']:.4g}.\n")
out.append("")

# ======================================================================================= B10 old-against-new
out.append("## B10. Section 5.8: the old-against-new record\n")
rec = repo / "tables" / "old_vs_new_record.md"
if rec.exists():
    out.append(f"Built by `lemcvt phase6 record` (`tables/old_vs_new_record.csv`, {sum(1 for _ in open(repo / 'tables' / 'old_vs_new_record.csv')) - 1} rows). Tile and stage rows:\n")
    lines = rec.read_text().splitlines()
    keep = [ln for ln in lines if "sweep ratio" not in ln]
    out.extend(keep)
    n_sw = sum(1 for ln in lines if "sweep ratio" in ln)
    out.append(f"\n({n_sw} resolution-sweep ratio rows are in the CSV only.)")
else:
    out.append("pending")
out.append("")
print("\n".join(out))
