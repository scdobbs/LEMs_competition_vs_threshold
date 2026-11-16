"""Appendix tables for the Phase 6 report of addendum 5 (Phase 6.0: seam fix, option (iii) test, measured inputs, budget).

Usage: PYTHONNOUSERSITE=1 LEMCVT_RESULTS=... LEMCVT_DATA=... python reports/v2/scripts/make_phase6_tables.py > appendix.md
Reads the latest run of each step under $LEMCVT_RESULTS; sections whose runs do not exist are marked pending.
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
out: list[str] = []


def fmt(v, f=".4g"):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    return format(v, f)


def ci(v, f=".3g"):
    return f"[{fmt(v[0], f)}, {fmt(v[1], f)}]" if v else "-"


def latest(step: str):
    ds = []
    for d in sorted(root.iterdir()):
        if f"_{step}_" not in d.name or not (d / "run.json").exists():
            continue
        try:
            p = json.loads((d / "params.json").read_text())
            r = json.loads((d / "run.json").read_text())
        except Exception:  # noqa: BLE001
            continue
        if r.get("step") != step or (r.get("config") or {}).get("crop"):
            continue
        ds.append((d, p, r))
    return ds[-1] if ds else (None, None, None)


# ======================================================================================= A1 seam fix
out.append("## A1. Section 2.1: AP1 flow-direction seam corrected within the tile\n")
d, p, r = latest("seamfix")
if p:
    s, ch, chk, re_, co = p["seam"], p["changes"], p["independent_recomputation"], p["reentry"], p["correction"]
    out.append(f"Run `{d.name}` (job {r.get('slurm_job_id')}, git {r['git_hash'][:8]}).\n")
    out.append("| quantity | value |")
    out.append("|---|---|")
    out.append(f"| seam cells (interior, stored code not the steepest descent, not a tie) | {s['n_seam']} on rows {s['rows']}, columns {s['cols_min_max']} |")
    out.append(f"| metre and feet recomputations agree on the seam cells | {s['metre_feet_codes_agree_on_seam']} ({s['n_metre_feet_disagree_on_seam']} disagree) |")
    out.append(f"| stored codes on the seam (code: count) | {s['old_code_counts']} |")
    out.append(f"| corrected codes (code: count) | {s['new_code_counts']} |")
    q = s["excess_drop_best_minus_orig_m"]
    out.append(f"| steepest drop minus drop to the stored receiver (m): min / median / max | {fmt(q.get('0'), '.3g')} / {fmt(q.get('50'), '.3g')} / {fmt(q.get('100'), '.3g')} |")
    q = s["drop_to_orig_receiver_m"]
    out.append(f"| drop to the stored receiver (m): min / median / max; stored receivers uphill or level | {fmt(q.get('0'), '.3g')} / {fmt(q.get('50'), '.3g')} / {fmt(q.get('100'), '.3g')}; {s['n_orig_receiver_uphill_or_level']} |")
    out.append(f"| old paths: stopped at an uphill receiver / reached the tile exit | {co['n_old_paths_stopped_uphill']} / {co['n_old_paths_exit_tile']} (distinct exit cells {re_['n_distinct_exit_cells_old']}) |")
    out.append(f"| new paths: stopped at an uphill receiver / reached the tile exit | {co['n_new_paths_stopped_uphill']} / {co['n_new_paths_exit_tile']} (distinct exit cells {re_['n_distinct_exit_cells_new']}) |")
    out.append(f"| edge cells receiving external inflow (re-entry would appear here) | {re_['edge_cells_with_external_inflow']} |")
    out.append(f"| path correction against the external-inflow recomputation: max abs difference (cells); cells differing by > 1e-6 cell | {chk['max_abs_diff_cells']:.3g}; {chk['n_cells_differing_gt_1e-6_cell']} |")
    out.append(f"| external inflow at interior cells > 0 / negative anywhere | {chk['external_inflow_interior_cells_gt_0']} / {chk['external_inflow_negative_cells']} |")
    out.append(f"| cells whose area changes (fraction of valid cells) | {ch['n_cells_area_changed']} ({100 * ch['frac_cells_area_changed']:.2f} %); increases {ch['n_increase']}, decreases {ch['n_decrease']}; rows touched {ch['n_rows_touched']} |")
    rq = ch["rel_change_quantiles"]
    aq = ch["abs_rel_change_quantiles"]
    out.append(f"| relative area change of the changed cells: quantiles 0 / 5 / 25 / 50 / 75 / 95 / 100 % | {' / '.join(fmt(rq.get(k), '.3g') for k in ('0', '5', '25', '50', '75', '95', '100'))} |")
    out.append(f"| absolute relative change: median / 95 % / max; max absolute change (cells) | {fmt(aq.get('50'), '.3g')} / {fmt(aq.get('95'), '.3g')} / {fmt(aq.get('100'), '.3g')}; {ch['max_abs_change_cells']:.4g} |")
    out.append(f"| cells with A >= 1e3 m^2 changing by more than 1 % (of {ch['n_cells_A_ge_1e3']}) | **{ch['n_cells_A_ge_1e3_changed_gt_1pct']}** |")
    out.append(f"| total area in the tile before / after (cells) | {ch['area_total_old_cells']:.1f} / {ch['area_total_new_cells']:.1f} |")
    cs = p.get("consistency")
    if cs:
        out.append("\nConsistency of the stored grids (the premise of the correction): interior cells whose stored area is not the accumulation of their in-tile donors (|ext| > 0.5 cell):\n")
        out.append("| directions | inconsistent interior cells | rows (count per row) | abs size: median / 90 % / max (cells) | sum of positives / negatives (cells) |")
        out.append("|---|---|---|---|---|")
        for k, lab in (("stored_area_vs_stored_fd", "stored fd"), ("stored_area_vs_corrected_fd", "corrected fd")):
            q = cs[k]
            aq = q.get("abs_quantiles_cells", {})
            out.append(f"| {lab} | **{q['n_inconsistent_interior']}** | {q['per_row']} | {fmt(aq.get('50'), '.0f')} / {fmt(aq.get('90'), '.0f')} / {fmt(aq.get('100'), '.0f')} | {q['sum_positive_cells']:.0f} / {q['sum_negative_cells']:.0f} |")
        st = cs.get("elevation_step_minus_local_reference", {})
        if st:
            out.append("\nElevation drop between consecutive rows minus the local reference (the mean of the four neighbouring drops), columns 2-1012:\n")
            out.append("| rows | mean drop (m) | median excess (m) | 5 % / 95 % (m) |")
            out.append("|---|---|---|---|")
            for k, v in st.items():
                out.append(f"| {k} | {v['mean_drop_m']:.4f} | **{v['median_m']:.4f}** | {v['q5_m']:.4f} / {v['q95_m']:.4f} |")
        ca = cs["corrected_area"]
        out.append(f"\nCorrected area grid: {ca['n_below_one_cell']} cells below one cell (minimum {ca['min_cells']:.0f} cells, rows {ca['rows_below_one_cell'][:10]}); premise holds: **{cs['premise_holds']}**.")
    out.append("\nChannel masks before and after the correction:\n")
    out.append("| threshold | A_crit (m^2) | channel cells before | after | gained | lost |")
    out.append("|---|---|---|---|---|---|")
    for k, v in ch["channel_mask"].items():
        out.append(f"| {k} | {v['A_crit']:.6g} | {v['n_channel_old']} | {v['n_channel_new']} | {v['n_gained']} | {v['n_lost']} |")
    if ch.get("checkpoint_acrit"):
        out.append(f"\n(joint and stage-2 A_crit from the checkpoint run `{ch['checkpoint_acrit']['run']}`.)")
    w = p.get("written", {})
    if w.get("fd"):
        out.append("\nGrids written (originals kept):\n")
        out.append("| grid | path | sha256 (12) | bytes | round trip identical | original | original sha256 (12) |")
        out.append("|---|---|---|---|---|---|---|")
        for k in ("fd", "area"):
            v = w[k]
            out.append(f"| {k} | `{v['path']}` | {v['sha256'][:12]} | {v['bytes']} | {v['round_trip_identical']} | `{v['original']}` | {v['original_sha256'][:12]} |")
        if w.get("manifest"):
            out.append(f"\nManifest: {w['manifest']['n_entries']} entries.")
    out.append("\nFirst seam cells processed (upstream first):\n")
    out.append("| row, col | area moved (cells) | old -> new code | drop to old / new receiver (m) | old path: cells, stop | new path: cells, stop |")
    out.append("|---|---|---|---|---|---|")
    for q in co["records_first_20"][:12]:
        out.append(f"| {q['row']}, {q['col']} | {q['area_moved'] / 0.951 if q['area_moved'] else 0:.1f} | {q['old_code']} -> {q['new_code']} | {fmt(q['drop_to_old_receiver'], '.3g')} / {fmt(q['drop_to_new_receiver'], '.3g')} | "
                   f"{q['old_path']['n_cells']}, {q['old_path']['stop']} | {q['new_path']['n_cells']}, {q['new_path']['stop']} |")
else:
    out.append("pending")
out.append("")

# ======================================================================================= A2 option (iii) window test
out.append("## A2. Section 2.2: the option (iii) window test (pre-registered)\n")
d, p, r = latest("phase6_inputs")
if p:
    out.append(f"Run `{d.name}` (job {r.get('slurm_job_id')}, git {r['git_hash'][:8]}); {p['settings']['n_boot']} bootstrap draws over 8-connected hilltop segments; "
               f"L_f = 11 cells (w_f = {p['settings']['w_f']}); pass tolerance +-{100 * p['settings']['pass_tol']:.0f} %; site rule: >= {p['sites']['AP']['option_iii_test']['min_pass']} of 5 tiles.\n")
    out.append("| tile | pixels (segments) | median crest width 2d (m): 25 / 50 / 75 % | windows tested (nominal m) | C_ht(L)/C_ht(L_f) at 7 / 9 / 11 / 15 / 21 / 25 / 31 / 41 / 51 m | max deviation in the tested range | pass |")
    out.append("|---|---|---|---|---|---|---|")
    for t in TILES:
        o = p["tiles"][t]["option_iii"]
        w = o["width_2x_distance_m"]
        ratios = " / ".join(f"{q['ratio_to_L_f']:.2f}" for q in o["rows"])
        out.append(f"| {t} | {o['n_pixels']} ({o['n_segments']}) | {w['25']:.1f} / {w['50']:.1f} / {w['75']:.1f} | {o['test']['windows_tested_nominal_m']} | {ratios} | "
                   f"{100 * o['test']['max_abs_deviation']:.0f} % | **{'yes' if o['test']['pass'] else 'no'}**{' (only L_f in range)' if o['test']['only_L_f_in_range'] else ''} |")
    out.append("\nC_ht(L) with the 68 % segment-bootstrap band, per tile (1/m):\n")
    out.append("| tile | " + " | ".join(f"{q['L_nominal_m']} m" for q in p["tiles"]["AP1"]["option_iii"]["rows"]) + " |")
    out.append("|---|" + "---|" * len(p["tiles"]["AP1"]["option_iii"]["rows"]))
    for t in TILES:
        o = p["tiles"][t]["option_iii"]
        out.append(f"| {t} | " + " | ".join(f"{q['C_ht']:.3g} {ci(q.get('C_ht_ci68'), '.2g')}" for q in o["rows"]) + " |")
    out.append("\nSite result:\n")
    out.append("| site | tiles passing | rule | option (iii) stands |")
    out.append("|---|---|---|---|")
    for s_, q in p["sites"].items():
        tt = q["option_iii_test"]
        out.append(f"| {s_} | {tt['n_pass']} of {tt['n_tiles']} ({', '.join(k for k, v in tt['per_tile'].items() if v)}) | >= {tt['min_pass']} | **{'yes' if tt['pass'] else 'no'}** |")
    out.append(f"\nDecision record: option (iii) stands = **{p['decision']['option_iii_stands']}** ({p['decision']['rule']}).")
else:
    out.append("pending")
out.append("")

# ======================================================================================= A3 measured inputs
out.append("## A3. Sections 1.1-1.3: the measured inputs that section 2.3 would write (not written; see A2)\n")
if p:
    sm = p["site_m"]
    out.append("| site | m_site | S_min | S(m) 1 % interval | 5 % interval | leave-one-out m | m-profile runs |")
    out.append("|---|---|---|---|---|---|---|")
    for s_, v in sm["provenance"].items():
        out.append(f"| {s_} | **{v['m_site']}** | {fmt(v.get('S_min'), '.3f')} | {v.get('interval_1pct', {}).get('intervals')} | {v.get('interval_5pct', {}).get('intervals')} | "
                   f"{ {k: round(x, 2) for k, x in (v.get('leave_one_out_m') or {}).items()} } | {len(v.get('m_profile_runs') or {})} runs |")
    out.append("\nk_s at theta = m_site (pooled through-origin chi-z slope; basin bootstrap; n_eff centred / through-origin) and K0 = U/k_s:\n")
    out.append("| tile | m_site | k_s at 1e3 m^2 [68 %] (basins; n_eff) | audit value, same theta | K0 = U/k_s (1/yr) | k_s at 1e4 m^2 [68 %] (basins; n_eff) | K0 at 1e4 (reported, not used) | site-pooled k_s 1e3 / 1e4 |")
    out.append("|---|---|---|---|---|---|---|---|")
    for t in TILES:
        q = p["tiles"][t]
        k3, k4 = q["ks"]["1000"], q["ks"]["10000"]
        sp = p["sites"][t[:2]]["ks_site_pooled"]
        out.append(f"| {t} | {q['m_site']} | {k3['ks']:.4g} {ci(k3.get('ks_ci68'))} ({k3['n_basins']}; {fmt(k3.get('n_eff_centred'), '.1f')} / {fmt(k3.get('n_eff_through_origin'), '.1f')}) | "
                   f"{fmt(k3.get('audit_ks_same_theta'), '.4g')} | **{k3['K0']:.3e}** | {fmt(k4.get('ks'), '.4g')} {ci(k4.get('ks_ci68'))} ({k4['n_basins']}; {fmt(k4.get('n_eff_centred'), '.1f')}) | "
                   f"{fmt(k4.get('K0'), '.3e')} | {fmt((sp.get('1000') or {}).get('ks'), '.4g')} / {fmt((sp.get('10000') or {}).get('ks'), '.4g')} |")
    out.append("\nC_ht and D0 = U/C_ht by option (iii) at L_f (segment bootstrap), beside the 02b set (measure set, no exclusion, plateau window L*):\n")
    out.append("| tile | option (iii) pixels | C_ht(L_f) [68 %] (1/m) | audit value | D0 [68 %] (m^2/yr) | 02b set: pixels, L* (m), C_ht [68 %] | 02b D0 | D0 ratio option (iii) / 02b |")
    out.append("|---|---|---|---|---|---|---|---|")
    for t in TILES:
        q = p["tiles"][t]
        o, b = q["option_iii"], q["set_02b"]
        out.append(f"| {t} | {o['n_pixels']} | {o['C_ht_at_L_f']:.4g} {ci(o.get('C_ht_ci68_at_L_f'))} | {fmt(o.get('audit_C_ht_at_L_f'), '.4g')} | **{fmt(o.get('D0'), '.3e')}** {ci(o.get('D0_ci68'))} | "
                   f"{b.get('n_pixels')}, {fmt(b.get('L_star_m'), '.1f')}, {fmt(b.get('C_ht'), '.4g')} {ci(b.get('C_ht_ci68'))} | {fmt(b.get('D0'), '.3e')} | {fmt(o['D0'] / b['D0'] if o.get('D0') and b.get('D0') else None, '.2f')} |")
out.append("")

# ======================================================================================= A4 budget
out.append("## A4. Section 3.0: the core-hour estimate of the Phase 6 rerun (not submitted)\n")
try:
    from lemcvt.analyses.exponent_site import site_summary
    from lemcvt.analyses.phase6 import estimate, fill_counts_from_m_profiles

    summ = site_summary()
    m_site = {s_: float(v["all"]["m_opt"]) for s_, v in summ["sites"].items()}
    fc = fill_counts_from_m_profiles(m_site)
    est = estimate(TILES, fill_counts=fc)
    out.append(f"Limits = {est['time_factor']} x the expected wall time rounded up to the hour; expected times from the measured per-evaluation costs "
               f"(partitioned {est['cost_assumptions']['partitioned_eval']} s, AD {est['cost_assumptions']['ad_eval']} s, Picard iteration {est['cost_assumptions']['picard_iteration']} s, "
               f"D-inf LU {est['cost_assumptions']['dinf_lu']} s). Unique areas inside the 5 % intervals at the site m (joint + stage 2, mean per tile): AP {fc['AP']}, GM {fc['GM']}.\n")
    out.append("| item | tasks | CPUs | expected h per task | limit h per task | core-hours (limits) | core-hours (expected) | memory | note |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for it in est["items"]:
        out.append(f"| {it['item']} | {it['tasks']} | {it['cpus']} | {it['expected_h_per_task']} | {it['limit_h_per_task']} | {it['core_hours_limit']} | {it['core_hours_expected']} | {it['mem']} | {it['note']} |")
    out.append(f"| **total** | | | | | **{est['total_core_hours_limit']:.0f}** | **{est['total_core_hours_expected']:.0f}** | | budget {est['budget_total']:.0f} without asking; largest single item {est['max_single_item_limit']:.0f} (per-submission line {est['budget_per_submission']:.0f}) |")
    out.append("\nPer tile, the unique drainage areas inside the 5 % intervals of the m-profile at the site m (joint / stage 2 at the v1 K0, D0):\n")
    out.append("| tile | joint 5 % interval (m^2) | unique areas | stage-2 5 % interval (m^2) | unique areas |")
    out.append("|---|---|---|---|---|")
    for t in TILES:
        q = fc["per_tile"].get(t)
        if q:
            out.append(f"| {t} | {q['mse_joint']['A_min']:.0f} - {q['mse_joint']['A_max']:.0f} | {q['mse_joint']['n_unique_inside']} | {q['mse_stage2']['A_min']:.0f} - {q['mse_stage2']['A_max']:.0f} | {q['mse_stage2']['n_unique_inside']} |")
except Exception as exc:  # noqa: BLE001
    out.append(f"(estimate not available here: {exc})")
out.append("")
print("\n".join(out))
