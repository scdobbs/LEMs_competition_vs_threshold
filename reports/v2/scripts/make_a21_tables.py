#!/usr/bin/env python3
"""Addendum 21 appendices (section 7, item 11): every table from the run directory as markdown (<run>/appendix_tables.md; pasted into
the report), with run ids, checksums and the Slurm accounting. usage: make_a21_tables.py
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a21_common import HEADS_CSV, JOB_DIR, MD_TILES, REPO, TTV_FULL, TTV_PRIMARY, a21_run_dir, load_json, out_path  # noqa: E402
from a21_rules import FIXED, FREE_D, HEAD_MODELS, ci, f, table  # noqa: E402
from a21_rules import OUT as _OUT_RULES  # noqa: E402

OUT: list[str] = []


def W(s: str = "") -> None:
    OUT.append(s)


def tbl(h, rows):
    _OUT_RULES.clear()
    table(h, rows)
    OUT.extend(_OUT_RULES)
    _OUT_RULES.clear()


def load(*p):
    q = out_path(*p)
    return load_json(q) if os.path.exists(q) else None


def sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cost() -> dict:
    ids = sorted({ln.split()[0] for ln in open(os.path.join(JOB_DIR, "JOBS.txt")) if ln.strip()})
    out = subprocess.check_output(["sacct", "-j", ",".join(ids), "-X", "-P", "-n", "--format=JobID,JobName,State,ElapsedRaw,AllocCPUS,TimelimitRaw"], text=True)
    by: dict = {}
    for line in out.strip().splitlines():
        jid, name, st, el, cpus, lim = (line.split("|") + [""] * 6)[:6]
        b = by.setdefault(name, {"tasks": 0, "used": 0.0, "lim": 0.0, "states": {}, "jobs": set()})
        b["tasks"] += 1
        b["used"] += float(el or 0) * int(cpus or 0) / 3600
        b["lim"] += float(lim or 0) / 60 * int(cpus or 0)
        b["states"][st.split()[0]] = b["states"].get(st.split()[0], 0) + 1
        b["jobs"].add(jid.split("_")[0])
    rss = subprocess.check_output(["sacct", "-j", ",".join(ids), "-P", "-n", "--format=MaxRSS"], text=True).split()
    peak = max((int(v.rstrip("K")) for v in rss if v.endswith("K")), default=0)
    return {"by": by, "used": sum(b["used"] for b in by.values()), "lim": sum(b["lim"] for b in by.values()), "peak_gb": peak / 1e6, "n_jobs": len(ids)}


def main() -> None:
    G = load("grid12", "grid12.json")
    PH = load("grid12", "phase.json")
    R = load("rules", "rules.json")
    FG = load("figures", "figures.json") or {}
    T = {t: load("tiles", f"{t}.json") for t in TTV_PRIMARY}
    S = {t: load("sjc", f"{t}.json") for t in TTV_PRIMARY}
    W("## 11. Appendices: all tables, run ids, checksums, Slurm accounting")
    W()
    W(f"Run directory: `$GROUP_HOME/lemcvt_results/{os.path.basename(a21_run_dir())}/` (grid12/, tiles/, sjc/, rules/, figures/). Jobs and logs: `$SCRATCH/lemcvt_jobs/a21/`.")
    W()
    W("### F1. Inputs and checksums")
    W()
    tbl(["input", "path", "sha256"], [["Steve's 64 picks", "data/ttv/ttv_heads_manual_steve_20261119.csv", sha(HEADS_CSV)],
                                       ["TTV site config (this addendum)", "configs/sites/TTV.yaml", sha(os.path.join(REPO, "configs", "sites", "TTV.yaml"))],
                                       ["full-site 1 m filled grid", TTV_FULL["filled"].replace(os.environ.get("GROUP_HOME", ""), "$GROUP_HOME"), T["TTV2"]["inputs"]["sha256"]["filled"][:16] + "… (tile TTV2 cut; the full grid's own sha256 in the stage B record)"]])
    W("### F2. Map-scale heads (decision 1.2): the picks on each grid")
    W()
    for b, sc in G["scales"].items():
        W(f"- **{b} m grid** ({sc['shape'][0]} x {sc['shape'][1]} cells): A at the picks p25/p50/p75 {f(sc['A_pct']['p25'], '.0f')} / {f(sc['A_pct']['p50'], '.0f')} / {f(sc['A_pct']['p75'], '.0f')} m² "
          f"(cell area median {f(sc['A_cell_pct']['p50'], '.0f')}); share within Table 1's 5-95 % {f(sc['frac_within_table1_5_95'], '.2f')}; S one cell median {f(sc['S1_pct']['p50'], '.3f')} "
          f"(q25-q75 {f(sc['S1_pct']['p25'], '.3f')}-{f(sc['S1_pct']['p75'], '.3f')}), two cells {f(sc['S2_pct']['p50'], '.3f')} ({sc['n_S2_short']} chains short of two cells); S undefined on {sc['n_S1_nan']} picks"
          + (f"; D-infinity A median {f(sc['A_dinf_pct']['p50'], '.0f')} m²" if "A_dinf_pct" in sc else ""))
    W()
    tbl(["pick", "A_12 (m²)", "cell A_12", "S_12 one cell", "S_12 two cells", "A_12 D-inf", "A_10", "S_10"],
        [[q["pick_id"], f(q["A"], ".0f"), f(q["A_cell"], ".0f"), f(q["S1"], ".3f"), f(q["S2"], ".3f"), f(q.get("A_dinf"), ".0f"), f(q10["A"], ".0f"), f(q10["S1"], ".3f")]
         for q, q10 in zip(G["scales"]["12"]["picks"], G["scales"]["10"]["picks"], strict=True)])
    W("### F3. Fits and field values (decision 1.2; rule H0b)")
    W()
    tbl(["fit", "n", "a (m²)", "a 95 %", "b", "b 95 %", "R²", "median A/(1978 S^-1.65)", "share in 0.5-2", "1.65 in b 95 %", "H0b"],
        [[k, v["n_used"], f(v["a"], ".0f"), ci(v["a_ci95"], ".0f"), f(v["b"], ".3f"), ci(v["b_ci95"], ".2f"), f(v["R2"], ".2f"), f(v["ratio_to_eq10"]["median"], ".2f"),
          f(v["ratio_to_eq10"]["frac_in_0.5_2"], ".2f"), f(v["H0b"]["b_1.65_inside_ci95"]), f(v["H0b"]["consistent"])] for k, v in G["fits"].items()])
    tbl(["set", "A_F (m²)", "C_F alpha 2 (m²)", "C_F alpha 1.65 (m²)", "A_F w/o 49", "C_F w/o 49", "A·S² p25 / p75", "S nan or <= 0"],
        [[k, f(v["A_F"], ".0f"), f(v["C_F"], ".1f"), f(v["C_F_a1.65"], ".1f"), f(v["A_F_without_49"], ".0f"), f(v["C_F_without_49"], ".1f"),
          f"{f(v['AS2_pct']['p25'], '.0f')} / {f(v['AS2_pct']['p75'], '.0f')}", v["n_S_nonpositive_or_nan"]] for k, v in G["field_values"].items()])
    if PH:
        W("### F4. Lattice-phase sensitivity of the 12 m measurement (added after Part A)")
        W()
        tbl(["phase (rows, cols)", "A_F (m²)", "C_F (m²)", "C_F 1.65", "b", "b 95 %", "median ratio to eq. 10", "H0b"],
            [[str(m["offset_rc"]), f(m["A_F"], ".0f"), f(m["C_F"], ".0f"), f(m["C_F_a1.65"], ".0f"), f(m["b"], ".2f"), ci(m["b_ci95"], ".2f"), f(m["median_ratio_to_eq10"], ".2f"), f(m["H0b"])] for m in PH["phases"]] +
            [["A20 window " + str(PH["a20_window_phase"]["offset_rc"]), f(PH["a20_window_phase"]["A_F"], ".0f"), f(PH["a20_window_phase"]["C_F"], ".0f"), f(PH["a20_window_phase"]["C_F_a1.65"], ".0f"),
              f(PH["a20_window_phase"]["b"], ".2f"), ci(PH["a20_window_phase"]["b_ci95"], ".2f"), f(PH["a20_window_phase"]["median_ratio_to_eq10"], ".2f"), f(PH["a20_window_phase"]["H0b"])]])
        s = PH["summary"]
        W(f"- A_F over the 16 phases: {f(s['A_F']['min'], '.0f')}-{f(s['A_F']['max'], '.0f')} m² (median {f(s['A_F']['p50'], '.0f')}); C_F {f(s['C_F']['min'], '.0f')}-{f(s['C_F']['max'], '.0f')} m² "
          f"(median {f(s['C_F']['p50'], '.0f')}); b {f(s['b_range'][0], '.2f')}-{f(s['b_range'][1], '.2f')}; H0b consistent on {s['n_H0b_consistent']} of {s['n_phases']} phases.")
        W()
    W("### F5. The criterion networks on the full 12 m grid and per tile (decision 1.3)")
    W()
    tbl(["alpha", "C", "channel cells (12 m, whole grid)", "heads", "unique M values"], [[k, f(v["C"], ".4g"), v["n_channel_cells"], v["n_heads"], v["n_unique_M"]] for k, v in G["criterion_full_grid"].items()])
    tbl(["tile", "12 m box (rows, cols)", "candidates", "reach the tile", "max path steps", "pick seeds reaching", "α0 heads / entries / 1 m cells / 1 m heads", "α2 heads / entries / 1 m cells / 1 m heads", "mapped: entries / cells", "seed area check (max rel diff, n)"],
        [[t, f"{g['box12'][0]}-{g['box12'][1]}, {g['box12'][2]}-{g['box12'][3]}", g["n_candidates"], g["n_reach_tile"], g["max_steps"], g["n_pick_seeds_reaching_tile"],
          (lambda n: f"{n['n_head_cells_12m']} / {n['n_entries']} / {n['n_channel_1m']} / {n['n_heads_1m']}")(T[t]["networks"]["A"]) if T[t] else "-",
          (lambda n: f"{n['n_head_cells_12m']} / {n['n_entries']} / {n['n_channel_1m']} / {n['n_heads_1m']}")(T[t]["networks"]["C"]) if T[t] else "-",
          (lambda n: f"{n['n_pick_entries']} / {n['n_channel_1m']}")(T[t]["networks"]["mapped"]) if (T[t] and "mapped" in T[t]["networks"]) else "-",
          (lambda c: f"{f(c['max_rel_diff'], '.1e')}, {c['n']}")(T[t]["networks"]["seed_area_check"]) if (T[t] and T[t]["networks"].get("seed_area_check")) else "-"]
         for t, g in G["tiles"].items()])
    W("### F6. Checks: the mask route, the rescaling, the bounds")
    W()
    tbl(["tile", "mask route same set", "rel diff MSE", "max |Δb|", "P4 stored MSE", "P4 direct at new U", "rel", "AD stored MSE", "AD direct at new U", "rel", "K0 (new U)", "PF(7,400) K/K0, D at new U"],
        [[t, f(r["mask_route_check"]["same_channel_set"]), f(r["mask_route_check"]["rel_diff"], ".1e"), f(r["mask_route_check"]["max_abs_diff_b"], ".1e"), f(r["rescale_check"]["P4"]["stored_mse"], ".4f"),
          f(r["rescale_check"]["P4"]["direct_mse_new_U"], ".4f"), f(r["rescale_check"]["P4"]["rel_diff"], ".1e"), f(r["rescale_check"]["AD"]["stored_mse"], ".4f"), f(r["rescale_check"]["AD"]["direct_mse_new_U"], ".4f"),
          f(r["rescale_check"]["AD"]["rel_diff"], ".1e"), f(r["K0"], ".3e"), f"{f(r['PF_7400_new_U']['K_over_K0'], '.3f')}, {f(r['PF_7400_new_U']['D'], '.4g')}"] for t, r in T.items() if r])
    W("### F7. The rescaled addendum 18 and 20 fits (K, D x 65/77; MSE unchanged)")
    W()
    tbl(["tile", "P4 A_crit", "P4 K", "P4 D", "P4 D at bound (0.0422)", "P4 MSE", "AD K", "AD D", "AD D at bound", "AD MSE", "P3 D", "PF(7,400) A20 K", "PF(7,400) A20 D", "PF(7,400) A20 MSE"],
        [[t, f(r["rescaled"]["P4"]["A_crit"], ".0f"), f(r["rescaled"]["P4"]["K"], ".3e"), f(r["rescaled"]["P4"]["D"], ".4g"), f(r["rescaled"]["P4"]["D_at_bound_rescaled"]), f(r["rescaled"]["P4"]["mse"], ".2f"),
          f(r["rescaled"]["AD"]["K"], ".3e"), f(r["rescaled"]["AD"]["D"], ".4g"), f(r["rescaled"]["AD"]["D_at_bound_rescaled"]), f(r["rescaled"]["AD"]["mse"], ".2f"), f(r["rescaled"]["P3"]["D"], ".4g"),
          f(r["rescaled"].get("PF_7400_a20", {}).get("K"), ".3e"), f(r["rescaled"].get("PF_7400_a20", {}).get("D"), ".4g"), f(r["rescaled"].get("PF_7400_a20", {}).get("mse"), ".2f")] for t, r in T.items() if r])
    W("### F8. D values per tile (section 2.2) and the relief tracing")
    W()
    rows = []
    for t, r in T.items():
        if not r:
            continue
        for name, blk in r["D"]["relief"].items():
            if "D_relief" not in blk:
                rows.append([t, name, blk.get("pixels", ""), blk["n_pixels"], blk["n_reached"], blk["n_used"], "-", "-", "-", "-", "-"])
                continue
            rows.append([t, name, blk.get("pixels", "inside the M&D polygon"), blk["n_pixels"], blk["n_reached"], blk["n_used"], f(blk["L_H_median_m"], ".0f"), f(blk["R_median_m"], ".1f"), f(blk["Sbar_median"], ".3f"), f(blk["D_relief"]["median"], ".4f"), ci(blk["D_relief"].get("ci68"), ".4f")])
    tbl(["tile", "network", "pixels", "n", "reached", "used", "L_H median (m)", "R median (m)", "mean slope", "D_relief", "68 %"], rows)
    tbl(["tile", "C_crest", "D_form = U/C", "D_form at U = 102", "D_form at U = 77", "E_ridge/C (median E)", "E_ridge/C (mean E)", "D_form/D_phys", "U/E_ridge", "D_relief/D_form", "(D_relief/D_form)(U/E_ridge)"],
        [[t, f(r["D"]["C_crest"], ".4f"), f(r["D"]["D_form"], ".4f"), f(r["D"]["D_form_U_sens"]["sub_basin_2"], ".4f"), f(r["D"]["D_form_U_sens"]["legacy_P0"], ".4f"), f(r["D"]["D_crest_ridge_median"], ".4f"),
          f(r["D"]["D_crest_ridge_mean"], ".4f"), f(r["D"]["D_form_over_D_phys"], ".2f"), f(r["D"]["U_over_E_ridge"], ".2f"), f(r["D"]["D_relief_over_D_form"], ".2f"), f(r["D"]["decomposition"]["D_relief_over_D_form_times_U_over_E_ridge"], ".2f")] for t, r in T.items() if r])
    W("### F9. Fixed runs and the free fits per tile (U = 6.5e-5, m = 0.33; MSE in m²)")
    W()
    rows = []
    for t, r in T.items():
        if not r:
            continue
        for k, v in r["fixed"].items():
            rows.append([t, k, v["network"], f(v["D"], ".4f"), f(v["D_inside_bounds"]), v["n_channel"], f(v["mse"], ".2f"), f(v["K_free_at_this_D"]["K_over_K0"], ".2f"), f(v["K_free_at_this_D"]["mse"], ".2f")])
    tbl(["tile", "run", "network", "D", "D inside bounds", "channel cells", "MSE at K0", "K/K0 if K free at this D", "MSE then"], rows)
    rows = []
    for t, r in T.items():
        if not r:
            continue
        for k, v in r["PF"].items():
            fk = v["free_KD"]
            rows.append([t, k, f(v["C"], ".4g"), v["n_channel"], v["n_channel_no_chi"], f(fk["K"], ".3e"), f(fk["K_over_K0"], ".2f"), f(fk["D"], ".4g"), fk["where"], f(fk["mse"], ".2f"), f(fk.get("mse_heldout_mean"), ".2f"),
                         f(v["K_fixed"]["D"], ".4g"), f(v["K_fixed"]["mse"], ".2f"), f(v["m_0.30_sensitivity"]["free_KD"]["K"], ".3e"), f(v["m_0.30_sensitivity"]["free_KD"]["D"], ".4g"), f(v["m_0.30_sensitivity"]["free_KD"]["mse"], ".2f")])
    tbl(["tile", "fit", "C", "channel cells", "no chi", "K", "K/K0", "D", "where", "MSE", "held-out mean", "D (K = K0)", "MSE (K = K0)", "K at m 0.30", "D at m 0.30", "MSE at m 0.30"], rows)
    W("### F10. SJ-C per tile")
    W()
    tbl(["tile", "C range", "unique values", "grid", "refine (targets / evaluated)", "evaluations", "s per evaluation", "C opt", "12 m heads", "channel cells", "K", "K/K0", "D", "where", "MSE", "1 % interval of C", "held-out mean", "monotone n(C)"],
        [[t, f"{f(s['setup']['C_range'][0], '.4g')}-{f(s['setup']['C_range'][1], '.4g')}", s["setup"]["n_unique_in_range"], s["setup"]["n_grid"], f"{s['refine']['n_targets']} / {s['refine']['n_evaluated']}", s["n_evaluations"],
          f(s["s_per_evaluation"], ".1f"), f(s["joint_optimum"]["C"], ".4g"), s["joint_optimum"]["n_heads_12m"], s["joint_optimum"]["n_channel"], f(s["joint_optimum"]["K"], ".3e"), f(s["joint_optimum"]["K_over_K0"], ".2f"),
          f(s["joint_optimum"]["D"], ".4g"), s["joint_optimum"]["where"], f(s["joint_optimum"]["mse"], ".2f"), str([[round(a, 1), round(b, 1)] for a, b in s["identifiability"]["level_0.01"]["intervals"]]),
          f((s.get("holdout") or {}).get("mse_heldout_mean"), ".2f"), f(s["setup"]["monotone_check"])] for t, s in S.items() if s])
    W("### F11. Head extraction (decision 1.4) on every surface, per tile")
    W()
    rows = []
    for t in TTV_PRIMARY:
        r = T.get(t)
        if not r:
            continue
        ex = dict(r["extraction"])
        if S.get(t):
            ex["SJ_C"] = S[t]["extraction"]
        for name in HEAD_MODELS + ["obs_full_lattice"]:
            e = ex.get(name)
            if not e:
                continue
            for key in ("a2", "a0", "a1.65", "perron"):
                m = e.get(key)
                if not m:
                    continue
                d = m["pick_dist_m"]
                rows.append([t, name, key, m.get("n_channel_12m", m.get("n_skeleton_cells")), m.get("n_heads_12m", m.get("n_heads")), m["n_heads_in_polygon"], m["n_picks_in_polygon"], f(m["ratio_heads_to_picks"], ".2f"),
                             f(d["median"], ".1f"), f(d.get("frac_within_25m"), ".2f"), f(m.get("frac_line_within_15m_of_channel"), ".2f"), f(m.get("frac_channel_within_15m_of_line"), ".2f"),
                             m["H"]["reading"] if m.get("H") else "-"])
    tbl(["tile", "surface", "set", "channel cells / skeleton px", "heads", "heads in polygon", "picks in polygon", "ratio", "median pick dist (m)", "share <= 25 m", "line share <= 15 m", "channel share <= 15 m of line", "reading"], rows)
    rows = []
    for t in TTV_PRIMARY:
        r = T.get(t)
        if not r:
            continue
        for name in HEAD_MODELS:
            e = r["extraction"].get(name) if name != "SJ_C" else (S[t]["extraction"] if S.get(t) else None)
            if not e:
                continue
            rs = e.get("residual")
            rows.append([t, name, e["n_valid_12m"], e.get("n_holes_1m"), f(rs["mse"], ".2f") if rs else "-", f(rs["mean"], ".2f") if rs else "-", f(rs["q05"], ".1f") if rs else "-", f(rs["q95"], ".1f") if rs else "-", f(e["wall_s"], ".0f")])
    tbl(["tile", "surface", "12 m cells", "1 m holes (channel cells without chi)", "residual MSE", "mean", "q05", "q95", "wall (s)"], rows)
    if R:
        W("### F12. Rule tables (section 6; from rules/tables.md)")
        W()
        with open(out_path("rules", "tables.md")) as fh:
            OUT.extend(fh.read().splitlines())
        W()
    W("### F13. Figures")
    W()
    for k, v in FG.items():
        W(f"- `{v['path']}` ({v['bytes'] / 1e6:.2f} MB): {k}")
    if G and G.get("figure"):
        W(f"- `{G['figure']['path']}` ({G['figure']['bytes'] / 1e6:.2f} MB): map-scale area against slope at the picks")
    W()
    W("### F14. Slurm accounting")
    W()
    c = cost()
    tbl(["job name", "tasks", "core-h used", "core-h of limits", "states", "job ids"],
        [[k, v["tasks"], f(v["used"], ".2f"), f(v["lim"], ".1f"), ", ".join(f"{s} {n}" for s, n in v["states"].items()), " ".join(sorted(v["jobs"]))] for k, v in sorted(c["by"].items())])
    W(f"Core-hours used: **{c['used']:.1f}**; CPU-weighted limits: {c['lim']:.1f}; peak MaxRSS {c['peak_gb']:.1f} GB; {c['n_jobs']} submissions.")
    with open(out_path("appendix_tables.md"), "w") as fh:
        fh.write("\n".join(OUT) + "\n")
    print(len(OUT), "lines ->", out_path("appendix_tables.md"), f"used {c['used']:.1f} core-h, limits {c['lim']:.1f}")
    _ = (np, MD_TILES, FIXED, FREE_D)


if __name__ == "__main__":
    main()
