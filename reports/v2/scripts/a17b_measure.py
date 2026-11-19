#!/usr/bin/env python3
"""Addendum 17 Part B, sections 5-6 of the directive and the go-ahead: the per-tile measurement, after the m-profile runs.

Everything is computed with the v2 functions (docs/measurement_methods.md section 0): `exponent_site.site_profile` for S(m),
`phase6_inputs.ks_at_theta` (pooled through-origin k_s at theta = site m over the network above A_c, the audit's basin filters,
n_eff, basin bootstrap), `phase6_inputs.option_iii_mask` + `window_test` (C_ht(L) on the option (iii) set, L_f = 11 m, the
pre-registered window-stability rule, D0 = E/C_ht), `audits.basin_moments` + `estimators` for the per-tile and site-pooled theta
(pooled and equal-weight, basin bootstrap 95 %) at A_c = 1e3 and 1e4 m2. Site m = argmin S(m) over the primary set (rule fixed in
the directive); the clean subset (primary minus Tier 2), the NCALM GM1-GM7 and the PAMAP AP1-AP5 are measured at the same site m and
profiled beside it. Rules 6.1-6.4 of the directive and go-ahead 2.3-2.4 are evaluated and written to measurement.json.
usage: a17b_measure.py --out DIR --primary primary_set.json [--n-boot 200] [--data $LEMCVT_DATA]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a17b_common import (  # noqa: E402
    A_C_KS,
    A_C_THETA,
    AP_PAMAP,
    E_M_PER_YR,
    GM_NCALM,
    L_F_M,
    MEASURED,
    RULE_62_TOL,
    RULE_64_MEDIAN_TOL,
    RULE_64_SINGLE_TOL,
    TIER1,
    TIER2,
    V2_SITE_M,
    WINDOW_TEST_PASS_FRAC,
    dump,
    site_of,
    slurm_record,
)

from lemcvt.analyses.exponent_site import latest_m_profile_runs, site_profile  # noqa: E402
from lemcvt.measure.audits import basin_moments, concat_moments, estimators  # noqa: E402
from lemcvt.measure.chi import channel_network  # noqa: E402
from lemcvt.measure.concavity import DEFAULT_THETAS  # noqa: E402
from lemcvt.measure.diagnostics import edge_outlet_labels  # noqa: E402
from lemcvt.measure.phase6_inputs import (  # noqa: E402
    W_F,
    _concat,
    _ks_from_moments,
    _measure_curvature_params,
    ks_at_theta,
    option_iii_mask,
    window_test,
)
from lemcvt.measure.pipeline import _metre_tile  # noqa: E402
from lemcvt.results import find_runs  # noqa: E402

ALL_PROFILED = None   # set in main


def latest_measure_params() -> dict[str, dict]:
    out = {}
    for r in find_runs("measure"):
        try:
            p = json.loads((r / "params.json").read_text())
        except Exception:  # noqa: BLE001
            continue
        for t, v in p.get("tiles", {}).items():
            out[t] = dict(v) | {"_run": r.name}
    return out


def theta_rows(fit, key):
    e = fit.get(key) or {}
    return {f"theta_{key}": e.get("theta_star"), f"theta_{key}_lo95": (e.get("theta_ci95") or [None, None])[0], f"theta_{key}_hi95": (e.get("theta_ci95") or [None, None])[1],
            f"ks_at_theta_{key}": e.get("ks", e.get("ks_pooled_at_theta")), f"n_eff_{key}": e.get("n_eff_centred", e.get("n_eff"))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--primary", required=True)
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument("--data", default=os.environ.get("LEMCVT_DATA"))
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    slurm_record(a.out)
    t_all = time.time()
    prim = json.load(open(a.primary))
    primary = prim["primary"]
    clean = prim["clean_subset"]
    groups = {"GM": {"primary": primary["GM"], "clean": clean["GM"], "ncalm": GM_NCALM},
              "AP": {"primary": primary["AP"], "clean": clean["AP"], "pamap": AP_PAMAP},
              "TTV": {"primary": primary["TTV"], "clean": clean["TTV"]}}
    # ------------------------------------------------------------- A. S(m) per site and group, the site m
    profiled = sorted(set(MEASURED) - {"AP1_3dep"} | set(primary["GM"]) | set(primary["AP"]) | set(primary["TTV"]))
    runs = latest_m_profile_runs(tiles=tuple(set(MEASURED)), m_min_required=0.05)
    missing = [t for t in profiled if t not in runs]
    print("m-profile runs:", len(runs), "missing:", missing, flush=True)
    sm = {}
    site_m = {}
    for site, g in groups.items():
        sm[site] = {}
        for gname, tiles in g.items():
            tl = [t for t in tiles if t in runs]
            if len(tl) < 2:
                sm[site][gname] = {"tiles": tl, "note": "fewer than 2 profiled tiles"}
                continue
            prof = site_profile(runs, tl)
            prof["runs"] = {t: runs[t][0].name for t in tl}
            if gname == "primary":
                prof["leave_one_out"] = {t: site_profile(runs, [u for u in tl if u != t])["m_opt"] for t in tl}
                prof["tile_optima_m"] = {t: runs[t][1]["v2"]["m_profile"]["optimum"]["m"] for t in tl}
                prof["tile_optima_A_crit"] = {t: runs[t][1]["v2"]["m_profile"]["optimum"]["A_crit"] for t in tl}
                row_at = {}
                for t in tl:
                    pm = runs[t][1]["v2"]["m_profile"]["profile_m"]
                    r = min(pm, key=lambda q: abs(q["m"] - prof["m_opt"]))
                    row_at[t] = {"A_crit": r["A_crit"], "K": r["K"], "D": r["D"], "mse": r["mse"]}
                prof["at_site_m"] = row_at
            prof.pop("per_tile_normalised", None) if gname != "primary" else None
            sm[site][gname] = prof
            print(site, gname, len(tl), "tiles: m_opt", prof["m_opt"], "1 %", prof["interval_0.01"]["intervals"], "5 %", prof["interval_0.05"]["intervals"], flush=True)
        site_m[site] = float(sm[site]["primary"]["m_opt"])
    # rule 6.2 and go-ahead 2.4
    rule62 = {}
    for site in ("GM", "AP"):
        v2 = V2_SITE_M[site]
        rec = {"v2_m": v2, "primary_m": site_m[site], "primary_moved_gt_0.05": bool(abs(site_m[site] - v2) > RULE_62_TOL)}
        if site == "GM":
            rec["ncalm_m"] = sm["GM"]["ncalm"].get("m_opt")
            rec["ncalm_moved_gt_0.05"] = bool(rec["ncalm_m"] is not None and abs(rec["ncalm_m"] - v2) > RULE_62_TOL)
        if site == "AP":
            rec["pamap_m_recomputed"] = sm["AP"]["pamap"].get("m_opt")
        rule62[site] = rec
    # ------------------------------------------------------------- B. per tile at the site m
    curv = _measure_curvature_params()
    mparams = latest_measure_params()
    thetas = DEFAULT_THETAS
    per = {}
    moments = {}
    tmom = {}
    rows = []
    curve_rows = []
    for t in MEASURED:
        t0 = time.time()
        site = site_of(t)
        td = _metre_tile(t)
        Z, A, FD, dx, dy = td.Z, td.A, td.FD, td.dx, td.filled.dy
        E = float(td.site.U)
        assert abs(E - E_M_PER_YR[site]) < 1e-12, (t, E)
        m_site = site_m[site]
        labels = edge_outlet_labels(A, FD)
        out = {"tile": t, "site": site, "E": E, "m_site": m_site, "inputs": td.provenance(),
               "group": ("tier1_continuity" if t in TIER1 else "pamap" if t in AP_PAMAP else "ncalm" if t in GM_NCALM else
                         "primary" if t in primary[site] else "screened"), "tier2": t in TIER2}
        # k_s at the site m
        moments[t] = {}
        out["ks"] = {}
        for a_c in (1e3, 1e4):
            r = ks_at_theta(Z, A, FD, dx, dy, m_site, a_c, labels, n_boot=a.n_boot)
            moments[t][a_c] = r.pop("_moments")
            r["K0"] = float(E / r["ks"]) if r.get("ks") else None
            out["ks"][f"{a_c:g}"] = r
        # per-tile theta (pooled and equal-weight) at 1e3 and 1e4
        out["theta"] = {}
        tmom[t] = {}
        for a_c in A_C_THETA:
            net = channel_network(Z, A, FD, a_c, dx, dy)
            mom = basin_moments(net, thetas, labels)
            tmom[t][a_c] = mom
            est = estimators(mom, thetas, n_boot=a.n_boot)
            out["theta"][f"{a_c:g}"] = {k: (v if k not in ("pooled", "equal", "median") or v is None else {kk: vv for kk, vv in v.items() if kk not in ("ssr_total", "ks_at_theta", "objective", "theta_per_basin")}) for k, v in est.items()}
        mp = mparams.get(t, {})
        out["measure_run"] = {"run": mp.get("_run"), "A_c_rollover": (mp.get("thresholds") or {}).get("A_c_primary"),
                              "theta_rollover": ((mp.get("chi_fits") or {}).get("primary") or {}).get("theta_star"),
                              "theta_1e3_fit_chi": ((mp.get("chi_fits") or {}).get("1000") or {}).get("theta_star"),
                              "theta_1e4_fit_chi": ((mp.get("chi_fits") or {}).get("10000") or {}).get("theta_star")}
        # option (iii) and the window test
        cp = curv.get(t)
        if not cp:
            raise FileNotFoundError(f"{t}: no measure run with a rollover A_c")
        mask, mask0, dist, diag = option_iii_mask(Z, A, FD, dx, dy, float(cp["A_c"]), W_F)
        wt = window_test(Z, mask, dist, dx, dy, E, w_f=W_F, n_boot=a.n_boot)
        wt["set"] = diag
        tested = [r for r in wt["rows"] if r["in_test_range"]]
        lw = max(tested, key=lambda r: r["L_nominal_m"]) if tested else next(r for r in wt["rows"] if r["w"] == W_F)
        wt["L_w_m"] = lw["L_nominal_m"]
        wt["C_ht_at_L_w"] = lw["C_ht"]
        wt["C_ht_ci68_at_L_w"] = lw.get("C_ht_ci68")
        wt["D0_B"] = float(E / lw["C_ht"]) if lw["C_ht"] and lw["C_ht"] > 0 else None
        out["option_iii"] = wt
        for r in wt["rows"]:
            curve_rows.append({"tile": t, "site": site, "group": out["group"], "w": r["w"], "L_m": r["L_nominal_m"], "C_ht": r["C_ht"], "n_pixels": r["n_pixels"],
                               "lo68": (r.get("C_ht_ci68") or [None, None])[0], "hi68": (r.get("C_ht_ci68") or [None, None])[1], "ratio_to_L_f": r["ratio_to_L_f"], "in_test_range": r["in_test_range"]})
        out["wall_s"] = time.time() - t0
        per[t] = out
        k3, k4 = out["ks"]["1000"], out["ks"]["10000"]
        th3, th4 = out["theta"]["1000"], out["theta"]["10000"]
        rows.append({"tile": t, "site": site, "group": out["group"], "tier2": t in TIER2, "m_site": m_site, "E": E,
                     **theta_rows(th3, "pooled"), **{k.replace("pooled", "equal"): v for k, v in theta_rows(th3, "equal").items()},
                     "theta_1e4_pooled": (th4.get("pooled") or {}).get("theta_star"), "theta_1e4_equal": (th4.get("equal") or {}).get("theta_star"),
                     "n_basins_1e3": th3.get("n_basins"), "n_basins_1e4": th4.get("n_basins"), "theta_rollover": out["measure_run"]["theta_rollover"], "A_c_rollover": float(cp["A_c"]),
                     "ks_1e3": k3.get("ks"), "ks_1e3_lo95": (k3.get("ks_ci95") or [None, None])[0], "ks_1e3_hi95": (k3.get("ks_ci95") or [None, None])[1], "n_eff_1e3": k3.get("n_eff_centred"),
                     "n_basins_ks_1e3": k3.get("n_basins"), "K0": k3.get("K0"), "ks_1e4": k4.get("ks"), "K0_1e4": k4.get("K0"), "n_eff_1e4": k4.get("n_eff_centred"),
                     "C_ht_Lf": wt["C_ht_at_L_f"], "C_ht_Lf_lo95": (wt.get("C_ht_ci95_at_L_f") or [None, None])[0], "C_ht_Lf_hi95": (wt.get("C_ht_ci95_at_L_f") or [None, None])[1],
                     "D0_A": wt["D0"], "L_w_m": wt["L_w_m"], "C_ht_Lw": wt["C_ht_at_L_w"], "D0_B": wt["D0_B"], "n_hilltop_px": wt["n_pixels"], "n_segments": wt["n_segments"],
                     "median_crest_width_m": wt["test"]["median_crest_width_m"], "windows_tested": len(wt["test"]["windows_tested_nominal_m"]), "max_abs_dev": wt["test"]["max_abs_deviation"],
                     "window_pass": wt["test"]["pass"], "only_Lf_in_range": wt["test"]["only_L_f_in_range"], "wall_s": out["wall_s"]})
        print(t, site, out["group"], f"theta3 {rows[-1]['theta_pooled']} ks3 {k3.get('ks')} n_eff {k3.get('n_eff_centred')} K0 {k3.get('K0')} C_ht {wt['C_ht_at_L_f']:.4g} D0A {wt['D0']} "
              f"Lw {wt['L_w_m']} D0B {wt['D0_B']} pass {wt['test']['pass']} ({out['wall_s']:.0f} s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "per_tile.csv"), index=False)
    pd.DataFrame(curve_rows).to_csv(os.path.join(a.out, "window_curves.csv"), index=False)
    # ------------------------------------------------------------- C. site summaries per group
    def summarise(site, tiles, name):
        tl = [t for t in tiles if t in per]
        d = df[df.tile.isin(tl)]
        s = {"tiles": tl, "n": len(tl), "m_site": site_m[site]}
        for a_c in (1e3, 1e4):
            parts = [moments[t][a_c] for t in tl if moments[t][a_c]["n"].size]
            s[f"ks_site_pooled_{a_c:g}"] = _ks_from_moments(_concat(parts), site_m[site], a_c, a.n_boot) if parts else None
            if s[f"ks_site_pooled_{a_c:g}"] and s[f"ks_site_pooled_{a_c:g}"].get("ks"):
                s[f"K0_site_pooled_{a_c:g}"] = float(E_M_PER_YR[site] / s[f"ks_site_pooled_{a_c:g}"]["ks"])
        for a_c in A_C_THETA:
            parts = [tmom[t][a_c] for t in tl if tmom[t][a_c]["n"].size]
            est = estimators(concat_moments(parts), thetas, n_boot=a.n_boot) if parts else None
            s[f"theta_site_{a_c:g}"] = {k: (v if k not in ("pooled", "equal", "median") or v is None else {kk: vv for kk, vv in v.items() if kk not in ("ssr_total", "ks_at_theta", "objective", "theta_per_basin")}) for k, v in est.items()} if est else None
        for col in ("K0", "D0_A", "D0_B", "C_ht_Lf", "ks_1e3", "theta_pooled"):
            v = d[col].astype(float)
            s[f"{col}_median"] = float(v.median()) if v.notna().any() else None
            s[f"{col}_min"] = float(v.min()) if v.notna().any() else None
            s[f"{col}_max"] = float(v.max()) if v.notna().any() else None
        n_pass = int(d["window_pass"].sum())
        s["window_test"] = {"n_tiles": len(tl), "n_pass": n_pass, "frac_pass": n_pass / len(tl) if tl else None, "pass_rule": f">= {WINDOW_TEST_PASS_FRAC:.0%} of the tiles (addendum 6 section 1: 4 of 5)",
                            "stands": bool(tl and n_pass / len(tl) >= WINDOW_TEST_PASS_FRAC), "per_tile": {t: bool(per[t]["option_iii"]["test"]["pass"]) for t in tl},
                            "trivial_only_Lf": [t for t in tl if per[t]["option_iii"]["test"]["only_L_f_in_range"]]}
        s["D0_status"] = "measurement (D0_A)" if s["window_test"]["stands"] else "bracket [D0_A, D0_B]"
        return s
    sites = {}
    for site, g in groups.items():
        sites[site] = {gname: summarise(site, tiles, gname) for gname, tiles in g.items()}
        sites[site]["S_m"] = sm[site]
    # ------------------------------------------------------------- D. rule 6.3 (AP PAMAP vs 3DEP) and 6.4 (GM FEMA vs NCALM)
    def ratio(x, y):
        return float(x / y) if x and y else None
    r63 = []
    for n in range(1, 6):
        pa, de = per.get(f"AP{n}"), per.get(f"AP{n}_3dep")
        if pa and de:
            r63.append({"tile": f"AP{n}", "ks_pamap": pa["ks"]["1000"]["ks"], "ks_3dep": de["ks"]["1000"]["ks"], "ks_ratio_3dep_over_pamap": ratio(de["ks"]["1000"]["ks"], pa["ks"]["1000"]["ks"]),
                        "C_ht_pamap": pa["option_iii"]["C_ht_at_L_f"], "C_ht_3dep": de["option_iii"]["C_ht_at_L_f"], "C_ht_ratio": ratio(de["option_iii"]["C_ht_at_L_f"], pa["option_iii"]["C_ht_at_L_f"]),
                        "theta_pamap": (pa["theta"]["1000"].get("pooled") or {}).get("theta_star"), "theta_3dep": (de["theta"]["1000"].get("pooled") or {}).get("theta_star"),
                        "theta_ratio": ratio((de["theta"]["1000"].get("pooled") or {}).get("theta_star"), (pa["theta"]["1000"].get("pooled") or {}).get("theta_star")),
                        "excluded": f"AP{n}_3dep" in TIER1})
    r64 = []
    for n in range(1, 8):
        nc, fe = per.get(f"GM{n}"), per.get(f"GM{n}_fema")
        if nc and fe:
            r64.append({"tile": f"GM{n}", "ks_ncalm": nc["ks"]["1000"]["ks"], "ks_fema": fe["ks"]["1000"]["ks"], "ks_ratio_fema_over_ncalm": ratio(fe["ks"]["1000"]["ks"], nc["ks"]["1000"]["ks"]),
                        "C_ht_ncalm": nc["option_iii"]["C_ht_at_L_f"], "C_ht_fema": fe["option_iii"]["C_ht_at_L_f"], "C_ht_ratio": ratio(fe["option_iii"]["C_ht_at_L_f"], nc["option_iii"]["C_ht_at_L_f"]),
                        "theta_ncalm": (nc["theta"]["1000"].get("pooled") or {}).get("theta_star"), "theta_fema": (fe["theta"]["1000"].get("pooled") or {}).get("theta_star")})
    def crit(rows, key):
        v = np.array([r[key] for r in rows if r[key] is not None])
        return {"median_ratio": float(np.median(v)) if v.size else None, "median_within_15pct": bool(v.size and abs(np.median(v) - 1) <= RULE_64_MEDIAN_TOL),
                "max_single_abs_dev": float(np.max(np.abs(v - 1))) if v.size else None, "all_within_30pct": bool(v.size and np.all(np.abs(v - 1) <= RULE_64_SINGLE_TOL))}
    rule64 = {"rows": r64, "ks": crit(r64, "ks_ratio_fema_over_ncalm"), "C_ht": crit(r64, "C_ht_ratio")}
    rule64["criteria_met"] = bool(rule64["ks"]["median_within_15pct"] and rule64["C_ht"]["median_within_15pct"] and rule64["ks"]["all_within_30pct"] and rule64["C_ht"]["all_within_30pct"])
    rule64["note"] = "report only (go-ahead 2.3): the primary GM set is a single survey; no pooling decision follows"
    rule63 = {"rows": r63, "ks": crit(r63, "ks_ratio_3dep_over_pamap"), "C_ht": crit(r63, "C_ht_ratio"), "theta": crit(r63, "theta_ratio"), "note": "report only (rule 6.3); AP1 is Tier 1 (continuity only)"}
    rec = {"primary_set_file": a.primary, "primary": primary, "clean_subset": clean, "site_m": site_m, "rule_62": rule62, "rule_61": {s: {g: sites[s][g]["window_test"] | {"D0_status": sites[s][g]["D0_status"]} for g in groups[s]} for s in groups},
           "rule_63": rule63, "rule_64": rule64, "sites": sites, "tiles": per, "m_profile_runs": {t: r.name for t, (r, _p) in runs.items()}, "m_profile_missing": missing,
           "settings": {"thetas": thetas.tolist(), "n_boot": a.n_boot, "A_c_ks": A_C_KS, "A_c_theta": list(A_C_THETA), "L_f_m": L_F_M, "w_f": W_F, "window_pass_frac": WINDOW_TEST_PASS_FRAC,
                        "E": E_M_PER_YR, "v2_site_m": V2_SITE_M}, "wall_s": time.time() - t_all}
    dump(rec, os.path.join(a.out, "measurement.json"))
    # ------------------------------------------------------------- E. figures: S(m) and C_ht(L) per site
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures", "a17b")
        os.makedirs(fdir, exist_ok=True)
        for site, g in groups.items():
            fig, ax = plt.subplots(figsize=(7, 4.5), dpi=200)
            cols = {"primary": "#1f77b4", "clean": "#2ca02c", "ncalm": "#7b2cbf", "pamap": "#ff7f0e"}
            for gname in g:
                s = sm[site].get(gname) or {}
                if "m" not in s:
                    continue
                m, S = np.array(s["m"]), np.array(s["S"]) / s["n_tiles"]
                ax.plot(m, S, color=cols[gname], lw=1.6, label=f"{gname} (n = {s['n_tiles']}): m = {s['m_opt']:.2f}, 1 % [{s['interval_0.01']['m_min']:.2f}, {s['interval_0.01']['m_max']:.2f}]")
                if gname == "primary":
                    for _t, c in s["per_tile_normalised"].items():
                        ax.plot(m, np.array(c), color="grey", lw=0.4, alpha=0.5)
            ax.axvline(site_m[site], color="k", lw=0.8, ls="--")
            if site in V2_SITE_M:
                ax.axvline(V2_SITE_M[site], color="r", lw=0.8, ls=":", label=f"v2 site m {V2_SITE_M[site]:.2f}")
            ax.set_ylim(0.95, 2.0)
            ax.set_xlabel("m")
            ax.set_ylabel("S(m) / n  (mean of MSE*_t(m) / MSE*_t,min)")
            ax.set_title(f"{site}: site exponent profile S(m) (grey: single tiles of the primary set)", fontsize=9)
            ax.legend(fontsize=6.5)
            for ext in ("png", "pdf"):
                fig.savefig(os.path.join(fdir, f"{site}_S_m.{ext}"), bbox_inches="tight")
            plt.close(fig)
            fig, ax = plt.subplots(figsize=(7, 4.5), dpi=200)
            for t in g["primary"]:
                rws = [r for r in curve_rows if r["tile"] == t]
                if not rws:
                    continue
                L = [r["L_m"] for r in rws]
                rr = [r["ratio_to_L_f"] for r in rws]
                ax.plot(L, rr, color="#1f77b4" if per[t]["option_iii"]["test"]["pass"] else "#d62728", lw=0.8, alpha=0.8, marker=".", ms=3)
                ax.plot([per[t]["option_iii"]["test"]["median_crest_width_m"]] * 2, [0.5, 0.55], color="grey", lw=0.5)
            ax.axhline(1.1, color="k", lw=0.6, ls="--")
            ax.axhline(0.9, color="k", lw=0.6, ls="--")
            ax.axvline(L_F_M, color="k", lw=0.6)
            ax.set_xscale("log")
            ax.set_ylim(0.4, 2.0)
            ax.set_xlabel("window L (m, nominal)")
            ax.set_ylabel("C_ht(L) / C_ht(L_f)")
            ax.set_title(f"{site} primary tiles: option (iii) C_ht(L) relative to L_f = 11 m (blue: window test passes, red: fails; ticks: median crest width)", fontsize=8)
            for ext in ("png", "pdf"):
                fig.savefig(os.path.join(fdir, f"{site}_C_ht_L.{ext}"), bbox_inches="tight")
            plt.close(fig)
        print("figures written", flush=True)
    except Exception as e:  # pragma: no cover
        print("figures failed", repr(e), flush=True)
    print("done", round(time.time() - t_all), "s", flush=True)


if __name__ == "__main__":
    main()
