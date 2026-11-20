#!/usr/bin/env python3
"""Addendum 19, Part A site level (sections 2.1, 2.4-2.8; rules R1, R11, R12; the bracket of decision 12): aggregates <run>/hilltops/*.

Per tile: the 2.1 table. Per site and group: the 2.4 summaries. 2.5: the pooled |C_ht| of the primary tiles against Perron et al. (2012)
at L_f, at each tile's L_w and at every window (segment bootstrap over the pooled segments), the legacy (approximate) manuscript route, rule R11.
2.6: sigma_C(L) normalised at 11 m, pooled per site, a two-segment continuous piecewise-linear fit in log-log with the break searched on a fine
grid, bootstrap over tiles (2,000 draws, seed 20261120); D0_N_S at the odd window nearest L_break; rule R12. 2.7: D_R, D_R/D_C, E* at
S_c = 1.27 and at the site's fitted S_c (er_star.fit_S_c on the pooled bracket pixels with the segment bootstrap), the addendum 8 medians.
2.8: the TTV anchor ratios. The bracket [D_C,lo, D_R,hi] per tile (decision 12, fixed here before any fit is classified).
usage: a19_site.py
"""
from __future__ import annotations

import glob
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a19_common import (  # noqa: E402
    ALL_HALF_WIDTHS,
    AP1_3DEP,
    AP_PAMAP,
    CLEAN,
    D0_MIN_PIXELS,
    E_SITE,
    E_STAR_BANDS,
    GM_NCALM,
    GM_OFFMESA,
    GROUPS,
    L_BREAK_LITERATURE_M,
    L_F_M,
    MEASURED_TILES,
    N_BOOT,
    PERRON_2012,
    PRIMARY,
    R1_MIN_FRAC_TILES,
    R11_RANGE,
    RESULTS_ROOT,
    SCAN_HALF_WIDTHS,
    SEED,
    TEST_HALF_WIDTHS,
    TTV_ANCHOR,
    W_F,
    dump,
    fnum,
    hold_n,
    load_json,
    out_path,
    partb_per_tile,
    slurm_record,
    weighted_median,
)

from lemcvt.analyses.er_star import S_C_REFERENCE, fit_S_c  # noqa: E402

A8_ER_STAR = os.path.join(RESULTS_ROOT, "20261113T015030Z_er_star_10tiles", "params.json")
SITE_GROUPS = {"GM": ["primary", "clean", "offmesa", "ncalm"], "AP": ["primary", "clean", "pamap", "ap1_3dep"], "TTV": ["primary", "clean"]}


def tiles_of(site, group):
    if group == "ap1_3dep":
        return AP1_3DEP
    return GROUPS[(site, group)] if (site, group) in GROUPS else {"ncalm": GM_NCALM, "pamap": AP_PAMAP, "offmesa": GM_OFFMESA}[group]


def med(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else None


def rng(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return [float(min(v)), float(max(v))] if v else None


def pooled_median_boot(vals: np.ndarray, seg: np.ndarray, n_boot: int, seed: int = SEED) -> dict:
    ok = np.isfinite(vals)
    vals, seg = vals[ok], seg[ok]
    if vals.size == 0:
        return {"median": None, "n": 0}
    order = np.argsort(vals, kind="stable")
    vs = vals[order]
    uniq, inv = np.unique(seg[order], return_inverse=True)
    rec = {"median": float(np.median(vals)), "n": int(vals.size), "n_segments": int(uniq.size)}
    if n_boot > 0 and uniq.size >= 5:
        r = np.random.default_rng(seed)
        bs = np.empty(n_boot)
        for i in range(n_boot):
            counts = np.bincount(r.integers(0, uniq.size, uniq.size), minlength=uniq.size).astype(float)
            w = counts[inv]
            cw = np.cumsum(w)
            bs[i] = vs[np.searchsorted(cw, 0.5 * cw[-1])] if cw[-1] > 0 else np.nan
        rec["ci68"] = [float(np.nanpercentile(bs, 16)), float(np.nanpercentile(bs, 84))]
        rec["ci95"] = [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]
    return rec


def piecewise_fit(x: np.ndarray, y: np.ndarray, breaks: np.ndarray) -> dict:
    """Continuous two-segment linear fit y = a + b1 (x - xb) [x <= xb], a + b2 (x - xb) [x > xb]; xb by least squares over ``breaks``."""
    best = None
    for xb in breaks:
        left = x <= xb
        if left.sum() < 2 or (~left).sum() < 2:
            continue
        X = np.column_stack([np.ones_like(x), np.where(left, x - xb, 0.0), np.where(~left, x - xb, 0.0)])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        sse = float(np.sum((y - X @ coef) ** 2))
        if best is None or sse < best["sse"]:
            best = {"xb": float(xb), "a": float(coef[0]), "b1": float(coef[1]), "b2": float(coef[2]), "sse": sse}
    return best


def main() -> None:
    t0 = time.time()
    out_dir = os.path.dirname(out_path("site", "partA.json"))
    slurm_record(out_dir)
    per_b = partb_per_tile()
    recs = {}
    for f in sorted(glob.glob(out_path("hilltops", "*.json"))):
        if os.path.basename(f) == "job.json":
            continue
        r = load_json(f)
        if "tile" in r:
            recs[r["tile"]] = r
    missing = [t for t in MEASURED_TILES if t not in recs]
    print(f"{len(recs)} tile records; missing {missing}", flush=True)
    out: dict = {"n_tiles": len(recs), "missing": missing, "tiles": {}, "sites": {}, "rules": {}}
    # ------------------------------------------------------------------ 2.1 per tile
    rows = []
    for tile, r in recs.items():
        pb = per_b[tile]
        d0 = r["D0_N"]
        A = d0.get("A", {})
        B = d0.get("B", {})
        Bp = d0.get("B_prime_rollover_width", {})
        unf = r.get("unf", {})
        row = {"tile": tile, "site": r["site"], "n_counted": d0["n_counted"], "usable": d0["usable"], "n_geom": r["set"]["n_geom"], "n_removed_road": r["set"]["n_removed_road"],
               "n_removed_segment": r["set"]["n_removed_segment"], "n_removed_nonconvex": r["set"]["n_removed_nonconvex"], "n_segments": r["set"].get("n_segments_counted"),
               "road_lines_in_tile": r["roads"].get("n_lines_in_tile"), "C_ht_Lf": A.get("C_ht"), "C_ht_Lf_ci68": A.get("C_ht_ci68"), "D0_N_A": A.get("D0"), "D0_N_A_ci68": A.get("D0_ci68"),
               "L_w_m": B.get("L_w_m"), "crest_width_1e3_m": B.get("crest_width_m"), "C_ht_Lw": B.get("C_ht"), "D0_N_B": B.get("D0"), "D0_N_B_ci68": B.get("D0_ci68"),
               "window_test_pass": d0.get("window_test_B", {}).get("pass"), "window_test_n": len(d0.get("window_test_B", {}).get("windows_tested_nominal_m", [])),
               "window_test_max_dev": d0.get("window_test_B", {}).get("max_abs_deviation"),
               "L_w_prime_m": Bp.get("L_w_m"), "crest_width_rollover_m": Bp.get("crest_width_m"), "D0_N_B_prime": Bp.get("D0"),
               "window_test_prime_pass": d0.get("window_test_B_prime_rollover_width", {}).get("pass"), "window_test_prime_n": len(d0.get("window_test_B_prime_rollover_width", {}).get("windows_tested_nominal_m", [])),
               "D0_R_A": fnum(pb["D0_A"]), "D0_R_B": fnum(pb["D0_B"]), "n_rollover": int(float(pb["n_hilltop_px"])), "A_c_rollover": fnum(pb["A_c_rollover"]),
               "D0_N_unf_A": unf.get("D0_A", {}).get("D0"), "D0_N_unf_B": unf.get("D0_B", {}).get("D0"), "n_unf": unf.get("n"),
               "D0_N_A_no_convex_drop": r.get("segfilt_no_convex_drop", {}).get("D0_A", {}).get("D0"),
               "rollover_recheck_rel": r["rollover_set"].get("check_rel")}
        for k_num, k_den, name in (("D0_N_A", "D0_R_A", "N_over_R_A"), ("D0_N_B", "D0_R_B", "N_over_R_B"), ("D0_N_A", "D0_N_unf_A", "N_over_unf_A"), ("D0_N_B", "D0_N_unf_B", "N_over_unf_B"),
                                   ("D0_N_A", "D0_N_A_no_convex_drop", "N_over_noconvexdrop_A"), ("D0_N_B_prime", "D0_N_A", "B_prime_over_A"), ("D0_N_B", "D0_N_A", "B_over_A")):
            row[name] = (row[k_num] / row[k_den]) if (row.get(k_num) and row.get(k_den)) else None
        dr = r["D_R"].get("net_1e3", {})
        row.update({"D_R": dr.get("D_R", {}).get("median"), "D_R_ci68": dr.get("D_R", {}).get("ci68"), "D_R_euclid": dr.get("D_R_euclid", {}).get("median"), "D_R_over_D_C": dr.get("D_R_over_D_C_median"),
                    "ratio_norm": dr.get("ratio_normalised", {}).get("median"), "L_H_median": dr.get("L_H_median_m"), "R_median": dr.get("R_median_m"), "n_traced": dr.get("n_pixels"), "n_used": dr.get("n_used"),
                    "n_left": dr.get("n_left_tile"), "E_star_1.27": dr.get("E_star_at_1.27", {}).get("median"), "E_bands": dr.get("E_star_bands")})
        pk = r["D_R"].get("picked_heads")
        if pk:
            row.update({"D_R_picked": pk.get("D_R", {}).get("median"), "D_R_picked_ci68": pk.get("D_R", {}).get("ci68"), "n_used_picked": pk.get("n_used"), "L_H_picked": pk.get("L_H_median_m"), "D_R_over_D_C_picked": pk.get("D_R_over_D_C_median")})
        mp = r["D_R"].get("model_P4")
        if mp:
            row.update({"D_R_model_P4": mp.get("D_R", {}).get("median"), "n_used_model_P4": mp.get("n_used"), "L_H_model_P4": mp.get("L_H_median_m"), "A_crit_P4": r["model_P4"]["A_crit"], "D_P4": r["model_P4"]["D"]})
        # the bracket (decision 12): fixed here
        if d0["usable"] and A.get("D0_ci68") and dr.get("D_R", {}).get("ci68"):
            row["bracket"] = [A["D0_ci68"][0], dr["D_R"]["ci68"][1]]
        else:
            row["bracket"] = None
        if r.get("legacy_v1_curvature_approx"):
            row["legacy_C_ht"] = r["legacy_v1_curvature_approx"].get("C_ht")
            row["legacy_D0"] = r["legacy_v1_curvature_approx"].get("D0")
            row["legacy_n"] = r["legacy_v1_curvature_approx"].get("n_hilltops")
        rows.append(row)
        out["tiles"][tile] = row
    pd.DataFrame(rows).to_csv(out_path("site", "per_tile.csv"), index=False)
    # ------------------------------------------------------------------ 2.4 site summaries and R1
    for site, groups in SITE_GROUPS.items():
        out["sites"][site] = {}
        for g in groups:
            tl = [t for t in tiles_of(site, g) if t in out["tiles"]]
            tr = [out["tiles"][t] for t in tl]
            us = [x for x in tr if x["usable"]]
            s = {"tiles": tl, "n": len(tl), "n_usable": len(us), "tiles_not_usable": [x["tile"] for x in tr if not x["usable"]],
                 "n_counted_median": med([x["n_counted"] for x in tr]), "n_counted_range": rng([x["n_counted"] for x in tr])}
            for k in ("D0_N_A", "D0_N_B", "D0_N_B_prime", "D0_R_A", "D0_R_B", "D0_N_unf_A", "D0_N_unf_B", "D0_N_A_no_convex_drop", "N_over_R_A", "N_over_R_B", "N_over_unf_A", "N_over_unf_B",
                      "N_over_noconvexdrop_A", "B_over_A", "B_prime_over_A", "C_ht_Lf", "D_R", "D_R_euclid", "D_R_over_D_C", "ratio_norm", "L_H_median", "E_star_1.27", "legacy_D0", "legacy_C_ht"):
                s[f"{k}_median"] = med([x.get(k) for x in us])
                s[f"{k}_range"] = rng([x.get(k) for x in us])
            s["window_test_pass_n"] = sum(1 for x in us if x["window_test_pass"])
            s["window_test_trivial_n"] = sum(1 for x in us if x["window_test_n"] <= 1)
            s["window_test_prime_pass_n"] = sum(1 for x in us if x["window_test_prime_pass"])
            s["window_test_prime_trivial_n"] = sum(1 for x in us if x["window_test_prime_n"] <= 1)
            s["L_w_values"] = {str(x["L_w_m"]): sum(1 for y in us if y["L_w_m"] == x["L_w_m"]) for x in us}
            s["L_w_prime_values"] = {str(x["L_w_prime_m"]): sum(1 for y in us if y["L_w_prime_m"] == x["L_w_prime_m"]) for x in us}
            out["sites"][site][g] = s
        prim = out["sites"][site]["primary"]
        need = int(np.ceil(R1_MIN_FRAC_TILES * prim["n"] - 1e-9))
        out["rules"].setdefault("R1", {})[site] = {"n_usable": prim["n_usable"], "N": prim["n"], "needed": need, "count_pass": bool(prim["n_usable"] >= need), "tiles_below_500": prim["tiles_not_usable"],
                                                   "qc_look": "pending (section 2.3 figures, read by eye)"}
    # ------------------------------------------------------------------ 2.5 Perron calibration (pooled pixels of the primary tiles) and R11
    pix_cache: dict[str, pd.DataFrame] = {}

    def pixels(tile):
        if tile not in pix_cache:
            pix_cache[tile] = pd.read_parquet(out_path("hilltops", f"{tile}_pixels.parquet"))
        return pix_cache[tile]

    out["perron"] = {}
    for site in ("GM", "AP"):
        tl = [t for t in PRIMARY[site] if t in recs]
        frames = []
        for i, t in enumerate(tl):
            df = pixels(t)
            df = df.assign(gseg=df["segment"].to_numpy() + 10_000_000 * i)
            lw = out["tiles"][t]["L_w_m"]
            df = df.assign(C_Lw=df[f"C_w{(lw - 1) // 2}"].to_numpy() if lw else np.nan)
            frames.append(df)
        pool = pd.concat(frames, ignore_index=True)
        seg = pool["gseg"].to_numpy()
        ref = PERRON_2012[site]["curvature_per_m"]
        rec = {"tiles": tl, "n_pixels": int(pool.shape[0]), "perron": PERRON_2012[site], "E_site": E_SITE[site], "E_matches_perron": bool(abs(E_SITE[site] * 1e6 - PERRON_2012[site]["E_m_per_Myr"]) < 1e-6),
               "by_window": {}}
        for w in ALL_HALF_WIDTHS:
            m = pooled_median_boot(pool[f"C_w{w}"].to_numpy(), seg, N_BOOT if w in (W_F,) else 400)
            m["L_nominal_m"] = 2 * w + 1
            m["ratio_to_perron"] = m["median"] / ref if m["median"] else None
            m["ratio_ci68"] = [m["ci68"][0] / ref, m["ci68"][1] / ref] if m.get("ci68") else None
            rec["by_window"][str(w)] = m
        rec["at_L_f"] = rec["by_window"][str(W_F)]
        mw = pooled_median_boot(pool["C_Lw"].to_numpy(), seg, N_BOOT)
        mw["ratio_to_perron"] = mw["median"] / ref if mw["median"] else None
        mw["ratio_ci68"] = [mw["ci68"][0] / ref, mw["ci68"][1] / ref] if mw.get("ci68") else None
        rec["at_L_w"] = mw
        in_range = [2 * w + 1 for w in ALL_HALF_WIDTHS if rec["by_window"][str(w)]["ratio_to_perron"] and R11_RANGE[0] <= rec["by_window"][str(w)]["ratio_to_perron"] <= R11_RANGE[1]]
        rec["windows_in_range_m"] = in_range
        rec["tile_median_of_C_ht_Lf"] = med([out["tiles"][t]["C_ht_Lf"] for t in tl if out["tiles"][t]["usable"]])
        rec["legacy_route"] = {"available": "approximation (lemcvt.measure.legacy_v1.legacy_curvature: block-mean grid, coarse D8, divides between coarse basins, 3x3 Laplacian; the "
                                            "topo_analysis first-order-divide module is not available, as the package labels it)",
                               "C_ht_site_median": med([out["tiles"][t].get("legacy_C_ht") for t in tl]), "D0_site_median": med([out["tiles"][t].get("legacy_D0") for t in tl]),
                               "C_ht_range": rng([out["tiles"][t].get("legacy_C_ht") for t in tl]), "n_hilltops_range": rng([out["tiles"][t].get("legacy_n") for t in tl])}
        rec["legacy_route"]["ratio_to_perron"] = rec["legacy_route"]["C_ht_site_median"] / ref if rec["legacy_route"]["C_ht_site_median"] else None
        ratio = rec["at_L_f"]["ratio_to_perron"]
        out["rules"].setdefault("R11", {})[site] = {"ratio_at_L_f": ratio, "ratio_ci68": rec["at_L_f"]["ratio_ci68"], "range": list(R11_RANGE), "pass": bool(ratio and R11_RANGE[0] <= ratio <= R11_RANGE[1]),
                                                    "windows_in_range_m": in_range, "D_statements_run": bool(ratio and R11_RANGE[0] <= ratio <= R11_RANGE[1])}
        out["perron"][site] = rec
        print(f"{site}: pooled |C_ht|(L_f) {rec['at_L_f']['median']:.4g} = {ratio:.2f} x Perron; in range at L = {in_range} [{time.time() - t0:.0f} s]", flush=True)
    out["rules"]["R11"]["TTV"] = {"pass": None, "note": "no Perron value; not gated", "D_statements_run": True}
    # ------------------------------------------------------------------ 2.6 scaling break and R12
    out["scaling_break"] = {}
    breaks = np.log10(np.arange(4.0, 29.01, 0.1))
    for site in ("GM", "AP", "TTV"):
        tl = [t for t in PRIMARY[site] if t in recs]
        Ls = np.array([2 * w + 1 for w in SCAN_HALF_WIDTHS], dtype=float)
        curves = {}
        for t in tl:
            sg = recs[t]["sigma_C"]
            s = np.array([sg[str(w)]["sigma"] for w in SCAN_HALF_WIDTHS])
            curves[t] = s / sg[str(W_F)]["sigma"]
        X = np.concatenate([np.log10(Ls) for _ in tl])
        Y = np.concatenate([np.log10(curves[t]) for t in tl])
        fit = piecewise_fit(X, Y, breaks)
        r = np.random.default_rng(SEED)
        bs = []
        for _ in range(N_BOOT):
            pick = r.integers(0, len(tl), len(tl))
            Xb = np.concatenate([np.log10(Ls) for _ in pick])
            Yb = np.concatenate([np.log10(curves[tl[j]]) for j in pick])
            fb = piecewise_fit(Xb, Yb, breaks)
            bs.append(10 ** fb["xb"] if fb else np.nan)
        bs = np.array(bs)
        L_break = 10 ** fit["xb"]
        ci68 = [float(np.nanpercentile(bs, 16)), float(np.nanpercentile(bs, 84))]
        odd = np.array([2 * w + 1 for w in SCAN_HALF_WIDTHS])
        L_S = int(odd[np.argmin(np.abs(odd - L_break))])
        w_S = (L_S - 1) // 2
        rec = {"tiles": tl, "L_nominal_m": Ls.tolist(), "sigma_curves_normalised": {t: curves[t].tolist() for t in tl}, "pooled_median_curve": np.median(np.array([curves[t] for t in tl]), axis=0).tolist(),
               "fit": fit, "L_break_m": float(L_break), "L_break_ci68": ci68, "L_break_ci95": [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))],
               "slopes_log": {"below": fit["b1"], "above": fit["b2"]}, "L_S_m": L_S, "w_S": w_S, "n_boot": N_BOOT, "seed": SEED}
        for t in tl:
            cv = recs[t]["curve_counted"].get(str(w_S))
            E = E_SITE[site]
            d_s = (E / cv["median"]) if cv and cv.get("median") and cv["median"] > 0 else None
            out["tiles"][t]["D0_N_S"] = d_s
            out["tiles"][t]["D0_N_S_ci68"] = [E / cv["ci68"][1], E / cv["ci68"][0]] if cv and cv.get("ci68") and cv["ci68"][0] > 0 else None
            out["tiles"][t]["L_S_m"] = L_S
            out["tiles"][t]["S_over_A"] = (d_s / out["tiles"][t]["D0_N_A"]) if (d_s and out["tiles"][t]["D0_N_A"]) else None
        us = [out["tiles"][t] for t in tl if out["tiles"][t]["usable"]]
        rec["D0_N_S_median"] = med([x.get("D0_N_S") for x in us])
        rec["S_over_A_median"] = med([x.get("S_over_A") for x in us])
        rec["S_over_A_range"] = rng([x.get("S_over_A") for x in us])
        inside_ci = bool(ci68[0] <= L_F_M <= ci68[1])
        in_lit = bool(L_BREAK_LITERATURE_M[0] <= L_break <= L_BREAK_LITERATURE_M[1])
        out["rules"].setdefault("R12", {})[site] = {"L_break_m": float(L_break), "ci68": ci68, "L_f_inside_ci68": inside_ci, "L_break_in_8_20": in_lit, "consistent": bool(inside_ci or in_lit),
                                                    "L_S_m": L_S, "D0_N_S_over_A_median": rec["S_over_A_median"]}
        out["scaling_break"][site] = rec
        out["sites"][site]["primary"]["D0_N_S_median"] = rec["D0_N_S_median"]
        print(f"{site}: L_break {L_break:.1f} m [{ci68[0]:.1f}, {ci68[1]:.1f}], slopes {fit['b1']:.2f}/{fit['b2']:.2f}; L_S {L_S} m; D0_N_S/A median {rec['S_over_A_median']} [{time.time() - t0:.0f} s]", flush=True)
    # ------------------------------------------------------------------ 2.7 E* at the fitted S_c (pooled bracket pixels), addendum 8 medians
    out["e_star"] = {}
    a8 = load_json(A8_ER_STAR) if os.path.exists(A8_ER_STAR) else None
    for site in ("GM", "AP", "TTV"):
        tl = [t for t in PRIMARY[site] if t in recs]
        frames = []
        for i, t in enumerate(tl):
            df = pixels(t)
            if "used_net_1e3" not in df:
                continue
            d = df[df["used_net_1e3"]]
            frames.append(pd.DataFrame({"tile": t, "C": d["C_Lf"].to_numpy(), "L": d["L_H_net_1e3"].to_numpy(), "S": (d["R_net_1e3"] / d["L_H_net_1e3"]).to_numpy(), "gseg": d["segment"].to_numpy() + 10_000_000 * i}))
        pool = pd.concat(frames, ignore_index=True)
        C, L, S, seg = pool["C"].to_numpy(), pool["L"].to_numpy(), pool["S"].to_numpy(), pool["gseg"].to_numpy()
        ok = np.isfinite(C) & np.isfinite(L) & np.isfinite(S) & (C > 0) & (L > 0) & (S > 0)
        C, L, S, seg = C[ok], L[ok], S[ok], seg[ok]
        fit = fit_S_c(C, L, S)
        uniq, inv = np.unique(seg, return_inverse=True)
        r = np.random.default_rng(SEED)
        bs = np.empty(400)
        for i in range(bs.size):
            counts = np.bincount(r.integers(0, uniq.size, uniq.size), minlength=uniq.size).astype(float)
            bs[i] = fit_S_c(C, L, S, counts[inv])["S_c"]
        fit["S_c_ci68"] = [float(np.percentile(bs, 16)), float(np.percentile(bs, 84))]
        fit["S_c_ci95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        fit["inside_own_ci68"] = bool(fit["S_c_ci68"][0] <= fit["S_c"] <= fit["S_c_ci68"][1])
        fit["n_boot"] = int(bs.size)
        E_ref = 2 * C * L / S_C_REFERENCE
        E_fit = 2 * C * L / fit["S_c"]
        bands = lambda e: {"lt_10": float(np.mean(e < E_STAR_BANDS[0])), "10_30": float(np.mean((e >= E_STAR_BANDS[0]) & (e <= E_STAR_BANDS[1]))), "gt_30": float(np.mean(e > E_STAR_BANDS[1]))}  # noqa: E731
        rec = {"tiles": tl, "n_pixels": int(C.size), "S_c_fit": fit, "S_c_reference": S_C_REFERENCE, "E_star_at_reference": {"median": float(np.median(E_ref)), "bands": bands(E_ref)},
               "E_star_at_fit": {"median": float(np.median(E_fit)), "bands": bands(E_fit), "applies": fit["inside_own_ci68"]}, "per_tile": {}}
        for t in tl:
            m = pool["tile"].to_numpy()[ok] == t
            if m.any():
                e_r, e_f = E_ref[m], E_fit[m]
                rec["per_tile"][t] = {"n": int(m.sum()), "E_star_ref_median": float(np.median(e_r)), "bands_ref": bands(e_r), "E_star_fit_median": float(np.median(e_f)), "bands_fit": bands(e_f)}
                out["tiles"][t]["E_star_fit"] = float(np.median(e_f))
        if a8:
            rec["addendum_8_normalised_ratio_medians"] = {t: a8["tiles"][t]["networks"]["primary_1e3"]["ratio_normalised"]["median"] for t in a8["tiles"] if t.startswith(site)}
            rec["addendum_8_S_c_pooled"] = a8["pooled"].get(site, {}).get("primary_1e3", {}).get("S_c_fit", {}).get("S_c")
        out["e_star"][site] = rec
        print(f"{site}: S_c fit {fit['S_c']:.3f} {fit['S_c_ci68']} inside {fit['inside_own_ci68']}; E* median at 1.27 {rec['E_star_at_reference']['median']:.1f}, bands {rec['E_star_at_reference']['bands']} [{time.time() - t0:.0f} s]", flush=True)
    # ------------------------------------------------------------------ 2.8 TTV anchor
    ttv = [out["tiles"][t] for t in PRIMARY["TTV"] if t in out["tiles"] and out["tiles"][t]["usable"]]
    anc = {}
    for k in ("D0_N_A", "D0_N_S", "D_R", "D0_N_B", "D0_R_A"):
        v = med([x.get(k) for x in ttv])
        anc[k] = {"site_median": v, "over_2.5e-3": v / TTV_ANCHOR["bedrock_convention_m2_per_yr"] if v else None, "over_5e-3": v / TTV_ANCHOR["soil_flux_convention_m2_per_yr"] if v else None}
    out["ttv_anchor"] = {"anchor": TTV_ANCHOR, "ratios": anc, "n_tiles": len(ttv)}
    # brackets table
    out["brackets"] = {t: {"bracket": x["bracket"], "D_C_lo": x["bracket"][0] if x["bracket"] else None, "D_R_hi": x["bracket"][1] if x["bracket"] else None, "usable": x["usable"]} for t, x in out["tiles"].items()}
    out["constants"] = {"D0_MIN_PIXELS": D0_MIN_PIXELS, "N_BOOT": N_BOOT, "SEED": SEED, "TEST_HALF_WIDTHS": list(TEST_HALF_WIDTHS), "hold_n_example_22": hold_n(22), "clean": {k[0] + ":" + k[1]: v for k, v in CLEAN.items()} if isinstance(CLEAN, dict) and all(isinstance(k, tuple) for k in CLEAN) else None}
    out["wall_s"] = time.time() - t0
    pd.DataFrame(list(out["tiles"].values())).to_csv(out_path("site", "per_tile.csv"), index=False)
    dump(out, out_path("site", "partA.json"))
    print("R1", out["rules"]["R1"], "\nR11", {k: (v.get("ratio_at_L_f"), v.get("pass")) for k, v in out["rules"]["R11"].items()}, "\nR12", {k: (v["L_break_m"], v["consistent"]) for k, v in out["rules"]["R12"].items()}, flush=True)
    _ = weighted_median


if __name__ == "__main__":
    main()
