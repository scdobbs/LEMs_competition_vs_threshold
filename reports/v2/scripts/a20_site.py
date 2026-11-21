#!/usr/bin/env python3
"""Addendum 20, Part A site level (sections 2.2-2.8, 3.1-3.2, 3.4; rules R1, R13, R11P; the bracket of decision 3).

Per tile: D0_P (r = 3 ... 15 m, fixed ranges, D-infinity), D0_N (decision 1) and its 20-px sensitivity, D0_P/D0_N, D_R, the two brackets.
Per site (primary tiles pooled): the per-block bin tables of every primary tile stacked, the plateau rule applied once, block bootstrap over
all blocks (2,000, seed 20261120), at every radius, with D-infinity at GM and AP; the same for Perron's own surveys (NCALM at GM, PAMAP at AP:
report only). Rules R1 (D0_N >= 500 px on >= 90 % of primary tiles), R13 (plateau on >= 90 % of primary tiles and on the pooled curve), R11P
(pooled |lap_h| at r = 7 m within 0.67-1.5 x Perron) with the reading written in the directive from the four-way comparison. The tile D_C is
D0_P where the tile has a plateau, else D0_N (decision 2). Bracket (decision 3): [lower 68 % of D_C, upper 68 % of D_R on the 10^3 network];
the bracket with D0_N at the lower end beside it. E* (decision 13 of addendum 19) on the new pixels; the TTV anchor with D0_P.
usage: a20_site.py
"""
from __future__ import annotations

import glob
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a20_common import (  # noqa: E402
    A18_SUMMARY,
    AP1_3DEP,
    AP_PAMAP,
    D0_MIN_PIXELS,
    E_SITE,
    E_STAR_BANDS,
    FIG_DIR,
    GM_NCALM,
    GM_OFFMESA,
    GROUPS,
    LAP_ZERO,
    MEASURED_TILES,
    N_BOOT_BLOCK,
    PERRON_2012,
    PERRON_R_PRIMARY,
    PERRON_RADII_M,
    PRIMARY,
    R1_MIN_FRAC_TILES,
    R11_RANGE,
    R13_MIN_FRAC_TILES,
    SEED,
    SURVEY_PAIRS,
    TTV_ANCHOR,
    a19_path,
    dump,
    load_json,
    out_path,
    slurm_record,
)
from a20_perron import N_BINS, d0_from_lap, estimate  # noqa: E402

from lemcvt.analyses.er_star import S_C_REFERENCE, fit_S_c  # noqa: E402

SITE_GROUPS = {"GM": ["primary", "clean", "offmesa", "ncalm"], "AP": ["primary", "clean", "pamap", "ap1_3dep"], "TTV": ["primary", "clean"]}
PERRON_SURVEY_GROUP = {"GM": "ncalm", "AP": "pamap"}
PLATEAU_CELL_X_MAX = 10.0


def tiles_of(site, group):
    if group == "ap1_3dep":
        return AP1_3DEP
    return GROUPS[(site, group)] if (site, group) in GROUPS else {"ncalm": GM_NCALM, "pamap": AP_PAMAP, "offmesa": GM_OFFMESA}[group]


def med(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else None


def rng_(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return [float(min(v)), float(max(v))] if v else None


def hold(n, N, frac):
    need = int(np.ceil(frac * N - 1e-9))
    return {"n": n, "N": N, "needed": need, "pass": bool(n >= need)}


def stack_tables(tiles: list[str], key: str) -> dict[str, np.ndarray]:
    out: dict[str, list] = {}
    for t in tiles:
        z = np.load(out_path("tiles", f"{t}_tables.npz"))
        for k in z.files:
            if not k.startswith(key + "__"):
                continue
            name = k.split("__", 1)[1]
            v = z[k]
            if v.ndim == 0:
                continue
            out.setdefault(name, []).append(v)
    return {k: np.concatenate(v, axis=0) for k, v in out.items()}


def pooled_median_block(vals: np.ndarray, blocks: np.ndarray, n_boot: int = N_BOOT_BLOCK, seed: int = SEED) -> dict:
    ok = np.isfinite(vals)
    v, b = vals[ok], blocks[ok]
    order = np.argsort(v, kind="stable")
    vs = v[order]
    uniq, inv = np.unique(b[order], return_inverse=True)
    r = np.random.default_rng(seed)
    bs = np.empty(n_boot)
    for i in range(n_boot):
        w = np.bincount(r.integers(0, uniq.size, uniq.size), minlength=uniq.size).astype(float)[inv]
        cw = np.cumsum(w)
        bs[i] = vs[np.searchsorted(cw, 0.5 * cw[-1])]
    return {"value": float(np.median(v)), "n": int(v.size), "n_blocks": int(uniq.size), "ci68": [float(np.percentile(bs, 16)), float(np.percentile(bs, 84))]}


def pooled_geomean_block(neg_vals: np.ndarray, blocks: np.ndarray, n_boot: int = N_BOOT_BLOCK, seed: int = SEED) -> dict:
    """Geometric mean of |v| over the convex cells (v < 0), block bootstrap."""
    ok = np.isfinite(neg_vals) & (neg_vals < -LAP_ZERO)
    lv, b = np.log10(-neg_vals[ok]), blocks[ok]
    uniq, inv = np.unique(b, return_inverse=True)
    s = np.bincount(inv, weights=lv)
    n = np.bincount(inv).astype(float)
    r = np.random.default_rng(seed)
    W = np.stack([np.bincount(r.integers(0, uniq.size, uniq.size), minlength=uniq.size) for _ in range(n_boot)]).astype(float)
    bs = 10 ** ((W @ s) / (W @ n))
    return {"value": float(10 ** (s.sum() / n.sum())), "n_convex": int(n.sum()), "frac_convex": float(ok.sum() / np.isfinite(neg_vals).sum()), "n_blocks": int(uniq.size),
            "ci68": [float(np.percentile(bs, 16)), float(np.percentile(bs, 84))]}


def main() -> None:
    t0 = time.time()
    slurm_record(os.path.dirname(out_path("site", "partA.json")))
    recs = {}
    for f in sorted(glob.glob(out_path("tiles", "*.json"))):
        if os.path.basename(f) == "job.json":
            continue
        r = load_json(f)
        if "tile" in r:
            recs[r["tile"]] = r
    missing = [t for t in MEASURED_TILES if t not in recs]
    print(f"{len(recs)} tiles; missing {missing}", flush=True)
    a19 = load_json(a19_path("site", "partA.json"))
    fits18 = load_json(os.path.join(A18_SUMMARY, "per_tile_fits.json"))
    out: dict = {"n_tiles": len(recs), "missing": missing, "tiles": {}, "sites": {}, "rules": {}, "pooled": {}, "four_way": {}}
    # ------------------------------------------------------------------ per tile
    for t, r in recs.items():
        a19t = a19["tiles"].get(t, {})
        d0n = r["D0_N"]
        A_ = d0n.get("A", {})
        row = {"tile": t, "site": r["site"], "n_counted": d0n["n_counted"], "usable_N": d0n["usable"], "n_geom": r["set"]["n_geom"], "n_removed_road": r["set"]["n_removed_road"],
               "n_removed_nonconvex": r["set"]["n_removed_nonconvex"], "n_segments": r["set"]["n_segments"], "segment_size_median_px": r["set"]["segment_size_median_px"],
               "n_counted_a19_seg50": r["set"]["a19_n_counted_segment50"], "same_as_a19_fragment0": r["set"]["same_as_a19_fragment0"],
               "C_ht_Lf": A_.get("C_ht"), "D0_N_A": A_.get("D0"), "D0_N_A_ci68": A_.get("D0_ci68"), "D0_N_A_ci68_block": r["D0_N"].get("A_block_bootstrap", {}).get("D0_ci68"),
               "D0_N_frag20": r.get("D0_N_fragment20", {}).get("D0"), "n_frag20": r.get("D0_N_fragment20", {}).get("n"), "usable_frag20": r.get("D0_N_fragment20", {}).get("usable"),
               "D0_N_a19_seg50": a19t.get("D0_N_A"), "D0_R_A": a19t.get("D0_R_A"),
               "D0_N_by_L": {v["L_nominal_m"]: v["D0"].get("D0") for v in d0n["curve"].values()}}
        for key, v in r["D0_P"].items():
            row[f"D0_P_{key}"] = v.get("D0_P")
            row[f"D0_P_{key}_ci68"] = v.get("D0_P_ci68")
            row[f"plateau_{key}"] = v.get("plateau")
            row[f"D0_fixed1_{key}"] = v.get("D0_fixed_1")
            row[f"D0_fixed3_{key}"] = v.get("D0_fixed_3")
            row[f"D0_P_{key}_noflat"] = v.get("D0_P_no_flat")
            row[f"plateau_{key}_noflat"] = v.get("no_flat_plateau")
        p7 = r["perron"]["r7"]
        row.update({"plateau_range_m2": p7["point"].get("range_m2"), "plateau_n_bins": p7["point"].get("n_bins"), "plateau_n_cells": p7["point"].get("n_cells"),
                    "plateau_stop": p7["point"].get("stop"), "plateau_start_convex_ok": p7["point"].get("start_bin_meets_convex_condition"),
                    "boot_share_plateau": p7.get("boot", {}).get("share_with_plateau"), "lap_h_r7": p7["point"].get("lap_h"), "lap_ci68_r7": p7.get("boot", {}).get("lap_ci68")})
        row["D0_P"] = row["D0_P_r7"]
        row["D0_P_ci68"] = row["D0_P_r7_ci68"]
        row["P_over_N"] = row["D0_P"] / row["D0_N_A"] if (row["D0_P"] and row["D0_N_A"]) else None
        row["P_over_R"] = row["D0_P"] / row["D0_R_A"] if (row["D0_P"] and row["D0_R_A"]) else None
        row["N_over_a19seg50"] = row["D0_N_A"] / row["D0_N_a19_seg50"] if (row["D0_N_A"] and row["D0_N_a19_seg50"]) else None
        # tile D_C: D0_P where a plateau, else D0_N (decision 2)
        if row["plateau_r7"] and row["D0_P_ci68"]:
            row["D_C"], row["D_C_ci68"], row["D_C_source"] = row["D0_P"], row["D0_P_ci68"], "D0_P"
        elif d0n["usable"]:
            row["D_C"], row["D_C_ci68"], row["D_C_source"] = row["D0_N_A"], row["D0_N_A_ci68"], "D0_N (no plateau)"
        else:
            row["D_C"], row["D_C_ci68"], row["D_C_source"] = None, None, "none"
        dr = r["D_R"].get("net_1e3", {})
        row.update({"D_R": dr.get("D_R", {}).get("median"), "D_R_ci68": dr.get("D_R", {}).get("ci68"), "D_R_ci68_block": dr.get("D_R_block_bootstrap", {}).get("ci68"),
                    "D_R_euclid": dr.get("D_R_euclid", {}).get("median"), "D_R_over_D_C_pixel": dr.get("D_R_over_D_C_median"), "ratio_norm": dr.get("ratio_normalised", {}).get("median"),
                    "L_H_median": dr.get("L_H_median_m"), "R_median": dr.get("R_median_m"), "n_traced": dr.get("n_pixels"), "n_used": dr.get("n_used"), "n_left": dr.get("n_left_tile"),
                    "n_R_nonpositive": dr.get("n_R_nonpositive"), "E_star_1.27": dr.get("E_star_at_1.27", {}).get("median"), "E_bands": dr.get("E_star_bands")})
        row["D_R_over_D0_P"] = row["D_R"] / row["D0_P"] if (row["D_R"] and row["D0_P"]) else None
        row["D_R_over_D0_N"] = row["D_R"] / row["D0_N_A"] if (row["D_R"] and row["D0_N_A"]) else None
        for name in ("snapped_heads", "picked_heads_a19_A3"):
            pk = r["D_R"].get(name)
            if pk:
                row.update({f"D_R_{name}": pk.get("D_R", {}).get("median"), f"D_R_{name}_ci68": pk.get("D_R", {}).get("ci68"), f"n_used_{name}": pk.get("n_used"),
                            f"n_left_{name}": pk.get("n_left_tile"), f"n_inside_{name}": pk.get("n_inside_md_polygon"), f"L_H_{name}": pk.get("L_H_median_m"),
                            f"D_R_euclid_{name}": pk.get("D_R_euclid", {}).get("median")})
        mp = r["D_R"].get("model_P4")
        if mp:
            row.update({"D_R_model_P4": mp.get("D_R", {}).get("median"), "L_H_model_P4": mp.get("L_H_median_m"), "A_crit_P4": r["model_P4"]["A_crit"], "D_P4": r["model_P4"]["D"],
                        "P4_D_over_D_R_model": r["model_P4"]["D"] / mp["D_R"]["median"] if mp.get("D_R", {}).get("median") else None})
        # brackets (decision 3), fixed here before any fit is classified
        det = bool(d0n["usable"] and row["D_R_ci68"])
        row["bracket_P"] = [row["D_C_ci68"][0], row["D_R_ci68"][1]] if (det and row["D_C_ci68"]) else None
        row["bracket_N"] = [row["D0_N_A_ci68"][0], row["D_R_ci68"][1]] if (det and row["D0_N_A_ci68"]) else None
        row["bracket_P_block"] = [row["D_C_ci68"][0], row["D_R_ci68_block"][1]] if (det and row["D_C_ci68"] and row["D_R_ci68_block"]) else None
        row["four_way"] = r["four_way"]
        out["tiles"][t] = row
    flat = [{k: (v if not isinstance(v, (dict, list)) else str(v)) for k, v in x.items()} for x in out["tiles"].values()]
    pd.DataFrame(flat).to_csv(out_path("site", "per_tile.csv"), index=False)
    # ------------------------------------------------------------------ site summaries, R1, R13
    for site, groups in SITE_GROUPS.items():
        out["sites"][site] = {}
        for g in groups:
            tl = [t for t in tiles_of(site, g) if t in out["tiles"]]
            tr = [out["tiles"][t] for t in tl]
            s = {"tiles": tl, "n": len(tl), "n_usable_N": sum(1 for x in tr if x["usable_N"]), "tiles_not_usable_N": [x["tile"] for x in tr if not x["usable_N"]],
                 "n_plateau_r7": sum(1 for x in tr if x["plateau_r7"]), "tiles_no_plateau": [x["tile"] for x in tr if not x["plateau_r7"]], "n_counted_median": med([x["n_counted"] for x in tr]),
                 "n_counted_range": rng_([x["n_counted"] for x in tr]), "n_usable_frag20": sum(1 for x in tr if x.get("usable_frag20")),
                 "n_plateau_r7_noflat": sum(1 for x in tr if x.get("plateau_r7_noflat"))}
            for k in ("D0_N_A", "D0_N_frag20", "D0_N_a19_seg50", "D0_R_A", "D0_P_r3", "D0_P_r5", "D0_P_r7", "D0_P_r10", "D0_P_r15", "D0_P_r7_dinf", "D0_P_r7_noflat", "D0_fixed1_r7", "D0_fixed3_r7",
                      "P_over_N", "P_over_R", "N_over_a19seg50", "D_R", "D_R_euclid", "D_R_over_D_C_pixel", "D_R_over_D0_P", "D_R_over_D0_N", "ratio_norm", "L_H_median", "E_star_1.27",
                      "D_R_model_P4", "P4_D_over_D_R_model", "boot_share_plateau", "C_ht_Lf", "lap_h_r7"):
                vals = [x.get(k) for x in tr if (x["usable_N"] if k.startswith(("D0_N", "D_R", "C_ht", "N_over", "ratio", "L_H", "E_star")) else True)]
                s[f"{k}_median"] = med(vals)
                s[f"{k}_range"] = rng_(vals)
            out["sites"][site][g] = s
        prim = out["sites"][site]["primary"]
        out["rules"].setdefault("R1", {})[site] = hold(prim["n_usable_N"], prim["n"], R1_MIN_FRAC_TILES) | {"tiles_below_500": prim["tiles_not_usable_N"],
                                                                                                           "fragment20": hold(prim["n_usable_frag20"], prim["n"], R1_MIN_FRAC_TILES)}
    # ------------------------------------------------------------------ pooled per site (2.4/2.5), R13 pooled part, R11P
    for site in ("GM", "AP", "TTV"):
        groups = [("primary", PRIMARY[site])] + ([(PERRON_SURVEY_GROUP[site], tiles_of(site, PERRON_SURVEY_GROUP[site]))] if site in PERRON_SURVEY_GROUP else [])
        out["pooled"][site] = {}
        for gname, tl in groups:
            tl = [t for t in tl if t in recs]
            pg: dict = {"tiles": tl}
            keys = [f"r{r:g}" for r in PERRON_RADII_M] + (["r7_dinf"] if site in ("GM", "AP") and gname == "primary" else [])
            for key in keys:
                tab = stack_tables(tl, key)
                if not tab:
                    continue
                est = estimate(tab, extra=("sq11",) if key == "r7" else ())
                pg[key] = est | {"D0": d0_from_lap(E_SITE[site], est)}
                if site in PERRON_2012:
                    ref = PERRON_2012[site]["curvature_per_m"]
                    pg[key]["ratio_to_perron"] = abs(est["point"]["lap_h"]) / ref if est["point"].get("plateau") else None
                    ci = est.get("boot", {}).get("lap_ci68")
                    pg[key]["ratio_ci68"] = sorted([abs(ci[0]) / ref, abs(ci[1]) / ref]) if ci else None
                    nf = est.get("no_flat")
                    if nf:
                        pg[key]["ratio_to_perron_noflat"] = abs(nf["point"]["lap_h"]) / ref if nf["point"]["plateau"] else None
                        nci = (nf.get("boot") or {}).get("lap_ci68")
                        pg[key]["ratio_ci68_noflat"] = sorted([abs(nci[0]) / ref, abs(nci[1]) / ref]) if nci else None
                    for R in (1, 3):
                        v = est[f"fixed_{R}"]["lap"]
                        pg[key][f"fixed_{R}_ratio"] = abs(v) / ref if v else None
                print(f"{site} {gname} {key}: plateau {est['point'].get('plateau')} {est['point'].get('range_m2')} lap_h {est['point'].get('lap_h')} ratio {pg[key].get('ratio_to_perron')} "
                      f"share {est.get('boot', {}).get('share_with_plateau')} [{time.time() - t0:.0f} s]", flush=True)
            out["pooled"][site][gname] = pg
        prim = out["sites"][site]["primary"]
        pooled_plateau = bool(out["pooled"][site]["primary"]["r7"]["point"].get("plateau"))
        tiles_rule = hold(prim["n_plateau_r7"], prim["n"], R13_MIN_FRAC_TILES)
        nfp = out["pooled"][site]["primary"]["r7"].get("no_flat", {}).get("point", {})
        out["rules"].setdefault("R13", {})[site] = tiles_rule | {"pooled_plateau": pooled_plateau, "holds": bool(tiles_rule["pass"] and pooled_plateau), "tiles_no_plateau": prim["tiles_no_plateau"],
                                                                 "sensitivity_no_flat": {"n_tiles": prim["n_plateau_r7_noflat"], "N": prim["n"], "pooled_plateau": bool(nfp.get("plateau")),
                                                                                         "pooled_range_m2": nfp.get("range_m2")}}
    # ------------------------------------------------------------------ four-way, pooled per site
    for site in ("GM", "AP", "TTV"):
        tl = [t for t in PRIMARY[site] if t in recs]
        pp = out["pooled"][site]["primary"]["r7"]["point"]
        frames_d, frames_p = [], []
        for i, t in enumerate(tl):
            d = pd.read_parquet(out_path("tiles", f"{t}_divide_pixels.parquet"), columns=["block", "C_Lf", "lap7"])
            if not recs[t]["D0_N"]["usable"]:
                d = d.iloc[0:0]
            frames_d.append(d.assign(gblock=d["block"].to_numpy() + 1_000_000 * i))
            if pp.get("plateau"):
                c = pd.read_parquet(out_path("tiles", f"{t}_plateau_cells.parquet"), columns=["block", "bin7", "lap7", "sq11"])
                c = c[c["bin7"].isin(pp["bins"])]
                frames_p.append(c.assign(gblock=c["block"].to_numpy() + 1_000_000 * i))
        D = pd.concat(frames_d, ignore_index=True)
        fw: dict = {"tiles": tl, "pooled_plateau_bins_m2": pp.get("range_m2"), "plateau_cells_complete": bool(pp.get("plateau") and pp["range_m2"][1] <= PLATEAU_CELL_X_MAX + 1e-9)}
        fw["divide_square11"] = {"median": pooled_median_block(D["C_Lf"].to_numpy(), D["gblock"].to_numpy()), "geomean": pooled_geomean_block(-D["C_Lf"].to_numpy(), D["gblock"].to_numpy())}
        fw["divide_disk7"] = {"median": pooled_median_block(-D["lap7"].to_numpy(), D["gblock"].to_numpy()), "geomean": pooled_geomean_block(D["lap7"].to_numpy(), D["gblock"].to_numpy())}
        if frames_p:
            P = pd.concat(frames_p, ignore_index=True)
            lapc = P["lap7"].to_numpy().astype(float)
            sqc = P["sq11"].to_numpy().astype(float)
            fw["plateau_disk7"] = {"median_convex": pooled_median_block(-lapc[lapc < -LAP_ZERO], P["gblock"].to_numpy()[lapc < -LAP_ZERO]), "geomean": pooled_geomean_block(lapc, P["gblock"].to_numpy()),
                                   "median_all": float(np.median(-lapc))}
            fw["plateau_square11"] = {"median_convex": pooled_median_block(-sqc[sqc < -LAP_ZERO], P["gblock"].to_numpy()[sqc < -LAP_ZERO]), "geomean": pooled_geomean_block(sqc, P["gblock"].to_numpy()),
                                      "median_all": float(np.nanmedian(-sqc))}
            fw["plateau_disk7"]["geomean_equals_pooled_lap_h"] = bool(abs(fw["plateau_disk7"]["geomean"]["value"] / abs(pp["lap_h"]) - 1) < 1e-5)
        if site in PERRON_2012:
            ref = PERRON_2012[site]["curvature_per_m"]
            for k in ("divide_square11", "divide_disk7", "plateau_disk7", "plateau_square11"):
                if k in fw:
                    for stat in list(fw[k]):
                        v = fw[k][stat]
                        if isinstance(v, dict) and "value" in v:
                            v["ratio_to_perron"] = v["value"] / ref
                            v["ratio_ci68"] = [v["ci68"][0] / ref, v["ci68"][1] / ref]
                            v["in_range"] = bool(R11_RANGE[0] <= v["value"] / ref <= R11_RANGE[1])
        out["four_way"][site] = fw
        print(site, "four-way:", {k: {s: (round(v["value"], 5), v.get("ratio_to_perron") and round(v["ratio_to_perron"], 2)) for s, v in fw[k].items() if isinstance(v, dict) and "value" in v}
                                  for k in ("divide_square11", "divide_disk7", "plateau_disk7", "plateau_square11") if k in fw}, f"[{time.time() - t0:.0f} s]", flush=True)
    # R11P with the reading
    for site in ("GM", "AP"):
        p = out["pooled"][site]["primary"]["r7"]
        ratio = p.get("ratio_to_perron")
        fw = out["four_way"][site]
        div7 = fw["divide_disk7"]["median"]
        out["rules"].setdefault("R11P", {})[site] = {"ratio": ratio, "ratio_ci68": p.get("ratio_ci68"), "range": list(R11_RANGE), "pass": bool(ratio is not None and R11_RANGE[0] <= ratio <= R11_RANGE[1]),
                                                     "divide_disk7_median_ratio": div7.get("ratio_to_perron"), "divide_disk7_in_range": div7.get("in_range"),
                                                     "radii_in_range": [r for r in PERRON_RADII_M if out["pooled"][site]["primary"][f"r{r:g}"].get("ratio_to_perron") and
                                                                        R11_RANGE[0] <= out["pooled"][site]["primary"][f"r{r:g}"]["ratio_to_perron"] <= R11_RANGE[1]],
                                                     "perron_survey_group": {k: out["pooled"][site][PERRON_SURVEY_GROUP[site]][k].get("ratio_to_perron") for k in out["pooled"][site][PERRON_SURVEY_GROUP[site]] if k.startswith("r")}}
    rp = out["rules"]["R11P"]
    if rp["AP"]["pass"] and not rp["AP"]["divide_disk7_in_range"]:
        rd = "pixel selection: at AP the plateau estimate passes while the divide-line estimate at the same window (7 m disk) does not"
    elif not rp["AP"]["pass"] and not rp["AP"]["divide_disk7_in_range"]:
        rd = "neither passes at AP: location, survey or method detail (the report says which the evidence favours)"
    elif rp["AP"]["pass"] and rp["AP"]["divide_disk7_in_range"]:
        rd = "both pass at AP at the 7 m disk: the addendum 19 AP difference was the window (11 m square vs 7 m disk), not pixel selection"
    else:
        rd = "the divide-line estimate passes at AP while the plateau estimate does not (no reading written)"
    out["rules"]["R11P"]["reading_AP"] = rd
    out["rules"]["R11P"]["perron_reproduced_both"] = bool(rp["GM"]["pass"] and rp["AP"]["pass"])
    out["rules"]["R11P"]["lower_end_D0"] = {s: ("D0_P" if rp[s]["pass"] else "both D0_P and D0_N reported; lower-end rules without a verdict") for s in ("GM", "AP")} | {"TTV": "D0_P (not gated)"}
    # ------------------------------------------------------------------ survey pairs (2.7)
    out["survey_pairs"] = {}
    for site, pairs in SURVEY_PAIRS.items():
        rows = []
        for old, new in pairs:
            o, n = out["tiles"].get(old), out["tiles"].get(new)
            if not (o and n):
                continue
            rows.append({"pair": f"{old}/{new}", "D0_P_old": o["D0_P"], "D0_P_new": n["D0_P"], "ratio_P_new_over_old": (n["D0_P"] / o["D0_P"]) if (o["D0_P"] and n["D0_P"]) else None,
                         "D0_N_old": o["D0_N_A"], "D0_N_new": n["D0_N_A"], "ratio_N_new_over_old": (n["D0_N_A"] / o["D0_N_A"]) if (o["D0_N_A"] and n["D0_N_A"]) else None,
                         "plateau_old": o["plateau_range_m2"], "plateau_new": n["plateau_range_m2"]})
        out["survey_pairs"][site] = {"pairs": rows, "ratio_P_median": med([r["ratio_P_new_over_old"] for r in rows]), "ratio_N_median": med([r["ratio_N_new_over_old"] for r in rows])}
    # ------------------------------------------------------------------ E* on the new pixels (pooled S_c fit, as addendum 19)
    out["e_star"] = {}
    for site in ("GM", "AP", "TTV"):
        tl = [t for t in PRIMARY[site] if t in recs and recs[t]["D0_N"]["usable"]]
        frames = []
        for i, t in enumerate(tl):
            d = pd.read_parquet(out_path("tiles", f"{t}_divide_pixels.parquet"), columns=["segment", "C_Lf", "L_H_net_1e3", "R_net_1e3", "used_net_1e3"])
            d = d[d["used_net_1e3"]]
            frames.append(pd.DataFrame({"tile": t, "C": d["C_Lf"].to_numpy(), "L": d["L_H_net_1e3"].to_numpy(), "S": (d["R_net_1e3"] / d["L_H_net_1e3"]).to_numpy(), "gseg": d["segment"].to_numpy() + 10_000_000 * i}))
        pool = pd.concat(frames, ignore_index=True)
        C, L, S, seg = (pool[k].to_numpy() for k in ("C", "L", "S", "gseg"))
        ok = np.isfinite(C) & np.isfinite(L) & np.isfinite(S) & (C > 0) & (L > 0) & (S > 0)
        C, L, S, seg = C[ok], L[ok], S[ok], seg[ok]
        fit = fit_S_c(C, L, S)
        uniq, inv = np.unique(seg, return_inverse=True)
        r = np.random.default_rng(SEED)
        bs = np.array([fit_S_c(C, L, S, np.bincount(r.integers(0, uniq.size, uniq.size), minlength=uniq.size).astype(float)[inv])["S_c"] for _ in range(400)])
        fit["S_c_ci68"] = [float(np.percentile(bs, 16)), float(np.percentile(bs, 84))]
        fit["inside_own_ci68"] = bool(fit["S_c_ci68"][0] <= fit["S_c"] <= fit["S_c_ci68"][1])
        bands = lambda e: {"lt_10": float(np.mean(e < E_STAR_BANDS[0])), "10_30": float(np.mean((e >= E_STAR_BANDS[0]) & (e <= E_STAR_BANDS[1]))), "gt_30": float(np.mean(e > E_STAR_BANDS[1]))}  # noqa: E731
        e_ref, e_fit = 2 * C * L / S_C_REFERENCE, 2 * C * L / fit["S_c"]
        out["e_star"][site] = {"n_pixels": int(C.size), "S_c_fit": fit, "S_c_reference": S_C_REFERENCE, "E_star_at_reference": {"median": float(np.median(e_ref)), "bands": bands(e_ref)},
                               "E_star_at_fit": {"median": float(np.median(e_fit)), "bands": bands(e_fit), "applies": fit["inside_own_ci68"]},
                               "per_tile": {t: {"E_star_ref_median": float(np.median(e_ref[pool["tile"].to_numpy()[ok] == t])), "E_star_fit_median": float(np.median(e_fit[pool["tile"].to_numpy()[ok] == t]))} for t in tl}}
    # ------------------------------------------------------------------ TTV anchor (addendum 19 decision 14, with D0_P)
    ttv = [out["tiles"][t] for t in PRIMARY["TTV"] if t in out["tiles"]]
    L_S = a19["scaling_break"]["TTV"]["L_S_m"]
    anc = {}
    for k, vals in (("D0_P", [x["D0_P"] for x in ttv]), ("D0_N_A", [x["D0_N_A"] for x in ttv if x["usable_N"]]), (f"D0_N_S_{L_S}m", [x["D0_N_by_L"].get(L_S) for x in ttv if x["usable_N"]]),
                    ("D_R", [x["D_R"] for x in ttv if x["usable_N"]])):
        v = med(vals)
        anc[k] = {"site_median": v, "n_tiles": len([x for x in vals if x]), "over_2.5e-3": v / TTV_ANCHOR["bedrock_convention_m2_per_yr"] if v else None, "over_5e-3": v / TTV_ANCHOR["soil_flux_convention_m2_per_yr"] if v else None}
    out["ttv_anchor"] = {"anchor": TTV_ANCHOR, "ratios": anc}
    out["brackets"] = {t: {"bracket_P": x["bracket_P"], "bracket_N": x["bracket_N"], "bracket_P_block": x["bracket_P_block"], "D_C_source": x["D_C_source"], "usable_N": x["usable_N"]} for t, x in out["tiles"].items()}
    out["a18_fits_for_figures"] = {t: {m: (fits18.get(f"{t}:primary") or {}).get(m, {}).get("D") for m in ("AD", "P4", "P3", "A1")} for t in out["tiles"]}
    out["constants"] = {"N_BINS": N_BINS, "R_PRIMARY": PERRON_R_PRIMARY, "D0_MIN_PIXELS": D0_MIN_PIXELS, "FIG_DIR": FIG_DIR}
    out["wall_s"] = time.time() - t0
    dump(out, out_path("site", "partA.json"))
    print("R1", {s: (v["n"], v["N"], v["pass"]) for s, v in out["rules"]["R1"].items()}, "\nR13", {s: (v["n"], v["N"], v["pooled_plateau"], v["holds"]) for s, v in out["rules"]["R13"].items()},
          "\nR11P", {s: (v["ratio"], v["pass"]) for s, v in out["rules"]["R11P"].items() if isinstance(v, dict) and "ratio" in v}, out["rules"]["R11P"]["reading_AP"], flush=True)


if __name__ == "__main__":
    main()
