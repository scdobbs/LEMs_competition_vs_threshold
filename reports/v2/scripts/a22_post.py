#!/usr/bin/env python3
"""Addendum 22 post-processing: the head metrics of decision 2.4 on every stored 12 m surface, the readings of section 6, the hypothesis
table inputs, the AD check geometry (2.5), the addendum 21 head verdicts for the comparison appendix, and the report tables.

Inputs: <run>/tiles/<tile>.json, <tile>_products.npz (obs, PF-A, PF-C, F2-relief, F2-phys); <run>/profiles/<model>_<tile>.json,
_products.npz (AD, SJ-C, P4). Outputs <run>/post/metrics.json, rules.json, tables.md, ad_check.json.
usage: a22_post.py
"""
from __future__ import annotations

import os
import sys
import time
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a22_common import (  # noqa: E402
    A_F,
    A_F_PLACEMENT_MEDIAN,
    BLOCK_M,
    C_F,
    C_F_PLACEMENT_MEDIAN,
    D_PHYS,
    D_RELIEF_RANGE,
    FITTED,
    FITTED_MODELS,
    K_PHYS_RANGE,
    MODELS,
    PLAIN,
    T_PRIMARY_M,
    TILES,
    TOL_VERDICT_RANGE,
    TOLERANCES_M,
    WHAT,
    a21_path,
    channel_length_m,
    classify_extra,
    criterion_sets,
    dump,
    json_ready,
    load_json,
    match_heads,
    out_path,
    reading_heads,
    unpack,
    valley_offset,
)

SENS = {"primary": ("a2", C_F), "area_rule": ("a0", A_F), "placement_median_c": ("a2", C_F_PLACEMENT_MEDIAN), "placement_median_a": ("a0", A_F_PLACEMENT_MEDIAN)}


def load_tile(tile: str) -> dict[str, Any]:
    rec = load_json(out_path("tiles", f"{tile}.json"))
    P = dict(np.load(out_path("tiles", f"{tile}_products.npz")))
    models = {k: dict(v) for k, v in rec["models"].items()}
    for name in ("AD", "SJ_C", "P4"):
        pj = out_path("profiles", f"{name}_{tile}.json")
        if not os.path.exists(pj):
            continue
        pr = load_json(pj)
        o = pr["optimum_poly"]
        thr = o.get("C", o.get("A_crit"))
        models[name] = {"fitted": FITTED[name], "K": o["K"], "D": o["D"], "K_over_K0": o["K"] / rec["K0"], "where": o["where"], "mse_poly": o["mse"], "mse_all_at_poly_fit": o["mse_other"],
                        "threshold": thr, "threshold_a21": pr["mask_check"]["a21_threshold"] if "a21_threshold" in pr["mask_check"] else None, "n_channel": o.get("n_channel"),
                        "a21_whole_tile_fit": {"K": pr["optimum_all"]["K"], "D": pr["optimum_all"]["D"], "mse_all": pr["optimum_all"]["mse"], "mse_poly": pr["optimum_all"]["mse_other"],
                                               "threshold": pr["optimum_all"].get("C", pr["optimum_all"].get("A_crit", pr["optimum_all"].get("r")))},
                        "mask_check": pr["mask_check"], "D_at_bound": pr.get("D_at_bound_poly"), "identifiability_poly": pr.get("identifiability_poly"), "n_evaluations": pr.get("n_evaluations"),
                        "s_per_evaluation": pr.get("s_per_evaluation"), "wall_s": pr.get("wall_s")}
        P.update(np.load(out_path("profiles", f"{name}_{tile}_products.npz")))
    models["obs"] = {"fitted": "none", "K": None, "D": None, "K_over_K0": None, "mse_poly": 0.0, "mse_all_at_poly_fit": 0.0, "threshold": None}
    return {"rec": rec, "P": P, "models": models}


def head_cells(P: dict[str, np.ndarray], name: str, key: str, C: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ch, hd = criterion_sets(P[f"{name}_M_{key}"], P[f"{name}_Mdon_{key}"], C)
    return ch, hd, P["poly12_centre"]


def tile_metrics(tile: str, T: dict[str, Any], sens: str) -> dict[str, Any]:
    rec, P = T["rec"], T["P"]
    key, C = SENS[sens]
    shape = tuple(rec["inputs"]["shape"])
    poly = unpack(P["poly_1m"], shape)
    inside12 = P["poly12_centre"]
    nb = inside12.shape[0]
    pr, pc = P["pick_rc"]
    inpoly = poly[pr, pc] if pr.size else np.zeros(0, bool)
    mapped_xy = np.stack([pr[inpoly], pc[inpoly]], axis=1).astype(float) if pr.size else np.zeros((0, 2))
    mapped_cells = [(min(int(r) // BLOCK_M, nb - 1), min(int(c) // BLOCK_M, nb - 1)) for r, c in mapped_xy]      # a pick in the 4 m remainder strip takes the last block
    mapped_ids = [int(i) for i in P["pick_ids"][inpoly]] if pr.size else []
    FD_obs = P["obs_FD12"]
    steep = rec["steep_threshold_obs"]["slope"]
    md_len = rec["md_traced"]["length_in_tile_and_polygon_m"]
    out: dict[str, Any] = {"tile": tile, "sensitivity": sens, "alpha_key": key, "C": C, "n_mapped": int(mapped_xy.shape[0]), "mapped_ids": mapped_ids, "md_length_m": md_len, "models": {}}
    for name in MODELS:
        if f"{name}_M_{key}" not in P:
            continue
        ch, hd, _ = head_cells(P, name, key, C)
        hr, hc = np.nonzero(hd & inside12)
        model_xy = np.stack([hr * BLOCK_M + BLOCK_M // 2, hc * BLOCK_M + BLOCK_M // 2], axis=1).astype(float)
        model_cells = [(int(r), int(c)) for r, c in zip(hr, hc, strict=True)]
        mrec: dict[str, Any] = {"n_heads_tile": int(hd.sum()), "n_heads_in_polygon": int(hr.size), "n_channel_in_polygon": int((ch & inside12).sum()),
                                "channel_length_m": channel_length_m(ch, P[f"{name}_FD12"], inside12), "by_t": {}}
        mrec["length_ratio"] = mrec["channel_length_m"] / md_len if md_len else None
        for t in TOLERANCES_M:
            mt = match_heads(mapped_xy, model_xy, t)
            offs = [valley_offset(FD_obs, mapped_cells[i], model_cells[j]) | {"mapped_id": mapped_ids[i], "dist_m": d} for i, j, d in mt["pairs"]]
            vals = [o["offset_m"] for o in offs if o["offset_m"] is not None]
            kinds = {k: sum(1 for o in offs if o["kind"] == k) for k in ("too short", "too long", "neighbouring hollow", "same cell")}
            extra_cells = [model_cells[j] for j in range(len(model_cells)) if j not in {p[1] for p in mt["pairs"]}]
            cls = classify_extra(P[f"{name}_slope12"], steep, extra_cells, nb)
            mrec["by_t"][f"{t:g}"] = {"n_matched": mt["n_matched"], "n_missed": mt["n_missed"], "n_extra": mt["n_extra"], "share_found": mt["share_found"], "share_real": mt["share_real"],
                                     "matched_dist_m": mt["matched_dist_m"], "pairs": [{"mapped_id": mapped_ids[i], "model_cell": list(model_cells[j]), "dist_m": d} for i, j, d in mt["pairs"]],
                                     "offsets": offs, "offset_values_m": vals, "median_offset_m": float(np.median(vals)) if vals else None, "offset_kinds": kinds,
                                     "extra_classes": {k: cls[k] for k in ("too-steep wall", "tile edge", "other")}, "extra_cells": cls["cells"],
                                     "missed_ids": [mapped_ids[i] for i in range(len(mapped_ids)) if i not in {p[0] for p in mt["pairs"]}]}
        out["models"][name] = mrec
    return out


def pool(tiles_m: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"n_mapped": int(sum(t["n_mapped"] for t in tiles_m)), "md_length_m": float(sum(t["md_length_m"] for t in tiles_m)), "models": {}}
    for name in MODELS:
        if not all(name in t["models"] for t in tiles_m):
            continue
        ms = [t["models"][name] for t in tiles_m]
        rec: dict[str, Any] = {"n_heads_in_polygon": int(sum(m["n_heads_in_polygon"] for m in ms)), "channel_length_m": float(sum(m["channel_length_m"] for m in ms)), "by_t": {}}
        rec["length_ratio"] = rec["channel_length_m"] / out["md_length_m"] if out["md_length_m"] else None
        for t in TOLERANCES_M:
            k = f"{t:g}"
            bt = [m["by_t"][k] for m in ms]
            n_m, n_x, n_s = sum(b["n_matched"] for b in bt), sum(b["n_extra"] for b in bt), sum(b["n_missed"] for b in bt)
            vals = [v for b in bt for v in b["offset_values_m"]]
            kinds = {kk: sum(b["offset_kinds"][kk] for b in bt) for kk in ("too short", "too long", "neighbouring hollow", "same cell")}
            n_pairs = n_m
            r = {"n_matched": n_m, "n_missed": n_s, "n_extra": n_x, "share_found": n_m / out["n_mapped"] if out["n_mapped"] else None,
                 "share_real": n_m / (n_m + n_x) if (n_m + n_x) else None, "median_offset_m": float(np.median(vals)) if vals else None,
                 "offset_q25_q75_m": [float(np.percentile(vals, 25)), float(np.percentile(vals, 75))] if vals else None, "offset_values_m": vals, "offset_kinds": kinds,
                 "share_too_short": kinds["too short"] / n_pairs if n_pairs else None, "share_too_long": kinds["too long"] / n_pairs if n_pairs else None,
                 "share_neighbouring_hollow": kinds["neighbouring hollow"] / n_pairs if n_pairs else None, "share_same_cell": kinds["same cell"] / n_pairs if n_pairs else None,
                 "matched_dist_median_m": float(np.median([d for b in bt for d in b["matched_dist_m"]])) if n_m else None,
                 "extra_classes": {kk: sum(b["extra_classes"][kk] for b in bt) for kk in ("too-steep wall", "tile edge", "other")}}
            r["reading"] = reading_heads(r["share_found"], r["share_real"], r["median_offset_m"])
            rec["by_t"][k] = r
        out["models"][name] = rec
    obs = out["models"].get("obs")
    if obs:
        for rec in out["models"].values():
            rec["length_ratio_vs_obs"] = rec["channel_length_m"] / obs["channel_length_m"] if obs["channel_length_m"] else None
            for k, r in rec["by_t"].items():
                o = obs["by_t"][k]
                r["share_found_vs_obs"] = r["share_found"] / o["share_found"] if o["share_found"] else None
                r["share_real_vs_obs"] = r["share_real"] / o["share_real"] if o["share_real"] else None
    return out


def readings(T: dict[str, dict[str, Any]], pooled: dict[str, Any]) -> dict[str, Any]:
    tp = f"{T_PRIMARY_M:g}"
    R: dict[str, Any] = {"t_primary": T_PRIMARY_M, "heads": {}, "misfit": {}, "K_physical": {}, "D_relief": {}, "tolerance_stability": {}}
    for name, rec in pooled["models"].items():
        R["heads"][name] = rec["by_t"][tp]["reading"] | {"share_found": rec["by_t"][tp]["share_found"], "share_real": rec["by_t"][tp]["share_real"], "median_offset_m": rec["by_t"][tp]["median_offset_m"]}
        verdicts = {k: rec["by_t"][k]["reading"]["reproduces_mapped_heads"] for k in rec["by_t"]}
        inrange = {k: v for k, v in verdicts.items() if TOL_VERDICT_RANGE[0] <= float(k) <= TOL_VERDICT_RANGE[1]}
        R["tolerance_stability"][name] = {"verdict_by_t": verdicts, "changes_between_15_and_40_m": len(set(inrange.values())) > 1}
    for name in FITTED_MODELS + ["F2_relief", "F2_phys"]:
        per = {}
        for tile in TILES:
            m, ad = T[tile]["models"].get(name), T[tile]["models"].get("AD")
            if m is None or ad is None:
                continue
            per[tile] = {"mse_poly": m["mse_poly"], "mse_poly_AD": ad["mse_poly"], "ratio_to_AD": m["mse_poly"] / ad["mse_poly"], "beats_AD": bool(m["mse_poly"] < ad["mse_poly"])}
        R["misfit"][name] = {"per_tile": per, "beats_AD_on_all_three": bool(per) and all(v["beats_AD"] for v in per.values()), "n_tiles_beats_AD": sum(v["beats_AD"] for v in per.values())}
    for name in FITTED_MODELS:
        kk, dd = {}, {}
        for tile in TILES:
            m = T[tile]["models"].get(name)
            if m is None:
                continue
            kk[tile] = {"K_over_K0": m["K_over_K0"], "physical": bool(K_PHYS_RANGE[0] <= m["K_over_K0"] <= K_PHYS_RANGE[1])}
            Dr = T[tile]["rec"]["D_relief"]
            dd[tile] = {"D": m["D"], "D_over_D_relief": m["D"] / Dr, "D_over_D_phys": m["D"] / D_PHYS, "explained_by_relief": bool(D_RELIEF_RANGE[0] <= m["D"] / Dr <= D_RELIEF_RANGE[1])}
        R["K_physical"][name] = {"per_tile": kk, "holds_on_all_three": bool(kk) and all(v["physical"] for v in kk.values())}
        R["D_relief"][name] = {"per_tile": dd, "holds_on_all_three": bool(dd) and all(v["explained_by_relief"] for v in dd.values())}
    return R


def a21_head_verdicts() -> dict[str, Any]:
    """The addendum 21 H1 measures (median pick-to-channel distance, heads / picks) pooled over TTV1-TTV3 at alpha = 2, unchanged."""
    out: dict[str, Any] = {}
    map21 = {"obs": "obs", "AD": "AD", "PF_A": "PF_A", "PF_C": "PF_C", "P4": "P4", "F2_relief": "F2_relief", "F2_phys": "F2_phys"}
    for name, k21 in map21.items():
        d, nh, npk = [], 0, 0
        for tile in TILES:
            e = load_json(a21_path("tiles", f"{tile}.json"))["extraction"][k21]["a2"]
            d += list(e["pick_dist_m"]["values"].values())
            nh += e["n_heads_in_polygon"]
            npk += e["n_picks_in_polygon"]
        out[name] = {"median_pick_to_channel_m": float(np.median(d)), "heads_in_polygon": nh, "picks": npk, "ratio": nh / npk, "H1": {"holds": bool(np.median(d) <= 25 and 0.67 <= nh / npk <= 1.5)}}
    d, nh, npk = [], 0, 0
    for tile in TILES:
        e = load_json(a21_path("sjc", f"{tile}.json"))["extraction"]["a2"]
        d += list(e["pick_dist_m"]["values"].values())
        nh += e["n_heads_in_polygon"]
        npk += e["n_picks_in_polygon"]
    out["SJ_C"] = {"median_pick_to_channel_m": float(np.median(d)), "heads_in_polygon": nh, "picks": npk, "ratio": nh / npk, "H1": {"holds": bool(np.median(d) <= 25 and 0.67 <= nh / npk <= 1.5)}}
    return out


def ad_check(T: dict[str, dict[str, Any]], M: dict[str, Any]) -> dict[str, Any]:
    """Decision 2.5 on TTV2: the AD extra heads classified 'too-steep wall', their principal axis, and a transect across it (12 m cells)."""
    tile = "TTV2"
    P = T[tile]["P"]
    if "AD" not in M["per_tile"]["primary"][tile]["models"]:
        return {"tile": tile, "missing": "AD products"}
    bt = M["per_tile"]["primary"][tile]["models"]["AD"]["by_t"][f"{T_PRIMARY_M:g}"]
    cells = np.array([[c["r"], c["c"]] for c in bt["extra_cells"]], dtype=float)
    wall = np.array([[c["r"], c["c"]] for c in bt["extra_cells"] if c["class"] == "too-steep wall"], dtype=float)
    use = wall if wall.shape[0] >= 3 else cells
    out: dict[str, Any] = {"tile": tile, "n_extra": int(cells.shape[0]), "n_wall": int(wall.shape[0]), "classes": bt["extra_classes"]}
    if use.shape[0] < 2:
        return out
    cen = use.mean(axis=0)
    u, s, vt = np.linalg.svd(use - cen, full_matrices=False)
    axis = vt[0]
    perp = np.array([-axis[1], axis[0]])
    proj = (use - cen) @ axis
    out.update({"centroid_rc": cen.tolist(), "axis_rc": axis.tolist(), "spread_along_m": float((proj.max() - proj.min()) * BLOCK_M), "rms_across_m": float(np.sqrt(np.mean(((use - cen) @ perp) ** 2)) * BLOCK_M),
                "line_end_cells": [(cen + axis * proj.min()).tolist(), (cen + axis * proj.max()).tolist()]})
    nb = P["obs_Zf12"].shape[0]
    half = 20
    ts = np.arange(-half, half + 1)
    pts = cen[None, :] + ts[:, None] * perp[None, :]
    rr, cc = np.clip(np.round(pts[:, 0]).astype(int), 0, nb - 1), np.clip(np.round(pts[:, 1]).astype(int), 0, nb - 1)
    prof = {"distance_m": (ts * BLOCK_M).tolist(), "cells": np.stack([rr, cc], axis=1).tolist()}
    for name in ("obs", "AD"):
        prof[f"z_{name}"] = P[f"{name}_Zf12"][rr, cc].astype(float).tolist()
        prof[f"slope_{name}"] = P[f"{name}_slope12"][rr, cc].astype(float).tolist()
    out["transect"] = prof
    sl_obs, sl_ad = P["obs_slope12"], P["AD_slope12"]
    wr, wc = use[:, 0].astype(int), use[:, 1].astype(int)
    out["slope_at_wall_heads"] = {"AD_median": float(np.nanmedian(sl_ad[wr, wc])), "obs_median": float(np.nanmedian(sl_obs[wr, wc])), "obs_95th_in_polygon": T[tile]["rec"]["steep_threshold_obs"]["slope"]}
    out["drainage_area_at_wall_heads_m2"] = {"AD_median": float(np.nanmedian(P["AD_A12"][wr, wc])), "obs_median_same_cells": float(np.nanmedian(P["obs_A12"][wr, wc]))}
    out["elevation_difference_at_wall_heads_m"] = {"AD_minus_obs_median": float(np.nanmedian(P["AD_Zf12"][wr, wc] - P["obs_Zf12"][wr, wc]))}
    return out


def fmt(v, nd=2, none="–"):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return none
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if abs(v) < 0.01 and v != 0:
        return f"{v:.2e}"
    return f"{v:.{nd}f}"


def tables(T: dict[str, dict[str, Any]], M: dict[str, Any], R: dict[str, Any], V21: dict[str, Any], ADC: dict[str, Any]) -> str:
    tp = f"{T_PRIMARY_M:g}"
    L: list[str] = []
    L.append("### Model key\n")
    L.append("| Plain name | What it is | Fitted parameters |\n|---|---|---|")
    for n in MODELS:
        L.append(f"| **{PLAIN[n]}** | {WHAT[n]} | {FITTED[n]} |")
    L.append("\n### Polygon cells per tile\n")
    L.append("| tile | 1 m cells in the M&D polygon | valid 1 m cells | share | profiler cells in polygon | 12 m cells (centre inside) | mapped heads inside | M&D traced length inside (m) |\n|---|---|---|---|---|---|---|---|")
    for t in TILES:
        r = T[t]["rec"]
        L.append(f"| {t} | {r['polygon']['n_cells_1m']:,} | {r['polygon']['n_valid_1m']:,} | {r['polygon']['share_of_tile']:.2f} | {r['polygon']['n_profiler_cells_in_polygon']:,} | {r['polygon']['n_cells_12m_centre_inside']:,} | "
                 f"{r['picks']['n_in_polygon']} | {r['md_traced']['length_in_tile_and_polygon_m']:,.0f} |")
    L.append("\n### Section 4 table: one row per model and tile (misfit inside the M&D area; t = 25 m)\n")
    L.append("| model | tile | misfit in M&D area (m²) | ÷ AD | whole-tile misfit at these parameters (m²) | K ÷ K₀ | D (m²/yr) | D ÷ physical D | D ÷ relief D | fitted threshold (M&D value) | matched | missed | extra | share found | share real | median offset (m) | channel length ÷ mapped |\n"
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for n in MODELS:
        for t in TILES:
            m = T[t]["models"].get(n)
            if m is None:
                continue
            ad = T[t]["models"].get("AD", {}).get("mse_poly", float("nan"))
            Dr = T[t]["rec"]["D_relief"]
            h = M["per_tile"]["primary"][t]["models"][n]
            b = h["by_t"][tp]
            thr = "–"
            if n == "SJ_C":
                thr = f"{m['threshold']:,.0f} m² area × slope² (876)"
            elif n == "P4":
                thr = f"{m['threshold']:,.0f} m² area (6,120)"
            elif n in ("PF_C", "F2_relief", "F2_phys"):
                thr = "876 m² area × slope² (fixed)"
            elif n == "PF_A":
                thr = "6,120 m² area (fixed)"
            L.append(f"| {PLAIN[n]} | {t} | {fmt(m['mse_poly'])} | {fmt(m['mse_poly'] / ad) if n != 'obs' else '–'} | {fmt(m['mse_all_at_poly_fit'])} | {fmt(m['K_over_K0'])} | "
                     f"{fmt(m['D'], 4)} | {fmt(m['D'] / D_PHYS if m['D'] else None, 1)} | {fmt(m['D'] / Dr if m['D'] else None)} | {thr} | {b['n_matched']} | {b['n_missed']} | {b['n_extra']} | "
                     f"{fmt(b['share_found'])} | {fmt(b['share_real'])} | {fmt(b['median_offset_m'], 0)} | {fmt(h['length_ratio'])} |")
    L.append("\n### Pooled over TTV1–TTV3 at t = 25 m (section 6 head reading)\n")
    L.append("| model | mapped | matched | missed | extra | share found | ÷ real landscape | share real | ÷ real landscape | median offset (m) | q25–q75 | too short | too long | neighbouring hollow | same cell | channel length (m) | ÷ mapped | ÷ real landscape | extra: too-steep wall | tile edge | other | reproduces the mapped heads |\n"
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    pm = M["pooled"]["primary"]
    for n in MODELS:
        r = pm["models"].get(n)
        if r is None:
            continue
        b = r["by_t"][tp]
        q = b["offset_q25_q75_m"]
        L.append(f"| {PLAIN[n]} | {pm['n_mapped']} | {b['n_matched']} | {b['n_missed']} | {b['n_extra']} | {fmt(b['share_found'])} | {fmt(b.get('share_found_vs_obs'))} | {fmt(b['share_real'])} | {fmt(b.get('share_real_vs_obs'))} | "
                 f"{fmt(b['median_offset_m'], 0)} | {fmt(q[0], 0) if q else '–'}–{fmt(q[1], 0) if q else '–'} | {fmt(b['share_too_short'])} | {fmt(b['share_too_long'])} | {fmt(b['share_neighbouring_hollow'])} | {fmt(b['share_same_cell'])} | "
                 f"{r['channel_length_m']:,.0f} | {fmt(r['length_ratio'])} | {fmt(r.get('length_ratio_vs_obs'))} | {b['extra_classes']['too-steep wall']} | {b['extra_classes']['tile edge']} | {b['extra_classes']['other']} | {fmt(b['reading']['reproduces_mapped_heads'])} |")
    L.append(f"\nM&D traced channel length inside the polygon on the three tiles: {pm['md_length_m']:,.0f} m.\n")
    L.append("\n### Every tolerance (pooled; share found / share real / median offset m / verdict)\n")
    L.append("| model | " + " | ".join(f"t = {t:g} m" for t in TOLERANCES_M) + " | verdict changes between 15 and 40 m |\n|---|" + "---|" * len(TOLERANCES_M) + "---|")
    for n in MODELS:
        r = pm["models"].get(n)
        if r is None:
            continue
        cells = []
        for t in TOLERANCES_M:
            b = r["by_t"][f"{t:g}"]
            cells.append(f"{fmt(b['share_found'])} / {fmt(b['share_real'])} / {fmt(b['median_offset_m'], 0)} / {'yes' if b['reading']['reproduces_mapped_heads'] else 'no'}")
        L.append(f"| {PLAIN[n]} | " + " | ".join(cells) + f" | {fmt(R['tolerance_stability'][n]['changes_between_15_and_40_m'])} |")
    L.append("\n### Sensitivities of the head finding (pooled, t = 25 m): share found / share real / median offset / extra\n")
    L.append("| model | area × slope² ≥ 876 m² (primary) | area ≥ 6,120 m² | area × slope² ≥ 817 m² (16-placement median) | area ≥ 6,228 m² (16-placement median) |\n|---|---|---|---|---|")
    for n in MODELS:
        cells = []
        for s in ("primary", "area_rule", "placement_median_c", "placement_median_a"):
            r = M["pooled"][s]["models"].get(n)
            if r is None:
                cells.append("–")
                continue
            b = r["by_t"][tp]
            cells.append(f"{fmt(b['share_found'])} / {fmt(b['share_real'])} / {fmt(b['median_offset_m'], 0)} / {b['n_extra']}")
        L.append(f"| {PLAIN[n]} | " + " | ".join(cells) + " |")
    L.append("\n### Misfit, K and D readings (section 6)\n")
    L.append("| model | misfit ÷ AD: TTV1 / TTV2 / TTV3 | beats AD on all three | K ÷ K₀: TTV1 / TTV2 / TTV3 | K physical (0.5–2 on all three) | D ÷ relief D: TTV1 / TTV2 / TTV3 | explained by relief (0.67–1.5 on all three) | D ÷ physical D: TTV1 / TTV2 / TTV3 |\n|---|---|---|---|---|---|---|---|")
    for n in FITTED_MODELS + ["F2_relief", "F2_phys"]:
        mi = R["misfit"].get(n, {}).get("per_tile", {})
        kk = R["K_physical"].get(n, {}).get("per_tile", {})
        dd = R["D_relief"].get(n, {}).get("per_tile", {})
        L.append(f"| {PLAIN[n]} | {' / '.join(fmt(mi[t]['ratio_to_AD']) for t in TILES if t in mi)} | {fmt(R['misfit'].get(n, {}).get('beats_AD_on_all_three'))} | "
                 f"{' / '.join(fmt(kk[t]['K_over_K0']) for t in TILES if t in kk) if kk else ' / '.join(fmt(T[t]['models'].get(n, {}).get('K_over_K0')) for t in TILES)} | {fmt(R['K_physical'].get(n, {}).get('holds_on_all_three'))} | "
                 f"{' / '.join(fmt(dd[t]['D_over_D_relief']) for t in TILES if t in dd) if dd else ' / '.join(fmt(T[t]['models'][n]['D'] / T[t]['rec']['D_relief']) for t in TILES if n in T[t]['models'])} | {fmt(R['D_relief'].get(n, {}).get('holds_on_all_three'))} | "
                 f"{' / '.join(fmt(dd[t]['D_over_D_phys'], 1) for t in TILES if t in dd) if dd else ' / '.join(fmt(T[t]['models'][n]['D'] / D_PHYS, 1) for t in TILES if n in T[t]['models'])} |")
    L.append("\n### Appendix: mask checks (mask = every cell reproduces the addendum 21 fit)\n")
    L.append("| tile | model | addendum 21 misfit (m²) | all-cells misfit here | relative difference | pass |\n|---|---|---|---|---|---|")
    for t in TILES:
        for k, c in T[t]["rec"]["mask_check"].items():
            L.append(f"| {t} | {k} | {c['a21_mse']:.6f} | {c['all_cells_mse']:.6f} | {c['rel_diff_mse']:.1e} | {fmt(c['pass'])} |")
        for n in ("AD", "SJ_C", "P4"):
            m = T[t]["models"].get(n)
            if m and "mask_check" in m:
                c = m["mask_check"]
                extra = f"; at the a21 threshold {c['rel_diff_at_a21_threshold']:.1e}" if c.get("rel_diff_at_a21_threshold") is not None else ""
                extra += f"; profile minimum below a21: {fmt(c.get('all_cells_min_below_a21'))}"
                L.append(f"| {t} | {n} (profile) | {c['a21_mse']:.6f} | {c['all_cells_mse']:.6f} | {c['rel_diff_mse']:.1e}{extra} | {fmt(c['pass'])} |")
    L.append("\n### Appendix: whole-tile misfits beside the polygon misfits\n")
    L.append("| tile | model | polygon misfit of the polygon fit | whole-tile misfit of the polygon fit | polygon misfit of the addendum 21 (whole-tile) fit | whole-tile misfit of the addendum 21 fit | a21 K ÷ K₀ | a21 D | polygon-fit K ÷ K₀ | polygon-fit D | a21 threshold | polygon-fit threshold |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
    for t in TILES:
        K0 = T[t]["rec"]["K0"]
        for n in MODELS:
            m = T[t]["models"].get(n)
            if m is None or n == "obs":
                continue
            w = m.get("a21_whole_tile_fit", {})
            L.append(f"| {t} | {PLAIN[n]} | {fmt(m['mse_poly'])} | {fmt(m['mse_all_at_poly_fit'])} | {fmt(w.get('mse_poly'))} | {fmt(w.get('mse_all'))} | {fmt(w.get('K', 0) / K0 if w.get('K') else None)} | {fmt(w.get('D'), 4)} | "
                     f"{fmt(m['K_over_K0'])} | {fmt(m['D'], 4)} | {fmt(m.get('threshold_a21'), 0) if n in ('SJ_C', 'P4') else '–'} | {fmt(m.get('threshold'), 0) if n in ('SJ_C', 'P4') else '–'} |")
    L.append("\n### Appendix: the addendum 21 head verdicts, unchanged (pooled TTV1–TTV3, α = 2; median pick-to-channel distance ≤ 25 m and heads ÷ picks in 0.67–1.5)\n")
    L.append("| model (plain name) | internal code | median pick-to-channel distance (m) | heads in polygon ÷ picks | addendum 21 H1 holds | addendum 22 reading (t = 25 m) |\n|---|---|---|---|---|---|")
    for n in MODELS:
        v = V21.get(n)
        if v is None:
            continue
        r22 = pm["models"][n]["by_t"][tp]["reading"]["reproduces_mapped_heads"] if n in pm["models"] else None
        L.append(f"| {PLAIN[n]} | {n} | {v['median_pick_to_channel_m']:.1f} | {v['ratio']:.2f} ({v['heads_in_polygon']}/{v['picks']}) | {fmt(v['H1']['holds'])} | {fmt(r22)} |")
    L.append("\n### Appendix: the AD check geometry (TTV2)\n")
    L.append(f"- extra AD heads at t = 25 m: {ADC.get('n_extra')}, of which on too-steep walls {ADC.get('n_wall')}; classes {ADC.get('classes')}")
    if "spread_along_m" in ADC:
        L.append(f"- principal axis of the wall heads: spread {ADC['spread_along_m']:.0f} m along, rms {ADC['rms_across_m']:.0f} m across; centroid cell {ADC['centroid_rc']}")
        L.append(f"- slope at the wall heads: AD median {ADC['slope_at_wall_heads']['AD_median']:.2f}, real landscape at the same cells {ADC['slope_at_wall_heads']['obs_median']:.2f} (95th percentile of the real landscape inside the polygon {ADC['slope_at_wall_heads']['obs_95th_in_polygon']:.2f})")
        L.append(f"- drainage area at the wall heads (12 m): AD median {ADC['drainage_area_at_wall_heads_m2']['AD_median']:,.0f} m², real landscape {ADC['drainage_area_at_wall_heads_m2']['obs_median_same_cells']:,.0f} m²; AD − real elevation median {ADC['elevation_difference_at_wall_heads_m']['AD_minus_obs_median']:.1f} m")
    return "\n".join(L) + "\n"


def main() -> None:
    t0 = time.time()
    T = {t: load_tile(t) for t in TILES}
    M: dict[str, Any] = {"per_tile": {}, "pooled": {}}
    for s in SENS:
        M["per_tile"][s] = {t: tile_metrics(t, T[t], s) for t in TILES}
        M["pooled"][s] = pool([M["per_tile"][s][t] for t in TILES])
        print(f"{s}: pooled at t = 25 m: " + "; ".join(f"{n} found {r['by_t']['25']['share_found']:.2f} real {r['by_t']['25']['share_real']:.2f} off {r['by_t']['25']['median_offset_m']}" for n, r in M["pooled"][s]["models"].items()) + f" [{time.time() - t0:.0f} s]", flush=True)
    R = readings(T, M["pooled"]["primary"])
    V21 = a21_head_verdicts()
    ADC = ad_check(T, M)
    fits = {t: {"K0": T[t]["rec"]["K0"], "D_relief": T[t]["rec"]["D_relief"], "D_relief_network": T[t]["rec"]["D_relief_network"], "polygon": T[t]["rec"]["polygon"], "md_traced": T[t]["rec"]["md_traced"],
                "steep_threshold_obs": T[t]["rec"]["steep_threshold_obs"], "mask_check": T[t]["rec"]["mask_check"], "a21_params_scored": T[t]["rec"]["a21_params_scored"],
                "models": {n: {k: v for k, v in m.items() if k != "identifiability_poly"} for n, m in T[t]["models"].items()}} for t in TILES}
    dump(json_ready({"fits": fits, "metrics": M}), out_path("post", "metrics.json"))
    dump(json_ready({"readings": R, "a21_head_verdicts": V21}), out_path("post", "rules.json"))
    dump(json_ready(ADC), out_path("post", "ad_check.json"))
    md = tables(T, M, R, V21, ADC)
    with open(out_path("post", "tables.md"), "w") as f:
        f.write(md)
    print(f"post: done [{time.time() - t0:.0f} s]", flush=True)


if __name__ == "__main__":
    main()
