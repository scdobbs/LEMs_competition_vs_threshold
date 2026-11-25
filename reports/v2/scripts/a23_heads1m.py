#!/usr/bin/env python3
"""Addendum 23, Part A item 4 (decision 2.4): the head-finder candidates on the 1 m lidar of the domain, calibrated and tested
across the two halves (TTV2 ground; TTV1 + TTV3 ground), with the reading H-A.

Candidates: multiple-flow-direction area (Freeman 1991, exponent 1.1) on the filled 1 m surface with the 10 m downstream slope;
D8 on the 1 m surface Gaussian-smoothed with sigma 3 m and 6 m (filled and routed again; area, slope and routing of the smoothed
surface); the addendum 22 12 m method as the reference. For each: the statistic T = area x slope^2, its upstream maximum M and the
donors' maximum Mdon; the threshold C is the median of T at the calibration picks (largest T within PICK_RADIUS_M of the pick); the
test half's heads (M >= C, Mdon < C, inside the test tiles' footprints, the domain and the polygon) are matched one-to-one to the
test half's picks at 10-50 m (25 m primary). Reported beside it, as sensitivities only: the picks' statistic at 1 and 5 m radius,
and the F1-optimal threshold on the calibration half. Outputs <run>/heads/heads1m.json, heads1m.npz, accepted.json.
usage: a23_heads1m.py
"""
from __future__ import annotations

import os
import sys
import time
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a23_common import (  # noqa: E402
    BLOCK_M,
    CANDIDATE_PLAIN,
    CANDIDATES,
    HA_SHARE_FOUND_MIN,
    HA_SHARE_REAL_MIN,
    HALVES,
    MFD_EXPONENT,
    PICK_RADIUS_M,
    PICK_RADIUS_SENS_M,
    T_PRIMARY_M,
    TOLERANCES_M,
    coarse_products,
    criterion_sets,
    dump,
    f1_optimal_threshold,
    fill_and_route,
    gaussian_smooth_nan,
    head_statistic_1m,
    json_ready,
    load_domain,
    match_heads,
    mfd_area,
    out_path,
    reading_ha,
    slurm_record,
    tile_footprint,
    value_near_picks,
)


def candidate_statistics(name: str, dom: dict[str, Any], t0: float) -> dict[str, Any]:
    Z, FD = dom["Z"], dom["FD"]
    if name == "mfd":
        Am = mfd_area(Z, FD, 1.0, MFD_EXPONENT)
        st = head_statistic_1m(Am, Z, FD, 1.0)
        out = {"A": Am, "Zf": Z, "FD": FD, "M": st["M"], "Mdon": st["Mdon"], "T": st["T"], "S": st["S"], "scale_m": 1.0, "origin_rc": (0, 0),
               "note": f"Freeman (1991) area, exponent {MFD_EXPONENT}; slope over 10 m downstream on the lidar D8"}
    elif name.startswith("d8s"):
        sigma = float(name[3:])
        Zs = gaussian_smooth_nan(Z, sigma)
        fr = fill_and_route(Zs, 1.0)
        st = head_statistic_1m(fr["A"], fr["Zf"], fr["FD"], 1.0)
        out = {"A": fr["A"], "Zf": fr["Zf"], "FD": fr["FD"], "M": st["M"], "Mdon": st["Mdon"], "T": st["T"], "S": st["S"], "scale_m": 1.0, "origin_rc": (0, 0),
               "note": f"Gaussian sigma {sigma:g} m (NaN-aware), pit-filled, D8 (data-edge cells are outlets); slope over 10 m downstream on that D8"}
    elif name == "ref12":
        P = coarse_products(Z, 1.0)
        S = P["S12"].astype(float)
        T = np.where(np.isfinite(P["A12"]) & np.isfinite(S), P["A12"] * np.where(S > 0, S, 0.0) ** 2, np.nan)
        out = {"A": P["A12"].astype(float), "Zf": P["Zf12"].astype(float), "FD": P["FD12"], "M": P["M_a2"], "Mdon": P["Mdon_a2"], "T": T, "S": S, "scale_m": float(BLOCK_M),
               "origin_rc": (0, 0), "note": "the addendum 22 method: 12 m block means on the box lattice, filled, D8, slope to the largest-area donor"}
    else:
        raise ValueError(name)
    print(f"{name}: statistics ready; finite M {np.isfinite(out['M']).sum():,} [{time.time() - t0:.0f} s]", flush=True)
    return out


def head_cells_1m(st: dict[str, Any], C: float, keep_1m: np.ndarray) -> np.ndarray:
    """Heads at C as (row, col) on the 1 m lattice, kept where their 1 m position is inside ``keep_1m``."""
    _ch, hd = criterion_sets(st["M"], st["Mdon"], C)
    hr, hc = np.nonzero(hd)
    b = st["scale_m"]
    if b != 1.0:
        hr, hc = (hr * int(b) + int(b) // 2), (hc * int(b) + int(b) // 2)
    ok = (hr < keep_1m.shape[0]) & (hc < keep_1m.shape[1])
    hr, hc = hr[ok], hc[ok]
    ok = keep_1m[hr, hc]
    return np.stack([hr[ok], hc[ok]], axis=1).astype(float)


def picks_statistic(st: dict[str, Any], rows: np.ndarray, cols: np.ndarray, radius_m: float) -> dict[str, np.ndarray]:
    b = st["scale_m"]
    if b == 1.0:
        return value_near_picks(st["T"], st["A"], rows, cols, radius_m)
    # the 12 m reference: the pick's block and its 3 x 3 neighbourhood (one block = 12 m), as addendum 21 decision 1.2
    rb, cb = rows // int(b), cols // int(b)
    return value_near_picks(st["T"], st["A"], rb, cb, max(radius_m, 1.0) if radius_m >= b else 1.0, dx=1.0)


def main() -> None:
    t0 = time.time()
    slurm_record(os.path.dirname(out_path("heads", "heads1m.json")))
    dom = load_domain()
    poly, domain = dom["poly"], dom["domain"]
    pk = dom["picks"]
    use = pk["in_domain"] & pk["in_poly"]
    ids, prow, pcol, ptile = pk["id"][use], pk["row"][use].astype(int), pk["col"][use].astype(int), pk["tile"][use]
    half_of_pick = {h: np.isin(ptile, tiles) for h, tiles in HALVES.items()}
    ground = {h: tile_footprint(dom, tiles) & domain & poly for h, tiles in HALVES.items()}
    rec: dict[str, Any] = {"n_picks_used": int(use.sum()), "picks_per_half": {h: int(v.sum()) for h, v in half_of_pick.items()}, "pick_radius_m": PICK_RADIUS_M,
                           "halves": {h: {"tiles": tiles, "n_picks": int(half_of_pick[h].sum()), "ground_cells": int(ground[h].sum())} for h, tiles in HALVES.items()},
                           "reading_H_A": {"share_found_min": HA_SHARE_FOUND_MIN, "share_real_min": HA_SHARE_REAL_MIN, "t_m": T_PRIMARY_M}, "candidates": {}}
    print(f"picks used {use.sum()} (TTV2 {half_of_pick['TTV2'].sum()}, TTV1+TTV3 {half_of_pick['TTV13'].sum()}) [{time.time() - t0:.0f} s]", flush=True)
    arrays: dict[str, np.ndarray] = {"pick_ids": ids.astype(np.int32), "pick_rows": prow.astype(np.int32), "pick_cols": pcol.astype(np.int32)}
    accepted: list[dict[str, Any]] = []
    for name in CANDIDATES:
        st = candidate_statistics(name, dom, t0)
        crec: dict[str, Any] = {"plain": CANDIDATE_PLAIN[name], "note": st["note"], "scale_m": st["scale_m"], "picks": {}, "halves": {}}
        for rad in (PICK_RADIUS_M, *PICK_RADIUS_SENS_M):
            vp = picks_statistic(st, prow, pcol, rad)
            Tp = vp["T_max"]
            crec["picks"][f"{rad:g}"] = {"T_at_picks": Tp.tolist(), "A_at_T_max": vp["A_at_T_max"].tolist(), "A_max_at_picks": vp["A_max"].tolist(),
                                        "n_finite": int(np.isfinite(Tp).sum()), "median_T_all": float(np.nanmedian(Tp)) if np.isfinite(Tp).any() else None,
                                        "median_A_all": float(np.nanmedian(vp["A_max"])) if np.isfinite(vp["A_max"]).any() else None,
                                        "T_pct": {str(q): float(np.nanpercentile(Tp, q)) for q in (5, 25, 50, 75, 95)} if np.isfinite(Tp).any() else None}
            if rad == PICK_RADIUS_M:
                arrays[f"{name}_T_at_picks"] = Tp
                T_primary = Tp
        pooled_pairs: dict[str, dict[str, int]] = {f"{t:g}": {"n_matched": 0, "n_mapped": 0, "n_heads": 0} for t in TOLERANCES_M}
        for cal_half, test_half in (("TTV2", "TTV13"), ("TTV13", "TTV2")):
            Tc = T_primary[half_of_pick[cal_half]]
            C = float(np.nanmedian(Tc))
            hrec: dict[str, Any] = {"calibrated_on": cal_half, "tested_on": test_half, "C": C, "n_cal_picks_finite": int(np.isfinite(Tc).sum()), "by_t": {}}
            mapped_xy = np.stack([prow[half_of_pick[test_half]], pcol[half_of_pick[test_half]]], axis=1).astype(float)
            heads_xy = head_cells_1m(st, C, ground[test_half])
            hrec["n_heads_on_test_ground"] = int(heads_xy.shape[0])
            hrec["n_mapped_on_test_ground"] = int(mapped_xy.shape[0])
            for t in TOLERANCES_M:
                mt = match_heads(mapped_xy, heads_xy, t)
                hrec["by_t"][f"{t:g}"] = {k: mt[k] for k in ("n_matched", "n_missed", "n_extra", "share_found", "share_real")} | {"median_dist_m": float(np.median(mt["matched_dist_m"])) if mt["matched_dist_m"] else None}
                pp = pooled_pairs[f"{t:g}"]
                pp["n_matched"] += mt["n_matched"]
                pp["n_mapped"] += mt["n_mapped"]
                pp["n_heads"] += mt["n_model"]
            b25 = hrec["by_t"][f"{T_PRIMARY_M:g}"]
            hrec["H_A_half"] = reading_ha(b25["share_found"], b25["share_real"])
            # the F1-optimal threshold on the calibration ground (sensitivity, not the reading)
            cal_xy = np.stack([prow[half_of_pick[cal_half]], pcol[half_of_pick[cal_half]]], axis=1).astype(float)
            grid = np.logspace(np.log10(max(C * 0.05, 1e-3)), np.log10(C * 20.0), 37)
            keep_cal = ground[cal_half]
            if st["scale_m"] != 1.0:
                b = int(st["scale_m"])
                nb_r, nb_c = st["M"].shape
                keep12 = np.zeros((nb_r, nb_c), dtype=bool)
                sub = keep_cal[b // 2: nb_r * b: b, b // 2: nb_c * b: b]
                keep12[: sub.shape[0], : sub.shape[1]] = sub
                best = f1_optimal_threshold(Tc, st["M"], st["Mdon"], keep12, cal_xy / b - 0.5 + 0.5, T_PRIMARY_M / b, grid)
                # (12 m cells against picks expressed in 12 m cell units; the tolerance scales with the cell)
            else:
                best = f1_optimal_threshold(Tc, st["M"], st["Mdon"], keep_cal, cal_xy, T_PRIMARY_M, grid)
            if best:
                heads_b = head_cells_1m(st, best["C"], ground[test_half])
                mtb = match_heads(mapped_xy, heads_b, T_PRIMARY_M)
                best["test_half_at_this_C"] = {k: mtb[k] for k in ("n_matched", "n_extra", "share_found", "share_real")}
            hrec["F1_optimal_on_calibration_half"] = best
            crec["halves"][cal_half] = hrec
            print(f"{name}: cal {cal_half} C {C:.3g} -> test {test_half}: {b25['n_matched']}/{mapped_xy.shape[0]} found ({b25['share_found']:.2f}), {heads_xy.shape[0]} heads, share real "
                  f"{b25['share_real'] if b25['share_real'] is not None else float('nan'):.2f}; H-A half {hrec['H_A_half']}; F1-opt C {best.get('C')} [{time.time() - t0:.0f} s]", flush=True)
        crec["pooled"] = {k: {"n_matched": v["n_matched"], "n_mapped": v["n_mapped"], "n_heads": v["n_heads"], "share_found": v["n_matched"] / v["n_mapped"] if v["n_mapped"] else None,
                              "share_real": v["n_matched"] / v["n_heads"] if v["n_heads"] else None} for k, v in pooled_pairs.items()}
        crec["H_A"] = bool(all(h["H_A_half"] for h in crec["halves"].values()))
        crec["C_all_picks"] = float(np.nanmedian(T_primary))
        crec["A_all_picks"] = float(np.nanmedian(np.array(crec["picks"][f"{PICK_RADIUS_M:g}"]["A_max_at_picks"])))
        # the whole-domain head count at the all-picks threshold (for the report's description)
        all_heads = head_cells_1m(st, crec["C_all_picks"], domain & poly)
        crec["n_heads_domain_polygon_at_C_all_picks"] = int(all_heads.shape[0])
        rec["candidates"][name] = crec
        if crec["H_A"]:
            accepted.append({"candidate": name, "pooled_share_found": crec["pooled"][f"{T_PRIMARY_M:g}"]["share_found"], "C_all_picks": crec["C_all_picks"], "A_all_picks": crec["A_all_picks"]})
        arrays[f"{name}_M"] = st["M"].astype(np.float32)
        arrays[f"{name}_Mdon"] = st["Mdon"].astype(np.float32)
        if st["scale_m"] == 1.0:
            arrays[f"{name}_A"] = st["A"].astype(np.float32)
        print(f"{name}: pooled at 25 m found {crec['pooled']['25']['share_found']:.2f} real {crec['pooled']['25']['share_real']:.2f}; H-A {crec['H_A']} [{time.time() - t0:.0f} s]", flush=True)
    acc = max(accepted, key=lambda q: q["pooled_share_found"]) if accepted else None
    rec["accepted"] = acc
    rec["n_accepted"] = len(accepted)
    rec["wall_s"] = time.time() - t0
    np.savez_compressed(out_path("heads", "heads1m.npz"), **arrays)
    dump(json_ready(rec), out_path("heads", "heads1m.json"))
    dump(json_ready({"accepted": acc, "all_accepted": accepted}), out_path("heads", "accepted.json"))
    print(f"heads1m: accepted {acc} [{time.time() - t0:.0f} s]", flush=True)


if __name__ == "__main__":
    main()
