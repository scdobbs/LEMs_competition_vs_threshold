#!/usr/bin/env python3
"""Addendum 22, per tile (TTV1-TTV3), the cheap part: the polygon misfit for every model that needs no threshold profile.

  - the mask check (decision 2.1): with the mask set to every cell, PF-A, PF-C, F2-relief, F2-phys, P4 (at its A_crit) and AD (at its
    K, D) reproduce the addendum 21 misfits (relative 1e-8);
  - the refits of PF-A and PF-C with the misfit over the M&D polygon only (one exact reduction per network: the closed-form box minimum
    in (U/K, U/D) of the polygon Gram terms), with the whole-tile misfit of the polygon fit beside it and the polygon misfit of the
    addendum 21 (whole-tile) fit;
  - the re-scores of F2-relief and F2-phys (K0, fixed D) over the polygon;
  - the polygon misfit of the addendum 21 P4 and AD parameters (their refits are `a22_profile.py`);
  - the surfaces of obs, PF-A, PF-C, F2-relief, F2-phys through the 12 m products of decision 2.3 (`<tile>_products.npz`);
  - the polygon cell counts, the M&D traced channel length inside the polygon and tile, the real landscape's steep-slope threshold.
Outputs <run>/tiles/<tile>.json, <tile>_products.npz.
usage: a22_tile.py --tile TTV2
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a22_common import (  # noqa: E402
    A_F,
    BLOCK_M,
    C_F,
    D_PHYS,
    MASK_CHECK_TOL,
    STEEP_PCT,
    a21_path,
    coarse_products,
    dump,
    fit_terms,
    json_ready,
    keep_masks,
    md_traced_length_in_tile_polygon,
    mse_at,
    out_path,
    partitioned_surface,
    poly12_centres,
    reduce_mask,
    slurm_record,
    terms_of,
    tile_setup,
    unpack,
)

from lemcvt.models.ad import advection_diffusion_model  # noqa: E402
from lemcvt.models.common import extract_grid_data  # noqa: E402
from lemcvt.optimize.v2 import ADProfiler, PartitionedProfiler  # noqa: E402


def rel(a: float, b: float) -> float:
    return abs(a / b - 1.0) if b else float("inf")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tile", required=True)
    a = p.parse_args()
    tile = a.tile
    t0 = time.time()
    slurm_record(os.path.dirname(out_path("tiles", f"{tile}.json")))
    S = tile_setup(tile)
    td, Z, A, FD, dx = S["td"], S["Z"], S["A"], S["FD"], S["dx"]
    U, m, K0, lo, hi = S["U"], S["m"], S["K0"], S["lo"], S["hi"]
    a21 = S["a21"]
    poly = S["poly"]
    prof = PartitionedProfiler(Z, A, FD, dx, U, m_values=[m])
    keep = keep_masks(prof, poly)
    nb = Z.shape[0] // BLOCK_M
    rec: dict = {"tile": tile, "U": U, "m": m, "K0": K0, "bounds_rescaled": {"K": list(S["Kb"]), "D": list(S["Db"])}, "D_phys": D_PHYS, "D_relief": S["D_relief"],
                 "D_relief_network": S["D_relief_network"], "inputs": td.provenance(),
                 "polygon": {"n_cells_1m": int(poly.sum()), "n_valid_1m": int(np.isfinite(Z).sum()), "n_profiler_cells": int(prof.n), "n_profiler_cells_in_polygon": int(keep["poly"].sum()),
                             "share_of_tile": float(poly.sum() / np.isfinite(Z).sum()), "n_cells_12m_centre_inside": int(poly12_centres(poly, nb).sum()), "n_cells_12m": int(nb * nb)},
                 "field_thresholds": {"A_F": A_F, "C_F": C_F}, "picks": {"n_on_tile": int(S["pick_rc"][0].size), "ids": S["pick_ids"],
                                                                           "n_in_polygon": int(poly[S["pick_rc"]].sum()) if S["pick_rc"][0].size else 0}}
    rec["md_traced"] = md_traced_length_in_tile_polygon(td)
    print(f"{tile}: polygon {rec['polygon']['n_cells_1m']:,} of {rec['polygon']['n_valid_1m']:,} 1 m cells ({rec['polygon']['share_of_tile']:.2f}); M&D traced length in tile∩polygon "
          f"{rec['md_traced']['length_in_tile_and_polygon_m']:.0f} m [{time.time() - t0:.0f} s]", flush=True)
    # ---------------------------------------------------------------- networks (addendum 21) and the partitioned fits
    hz = np.load(a21_path("tiles", f"{tile}_heads.npz"))
    nets = {"A": unpack(hz["net_A_1m"], Z.shape), "C": unpack(hz["net_C_1m"], Z.shape)}
    rec["models"], rec["mask_check"] = {}, {}
    surfaces: dict[str, np.ndarray] = {"obs": Z}
    for net, name in (("A", "PF_A"), ("C", "PF_C")):
        t1 = time.perf_counter()
        red = reduce_mask(prof, nets[net], m)
        T = {k: terms_of(red, prof.Z_obs_flat, v) for k, v in keep.items()}
        f_all, f_poly = fit_terms(T["all"], U, K0, lo, hi), fit_terms(T["poly"], U, K0, lo, hi)
        stored = a21["PF"][name]["free_KD"]
        chk = {"a21_mse": stored["mse"], "all_cells_mse": f_all["free_KD"]["mse"], "rel_diff_mse": rel(f_all["free_KD"]["mse"], stored["mse"]),
               "rel_diff_K": rel(f_all["free_KD"]["K"], stored["K"]), "rel_diff_D": rel(f_all["free_KD"]["D"], stored["D"])}
        chk["pass"] = bool(chk["rel_diff_mse"] <= MASK_CHECK_TOL and chk["rel_diff_K"] <= 1e-6 and chk["rel_diff_D"] <= 1e-6)
        rec["mask_check"][name] = chk
        fp = f_poly["free_KD"]
        rec["models"][name] = {"fitted": "K, D", "network": net, "threshold": A_F if net == "A" else C_F, "n_channel": int(red.channel_mask.sum()),
                               "K": fp["K"], "D": fp["D"], "K_over_K0": fp["K_over_K0"], "where": fp["where"], "mse_poly": fp["mse"],
                               "mse_all_at_poly_fit": mse_at(T["all"], U, fp["K"], fp["D"]), "n_cells_poly": T["poly"]["n"], "n_cells_all": T["all"]["n"],
                               "a21_whole_tile_fit": {"K": stored["K"], "D": stored["D"], "mse_all": stored["mse"], "mse_poly": mse_at(T["poly"], U, stored["K"], stored["D"])},
                               "K_fixed_poly": f_poly["K_fixed"], "wall_s": time.perf_counter() - t1}
        Zs, holes = partitioned_surface(prof, red, fp["K"], fp["D"], Z)
        rec["models"][name]["surface_holes"] = holes
        surfaces[name] = Zs
        print(f"{tile}: {name} polygon fit K/K0 {fp['K_over_K0']:.3f} D {fp['D']:.4g} ({fp['where']}) mse_poly {fp['mse']:.2f} (whole tile at these {rec['models'][name]['mse_all_at_poly_fit']:.2f}; "
              f"a21 fit mse_poly {rec['models'][name]['a21_whole_tile_fit']['mse_poly']:.2f}); mask check {'PASS' if chk['pass'] else 'FAIL'} ({chk['rel_diff_mse']:.1e}) [{time.time() - t0:.0f} s]", flush=True)
        if net == "C":
            for dname, Dv in (("F2_relief", S["D_relief"]), ("F2_phys", D_PHYS)):
                st = a21["fixed"][dname]
                mse_all = mse_at(T["all"], U, K0, Dv)
                chk2 = {"a21_mse": st["mse"], "all_cells_mse": mse_all, "rel_diff_mse": rel(mse_all, st["mse"]), "rel_diff_D": rel(Dv, st["D"])}
                chk2["pass"] = bool(chk2["rel_diff_mse"] <= MASK_CHECK_TOL and chk2["rel_diff_D"] <= 1e-9)
                rec["mask_check"][dname] = chk2
                rec["models"][dname] = {"fitted": "none", "network": "C", "threshold": C_F, "n_channel": int(red.channel_mask.sum()), "K": K0, "D": Dv, "K_over_K0": 1.0, "where": "fixed",
                                        "mse_poly": mse_at(T["poly"], U, K0, Dv), "mse_all_at_poly_fit": mse_all, "n_cells_poly": T["poly"]["n"], "n_cells_all": T["all"]["n"],
                                        "a21_whole_tile_fit": {"K": K0, "D": Dv, "mse_all": st["mse"], "mse_poly": mse_at(T["poly"], U, K0, Dv)}}
                Zs, holes = partitioned_surface(prof, red, K0, Dv, Z)
                rec["models"][dname]["surface_holes"] = holes
                surfaces[dname] = Zs
                print(f"{tile}: {dname} D {Dv:.4g}: mse_poly {rec['models'][dname]['mse_poly']:.2f}, whole tile {mse_all:.2f}; mask check {'PASS' if chk2['pass'] else 'FAIL'} ({chk2['rel_diff_mse']:.1e})", flush=True)
        del red, T
    # ---------------------------------------------------------------- the addendum 21 P4 and AD parameters scored over the polygon
    p4 = a21["rescaled"]["P4"]
    red4 = prof.reduce(float(p4["A_crit"]), m)
    T4 = {k: terms_of(red4, prof.Z_obs_flat, v) for k, v in keep.items()}
    mse4_all = mse_at(T4["all"], U, p4["K"], p4["D"])
    rec["mask_check"]["P4_at_a21_A_crit"] = {"a21_mse": p4["mse"], "all_cells_mse": mse4_all, "rel_diff_mse": rel(mse4_all, p4["mse"]), "pass": bool(rel(mse4_all, p4["mse"]) <= MASK_CHECK_TOL)}
    rec["a21_params_scored"] = {"P4": {"A_crit": p4["A_crit"], "K": p4["K"], "D": p4["D"], "mse_all": p4["mse"], "mse_poly": mse_at(T4["poly"], U, p4["K"], p4["D"]),
                                       "poly_fit_at_a21_A_crit": fit_terms(T4["poly"], U, K0, lo, hi)["free_KD"]}}
    del red4, T4
    ad = a21["rescaled"]["AD"]
    adm = advection_diffusion_model(Z, A, FD, dx, m)
    gd = extract_grid_data(Z, A, FD, dx)
    obs_all = Z[gd["rev_j"], gd["rev_i"]].astype(float)
    obs_poly = np.where(poly[gd["rev_j"], gd["rev_i"]], obs_all, np.nan)
    pa, pp = ADProfiler(adm, obs_all, U), ADProfiler(adm, obs_poly, U)
    t1 = time.perf_counter()
    red_ad = pa.reduce(ad["K"] / ad["D"])
    y = U / ad["D"]
    mse_ad_all = pa.mse_at(red_ad, y)
    rec["mask_check"]["AD_at_a21_K_D"] = {"a21_mse": ad["mse"], "all_cells_mse": mse_ad_all, "rel_diff_mse": rel(mse_ad_all, ad["mse"]), "pass": bool(rel(mse_ad_all, ad["mse"]) <= MASK_CHECK_TOL),
                                          "wall_s_one_reduction": time.perf_counter() - t1}
    rec["a21_params_scored"]["AD"] = {"K": ad["K"], "D": ad["D"], "r": ad["K"] / ad["D"], "mse_all": ad["mse"], "mse_poly": pp.mse_at(red_ad, y), "n_cells_poly": int(np.isfinite(obs_poly).sum()),
                                      "n_cells_all": int(np.isfinite(obs_all).sum())}
    print(f"{tile}: a21 P4 (A_crit {p4['A_crit']:.0f}) mse_poly {rec['a21_params_scored']['P4']['mse_poly']:.2f} (all {p4['mse']:.2f}); a21 AD mse_poly {rec['a21_params_scored']['AD']['mse_poly']:.2f} "
          f"(all {ad['mse']:.2f}); checks P4 {rec['mask_check']['P4_at_a21_A_crit']['rel_diff_mse']:.1e} AD {rec['mask_check']['AD_at_a21_K_D']['rel_diff_mse']:.1e} [{time.time() - t0:.0f} s]", flush=True)
    del red_ad, pa, pp, adm
    # ---------------------------------------------------------------- the 12 m products (decision 2.3)
    arrays: dict[str, np.ndarray] = {"poly_1m": np.packbits(poly), "poly12_centre": poly12_centres(poly, nb), "pick_rc": np.stack(S["pick_rc"]).astype(np.int32) if S["pick_rc"][0].size else np.zeros((2, 0), np.int32),
                                     "pick_ids": np.array(S["pick_ids"], dtype=np.int32)}
    rec["products"] = {}
    for name, Zs in surfaces.items():
        t1 = time.perf_counter()
        P = coarse_products(Zs, dx)
        for k, v in P.items():
            arrays[f"{name}_{k}"] = v
        ch2, hd2 = (np.isfinite(P["M_a2"]) & (P["M_a2"] >= C_F)), None
        hd2 = ch2 & ~(P["Mdon_a2"] >= C_F)
        inside = arrays["poly12_centre"]
        rec["products"][name] = {"n_valid_12m": int(np.isfinite(P["Z12"]).sum()), "n_channel_a2": int(ch2.sum()), "n_heads_a2": int(hd2.sum()), "n_heads_a2_in_polygon": int((hd2 & inside).sum()),
                                 "wall_s": time.perf_counter() - t1}
        if name == "obs":
            sl = P["slope12"][inside & np.isfinite(P["slope12"])]
            rec["steep_threshold_obs"] = {"percentile": STEEP_PCT, "slope": float(np.percentile(sl, STEEP_PCT)), "n_cells": int(sl.size), "median_slope": float(np.median(sl))}
        print(f"{tile}: {name}: 12 m channels {ch2.sum()} heads {hd2.sum()} (in polygon {rec['products'][name]['n_heads_a2_in_polygon']}) [{time.time() - t0:.0f} s]", flush=True)
    np.savez_compressed(out_path("tiles", f"{tile}_products.npz"), **arrays)
    rec["wall_s"] = time.time() - t0
    dump(json_ready(rec), out_path("tiles", f"{tile}.json"))
    print(f"{tile}: done; mask checks " + ", ".join(f"{k} {'PASS' if v['pass'] else 'FAIL'}" for k, v in rec["mask_check"].items()) + f" [{time.time() - t0:.0f} s]", flush=True)


if __name__ == "__main__":
    main()
