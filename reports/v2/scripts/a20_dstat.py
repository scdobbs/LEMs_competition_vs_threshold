#!/usr/bin/env python3
"""Addendum 20, decision 6e (section 5.3), per addendum 18 spec: the D0-fixed variants P1, P2 and A2 refitted at the new D0.

Ends: "P" = the tile's D_C of decision 2 (D0_P at r = 7 m where the tile has a plateau, else D0_N), "N" = D0_N (decision 1, the sensitivity).
Route (the one the addendum 19 report states in its section 1.1, code `a19_dstat`): P1 and P2 from the Gram terms stored in every row of the
addendum 18 partitioned run (exact on the stored A_crit rows); A2 from the stored AD rows by exact quadratic reconstruction in y = U/D at
each r, the grid minimum refined by Brent with fresh AD evaluations, and one fresh evaluation at the grid minimum as the check (rel. 1e-8).
Also one fresh partitioned evaluation per tile at P2's optimum (the Gram route checked against a direct reduction, rel. 1e-8). No P4, P3,
A1 or AD is rerun.
usage: a20_dstat.py --spec TILE:GROUP
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a19_dstat  # noqa: E402
from a18_common import TRUNK_A_M2  # noqa: E402
from a18_fits import ad_with_anchors  # noqa: E402
from a19_dstat import mse_quadratic, partitioned_at_D0, reconstruct_quadratic  # noqa: E402
from a20_common import A18_SUMMARY, RECON_TOL, RESULTS_ROOT, a18_specs, dump, load_json, out_path, slurm_record  # noqa: E402

from lemcvt.config import load_tile  # noqa: E402


def terms_of(row: dict, part: str) -> dict:
    """The stored Gram terms of one addendum 18 row in the flat form `sensitivity._q` / `_fit` take (G00, G01, G11, g0, g1, rr, n).
    Replaces `a19_dstat.terms_of`, which built nested arrays that `_q` cannot read (that code had never run)."""
    return {k: (int(row[f"{k}_{part}"]) if k == "n" else float(row[f"{k}_{part}"])) for k in ("G00", "G01", "G11", "g0", "g1", "rr", "n")}


a19_dstat.terms_of = terms_of


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--spec", required=True)
    a = p.parse_args()
    t0 = time.time()
    tile, group = a.spec.split(":")
    slurm_record(os.path.dirname(out_path("dstat", f"{tile}_{group}.json")))
    spec = a18_specs()[a.spec]
    fits = load_json(os.path.join(A18_SUMMARY, "per_tile_fits.json"))[a.spec]
    tl = load_json(out_path("site", "partA.json"))["tiles"].get(tile)
    rec: dict = {"spec": a.spec, "tile": tile, "group": group, "site": spec["site"], "m": spec["m"], "K0": spec["K0"], "D0_R_A": spec.get("D0A"), "D0_R_B": spec.get("D0B"),
                 "run_part": fits["run_part"], "run_ad": fits["run_ad"]}
    D0 = {}
    if tl:
        if tl.get("D_C") is not None:
            D0["P"] = tl["D_C"]
        if tl.get("usable_N") and tl.get("D0_N_A"):
            D0["N"] = tl["D0_N_A"]
        rec.update({"D_C_source": tl.get("D_C_source"), "D0_P": tl.get("D0_P"), "D0_N_A": tl.get("D0_N_A"), "usable_N": tl.get("usable_N"), "plateau_r7": tl.get("plateau_r7"),
                    "bracket_P": tl.get("bracket_P"), "bracket_N": tl.get("bracket_N"), "D_R": tl.get("D_R"), "D_R_ci68": tl.get("D_R_ci68")})
    rec["D0_used"] = D0
    rec["usable"] = bool(D0)
    if not D0:
        rec["reason"] = "no D0_P plateau and fewer than 500 D0_N pixels" if tl else "tile not measured"
        dump(rec, out_path("dstat", f"{tile}_{group}.json"))
        print(a.spec, rec["reason"], flush=True)
        return
    td = load_tile(tile)
    mp = td.cfg.model
    U, K0 = float(spec["E"]), float(spec["K0"])
    (K_lo, K_hi), (D_lo, D_hi) = mp.K_bounds, mp.D_bounds
    lo = np.array([U / K_hi, U / D_hi])
    hi = np.array([U / K_lo, U / D_lo])
    df = pd.read_parquet(os.path.join(RESULTS_ROOT, fits["run_part"], "acrit_profile.parquet"))
    rows = df.to_dict("records")
    rec["partitioned"] = {end: partitioned_at_D0(rows, U, K0, v, lo, hi) for end, v in D0.items()}
    rec["n_rows_part"] = len(rows)
    if fits.get("P2_A") and fits.get("P1_A"):
        chk = partitioned_at_D0(rows, U, K0, float(fits["P2_A"]["D"]), lo, hi)
        rec["route_check_vs_a18"] = {"D0A": float(fits["P2_A"]["D"]), "P2_recomputed": chk["P2"]["mse"], "P2_a18": fits["P2_A"]["mse"], "P1_recomputed": chk["P1"]["mse"], "P1_a18": fits["P1_A"]["mse"],
                                     "rel_P2": abs(chk["P2"]["mse"] / fits["P2_A"]["mse"] - 1), "rel_P1": abs(chk["P1"]["mse"] / fits["P1_A"]["mse"] - 1),
                                     "same_A_crit_P2": bool(chk["P2"]["A_crit"] == fits["P2_A"]["A_crit"])}
    for end, v in rec["partitioned"].items():
        for name in ("P1", "P2"):
            if v.get(name):
                v[name]["D_at_bound"] = bool(np.isclose(v[name]["D"], D_lo, rtol=1e-9) or np.isclose(v[name]["D"], D_hi, rtol=1e-9))
                v[name]["D0_outside_bounds"] = bool(not (D_lo <= D0[end] <= D_hi))
    print(f"{a.spec}: " + "; ".join(f"{e}: D0 {D0[e]:.4g} P1 {v['P1']['mse']:.3f} @ {v['P1']['A_crit']:.0f}, P2 {v['P2']['mse']:.3f} @ {v['P2']['A_crit']:.0f}" for e, v in rec["partitioned"].items()) + f" [{time.time() - t0:.0f} s]", flush=True)
    # the direct check of the Gram route: one fresh partitioned reduction at P2's A_crit, (K, D) of P2
    from lemcvt.optimize.v2 import PartitionedProfiler
    Z0, A, FD, dx = td.Z, td.A, td.FD, td.dx
    m = float(spec["m"])
    if not spec.get("trunk"):
        end0 = "P" if "P" in rec["partitioned"] else "N"
        p2 = rec["partitioned"][end0]["P2"]
        prof_p = PartitionedProfiler(Z0, A, FD, dx, U, m_values=[m])
        red = prof_p.reduce(float(p2["A_crit"]), m)
        t_all = red.mse_terms(prof_p.Z_obs_flat)
        direct = red.mse_from_terms(t_all, U / p2["K"], U / p2["D"])
        rec["gram_route_check"] = {"end": end0, "A_crit": p2["A_crit"], "stored": p2["mse"], "direct": float(direct), "rel_diff": abs(p2["mse"] / direct - 1), "pass": bool(abs(p2["mse"] / direct - 1) <= RECON_TOL)}
        del prof_p, red
    else:
        rec["gram_route_check"] = {"note": "trunk-anchored spec: the stored rows include the anchors; the direct check is the AD one below"}
    # AD: A2 by reconstruction + Brent
    dfa = pd.read_parquet(os.path.join(RESULTS_ROOT, fits["run_ad"], "r_profile.parquet"))
    arows = [q for q in dfa.to_dict("records") if q.get("kind") == "grid"]
    coefs = {q["r"]: reconstruct_quadratic(q, U) for q in arows}
    n_avail = sum(1 for c in coefs.values() if c is not None)
    rec["ad"] = {"n_grid_rows": len(arows), "n_reconstructed": n_avail}
    if n_avail:
        from lemcvt.models.ad import advection_diffusion_model
        from lemcvt.models.common import extract_grid_data
        from lemcvt.optimize.v2 import ADProfiler
        ad_model = advection_diffusion_model(Z0, A, FD, dx, m)
        gd = extract_grid_data(Z0, A, FD, dx)
        obs = Z0[gd["rev_j"], gd["rev_i"]].astype(float)
        trunk_mask = np.isfinite(A) & (A >= TRUNK_A_M2) if spec.get("trunk") else np.zeros(A.shape, dtype=bool)
        keep_all = ~trunk_mask[gd["rev_j"], gd["rev_i"]]
        if spec.get("trunk"):
            ad_model, _n = ad_with_anchors(ad_model, obs, np.where(~keep_all)[0])
        prof = ADProfiler(ad_model, obs, U)

        def direct(r: float, y: float) -> float:
            red_ = prof.reduce(float(r))
            res = obs - red_["h"]
            g = red_["g"]
            ok = np.isfinite(res) & np.isfinite(g) & keep_all
            e = res[ok] - y * g[ok]
            return float(e @ e / ok.sum())

        from scipy.optimize import minimize_scalar
        for end, Dv in D0.items():
            yN = U / Dv
            grid = []
            for q in arows:
                c = coefs[q["r"]]
                feas = K_lo <= q["r"] * Dv <= K_hi
                grid.append({"r": q["r"], "mse": mse_quadratic(c, yN) if (c is not None and feas) else float("inf"), "available": c is not None, "feasible": feas})
            feas_rows = [g for g in grid if np.isfinite(g["mse"])]
            res: dict = {"D0": Dv, "n_feasible": len(feas_rows), "n_unavailable_feasible": sum(1 for g in grid if g["feasible"] and not g["available"]), "D0_outside_bounds": bool(not (D_lo <= Dv <= D_hi))}
            if not feas_rows:
                res["note"] = "no feasible reconstructed row"
                rec["ad"][f"A2_{end}"] = res
                continue
            jb = min(range(len(feas_rows)), key=lambda j: feas_rows[j]["mse"])
            gb = feas_rows[jb]
            res["grid_optimum"] = {"r": gb["r"], "mse": gb["mse"], "K": gb["r"] * Dv, "at_feasible_edge": bool(jb in (0, len(feas_rows) - 1))}
            d_chk = direct(gb["r"], yN)
            res["reconstruction_check"] = {"reconstructed": gb["mse"], "direct": d_chk, "rel_diff": abs(gb["mse"] / d_chk - 1), "pass": bool(abs(gb["mse"] / d_chk - 1) <= RECON_TOL)}
            lg = np.log10([g["r"] for g in feas_rows])
            lo_, hi_ = lg[max(jb - 1, 0)], lg[min(jb + 1, len(lg) - 1)]
            evals = []

            def obj(lr, Dv=Dv, yN=yN, evals=evals):
                r_ = 10 ** lr
                v = direct(r_, yN) if K_lo <= r_ * Dv <= K_hi else float("inf")
                evals.append((r_, v))
                return v

            if lo_ < hi_:
                rb = minimize_scalar(obj, bounds=(lo_, hi_), method="bounded", options={"xatol": 1e-6})
                r_opt, mse_opt = (10 ** rb.x, rb.fun) if rb.fun <= gb["mse"] else (gb["r"], gb["mse"])
                res["brent"] = {"nfev": int(rb.nfev), "improved_on_grid": bool(rb.fun < gb["mse"])}
            else:
                r_opt, mse_opt = gb["r"], gb["mse"]
            res["optimum"] = {"r": float(r_opt), "mse": float(mse_opt), "K": float(r_opt * Dv), "D": Dv, "K_over_K0": float(r_opt * Dv / K0),
                              "K_at_bound": bool(np.isclose(r_opt * Dv, K_lo, rtol=1e-6) or np.isclose(r_opt * Dv, K_hi, rtol=1e-6)), "n_fresh_evaluations": len(evals) + 1}
            rec["ad"][f"A2_{end}"] = res
            print(f"{a.spec}: A2_{end} grid {gb['mse']:.3f} -> {mse_opt:.3f} (K/K0 {r_opt * Dv / K0:.3f}); recon check {res['reconstruction_check']['rel_diff']:.1e} [{time.time() - t0:.0f} s]", flush=True)
    rec["a18"] = {k: fits.get(k) for k in ("P4", "P3", "AD", "A1", "P1_A", "P1_B", "P2_A", "P2_B", "A2_A", "A2_B", "heldout_part", "heldout_AD", "lopsided", "has_D0", "R_MSE", "F1", "F2")}
    rec["wall_s"] = time.time() - t0
    dump(rec, out_path("dstat", f"{tile}_{group}.json"))


if __name__ == "__main__":
    main()
