#!/usr/bin/env python3
"""Addendum 19, decision 7 (section 5.3), per addendum 18 spec: the D0-fixed variants refitted at D0_N from the stored rows.

P1_N [A_crit] at (K0, D0_N) and P2_N [K, A_crit] at D0_N, in sample and on the held-out halves, come exactly from the Gram terms stored in
every row of the addendum 18 partitioned run (acrit_profile.parquet: G00..rr for all/h1/h2); the optimum is the best stored row (the thin grid,
0.35 % in A_crit, plus the staircase and fill rows of the old variants). A2_N [K] (AD with D = D0_N) comes from the stored AD rows by exact
reconstruction: at each r the MSE is a quadratic in y = U/D whose three coefficients are fixed by the stored values at the AD optimum (with the
stationarity condition when interior), at A1's y and at the old A2's y; the grid minimum is then refined by Brent with fresh AD evaluations, as
addendum 18 did, and one fresh evaluation at the grid minimum checks the reconstruction (rel. tol 1e-8). Rows without three stored values
are marked unavailable. No P4, P3, A1 or AD is rerun. D0_N at both ends (A at L_f; B at L_w as written).
usage: a19_dstat.py --spec TILE:GROUP
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a18_common import TRUNK_A_M2  # noqa: E402
from a18_fits import ad_with_anchors  # noqa: E402
from a19_common import A18_SUMMARY, RESULTS_ROOT, a18_specs, dump, load_json, out_path, slurm_record  # noqa: E402

from lemcvt.analyses.sensitivity import _fit, _q, greedy_basin_halves  # noqa: E402
from lemcvt.config import load_tile  # noqa: E402

RECON_TOL = 1e-8


def terms_of(row: dict, part: str) -> dict:
    return {"G": np.array([[row[f"G00_{part}"], row[f"G01_{part}"]], [row[f"G01_{part}"], row[f"G11_{part}"]]]), "g": np.array([row[f"g0_{part}"], row[f"g1_{part}"]]),
            "rr": row[f"rr_{part}"], "n": int(row[f"n_{part}"])}


def partitioned_at_D0(rows: list[dict], U: float, K0: float, D0: float, lo, hi) -> dict:
    x0, yf = U / K0, U / D0
    best: dict = {}
    for name, kind in (("P1", "fix_xy"), ("P2", "fix_y")):
        cand = []
        for q in rows:
            t = terms_of(q, "all")
            if t["n"] == 0 or not np.isfinite(t["rr"]):
                continue
            x, y, where = _fit(t, kind, lo, hi, x0, yf)
            cand.append((_q(t, x, y), q, x, y, where))
        if not cand:
            continue
        mse, q, x, y, where = min(cand, key=lambda c: c[0])
        rec = {"A_crit": float(q["A_crit"]), "mse": float(mse), "K": U / x, "D": U / y, "where": where, "n_channel": int(q["n_channel"]), "row_kind": q.get("kind"), "n_rows": len(cand)}
        srt = sorted(c[1]["A_crit"] for c in cand)
        j = srt.index(q["A_crit"])
        rec["neighbour_rows"] = [srt[max(j - 1, 0)], srt[min(j + 1, len(srt) - 1)]]
        rec["resolution_rel"] = (rec["neighbour_rows"][1] - rec["neighbour_rows"][0]) / (2 * q["A_crit"])
        if name == "P2":
            held = {}
            for part, other in (("h1", "h2"), ("h2", "h1")):
                cand_h = []
                for qq in rows:
                    t = terms_of(qq, part)
                    if t["n"] == 0 or not np.isfinite(t["rr"]):
                        continue
                    xh, yh, wh = _fit(t, kind, lo, hi, x0, yf)
                    cand_h.append((_q(t, xh, yh), qq, xh, yh, wh))
                if cand_h:
                    mh, qh, xh, yh, wh = min(cand_h, key=lambda c: c[0])
                    held[f"fit_{part}"] = {"A_crit": float(qh["A_crit"]), "mse": float(mh), "K": U / xh, "D": U / yh, "where": wh, f"score_{other}": _q(terms_of(qh, other), xh, yh)}
            if "fit_h1" in held and "fit_h2" in held:
                held["mse_heldout_mean"] = 0.5 * (held["fit_h1"]["score_h2"] + held["fit_h2"]["score_h1"])
            rec["heldout"] = held
        best[name] = rec
    return best


def reconstruct_quadratic(q: dict, U: float) -> np.ndarray | None:
    """(c0, c1, c2) of MSE(y) = c0 - 2 c1 y + c2 y^2 at one stored r, from the stored values; None if fewer than three constraints."""
    pts = []
    if np.isfinite(q.get("mse_AD_all", np.inf)):
        y_ad = U / q["D_AD_all"]
        pts.append((y_ad, q["mse_AD_all"]))
    else:
        return None
    for name in ("A1", "A2_A", "A2_B"):
        v = q.get(f"mse_{name}_all", np.inf)
        if v is not None and np.isfinite(v):
            pts.append((U / q[f"D_{name}_all"], v))
    rows_ = []
    rhs = []
    s = pts[0][0]                       # scale y by the AD optimum for conditioning
    for y, m in pts:
        rows_.append([1.0, -2.0 * (y / s), (y / s) ** 2])
        rhs.append(m)
    if q.get("where_AD_all") == "interior":
        rows_.append([0.0, 1.0, -(pts[0][0] / s)])     # stationarity: c1' = y* c2' (in scaled units)
        rhs.append(0.0)
    Amat, b = np.array(rows_), np.array(rhs)
    if np.linalg.matrix_rank(Amat) < 3:
        return None
    c, *_ = np.linalg.lstsq(Amat, b, rcond=None)
    pred = Amat @ c
    if np.max(np.abs(pred[: len(pts)] - b[: len(pts)])) > 1e-6 * max(1.0, np.max(np.abs(b[: len(pts)]))):
        return None
    return np.array([c[0], c[1] / s, c[2] / s ** 2])


def mse_quadratic(c: np.ndarray, y: float) -> float:
    return float(c[0] - 2.0 * c[1] * y + c[2] * y * y)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--spec", required=True)
    p.add_argument("--no-brent", action="store_true")
    a = p.parse_args()
    t0 = time.time()
    tile, group = a.spec.split(":")
    out_dir = os.path.dirname(out_path("dstat", f"{tile}_{group}.json"))
    slurm_record(out_dir)
    spec = a18_specs()[a.spec]
    fits = load_json(os.path.join(A18_SUMMARY, "per_tile_fits.json"))[a.spec]
    partA = load_json(out_path("site", "partA.json"))
    tl = partA["tiles"].get(tile)
    rec: dict = {"spec": a.spec, "tile": tile, "group": group, "site": spec["site"], "m": spec["m"], "K0": spec["K0"], "D0_R_A": spec.get("D0A"), "D0_R_B": spec.get("D0B"),
                 "run_part": fits["run_part"], "run_ad": fits["run_ad"]}
    if not tl or not tl["usable"]:
        rec["usable"] = False
        rec["reason"] = "no D0_N (fewer than 500 counted pixels)" if tl else "tile not measured"
        dump(rec, out_path("dstat", f"{tile}_{group}.json"))
        print(a.spec, "no D0_N", flush=True)
        return
    D0N = {"A": tl["D0_N_A"], "B": tl["D0_N_B"]}
    rec.update({"usable": True, "D0_N_A": D0N["A"], "D0_N_B": D0N["B"], "L_w_m": tl["L_w_m"], "bracket": tl["bracket"], "D_R": tl.get("D_R"), "D_R_ci68": tl.get("D_R_ci68")})
    td = load_tile(tile)
    mp = td.cfg.model
    U, K0 = float(spec["E"]), float(spec["K0"])
    (K_lo, K_hi), (D_lo, D_hi) = mp.K_bounds, mp.D_bounds
    lo = np.array([U / K_hi, U / D_hi])
    hi = np.array([U / K_lo, U / D_lo])
    # partitioned variants from the stored Gram rows
    df = pd.read_parquet(os.path.join(RESULTS_ROOT, fits["run_part"], "acrit_profile.parquet"))
    rows = df.to_dict("records")
    rec["partitioned"] = {end: partitioned_at_D0(rows, U, K0, D0N[end], lo, hi) for end in ("A", "B")}
    rec["n_rows_part"] = len(rows)
    print(f"{a.spec}: P1_N/P2_N from {len(rows)} rows: " + "; ".join(f"{e}: P1 {v['P1']['mse']:.3f} @ {v['P1']['A_crit']:.0f}, P2 {v['P2']['mse']:.3f} @ {v['P2']['A_crit']:.0f} K/K0 {v['P2']['K'] / K0:.3f}" for e, v in rec["partitioned"].items()) + f" [{time.time() - t0:.0f} s]", flush=True)
    # AD: A2_N by reconstruction + Brent
    dfa = pd.read_parquet(os.path.join(RESULTS_ROOT, fits["run_ad"], "r_profile.parquet"))
    arows = [q for q in dfa.to_dict("records") if q.get("kind") == "grid"]
    coefs = {q["r"]: reconstruct_quadratic(q, U) for q in arows}
    n_avail = sum(1 for c in coefs.values() if c is not None)
    rec["ad"] = {"n_grid_rows": len(arows), "n_reconstructed": n_avail}
    need_model = n_avail > 0 and not a.no_brent
    ad_model = prof = obs = keep_all = None
    if need_model:
        from lemcvt.models.ad import advection_diffusion_model
        from lemcvt.models.common import extract_grid_data
        from lemcvt.optimize.v2 import ADProfiler
        Z0, A, FD, dx = td.Z, td.A, td.FD, td.dx
        m = float(spec["m"])
        ad_model = advection_diffusion_model(Z0, A, FD, dx, m)
        gd = extract_grid_data(Z0, A, FD, dx)
        obs = Z0[gd["rev_j"], gd["rev_i"]].astype(float)
        trunk_mask = np.isfinite(A) & (A >= TRUNK_A_M2) if spec.get("trunk") else np.zeros(A.shape, dtype=bool)
        keep_all = ~trunk_mask[gd["rev_j"], gd["rev_i"]]
        if spec.get("trunk"):
            ad_model, _n = ad_with_anchors(ad_model, obs, np.where(~keep_all)[0])
        prof = ADProfiler(ad_model, obs, U)
        _ = greedy_basin_halves

    def direct(r: float, y: float) -> float:
        red = prof.reduce(float(r))
        res = obs - red["h"]
        g = red["g"]
        ok = np.isfinite(res) & np.isfinite(g) & keep_all
        e = res[ok] - y * g[ok]
        return float(e @ e / ok.sum())

    for end in ("A", "B"):
        D0 = D0N[end]
        yN = U / D0
        grid = []
        for q in arows:
            c = coefs[q["r"]]
            K2 = q["r"] * D0
            feas = K_lo <= K2 <= K_hi
            grid.append({"r": q["r"], "mse": mse_quadratic(c, yN) if (c is not None and feas) else float("inf"), "available": c is not None, "feasible": feas})
        feas_rows = [g for g in grid if np.isfinite(g["mse"])]
        res: dict = {"n_feasible": len(feas_rows), "n_unavailable_feasible": sum(1 for g in grid if g["feasible"] and not g["available"])}
        if not feas_rows:
            res["note"] = "no feasible reconstructed row"
            rec["ad"][f"A2_N_{end}"] = res
            continue
        jbest = min(range(len(feas_rows)), key=lambda j: feas_rows[j]["mse"])
        gbest = feas_rows[jbest]
        res["grid_optimum"] = {"r": gbest["r"], "mse": gbest["mse"], "K": gbest["r"] * D0, "D": D0, "at_feasible_edge": bool(jbest in (0, len(feas_rows) - 1))}
        if need_model:
            t1 = time.perf_counter()
            d_chk = direct(gbest["r"], yN)
            res["reconstruction_check"] = {"r": gbest["r"], "reconstructed": gbest["mse"], "direct": d_chk, "rel_diff": abs(gbest["mse"] / d_chk - 1), "pass": bool(abs(gbest["mse"] / d_chk - 1) <= RECON_TOL), "wall_s": time.perf_counter() - t1}
            from scipy.optimize import minimize_scalar
            lg = np.log10([g["r"] for g in feas_rows])
            lo_, hi_ = lg[max(jbest - 1, 0)], lg[min(jbest + 1, len(lg) - 1)]
            evals = []

            def obj(lr):
                r_ = 10 ** lr
                K2 = r_ * D0
                v = direct(r_, yN) if K_lo <= K2 <= K_hi else float("inf")
                evals.append((r_, v))
                return v

            if lo_ < hi_:
                rb = minimize_scalar(obj, bounds=(lo_, hi_), method="bounded", options={"xatol": 1e-6})
                r_opt, mse_opt = (10 ** rb.x, rb.fun) if rb.fun <= gbest["mse"] else (gbest["r"], gbest["mse"])
                res["brent"] = {"nfev": int(rb.nfev), "success": bool(rb.success), "bracket_log10": [float(lo_), float(hi_)], "improved_on_grid": bool(rb.fun < gbest["mse"])}
            else:
                r_opt, mse_opt = gbest["r"], gbest["mse"]
            res["optimum"] = {"r": float(r_opt), "mse": float(mse_opt), "K": float(r_opt * D0), "D": D0, "K_at_bound": bool(np.isclose(r_opt * D0, K_lo, rtol=1e-6) or np.isclose(r_opt * D0, K_hi, rtol=1e-6)), "n_fresh_evaluations": len(evals) + 1}
        else:
            res["optimum"] = res["grid_optimum"] | {"K_at_bound": None, "note": "grid only"}
        rec["ad"][f"A2_N_{end}"] = res
        print(f"{a.spec}: A2_N_{end} grid {gbest['mse']:.3f} @ r {gbest['r']:.3e} -> {res['optimum']['mse']:.3f} (K/K0 {res['optimum']['K'] / K0:.3f}); recon check {res.get('reconstruction_check', {}).get('rel_diff')} [{time.time() - t0:.0f} s]", flush=True)
    # the addendum 18 values for the comparisons
    rec["a18"] = {k: fits.get(k) for k in ("P4", "P3", "AD", "A1", "P1_A", "P1_B", "P2_A", "P2_B", "A2_A", "A2_B", "heldout_part", "heldout_AD", "lopsided", "has_D0", "R_MSE", "F1", "F2")}
    rec["wall_s"] = time.time() - t0
    dump(rec, out_path("dstat", f"{tile}_{group}.json"))
    _ = math


if __name__ == "__main__":
    main()
