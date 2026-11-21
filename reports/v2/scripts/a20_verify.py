#!/usr/bin/env python3
"""Addendum 20, section 2.1: verification of the decision-2 estimator (disk least-squares quadratic) before use.

(1) An analytic paraboloid z = 300 + 0.12 x - 0.07 y - (x^2 + y^2)/(2 R0) + 0.003 x y (lap = -2/R0, |grad z| known at every cell);
(2) a plane z = 300 + 0.2 x + 0.1 y (lap = 0, |grad z| = sqrt(0.05)); both on 1 m and on the 0.9752 x 0.9751 m PAMAP lattice, at every radius;
(3) the convolution against an independent per-cell numpy lstsq on a noisy paraboloid at 200 random cells;
(4) the plateau rule on a constructed bin table (a known plateau of 4 bins, then a jump) and the exclusion margin.
Writes <run>/verify/verify.json; exits non-zero if any check fails.
usage: a20_verify.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a20_common import N_BOOT_BLOCK, PERRON_RADII_M, dump, out_path  # noqa: E402
from a20_perron import N_BINS, direct_quadratic, disk_quadratic, estimate, plateau, valid_interior  # noqa: E402

TOL_LAP = 1e-9        # m^-1, exact polynomial up to float rounding
TOL_GRAD = 1e-9


def grids(ny, nx, dx, dy):
    ii, jj = np.mgrid[0:ny, 0:nx]
    X = (jj - nx // 2) * dx
    Y = -(ii - ny // 2) * dy
    return X, Y


def main() -> None:
    out: dict = {"checks": [], "all_pass": True}
    R0 = 40.0
    for dx, dy in ((1.0, 1.0), (0.9751997044335207, 0.9751305418722148)):
        X, Y = grids(301, 331, dx, dy)
        par = 300 + 0.12 * X - 0.07 * Y - (X ** 2 + Y ** 2) / (2 * R0) + 0.003 * X * Y
        gx = 0.12 - X / R0 + 0.003 * Y
        gy = -0.07 - Y / R0 + 0.003 * X
        pla = 300 + 0.2 * X + 0.1 * Y
        for r in PERRON_RADII_M:
            q = disk_quadratic(par, r, dx, dy)
            ok = q["valid"]
            e_lap = float(np.nanmax(np.abs(q["lap"][ok] + 2 / R0)))
            e_grad = float(np.nanmax(np.abs(q["grad"][ok] - np.hypot(gx, gy)[ok])))
            p = disk_quadratic(pla, r, dx, dy)
            e_lap_p = float(np.nanmax(np.abs(p["lap"][p["valid"]])))
            e_grad_p = float(np.nanmax(np.abs(p["grad"][p["valid"]] - np.hypot(0.2, 0.1))))
            passed = bool(e_lap <= TOL_LAP and e_grad <= TOL_GRAD and e_lap_p <= TOL_LAP and e_grad_p <= TOL_GRAD)
            out["checks"].append({"test": "paraboloid+plane", "dx": dx, "dy": dy, "r_m": r, "n_disk": q["n_disk"], "max_err_lap_paraboloid": e_lap, "max_err_grad_paraboloid": e_grad,
                                  "max_err_lap_plane": e_lap_p, "max_err_grad_plane": e_grad_p, "lap_true": -2 / R0, "pass": passed})
            out["all_pass"] &= passed
    # (3) against an independent per-cell lstsq on a noisy surface
    rng = np.random.default_rng(20261120)
    X, Y = grids(201, 201, 1.0, 1.0)
    noisy = 300 + 0.12 * X - (X ** 2 + Y ** 2) / (2 * R0) + rng.normal(0, 0.05, X.shape)
    for r in PERRON_RADII_M:
        q = disk_quadratic(noisy, r, 1.0, 1.0)
        rows, cols = np.nonzero(q["valid"])
        pick = rng.integers(0, rows.size, 200)
        d_lap, d_grad = [], []
        for k in pick:
            g, lap = direct_quadratic(noisy, r, 1.0, 1.0, rows[k], cols[k])
            d_lap.append(abs(lap - q["lap"][rows[k], cols[k]]))
            d_grad.append(abs(g - q["grad"][rows[k], cols[k]]))
        passed = bool(max(d_lap) <= 1e-9 and max(d_grad) <= 1e-9)
        out["checks"].append({"test": "convolution vs per-cell lstsq (noise 5 cm)", "r_m": r, "max_abs_diff_lap": float(max(d_lap)), "max_abs_diff_grad": float(max(d_grad)), "n_cells": 200, "pass": passed})
        out["all_pass"] &= passed
    # (4) margin: a cell exactly r + 2 cells from the frame is excluded, r + 3 kept
    Z = np.zeros((60, 60))
    Z[30, 30] = np.nan
    v = valid_interior(Z, 7.0, 1.0)
    margin_ok = bool((not v[8, 30]) and v[9, 30] and (not v[30, 39]) and v[30, 40] and (not v[30, 21]) and v[30, 20])
    out["checks"].append({"test": "exclusion margin r + 2 cells (r = 7 m, 1 m cells; a cell is excluded when a cell outside the tile or a nodata cell lies <= r + 2 cells away): row 8 excluded, row 9 kept (row -1 is 10 cells away); nodata at (30,30): cols 39 and 21 excluded, 40 and 20 kept", "pass": margin_ok})
    out["all_pass"] &= margin_ok
    # (5) plateau rule on a constructed table
    n = np.zeros(N_BINS)
    nc = np.zeros(N_BINS)
    L = np.full(N_BINS, -0.01)
    n[0], n[1:6] = 150, 1000                     # bin 0 below 200 cells -> start at 1
    nc[0], nc[1:6] = 100, 900
    L[1:5] = [-0.010, -0.011, -0.0105, -0.0098]
    L[5] = -0.02                                  # jump: |L/L_plat - 1| ~ 0.95 -> stop
    sl = np.log10(-L) * nc
    p = plateau(n, nc, sl)
    expect = -(10 ** (np.sum(np.log10(-L[1:5]) * 900) / 3600))
    rule_ok = bool(p["plateau"] and p["bins"] == [1, 2, 3, 4] and abs(p["lap_h"] / expect - 1) < 1e-12)
    out["checks"].append({"test": "plateau rule on a constructed table (bins 1-4 then a jump)", "bins": p["bins"], "lap_h": p["lap_h"], "expected": float(expect), "stop": p["stop"], "pass": rule_ok})
    out["all_pass"] &= rule_ok
    nc2 = nc.copy()
    nc2[3] = 500                                  # fraction convex 0.5 at bin 3 -> stop after bin 2 -> 2 bins -> no plateau
    p2 = plateau(n, nc2, np.log10(-L) * nc2)
    rule2_ok = bool((not p2["plateau"]) and p2["bins"] == [1, 2])
    out["checks"].append({"test": "plateau rule: convex fraction 0.5 at the third bin -> 2 bins -> no plateau", "bins": p2["bins"], "pass": rule2_ok})
    out["all_pass"] &= rule2_ok
    # bootstrap on identical blocks reproduces the point value
    tab = {"n": np.tile(n, (50, 1)), "n_convex": np.tile(nc, (50, 1)), "slog": np.tile(sl, (50, 1)), "ssum": np.tile(L * n, (50, 1)),
           "fixed_1_n": np.ones(50), "fixed_1_slog": np.full(50, -2.0), "fixed_3_n": np.ones(50), "fixed_3_slog": np.full(50, -2.0)}
    e = estimate(tab, n_boot=200)
    pt = e["point"]["lap_h"]          # summed over 50 blocks bin 0 has 7,500 cells, so the start bin is 0 here
    boot_ok = bool(e["boot"]["share_with_plateau"] == 1.0 and abs(e["boot"]["lap_ci68"][0] / pt - 1) < 1e-9 and abs(e["boot"]["lap_ci68"][1] / pt - 1) < 1e-9)
    out["checks"].append({"test": "block bootstrap over identical blocks returns the point value", "ci68": e["boot"]["lap_ci68"], "point": pt, "bins": e["point"]["bins"], "pass": boot_ok})
    out["all_pass"] &= boot_ok
    out["n_boot_block"] = N_BOOT_BLOCK
    dump(out, out_path("verify", "verify.json"))
    for c in out["checks"]:
        print({k: (f"{v:.2e}" if isinstance(v, float) else v) for k, v in c.items()})
    print("ALL PASS" if out["all_pass"] else "FAILURES")
    sys.exit(0 if out["all_pass"] else 1)


if __name__ == "__main__":
    main()
