"""Synthetic Peclet-number resolution analysis (``supp_fig5``). Exact port of the three
computations in the notebook: (1) effective k_s from the centre-line Laplacian with the
Dirichlet-outlet planform solver, (2) slope-area regression (fixed theta) on the no-flux
solver with 20 % trim, (3) local k_s = S A^m on the no-flux solver (the printed table).
No AP dependence; regression-only in Phase 6.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from lemcvt.measure.slope_area import fit_slope_area, ks_local_from_SA
from lemcvt.models.synthetic import (
    A_top_for_target_inlet_area,
    D_from_Pe,
    Pe_from_D_Ain,
    channel_slope_centerline,
    interior_slice,
    interior_slice_indices,
    laplacian_2d,
    solve_advection_diffusion_planform,
    solve_advection_diffusion_planform_noflux,
)
from lemcvt.results import RunRecorder

logger = logging.getLogger(__name__)

PARAMS = {
    "Lx": 300.0, "Ly": 200.0, "U": 1.47e-4, "K": 1.0e-4, "m": 0.35, "L_PE": 50.0, "PE_BAL": 100.0,
    "A_INLET_BAL_ANCHOR": 2.59e8, "PRIMARY_FACTOR": 1.0, "MIN_POINTS": 3,
}


def _dx_targets():
    return np.logspace(np.log10(55.0), np.log10(1.0), 25)[::-1]


def _grid_dims(dx_t, Lx, Ly):
    Nx = int(round(Lx / dx_t)) + 1
    Ny = int(round(Ly / dx_t)) + 1
    if Nx % 2 == 0:
        Nx += 1
    return Nx, Ny


def curve_laplacian(D_use, A_inlet_target, p=PARAMS, frac_trim=0.0):
    Lx, Ly, U, K, m = p["Lx"], p["Ly"], p["U"], p["K"], p["m"]
    ks_ref = U / K
    rows = []
    for dx_t in _dx_targets():
        Nx, Ny = _grid_dims(dx_t, Lx, Ly)
        A_top_use = A_top_for_target_inlet_area(Nx, Ny, Lx, Ly, A_inlet_target)
        Z, A, dx, dy = solve_advection_diffusion_planform(U, K, m, D_use, Nx, Ny, Lx, Ly, A_top_use)
        ic = (Nx - 1) // 2
        sl = interior_slice(Ny, frac_trim)
        lap = laplacian_2d(Z, dx, dy)[:, ic]
        ks_eff = ks_ref + (D_use / K) * lap
        ks_med = float(np.nanmedian(ks_eff[sl]))
        ks_norm = ks_med / ks_ref
        rows.append({"variant": "laplacian", "dx_target": float(dx_t), "dx": float(dx), "Nx": Nx, "Ny": Ny,
                     "ks_norm": float(ks_norm),
                     "valid": bool(np.isfinite(dx) and dx > 0 and np.isfinite(ks_norm) and ks_norm > 0)})
    return rows


def curve_regression(D_use, A_inlet_target, p=PARAMS, frac_trim=0.2, theta_mode="fixed"):
    Lx, Ly, U, K, m = p["Lx"], p["Ly"], p["U"], p["K"], p["m"]
    ks_ref = U / K
    rows = []
    for dx_t in _dx_targets():
        Nx, Ny = _grid_dims(dx_t, Lx, Ly)
        A_top_use = A_top_for_target_inlet_area(Nx, Ny, Lx, Ly, A_inlet_target)
        Z, A, dx, dy = solve_advection_diffusion_planform_noflux(U=U, K=K, m=m, D=D_use, Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly,
                                                                A_top=A_top_use)
        ic = (Nx - 1) // 2
        A_all = A[:, ic].copy()
        S_all = channel_slope_centerline(Z, Nx, Ny, dx, dy)
        j0, j1 = interior_slice_indices(Ny, frac_trim)
        sl = slice(j0, j1)
        A_use = A_all[sl].copy()
        S_use = S_all[sl].copy()
        fit = fit_slope_area(A_use, S_use, theta_fixed=m if theta_mode == "fixed" else None,
                             min_points=p["MIN_POINTS"])
        ks_est = fit["ks"]
        ks_norm = ks_est / ks_ref if np.isfinite(ks_est) else np.nan
        rows.append({"variant": "regression", "dx_target": float(dx_t), "dx": float(dx), "Nx": Nx, "Ny": Ny,
                     "ks": float(ks_est) if np.isfinite(ks_est) else np.nan, "ks_norm": float(ks_norm),
                     "theta": float(fit["theta"]) if np.isfinite(fit["theta"]) else np.nan,
                     "r2": float(fit["r2"]) if np.isfinite(fit["r2"]) else np.nan, "n_used": int(fit["n_good"]),
                     "frac_trim": frac_trim, "j0": j0, "j1": j1,
                     "valid": bool(np.isfinite(ks_norm) and ks_norm > 0 and dx > 0)})
    return rows


def curve_local_ks(D_use, A_inlet_target, p=PARAMS, frac_trim=0.2):
    Lx, Ly, U, K, m = p["Lx"], p["Ly"], p["U"], p["K"], p["m"]
    ks_ref = U / K
    rows = []
    for dx_t in _dx_targets():
        Nx, Ny = _grid_dims(dx_t, Lx, Ly)
        A_top_use = A_top_for_target_inlet_area(Nx, Ny, Lx, Ly, A_inlet_target)
        Z, A, dx, dy = solve_advection_diffusion_planform_noflux(U=U, K=K, m=m, D=D_use, Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly,
                                                                A_top=A_top_use)
        ic = (Nx - 1) // 2
        sl = interior_slice(Ny, frac_trim)
        A_use = A[sl, ic].copy()
        S_use = channel_slope_centerline(Z, Nx, Ny, dx, dy)[sl]
        _ks_local, _mask, n_good, ks_med, ks_q25, ks_q75 = ks_local_from_SA(A_use, S_use, m=m, min_points=p["MIN_POINTS"])
        rows.append({"variant": "local_ks", "dx_target": float(dx_t), "dx": float(dx), "Nx": Nx, "Ny": Ny,
                     "n_good": int(n_good),
                     "ks_med": float(ks_med) if np.isfinite(ks_med) else np.nan,
                     "ks_q25": float(ks_q25) if np.isfinite(ks_q25) else np.nan,
                     "ks_q75": float(ks_q75) if np.isfinite(ks_q75) else np.nan,
                     "ks_med_norm": float(ks_med / ks_ref) if np.isfinite(ks_med) else np.nan,
                     "ks_q25_norm": float(ks_q25 / ks_ref) if np.isfinite(ks_q25) else np.nan,
                     "ks_q75_norm": float(ks_q75 / ks_ref) if np.isfinite(ks_q75) else np.nan,
                     "valid": bool(np.isfinite(ks_med) and ks_med > 0 and dx > 0)})
    return rows


def run_synthetic_pe(run_id: str | None = None, results_root=None) -> RunRecorder:
    p = PARAMS
    rec = RunRecorder("synthetic_pe", "supp_fig5", run_id=run_id, root=results_root)
    A_inlet_bal = float(p["A_INLET_BAL_ANCHOR"])
    D_bal = D_from_Pe(p["PE_BAL"], A_inlet_bal, p["K"], p["L_PE"], p["m"])
    D_primary = D_bal * p["PRIMARY_FACTOR"]
    Pe_primary = Pe_from_D_Ain(D_primary, A_inlet_bal, p["K"], p["L_PE"], p["m"])
    rec.add_params("params", dict(p))
    rec.add_params("baseline", {"A_inlet_bal": A_inlet_bal, "D_bal": D_bal, "D_primary": D_primary, "Pe_primary": Pe_primary,
                                "ks_ref": p["U"] / p["K"]})
    logger.info("[primary] Pe_bal=%g, L_PE=%g m -> D_bal=%.3e, A_inlet_bal=%.3e", p["PE_BAL"], p["L_PE"], D_bal, A_inlet_bal)
    rows = curve_laplacian(D_primary, A_inlet_bal) + curve_regression(D_primary, A_inlet_bal) + curve_local_ks(D_primary, A_inlet_bal)
    df = pd.DataFrame(rows)
    rec.write_table(df, "curves")
    summary = [r for r in rows if r["variant"] == "local_ks" and np.isfinite(r["ks_med"])]
    rec.add_params("summary_local_ks", sorted(summary, key=lambda r: r["dx"]))
    rec.finalize({"analysis": "synthetic_pe"})
    return rec
