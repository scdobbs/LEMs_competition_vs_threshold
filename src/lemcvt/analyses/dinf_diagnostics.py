"""Diagnostics for the D-infinity partitioned model (addendum 2, section 2.6). Read-only.

Phase 2 reproduced v1's D-infinity numbers exactly, including the partitioned-model MSEs
that are 10^3 x larger than their D8 counterparts (AP1 3.77e6 against 2195 ft^2; GM1
4240 against 792 m^2) while the AD model is unaffected. This module measures where the
error sits and how the D-infinity channel integration behaves, without changing any
model code:

* error maps Z - Z_obs for both routings, split into channel, hillslope and boundary cells;
* a traced re-run of the model's channel integration (``_precompute_channel_network_dinf``),
  verified bit for bit against the model, that records for every cell which branch set
  its elevation (both receivers known, cardinal-only fallback, diagonal-only fallback,
  unreached), the receivers used, the step length, and whether the integration path
  passed through cells of smaller D-infinity area (hillslope cells) before reaching it;
* connectivity of the D-infinity channel set to the Dirichlet boundary, monotonicity of
  channel elevation along the receivers used, and termination of every path;
* channel-cell counts under both routings at the same A_crit and the A_crit that equalises them;
* a synthetic case (a V-shaped valley aligned with the grid, with equal side and axial
  slopes so that every D-infinity angle is exactly cardinal or diagonal) on which D8 and
  D-infinity must agree, plus a rotated V on which they must not.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from collections.abc import Sequence
from typing import Any

import numpy as np

from lemcvt.analyses.d8_vs_dinf import V1_CONFIGS
from lemcvt.config import config_snapshot, load_tile
from lemcvt.hydro.d8 import d8_area, d8_flow_direction
from lemcvt.hydro.dinf import compute_dinf, compute_dinf_area, compute_dinf_flow_direction, get_receivers
from lemcvt.hydro.fill import priority_flood_fill
from lemcvt.models.ad import advection_diffusion_model
from lemcvt.models.common import extract_grid_data, make_boundary_mask, precompute_channel_network
from lemcvt.models.dinf_models import (
    _precompute_channel_network_dinf,
    advection_diffusion_model_dinf,
    partitioned_threshold_model_dinf,
)
from lemcvt.models.partitioned import partitioned_threshold_model
from lemcvt.results import RunRecorder

logger = logging.getLogger(__name__)

BRANCH = {"boundary": 0, "both": 1, "cardinal_only": 2, "diagonal_only": 3, "unreached": -1}


def trace_dinf_integration(gd: dict, m: float, U: float, K: float, r_facet: np.ndarray, facet_idx: np.ndarray,
                           A_dinf: np.ndarray) -> dict[str, np.ndarray]:
    """Replay the model's D-infinity channel integration, recording how each cell got its value.

    The loop body and visiting order are those of ``_precompute_channel_network_dinf``; the
    caller checks that ``Z`` equals the model's ``Z_channel_all`` exactly.
    """
    Z0, dx = gd["Z0"], gd["dx"]
    Ny, Nx, n = gd["Ny"], gd["Nx"], gd["n"]
    valid, idx_map = gd["valid"], gd["idx_map"]
    rev_i, rev_j = gd["rev_i"], gd["rev_j"]
    A_flat = A_dinf[rev_j, rev_i]
    S0 = (U / K) / (A_flat ** m)

    Z = np.full(n, np.nan)
    branch = np.full(n, BRANCH["unreached"], dtype=np.int8)
    k1_used = np.full(n, -1, dtype=np.int64)
    k2_used = np.full(n, -1, dtype=np.int64)
    step_len = np.full(n, np.nan)                 # effective dL = (z_here - z_ref) / S0
    path_min_area = np.full(n, np.nan)            # smallest A_dinf on the receiver chain used (incl. self)
    through_smaller = np.zeros(n, dtype=bool)     # a receiver used has smaller area than this cell
    depth = np.full(n, -1, dtype=np.int64)

    boundary_mask = make_boundary_mask(valid)
    for j_b, i_b in zip(*np.where(boundary_mask), strict=True):
        k = idx_map[j_b, i_b]
        Z[k] = Z0[j_b, i_b]
        branch[k] = BRANCH["boundary"]
        path_min_area[k] = A_flat[k]
        depth[k] = 0

    upstream: list[list[int]] = [[] for _ in range(n)]
    for k in range(n):
        i_c, j_c = rev_i[k], rev_j[k]
        fac = facet_idx[j_c, i_c]
        if fac < 0:
            continue
        j1, i1, j2, i2, _p1, _p2 = get_receivers(j_c, i_c, fac, r_facet[j_c, i_c])
        if 0 <= i1 < Nx and 0 <= j1 < Ny and valid[j1, i1]:
            upstream[idx_map[j1, i1]].append(k)
        if 0 <= i2 < Nx and 0 <= j2 < Ny and valid[j2, i2]:
            upstream[idx_map[j2, i2]].append(k)

    queue = deque(k for k in range(n) if not np.isnan(Z[k]))
    while queue:
        kc = queue.popleft()
        for ku in upstream[kc]:
            if not np.isnan(Z[ku]):
                continue
            i_u, j_u = rev_i[ku], rev_j[ku]
            fac = facet_idx[j_u, i_u]
            if fac < 0:
                continue
            r = r_facet[j_u, i_u]
            j1, i1, j2, i2, _p1, _p2 = get_receivers(j_u, i_u, fac, r)
            cos_r, sin_r = np.cos(r), np.sin(r)
            k1 = idx_map[j1, i1] if (0 <= i1 < Nx and 0 <= j1 < Ny and valid[j1, i1]) else -1
            k2 = idx_map[j2, i2] if (0 <= i2 < Nx and 0 <= j2 < Ny and valid[j2, i2]) else -1
            z1k = k1 >= 0 and not np.isnan(Z[k1])
            z2k = k2 >= 0 and not np.isnan(Z[k2])
            s = S0[ku]
            if not np.isfinite(s):
                continue
            if z1k and z2k and cos_r > 1e-12:
                z = Z[k1] + (s * dx - (Z[k1] - Z[k2]) * sin_r) / cos_r
                b, ka, kb = BRANCH["both"], k1, k2
                zref = Z[k1]
            elif z1k:
                z = Z[k1] + s * dx
                b, ka, kb = BRANCH["cardinal_only"], k1, -1
                zref = Z[k1]
            elif z2k:
                z = Z[k2] + s * dx * np.sqrt(2.0)
                b, ka, kb = BRANCH["diagonal_only"], -1, k2
                zref = Z[k2]
            else:
                continue
            if not np.isfinite(z):
                continue
            Z[ku] = z
            branch[ku] = b
            k1_used[ku], k2_used[ku] = ka, kb
            step_len[ku] = (z - zref) / s
            used = [kk for kk in (ka, kb) if kk >= 0]
            path_min_area[ku] = min([A_flat[ku], *[path_min_area[kk] for kk in used]])
            through_smaller[ku] = any(A_flat[kk] < A_flat[ku] for kk in used)
            depth[ku] = 1 + max(depth[kk] for kk in used)
            queue.append(ku)
    return {"Z": Z, "branch": branch, "k1_used": k1_used, "k2_used": k2_used, "step_len": step_len,
            "path_min_area": path_min_area, "through_smaller": through_smaller, "depth": depth, "S0": S0,
            "A_flat": A_flat}


def _class_errors(err: np.ndarray, classes: dict[str, np.ndarray]) -> dict[str, Any]:
    sse_total = float(np.nansum(err ** 2))
    out: dict[str, Any] = {"sse_total": sse_total, "n_total": int(np.sum(np.isfinite(err)))}
    for name, msk in classes.items():
        e = err[msk]
        e = e[np.isfinite(e)]
        out[name] = {"n": int(e.size), "sse_share": float(np.sum(e ** 2) / sse_total) if sse_total > 0 else np.nan,
                     "rms": float(np.sqrt(np.mean(e ** 2))) if e.size else np.nan,
                     "max_abs": float(np.max(np.abs(e))) if e.size else np.nan,
                     "mean": float(np.mean(e)) if e.size else np.nan}
    return out


def diagnose_tile(td, cfg: dict[str, Any], rec: RunRecorder, save_grids: bool = True) -> dict[str, Any]:
    tile = td.cfg.tile
    Z0, A, FD, dx = td.Z, td.A, td.FD, td.dx
    m, U, K, D, A_crit = float(cfg["m"]), float(cfg["U"]), float(cfg["K"]), float(cfg["D"]), float(cfg["A_crit"])
    relief = float(np.nanmax(Z0) - np.nanmin(Z0))
    angles, r_facet, facet_idx, A_dinf = compute_dinf(Z0, dx, FD)

    # models (as d8_vs_dinf runs them)
    Z_pt_d8, mask_d8 = partitioned_threshold_model(Z0, A, FD, dx, m, U, K)(A_crit, D)
    pt_dinf = partitioned_threshold_model_dinf(Z0, A, FD, dx, m, U, K, angles, r_facet, facet_idx, A_dinf)
    Z_pt_dinf, mask_dinf = pt_dinf(A_crit, D)
    Z_ad_d8 = advection_diffusion_model(Z0, A, FD, dx, m)(U, K, D)
    Z_ad_dinf = advection_diffusion_model_dinf(Z0, A, FD, dx, m, angles, r_facet, facet_idx, A_dinf)(U, K, D)

    valid = np.isfinite(Z0)
    boundary = make_boundary_mask(valid)
    res: dict[str, Any] = {"params": cfg, "relief": relief,
                           "mse": {"pt_d8": float(np.nanmean((Z_pt_d8 - Z0) ** 2)), "pt_dinf": float(np.nanmean((Z_pt_dinf - Z0) ** 2)),
                                   "ad_d8": float(np.nanmean((Z_ad_d8 - Z0) ** 2)), "ad_dinf": float(np.nanmean((Z_ad_dinf - Z0) ** 2))}}

    # 1. error maps by class
    for name, Z, msk in (("pt_d8", Z_pt_d8, mask_d8), ("pt_dinf", Z_pt_dinf, mask_dinf)):
        err = Z - Z0
        classes = {"channel": msk & ~boundary, "hillslope": ~msk & ~boundary & valid, "boundary": boundary}
        res[f"errors_{name}"] = _class_errors(err, classes)
        res[f"errors_{name}"]["n_nan_output"] = int(np.sum(~np.isfinite(Z) & valid))
        if save_grids:
            rec.write_grid(td.filled.with_data(err), f"{tile}_{name}_error")
    res["errors_ad_dinf"] = _class_errors(Z_ad_dinf - Z0, {"channel_dinf": mask_dinf & ~boundary,
                                                             "hillslope": ~mask_dinf & ~boundary & valid, "boundary": boundary})

    # 2. the D-infinity channel set and its integration
    gd = extract_grid_data(Z0, A, FD, dx)
    Z_model, _ds, k_out, bc_inds, _bc_vals = _precompute_channel_network_dinf(gd, m, U, K, angles, r_facet, facet_idx, A_dinf)
    tr = trace_dinf_integration(gd, m, U, K, r_facet, facet_idx, A_dinf)
    same = np.array_equal(np.isnan(Z_model), np.isnan(tr["Z"])) and np.array_equal(Z_model[~np.isnan(Z_model)], tr["Z"][~np.isnan(tr["Z"])])
    res["trace_reproduces_model"] = bool(same)
    n = gd["n"]
    ch = tr["A_flat"] >= A_crit
    ch[k_out] = True
    b = tr["branch"]
    res["integration_branches_all_cells"] = {k: int(np.sum(b == v)) for k, v in BRANCH.items()}
    res["integration_branches_channel_cells"] = {k: int(np.sum((b == v) & ch)) for k, v in BRANCH.items()}
    res["n_cells"], res["n_channel_dinf"], res["n_channel_d8"] = int(n), int(ch.sum()), int(mask_d8.sum())

    # 2a connectivity: channel cells reachable from boundary channel cells through channel-only receiver links
    k1u, k2u = tr["k1_used"], tr["k2_used"]
    connected = np.zeros(n, dtype=bool)
    is_bc = np.zeros(n, dtype=bool)
    is_bc[bc_inds] = True
    connected[is_bc & ch] = True
    ups_ch: list[list[int]] = [[] for _ in range(n)]
    for k in np.nonzero(ch)[0]:
        for kk in (k1u[k], k2u[k]):
            if kk >= 0 and ch[kk]:
                ups_ch[kk].append(int(k))
    dq = deque(np.nonzero(connected)[0].tolist())
    while dq:
        kc = dq.popleft()
        for ku in ups_ch[kc]:
            if not connected[ku]:
                connected[ku] = True
                dq.append(ku)
    res["channel_connectivity"] = {
        "n_channel": int(ch.sum()),
        "n_connected_to_boundary_through_channel_cells": int(np.sum(connected & ch)),
        "n_not_connected": int(np.sum(ch & ~connected)),
        "n_channel_integrated_through_smaller_area_cell": int(np.sum(ch & tr["through_smaller"])),
        "n_channel_path_passes_hillslope": int(np.sum(ch & (tr["path_min_area"] < A_crit))),
        "n_channel_unreached_nan": int(np.sum(ch & (b == BRANCH["unreached"]))),
    }
    # the same for D8 (every channel cell drains through channel cells by construction; verify)
    Zc_d8, ds_d8, k_out_d8, _bi, _bv = precompute_channel_network(gd, m, U, K)
    ch8 = gd["A_flat"] >= A_crit
    ch8[k_out_d8] = True
    rec_ok = ds_d8 >= 0
    res["channel_connectivity_d8"] = {"n_channel": int(ch8.sum()),
                                      "n_channel_whose_receiver_is_hillslope": int(np.sum(ch8 & rec_ok & ~ch8[np.maximum(ds_d8, 0)])),
                                      "n_channel_unreached_nan": int(np.sum(ch8 & np.isnan(Zc_d8)))}

    # 2b monotonicity along the receivers used
    zc = tr["Z"]
    viol = np.zeros(n, dtype=bool)
    drop = np.full(n, np.nan)
    for k in np.nonzero(ch & (b > 0))[0]:
        zs = [zc[kk] for kk in (k1u[k], k2u[k]) if kk >= 0]
        drop[k] = zc[k] - max(zs)
        viol[k] = zc[k] <= max(zs)
    res["channel_monotonicity"] = {
        "n_channel_integrated": int(np.sum(ch & (b > 0))),
        "n_not_higher_than_receiver_used": int(viol.sum()),
        "min_drop_to_receiver": float(np.nanmin(drop)) if np.any(np.isfinite(drop)) else None,
        "median_drop_to_receiver": float(np.nanmedian(drop)) if np.any(np.isfinite(drop)) else None,
        "n_channel_elev_above_obs_max_plus_relief": int(np.sum(ch & (zc > np.nanmax(Z0) + relief))),
        "channel_elev_max": float(np.nanmax(zc[ch])), "obs_max": float(np.nanmax(Z0)),
        "channel_elev_rms_error_vs_obs": float(np.sqrt(np.nanmean((zc[ch] - Z0[gd['rev_j'][ch], gd['rev_i'][ch]]) ** 2))),
    }
    # D8 channel elevations for comparison
    res["channel_monotonicity_d8"] = {
        "channel_elev_max": float(np.nanmax(Zc_d8[ch8])),
        "channel_elev_rms_error_vs_obs": float(np.sqrt(np.nanmean((Zc_d8[ch8] - Z0[gd['rev_j'][ch8], gd['rev_i'][ch8]]) ** 2))),
    }
    # 2c step lengths and S0 along paths
    sl = tr["step_len"][ch & (b > 0)]
    res["step_length_channel"] = {"min": float(np.nanmin(sl)), "q01": float(np.nanpercentile(sl, 1)),
                                  "median": float(np.nanmedian(sl)), "q99": float(np.nanpercentile(sl, 99)),
                                  "max": float(np.nanmax(sl)), "n_negative": int(np.sum(sl < 0)),
                                  "n_above_dx_sqrt2": int(np.sum(sl > dx * math.sqrt(2) * 1.0001)), "dx": dx}
    res["S0_at_A_crit"] = float((U / K) / A_crit ** m)
    res["S0_channel_max"] = float(np.nanmax(tr["S0"][ch]))
    res["S0_cell_area_cell"] = float((U / K) / (dx * dx) ** m)

    # 3. area distributions and A_crit equivalence
    af, a8 = A_dinf[valid], A[valid]
    qs = [0.5, 0.9, 0.99, 0.999, 1.0]
    res["area_quantiles"] = {"q": qs, "A_d8": [float(np.quantile(a8, q)) for q in qs], "A_dinf": [float(np.quantile(af, q)) for q in qs]}
    n8 = int(np.sum(a8 >= A_crit))
    res["A_crit_dinf_matching_d8_count"] = float(np.sort(af)[::-1][n8 - 1]) if n8 > 0 else None
    res["n_channel_at_A_crit"] = {"d8": n8, "dinf": int(np.sum(af >= A_crit))}
    res["n_channel_dinf_at_A_crit_ladder"] = {str(a): int(np.sum(af >= a)) for a in (1e3, 2e3, 5e3, 1e4, 2e4)}
    res["n_channel_d8_at_A_crit_ladder"] = {str(a): int(np.sum(a8 >= a)) for a in (1e3, 2e3, 5e3, 1e4, 2e4)}
    if save_grids:
        rec.write_grid(td.filled.with_data(np.where(mask_dinf, 1.0, 0.0) + np.where(mask_d8, 2.0, 0.0)), f"{tile}_masks_dinf1_d8_2")
        zc2 = np.full(Z0.shape, np.nan)
        zc2[gd["rev_j"], gd["rev_i"]] = zc
        rec.write_grid(td.filled.with_data(zc2), f"{tile}_Zchannel_dinf")
        br2 = np.full(Z0.shape, np.nan)
        br2[gd["rev_j"], gd["rev_i"]] = b.astype(float)
        rec.write_grid(td.filled.with_data(br2), f"{tile}_integration_branch")
    logger.info("%s: PT MSE d8=%.4e dinf=%.4e; channel cells d8=%d dinf=%d; dinf channel through hillslope=%d, "
                "non-monotone=%d, trace ok=%s", tile, res["mse"]["pt_d8"], res["mse"]["pt_dinf"], res["n_channel_d8"],
                res["n_channel_dinf"], res["channel_connectivity"]["n_channel_path_passes_hillslope"],
                res["channel_monotonicity"]["n_not_higher_than_receiver_used"], same)
    return res


# ----------------------------------------------------------------------------- synthetic

def _padded_hydro(z: np.ndarray, dx: float, pad: int = 12):
    """Fill, D8, area and D-infinity routing computed on a padded copy and cropped (tests/synthetic_tile)."""
    ny, nx = z.shape
    zp = np.pad(z, pad, mode="reflect")
    yy, xx = np.indices(zp.shape)
    dist = np.maximum.reduce([np.maximum(pad - yy, 0), np.maximum(yy - (ny + pad - 1), 0),
                              np.maximum(pad - xx, 0), np.maximum(xx - (nx + pad - 1), 0)])
    zp = zp - 2.0 * dist * dx
    zf = priority_flood_fill(zp, dx)
    fd = d8_flow_direction(zf)
    area = d8_area(fd, zf, dx)
    angles, r_facet, facet_idx = compute_dinf_flow_direction(zf, dx, FD_d8=fd)
    A_dinf = compute_dinf_area(zf, dx, angles, r_facet, facet_idx)
    sl = (slice(pad, pad + ny), slice(pad, pad + nx))
    return (zf[sl].copy(), fd[sl].copy(), area[sl].copy(), angles[sl].copy(), r_facet[sl].copy(),
            facet_idx[sl].copy(), A_dinf[sl].copy())


def v_valley(ny: int, nx: int, dx: float, side_slope: float, axial_slope: float, z_base: float = 100.0) -> np.ndarray:
    """z = side_slope*|x - x_c| + axial_slope*(L_y - y): a V-valley whose axis runs south (row index increasing)."""
    y, x = np.meshgrid(np.arange(ny) * dx, np.arange(nx) * dx, indexing="ij")
    xc = 0.5 * (nx - 1) * dx
    return z_base + side_slope * np.abs(x - xc) + axial_slope * ((ny - 1) * dx - y)


def synthetic_case(name: str, z: np.ndarray, dx: float, m: float, U: float, K: float, D: float, A_crit: float,
                   rec: RunRecorder) -> dict[str, Any]:
    zf, fd, area, angles, r_facet, facet_idx, A_dinf = _padded_hydro(z, dx)
    Z_d8, mask_d8 = partitioned_threshold_model(zf, area, fd, dx, m, U, K)(A_crit, D)
    Z_di, mask_di = partitioned_threshold_model_dinf(zf, area, fd, dx, m, U, K, angles, r_facet, facet_idx, A_dinf)(A_crit, D)
    Zad8 = advection_diffusion_model(zf, area, fd, dx, m)(U, K, D)
    Zadi = advection_diffusion_model_dinf(zf, area, fd, dx, m, angles, r_facet, facet_idx, A_dinf)(U, K, D)
    gd = extract_grid_data(zf, area, fd, dx)
    tr = trace_dinf_integration(gd, m, U, K, r_facet, facet_idx, A_dinf)
    relief = float(np.nanmax(zf) - np.nanmin(zf))
    ch = tr["A_flat"] >= A_crit
    # analytic check along the axis: z_axis = z_bc + (U/K) sum A^-m dx (cardinal steps), using the D8 area
    xc = (zf.shape[1] - 1) // 2
    axis_rows = np.arange(zf.shape[0] - 1, -1, -1)
    z_axis_model = Z_d8[axis_rows, xc]
    z_exp = np.empty_like(z_axis_model)
    z_exp[0] = zf[axis_rows[0], xc]
    for t in range(1, axis_rows.size):
        z_exp[t] = z_exp[t - 1] + (U / K) * area[axis_rows[t], xc] ** (-m) * dx
    on_axis_channel = area[axis_rows, xc] >= A_crit
    out = {
        "name": name, "shape": list(z.shape), "dx": dx, "relief": relief,
        "angles_unique_deg": sorted({round(float(a), 6) for a in np.degrees(angles[np.isfinite(angles)]).ravel()})[:16],
        "n_angles_not_multiple_of_45": int(np.sum(np.abs(np.remainder(np.degrees(angles[np.isfinite(angles)]), 45.0)) > 1e-9)),
        "max_abs_dA_dinf_minus_d8": float(np.nanmax(np.abs(A_dinf - area))),
        "max_abs_dZ_pt_dinf_minus_d8_over_relief": float(np.nanmax(np.abs(Z_di - Z_d8)) / relief),
        "max_abs_dZ_ad_dinf_minus_d8_over_relief": float(np.nanmax(np.abs(Zadi - Zad8)) / relief),
        "n_channel_d8": int(mask_d8.sum()), "n_channel_dinf": int(mask_di.sum()),
        "n_mask_differs": int(np.sum(mask_d8 != mask_di)),
        "integration_branches_channel_cells": {k: int(np.sum((tr["branch"] == v) & ch)) for k, v in BRANCH.items()},
        "n_channel_path_passes_hillslope": int(np.sum(ch & (tr["path_min_area"] < A_crit))),
        "n_channel_integrated_through_smaller_area_cell": int(np.sum(ch & tr["through_smaller"])),
        "axis_channel_max_abs_dz_model_minus_analytic_over_relief": float(np.max(np.abs(z_axis_model[on_axis_channel] - z_exp[on_axis_channel])) / relief) if on_axis_channel.any() else None,
        "mse_pt_d8_vs_input": float(np.nanmean((Z_d8 - zf) ** 2)), "mse_pt_dinf_vs_input": float(np.nanmean((Z_di - zf) ** 2)),
    }
    rec.write_grid(_grid(zf, dx), f"synthetic_{name}_input")
    rec.write_grid(_grid(Z_di - Z_d8, dx), f"synthetic_{name}_pt_dinf_minus_d8")
    logger.info("synthetic %s: dA max=%.3g, dZ_pt/relief=%.3g, dZ_ad/relief=%.3g, odd angles=%d", name,
                out["max_abs_dA_dinf_minus_d8"], out["max_abs_dZ_pt_dinf_minus_d8_over_relief"],
                out["max_abs_dZ_ad_dinf_minus_d8_over_relief"], out["n_angles_not_multiple_of_45"])
    return out


def _grid(data: np.ndarray, dx: float):
    from rasterio.transform import Affine

    from lemcvt.io.grids import Grid

    return Grid(data=np.asarray(data, dtype=np.float64), transform=Affine(dx, 0.0, 0.0, 0.0, -dx, 0.0), crs=None,
                nodata=None, name="synthetic")


def run_dinf_diagnostics(tiles: Sequence[str] = ("AP1", "GM1"), synthetic: bool = True, run_id: str | None = None,
                         results_root=None, save_grids: bool = True) -> RunRecorder:
    rec = RunRecorder("dinf_diagnostics", "_".join(tiles) if tiles else "synthetic", run_id=run_id, root=results_root)
    out: dict[str, Any] = {}
    snaps = {}
    for tile in tiles:
        td = load_tile(tile)
        snaps[tile] = config_snapshot(td.cfg)
        out[tile] = diagnose_tile(td, V1_CONFIGS[tile], rec, save_grids=save_grids)
    rec.add_params("tiles", out)
    if synthetic:
        syn = {}
        # GM-like parameters; D8 == D-inf when every slope direction is cardinal or diagonal
        m, U, K, D = 0.4, 1.47e-4, 2.2e-5, 1.0e-2
        n, dx = 201, 1.0
        syn["plane_south"] = synthetic_case("plane_south", v_valley(n, n, dx, 0.0, 0.1), dx, m, U, K, D, 500.0, rec)
        syn["v_equal_slopes"] = synthetic_case("v_equal_slopes", v_valley(n, n, dx, 0.1, 0.1), dx, m, U, K, D, 500.0, rec)
        syn["v_rotated_20deg"] = synthetic_case("v_rotated_20deg", v_valley(n, n, dx, 0.1, 0.1 * math.tan(math.radians(20))),
                                                dx, m, U, K, D, 500.0, rec)
        rec.add_params("synthetic", syn)
    rec.finalize({"tiles": snaps, "v1_configs": V1_CONFIGS})
    return rec
