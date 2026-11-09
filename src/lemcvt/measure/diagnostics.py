"""Measurement diagnostics for the channel-threshold decision (addendum 2, section 6).

``lemcvt measure-diagnostics --tiles all`` writes, per tile and pooled per site, the
evidence the threshold rule needs. Nothing here writes to a tile config.

Pre-registered choices (fixed before the run; stated here so the report can quote them):

* ``N_MIN_FIXED_BASINS = 50``: a basin enters the basin-fixed ladder (6.1) when it holds
  at least 50 channel pixels at A_c = 1e4 m^2, i.e. at least ~50 m of channel above
  1e4 m^2; 10 pixels (the fitting minimum) would admit basins whose chi range at 1e4 is a
  few metres and whose theta is unconstrained.
* ``MIN_RELIEF_M = 10``: a basin enters a chi fit only if its channel relief
  (max z - z_outlet) is at least 10 m, ten times the vertical noise of the lidar/TIN
  surfaces and the vertical interval of the slope-area method.
* ``MIN_CHI_RANGE_M = 1`` at ``THETA_REF_CHI = 0.45``: the chi range of the basin,
  evaluated at a fixed reference concavity so that the criterion does not move with the
  theta being fitted, must be at least 1 m (A0 = 1 m^2), about 40-100 m of channel at
  A ~ 1e4 m^2; a through-origin regression over a shorter predictor range has no leverage.
* ``MIN_PIXELS_PER_BASIN = 10`` as in `lemcvt.measure.concavity`.
* Bootstrap unit: the basin (resampled with replacement, 200 draws). Steps with fewer than
  ``MIN_BASINS_FLAG = 10`` basins are flagged.
* Scaling break (6.3): binned medians of the D8 slope in 0.1-decade bins above the
  slope-area rollover, continuous piecewise-linear fits with 1, 2 and 3 segments by
  weighted least squares (weights = pixel counts; the maximum-likelihood breakpoint under
  Gaussian bin errors), BIC = n ln(SSR_w/n) + k ln n with k = 2, 4, 6; breakpoint
  bootstrap over pixels (200 draws).
* Edge truncation (6.5): a basin is "truncated" when its catchment inside the tile is not
  the whole catchment: the area at its outlet exceeds the in-tile pixel count times the
  cell area by more than 5 %, or its catchment touches the tile edge away from the outlet
  (more than 3 edge pixels).
* Curvature synthetic (6.6): parabolic ridges with crest widths spanning the observed
  hilltop-to-channel distances of each site, with and without TIN facets of the size
  measured on the AP tiles and with Gaussian noise of the roughness measured on GM;
  recovered C_ht is the median of -laplacian over crest pixels.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, label, uniform_filter

from lemcvt.config import TileConfig, config_snapshot, load_tile, load_tile_config
from lemcvt.hydro.d8 import d8_receiver_arrays
from lemcvt.measure.chi import ChannelNetwork, channel_network, chi_for_thetas, relative_elevation
from lemcvt.measure.concavity import DEFAULT_THETAS
from lemcvt.measure.hilltop_curvature import hilltop_mask, total_curvature
from lemcvt.measure.thresholds import d8_slope, slope_area_rollover
from lemcvt.results import RunRecorder, find_runs

logger = logging.getLogger(__name__)

FINE_LADDER = tuple(float(f"{v:.6g}") for v in 10 ** np.arange(2.0, 5.0001, 0.1))     # 31 steps, 0.1 decade
FIXED_LADDER = tuple(a for a in FINE_LADDER if a <= 1e4 + 1e-6)                        # 1e2 .. 1e4
A_REF_FIXED = 1e4
N_MIN_FIXED_BASINS = 50
MIN_RELIEF_M = 10.0
MIN_CHI_RANGE_M = 1.0
THETA_REF_CHI = 0.45
MIN_PIXELS_PER_BASIN = 10
MIN_BASINS_FLAG = 10
N_BOOT = 200
THETA_REF_COMMON = 0.45
CURV_HALF_WIDTHS = (1, 2, 3, 4, 5, 7, 10, 12, 15, 20, 25, 30)


# ----------------------------------------------------------------------------- basins

def edge_outlet_labels(area: np.ndarray, fd: np.ndarray) -> np.ndarray:
    """Flat index of the tile-edge pixel every cell drains to (-1 if it reaches none).

    Cells are processed in descending area (receivers before donors; ties by flat index).
    """
    ny, nx = area.shape
    r_to, c_to, good = d8_receiver_arrays(fd)
    lab = np.full(ny * nx, -1, dtype=np.int64)
    edge = np.zeros((ny, nx), dtype=bool)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    flat = np.arange(ny * nx)
    lab[edge.ravel()] = flat[edge.ravel()]
    a = np.where(np.isfinite(area), area, -np.inf).ravel()
    order = np.lexsort((flat, -a))
    rt, ct, gd = r_to.ravel(), c_to.ravel(), good.ravel()
    ed = edge.ravel()
    for k in order.tolist():
        if ed[k] or not gd[k]:
            continue
        lab[k] = lab[rt[k] * nx + ct[k]]
    return lab.reshape(ny, nx)


def basin_truncation(area: np.ndarray, labels: np.ndarray, frac_tol: float = 0.05, max_edge_px: int = 3) -> pd.DataFrame:
    """Per edge-outlet basin: in-tile pixel count, outlet area, external area fraction, edge contact, truncated flag."""
    ny, nx = area.shape
    cell = float(np.nanmin(area))
    lab = labels.ravel()
    ok = lab >= 0
    edge = np.zeros((ny, nx), dtype=bool)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    ids, inv = np.unique(lab[ok], return_inverse=True)
    n_px = np.bincount(inv, minlength=ids.size)
    n_edge = np.bincount(inv, weights=edge.ravel()[ok].astype(float), minlength=ids.size)
    A_out = area.ravel()[ids]
    external = A_out - n_px * cell
    frac_ext = external / A_out
    truncated = (frac_ext > frac_tol) | (n_edge > max_edge_px)
    return pd.DataFrame({"outlet_flat": ids, "n_px": n_px, "n_edge_px": n_edge.astype(int), "A_outlet": A_out,
                         "external_area_frac": frac_ext, "truncated": truncated})


def _restrict_area(area: np.ndarray, labels: np.ndarray, keep: set[int] | None) -> np.ndarray:
    if keep is None:
        return area
    return np.where(np.isin(labels, list(keep)), area, np.nan)


# ----------------------------------------------------------------------------- chi statistics

def basin_stats(net: ChannelNetwork, thetas: np.ndarray, labels: np.ndarray, min_pixels: int = MIN_PIXELS_PER_BASIN,
                min_relief: float = MIN_RELIEF_M, min_chi_range: float = MIN_CHI_RANGE_M,
                theta_ref: float = THETA_REF_CHI) -> dict[str, Any]:
    """Per-basin through-origin statistics (sxx, sxy, syy per theta) after the pre-registered filters."""
    chi = chi_for_thetas(net, thetas)
    chi_ref = chi_for_thetas(net, np.array([theta_ref]))[:, 0]
    zrel = relative_elevation(net)
    rows = []
    sxx, sxy, syy = [], [], []
    for o in net.outlet_index:
        idx = np.nonzero(net.basin == o)[0]
        idx = idx[idx != o]
        n = idx.size
        relief = float(zrel[idx].max()) if n else 0.0
        chir = float(chi_ref[idx].max()) if n else 0.0
        keep = n >= min_pixels and relief >= min_relief and chir >= min_chi_range
        rows.append({"outlet_flat": int(labels[net.rows[o], net.cols[o]]), "n_px": int(n), "relief": relief,
                     "chi_range_ref": chir, "kept": bool(keep), "edge_outlet": bool(net.rows[o] in (0, net.mask.shape[0] - 1)
                                                                                or net.cols[o] in (0, net.mask.shape[1] - 1))})
        if keep:
            x, y = chi[idx], zrel[idx]
            sxx.append((x * x).sum(axis=0))
            sxy.append((x * y[:, None]).sum(axis=0))
            syy.append(float((y * y).sum()))
    return {"basins": pd.DataFrame(rows), "sxx": np.array(sxx), "sxy": np.array(sxy), "syy": np.array(syy),
            "n_px_kept": int(sum(r["n_px"] for r in rows if r["kept"])),
            "n_dropped_pixels": sum(1 for r in rows if r["n_px"] < min_pixels),
            "n_dropped_relief": sum(1 for r in rows if r["n_px"] >= min_pixels and r["relief"] < min_relief),
            "n_dropped_chi": sum(1 for r in rows if r["n_px"] >= min_pixels and r["relief"] >= min_relief and r["chi_range_ref"] < min_chi_range)}


def fit_from_stats(sxx: np.ndarray, sxy: np.ndarray, syy: np.ndarray, thetas: np.ndarray, n_boot: int = N_BOOT,
                   seed: int = 0) -> dict[str, Any]:
    """theta*, pooled k_s, R2 and basin-bootstrap intervals from stacked per-basin statistics."""
    nb = sxx.shape[0] if sxx.ndim == 2 else 0
    out: dict[str, Any] = {"n_basins": int(nb), "flag_few_basins": bool(nb < MIN_BASINS_FLAG)}
    if nb == 0:
        out.update({"theta_star": None, "ks": None, "r2": None})
        return out
    with np.errstate(invalid="ignore", divide="ignore"):
        ssr_b = syy[:, None] - sxy * sxy / sxx
    ssr = np.nansum(ssr_b, axis=0)
    j = int(np.nanargmin(ssr))
    ks = float(sxy[:, j].sum() / sxx[:, j].sum())
    ss_tot = float(syy.sum())
    out.update({"theta_star": float(thetas[j]), "ks": ks, "r2": float(1 - ssr[j] / ss_tot) if ss_tot > 0 else None,
                "ssr_total": ssr.tolist(), "theta_at_edge": bool(j in (0, thetas.size - 1))})
    if n_boot > 0 and nb >= 3:
        rng = np.random.default_rng(seed)
        th = np.empty(n_boot)
        kk = np.empty(n_boot)
        for i in range(n_boot):
            w = np.bincount(rng.integers(0, nb, nb), minlength=nb).astype(float)
            tot = np.nansum(ssr_b * w[:, None], axis=0)
            jj = int(np.nanargmin(tot))
            th[i] = thetas[jj]
            kk[i] = (sxy[:, jj] * w).sum() / (sxx[:, jj] * w).sum()
        out.update({"theta_ci68": [float(np.percentile(th, 16)), float(np.percentile(th, 84))],
                    "theta_ci95": [float(np.percentile(th, 2.5)), float(np.percentile(th, 97.5))],
                    "ks_ci68": [float(np.percentile(kk, 16)), float(np.percentile(kk, 84))],
                    "ks_ci95": [float(np.percentile(kk, 2.5)), float(np.percentile(kk, 97.5))], "n_boot": n_boot,
                    "bootstrap_unit": "basin"})
    return out


def ks_at_theta_from_stats(net: ChannelNetwork, theta: float, labels: np.ndarray, **filters) -> dict[str, Any]:
    st = basin_stats(net, np.array([theta]), labels, **filters)
    if st["sxx"].size == 0:
        return {"theta": theta, "ks": None, "n_basins": 0}
    ks = float(st["sxy"][:, 0].sum() / st["sxx"][:, 0].sum())
    ss_res = float((st["syy"] - st["sxy"][:, 0] ** 2 / st["sxx"][:, 0]).sum())
    ss_tot = float(st["syy"].sum())
    return {"theta": theta, "ks": ks, "r2": 1 - ss_res / ss_tot if ss_tot > 0 else None, "n_basins": int(st["sxx"].shape[0]),
            "n_px": st["n_px_kept"]}


def ladder(Z, A, FD, dx, dy, labels, thresholds: Sequence[float], thetas: np.ndarray, keep: set[int] | None = None,
           n_boot: int = N_BOOT, seed: int = 0, store_stats: dict | None = None, tag: str = "") -> list[dict[str, Any]]:
    steps = []
    for a_c in thresholds:
        t0 = time.perf_counter()
        try:
            net = channel_network(Z, _restrict_area(A, labels, keep), FD, a_c, dx, dy)
            st = basin_stats(net, thetas, labels)
            fit = fit_from_stats(st["sxx"], st["sxy"], st["syy"], thetas, n_boot=n_boot, seed=seed)
            fit.update({"A_c": a_c, "n_px": st["n_px_kept"], "n_basins_total": int(net.n_basins),
                        "n_dropped_pixels": st["n_dropped_pixels"], "n_dropped_relief": st["n_dropped_relief"],
                        "n_dropped_chi": st["n_dropped_chi"], "wall_s": time.perf_counter() - t0})
            if store_stats is not None:
                store_stats[a_c] = (st["sxx"], st["sxy"], st["syy"])
        except ValueError as exc:
            fit = {"A_c": a_c, "error": str(exc), "theta_star": None, "ks": None, "n_basins": 0, "flag_few_basins": True}
        steps.append(fit)
        logger.info("%s A_c=%-8g theta*=%s ks=%s basins=%s px=%s R2=%s", tag, a_c, fit.get("theta_star"),
                    None if fit.get("ks") is None else f"{fit['ks']:.3g}", fit.get("n_basins"), fit.get("n_px"),
                    None if fit.get("r2") is None else f"{fit['r2']:.3f}")
    return steps


def fixed_basin_set(Z, A, FD, dx, dy, labels, a_ref: float = A_REF_FIXED, n_min: int = N_MIN_FIXED_BASINS) -> tuple[set[int], dict]:
    net = channel_network(Z, A, FD, a_ref, dx, dy)
    counts: dict[int, int] = {}
    for o in net.outlet_index:
        lab = int(labels[net.rows[o], net.cols[o]])
        counts[lab] = counts.get(lab, 0) + int(np.sum(net.basin == o))
    keep = {k for k, v in counts.items() if v >= n_min and k >= 0}
    return keep, {"A_ref": a_ref, "N_min": n_min, "n_basins_at_ref": len(counts), "n_kept": len(keep),
                  "kept_outlets": sorted(keep)}


# ----------------------------------------------------------------------------- scaling break

def _pw_design(x: np.ndarray, breaks: Sequence[float]) -> np.ndarray:
    cols = [np.ones_like(x), x] + [np.maximum(x - b, 0.0) for b in breaks]
    return np.column_stack(cols)


def _wls(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, float]:
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    r = y - X @ beta
    return beta, float(np.sum(w * r * r))


def piecewise_fits(x: np.ndarray, y: np.ndarray, w: np.ndarray, min_sep: int = 2) -> dict[str, Any]:
    """1-, 2- and 3-segment continuous piecewise-linear WLS fits with breaks on bin centres; BIC per model."""
    n = x.size
    out: dict[str, Any] = {"n_bins": int(n)}
    if n < 6:
        out["error"] = "fewer than 6 bins above the rollover"
        return out
    b1, ssr1 = _wls(_pw_design(x, []), y, w)
    out["one"] = {"slope": float(b1[1]), "intercept": float(b1[0]), "ssr": ssr1, "bic": n * math.log(ssr1 / n) + 2 * math.log(n)}
    best2 = None
    for i in range(min_sep, n - min_sep):
        beta, ssr = _wls(_pw_design(x, [x[i]]), y, w)
        if best2 is None or ssr < best2[1]:
            best2 = (x[i], ssr, beta)
    bx, ssr2, beta2 = best2
    out["two"] = {"break_log10A": float(bx), "slope_below": float(beta2[1]), "slope_above": float(beta2[1] + beta2[2]),
                  "ssr": ssr2, "bic": n * math.log(ssr2 / n) + 4 * math.log(n)}
    best3 = None
    for i in range(min_sep, n - 2 * min_sep):
        for j in range(i + min_sep, n - min_sep):
            beta, ssr = _wls(_pw_design(x, [x[i], x[j]]), y, w)
            if best3 is None or ssr < best3[2]:
                best3 = (x[i], x[j], ssr, beta)
    if best3 is not None:
        bi, bj, ssr3, beta3 = best3
        out["three"] = {"breaks_log10A": [float(bi), float(bj)], "slopes": [float(beta3[1]), float(beta3[1] + beta3[2]),
                                                                        float(beta3[1] + beta3[2] + beta3[3])],
                        "ssr": ssr3, "bic": n * math.log(ssr3 / n) + 6 * math.log(n)}
    bics = {k: out[k]["bic"] for k in ("one", "two", "three") if k in out}
    out["bic_best"] = min(bics, key=bics.get)
    return out


def _binned_medians(la: np.ndarray, ls: np.ndarray, edges: np.ndarray, min_count: int = 50):
    idx = np.digitize(la, edges) - 1
    ok = (idx >= 0) & (idx < edges.size - 1)
    idx, ls_ok = idx[ok], ls[ok]
    counts = np.bincount(idx, minlength=edges.size - 1)
    med = np.full(edges.size - 1, np.nan)
    order = np.argsort(idx, kind="stable")
    s_sorted = ls_ok[order]
    bounds = np.concatenate([[0], np.cumsum(counts)])
    for b in range(edges.size - 1):
        if counts[b] >= min_count:
            med[b] = np.median(s_sorted[bounds[b]:bounds[b + 1]])
    centers = 0.5 * (edges[:-1] + edges[1:])
    good = np.isfinite(med)
    return centers[good], med[good], counts[good].astype(float)


def scaling_break(Z, A, FD, dx, dy, A_roll: float, n_boot: int = N_BOOT, seed: int = 0, a_max: float = 1e7,
                  pixels: tuple[np.ndarray, np.ndarray] | None = None) -> dict[str, Any]:
    """Piecewise-linear log S - log A above the rollover with BIC model choice and a pixel bootstrap of the break."""
    if pixels is None:
        S = d8_slope(Z, FD, dx, dy)
        ok = np.isfinite(S) & np.isfinite(A) & (S > 0) & (A >= A_roll) & (A <= a_max)
        la, ls = np.log10(A[ok]), np.log10(S[ok])
    else:
        la, ls = pixels
        sel = la >= np.log10(A_roll)
        la, ls = la[sel], ls[sel]
    edges = np.arange(np.floor(np.log10(A_roll) * 10) / 10, np.log10(a_max) + 0.1, 0.1)
    x, y, w = _binned_medians(la, ls, edges)
    fits = piecewise_fits(x, y, w)
    fits.update({"A_rollover": A_roll, "n_pixels": int(la.size), "bin_log10A": x.tolist(), "bin_median_log10S": y.tolist(),
                 "bin_counts": w.tolist()})
    if "two" in fits:
        rng = np.random.default_rng(seed)
        brks = []
        for _ in range(n_boot):
            samp = rng.integers(0, la.size, la.size)
            xb, yb, wb = _binned_medians(la[samp], ls[samp], edges)
            f = piecewise_fits(xb, yb, wb)
            if "two" in f:
                brks.append(f["two"]["break_log10A"])
        brks = np.array(brks)
        fits["break_bootstrap"] = {"n": int(brks.size), "ci68_log10A": [float(np.percentile(brks, 16)), float(np.percentile(brks, 84))],
                                   "ci95_log10A": [float(np.percentile(brks, 2.5)), float(np.percentile(brks, 97.5))],
                                   "median_log10A": float(np.median(brks)), "unit": "pixel"}
        bl = fits["two"]["break_log10A"]
        at = (la >= bl - 0.05) & (la < bl + 0.05)
        as2 = (10 ** la[at]) * (10 ** ls[at]) ** 2
        fits["A_S2_at_break"] = {"n": int(as2.size), "q16": float(np.percentile(as2, 16)) if as2.size else None,
                                 "median": float(np.median(as2)) if as2.size else None,
                                 "q84": float(np.percentile(as2, 84)) if as2.size else None}
        fits["break_A_m2"] = float(10 ** bl)
    return fits


# ----------------------------------------------------------------------------- hilltop widths, facets, roughness

def hilltop_widths(Z, A, FD, dx, dy, A_c: float) -> dict[str, Any]:
    mask, diag = hilltop_mask(Z, A, FD, A_c, dx, dy)
    channel = np.isfinite(A) & (A >= A_c)
    dist = distance_transform_edt(~channel, sampling=(dy, dx))
    d = dist[mask]
    return {"A_c": A_c, "n_hilltop": int(mask.sum()), "distance_to_channel_m": {q: float(np.percentile(d, q)) for q in (10, 25, 50, 75, 90)},
            "hilltop_width_2x_m": {q: float(2 * np.percentile(d, q)) for q in (10, 50, 90)}, **diag}


def facet_size(Z, dx, dy, lap_tol: float = 2e-3) -> dict[str, Any]:
    """Size of planar patches: connected components of |laplacian(3x3)| < lap_tol (TIN facets on AP)."""
    C = total_curvature(Z, 1, dx, dy)
    flat = np.abs(C) < lap_tol
    lab, n = label(flat)
    sizes = np.bincount(lab.ravel())[1:]
    sizes = sizes[sizes >= 4]
    eq_len = np.sqrt(sizes * dx * dy)
    return {"lap_tol": lap_tol, "frac_flat_pixels": float(flat[np.isfinite(C)].mean()), "n_patches": int(sizes.size),
            "equiv_length_m": {q: float(np.percentile(eq_len, q)) for q in (50, 75, 90, 99)} if sizes.size else None,
            "mean_patch_area_m2": float(np.mean(sizes * dx * dy)) if sizes.size else None}


def roughness(Z, dx, dy, w: int = 2) -> dict[str, Any]:
    """Std of the residual from a (2w+1)^2 running mean, on all finite pixels and on the gentlest quartile."""
    zm = uniform_filter(np.nan_to_num(Z, nan=float(np.nanmean(Z))), size=2 * w + 1, mode="nearest")
    r = (Z - zm)[np.isfinite(Z)]
    return {"window_m": (2 * w + 1) * dx, "std_all": float(np.std(r)), "mad_all": float(np.median(np.abs(r - np.median(r))) * 1.4826)}


def synthetic_ridges(widths_m: Sequence[float], c_true: float, facet_len_m: float, noise_sigma: float, dx: float = 1.0,
                     half_widths=CURV_HALF_WIDTHS, L_star: dict[str, float] | None = None, seed: int = 0) -> dict[str, Any]:
    """Recovered crest curvature against window length for parabolic ridges; variants smooth, TIN, noise, TIN+noise."""
    from scipy.interpolate import LinearNDInterpolator
    from scipy.spatial import Delaunay

    rng = np.random.default_rng(seed)
    n = 401
    ys, xs = np.mgrid[0:n, 0:n] * dx
    xc = 0.5 * (n - 1) * dx
    out: dict[str, Any] = {"c_true": c_true, "facet_len_m": facet_len_m, "noise_sigma": noise_sigma, "dx": dx,
                           "window_lengths_m": [(2 * w + 1) * dx for w in half_widths], "cases": []}
    for W in widths_m:
        u = np.abs(xs - xc)
        half = W / 2
        z = np.where(u <= half, -0.5 * c_true * u ** 2, -0.5 * c_true * half ** 2 - c_true * half * (u - half))
        z = z + 200.0 + 0.02 * ys        # gentle axial tilt
        variants = {"smooth": z}
        # TIN: sample at random points of mean spacing facet_len_m, triangulate, linear interpolation
        if facet_len_m > 0:
            npts = int((n * dx / facet_len_m) ** 2)
            px = rng.uniform(0, (n - 1) * dx, npts)
            py = rng.uniform(0, (n - 1) * dx, npts)
            px = np.concatenate([px, [0, 0, (n - 1) * dx, (n - 1) * dx]])
            py = np.concatenate([py, [0, (n - 1) * dx, 0, (n - 1) * dx]])
            uu = np.abs(px - xc)
            pz = np.where(uu <= half, -0.5 * c_true * uu ** 2, -0.5 * c_true * half ** 2 - c_true * half * (uu - half)) + 200.0 + 0.02 * py
            tri = Delaunay(np.column_stack([px, py]))
            f = LinearNDInterpolator(tri, pz)
            variants["tin"] = f(np.column_stack([xs.ravel(), ys.ravel()])).reshape(n, n)
        if noise_sigma > 0:
            variants["noise"] = z + rng.normal(0, noise_sigma, z.shape)
            if "tin" in variants:
                variants["tin_noise"] = variants["tin"] + rng.normal(0, noise_sigma, z.shape)
        crest = (u <= dx * 0.51)
        case = {"width_m": W, "recovered": {}}
        for name, zz in variants.items():
            rec = []
            for w in half_widths:
                C = total_curvature(zz, w, dx, dx)
                sel = crest.copy()
                sel[:w + 1, :] = False
                sel[-(w + 1):, :] = False
                v = -C[sel]
                rec.append(float(np.nanmedian(v)))
            case["recovered"][name] = rec
            if L_star:
                case.setdefault("bias_at_L_star", {})[name] = {
                    site: float(np.interp(L, out["window_lengths_m"], rec) / c_true - 1.0) for site, L in L_star.items()}
        out["cases"].append(case)
    return out


# ----------------------------------------------------------------------------- per tile

def _metre_tile(tile: str, crop=None):
    cfg = load_tile_config(tile)
    if cfg.reproduce_v1_units:
        cfg = TileConfig(**{**cfg.__dict__, "reproduce_v1_units": False})
    return load_tile(cfg, crop=crop, sanity=crop is None)


def diagnose_tile(tile: str, thetas: np.ndarray, n_boot: int, stats_store: dict, pixel_store: dict,
                  latest_measure: dict | None, crop=None) -> dict[str, Any]:
    t0 = time.perf_counter()
    td = _metre_tile(tile, crop)
    Z, A, FD, dx, dy = td.Z, td.A, td.FD, td.dx, td.filled.dy
    out: dict[str, Any] = {"tile": tile, "site": td.site.site, "U": td.site.U, "inputs": td.provenance()}
    labels = edge_outlet_labels(A, FD)
    trunc = basin_truncation(A, labels)
    interior = set(trunc.loc[~trunc.truncated, "outlet_flat"].astype(int).tolist())
    out["truncation_basins"] = {"n_basins": int(len(trunc)), "n_truncated": int(trunc.truncated.sum()), "n_interior": len(interior),
                                "interior_outlets": sorted(interior)[:200]}

    # 6.2 fine ladder (free), storing per-basin stats for the site pooling
    stats_store[tile] = {}
    out["fine_ladder"] = ladder(Z, A, FD, dx, dy, labels, FINE_LADDER, thetas, n_boot=n_boot, store_stats=stats_store[tile],
                                tag=f"{tile} free")
    # 6.1 basin-fixed ladder
    keep, info = fixed_basin_set(Z, A, FD, dx, dy, labels)
    out["fixed_basins"] = info
    out["fixed_ladder"] = ladder(Z, A, FD, dx, dy, labels, FIXED_LADDER, thetas, keep=keep, n_boot=n_boot, tag=f"{tile} fixed")
    # 6.5 edge truncation per step and interior-only ladder
    frac = []
    for a_c in FINE_LADDER:
        ch = np.isfinite(A) & (A >= a_c)
        lab = labels[ch]
        tr = np.isin(lab, trunc.loc[trunc.truncated, "outlet_flat"].to_numpy())
        frac.append({"A_c": a_c, "n_channel_px": int(ch.sum()), "frac_in_truncated_basins": float(tr.mean()) if ch.any() else None,
                     "n_px_unlabelled": int(np.sum(lab < 0))})
    out["truncation_by_step"] = frac
    out["interior_ladder"] = ladder(Z, A, FD, dx, dy, labels, FINE_LADDER, thetas, keep=interior, n_boot=n_boot,
                                    tag=f"{tile} interior")
    # 6.3 scaling break
    roll = slope_area_rollover(Z, A, FD, dx, dy)
    A_roll = roll.A_rollover if np.isfinite(roll.A_rollover) else 100.0
    S = d8_slope(Z, FD, dx, dy)
    ok = np.isfinite(S) & np.isfinite(A) & (S > 0) & (A >= 10.0)
    pixel_store[tile] = (np.log10(A[ok]), np.log10(S[ok]))
    out["scaling_break"] = scaling_break(Z, A, FD, dx, dy, A_roll, n_boot=n_boot)
    out["scaling_break"]["rollover_note"] = roll.note
    # 6.4 common-theta ks at 1e4 (site break added at pooling time)
    net4 = channel_network(Z, A, FD, 1e4, dx, dy)
    out["ks_common_theta_1e4"] = {"theta_0.45": ks_at_theta_from_stats(net4, THETA_REF_COMMON, labels)}
    # 6.6 widths, facets, roughness
    out["hilltop_widths_1e4"] = hilltop_widths(Z, A, FD, dx, dy, 1e4)
    out["facets"] = facet_size(Z, dx, dy)
    out["roughness"] = roughness(Z, dx, dy)
    if latest_measure and tile in latest_measure:
        cur = latest_measure[tile].get("curvature", {})
        out["curvature_scale_curve"] = {k: cur.get(k) for k in ("window_lengths_m", "C_ht_median", "C_ht_q16", "C_ht_q84",
                                                                "L_star_m", "C_ht", "A_c")}
    out["wall_s"] = time.perf_counter() - t0
    logger.info("%s done in %.0f s", tile, out["wall_s"])
    return out


def pool_site(site: str, tiles: list[str], stats_store: dict, thetas: np.ndarray, n_boot: int) -> list[dict[str, Any]]:
    steps = []
    for a_c in FINE_LADDER:
        parts = [stats_store[t][a_c] for t in tiles if a_c in stats_store.get(t, {}) and stats_store[t][a_c][0].size]
        if not parts:
            steps.append({"A_c": a_c, "theta_star": None, "n_basins": 0, "flag_few_basins": True})
            continue
        sxx = np.vstack([p[0] for p in parts])
        sxy = np.vstack([p[1] for p in parts])
        syy = np.concatenate([p[2] for p in parts])
        fit = fit_from_stats(sxx, sxy, syy, thetas, n_boot=n_boot)
        fit["A_c"] = a_c
        fit["n_tiles"] = len(parts)
        steps.append(fit)
    return steps


def run_measure_diagnostics(tiles: Sequence[str], n_boot: int = N_BOOT, run_id: str | None = None, results_root=None,
                            thetas=DEFAULT_THETAS, crop=None) -> RunRecorder:
    rec = RunRecorder("measure_diagnostics", "_".join(tiles) if len(tiles) <= 3 else f"{len(tiles)}tiles", run_id=run_id,
                      root=results_root)
    thetas = np.asarray(thetas, dtype=float)
    # latest per-tile measure results (for the C_ht(L) curves and L*)
    latest_measure: dict[str, Any] = {}
    for r in find_runs("measure"):
        try:
            import json

            p = json.loads((r / "params.json").read_text())
            for t, v in p.get("tiles", {}).items():
                latest_measure[t] = v
        except Exception:  # noqa: BLE001
            continue
    stats_store: dict = {}
    pixel_store: dict = {}
    results = {}
    snaps = {}
    for tile in tiles:
        results[tile] = diagnose_tile(tile, thetas, n_boot, stats_store, pixel_store, latest_measure, crop=crop)
        snaps[tile] = config_snapshot(load_tile_config(tile))
    sites = sorted({t[:2] for t in tiles})
    pooled: dict[str, Any] = {}
    for site in sites:
        st_tiles = [t for t in tiles if t.startswith(site)]
        pooled[site] = {"tiles": st_tiles, "ladder": pool_site(site, st_tiles, stats_store, thetas, n_boot)}
        # pooled scaling break from all pixels of the site
        la = np.concatenate([pixel_store[t][0] for t in st_tiles])
        ls = np.concatenate([pixel_store[t][1] for t in st_tiles])
        rolls = [results[t]["scaling_break"]["A_rollover"] for t in st_tiles]
        pooled[site]["scaling_break"] = scaling_break(None, None, None, None, None, float(np.median(rolls)), n_boot=n_boot,
                                                      pixels=(la, ls))
        brk = pooled[site]["scaling_break"].get("break_A_m2")
        pooled[site]["theta_at_break"] = None
        if brk:
            lad = pooled[site]["ladder"]
            ths = [(s["A_c"], s["theta_star"]) for s in lad if s.get("theta_star") is not None]
            if ths:
                a_near = min(ths, key=lambda p: abs(math.log10(p[0]) - math.log10(brk)))
                pooled[site]["theta_at_break"] = {"A_c_step": a_near[0], "theta": a_near[1]}
        # 6.4 common-theta k_sn per tile at the site break and at 1e4
        for t in st_tiles:
            td = _metre_tile(t, crop)
            labels = edge_outlet_labels(td.A, td.FD)
            ks_c: dict[str, Any] = {}
            for a_c_name, a_c in (("site_break", brk), ("1e4", 1e4)):
                if not a_c:
                    continue
                net = channel_network(td.Z, td.A, td.FD, float(a_c), td.dx, td.filled.dy)
                entry = {"A_c": float(a_c), "theta_0.45": ks_at_theta_from_stats(net, THETA_REF_COMMON, labels)}
                if pooled[site]["theta_at_break"]:
                    th = pooled[site]["theta_at_break"]["theta"]
                    entry[f"theta_site_{th:.2f}"] = ks_at_theta_from_stats(net, th, labels)
                ks_c[a_c_name] = entry
            results[t]["ks_common_theta"] = ks_c
    rec.add_params("tiles", results)
    rec.add_params("pooled", pooled)
    rec.add_params("settings", {"fine_ladder": list(FINE_LADDER), "fixed_ladder": list(FIXED_LADDER), "A_ref_fixed": A_REF_FIXED,
                                "N_min_fixed_basins": N_MIN_FIXED_BASINS, "min_relief_m": MIN_RELIEF_M,
                                "min_chi_range_m": MIN_CHI_RANGE_M, "theta_ref_chi": THETA_REF_CHI,
                                "min_pixels_per_basin": MIN_PIXELS_PER_BASIN, "min_basins_flag": MIN_BASINS_FLAG,
                                "n_boot": n_boot, "bootstrap_unit_chi": "basin", "bootstrap_unit_break": "pixel",
                                "theta_ref_common": THETA_REF_COMMON, "thetas": thetas.tolist()})
    # 6.6 synthetic ridges with site-derived widths, facets and roughness
    syn = {}
    for site in sites:
        st_tiles = [t for t in tiles if t.startswith(site)]
        widths = sorted({round(results[t]["hilltop_widths_1e4"]["hilltop_width_2x_m"][q]) for t in st_tiles for q in (10, 50, 90)})
        ap_tiles = [t for t in tiles if t.startswith("AP")]
        facet = float(np.median([results[t]["facets"]["equiv_length_m"][50] for t in ap_tiles])) if ap_tiles else 0.0
        gm_tiles = [t for t in tiles if t.startswith("GM")]
        sigma = float(np.median([results[t]["roughness"]["mad_all"] for t in gm_tiles])) if gm_tiles else 0.05
        c_true = float(np.median([results[t].get("curvature_scale_curve", {}).get("C_ht") or 0.01 for t in st_tiles]))
        L_star = {t: results[t].get("curvature_scale_curve", {}).get("L_star_m") for t in st_tiles}
        L_star = {k: v for k, v in L_star.items() if v}
        syn[site] = synthetic_ridges(widths, c_true, facet if site == "AP" else 0.0, sigma, L_star=L_star)
        syn[site]["widths_source"] = "2 x hilltop-to-channel distance at A_c = 1e4 (10/50/90 % over the site's tiles)"
    rec.add_params("synthetic_ridges", syn)
    # long tables for plotting
    rows = []
    for t, r in results.items():
        for kind in ("fine_ladder", "fixed_ladder", "interior_ladder"):
            for s in r[kind]:
                rows.append({"tile": t, "site": r["site"], "ladder": kind, **{k: v for k, v in s.items() if not isinstance(v, list | dict)},
                             "theta_lo68": (s.get("theta_ci68") or [None, None])[0], "theta_hi68": (s.get("theta_ci68") or [None, None])[1],
                             "theta_lo95": (s.get("theta_ci95") or [None, None])[0], "theta_hi95": (s.get("theta_ci95") or [None, None])[1],
                             "ks_lo68": (s.get("ks_ci68") or [None, None])[0], "ks_hi68": (s.get("ks_ci68") or [None, None])[1]})
    for site, p in pooled.items():
        for s in p["ladder"]:
            rows.append({"tile": f"{site}_pooled", "site": site, "ladder": "pooled", **{k: v for k, v in s.items() if not isinstance(v, list | dict)},
                         "theta_lo68": (s.get("theta_ci68") or [None, None])[0], "theta_hi68": (s.get("theta_ci68") or [None, None])[1],
                         "theta_lo95": (s.get("theta_ci95") or [None, None])[0], "theta_hi95": (s.get("theta_ci95") or [None, None])[1]})
    rec.write_table(pd.DataFrame(rows), "ladders")
    rec.finalize({"tiles": snaps, "crop": crop})
    return rec
