"""Addendum 8, section 2: is the D gap the signature of nonuniform hillslope curvature? (E*-R*, pre-registered)

Every quantity comes from the observed DEM in metres; none from a model (addendum 8, 2).

2.1 Definitions (also in docs/measurement_methods.md)
    C_ht   -laplacian of the quadratic surface at L_f = 11 m at each option (iii) hilltop pixel (the configs' estimator).
    L_H    hillslope length: the D8 steepest-descent path from the hilltop pixel to the first cell of the channel
           network, the network being A >= A_c (primary 1e3 m^2; sensitivity 1e4 m^2 and the rollover network).
    R      elevation drop along that path; mean slope Sbar = R / L_H.
    E* = 2 C_ht L_H / S_c,  R* = Sbar / S_c.
    Steady nonlinear curve (Roering et al. 2007, as in Hurst et al. 2012):
        R* = (1/E*) [sqrt(1 + E*^2) - ln(0.5 (1 + sqrt(1 + E*^2))) - 1].
    The curve is verified against a numerical integration of the 1-D steady nonlinear profile before use
    (:func:`verify_curve`).

    Linear limit, corrected. For small E* the curve gives R* = E*/4, not E*/2 as addendum 8, 2.1 states: the steady
    linear-diffusion hillslope is the parabola z = -(U/2D) x^2, whose mean slope R/L_H = C_ht L_H / 2 is half of the
    toe slope C_ht L_H. The ratio Sbar / (C_ht L_H) is therefore 0.5 on a linear hillslope, and the nonlinear
    signature is a value *below 0.5*. The addendum's thresholds (median below 0.67, 68 % interval below 1) are read
    relative to the linear value, i.e. on the normalised ratio 2 Sbar / (C_ht L_H) (= 1 on a linear hillslope); the
    literal ratio is reported beside it. A perfectly linear hillslope passes the literal reading, so it cannot be the
    intended one; the report states this.

2.2 Per tile and pooled per site: (1) the median of the linear ratio with a segment bootstrap (segments = 8-connected
    groups of hilltop pixels, as in the option (iii) window test); (2) S_c by least squares of log10 R* against the
    curve over hilltop pixels, with the segment bootstrap and the 68 % interval; (3) the E*-R* figure (lemcvt figures).

2.3 Rule, per site: supported if on >= 4 of 5 tiles (a) the median normalised ratio is below 0.67 with its 68 %
    interval below 1, and (b) the fitted S_c lies inside its own bootstrap 68 % interval and is below 1.27. Applied
    on all three networks; the primary network decides.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy.ndimage import label
from scipy.optimize import minimize_scalar

from lemcvt.analyses.d_diagnostic import curvature_network_threshold
from lemcvt.config import apply_d0_variant, config_snapshot, load_tile
from lemcvt.hydro.d8 import d8_receiver_arrays
from lemcvt.measure.hilltop_curvature import _weighted_median, total_curvature
from lemcvt.measure.phase6_inputs import W_F, option_iii_mask
from lemcvt.results import RunRecorder

logger = logging.getLogger(__name__)

NETWORKS = ("primary_1e3", "sensitivity_1e4", "rollover")
NETWORK_AREAS = {"primary_1e3": 1e3, "sensitivity_1e4": 1e4}
S_C_REFERENCE = 1.27
S_C_BOUNDS = (0.05, 20.0)
RULE = {"ratio_median_below": 0.67, "ratio_ci68_below": 1.0, "S_c_below": S_C_REFERENCE, "min_tiles": 4}
N_BOOT = 200
LINEAR_RATIO = 0.5          # Sbar / (C_ht L_H) on the steady linear-diffusion (parabolic) hillslope


# ----------------------------------------------------------------------------- the curve and its verification

def steady_nonlinear_R(E: np.ndarray | float) -> np.ndarray:
    """R*(E*) of Roering et al. (2007); R* = E*/4 + O(E*^3) for small E*, 1 as E* -> infinity."""
    E = np.asarray(E, dtype=float)
    out = np.empty_like(E)
    small = E < 1e-4
    s = np.sqrt(1.0 + E[~small] ** 2)
    out[~small] = (s - np.log(0.5 * (1.0 + s)) - 1.0) / E[~small]
    out[small] = E[small] / 4.0
    return out


def integrate_profile_R(E: float, n: int = 20001) -> float:
    """R* from the 1-D steady nonlinear hillslope: q = U x, -D z' / (1 - (z'/S_c)^2) = U x, so with u = E* x / L_H
    the slope is S_c (sqrt(1 + u^2) - 1) / u and R* = (1/E*) int_0^{E*} (sqrt(1 + u^2) - 1) / u du (Simpson)."""
    u = np.linspace(0.0, E, n)
    f = np.empty_like(u)
    f[0] = 0.0
    f[1:] = (np.sqrt(1.0 + u[1:] ** 2) - 1.0) / u[1:]
    h = u[1] - u[0]
    simpson = (h / 3.0) * (f[0] + f[-1] + 4.0 * f[1:-1:2].sum() + 2.0 * f[2:-1:2].sum())
    return float(simpson / E)


def verify_curve(E_values=(0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0), tol: float = 1e-6) -> dict[str, Any]:
    """Compare the published form with the integration at several E*; the linear limit R*/E* -> 1/4."""
    rows = []
    worst = 0.0
    for E in E_values:
        a, b = float(steady_nonlinear_R(E)), integrate_profile_R(E)
        d = abs(a - b) / b
        worst = max(worst, d)
        rows.append({"E_star": E, "R_published": a, "R_integrated": b, "rel_diff": d})
    lin = float(steady_nonlinear_R(1e-3) / 1e-3)
    return {"rows": rows, "max_rel_diff": worst, "pass": bool(worst <= tol), "tol": tol,
            "linear_limit_R_over_E": lin, "linear_limit_expected": 0.25,
            "note": "addendum 8, 2.1 states the linear limit as R* = E*/2; the curve (and the parabolic profile) give E*/4"}


# ----------------------------------------------------------------------------- hillslope length and relief

def trace_to_channel(Z: np.ndarray, A: np.ndarray, FD: np.ndarray, dx: float, dy: float, rows: np.ndarray, cols: np.ndarray,
                     A_c: float) -> dict[str, np.ndarray]:
    """D8 steepest-descent path from each (row, col) to the first cell with A >= A_c: path length L_H, drop R, the
    channel cell reached, and whether the path left the tile first."""
    r_to, c_to, good = d8_receiver_arrays(FD)
    n = rows.size
    cur_r, cur_c = rows.astype(np.int64).copy(), cols.astype(np.int64).copy()
    L = np.zeros(n)
    reached = np.zeros(n, dtype=bool)
    left = np.zeros(n, dtype=bool)
    active = np.ones(n, dtype=bool)
    z0 = Z[rows, cols]
    diag_len = math.hypot(dx, dy)
    for _ in range(Z.shape[0] * Z.shape[1]):
        if not active.any():
            break
        ok = active & good[cur_r, cur_c]
        left |= active & ~ok
        active &= ok
        nr, nc = r_to[cur_r, cur_c], c_to[cur_r, cur_c]
        dr, dc = nr - cur_r, nc - cur_c
        step = np.where((dr != 0) & (dc != 0), diag_len, np.where(dr != 0, dy, dx))
        L[ok] += step[ok]
        cur_r[ok], cur_c[ok] = nr[ok], nc[ok]
        hit = ok & (np.nan_to_num(A[cur_r, cur_c], nan=-1.0) >= A_c)
        reached |= hit
        active &= ~hit
    R = z0 - Z[cur_r, cur_c]
    return {"L_H": L, "R": R, "reached": reached, "left_tile": left, "end_row": cur_r, "end_col": cur_c}


# ----------------------------------------------------------------------------- statistics

def _segment_bootstrap_median(vals: np.ndarray, seg: np.ndarray, n_boot: int, seed: int = 0) -> dict[str, Any]:
    ok = np.isfinite(vals)
    vals, seg = vals[ok], seg[ok]
    if vals.size == 0:
        return {"median": float("nan"), "n": 0}
    rec: dict[str, Any] = {"median": float(np.median(vals)), "n": int(vals.size), "n_segments": int(np.unique(seg).size)}
    uniq = np.unique(seg)
    if n_boot > 0 and uniq.size >= 5:
        rng = np.random.default_rng(seed)
        bs = np.empty(n_boot)
        for i in range(n_boot):
            pick = rng.choice(uniq, uniq.size, replace=True)
            counts = np.bincount(pick, minlength=int(seg.max()) + 1)
            bs[i] = _weighted_median(vals, counts[seg])
        rec["ci68"] = [float(np.percentile(bs, 16)), float(np.percentile(bs, 84))]
        rec["ci95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    return rec


def _sc_misfit(log_sc: float, C: np.ndarray, L: np.ndarray, S: np.ndarray, w: np.ndarray) -> float:
    sc = 10.0 ** log_sc
    E = 2.0 * C * L / sc
    model = steady_nonlinear_R(E)
    obs = S / sc
    r = np.log10(obs) - np.log10(model)
    return float(np.sum(w * r * r) / np.sum(w))


def fit_S_c(C: np.ndarray, L: np.ndarray, S: np.ndarray, w: np.ndarray | None = None) -> dict[str, Any]:
    """S_c by least squares of log10 R* against the curve (the only free parameter; E* and R* both scale with S_c)."""
    w = np.ones_like(C) if w is None else w
    lo, hi = math.log10(S_C_BOUNDS[0]), math.log10(S_C_BOUNDS[1])
    res = minimize_scalar(_sc_misfit, bounds=(lo, hi), args=(C, L, S, w), method="bounded", options={"xatol": 1e-6})
    sc = float(10.0 ** res.x)
    return {"S_c": sc, "misfit_log10_rms": float(math.sqrt(res.fun)), "at_bound": bool(abs(res.x - lo) < 1e-4 or abs(res.x - hi) < 1e-4),
            "nfev": int(res.nfev)}


def _fit_with_bootstrap(C: np.ndarray, L: np.ndarray, S: np.ndarray, seg: np.ndarray, n_boot: int, seed: int = 0) -> dict[str, Any]:
    ok = np.isfinite(C) & np.isfinite(L) & np.isfinite(S) & (C > 0) & (L > 0) & (S > 0)
    C, L, S, seg = C[ok], L[ok], S[ok], seg[ok]
    if C.size < 5:
        return {"S_c": float("nan"), "n": int(C.size)}
    fit = fit_S_c(C, L, S)
    fit["n"] = int(C.size)
    fit["n_segments"] = int(np.unique(seg).size)
    uniq = np.unique(seg)
    if n_boot > 0 and uniq.size >= 5:
        rng = np.random.default_rng(seed)
        bs = np.empty(n_boot)
        for i in range(n_boot):
            pick = rng.choice(uniq, uniq.size, replace=True)
            counts = np.bincount(pick, minlength=int(seg.max()) + 1)
            w = counts[seg].astype(float)
            bs[i] = fit_S_c(C, L, S, w)["S_c"]
        fit["S_c_ci68"] = [float(np.percentile(bs, 16)), float(np.percentile(bs, 84))]
        fit["S_c_ci95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        fit["S_c_bootstrap_range"] = [float(np.min(bs)), float(np.max(bs))]
        fit["S_c_bootstrap_median"] = float(np.median(bs))
    return fit


def _verdict(ratio_norm: dict[str, Any], ratio_lit: dict[str, Any], sc: dict[str, Any]) -> dict[str, Any]:
    ci = ratio_norm.get("ci68")
    a = bool(np.isfinite(ratio_norm["median"]) and ratio_norm["median"] < RULE["ratio_median_below"] and ci is not None and ci[1] < RULE["ratio_ci68_below"])
    ci_l = ratio_lit.get("ci68")
    a_lit = bool(np.isfinite(ratio_lit["median"]) and ratio_lit["median"] < RULE["ratio_median_below"] and ci_l is not None and ci_l[1] < RULE["ratio_ci68_below"])
    sci = sc.get("S_c_ci68")
    inside = bool(sci is not None and np.isfinite(sc.get("S_c", float("nan"))) and sci[0] <= sc["S_c"] <= sci[1])
    rng = sc.get("S_c_bootstrap_range")
    inside_range = bool(rng is not None and np.isfinite(sc.get("S_c", float("nan"))) and rng[0] <= sc["S_c"] <= rng[1])
    b = bool(inside and sc.get("S_c", float("inf")) < RULE["S_c_below"])
    return {"a_ratio_pass_normalised": a, "a_ratio_pass_literal": a_lit, "b_S_c_pass": b, "S_c_inside_ci68": inside,
            "S_c_inside_bootstrap_range": inside_range, "S_c_below_reference": bool(sc.get("S_c", float("inf")) < RULE["S_c_below"]),
            "pass": bool(a and b), "pass_literal_ratio": bool(a_lit and b)}


def _site_rule(per_tile: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for site in ("AP", "GM"):
        tiles = [t for t in per_tile if t.startswith(site)]
        if not tiles:
            continue
        v = [per_tile[t]["verdict"] for t in tiles]
        out[site] = {"n_tiles": len(tiles), "n_pass": sum(x["pass"] for x in v), "n_a_pass": sum(x["a_ratio_pass_normalised"] for x in v),
                     "n_b_pass": sum(x["b_S_c_pass"] for x in v), "n_pass_literal": sum(x["pass_literal_ratio"] for x in v),
                     "supported": bool(sum(x["pass"] for x in v) >= RULE["min_tiles"]),
                     "supported_literal_ratio": bool(sum(x["pass_literal_ratio"] for x in v) >= RULE["min_tiles"])}
    return out


# ----------------------------------------------------------------------------- per tile

def er_star_tile(tile: str, n_boot: int = N_BOOT, A_c_rollover: float | None = None, crop=None, sanity: bool | None = None) -> tuple[dict[str, Any], pd.DataFrame]:
    """E*-R* quantities of every option (iii) hilltop pixel of one tile, on the three networks."""
    td = load_tile(tile, crop=crop, sanity=(crop is None) if sanity is None else sanity)
    mp = apply_d0_variant(td.cfg.model, "A")
    Z, A, FD, dx, dy = td.Z, td.A, td.FD, td.dx, td.filled.dy
    src = ""
    if A_c_rollover is None:
        A_c_rollover, src = curvature_network_threshold(tile)
    mask, _m0, dist, diag = option_iii_mask(Z, A, FD, dx, dy, float(A_c_rollover), W_F)
    seg_grid, n_seg = label(mask, structure=np.ones((3, 3)))
    rows, cols = np.nonzero(mask)
    seg = seg_grid[rows, cols]
    C = -total_curvature(Z, W_F, dx, dy)[rows, cols]
    L_f = (2 * W_F + 1) * dx
    info: dict[str, Any] = {"tile": tile, "A_c_rollover": float(A_c_rollover), "A_c_source": src, "L_f_m": float(L_f), "n_hilltop": int(rows.size),
                            "n_segments": int(n_seg), "option_iii": diag, "n_C_ht_nonpositive": int(np.sum(~(C > 0))),
                            "C_ht_median": float(np.median(C)), "C_ht_config": mp.D0 and float(mp.U / mp.D0), "U": mp.U, "D0_A": mp.D0, "D0_B": mp.D0_B,
                            "networks": {}}
    frames = []
    areas = dict(NETWORK_AREAS) | {"rollover": float(A_c_rollover)}
    for net, A_c in areas.items():
        tr = trace_to_channel(Z, A, FD, dx, dy, rows, cols, A_c)
        okp = tr["reached"] & (tr["L_H"] > 0) & (C > 0) & (tr["R"] > 0)
        L_H, R = tr["L_H"], tr["R"]
        S = np.where(L_H > 0, R / np.where(L_H > 0, L_H, 1.0), np.nan)
        ratio_lit = np.where(okp, S / (C * L_H), np.nan)                  # 0.5 on a linear hillslope
        ratio_norm = ratio_lit / LINEAR_RATIO                             # 1 on a linear hillslope
        stat_lit = _segment_bootstrap_median(ratio_lit, seg, n_boot)
        stat_norm = _segment_bootstrap_median(ratio_norm, seg, n_boot)
        sc = _fit_with_bootstrap(np.where(okp, C, np.nan), L_H, S, seg, n_boot)
        E = 2.0 * C * L_H / sc["S_c"] if np.isfinite(sc.get("S_c", float("nan"))) else np.full(rows.size, np.nan)
        rec = {"A_c": float(A_c), "n_pixels": int(rows.size), "n_reached": int(tr["reached"].sum()), "n_left_tile": int(tr["left_tile"].sum()),
               "n_used": int(okp.sum()), "n_R_nonpositive": int(np.sum(tr["reached"] & ~(tr["R"] > 0))),
               "L_H_median_m": float(np.median(L_H[okp])) if okp.any() else float("nan"),
               "L_H_q16_q84_m": [float(np.percentile(L_H[okp], 16)), float(np.percentile(L_H[okp], 84))] if okp.any() else None,
               "R_median_m": float(np.median(R[okp])) if okp.any() else float("nan"),
               "Sbar_median": float(np.median(S[okp])) if okp.any() else float("nan"),
               "ratio_literal": stat_lit, "ratio_normalised": stat_norm, "S_c_fit": sc,
               "E_star_median_at_fit": float(np.nanmedian(E[okp])) if okp.any() else float("nan"),
               "E_star_median_at_1.27": float(np.median(2.0 * C[okp] * L_H[okp] / S_C_REFERENCE)) if okp.any() else float("nan"),
               "R_star_median_at_fit": float(np.nanmedian(S[okp] / sc["S_c"])) if okp.any() and np.isfinite(sc.get("S_c", float("nan"))) else float("nan")}
        rec["verdict"] = _verdict(stat_norm, stat_lit, sc)
        info["networks"][net] = rec
        frames.append(pd.DataFrame({"tile": tile, "network": net, "A_c": float(A_c), "row": rows, "col": cols, "segment": seg, "C_ht": C,
                                    "L_H": L_H, "R": R, "Sbar": S, "reached": tr["reached"], "left_tile": tr["left_tile"], "used": okp,
                                    "ratio_literal": ratio_lit, "ratio_normalised": ratio_norm, "E_star_at_fit": E, "R_star_at_fit": S / sc["S_c"] if np.isfinite(sc.get("S_c", float("nan"))) else np.nan}))
        logger.info("%s %s (A_c %.4g): %d of %d pixels reach a channel (%d leave the tile); L_H median %.1f m; ratio 2Sbar/(C L) median %.3f %s; S_c %.3f %s%s",
                    tile, net, A_c, rec["n_reached"], rec["n_pixels"], rec["n_left_tile"], rec["L_H_median_m"], stat_norm["median"], stat_norm.get("ci68"),
                    sc.get("S_c", float("nan")), sc.get("S_c_ci68"), " (at bound)" if sc.get("at_bound") else "")
    return info, pd.concat(frames, ignore_index=True)


def run_er_star(tiles, n_boot: int = N_BOOT, run_id: str | None = None, results_root=None, crop=None, A_c_rollover: float | None = None) -> RunRecorder:
    rec = RunRecorder("er_star", "_".join(tiles) if len(tiles) <= 3 else f"{len(tiles)}tiles", run_id=run_id, root=results_root)
    t0 = time.perf_counter()
    rec.add_params("rule", RULE)
    rec.add_params("linear_ratio", LINEAR_RATIO)
    rec.add_params("curve_verification", verify_curve())
    per_tile: dict[str, Any] = {}
    frames = []
    snaps = {}
    for tile in tiles:
        info, df = er_star_tile(tile, n_boot=n_boot, A_c_rollover=A_c_rollover, crop=crop)
        per_tile[tile] = info
        frames.append(df)
        snaps[tile] = config_snapshot(load_tile(tile, crop=crop, sanity=False).cfg)
    df_all = pd.concat(frames, ignore_index=True)
    rec.write_table(df_all, "hilltop_pixels")
    # pooled per site, per network
    pooled: dict[str, Any] = {}
    for site in ("AP", "GM"):
        sub = df_all[df_all.tile.str.startswith(site)]
        if sub.empty:
            continue
        pooled[site] = {}
        for net in NETWORKS:
            s = sub[(sub.network == net)]
            seg = (s.segment.to_numpy() + 100000 * s.tile.str[2:].astype(int).to_numpy()).astype(np.int64)
            seg = np.unique(seg, return_inverse=True)[1]
            used = s.used.to_numpy()
            rl = _segment_bootstrap_median(np.where(used, s.ratio_literal.to_numpy(), np.nan), seg, n_boot)
            rn = _segment_bootstrap_median(np.where(used, s.ratio_normalised.to_numpy(), np.nan), seg, n_boot)
            sc = _fit_with_bootstrap(np.where(used, s.C_ht.to_numpy(), np.nan), s.L_H.to_numpy(), s.Sbar.to_numpy(), seg, n_boot)
            pooled[site][net] = {"n_used": int(used.sum()), "ratio_literal": rl, "ratio_normalised": rn, "S_c_fit": sc, "verdict": _verdict(rn, rl, sc)}
    rec.add_params("tiles", per_tile)
    rec.add_params("pooled", pooled)
    site_rule = {net: _site_rule({t: {"verdict": per_tile[t]["networks"][net]["verdict"]} for t in per_tile}) for net in NETWORKS}
    rec.add_params("site_rule", site_rule)
    rec.add_params("decides", "primary_1e3")
    rec.add_params("wall_s", time.perf_counter() - t0)
    for net in NETWORKS:
        logger.info("site rule on %s: %s", net, site_rule[net])
    rec.finalize({"tiles": snaps, "crop": crop, "n_boot": n_boot})
    return rec
