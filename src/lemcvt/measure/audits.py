"""Audits of the chi estimator and the hilltop set (addendum 3, section 4).

``lemcvt audit measure --tiles all`` reruns the free ladder, the basin-fixed ladder and the
site pooling of ``lemcvt measure-diagnostics`` with three estimators of theta side by side,
reports the effective number of basins behind every pooled value, compares truncated with
complete basins once (A_c = 1e3 m^2), and measures hilltop curvature with and without the
near-channel exclusion of curvature option (i).

Estimators (all on the same per-basin statistics, after the pre-registered filters of
``lemcvt.measure.diagnostics``):

* ``pooled``: the 02c estimator, theta* = argmin of the pooled through-origin SSR
  sum_i SSR_i(theta). Each basin's leverage is its chi spread, so the largest basins dominate.
* ``equal``: theta* = argmin of sum_i SSR_i(theta) / SSR_i,null, where SSR_i,null is basin
  i's sum of squares about its own mean elevation; every basin counts equally.
* ``median``: the median over basins of each basin's own best theta.

n_eff = 1 / sum_i w_i^2 with w_i basin i's share of the total weight; two weights are given:
the centred chi spread sum (chi - chi_bar)^2 the addendum names, and the through-origin
leverage sum chi^2 that the pooled estimator actually uses. Both are evaluated at the pooled
theta*. Where n_eff < 3 the basin bootstrap is not a usable uncertainty and is flagged.

Hilltop exclusion: hilltop pixels closer than L*/2 to a channel pixel of the network at
A_c (1e3 and 1e4 m^2) are dropped; L* is the tile's window length chosen by the measure run.
C_ht is reported at that fixed L* and at the plateau scale re-selected on each set, with a
segment bootstrap, and D0 = U / C_ht.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, label

from lemcvt.config import config_snapshot, load_tile_config
from lemcvt.measure.chi import ChannelNetwork, channel_network, chi_for_thetas, relative_elevation
from lemcvt.measure.concavity import DEFAULT_THETAS
from lemcvt.measure.diagnostics import (
    FINE_LADDER,
    FIXED_LADDER,
    MIN_BASINS_FLAG,
    MIN_CHI_RANGE_M,
    MIN_PIXELS_PER_BASIN,
    MIN_RELIEF_M,
    N_BOOT,
    THETA_REF_CHI,
    _metre_tile,
    _restrict_area,
    basin_truncation,
    edge_outlet_labels,
    fixed_basin_set,
    synthetic_ridges,
)
from lemcvt.measure.hilltop_curvature import DEFAULT_HALF_WIDTHS, _weighted_median, hilltop_mask, select_scale, total_curvature
from lemcvt.results import RunRecorder, find_runs

logger = logging.getLogger(__name__)

N_EFF_FLAG = 3.0
HILLTOP_NETWORKS = (1e3, 1e4)
T8_STEPS = {"AP": (1000.0, 10000.0, 1995.26), "GM": (1000.0, 10000.0, 398.107)}


# ----------------------------------------------------------------------------- per-basin moments

def basin_moments(net: ChannelNetwork, thetas: np.ndarray, labels: np.ndarray, truncated_outlets: set[int] | None = None,
                  min_pixels: int = MIN_PIXELS_PER_BASIN, min_relief: float = MIN_RELIEF_M,
                  min_chi_range: float = MIN_CHI_RANGE_M, theta_ref: float = THETA_REF_CHI) -> dict[str, Any]:
    """Per kept basin: n, sum chi, sum chi^2, sum chi z, sum z, sum z^2 (chi per theta), and the truncation flag."""
    chi = chi_for_thetas(net, thetas)
    chi_ref = chi_for_thetas(net, np.array([theta_ref]))[:, 0]
    zrel = relative_elevation(net)
    n_l, sx_l, sxx_l, sxy_l, sy_l, syy_l, tr_l, out_l = [], [], [], [], [], [], [], []
    n_total = 0
    for o in net.outlet_index:
        idx = np.nonzero(net.basin == o)[0]
        idx = idx[idx != o]
        n = idx.size
        n_total += 1
        relief = float(zrel[idx].max()) if n else 0.0
        chir = float(chi_ref[idx].max()) if n else 0.0
        if not (n >= min_pixels and relief >= min_relief and chir >= min_chi_range):
            continue
        x, y = chi[idx], zrel[idx]
        n_l.append(n)
        sx_l.append(x.sum(axis=0))
        sxx_l.append((x * x).sum(axis=0))
        sxy_l.append((x * y[:, None]).sum(axis=0))
        sy_l.append(float(y.sum()))
        syy_l.append(float((y * y).sum()))
        lab = int(labels[net.rows[o], net.cols[o]])
        out_l.append(lab)
        tr_l.append(bool(truncated_outlets is not None and lab in truncated_outlets))
    k = thetas.size
    return {"n": np.array(n_l, dtype=float), "sx": np.array(sx_l).reshape(-1, k), "sxx": np.array(sxx_l).reshape(-1, k),
            "sxy": np.array(sxy_l).reshape(-1, k), "sy": np.array(sy_l), "syy": np.array(syy_l),
            "truncated": np.array(tr_l, dtype=bool), "outlet_flat": np.array(out_l, dtype=np.int64),
            "n_basins_total": n_total, "n_px_kept": int(sum(n_l))}


def concat_moments(parts: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ("n", "sx", "sxx", "sxy", "sy", "syy", "truncated", "outlet_flat")
    out = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    out["n_basins_total"] = sum(p["n_basins_total"] for p in parts)
    out["n_px_kept"] = sum(p["n_px_kept"] for p in parts)
    return out


def subset_moments(mom: dict[str, Any], sel: np.ndarray) -> dict[str, Any]:
    out = {k: (v[sel] if isinstance(v, np.ndarray) and v.shape[:1] == sel.shape else v) for k, v in mom.items()}
    out["n_px_kept"] = int(mom["n"][sel].sum())
    return out


def _n_eff(w: np.ndarray) -> float:
    w = np.asarray(w, dtype=float)
    w = w[np.isfinite(w) & (w > 0)]
    if w.size == 0:
        return float("nan")
    w = w / w.sum()
    return float(1.0 / np.sum(w * w))


def _ci(v: np.ndarray) -> dict[str, list[float]]:
    return {"ci68": [float(np.percentile(v, 16)), float(np.percentile(v, 84))],
            "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]}


def estimators(mom: dict[str, Any], thetas: np.ndarray, n_boot: int = N_BOOT, seed: int = 0) -> dict[str, Any]:
    """The pooled, equal-weight and median estimators with n_eff and basin-bootstrap intervals."""
    nb = int(mom["n"].size)
    out: dict[str, Any] = {"n_basins": nb, "n_px": mom.get("n_px_kept"), "n_basins_total": mom.get("n_basins_total"),
                           "flag_few_basins": bool(nb < MIN_BASINS_FLAG)}
    if nb == 0:
        out.update({"pooled": None, "equal": None, "median": None})
        return out
    sxx, sxy, syy, sx, sy, n = mom["sxx"], mom["sxy"], mom["syy"], mom["sx"], mom["sy"], mom["n"]
    with np.errstate(invalid="ignore", divide="ignore"):
        ssr_b = syy[:, None] - sxy * sxy / sxx                       # (nb, k) residual SS per basin per theta
        ks_b = sxy / sxx
    ssr_null = syy - sy * sy / n                                        # about the basin's own mean z
    ssr_null = np.where(ssr_null > 0, ssr_null, np.nan)
    norm = ssr_b / ssr_null[:, None]                                    # (nb, k)

    # pooled through-origin (02c)
    ssr = np.nansum(ssr_b, axis=0)
    j = int(np.nanargmin(ssr))
    ks_pooled = float(sxy[:, j].sum() / sxx[:, j].sum())
    ss_tot = float(syy.sum())
    w_c = sxx[:, j] - sx[:, j] ** 2 / n                                 # centred chi spread, the addendum's weight
    w_u = sxx[:, j]                                                     # through-origin leverage
    pooled = {"theta_star": float(thetas[j]), "ks": ks_pooled, "r2": float(1 - ssr[j] / ss_tot) if ss_tot > 0 else None,
              "theta_at_edge": bool(j in (0, thetas.size - 1)),
              "n_eff_centred": _n_eff(w_c), "n_eff_through_origin": _n_eff(w_u),
              "share_largest_basin_centred": float(np.max(w_c) / np.sum(w_c)) if np.sum(w_c) > 0 else None,
              "ssr_total": ssr.tolist()}
    pooled["n_eff_lt_3"] = bool(pooled["n_eff_centred"] < N_EFF_FLAG)
    # equal weight
    obj = np.nansum(norm, axis=0)
    je = int(np.nanargmin(obj))
    ksb_e = ks_b[:, je]
    ksb_e = ksb_e[np.isfinite(ksb_e)]
    equal = {"theta_star": float(thetas[je]), "ks_pooled_at_theta": float(sxy[:, je].sum() / sxx[:, je].sum()),
             "ks_basin_median": float(np.median(ksb_e)) if ksb_e.size else None,
             "ks_basin_q16": float(np.percentile(ksb_e, 16)) if ksb_e.size else None,
             "ks_basin_q84": float(np.percentile(ksb_e, 84)) if ksb_e.size else None,
             "mean_basin_r2": float(1 - np.nanmean(norm[:, je])), "theta_at_edge": bool(je in (0, thetas.size - 1)),
             "objective": obj.tolist(), "n_eff": float(np.sum(np.isfinite(norm[:, je])))}
    # per-basin best theta
    jb = np.nanargmin(np.where(np.isfinite(ssr_b), ssr_b, np.inf), axis=1)
    theta_b = thetas[jb]
    ks_own = np.array([ks_b[i, jb[i]] for i in range(nb)])
    median = {"theta_star": float(np.median(theta_b)), "theta_q16": float(np.percentile(theta_b, 16)),
              "theta_q84": float(np.percentile(theta_b, 84)), "frac_at_edge": float(np.mean((jb == 0) | (jb == thetas.size - 1))),
              "ks_at_own_theta_median": float(np.nanmedian(ks_own)), "theta_per_basin": theta_b.tolist()}
    if n_boot > 0 and nb >= 3:
        rng = np.random.default_rng(seed)
        th_p = np.empty(n_boot)
        th_e = np.empty(n_boot)
        th_m = np.empty(n_boot)
        ks_p = np.empty(n_boot)
        ks_e = np.empty(n_boot)
        for i in range(n_boot):
            idx = rng.integers(0, nb, nb)
            w = np.bincount(idx, minlength=nb).astype(float)
            jj = int(np.nanargmin(np.nansum(ssr_b * w[:, None], axis=0)))
            th_p[i] = thetas[jj]
            ks_p[i] = (sxy[:, jj] * w).sum() / (sxx[:, jj] * w).sum()
            jje = int(np.nanargmin(np.nansum(norm * w[:, None], axis=0)))
            th_e[i] = thetas[jje]
            ks_e[i] = (sxy[:, jje] * w).sum() / (sxx[:, jje] * w).sum()
            th_m[i] = np.median(theta_b[idx])
        pooled.update({"theta_" + k: v for k, v in _ci(th_p).items()})
        pooled.update({"ks_" + k: v for k, v in _ci(ks_p).items()})
        equal.update({"theta_" + k: v for k, v in _ci(th_e).items()})
        equal.update({"ks_" + k: v for k, v in _ci(ks_e).items()})
        median.update({"theta_" + k: v for k, v in _ci(th_m).items()})
        out["n_boot"] = n_boot
        out["bootstrap_unit"] = "basin"
    out.update({"pooled": pooled, "equal": equal, "median": median})
    return out


def ladder(Z, A, FD, dx, dy, labels, thresholds: Sequence[float], thetas: np.ndarray, truncated_outlets: set[int],
           keep: set[int] | None = None, n_boot: int = N_BOOT, store: dict | None = None, tag: str = "") -> list[dict[str, Any]]:
    steps = []
    for a_c in thresholds:
        t0 = time.perf_counter()
        try:
            net = channel_network(Z, _restrict_area(A, labels, keep), FD, a_c, dx, dy)
            mom = basin_moments(net, thetas, labels, truncated_outlets)
            est = estimators(mom, thetas, n_boot=n_boot)
            est.update({"A_c": a_c, "n_truncated_basins": int(mom["truncated"].sum()), "wall_s": time.perf_counter() - t0})
            if store is not None:
                store[a_c] = mom
        except ValueError as exc:
            est = {"A_c": a_c, "error": str(exc), "n_basins": 0, "pooled": None, "equal": None, "median": None,
                   "flag_few_basins": True}
        steps.append(est)
        p, e, m = est.get("pooled") or {}, est.get("equal") or {}, est.get("median") or {}
        logger.info("%s A_c=%-8g basins=%s pooled theta=%s (n_eff %s) equal theta=%s median theta=%s", tag, a_c,
                    est.get("n_basins"), p.get("theta_star"), None if p.get("n_eff_centred") is None else f"{p['n_eff_centred']:.1f}",
                    e.get("theta_star"), m.get("theta_star"))
    return steps


def _strip(est: dict[str, Any]) -> dict[str, Any]:
    """The estimator record without the long per-theta curves (for the JSON)."""
    out = dict(est)
    for k in ("pooled", "equal", "median"):
        if out.get(k):
            out[k] = {kk: vv for kk, vv in out[k].items() if kk not in ("ssr_total", "objective", "theta_per_basin")}
    return out


# ----------------------------------------------------------------------------- hilltops

def _basin_lookup(net: ChannelNetwork) -> np.ndarray:
    """Channel-pixel labels (row-major order of the mask, as basin_labels assigns them) -> network basin id."""
    ch_rows, ch_cols = np.nonzero(net.mask)
    pos = {(int(r), int(c)): i for i, (r, c) in enumerate(zip(net.rows, net.cols, strict=True))}
    lookup = np.zeros(ch_rows.size, dtype=np.int64)
    for i, (r, c) in enumerate(zip(ch_rows.tolist(), ch_cols.tolist(), strict=True)):
        lookup[i] = net.basin[pos[(r, c)]]
    return lookup


def curvature_on_mask(C_grids: dict[int, np.ndarray], mask: np.ndarray, U: float, dx: float, w_fixed: int,
                      n_boot: int = N_BOOT, seed: int = 0) -> dict[str, Any]:
    """C_ht(L) on a hilltop mask, the re-selected plateau scale, and C_ht with a segment bootstrap at w_fixed and w*."""
    hw = sorted(C_grids)
    L = np.array([(2 * w + 1) * dx for w in hw])
    seg, n_seg = label(mask, structure=np.ones((3, 3)))
    if int(mask.sum()) < 20:
        return {"n_pixels": int(mask.sum()), "n_segments": int(n_seg), "window_lengths_m": L.tolist(), "C_ht_median_curve": None,
                "w_star_reselected": None, "L_star_reselected_m": None, "reselect_note": "fewer than 20 hilltop pixels; no curvature",
                "at_fixed_L": {"w": int(w_fixed), "L_m": float((2 * w_fixed + 1) * dx), "C_ht": float("nan"), "D0": None, "n_pixels": int(mask.sum())},
                "at_reselected_L": {"w": None, "L_m": None, "C_ht": float("nan"), "D0": None, "n_pixels": int(mask.sum())}}
    med = []
    for w in hw:
        v = -C_grids[w][mask]
        v = v[np.isfinite(v)]
        med.append(float(np.median(v)) if v.size else float("nan"))
    i_star, note = select_scale(L, np.array(med))

    def at(w: int) -> dict[str, Any]:
        vals = -C_grids[w][mask]
        seg_ids = seg[mask]
        ok = np.isfinite(vals)
        vals, seg_ids = vals[ok], seg_ids[ok]
        C_ht = float(np.median(vals)) if vals.size else float("nan")
        rec = {"w": int(w), "L_m": float((2 * w + 1) * dx), "C_ht": C_ht, "D0": float(U / C_ht) if C_ht > 0 else None,
               "n_pixels": int(vals.size), "frac_convex": float(np.mean(vals > 0)) if vals.size else None}
        if n_boot > 0 and n_seg >= 5 and vals.size:
            rng = np.random.default_rng(seed)
            uniq = np.unique(seg_ids)
            bs = np.empty(n_boot)
            for i in range(n_boot):
                pick = rng.choice(uniq, uniq.size, replace=True)
                counts = np.bincount(pick, minlength=int(seg_ids.max()) + 1)
                bs[i] = _weighted_median(vals, counts[seg_ids])
            rec["C_ht_ci68"] = [float(np.percentile(bs, 16)), float(np.percentile(bs, 84))]
            rec["C_ht_ci95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
            rec["D0_ci68"] = [float(U / rec["C_ht_ci68"][1]), float(U / rec["C_ht_ci68"][0])]
            rec["D0_ci95"] = [float(U / rec["C_ht_ci95"][1]), float(U / rec["C_ht_ci95"][0])]
        return rec

    return {"n_pixels": int(mask.sum()), "n_segments": int(n_seg), "window_lengths_m": L.tolist(), "C_ht_median_curve": med,
            "w_star_reselected": int(hw[i_star]), "L_star_reselected_m": float(L[i_star]), "reselect_note": note,
            "at_fixed_L": at(w_fixed), "at_reselected_L": at(hw[i_star])}


def hilltop_audit(Z, A, FD, dx, dy, U: float, L_star_m: float, w_star: int, A_c_rollover: float | None,
                  n_boot: int, half_widths=DEFAULT_HALF_WIDTHS) -> dict[str, Any]:
    C_grids = {w: total_curvature(Z, w, dx, dy) for w in half_widths}
    out: dict[str, Any] = {"L_star_m": L_star_m, "w_star": w_star, "networks": {}}
    networks = list(HILLTOP_NETWORKS) + ([A_c_rollover] if A_c_rollover else [])
    for a_c in networks:
        key = "rollover" if (A_c_rollover and a_c == A_c_rollover) else f"{a_c:g}"
        net = channel_network(Z, A, FD, a_c, dx, dy)
        mask0, diag = hilltop_mask(Z, A, FD, a_c, dx, dy, basin_of_channel=_basin_lookup(net))
        channel = np.isfinite(A) & (A >= a_c)
        dist = distance_transform_edt(~channel, sampling=(dy, dx))
        rec: dict[str, Any] = {"A_c": a_c, "n_channel_px": int(channel.sum()), "mask_diagnostics": diag}
        for excl, name in ((False, "all"), (True, "excluded")):
            mask = mask0 & (dist >= 0.5 * L_star_m) if excl else mask0
            d = dist[mask]
            r = curvature_on_mask(C_grids, mask, U, dx, w_star, n_boot=n_boot)
            r["exclusion"] = {"rule": f"drop hilltop pixels closer than L*/2 = {0.5 * L_star_m:g} m to a channel pixel" if excl else "none",
                              "n_dropped": int(mask0.sum() - mask.sum())}
            r["width_2x_distance_m"] = {str(q): float(2 * np.percentile(d, q)) for q in (5, 10, 25, 50, 75, 90, 95)} if d.size else None
            rec[name] = r
        out["networks"][key] = rec
        logger.info("hilltops A_c=%g: %d -> %d pixels after exclusion; C_ht(L*) %.4g -> %.4g", a_c, rec["all"]["n_pixels"],
                    rec["excluded"]["n_pixels"], rec["all"]["at_fixed_L"]["C_ht"], rec["excluded"]["at_fixed_L"]["C_ht"])
    return out


# ----------------------------------------------------------------------------- driver

def _latest_params(step: str) -> dict[str, Any] | None:
    runs = find_runs(step)
    for r in reversed(runs):
        try:
            return json.loads((r / "params.json").read_text()) | {"_run": r.name}
        except Exception:  # noqa: BLE001
            continue
    return None


def audit_tile(tile: str, thetas: np.ndarray, n_boot: int, store: dict, measure_params: dict, crop=None) -> dict[str, Any]:
    t0 = time.perf_counter()
    td = _metre_tile(tile, crop)
    Z, A, FD, dx, dy = td.Z, td.A, td.FD, td.dx, td.filled.dy
    out: dict[str, Any] = {"tile": tile, "site": td.site.site, "U": td.site.U, "inputs": td.provenance()}
    labels = edge_outlet_labels(A, FD)
    trunc = basin_truncation(A, labels)
    truncated = set(trunc.loc[trunc.truncated, "outlet_flat"].astype(int).tolist())
    out["truncation_basins"] = {"n_basins": int(len(trunc)), "n_truncated": len(truncated), "n_interior": int((~trunc.truncated).sum())}
    store[tile] = {}
    out["free_ladder"] = [_strip(s) for s in ladder(Z, A, FD, dx, dy, labels, FINE_LADDER, thetas, truncated, n_boot=n_boot,
                                                    store=store[tile], tag=f"{tile} free")]
    keep, info = fixed_basin_set(Z, A, FD, dx, dy, labels)
    out["fixed_basins"] = info
    out["fixed_ladder"] = [_strip(s) for s in ladder(Z, A, FD, dx, dy, labels, FIXED_LADDER, thetas, truncated, keep=keep,
                                                     n_boot=n_boot, tag=f"{tile} fixed")]
    # truncated against complete basins, once, at A_c = 1e3 (equal-weight and pooled)
    mom = store[tile].get(1000.0)
    tc: dict[str, Any] = {"A_c": 1000.0}
    if mom is not None and mom["n"].size:
        for name, sel in (("truncated", mom["truncated"]), ("complete", ~mom["truncated"])):
            tc[name] = _strip(estimators(subset_moments(mom, sel), thetas, n_boot=n_boot)) if sel.sum() else {"n_basins": 0}
    out["truncated_vs_complete_1e3"] = tc
    # hilltops
    cur = measure_params.get(tile, {}).get("curvature", {}) if measure_params else {}
    L_star = float(cur.get("L_star_m", 0) or 0)
    w_star = int(cur.get("w_star", 0) or 0)
    if not L_star:
        L_star, w_star = (2 * 10 + 1) * dx, 10
        out["hilltop_note"] = "no measure run found; L* = 21 cells used"
    out["hilltops"] = hilltop_audit(Z, A, FD, dx, dy, td.site.U, L_star, w_star, cur.get("A_c"), n_boot)
    out["hilltops"]["measure_run_C_ht"] = {"C_ht": cur.get("C_ht"), "A_c": cur.get("A_c"), "L_star_m": cur.get("L_star_m"),
                                           "n_hilltop": cur.get("n_hilltop"), "run": measure_params.get(tile, {}).get("_run")}
    out["wall_s"] = time.perf_counter() - t0
    logger.info("%s done in %.0f s", tile, out["wall_s"])
    return out


def run_measure_audit(tiles: Sequence[str], n_boot: int = N_BOOT, run_id: str | None = None, results_root=None,
                      thetas=DEFAULT_THETAS, crop=None) -> RunRecorder:
    rec = RunRecorder("measure_audit", "_".join(tiles) if len(tiles) <= 3 else f"{len(tiles)}tiles", run_id=run_id, root=results_root)
    thetas = np.asarray(thetas, dtype=float)
    # per-tile L*, w*, rollover A_c and C_ht from the latest measure run of each tile
    measure_params: dict[str, Any] = {}
    for r in find_runs("measure"):
        try:
            p = json.loads((r / "params.json").read_text())
            for t, v in p.get("tiles", {}).items():
                measure_params[t] = v | {"_run": r.name}
        except Exception:  # noqa: BLE001
            continue
    diag = _latest_params("measure_diagnostics") or {}
    store: dict = {}
    results = {}
    snaps = {}
    for tile in tiles:
        results[tile] = audit_tile(tile, thetas, n_boot, store, measure_params, crop=crop)
        snaps[tile] = config_snapshot(load_tile_config(tile))
    sites = sorted({t[:2] for t in tiles})
    pooled: dict[str, Any] = {}
    for site in sites:
        st_tiles = [t for t in tiles if t.startswith(site)]
        steps = []
        for a_c in FINE_LADDER:
            parts = [store[t][a_c] for t in st_tiles if a_c in store.get(t, {}) and store[t][a_c]["n"].size]
            if not parts:
                steps.append({"A_c": a_c, "n_basins": 0, "pooled": None, "equal": None, "median": None})
                continue
            est = _strip(estimators(concat_moments(parts), thetas, n_boot=n_boot))
            est.update({"A_c": a_c, "n_tiles": len(parts)})
            steps.append(est)
        pooled[site] = {"tiles": st_tiles, "ladder": steps}
    rec.add_params("tiles", results)
    rec.add_params("pooled", pooled)
    # synthetic ridges on the post-exclusion width distribution (1e4 network), per site
    syn = {}
    for site in sites:
        st_tiles = [t for t in tiles if t.startswith(site)]
        wd = {t: results[t]["hilltops"]["networks"]["10000"]["excluded"].get("width_2x_distance_m") for t in st_tiles}
        wd = {t: v for t, v in wd.items() if v}
        if not wd:
            continue
        widths = sorted({max(2, round(v[q])) for v in wd.values() for q in ("5", "10", "25", "50", "75", "90")}
                        | {4, 8, 12, 16, 20, 30, 40, 60, 80, 120})
        dtiles = diag.get("tiles", {})
        ap_tiles = [t for t in dtiles if t.startswith("AP") and dtiles[t].get("facets", {}).get("equiv_length_m")]
        facet = float(np.median([dtiles[t]["facets"]["equiv_length_m"]["50"] for t in ap_tiles])) if ap_tiles else 2.76
        gm_tiles = [t for t in dtiles if t.startswith("GM") and dtiles[t].get("roughness")]
        sigma = float(np.median([dtiles[t]["roughness"]["mad_all"] for t in gm_tiles])) if gm_tiles else 0.043
        c_true = float(np.median([results[t]["hilltops"]["networks"]["10000"]["excluded"]["at_fixed_L"]["C_ht"] for t in st_tiles]))
        L_star = {t: results[t]["hilltops"]["L_star_m"] for t in st_tiles}
        syn[site] = synthetic_ridges(widths, c_true, facet if site == "AP" else 0.0, sigma, L_star=L_star)
        syn[site]["widths_source"] = "2 x hilltop-to-channel distance of the pixels kept after the L*/2 exclusion, network at A_c = 1e4"
        var = "tin_noise" if site == "AP" else "noise"
        Ws = np.array([c["width_m"] for c in syn[site]["cases"]])
        o = np.argsort(Ws)
        bias_tiles = {}
        for t in st_tiles:
            b = np.array([c["bias_at_L_star"][var][t] for c in syn[site]["cases"]])[o]
            w_t = wd.get(t)
            bias_tiles[t] = {"variant": var, "bias_at_median_width": float(np.interp(w_t["50"], Ws[o], b)) if w_t else None,
                             "bias_at_q25_width": float(np.interp(w_t["25"], Ws[o], b)) if w_t else None,
                             "width_median_m": w_t["50"] if w_t else None, "width_q25_m": w_t["25"] if w_t else None}
        syn[site]["bias_post_exclusion"] = bias_tiles
    rec.add_params("synthetic_ridges_post_exclusion", syn)
    rec.add_params("settings", {"fine_ladder": list(FINE_LADDER), "fixed_ladder": list(FIXED_LADDER), "n_boot": n_boot,
                                "bootstrap_unit": "basin (chi), segment (curvature)", "n_eff_flag": N_EFF_FLAG,
                                "hilltop_networks": list(HILLTOP_NETWORKS), "thetas": thetas.tolist(),
                                "measure_diagnostics_run": diag.get("_run"),
                                "filters": {"min_pixels": MIN_PIXELS_PER_BASIN, "min_relief_m": MIN_RELIEF_M,
                                            "min_chi_range_m": MIN_CHI_RANGE_M, "theta_ref_chi": THETA_REF_CHI}})
    rows = []
    for t, r in results.items():
        for kind in ("free_ladder", "fixed_ladder"):
            for s in r[kind]:
                for est in ("pooled", "equal", "median"):
                    e = s.get(est)
                    if not e:
                        continue
                    rows.append({"tile": t, "site": r["site"], "ladder": kind, "estimator": est, "A_c": s["A_c"], "n_basins": s.get("n_basins"),
                                 "theta_star": e.get("theta_star"), "ks": e.get("ks", e.get("ks_pooled_at_theta")),
                                 "theta_lo68": (e.get("theta_ci68") or [None, None])[0], "theta_hi68": (e.get("theta_ci68") or [None, None])[1],
                                 "theta_lo95": (e.get("theta_ci95") or [None, None])[0], "theta_hi95": (e.get("theta_ci95") or [None, None])[1],
                                 "n_eff_centred": e.get("n_eff_centred"), "n_eff_through_origin": e.get("n_eff_through_origin")})
    for site, p in pooled.items():
        for s in p["ladder"]:
            for est in ("pooled", "equal", "median"):
                e = s.get(est)
                if not e:
                    continue
                rows.append({"tile": f"{site}_pooled", "site": site, "ladder": "pooled", "estimator": est, "A_c": s["A_c"],
                             "n_basins": s.get("n_basins"), "theta_star": e.get("theta_star"), "ks": e.get("ks", e.get("ks_pooled_at_theta")),
                             "theta_lo68": (e.get("theta_ci68") or [None, None])[0], "theta_hi68": (e.get("theta_ci68") or [None, None])[1],
                             "theta_lo95": (e.get("theta_ci95") or [None, None])[0], "theta_hi95": (e.get("theta_ci95") or [None, None])[1],
                             "n_eff_centred": e.get("n_eff_centred"), "n_eff_through_origin": e.get("n_eff_through_origin")})
    rec.write_table(pd.DataFrame(rows), "ladders")
    rec.finalize({"tiles": snaps, "crop": crop})
    return rec
