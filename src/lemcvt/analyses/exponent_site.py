"""Site-level exponent profile and the unit-consistent K recovery (addendum 4, sections 6.1 and 6.2).

Reads results only. From the latest v2 ``m_profile`` run of every tile (AP in metres) it builds

    S(m) = sum_t MSE*_t(m) / MSE*_t,min

over the five tiles of a site, where ``MSE*_t(m)`` is the tile's profile MSE minimised over A_crit
and (K, D) at that m (the ``profile_m`` rows of the run). It reports the site optimum, the 1 % and
5 % intervals in m (where S <= S_min (1 + level)), the same with each tile left out in turn, and the
measured site-pooled theta(A_c) with its 95 % band at the median fitted A_crit of the site
(from the latest ``measure_audit`` run, pooled and equal-weight estimators).

Section 6.2: at the site-level m, the fitted K is the joint-profile K at that fixed m (the m-profile
row minimised over A_crit) and k_s is the pooled through-origin chi-z slope at theta = m over the
tile's channel network above A_c = 1e3 and 1e4 m^2 (``ks_at_theta`` of the audit's free ladder, so
the measurement never sees the model's A_crit). The ratio U/k_s / K_fit is reported per tile.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from lemcvt.config import TILE_IDS, load_site
from lemcvt.results import find_runs

logger = logging.getLogger(__name__)

LEVELS = (0.01, 0.05)


def latest_m_profile_runs(tiles=TILE_IDS, m_min_required: float | None = 0.05) -> dict[str, tuple[Path, dict]]:
    """Latest v2 m_profile run per tile on metre grids; with ``m_min_required`` only runs whose m grid reaches it."""
    out: dict[str, tuple[Path, dict]] = {}
    for r in find_runs("tile_run"):
        try:
            p = json.loads((r / "params.json").read_text())
            rj = json.loads((r / "run.json").read_text())
        except Exception:  # noqa: BLE001
            continue
        if p.get("optimizer") != "v2" or "m_profile" not in p.get("v2", {}) or (rj.get("config") or {}).get("crop"):
            continue
        if p["tile"] not in tiles or p.get("inputs", {}).get("z_units_working") != "m":
            continue
        mv = p["v2"]["m_profile"]["grid"]["m_values"]
        if m_min_required is not None and min(mv) > m_min_required + 1e-9:
            continue
        out[p["tile"]] = (r, p)
    return out


def _intervals(m: np.ndarray, S: np.ndarray, level: float) -> dict[str, Any]:
    smin = float(np.nanmin(S))
    idx = np.where(S <= smin * (1 + level))[0]
    runs = []
    if idx.size:
        start = prev = idx[0]
        for i in idx[1:]:
            if i != prev + 1:
                runs.append((start, prev))
                start = i
            prev = i
        runs.append((start, prev))
    return {"m_min": float(m[idx].min()) if idx.size else None, "m_max": float(m[idx].max()) if idx.size else None,
            "intervals": [[float(m[a]), float(m[b])] for a, b in runs], "n_grid_points": int(idx.size),
            "touches_grid_edge": bool(idx.size and (idx[0] == 0 or idx[-1] == m.size - 1))}


def site_profile(runs: dict[str, tuple[Path, dict]], tiles: list[str]) -> dict[str, Any]:
    """S(m) over ``tiles`` on their common m grid."""
    curves = {}
    for t in tiles:
        pm = runs[t][1]["v2"]["m_profile"]["profile_m"]
        m = np.round([q["m"] for q in pm], 2)
        mse = np.array([q["mse"] for q in pm])
        curves[t] = dict(zip(m.tolist(), (mse / mse.min()).tolist(), strict=True))
    common = sorted(set.intersection(*(set(c) for c in curves.values())))
    m = np.array(common)
    S = np.array([sum(curves[t][mm] for t in tiles) for mm in common])
    j = int(np.argmin(S))
    out = {"tiles": tiles, "m": m.tolist(), "S": S.tolist(), "m_opt": float(m[j]), "S_min": float(S[j]), "n_tiles": len(tiles),
           "per_tile_normalised": {t: [curves[t][mm] for mm in common] for t in tiles}}
    for lev in LEVELS:
        out[f"interval_{lev:g}"] = _intervals(m, S, lev)
    return out


def site_summary(tiles=TILE_IDS, runs: dict | None = None) -> dict[str, Any]:
    runs = runs if runs is not None else latest_m_profile_runs(tiles)
    out: dict[str, Any] = {"runs": {t: r.name for t, (r, _p) in runs.items()}, "sites": {}}
    for site in sorted({t[:2] for t in runs}):
        st = [t for t in TILE_IDS if t.startswith(site) and t in runs]
        rec: dict[str, Any] = {"all": site_profile(runs, st), "leave_one_out": {}}
        for t in st:
            rest = [u for u in st if u != t]
            if len(rest) >= 2:
                rec["leave_one_out"][t] = site_profile(runs, rest)
        rec["tile_optima"] = {t: runs[t][1]["v2"]["m_profile"]["optimum"] for t in st}
        a = np.array([runs[t][1]["v2"]["m_profile"]["optimum"]["A_crit"] for t in st])
        rec["median_fitted_A_crit"] = float(np.median(a))
        m_site = rec["all"]["m_opt"]
        a_at = []
        for t in st:
            pm = runs[t][1]["v2"]["m_profile"]["profile_m"]
            row = min(pm, key=lambda q: abs(q["m"] - m_site))
            a_at.append(row["A_crit"])
        rec["median_A_crit_at_site_m"] = float(np.median(a_at))
        out["sites"][site] = rec
    return out


# ----------------------------------------------------------------------------- measured theta(A_c) bands

def latest_audit() -> tuple[Path, dict] | None:
    runs = [r for r in find_runs("measure_audit") if (r / "params.json").exists()]
    if not runs:
        return None
    return runs[-1], json.loads((runs[-1] / "params.json").read_text())


def theta_band_at(audit: dict, site: str, A_c: float, estimator: str = "pooled") -> dict[str, Any] | None:
    """Site-pooled theta* and 95 % band of ``estimator`` interpolated in log A_c at ``A_c``."""
    steps = [s for s in audit.get("pooled", {}).get(site, {}).get("ladder", []) if s.get(estimator) and s[estimator].get("theta_star") is not None]
    if not steps:
        return None
    A = np.array([s["A_c"] for s in steps])
    th = np.array([s[estimator]["theta_star"] for s in steps])
    lo = np.array([(s[estimator].get("theta_ci95") or [np.nan, np.nan])[0] for s in steps])
    hi = np.array([(s[estimator].get("theta_ci95") or [np.nan, np.nan])[1] for s in steps])
    nb = np.array([s.get("n_basins", 0) for s in steps])
    la, x = np.log10(A), np.log10(A_c)
    if x < la.min() or x > la.max():
        return None
    return {"A_c": float(A_c), "estimator": estimator, "theta": float(np.interp(x, la, th)), "lo95": float(np.interp(x, la, lo)),
            "hi95": float(np.interp(x, la, hi)), "n_basins_nearest": int(nb[np.argmin(np.abs(la - x))])}


# ----------------------------------------------------------------------------- 6.2 K recovery

def k_recovery(summary: dict, audit: dict, runs: dict | None = None, thresholds=(1e3, 1e4)) -> list[dict[str, Any]]:
    runs = runs if runs is not None else latest_m_profile_runs()
    thetas = np.asarray(audit["settings"]["thetas"], float)
    rows = []
    for site, rec in summary["sites"].items():
        m_site = rec["all"]["m_opt"]
        j = int(np.argmin(np.abs(thetas - m_site)))
        if abs(thetas[j] - m_site) > 1e-6:
            logger.warning("%s: site m %.2f is not on the audit's theta grid (nearest %.2f)", site, m_site, thetas[j])
        U = load_site(site).U
        for t in rec["tile_optima"]:
            pm = runs[t][1]["v2"]["m_profile"]["profile_m"]
            row = min(pm, key=lambda q: abs(q["m"] - m_site))
            K_fit = row["K"]
            r: dict[str, Any] = {"tile": t, "site": site, "m_site": m_site, "A_crit_fit": row["A_crit"], "K_fit": K_fit, "D_fit": row["D"],
                                 "mse_fit": row["mse"], "theta_used": float(thetas[j])}
            steps = {s["A_c"]: s for s in audit["tiles"][t]["free_ladder"]}
            for a_c in thresholds:
                s = steps.get(float(a_c))
                ks = (s or {}).get("pooled", {}) or {}
                curve = ks.get("ks_at_theta")
                k_s = float(curve[j]) if curve else None
                r[f"ks_{a_c:g}"] = k_s
                r[f"n_basins_{a_c:g}"] = (s or {}).get("n_basins")
                r[f"n_eff_{a_c:g}"] = ks.get("n_eff_centred")
                r[f"ratio_{a_c:g}"] = (U / k_s) / K_fit if k_s else None
            # site-pooled k_s at the same theta for reference
            for a_c in thresholds:
                ps = next((s for s in audit["pooled"][site]["ladder"] if s["A_c"] == float(a_c)), None)
                curve = ((ps or {}).get("pooled") or {}).get("ks_at_theta")
                r[f"ks_site_pooled_{a_c:g}"] = float(curve[j]) if curve else None
            rows.append(r)
    return rows
