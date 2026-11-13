"""Test 2 analyse phase: directive sections 5-8 (``reports/directives/head_extraction_test2_directive.md``), as
amended by addendum 10, 1.3 and 4 (``lemcvt phase7 heads-analyse``).

Inputs
- the extracted heads of a ``lemcvt phase7 heads`` run (default ``20261113T022845Z_heads_10tiles``); the fit set is
  Method P, retained, ``head_class == hollow`` of one smoothing set (primary ``OWF+DT2`` at both sites), edited by
  ``heads_review.csv`` beside the CSV if one exists (none did at the review gate: every retained hollow head is kept);
- the tile grids through the lemcvt loader (metres; the stored D8 area and flow directions; the filled DEM);
- the A_crit bracket of each tile from its latest refined Phase 6 partitioned run
  (``optimizer_v2._latest_refined_partitioned``): stage 2 at (K0, D0_A), stage 2 at D0_B where present, the linear
  joint optimum and its 1 % / 5 % misfit intervals.

Section 5 measurements are recomputed here from the grids (A_D8, S_up_5/10/20, S_centred_10) and checked against the
extract-phase CSV; A from D-infinity routing on the filled tile is the routing sensitivity. The D-inf area is computed on
the 1 km tile only, while the stored D8 area came from the larger DEM: heads whose D8 catchment crosses the tile edge
(stored A_D8 > A_D8 recomputed on the tile) are flagged ``truncated_on_tile``.
"""

from __future__ import annotations

import json
import logging
import time
import zlib
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .config import ANALYSIS, EXTRACTION, QC  # noqa: E402
from .figures import C_MUTED, C_TEXT, _style, fig_qc_map  # noqa: E402
from .paths import d8_accumulate, detect_heads, path_slopes  # noqa: E402

logger = logging.getLogger(__name__)

HEADS_RUN_DEFAULT = "20261113T022845Z_heads_10tiles"
PRIMARY_SET = "OWF+DT2"
SETS = ["OWF", "OWF+DT0.5", "OWF+DT2", "OWF+DT5"]
SLOPES = ["S_up_10", "S_up_5", "S_up_20", "S_centred_10"]
ROUTINGS = {"D8": "A_D8_m2", "Dinf": "A_dinf_m2"}
STAGES = ["joint", "stage2", "stage2_B"]
MIN_TILE_FIT = 5            # heads with a finite slope needed for a per-tile fit
CHANCE_POINTS = 20000       # random interior points for the chance level of the 20 m match
# Clubb et al. (2014) p. 4296: average contributing area and AS^2 of the field-mapped heads (literature memo, item 1)
CLUBB_OHIO = {"Mid Bailey Run": {"n_heads": 53, "mean_A_m2": 413.0, "mean_AS2_m2": 278.0, "R2_area_slope": 0.0391},
              "Indian Creek": {"n_heads": 36, "mean_A_m2": 1035.0, "mean_AS2_m2": 640.0, "R2_area_slope": 0.1342}}
C_TILE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"]
C_SITE = {"AP": "#2a78d6", "GM": "#eb6834"}


# ----------------------------------------------------------------------------- pure helpers (tested)

def iqr_overlap(lo: float, hi: float, q1: float, q3: float) -> bool:
    """True if the closed interval [lo, hi] (the A_crit bracket, either order) intersects [q1, q3]."""
    a, b = min(lo, hi), max(lo, hi)
    return bool(max(a, q1) <= min(b, q3))


def fit_loglog(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """OLS of y on x and the reduced-major-axis slope sign(r) sd(y)/sd(x); x, y already log10."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    out = {"n": n, "b_ols": np.nan, "a_ols": np.nan, "b_rma": np.nan, "r": np.nan, "r2": np.nan}
    if n < 3:
        return out
    xd, yd = x - x.mean(), y - y.mean()
    sxx, syy, sxy = float(xd @ xd), float(yd @ yd), float(xd @ yd)
    if sxx <= 0 or syy <= 0:
        return out
    b = sxy / sxx
    r = sxy / np.sqrt(sxx * syy)
    out.update(b_ols=b, a_ols=float(y.mean() - b * x.mean()), b_rma=float(np.sign(r) * np.sqrt(syy / sxx)), r=float(r), r2=float(r * r))
    return out


def fit_pooled(x: np.ndarray, y: np.ndarray, g: np.ndarray) -> dict[str, Any]:
    """Pooled slope with one intercept per group (within-group demeaning): OLS, RMA and the within-group r^2."""
    x, y, g = np.asarray(x, float), np.asarray(y, float), np.asarray(g)
    xd, yd = x.copy(), y.copy()
    inter: dict[Any, tuple[float, float]] = {}
    for k in np.unique(g):
        m = g == k
        inter[k] = (float(x[m].mean()), float(y[m].mean()))
        xd[m] -= inter[k][0]
        yd[m] -= inter[k][1]
    sxx, syy, sxy = float(xd @ xd), float(yd @ yd), float(xd @ yd)
    out: dict[str, Any] = {"n": len(x), "n_groups": len(inter), "b_ols": np.nan, "b_rma": np.nan, "r2_within": np.nan, "means": inter}
    if len(x) < 3 or sxx <= 0 or syy <= 0:
        return out
    r = sxy / np.sqrt(sxx * syy)
    out.update(b_ols=sxy / sxx, b_rma=float(np.sign(r) * np.sqrt(syy / sxx)), r2_within=float(r * r))
    return out


def bootstrap_fit(x: np.ndarray, y: np.ndarray, g: np.ndarray, n_boot: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Resample observations with replacement within each group; pooled (tile-intercept) OLS and RMA slopes and the
    group means of x and y per resample (for 6c predictions). Degenerate resamples give NaN."""
    x, y, g = np.asarray(x, float), np.asarray(y, float), np.asarray(g)
    keys = list(np.unique(g))
    idx = [np.nonzero(g == k)[0] for k in keys]
    b_ols = np.full(n_boot, np.nan)
    b_rma = np.full(n_boot, np.nan)
    mx = np.full((n_boot, len(keys)), np.nan)
    my = np.full((n_boot, len(keys)), np.nan)
    for b in range(n_boot):
        sxx = syy = sxy = 0.0
        for k, ii in enumerate(idx):
            s = ii[rng.integers(0, len(ii), len(ii))]
            xs, ys = x[s], y[s]
            mx[b, k], my[b, k] = xs.mean(), ys.mean()
            xd, yd = xs - mx[b, k], ys - my[b, k]
            sxx += float(xd @ xd)
            syy += float(yd @ yd)
            sxy += float(xd @ yd)
        if sxx > 0 and syy > 0:
            b_ols[b] = sxy / sxx
            b_rma[b] = np.sign(sxy) * np.sqrt(syy / sxx)
    return {"b_ols": b_ols, "b_rma": b_rma, "mean_x": mx, "mean_y": my, "keys": np.array(keys)}


def ci(v: np.ndarray, pct=(2.5, 97.5)) -> tuple[float, float]:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return (np.nan, np.nan)
    lo, hi = np.percentile(v, pct)
    return float(lo), float(hi)


def includes(lo: float, hi: float, val: float) -> bool | None:
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return None
    return bool(lo <= val <= hi)


def partition_heads_rc(area: np.ndarray, ui: np.ndarray, uj: np.ndarray, a_crit: float) -> tuple[np.ndarray, np.ndarray]:
    """Partition channel heads: cells where A first reaches A_crit along a flow path (A >= A_crit and no D8 donor with
    A >= A_crit). Thin wrapper of :func:`lemcvt.analyses.heads.paths.detect_heads`."""
    return detect_heads(area, ui, uj, a_crit)


def match_fraction(a_xy: np.ndarray, b_xy: np.ndarray, radius: float) -> tuple[float, np.ndarray]:
    """Fraction of points ``a`` with a point of ``b`` within ``radius``, and the nearest distances."""
    if len(a_xy) == 0:
        return np.nan, np.array([])
    if len(b_xy) == 0:
        return 0.0, np.full(len(a_xy), np.inf)
    d, _ = cKDTree(b_xy).query(a_xy)
    return float(np.mean(d <= radius)), d


# ----------------------------------------------------------------------------- inputs

def acrit_phase6(tile_id: str) -> dict[str, Any]:
    """Stage-2 (D0_A, and D0_B where present) and linear joint A_crit with the joint 1 % / 5 % intervals."""
    from lemcvt.analyses.optimizer_v2 import _latest_refined_partitioned

    r = _latest_refined_partitioned(tile_id)
    v = json.loads((r / "params.json").read_text())["v2"]["partitioned"]
    ji = v["joint_identifiability"]
    out: dict[str, Any] = {"tile": tile_id, "source_run": r.name, "m": v.get("m"),
                           "joint": float(v["joint_optimum"]["A_crit"]), "joint_where": v["joint_optimum"].get("where"),
                           "joint_K": v["joint_optimum"].get("K"), "joint_D": v["joint_optimum"].get("D"),
                           "stage2": float(v["stage2_optimum"]["A_crit"]), "stage2_D": v["stage2_optimum"].get("D"),
                           "stage2_B": float(v["stage2_B_optimum"]["A_crit"]) if v.get("stage2_B_optimum") else np.nan,
                           "stage2_B_D": (v.get("stage2_B_optimum") or {}).get("D")}
    for lev in ("0.01", "0.05"):
        lv = ji.get(f"level_{lev}") or {}
        out[f"joint_{lev}_min"] = lv.get("A_crit_min", np.nan)
        out[f"joint_{lev}_max"] = lv.get("A_crit_max", np.nan)
        out[f"joint_{lev}_n_intervals"] = lv.get("n_intervals", np.nan)
    return out


def load_fit_heads(heads_csv: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Retained Method P hollow-class heads of every smoothing set, after ``heads_review.csv`` if present."""
    H = pd.read_csv(heads_csv)
    fit = H[(H.method == "P") & H.retained.astype(bool) & (H.head_class == "hollow")].copy()
    rev_path = Path(heads_csv).parent / "heads_review.csv"
    info: dict[str, Any] = {"heads_csv": str(heads_csv), "review_csv": str(rev_path), "review_exists": rev_path.exists()}
    if rev_path.exists():
        rev = pd.read_csv(rev_path)
        drop = set(rev.loc[~rev["keep"].astype(str).str.lower().isin(["true", "1", "keep", "yes"]), "head_id"])
        info["n_dropped_by_review"] = int(fit.head_id.isin(drop).sum())
        fit = fit[~fit.head_id.isin(drop)]
    else:
        info["n_dropped_by_review"] = 0
        info["note"] = "no heads_review.csv: every retained hollow-class Method P head is kept"
    return fit.reset_index(drop=True), info


# ----------------------------------------------------------------------------- per tile measurement (section 5)

def measure_tile(site: str, t: int, fit: pd.DataFrame, acr: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    from lemcvt.analyses.width.network import FlowGraph
    from lemcvt.analyses.width.tiles import load_tile
    from lemcvt.hydro.dinf import compute_dinf

    t0 = time.time()
    tile = load_tile(site, t)
    graph = FlowGraph(tile, QC["trunk_area_m2"], 0.0)
    _, _, _, A_dinf = compute_dinf(tile.z, tile.dx, tile.fd)
    A_tile = d8_accumulate(tile.fd, tile.dx * tile.dy)   # topological order: exact on filled flats too
    t_dinf = time.time() - t0
    sub = fit[(fit.site == site) & (fit.tile == t)].copy()
    L_values = [5.0, 10.0, 20.0]
    recs = []
    for _, h in sub.iterrows():
        i, j = int(h.row), int(h.col)
        s = path_slopes(graph, i, j, L_values)
        rec = {"head_id": h.head_id, "A_D8_m2": float(tile.area[i, j]), "A_dinf_m2": float(A_dinf[i, j]),
               "A_D8_tile_m2": float(A_tile[i, j])}
        for k in SLOPES:
            rec[k] = s[k]
        rec["d_A_vs_csv"] = rec["A_D8_m2"] - float(h.area_D8_m2)
        rec["d_S_up_10_vs_csv"] = (s["S_up_10"] - h.S_up_10) if np.isfinite(h.S_up_10) else np.nan
        recs.append(rec)
    m = pd.DataFrame(recs, columns=["head_id", "A_D8_m2", "A_dinf_m2", "A_D8_tile_m2", *SLOPES, "d_A_vs_csv", "d_S_up_10_vs_csv"])
    sub = sub.drop(columns=[c for c in SLOPES if c in sub.columns]).merge(m, on="head_id", how="left")
    sub["truncated_on_tile"] = sub.A_D8_m2 > sub.A_D8_tile_m2 * 1.01 + 1e-9
    sub["AS2_m2"] = sub.A_D8_m2 * sub.S_up_10**2
    sub["AS2_dinf_m2"] = sub.A_dinf_m2 * sub.S_up_10**2

    # partition heads at each A_crit, and the 20 m correspondence (6d)
    ny, nx = tile.shape
    buf = EXTRACTION["edge_buffer_m"]
    radius = ANALYSIS["spatial_match_m"]
    # chance level: uniform points inside the edge buffer
    xr = rng.uniform(tile.x0 + buf, tile.x0 + nx * tile.dx - buf, CHANCE_POINTS)
    yr = rng.uniform(tile.y0 - ny * tile.dy + buf, tile.y0 - buf, CHANCE_POINTS)
    rand_xy = np.column_stack([xr, yr])
    parts, corr = [], []
    for st in STAGES:
        a = acr.get(st, np.nan)
        if not np.isfinite(a):
            continue
        r, c = partition_heads_rc(tile.area, graph.ui, graph.uj, a)
        x, y = tile.rc_to_xy(r, c)
        edge = np.minimum(np.minimum(r, c), np.minimum(ny - 1 - r, nx - 1 - c)).astype(float) * tile.pixel_size
        P = pd.DataFrame({"site": site, "tile": t, "stage": st, "A_crit_m2": a, "row": r, "col": c, "x": x, "y": y,
                          "area_D8_m2": tile.area[r, c], "edge_distance_m": edge, "interior": edge >= buf})
        parts.append(P)
        pxy = P[["x", "y"]].to_numpy()
        f_heads = {}
        for sm, hs in sub.groupby("smoothing"):
            fr, d = match_fraction(hs[["x", "y"]].to_numpy(), pxy, radius)
            sub.loc[hs.index, f"dist_partition_{st}_m"] = d
            f_heads[sm] = fr
        f_rand, _ = match_fraction(rand_xy, pxy, radius)
        Pint = P[P.interior]
        for sm, hs in sub.groupby("smoothing"):
            fr_p, _ = match_fraction(Pint[["x", "y"]].to_numpy(), hs[["x", "y"]].to_numpy(), radius)
            corr.append({"site": site, "tile": t, "smoothing": sm, "stage": st, "A_crit_m2": a, "n_heads": len(hs),
                         "n_partition_heads": len(P), "n_partition_heads_interior": len(Pint),
                         "frac_heads_with_partition_head": f_heads[sm], "frac_partition_heads_with_head": fr_p,
                         "chance_frac_random_point_with_partition_head": f_rand})
    info = {"tile": f"{site}{t}", "n_fit_heads_all_sets": len(sub), "dinf_s": round(t_dinf, 1), "elapsed_s": round(time.time() - t0, 1),
            "max_abs_dA_vs_csv": float(np.nanmax(np.abs(sub.d_A_vs_csv))) if len(sub) else np.nan,
            "max_abs_dS10_vs_csv": float(np.nanmax(np.abs(sub.d_S_up_10_vs_csv))) if sub.d_S_up_10_vs_csv.notna().any() else np.nan,
            "frac_cells_storedA_eq_tileA": float(np.mean(np.isclose(A_tile, tile.area, rtol=0.01)))}
    logger.info("%s%d: %d fit heads (all sets), D-inf %.0f s, total %.0f s", site, t, len(sub), t_dinf, time.time() - t0)
    return {"heads": sub, "partition": pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(),
            "corr": pd.DataFrame(corr), "info": info, "tile": tile}


# ----------------------------------------------------------------------------- analyses (section 6)

def table_6a(H: pd.DataFrame, acr: dict[str, dict], area_col: str = "A_D8_m2") -> pd.DataFrame:
    rows = []
    for (sm, site, t), s in H.groupby(["smoothing", "site", "tile"]):
        a = s[area_col].to_numpy()
        ac = acr[f"{site}{t}"]
        q = np.percentile(a, [5, 25, 50, 75, 95]) if len(a) else [np.nan] * 5
        rows.append({"smoothing": sm, "routing": area_col, "site": site, "tile": t, "n_heads": len(a),
                     "A_p05": q[0], "A_q1": q[1], "A_median": q[2], "A_q3": q[3], "A_p95": q[4],
                     "A_crit_stage2": ac["stage2"], "A_crit_stage2_B": ac["stage2_B"], "A_crit_joint": ac["joint"],
                     "joint_0.01_min": ac["joint_0.01_min"], "joint_0.01_max": ac["joint_0.01_max"],
                     "joint_0.05_min": ac["joint_0.05_min"], "joint_0.05_max": ac["joint_0.05_max"],
                     "bracket_overlaps_IQR": iqr_overlap(ac["stage2"], ac["joint"], q[1], q[3]) if len(a) else None,
                     "bracketB_overlaps_IQR": (iqr_overlap(ac["stage2_B"], ac["joint"], q[1], q[3]) if len(a) and np.isfinite(ac["stage2_B"]) else None),
                     "ratio_joint_over_median": ac["joint"] / q[2] if len(a) else np.nan,
                     "ratio_stage2_over_median": ac["stage2"] / q[2] if len(a) else np.nan,
                     "joint_percentile_in_heads": float(stats.percentileofscore(a, ac["joint"])) if len(a) else np.nan,
                     "stage2_percentile_in_heads": float(stats.percentileofscore(a, ac["stage2"])) if len(a) else np.nan,
                     "kargere_r2_m2": ANALYSIS["kargere_r2_m2"] if site == "GM" else np.nan})
    return pd.DataFrame(rows)


def fits_6b(H: pd.DataFrame, slope_col: str, area_col: str, n_boot: int, seed: int, per_tile: bool = True) -> tuple[list[dict], dict]:
    """Per-tile and pooled (per site, tile intercepts) fits of log10 A on log10 S with bootstrap intervals."""
    rows: list[dict] = []
    boots: dict = {}
    for (sm, site), s in H.groupby(["smoothing", "site"]):
        s = s[np.isfinite(s[slope_col]) & (s[slope_col] > 0) & np.isfinite(s[area_col]) & (s[area_col] > 0)]
        x, y, g = np.log10(s[slope_col].to_numpy()), np.log10(s[area_col].to_numpy()), s.tile.to_numpy()
        base = {"smoothing": sm, "site": site, "slope": slope_col, "routing": area_col}
        groups = []
        if per_tile:
            for t in sorted(np.unique(g)):
                m = g == t
                f = fit_loglog(x[m], y[m])
                rec = {**base, "scope": f"{site}{t}", "n": int(m.sum())}
                if m.sum() >= MIN_TILE_FIT and np.isfinite(f["b_ols"]):
                    bt = bootstrap_fit(x[m], y[m], g[m], n_boot, np.random.default_rng([seed, int(t), zlib.crc32(str(site).encode())]))
                    rec.update(_summ(f["b_ols"], f["b_rma"], f["r2"], bt))
                    rec["a_ols"] = f["a_ols"]
                else:
                    rec.update(b_ols=f["b_ols"], b_rma=f["b_rma"], r2=f["r2"], note=f"n < {MIN_TILE_FIT}: no fit reported")
                rows.append(rec)
        keep = np.isin(g, [t for t in np.unique(g) if (g == t).sum() >= 2])
        groups = list(np.unique(g[keep]))
        fp = fit_pooled(x[keep], y[keep], g[keep])
        bt = bootstrap_fit(x[keep], y[keep], g[keep], n_boot, np.random.default_rng([seed, 99, zlib.crc32(f"{site}{sm}{slope_col}{area_col}".encode())]))
        rec = {**base, "scope": f"{site} pooled", "n": int(keep.sum()), "n_tiles": len(groups)}
        rec.update(_summ(fp["b_ols"], fp["b_rma"], fp["r2_within"], bt))
        rows.append(rec)
        boots[(sm, site)] = {"fit": fp, "boot": bt, "x": x[keep], "y": y[keep], "g": g[keep]}
    return rows, boots


def _summ(b_ols: float, b_rma: float, r2: float, bt: dict) -> dict:
    lo, hi = ci(bt["b_ols"], ANALYSIS["ci_percentiles"])
    rlo, rhi = ci(bt["b_rma"], ANALYSIS["ci_percentiles"])
    out: dict[str, Any] = {"b_ols": b_ols, "b_ols_lo": lo, "b_ols_hi": hi, "b_rma": b_rma, "b_rma_lo": rlo, "b_rma_hi": rhi, "r2": r2,
           "n_boot_valid": int(np.isfinite(bt["b_ols"]).sum()),
           "frac_boot_rma_positive": float(np.nanmean(bt["b_rma"] > 0)) if np.isfinite(bt["b_rma"]).any() else np.nan}
    for ref in ANALYSIS["reference_exponents"]:
        out[f"ols_ci_includes_{ref:g}"] = includes(lo, hi, ref)
        out[f"rma_ci_includes_{ref:g}"] = includes(rlo, rhi, ref)
    return out


def table_as2(H: pd.DataFrame, acr: dict[str, dict], col: str = "AS2_m2", slope_col: str = "S_up_10") -> pd.DataFrame:
    lo, hi = ANALYSIS["md_envelope_AS2_m2"]
    rows = []
    for (sm, site), s0 in H.groupby(["smoothing", "site"]):
        for scope, s in [(f"{site} pooled", s0)] + [(f"{site}{t}", st) for t, st in s0.groupby("tile")]:
            v = s[col].to_numpy()
            v = v[np.isfinite(v)]
            S = s[slope_col].to_numpy()
            S = S[np.isfinite(S)]
            q = np.percentile(v, [5, 25, 50, 75, 95]) if len(v) else [np.nan] * 5
            rec: dict[str, Any] = {"smoothing": sm, "site": site, "scope": scope, "n": len(v), "AS2_p05": q[0], "AS2_q1": q[1], "AS2_median": q[2],
                   "AS2_q3": q[3], "AS2_p95": q[4], "frac_below_500": float(np.mean(v < lo)) if len(v) else np.nan,
                   "frac_in_envelope": float(np.mean((v >= lo) & (v <= hi))) if len(v) else np.nan,
                   "frac_above_4000": float(np.mean(v > hi)) if len(v) else np.nan,
                   "S_median": float(np.median(S)) if len(S) else np.nan,
                   "S_q1": float(np.percentile(S, 25)) if len(S) else np.nan, "S_q3": float(np.percentile(S, 75)) if len(S) else np.nan}
            if not scope.endswith("pooled"):
                ac = acr[scope]
                for st in ("stage2", "joint"):
                    a = ac[st]
                    rec[f"A_crit_{st}"] = a
                    rec[f"A_crit_{st}_x_Smed2"] = a * rec["S_median"] ** 2
                    rec[f"A_crit_{st}_x_Smed2_in_envelope"] = bool(lo <= a * rec["S_median"] ** 2 <= hi) if np.isfinite(rec["S_median"]) else None
                    # slopes at which A_crit sits on the envelope: S = sqrt(AS2 / A_crit)
                    rec[f"S_env_lo_at_{st}"] = float(np.sqrt(lo / a))
                    rec[f"S_env_hi_at_{st}"] = float(np.sqrt(hi / a))
            rows.append(rec)
    return pd.DataFrame(rows)


def predict_6c(boots: dict, H: pd.DataFrame, acr: dict[str, dict], slope_col: str) -> pd.DataFrame:
    rows = []
    for (sm, site), B in boots.items():
        fp, bt = B["fit"], B["boot"]
        for k, t in enumerate(bt["keys"]):
            s = H[(H.smoothing == sm) & (H.site == site) & (H.tile == t)]
            Smed = float(np.nanmedian(s[slope_col]))
            mx, my = fp["means"][t]
            lp = my + fp["b_ols"] * (np.log10(Smed) - mx)
            lpb = bt["mean_y"][:, k] + bt["b_ols"] * (np.log10(Smed) - bt["mean_x"][:, k])
            plo, phi = ci(lpb)
            ac = acr[f"{site}{t}"]
            pred = 10**lp
            rows.append({"smoothing": sm, "site": site, "tile": t, "S_median": Smed, "A_pred_m2": pred, "A_pred_lo": 10**plo, "A_pred_hi": 10**phi,
                         "A_crit_stage2": ac["stage2"], "A_crit_joint": ac["joint"],
                         "pred_in_bracket": bool(min(ac["stage2"], ac["joint"]) <= pred <= max(ac["stage2"], ac["joint"])),
                         "ratio_joint_over_pred": ac["joint"] / pred})
    return pd.DataFrame(rows)


def ap5_6e(H: pd.DataFrame, acr: dict[str, dict]) -> dict[str, Any]:
    ap = H[H.site == "AP"]
    a5 = ap[ap.tile == 5].A_D8_m2.to_numpy()
    a14 = ap[ap.tile != 5].A_D8_m2.to_numpy()
    ac = acr["AP5"]
    out: dict[str, Any] = {"n_AP5": len(a5), "AP5_areas_m2": sorted(float(v) for v in a5), "A_crit": {k: ac[k] for k in STAGES},
                           "A_crit_percentile_in_AP5_heads": {k: float(stats.percentileofscore(a5, ac[k])) for k in STAGES if len(a5)},
                           "AP5_median": float(np.median(a5)) if len(a5) else None, "AP1_4_median": float(np.median(a14)) if len(a14) else None,
                           "n_AP1_4": len(a14)}
    if len(a5) and len(a14):
        mw = stats.mannwhitneyu(a5, a14, alternative="two-sided")
        ks = stats.ks_2samp(a5, a14)
        out.update(mannwhitney_p=float(mw.pvalue), ks_D=float(ks.statistic), ks_p=float(ks.pvalue))
    return out


def ap_lit_6f(H: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ap = H[H.site == "AP"]
    for scope, s in [("AP pooled", ap)] + [(f"AP{t}", st) for t, st in ap.groupby("tile")]:
        a = s.A_D8_m2.to_numpy()
        as2 = s.AS2_m2.to_numpy()
        as2 = as2[np.isfinite(as2)]
        rows.append({"scope": scope, "n": len(a), "mean_A_m2": float(a.mean()), "median_A_m2": float(np.median(a)),
                     "A_q1": float(np.percentile(a, 25)), "A_q3": float(np.percentile(a, 75)), "A_min": float(a.min()), "A_max": float(a.max()),
                     "mean_AS2_m2": float(as2.mean()) if len(as2) else np.nan, "median_AS2_m2": float(np.median(as2)) if len(as2) else np.nan,
                     "ratio_mean_A_over_MidBailey_413": float(a.mean() / 413.0), "ratio_mean_A_over_IndianCreek_1035": float(a.mean() / 1035.0)})
    for k, v in CLUBB_OHIO.items():
        rows.append({"scope": f"Clubb 2014 {k}", "n": v["n_heads"], "mean_A_m2": v["mean_A_m2"], "mean_AS2_m2": v["mean_AS2_m2"]})
    return pd.DataFrame(rows)


def verdicts(t6a: pd.DataFrame, fits: pd.DataFrame, n_by_tile: pd.Series) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for site in ("AP", "GM"):
        n = n_by_tile.loc[site] if site in n_by_tile.index.get_level_values(0) else pd.Series(dtype=int)
        n_ok = int((n >= 10).sum())
        valid = n_ok >= 4
        v: dict[str, Any] = {"n_hollow_heads_by_tile": {f"{site}{t}": int(k) for t, k in n.items()}, "n_tiles_with_ge10": n_ok,
                             "method_match_criterion": "not applicable: Method D (DrEICH) did not run (binary aborted on every tile)",
                             "extraction_valid": valid}
        a = t6a[t6a.site == site]
        n_ov = int(a.bracket_overlaps_IQR.fillna(False).astype(bool).sum())
        if n_ov >= 4:
            pv = "A_crit falls where independently extracted heads are"
        elif n_ov == 3:
            pv = "mixed"
        else:
            med = float(np.median(a.ratio_joint_over_median))
            pv = f"systematic offset (median joint A_crit / median head A = {med:.3g}; " + ("A_crit larger" if med > 1 else "A_crit smaller") + ")"
        v.update(n_tiles_bracket_overlaps_IQR=n_ov, tiles_overlap=[f"{site}{t}" for t in a[a.bracket_overlaps_IQR.fillna(False).astype(bool)].tile],
                 placement_verdict=pv)
        p = fits[fits.scope == f"{site} pooled"].iloc[0]
        if not valid:
            v["area_slope_verdict"] = "extraction failed at this site (6b-6d not judged)"
        elif p.b_ols < 0 and p.b_ols_hi < 0:
            v["area_slope_verdict"] = (f"inverse relation present (OLS b = {p.b_ols:.3g} [{p.b_ols_lo:.3g}, {p.b_ols_hi:.3g}]; "
                                       f"includes -2: {p['ols_ci_includes_-2']}, includes -1.65: {p['ols_ci_includes_-1.65']})")
        elif includes(p.b_ols_lo, p.b_ols_hi, 0.0):
            v["area_slope_verdict"] = f"no detectable relation (OLS b = {p.b_ols:.3g} [{p.b_ols_lo:.3g}, {p.b_ols_hi:.3g}])"
        else:
            v["area_slope_verdict"] = f"positive relation (OLS b = {p.b_ols:.3g} [{p.b_ols_lo:.3g}, {p.b_ols_hi:.3g}]); not an inverse relation"
        v["relation_6c_applies"] = bool(valid and not includes(p.b_ols_lo, p.b_ols_hi, 0.0))
        out[site] = v
    return out


# ----------------------------------------------------------------------------- figures (section 8)

def _save(fig, out: Path, df: pd.DataFrame) -> None:
    fig.savefig(out, dpi=170)
    plt.close(fig)
    df.to_csv(out.with_suffix(".csv"), index=False)


def fig_head_areas(H: pd.DataFrame, t6a: pd.DataFrame, site: str, out: Path, set_label: str) -> None:
    s = H[H.site == site]
    a = t6a[t6a.site == site].set_index("tile")
    tiles = sorted(a.index)
    fig, ax = plt.subplots(figsize=(7.0, 4.6), facecolor="white")
    data = [s[s.tile == t].A_D8_m2.to_numpy() for t in tiles]
    pos = np.arange(len(tiles), dtype=float)
    ax.boxplot(data, positions=pos, widths=0.42, showfliers=False, patch_artist=True,
               boxprops=dict(facecolor="#dbe8f7", edgecolor=C_MUTED), medianprops=dict(color=C_TEXT, lw=1.6),
               whiskerprops=dict(color=C_MUTED), capprops=dict(color=C_MUTED))
    for k, t in enumerate(tiles):
        v = data[k]
        ax.plot(pos[k] + np.random.default_rng(k).uniform(-0.12, 0.12, len(v)), v, "o", ms=3, color=C_TILE[k], alpha=0.75, ls="none")
        r = a.loc[t]
        xb = pos[k] + 0.36
        ax.plot([xb, xb], [r.A_crit_stage2, r.A_crit_joint], color=C_TEXT, lw=2.2, solid_capstyle="round")
        ax.plot(xb, r.A_crit_joint, "o", ms=8, color=C_TEXT, mec="white", mew=1.0)
        ax.plot(xb, r.A_crit_stage2, "o", ms=7, color="white", mec=C_TEXT, mew=1.2)
        ax.plot([xb + 0.07] * 2, [r["joint_0.05_min"], r["joint_0.05_max"]], color=C_MUTED, lw=1.0)
    if site == "GM":
        ax.axhline(ANALYSIS["kargere_r2_m2"], color="#4a3aa7", lw=1.2, ls=(0, (4, 3)))
        ax.text(pos[-1] + 0.55, ANALYSIS["kargere_r2_m2"], "Kargère r² = 62² m²", fontsize=7, color=C_MUTED, va="bottom", ha="right")
    ax.set_yscale("log")
    ax.set_xticks(pos, [f"{site}{t}\n(n = {len(d)})" for t, d in zip(tiles, data, strict=False)])
    ax.set_xlim(-0.6, len(tiles) - 0.2)
    ax.set_ylabel("drainage area A_D8 at head (m²)", fontsize=9)
    ax.set_title(f"{site}: hollow-class head areas ({set_label})\nbar = A_crit bracket (open = stage 2, filled = linear joint); "
                 "thin bar = joint 5 % interval", fontsize=8, loc="left", color=C_TEXT)
    _style(ax)
    fig.tight_layout()
    rows = [dict(layer="head", site=site, tile=r.tile, head_id=r.head_id, A_D8_m2=r.A_D8_m2) for r in s.itertuples()]
    rows += [dict(layer="bracket", site=site, tile=t, A_crit_stage2=a.loc[t].A_crit_stage2, A_crit_joint=a.loc[t].A_crit_joint,
                  joint_05_min=a.loc[t]["joint_0.05_min"], joint_05_max=a.loc[t]["joint_0.05_max"]) for t in tiles]
    if site == "GM":
        rows.append(dict(layer="kargere_r2", site=site, A_D8_m2=ANALYSIS["kargere_r2_m2"]))
    _save(fig, out, pd.DataFrame(rows))


def fig_area_slope(H: pd.DataFrame, boots: dict, t6a: pd.DataFrame, site: str, sm: str, out: Path, slope_col: str = "S_up_10") -> None:
    s = H[(H.site == site) & np.isfinite(H[slope_col])]
    B = boots[(sm, site)]
    fp, bt = B["fit"], B["boot"]
    fig, ax = plt.subplots(figsize=(6.4, 5.0), facecolor="white")
    Sg = np.logspace(np.log10(0.05), np.log10(1.5), 50)
    lo, hi = ANALYSIS["md_envelope_AS2_m2"]
    ax.fill_between(Sg, lo / Sg**2, hi / Sg**2, color="#efeeea", zorder=0, label="M&D (1992) envelope, AS² = 500-4000 m²")
    tiles = sorted(s.tile.unique())
    for k, t in enumerate(tiles):
        st = s[s.tile == t]
        ax.plot(st[slope_col], st.A_D8_m2, "o", ms=5, color=C_TILE[k], mec="white", mew=0.6, ls="none", label=f"{site}{t} (n = {len(st)})")
    xm, ym = float(np.mean(B["x"])), float(np.mean(B["y"]))
    xs = np.linspace(B["x"].min(), B["x"].max(), 40)
    yl = ym + fp["b_ols"] * (xs - xm)
    band = np.array([ym + b * (xs - xm) for b in bt["b_ols"][np.isfinite(bt["b_ols"])]])
    blo, bhi = np.percentile(band, [2.5, 97.5], axis=0)
    ax.fill_between(10**xs, 10**blo, 10**bhi, color="#2a78d6", alpha=0.15, lw=0)
    ax.plot(10**xs, 10**yl, color=C_TEXT, lw=2, label=f"pooled OLS (tile intercepts, mean shown), b = {fp['b_ols']:.2f}")
    for ref, ls in ((-2.0, (0, (5, 3))), (-1.65, (0, (1, 2)))):
        ax.plot(10**xs, 10 ** (ym + ref * (xs - xm)), color=C_MUTED, lw=1.2, ls=ls, label=f"slope {ref:g} through the centroid")
    a = t6a[t6a.site == site].set_index("tile")
    br = []
    for k, t in enumerate(tiles):
        Sm = float(np.nanmedian(s[s.tile == t][slope_col]))
        r = a.loc[t]
        ax.plot([Sm, Sm], [r.A_crit_stage2, r.A_crit_joint], color=C_TILE[k], lw=3, alpha=0.9, solid_capstyle="round")
        ax.plot(Sm, r.A_crit_joint, "s", ms=7, color=C_TILE[k], mec=C_TEXT, mew=0.8)
        br.append(dict(layer="bracket", site=site, tile=t, S_median=Sm, A_crit_stage2=r.A_crit_stage2, A_crit_joint=r.A_crit_joint))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.08, 1.3)
    ax.set_ylim(0.5, 2e5)
    ax.set_xlabel(f"{slope_col} at head (m/m)", fontsize=9)
    ax.set_ylabel("A_D8 at head (m²)", fontsize=9)
    ax.set_title(f"{site}, {sm}: area-slope at hollow heads\nbars = A_crit bracket at tile median slope (square = joint)",
                 fontsize=8, loc="left", color=C_TEXT)
    ax.legend(fontsize=6.5, loc="lower left", frameon=True, framealpha=0.9)
    _style(ax)
    fig.tight_layout()
    rows = [dict(layer="head", site=site, tile=r.tile, head_id=r.head_id, S=getattr(r, slope_col), A_D8_m2=r.A_D8_m2) for r in s.itertuples()]
    rows += [dict(layer="fit", site=site, S=10**x0, A_fit=10**y0, A_band_lo=10**l0, A_band_hi=10**h0) for x0, y0, l0, h0 in zip(xs, yl, blo, bhi, strict=False)]
    rows += br
    _save(fig, out, pd.DataFrame(rows))


def fig_spatial(corr: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), facecolor="white", sharey=True)
    for ax, site in zip(axes, ("AP", "GM"), strict=False):
        c = corr[(corr.site == site) & corr.stage.isin(["joint", "stage2"])]
        tiles = sorted(c.tile.unique())
        w = 0.19
        series = [("joint", "frac_heads_with_partition_head", "#2a78d6", "heads with partition head ≤ 20 m (joint)"),
                  ("joint", "frac_partition_heads_with_head", "#8fb8ea", "partition heads (joint) with head ≤ 20 m"),
                  ("stage2", "frac_heads_with_partition_head", "#eb6834", "heads with partition head ≤ 20 m (stage 2)"),
                  ("stage2", "frac_partition_heads_with_head", "#f3b08f", "partition heads (stage 2) with head ≤ 20 m")]
        for k, (st, col, colr, lab) in enumerate(series):
            v = [float(c[(c.tile == t) & (c.stage == st)][col].iloc[0]) for t in tiles]
            ax.bar(np.arange(len(tiles)) + (k - 1.5) * w, v, w * 0.9, color=colr, label=lab)
        for k, st in enumerate(("joint", "stage2")):
            ch = [float(c[(c.tile == t) & (c.stage == st)].chance_frac_random_point_with_partition_head.iloc[0]) for t in tiles]
            xo = -1.5 * w if st == "joint" else 0.5 * w
            ax.plot(np.arange(len(tiles)) + xo, ch, "_", ms=14, mew=2, color=C_TEXT, label="chance (random point)" if k == 0 else None)
        ax.set_xticks(np.arange(len(tiles)), [f"{site}{t}" for t in tiles])
        ax.set_ylim(0, 1)
        ax.set_title(f"{site}: 20 m correspondence", fontsize=9, loc="left", color=C_TEXT)
        _style(ax)
    axes[0].set_ylabel("fraction", fontsize=9)
    h, lab = axes[1].get_legend_handles_labels()
    fig.legend(h, lab, fontsize=7, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    _save(fig, out, corr)


def fig_sensitivity(sens: pd.DataFrame, out: Path) -> None:
    groups = [("smoothing set", [("OWF", "S_up_10", "A_D8_m2"), ("OWF+DT0.5", "S_up_10", "A_D8_m2"), ("OWF+DT2", "S_up_10", "A_D8_m2"), ("OWF+DT5", "S_up_10", "A_D8_m2")],
               lambda r: r[0]),
              ("routing", [(PRIMARY_SET, "S_up_10", "A_D8_m2"), (PRIMARY_SET, "S_up_10", "A_dinf_m2")], lambda r: "D8" if r[2] == "A_D8_m2" else "D-inf"),
              ("slope window", [(PRIMARY_SET, k, "A_D8_m2") for k in SLOPES], lambda r: r[1])]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4), facecolor="white", sharey="row")
    rows = []
    for c, (name, variants, lab) in enumerate(groups):
        for site, dx in (("AP", -0.1), ("GM", 0.1)):
            for k, v in enumerate(variants):
                r = sens[(sens.site == site) & (sens.smoothing == v[0]) & (sens.slope == v[1]) & (sens.routing == v[2])]
                if not len(r):
                    continue
                rr0 = r.iloc[0]
                axes[0, c].errorbar(k + dx, rr0.b_ols, yerr=[[rr0.b_ols - rr0.b_ols_lo], [rr0.b_ols_hi - rr0.b_ols]], fmt="o", ms=6, color=C_SITE[site],
                                    capsize=3, label=site if k == 0 else None)
                axes[1, c].plot(k + dx, rr0.median_A, "o", ms=6, color=C_SITE[site], label=site if k == 0 else None)
                rows.append({"panel": name, "variant": lab(v), "site": site, **{kk: rr0[kk] for kk in ("smoothing", "slope", "routing", "n", "b_ols", "b_ols_lo", "b_ols_hi", "b_rma", "b_rma_lo", "b_rma_hi", "r2", "median_A")}})
        for rr in (0, 1):
            axes[rr, c].set_xticks(range(len(variants)), [lab(v) for v in variants], fontsize=7)
            _style(axes[rr, c])
        for ref in (-2.0, -1.65, 0.0):
            axes[0, c].axhline(ref, color=C_MUTED, lw=0.8, ls=(0, (3, 3)) if ref else "-")
        axes[1, c].set_yscale("log")
        axes[0, c].set_title(f"vs {name}", fontsize=9, loc="left", color=C_TEXT)
    axes[0, 0].set_ylabel("pooled OLS exponent b (95 % CI)", fontsize=8)
    axes[1, 0].set_ylabel("median head A (m²)", fontsize=8)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"Sensitivity (defaults: {PRIMARY_SET}, S_up_10, D8). Curvature-threshold sensitivity unavailable: only kt = 0.1 heads were saved.",
                 fontsize=8.5, x=0.01, ha="left", color=C_TEXT)
    fig.tight_layout()
    _save(fig, out, pd.DataFrame(rows))


# ----------------------------------------------------------------------------- driver

def analyse(out_dir: Path, heads_csv: Path, tiles: list[str], n_boot: int | None = None) -> dict[str, Any]:
    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    (fig_dir / "qc_maps_fitset").mkdir(parents=True, exist_ok=True)
    n_boot = int(n_boot or ANALYSIS["bootstrap_n"])
    seed = int(ANALYSIS["random_seed"])
    rng = np.random.default_rng(seed)
    t_start = time.time()
    fit, rinfo = load_fit_heads(heads_csv)
    keys = [(s[:2], int(s[2:])) for s in tiles]
    acr = {f"{s}{t}": acrit_phase6(f"{s}{t}") for s, t in keys}
    pd.DataFrame(acr.values()).to_csv(out_dir / "acrit_bracket_phase6.csv", index=False)

    Hs, Ps, Cs, infos = [], [], [], {}
    for s, t in keys:
        r = measure_tile(s, t, fit, acr[f"{s}{t}"], rng)
        Hs.append(r["heads"])
        Ps.append(r["partition"])
        Cs.append(r["corr"])
        infos[f"{s}{t}"] = r["info"]
        # QC map of the primary fit set with the partition heads at the joint A_crit (section 8.4)
        hp = r["heads"][r["heads"].smoothing == PRIMARY_SET].copy()
        hp["retained"] = True
        part = r["partition"][r["partition"].stage == "joint"]
        try:
            fig_qc_map(r["tile"], hp, part, QC, EXTRACTION, fig_dir / "qc_maps_fitset" / f"{s}{t}.png")
            pd.concat([hp.assign(layer="head")[["layer", "head_id", "x", "y", "A_D8_m2", "head_class"]],
                       part.assign(layer="partition_head_joint")[["layer", "x", "y", "area_D8_m2"]]]).to_csv(
                fig_dir / "qc_maps_fitset" / f"{s}{t}.csv", index=False)
        except Exception as exc:  # noqa: BLE001
            logger.warning("QC map %s%d skipped: %s", s, t, exc)
    H = pd.concat(Hs, ignore_index=True)
    P = pd.concat(Ps, ignore_index=True)
    C = pd.concat(Cs, ignore_index=True)
    H.to_csv(out_dir / "heads_measured.csv", index=False)
    P.to_csv(out_dir / "partition_heads.csv", index=False)
    C.to_csv(out_dir / "spatial_correspondence_6d.csv", index=False)

    counts = H.groupby(["smoothing", "site", "tile"]).agg(n_heads=("head_id", "size"), n_S_up_10=("S_up_10", lambda v: int(np.isfinite(v).sum())),
                                                         n_truncated_on_tile=("truncated_on_tile", "sum")).reset_index()
    counts.to_csv(out_dir / "fit_set_counts.csv", index=False)

    t6a = pd.concat([table_6a(H, acr, "A_D8_m2"), table_6a(H, acr, "A_dinf_m2")], ignore_index=True)
    t6a.to_csv(out_dir / "head_areas_6a.csv", index=False)

    fit_rows, sens_rows = [], []
    boots_primary: dict = {}
    for sc in SLOPES:
        for rname, rcol in ROUTINGS.items():
            rows, boots = fits_6b(H, sc, rcol, n_boot, seed, per_tile=True)
            fit_rows += rows
            if sc == "S_up_10" and rname == "D8":
                boots_primary = boots
    F = pd.DataFrame(fit_rows)
    F.to_csv(out_dir / "area_slope_fits_6b_all.csv", index=False)
    pooled = F[F.scope.str.endswith("pooled")].copy()
    med = H.groupby(["smoothing", "site"]).agg(median_A_D8=("A_D8_m2", "median"), median_A_dinf=("A_dinf_m2", "median")).reset_index()
    for _, pr in pooled.iterrows():
        mm = med[(med.smoothing == pr.smoothing) & (med.site == pr.site)].iloc[0]
        sens_rows.append({**pr.to_dict(), "median_A": mm.median_A_D8 if pr.routing == "A_D8_m2" else mm.median_A_dinf})
    sens = pd.DataFrame(sens_rows)
    sens.to_csv(out_dir / "sensitivity.csv", index=False)
    prim = F[(F.slope == "S_up_10") & (F.routing == "A_D8_m2")]
    prim.to_csv(out_dir / "area_slope_fits_6b.csv", index=False)

    as2 = table_as2(H, acr)
    as2.to_csv(out_dir / "AS2_envelope.csv", index=False)

    # verdicts and 6c-6f on the primary set (and on GM OWF as the GM sensitivity)
    out_v: dict[str, Any] = {}
    p6c = []
    for sm in (PRIMARY_SET, "OWF"):
        Hm = H[H.smoothing == sm]
        n_by = Hm.groupby(["site", "tile"]).size()
        v = verdicts(t6a[(t6a.smoothing == sm) & (t6a.routing == "A_D8_m2")], prim[prim.smoothing == sm], n_by)
        out_v[sm] = v
        for site in ("AP", "GM"):
            if v[site]["relation_6c_applies"] and (sm, site) in boots_primary:
                p6c.append(predict_6c({(sm, site): boots_primary[(sm, site)]}, Hm, acr, "S_up_10"))
    if p6c:
        pd.concat(p6c, ignore_index=True).to_csv(out_dir / "prediction_6c.csv", index=False)
    Hp = H[H.smoothing == PRIMARY_SET]
    e = ap5_6e(Hp, acr) if "AP5" in acr else {}
    f6 = ap_lit_6f(Hp) if (Hp.site == "AP").any() else pd.DataFrame()
    f6.to_csv(out_dir / "ap_literature_6f.csv", index=False)

    # figures
    for sm in (PRIMARY_SET, "OWF"):
        Hm = H[H.smoothing == sm]
        tm = t6a[(t6a.smoothing == sm) & (t6a.routing == "A_D8_m2")]
        tag = sm.replace("+", "_")
        for site in ("AP", "GM"):
            if not (Hm.site == site).any():
                continue
            fig_head_areas(Hm, tm, site, fig_dir / f"fig1_head_areas_{site}_{tag}.png", sm)
            if (sm, site) in boots_primary:
                fig_area_slope(Hm, boots_primary, tm, site, sm, fig_dir / f"fig2_area_slope_{site}_{tag}.png")
    fig_spatial(C[C.smoothing == PRIMARY_SET], fig_dir / f"fig3_spatial_correspondence_{PRIMARY_SET.replace('+', '_')}.png")
    fig_spatial(C[C.smoothing == "OWF"], fig_dir / "fig3_spatial_correspondence_OWF.png")
    fig_sensitivity(sens, fig_dir / "fig5_sensitivity.png")

    summary: dict[str, Any] = {"heads": rinfo, "primary_set": PRIMARY_SET, "n_boot": n_boot, "seed": seed, "A_crit": acr, "tiles": infos,
                               "verdicts": out_v, "ap5_6e": e, "elapsed_s": round(time.time() - t_start, 1),
                               "deviations": ["curvature-threshold sensitivity not run: the extract run saved measured heads at kt = 0.1 only",
                                              "Method D did not run, so the 60 % method-match criterion of section 7 is not applicable",
                                              "D-inf area is computed on the 1 km tile only (stored D8 area came from the larger DEM); see truncated_on_tile"]}
    with open(out_dir / "summary.json", "w") as fh:
        json.dump(_json_safe(summary), fh, indent=1)
    return summary


def _json_safe(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, np.ndarray):
        return _json_safe(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def run_heads_analyse(tiles: list[str], heads_run: str = HEADS_RUN_DEFAULT, n_boot: int | None = None, run_id: str | None = None,
                      results_root=None):
    """``lemcvt phase7 heads-analyse``: the analyse phase into a new run directory."""
    from lemcvt.analyses.phase7 import _snapshot
    from lemcvt.results import RunRecorder
    from lemcvt.results import results_root as _root

    root = Path(results_root) if results_root is not None else _root()
    heads_csv = root / heads_run / "head_extraction" / "heads_extracted.csv"
    if not heads_csv.exists():
        raise FileNotFoundError(heads_csv)
    rec = RunRecorder("heads_analyse", f"{len(tiles)}tiles", run_id=run_id, root=results_root)
    summary = analyse(rec.dir / "head_analysis", heads_csv, list(tiles), n_boot=n_boot)
    rec.add_params("heads_run", heads_run)
    rec.add_params("summary", _json_safe(summary))
    rec.finalize({"tiles": _snapshot(list(tiles)), "heads_run": heads_run, "analysis": ANALYSIS})
    return rec
