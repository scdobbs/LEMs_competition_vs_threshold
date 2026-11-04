"""Head-quality diagnostics requested after the first run (2026-11-04):

    python -m acrit_slope.qc_diagnostics

1. Redraws figures/qc_maps/<tile>.png with log10 drainage area overlaid on the
   hillshade, kept heads marked by class, excluded heads by reason.
2. Classifies every kept head (heads_qc.csv):
   - ``dist_to_trunk_m``: distance to the nearest pixel with A >= trunk_area_m2
     (1e5 m², the manuscript's channel-slope floor);
   - ``dist_to_10acrit_m``: distance to the nearest pixel with A >= 10 A_crit;
   - ``S_0_10``, ``S_10_20``: slope of the upslope path over 0-10 and 10-20 m;
     ``profile_shape`` = plateau (S_up_20 < plateau_slope_max), steepening
     (S_10_20 / S_0_10 > 1.25), flattening (< 0.8) or uniform;
   - ``head_class``: trunk_floor (within trunk_buffer_m of the trunk and
     S_up_10 < floor_slope_max), plateau, or hollow (everything else).
3. Per-tile class counts (qc_class_summary.csv) and the variant-1 / variant-2
   fits repeated on hollow-class heads only (fits_hollow_only.json).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt

from width_analysis.network import FlowGraph
from width_analysis.tiles import REPO_ROOT, load_tile

from . import figures
from .config import A_CRIT_M2, CONFIG, TILE_KEYS
from .fits import ci, fit_common_slope, fit_ols, two_level_bootstrap
from .run import _json_safe

QC = {
    "trunk_area_m2": 1.0e5,
    "trunk_buffer_m": 30.0,
    "floor_slope_max": 0.10,
    "plateau_slope_max": 0.05,
    "steepening_ratio": 1.25,
    "flattening_ratio": 0.80,
    "area_overlay_min_m2": 1.0e3,
}


def _dist_grid(mask: np.ndarray, dx: float, dy: float) -> np.ndarray:
    if not mask.any():
        return np.full(mask.shape, np.inf)
    return distance_transform_edt(~mask, sampling=(dy, dx))


def classify(heads: pd.DataFrame, tile, qc: dict) -> pd.DataFrame:
    h = heads.copy()
    d_trunk = _dist_grid(tile.area >= qc["trunk_area_m2"], tile.dx, tile.dy)
    d_10 = _dist_grid(tile.area >= 10 * h.a_crit_m2.iloc[0], tile.dx, tile.dy)
    h["dist_to_trunk_m"] = d_trunk[h.row.to_numpy(), h.col.to_numpy()]
    h["dist_to_10acrit_m"] = d_10[h.row.to_numpy(), h.col.to_numpy()]
    h["S_0_10"] = h["S_up_10"]
    h["S_10_20"] = 2 * h["S_up_20"] - h["S_up_10"]          # (z20 - z10) / 10
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = h["S_10_20"] / h["S_0_10"]
    shape = np.where(h["S_up_20"] < qc["plateau_slope_max"], "plateau",
             np.where(ratio > qc["steepening_ratio"], "steepening",
             np.where(ratio < qc["flattening_ratio"], "flattening", "uniform")))
    h["profile_shape"] = shape
    floor = (h["dist_to_trunk_m"] <= qc["trunk_buffer_m"]) & (h["S_up_10"] < qc["floor_slope_max"])
    h["head_class"] = np.where(floor, "trunk_floor", np.where(shape == "plateau", "plateau", "hollow"))
    return h


def main(argv=None):
    out_dir = REPO_ROOT / CONFIG["output_dir"]
    heads = pd.read_csv(out_dir / "heads.csv")
    rows, summ = [], []
    for site, t in TILE_KEYS:
        tile = load_tile(site, t)
        sub = heads[(heads.site == site) & (heads.tile == t)].copy()
        kept = classify(sub[sub.kept], tile, QC)
        sub = pd.concat([kept, sub[~sub.kept]])
        rows.append(sub)
        n = len(kept)
        rec = dict(site=site, tile=t, A_crit_m2=A_CRIT_M2[(site, t)], n_kept=n)
        for c in ("hollow", "trunk_floor", "plateau"):
            rec[f"n_{c}"] = int((kept.head_class == c).sum())
        for s in ("steepening", "uniform", "flattening", "plateau"):
            rec[f"n_shape_{s}"] = int((kept.profile_shape == s).sum())
        rec["n_within_30m_of_trunk"] = int((kept.dist_to_trunk_m <= QC["trunk_buffer_m"]).sum())
        rec["n_within_15m_of_10Acrit"] = int((kept.dist_to_10acrit_m <= 15).sum())
        hol = kept[kept.head_class == "hollow"]
        rec["S_up_10_median_all"] = kept.S_up_10.median()
        rec["S_up_10_median_hollow"] = hol.S_up_10.median() if len(hol) else np.nan
        summ.append(rec)
        figures.fig_qc_map_area(tile, sub, CONFIG, QC, out_dir / "figures" / "qc_maps" / f"{site}{t}.png")
        print(f"[{site}{t}] kept {n}: hollow {rec['n_hollow']}, trunk_floor {rec['n_trunk_floor']}, plateau {rec['n_plateau']}; "
              f"shapes steepening {rec['n_shape_steepening']} uniform {rec['n_shape_uniform']} flattening {rec['n_shape_flattening']}", flush=True)
        del tile
    hq = pd.concat(rows)
    hq.to_csv(out_dir / "heads_qc.csv", index=False)
    summary = pd.DataFrame(summ)
    summary.to_csv(out_dir / "qc_class_summary.csv", index=False)
    print(summary.to_string(), flush=True)

    # fits on hollow-class heads only
    rng = np.random.default_rng(CONFIG["random_seed"] + 1)
    keys = list(TILE_KEYS)
    logA = {k: float(np.log10(A_CRIT_M2[k])) for k in keys}
    site = {k: k[0] for k in keys}
    hs = {k: hq[(hq.site == k[0]) & (hq.tile == k[1]) & (hq.kept == True) & (hq.head_class == "hollow")].S_up_10.dropna().to_numpy() for k in keys}  # noqa: E712
    x = np.array([np.log10(np.median(hs[k])) for k in keys])
    y = np.array([logA[k] for k in keys])
    g = np.array([site[k] for k in keys])
    out = dict(qc=QC, n_hollow={f"{k[0]}{k[1]}": int(len(hs[k])) for k in keys},
               median_S_up_10_hollow={f"{k[0]}{k[1]}": float(10 ** v) for k, v in zip(keys, x)})
    f = fit_ols(x, y)
    b = two_level_bootstrap(keys, logA, site, hs, "OLS", CONFIG["bootstrap_n"], rng)
    out["v1_all_OLS"] = dict(b=f["b"], ci_b=ci(b["b"], CONFIG["ci_percentiles"]), r2=f["r2"], n=10)
    f = fit_common_slope(x, y, g)
    b = two_level_bootstrap(keys, logA, site, hs, "common_slope", CONFIG["bootstrap_n"], rng)
    out["v2_common_slope"] = dict(b=f["b"], ci_b=ci(b["b"], CONFIG["ci_percentiles"]), r2=f["r2"], intercepts=f["intercepts"])
    with open(out_dir / "fits_hollow_only.json", "w") as fh:
        json.dump(_json_safe(out), fh, indent=1)
    for k in ("v1_all_OLS", "v2_common_slope"):
        print(k, out[k], flush=True)


if __name__ == "__main__":
    main()
