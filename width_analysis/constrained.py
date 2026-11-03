"""Constrained bank pick, along-stream averaged cross-sections, point
classification, exceedance statistics and window sensitivity
(reports/directives/channel_width_constrained_directive.md).

Nothing here changes the Phase 3 measures in widths.py; it adds a second,
config-controlled pick that only accepts a convex break sitting within a
channel-scale window (relief 0.3-1.5 m, offset <= 6 m from the thalweg).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import beta

from .network import FlowGraph
from .tiles import Tile
from .transect import Transect, make_transect
from .widths import find_thalweg, shape_class, stage_label, width_at_stage

CLASSES = ("inner_channel", "wide_or_terrace", "unresolved_narrow", "ambiguous")


# ---------------------------------------------------------------------------
# side pick
# ---------------------------------------------------------------------------

def convex_breaks(s, z, i_thal: int, cfg: dict):
    """All candidate bank tops on each side, nearest first.

    A candidate is a local maximum of convexity (-d2z/ds2 of the Gaussian
    smoothed profile) above ``bankbreak_min_convexity_1_per_m`` whose mean
    slope from the thalweg is at least ``bankbreak_min_bank_slope``.
    Returns {"left": [(offset, relief, s)], "right": [...]} with relief from
    the raw profile.
    """
    step = s[1] - s[0]
    zs = gaussian_filter1d(z, cfg["bankbreak_smooth_sigma_m"] / step, mode="nearest")
    conv = -np.gradient(np.gradient(zs, step), step)
    n = len(s)
    pad = int(np.ceil(3.0 * cfg["bankbreak_smooth_sigma_m"] / step)) + 1
    out = {"left": [], "right": []}
    for side, rng in (("left", range(i_thal - 1, pad - 1, -1)), ("right", range(i_thal + 1, n - pad))):
        for k in rng:
            if conv[k] >= cfg["bankbreak_min_convexity_1_per_m"] and conv[k] >= conv[k - 1] and conv[k] >= conv[k + 1]:
                ds = abs(s[k] - s[i_thal])
                if ds > 0 and (zs[k] - zs[i_thal]) / ds >= cfg["bankbreak_min_bank_slope"]:
                    out[side].append((float(ds), float(z[k] - z[i_thal]), float(s[k])))
    return out


def pick_side(cands: list, ccfg: dict) -> dict:
    """Constrained pick on one side.

    ``bank``: first candidate (nearest the thalweg) with relief in
    [bank_min_relief_m, bank_max_relief_m] and offset <= bank_max_offset_m.
    ``none_within_window``: a candidate in the relief window exists only
    beyond the offset limit. ``no_break``: no candidate in the relief window.
    """
    lo, hi, off = ccfg["bank_min_relief_m"], ccfg["bank_max_relief_m"], ccfg["bank_max_offset_m"]
    beyond = None
    for (d, r, sk) in cands:                         # nearest first
        if lo <= r <= hi:
            if d <= off:
                return dict(side_class="bank", offset=d, relief=r, s=sk)
            if beyond is None:
                beyond = (d, r, sk)
    if beyond is not None:
        return dict(side_class="none_within_window", offset=beyond[0], relief=beyond[1], s=beyond[2])
    return dict(side_class="no_break", offset=np.nan, relief=np.nan, s=np.nan)


def classify(left: str, right: str, shape: str) -> str:
    if left == "bank" and right == "bank":
        return "inner_channel"
    sides = {left, right}
    if "none_within_window" in sides and sides <= {"bank", "none_within_window"}:
        return "wide_or_terrace"
    if left == "no_break" and right == "no_break" and shape == "v_shape":
        return "unresolved_narrow"
    return "ambiguous"


# ---------------------------------------------------------------------------
# one transect
# ---------------------------------------------------------------------------

def measure_transect(s, z, pixel_size: float, cfg: dict, ccfg: dict) -> dict:
    """Constrained measures on one cross-section (keys are per-transect columns)."""
    i_t = find_thalweg(s, z, cfg["thalweg_search_m"])
    sc, flat_w = shape_class(s, z, i_t, cfg)
    cands = convex_breaks(s, z, i_t, cfg)
    L = pick_side(cands["left"], ccfg)
    R = pick_side(cands["right"], ccfg)
    cls = classify(L["side_class"], R["side_class"], sc)
    out = dict(thalweg_s=float(s[i_t]), z_thalweg=float(z[i_t]), shape_class=sc, flat_span_m=flat_w,
               left_class=L["side_class"], right_class=R["side_class"], point_class=cls,
               s_bank_left=L["s"], s_bank_right=R["s"], relief_left_m=L["relief"], relief_right_m=R["relief"],
               offset_left_m=L["offset"], offset_right_m=R["offset"],
               n_breaks_left=len(cands["left"]), n_breaks_right=len(cands["right"]),
               W_bank=np.nan, W_stage_bank=np.nan, h_bank_mean=np.nan)
    if cls == "inner_channel":
        out["W_bank"] = float(R["s"] - L["s"])
        h = 0.5 * (L["relief"] + R["relief"])
        out["h_bank_mean"] = h
        out["W_stage_bank"] = float(width_at_stage(s, z, i_t, h, cfg["crest_drop_m"])[0])
    for hf in cfg["fixed_stages_m"]:
        out[f"W_stage_{stage_label(hf)}"] = float(width_at_stage(s, z, i_t, hf, cfg["crest_drop_m"])[0])
    return out


# ---------------------------------------------------------------------------
# five transects along the stream
# ---------------------------------------------------------------------------

@dataclass
class MultiTransect:
    offsets: list
    transects: list                 # Transect or None per offset
    rows: list = field(default_factory=list)
    cols: list = field(default_factory=list)


def pixel_at_offset(graph: FlowGraph, row: int, col: int, offset_m: float):
    """Pixel on the main stem at along-path distance |offset_m| (downstream if positive)."""
    if offset_m == 0:
        return row, col
    walk = graph.walk(row, col, abs(offset_m) + graph.de, "down" if offset_m > 0 else "up")
    if len(walk) < 2 or walk[-1][2] < abs(offset_m) - graph.de:
        return None
    k = int(np.argmin([abs(d - abs(offset_m)) for (_, _, d) in walk]))
    return walk[k][0], walk[k][1]


def multi_transect(tile: Tile, graph: FlowGraph, row: int, col: int, cfg: dict, ccfg: dict) -> MultiTransect:
    mt = MultiTransect(offsets=list(ccfg["transect_offsets_m"]), transects=[])
    for off in mt.offsets:
        rc = pixel_at_offset(graph, row, col, off)
        if rc is None:
            mt.transects.append(None)
            mt.rows.append(-1)
            mt.cols.append(-1)
            continue
        mt.transects.append(make_transect(tile, graph, rc[0], rc[1], cfg))
        mt.rows.append(rc[0])
        mt.cols.append(rc[1])
    return mt


def _median_or_nan(v):
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    return float(np.median(v)) if len(v) else np.nan


def aggregate(per_transect: list[dict], pixel_size: float, ccfg: dict) -> dict:
    """Point-level class, widths and flags from the per-transect measures (None entries = missing transect)."""
    valid = [m for m in per_transect if m is not None]
    n_t = len(valid)
    flags = []
    out = dict(n_transects=n_t)
    if n_t == 0:
        out.update(point_class="ambiguous", qc_flags="no_valid_transect")
        return out
    classes = [m["point_class"] for m in valid]
    cnt = Counter(classes).most_common()
    if len(cnt) > 1 and cnt[0][1] == cnt[1][1]:
        cls = "ambiguous"
        flags.append("class_tie")
    else:
        cls = cnt[0][0]
    n_inner = sum(c == "inner_channel" for c in classes)
    if max(n_inner, n_t - n_inner) < ccfg["min_agreeing_transects"]:
        flags.append("inconsistent_along_stream")
    out["point_class"] = cls
    out["n_inner_channel_transects"] = n_inner
    out["class_counts"] = ";".join(f"{k}:{v}" for k, v in sorted(Counter(classes).items()))
    out["left_class"] = Counter(m["left_class"] for m in valid).most_common(1)[0][0]
    out["right_class"] = Counter(m["right_class"] for m in valid).most_common(1)[0][0]
    out["shape_class"] = Counter(m["shape_class"] for m in valid).most_common(1)[0][0]
    wb = [m["W_bank"] for m in valid if np.isfinite(m["W_bank"])]
    out["W_bank"] = _median_or_nan(wb)
    out["W_bank_n_valid"] = len(wb)
    out["W_bank_spread"] = float(max(wb) - min(wb)) if len(wb) > 1 else (0.0 if wb else np.nan)
    out["W_stage_bank"] = _median_or_nan([m["W_stage_bank"] for m in valid])
    out["h_bank_mean"] = _median_or_nan([m["h_bank_mean"] for m in valid])
    out["relief_left_m"] = _median_or_nan([m["relief_left_m"] for m in valid if m["left_class"] == "bank"])
    out["relief_right_m"] = _median_or_nan([m["relief_right_m"] for m in valid if m["right_class"] == "bank"])
    out["ratio_bank_over_stage"] = (out["W_bank"] / out["W_stage_bank"]
                                    if np.isfinite(out["W_bank"]) and out["W_stage_bank"] > 0 else np.nan)
    for key in ("W_stage_0.25", "W_stage_0.5", "W_stage_1.0", "flat_span_m"):
        out[key] = _median_or_nan([m[key] for m in valid])
    if np.isfinite(out["relief_left_m"]) and np.isfinite(out["relief_right_m"]):
        r = out["relief_left_m"] / out["relief_right_m"]
        if r > ccfg["asymmetry_factor"] or r < 1.0 / ccfg["asymmetry_factor"]:
            flags.append("asymmetric_banks")
    # censoring bounds for the fit and the exceedance counts
    out["censor_left_bound"] = 2.0 * pixel_size
    out["censor_right_bound"] = 2.0 * ccfg["bank_max_offset_m"]
    if cls == "inner_channel":
        out["W_fit"], out["fit_censor"] = out["W_bank"], "none"
    elif cls == "wide_or_terrace":
        out["W_fit"], out["fit_censor"] = out["censor_right_bound"], "right"
    elif cls == "unresolved_narrow":
        out["W_fit"], out["fit_censor"] = out["censor_left_bound"], "left"
    else:
        out["W_fit"], out["fit_censor"] = np.nan, "excluded"
    if cls == "inner_channel" and n_t < len(ccfg["transect_offsets_m"]):
        flags.append(f"only_{n_t}_transects")
    out["qc_flags"] = ";".join(flags)
    return out


def measure_point(mt: MultiTransect, pixel_size: float, cfg: dict, ccfg: dict) -> tuple[dict, list]:
    per = [measure_transect(t.s, t.z, pixel_size, cfg, ccfg) if t is not None else None for t in mt.transects]
    return aggregate(per, pixel_size, ccfg), per


# ---------------------------------------------------------------------------
# exceedance and sensitivity
# ---------------------------------------------------------------------------

def shown_to_exceed(row, threshold: float) -> bool:
    """Is this point's width shown to exceed ``threshold``?

    inner_channel: W_bank > T. wide_or_terrace: True for T <= 2 x offset limit
    (the width is at least that), otherwise not shown. unresolved_narrow:
    never shown (the width is at most 2 delta). ambiguous: not shown.
    """
    c = row["point_class"]
    if c == "inner_channel":
        return bool(row["W_bank"] > threshold)
    if c == "wide_or_terrace":
        return bool(threshold <= row["censor_right_bound"])
    return False


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    lo = beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return (float(lo), float(hi))


def exceedance_table(points: pd.DataFrame, edges_by_site: dict, thresholds) -> pd.DataFrame:
    rows = []
    for site, d in points.groupby("site"):
        e = np.asarray(edges_by_site[site])
        b = np.clip(np.searchsorted(e, d["area_m2"].to_numpy(), side="right") - 1, 0, len(e) - 2)
        for k in range(len(e) - 1):
            sub = d[b == k]
            for T in thresholds:
                n = len(sub)
                x = int(sum(shown_to_exceed(r, T) for _, r in sub.iterrows()))
                lo, hi = clopper_pearson(x, n)
                rows.append(dict(site=site, bin_index=k, A_lo=e[k], A_hi=e[k + 1], threshold_m=T, n=n, n_exceed=x,
                                 fraction=x / n if n else np.nan, ci_lo=lo, ci_hi=hi))
        for T in thresholds:                                # site totals
            n = len(d)
            x = int(sum(shown_to_exceed(r, T) for _, r in d.iterrows()))
            lo, hi = clopper_pearson(x, n)
            rows.append(dict(site=site, bin_index=-1, A_lo=e[0], A_hi=e[-1], threshold_m=T, n=n, n_exceed=x,
                             fraction=x / n if n else np.nan, ci_lo=lo, ci_hi=hi))
    return pd.DataFrame(rows)


def sensitivity_table(store: dict, pixel_sizes: dict, cfg: dict, ccfg: dict) -> tuple[pd.DataFrame, dict]:
    """Re-classify every point over the (offset, relief) grid.

    ``store`` maps point_id -> MultiTransect; ``pixel_sizes`` maps point_id ->
    pixel size. Returns (summary table, {(offset, relief): per-point DataFrame}).
    """
    rows, per_setting = [], {}
    for off in ccfg["sensitivity_offsets_m"]:
        for rel in ccfg["sensitivity_reliefs_m"]:
            c2 = dict(ccfg, bank_max_offset_m=off, bank_max_relief_m=rel)
            recs = []
            for pid, mt in store.items():
                agg, _ = measure_point(mt, pixel_sizes[pid], cfg, c2)
                agg["point_id"] = pid
                recs.append(agg)
            df = pd.DataFrame(recs)
            per_setting[(off, rel)] = df
            cnt = df.point_class.value_counts()
            wb = df.loc[df.point_class == "inner_channel", "W_bank"]
            pile = float(np.mean(wb >= 2 * off - 1.0)) if len(wb) else np.nan
            rows.append(dict(bank_max_offset_m=off, bank_max_relief_m=rel, n=len(df),
                             n_inner_channel=int(cnt.get("inner_channel", 0)), n_wide_or_terrace=int(cnt.get("wide_or_terrace", 0)),
                             n_unresolved_narrow=int(cnt.get("unresolved_narrow", 0)), n_ambiguous=int(cnt.get("ambiguous", 0)),
                             median_W_bank=float(wb.median()) if len(wb) else np.nan,
                             frac_wide_or_terrace=float(cnt.get("wide_or_terrace", 0)) / len(df) if len(df) else np.nan,
                             frac_W_bank_within_1m_of_limit=pile))
    return pd.DataFrame(rows), per_setting


def window_is_binding(sens: pd.DataFrame, ccfg: dict) -> tuple[bool, list[str]]:
    """Binding if W_bank piles up against 2 x offset, or the wide_or_terrace fraction moves sharply with the offset."""
    reasons = []
    d = sens[sens.bank_max_relief_m == ccfg["bank_max_relief_m"]]
    at_default = d[d.bank_max_offset_m == ccfg["bank_max_offset_m"]]
    if len(at_default) and at_default.frac_W_bank_within_1m_of_limit.iloc[0] > ccfg["binding_pileup_fraction"]:
        reasons.append(f"{100 * at_default.frac_W_bank_within_1m_of_limit.iloc[0]:.0f} % of W_bank within 1 m of 2 x offset at the default window")
    offs = sorted(d.bank_max_offset_m.unique())
    if len(offs) >= 2:
        f0 = d[d.bank_max_offset_m == offs[0]].frac_wide_or_terrace.iloc[0]
        f1 = d[d.bank_max_offset_m == offs[-1]].frac_wide_or_terrace.iloc[0]
        if abs(f1 - f0) > ccfg["binding_wide_jump"]:
            reasons.append(f"wide_or_terrace fraction moves from {f0:.2f} at offset {offs[0]:g} m to {f1:.2f} at {offs[-1]:g} m")
    return (len(reasons) > 0), reasons
