"""Constrained bank pick, along-stream averaged cross-sections, point
classification, exceedance statistics and window sensitivity
(reports/directives/channel_width_constrained_directive.md, with Steve's
2026-11-02 amendments: relief ceiling 4 m, one_bank rule, unresolved_wide,
W_trench and W_bed reported separately, trench depth).

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

CLASSES = ("inner_channel", "one_bank", "wide_or_terrace", "unresolved_narrow", "unresolved_wide", "ambiguous")
RESOLVED = ("inner_channel", "one_bank")


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


def pick_side(cands: list, ccfg: dict, max_relief_within_offset: float = np.nan) -> dict:
    """Constrained pick on one side.

    ``bank``: first candidate (nearest the thalweg) with relief in
    [bank_min_relief_m, bank_max_relief_m] and offset <= bank_max_offset_m.
    ``none_within_window``: a candidate in the relief window exists only
    beyond the offset limit. ``wall``: no candidate in the relief window, and
    the profile rises above the relief ceiling within the offset limit (a
    valley wall or cliff bounding the channel; its first break, if any, is
    above the ceiling). ``no_break``: none of the above. The nearest break of
    any relief is always recorded as first_offset / first_relief.
    """
    lo, hi, off = ccfg["bank_min_relief_m"], ccfg["bank_max_relief_m"], ccfg["bank_max_offset_m"]
    first = cands[0] if cands else (np.nan, np.nan, np.nan)
    base = dict(first_offset=first[0], first_relief=first[1])
    beyond = None
    for (d, r, sk) in cands:                         # nearest first
        if lo <= r <= hi:
            if d <= off:
                return dict(side_class="bank", offset=d, relief=r, s=sk, **base)
            if beyond is None:
                beyond = (d, r, sk)
    if beyond is not None:
        return dict(side_class="none_within_window", offset=beyond[0], relief=beyond[1], s=beyond[2], **base)
    if np.isfinite(max_relief_within_offset) and max_relief_within_offset > hi:
        return dict(side_class="wall", offset=first[0], relief=first[1], s=first[2], **base)
    return dict(side_class="no_break", offset=np.nan, relief=np.nan, s=np.nan, **base)


def classify(left: str, right: str, shape: str, w_stage_025: float = np.nan, bound: float = np.inf) -> str:
    """Point class from the two side classes.

    inner_channel: bank + bank. one_bank: bank + wall. wide_or_terrace: at
    least one none_within_window, the other bank or none_within_window.
    unresolved_narrow: no_break + no_break with W_stage_0.25 <= 2 delta.
    unresolved_wide: no_break + no_break otherwise (a flat floor, or a section
    wider than 2 delta at 0.25 m with no bank anywhere). ambiguous: anything else.
    """
    sides = {left, right}
    if left == "bank" and right == "bank":
        return "inner_channel"
    if sides == {"bank", "wall"}:
        return "one_bank"
    if "none_within_window" in sides and sides <= {"bank", "none_within_window"}:
        return "wide_or_terrace"
    if left == "no_break" and right == "no_break":
        if shape == "v_shape" and np.isfinite(w_stage_025) and w_stage_025 <= bound:
            return "unresolved_narrow"
        return "unresolved_wide"
    return "ambiguous"


# ---------------------------------------------------------------------------
# one transect
# ---------------------------------------------------------------------------

def measure_transect(s, z, pixel_size: float, cfg: dict, ccfg: dict) -> dict:
    """Constrained measures on one cross-section (keys are per-transect columns)."""
    i_t = find_thalweg(s, z, cfg["thalweg_search_m"])
    sc, flat_w = shape_class(s, z, i_t, cfg)
    bound = cfg["censor_multiple"] * pixel_size
    stages = {f"W_stage_{stage_label(hf)}": float(width_at_stage(s, z, i_t, hf, cfg["crest_drop_m"])[0])
              for hf in cfg["fixed_stages_m"]}
    cands = convex_breaks(s, z, i_t, cfg)
    off = ccfg["bank_max_offset_m"]
    zl = z[(s < s[i_t]) & (s >= s[i_t] - off)]
    zr = z[(s > s[i_t]) & (s <= s[i_t] + off)]
    L = pick_side(cands["left"], ccfg, float(zl.max() - z[i_t]) if len(zl) else np.nan)
    R = pick_side(cands["right"], ccfg, float(zr.max() - z[i_t]) if len(zr) else np.nan)
    cls = classify(L["side_class"], R["side_class"], sc, stages.get("W_stage_0.25", np.nan), bound)
    out = dict(thalweg_s=float(s[i_t]), z_thalweg=float(z[i_t]), shape_class=sc, flat_span_m=flat_w,
               left_class=L["side_class"], right_class=R["side_class"], point_class=cls,
               s_bank_left=L["s"], s_bank_right=R["s"], relief_left_m=L["relief"], relief_right_m=R["relief"],
               offset_left_m=L["offset"], offset_right_m=R["offset"],
               first_relief_left_m=L["first_relief"], first_relief_right_m=R["first_relief"],
               n_breaks_left=len(cands["left"]), n_breaks_right=len(cands["right"]),
               W_trench=np.nan, W_stage_bank=np.nan, trench_depth_m=np.nan)
    out.update(stages)
    out["W_bed"] = stages["W_stage_0.5"]
    if cls == "inner_channel":
        out["W_trench"] = float(R["s"] - L["s"])
        h = 0.5 * (L["relief"] + R["relief"])
        out["trench_depth_m"] = h
        out["W_stage_bank"] = float(width_at_stage(s, z, i_t, h, cfg["crest_drop_m"])[0])
    elif cls == "one_bank":
        h = L["relief"] if L["side_class"] == "bank" else R["relief"]
        out["trench_depth_m"] = float(h)
        out["W_stage_bank"] = float(width_at_stage(s, z, i_t, h, cfg["crest_drop_m"])[0])
        out["W_trench"] = out["W_stage_bank"]                 # width at the accepted bank's relief
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
    n_inner = sum(c in RESOLVED for c in classes)
    if max(n_inner, n_t - n_inner) < ccfg["min_agreeing_transects"]:
        flags.append("inconsistent_along_stream")
    out["point_class"] = cls
    out["n_resolved_transects"] = n_inner
    out["class_counts"] = ";".join(f"{k}:{v}" for k, v in sorted(Counter(classes).items()))
    out["left_class"] = Counter(m["left_class"] for m in valid).most_common(1)[0][0]
    out["right_class"] = Counter(m["right_class"] for m in valid).most_common(1)[0][0]
    out["shape_class"] = Counter(m["shape_class"] for m in valid).most_common(1)[0][0]
    same = [m for m in valid if m["point_class"] == cls]       # transects agreeing with the point class
    wt = [m["W_trench"] for m in same if np.isfinite(m["W_trench"])]
    out["W_trench"] = _median_or_nan(wt)
    out["W_trench_n_valid"] = len(wt)
    out["W_trench_spread"] = float(max(wt) - min(wt)) if len(wt) > 1 else (0.0 if wt else np.nan)
    wbed = [m["W_bed"] for m in valid if np.isfinite(m["W_bed"])]
    out["W_bed"] = _median_or_nan(wbed)
    out["W_bed_n_valid"] = len(wbed)
    out["W_bed_spread"] = float(max(wbed) - min(wbed)) if len(wbed) > 1 else (0.0 if wbed else np.nan)
    out["W_stage_bank"] = _median_or_nan([m["W_stage_bank"] for m in same])
    out["trench_depth_m"] = _median_or_nan([m["trench_depth_m"] for m in same])
    out["relief_left_m"] = _median_or_nan([m["relief_left_m"] for m in valid if m["left_class"] == "bank"])
    out["relief_right_m"] = _median_or_nan([m["relief_right_m"] for m in valid if m["right_class"] == "bank"])
    out["first_relief_left_m"] = _median_or_nan([m["first_relief_left_m"] for m in valid])
    out["first_relief_right_m"] = _median_or_nan([m["first_relief_right_m"] for m in valid])
    out["ratio_trench_over_stage"] = (out["W_trench"] / out["W_stage_bank"]
                                      if cls == "inner_channel" and np.isfinite(out["W_trench"]) and out["W_stage_bank"] > 0 else np.nan)
    for key in ("W_stage_0.25", "W_stage_0.5", "W_stage_1.0", "flat_span_m"):
        out[key] = _median_or_nan([m[key] for m in valid])
    if cls == "one_bank":
        flags.append("wall_bounded")
    if np.isfinite(out["relief_left_m"]) and np.isfinite(out["relief_right_m"]):
        r = out["relief_left_m"] / out["relief_right_m"]
        if r > ccfg["asymmetry_factor"] or r < 1.0 / ccfg["asymmetry_factor"]:
            flags.append("asymmetric_banks")
    # censoring bounds for the fit and the exceedance counts
    out["censor_left_bound"] = 2.0 * pixel_size
    out["censor_right_bound"] = 2.0 * ccfg["bank_max_offset_m"]
    if cls in RESOLVED:
        out["W_fit"], out["fit_censor"] = out["W_trench"], "none"
    elif cls == "wide_or_terrace":
        out["W_fit"], out["fit_censor"] = out["censor_right_bound"], "right"
    elif cls == "unresolved_narrow":
        out["W_fit"], out["fit_censor"] = out["censor_left_bound"], "left"
    else:
        out["W_fit"], out["fit_censor"] = np.nan, "excluded"
    if cls in RESOLVED and n_t < len(ccfg["transect_offsets_m"]):
        flags.append(f"only_{n_t}_transects")
    out["qc_flags"] = ";".join(flags)
    return out


def measure_point(mt: MultiTransect, pixel_size: float, cfg: dict, ccfg: dict) -> tuple[dict, list]:
    per = [measure_transect(t.s, t.z, pixel_size, cfg, ccfg) if t is not None else None for t in mt.transects]
    return aggregate(per, pixel_size, ccfg), per


# ---------------------------------------------------------------------------
# exceedance and sensitivity
# ---------------------------------------------------------------------------

def shown_to_exceed(row, threshold: float, measure: str = "bed") -> bool:
    """Is this point's width shown to exceed ``threshold``?

    measure="bed" (conservative): resolved points (inner_channel, one_bank)
    and wide_or_terrace points count when W_bed = W_stage_0.5 > T.
    measure="trench": resolved points count when W_trench > T;
    wide_or_terrace points count for T <= 2 x offset limit.
    unresolved_narrow, unresolved_wide and ambiguous points are never shown
    to exceed (their W_stage_0.5 is a valley floor or hollow, not a bed).
    """
    c = row["point_class"]
    if measure == "bed":
        if c in RESOLVED or c == "wide_or_terrace":
            return bool(row["W_bed"] > threshold)
        return False
    if c in RESOLVED:
        return bool(row["W_trench"] > threshold)
    if c == "wide_or_terrace":
        return bool(threshold <= row["censor_right_bound"])
    return False


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    lo = beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return (float(lo), float(hi))


def exceedance_table(points: pd.DataFrame, edges_by_site: dict, thresholds, measures=("bed", "trench")) -> pd.DataFrame:
    rows = []
    for site, d in points.groupby("site"):
        e = np.asarray(edges_by_site[site])
        b = np.clip(np.searchsorted(e, d["area_m2"].to_numpy(), side="right") - 1, 0, len(e) - 2)
        groups = [(k, d[b == k], e[k], e[k + 1]) for k in range(len(e) - 1)] + [(-1, d, e[0], e[-1])]
        for k, sub, alo, ahi in groups:
            for meas in measures:
                for T in thresholds:
                    n = len(sub)
                    x = int(sum(shown_to_exceed(r, T, meas) for _, r in sub.iterrows()))
                    lo, hi = clopper_pearson(x, n)
                    rows.append(dict(site=site, bin_index=k, A_lo=alo, A_hi=ahi, measure=meas, threshold_m=T, n=n,
                                     n_exceed=x, fraction=x / n if n else np.nan, ci_lo=lo, ci_hi=hi))
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
            wb = df.loc[df.point_class.isin(RESOLVED), "W_trench"]
            pile = float(np.mean(wb >= 2 * off - 1.0)) if len(wb) else np.nan
            rows.append(dict(bank_max_offset_m=off, bank_max_relief_m=rel, n=len(df),
                             n_inner_channel=int(cnt.get("inner_channel", 0)), n_one_bank=int(cnt.get("one_bank", 0)),
                             n_wide_or_terrace=int(cnt.get("wide_or_terrace", 0)),
                             n_unresolved_narrow=int(cnt.get("unresolved_narrow", 0)), n_unresolved_wide=int(cnt.get("unresolved_wide", 0)),
                             n_ambiguous=int(cnt.get("ambiguous", 0)),
                             median_W_trench=float(wb.median()) if len(wb) else np.nan,
                             frac_wide_or_terrace=float(cnt.get("wide_or_terrace", 0)) / len(df) if len(df) else np.nan,
                             frac_W_trench_within_1m_of_limit=pile))
    return pd.DataFrame(rows), per_setting


def window_is_binding(sens: pd.DataFrame, ccfg: dict) -> tuple[bool, list[str]]:
    """Binding if W_trench piles up against 2 x offset, or the wide_or_terrace fraction moves sharply with the offset."""
    reasons = []
    d = sens[sens.bank_max_relief_m == ccfg["bank_max_relief_m"]]
    at_default = d[d.bank_max_offset_m == ccfg["bank_max_offset_m"]]
    if len(at_default) and at_default.frac_W_trench_within_1m_of_limit.iloc[0] > ccfg["binding_pileup_fraction"]:
        reasons.append(f"{100 * at_default.frac_W_trench_within_1m_of_limit.iloc[0]:.0f} % of W_trench within 1 m of 2 x offset at the default window")
    offs = sorted(d.bank_max_offset_m.unique())
    if len(offs) >= 2:
        f0 = d[d.bank_max_offset_m == offs[0]].frac_wide_or_terrace.iloc[0]
        f1 = d[d.bank_max_offset_m == offs[-1]].frac_wide_or_terrace.iloc[0]
        if abs(f1 - f0) > ccfg["binding_wide_jump"]:
            reasons.append(f"wide_or_terrace fraction moves from {f0:.2f} at offset {offs[0]:g} m to {f1:.2f} at {offs[-1]:g} m")
    return (len(reasons) > 0), reasons
