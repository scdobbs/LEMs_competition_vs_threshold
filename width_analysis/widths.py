"""Width measures on a single cross-section z(s).

All functions take the transect distance ``s`` (m, monotonic, uniform step),
elevation ``z`` (m) and the thalweg index. Nothing here touches rasters.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d


# ---------------------------------------------------------------------------
# thalweg
# ---------------------------------------------------------------------------

def find_thalweg(s: np.ndarray, z: np.ndarray, search_half_width: float) -> int:
    """Index of the lowest sample within +/- search_half_width of s = 0."""
    m = np.abs(s) <= search_half_width + 1e-9
    idx = np.flatnonzero(m)
    zmin = np.min(z[idx])
    ties = idx[z[idx] <= zmin + 1e-6]          # flat beds: take the middle of the lowest run
    return int(ties[len(ties) // 2])


# ---------------------------------------------------------------------------
# contiguous span helpers
# ---------------------------------------------------------------------------

def _span_bounds(s, inside: np.ndarray, i0: int) -> tuple[float, float, bool, bool]:
    """Edges of the contiguous True-run of ``inside`` containing i0.

    Returns (s_left, s_right, hits_left_end, hits_right_end). Edges are placed
    half a step beyond the last inside sample (i.e. midway to the first outside
    sample), so a run of n samples has width n * step.
    """
    step = s[1] - s[0]
    L = i0
    while L - 1 >= 0 and inside[L - 1]:
        L -= 1
    R = i0
    while R + 1 < len(s) and inside[R + 1]:
        R += 1
    return s[L] - 0.5 * step, s[R] + 0.5 * step, L == 0, R == len(s) - 1


def _span_bounds_interp(s, z, i0: int, level: float) -> tuple[float, float, bool, bool]:
    """Edges where z crosses ``level`` on either side of i0, linearly interpolated."""
    n = len(s)
    L = i0
    while L - 1 >= 0 and z[L - 1] <= level:
        L -= 1
    R = i0
    while R + 1 < n and z[R + 1] <= level:
        R += 1
    if L == 0:
        sl = s[0]
    else:
        f = (level - z[L]) / (z[L - 1] - z[L]) if z[L - 1] != z[L] else 0.0
        sl = s[L] + f * (s[L - 1] - s[L])
    if R == n - 1:
        sr = s[-1]
    else:
        f = (level - z[R]) / (z[R + 1] - z[R]) if z[R + 1] != z[R] else 0.0
        sr = s[R] + f * (s[R + 1] - s[R])
    return float(sl), float(sr), L == 0, R == n - 1


# ---------------------------------------------------------------------------
# a. height-above-thalweg widths
# ---------------------------------------------------------------------------

def stage_label(h: float) -> str:
    """0.25 -> '0.25', 0.5 -> '0.5', 1.0 -> '1.0' (column-name suffix)."""
    t = f"{h:.2f}".rstrip("0")
    return t + "0" if t.endswith(".") else t


def width_at_stage(s, z, i_thal: int, h: float) -> tuple[float, float, float, bool]:
    """Width of the contiguous span around the thalweg with z - z_thalweg <= h.

    Returns (W, s_left, s_right, hits_transect_end).
    """
    level = z[i_thal] + h
    sl, sr, hl, hr = _span_bounds_interp(s, z, i_thal, level)
    return sr - sl, sl, sr, (hl or hr)


def hand_curve(s, z, i_thal: int, cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """W(h) for h = h_min .. h_max step h_step. Returns (h, W, hits_end)."""
    h = np.arange(cfg["h_min_m"], cfg["h_max_m"] + 0.5 * cfg["h_step_m"], cfg["h_step_m"])
    W = np.empty_like(h)
    hits = np.zeros(len(h), dtype=bool)
    for k, hk in enumerate(h):
        W[k], _, _, hits[k] = width_at_stage(s, z, i_thal, hk)
    return h, W, hits


def bankfull_pick(h: np.ndarray, W: np.ndarray, hits: np.ndarray) -> dict:
    """Stage where widening accelerates.

    Primary: the stage minimizing W/h (the hydraulic-depth proxy is smallest
    just before the banks are overtopped). Alternative recorded: stage
    maximizing d2W/dh2. Stages whose span reaches the transect end are not
    eligible. For a pure V-shape W/h is constant and the pick is arbitrary;
    that is reported through ``ratio_range`` (max/min of W/h) so that flat
    curves can be recognised downstream.
    """
    ok = ~hits & np.isfinite(W) & (W > 0)
    out = dict(h_bankfull=np.nan, W_bankfull=np.nan, h_bankfull_alt=np.nan, W_bankfull_alt=np.nan,
               ratio_range=np.nan, method="min_W_over_h")
    if ok.sum() < 3:
        return out
    ratio = np.where(ok, W / h, np.inf)
    k = int(np.argmin(ratio))
    out["h_bankfull"] = float(h[k])
    out["W_bankfull"] = float(W[k])
    out["ratio_range"] = float(np.max(ratio[ok]) / np.min(ratio[ok]))
    d2 = np.gradient(np.gradient(W, h), h)
    d2 = np.where(ok, d2, -np.inf)
    k2 = int(np.argmax(d2))
    out["h_bankfull_alt"] = float(h[k2])
    out["W_bankfull_alt"] = float(W[k2])
    return out


# ---------------------------------------------------------------------------
# b. bank-break width
# ---------------------------------------------------------------------------

def bank_break_width(s, z, i_thal: int, cfg: dict) -> dict:
    """Distance between the convex (concave-down) curvature maxima flanking the thalweg.

    z(s) is smoothed with a Gaussian of sigma = bankbreak_smooth_sigma_m, the
    curvature d2z/ds2 is taken by central differences, and on each side of the
    thalweg the first local maximum of convexity (-d2z/ds2) that satisfies both
    thresholds is the bank top:
      * convexity >= bankbreak_min_convexity_1_per_m
      * mean slope from the thalweg to the candidate >= bankbreak_min_bank_slope
    Candidates within 3 sigma + 1 sample of the transect ends are ignored
    (edge effects of the smoothing filter).
    """
    step = s[1] - s[0]
    zs = gaussian_filter1d(z, cfg["bankbreak_smooth_sigma_m"] / step, mode="nearest")
    conv = -np.gradient(np.gradient(zs, step), step)
    n = len(s)
    pad = int(np.ceil(3.0 * cfg["bankbreak_smooth_sigma_m"] / step)) + 1   # ignore filter edge effects
    picks = {}
    for side, rng in (("left", range(i_thal - 1, pad - 1, -1)), ("right", range(i_thal + 1, n - pad))):
        picks[side] = np.nan
        for k in rng:
            if conv[k] >= cfg["bankbreak_min_convexity_1_per_m"] and conv[k] >= conv[k - 1] and conv[k] >= conv[k + 1]:
                ds = abs(s[k] - s[i_thal])
                if ds > 0 and (zs[k] - zs[i_thal]) / ds >= cfg["bankbreak_min_bank_slope"]:
                    picks[side] = float(s[k])
                    break
    W = picks["right"] - picks["left"] if np.isfinite(picks["left"]) and np.isfinite(picks["right"]) else np.nan
    return dict(W_bankbreak=W, s_bank_left=picks["left"], s_bank_right=picks["right"])


# ---------------------------------------------------------------------------
# c. valley-floor width
# ---------------------------------------------------------------------------

def valley_floor_width(s, z, i_thal: int, cfg: dict) -> dict:
    """Contiguous span around the thalweg where |dz/ds| (smoothed) < valley_slope_threshold."""
    step = s[1] - s[0]
    sig = cfg["valley_smooth_sigma_m"]
    zs = gaussian_filter1d(z, sig / step, mode="nearest") if sig > 0 else z
    slope = np.abs(np.gradient(zs, step))
    inside = slope < cfg["valley_slope_threshold"]
    inside[i_thal] = True
    sl, sr, hl, hr = _span_bounds(s, inside, i_thal)
    return dict(W_valley=sr - sl, s_valley_left=sl, s_valley_right=sr, valley_hits_end=(hl or hr))


# ---------------------------------------------------------------------------
# censoring and shape class
# ---------------------------------------------------------------------------

def censor(width: float, pixel_size: float, multiple: float) -> tuple[bool, float]:
    bound = multiple * pixel_size
    return (bool(np.isfinite(width) and width < bound), bound)


def shape_class(s, z, i_thal: int, cfg: dict) -> tuple[str, float]:
    """flat_bed if samples within flat_bed_tol_m of the thalweg span >= flat_bed_min_width_m contiguously."""
    inside = (z - z[i_thal]) <= cfg["flat_bed_tol_m"]
    sl, sr, _, _ = _span_bounds(s, inside, i_thal)
    w = sr - sl
    return ("flat_bed" if w >= cfg["flat_bed_min_width_m"] else "v_shape"), float(w)


# ---------------------------------------------------------------------------
# everything for one cross-section
# ---------------------------------------------------------------------------

def measure_cross_section(s: np.ndarray, z: np.ndarray, pixel_size: float, cfg: dict) -> dict:
    """All width measures, flags and picks for one transect. Keys are stable column names."""
    i_t = find_thalweg(s, z, cfg["thalweg_search_m"])
    flags = []
    out: dict = dict(thalweg_index=i_t, s_thalweg=float(s[i_t]), z_thalweg=float(z[i_t]))

    h, W, hits = hand_curve(s, z, i_t, cfg)
    out["hand_h"] = h
    out["hand_W"] = W
    bf = bankfull_pick(h, W, hits)
    out["W_hand_bankfull"] = bf["W_bankfull"]
    out["h_bankfull"] = bf["h_bankfull"]
    out["bankfull_method"] = bf["method"]
    out["h_bankfull_alt_d2W"] = bf["h_bankfull_alt"]
    out["W_hand_bankfull_alt_d2W"] = bf["W_bankfull_alt"]
    out["W_over_h_range"] = bf["ratio_range"]
    if not np.isfinite(bf["W_bankfull"]):
        flags.append("no_bankfull_pick")
    if np.isfinite(bf["ratio_range"]) and bf["ratio_range"] < 1.1:
        flags.append("W_over_h_flat")
    if hits.any():
        flags.append("hand_hits_transect_end_at_h=%.2f" % h[np.argmax(hits)])

    for hf in cfg["fixed_stages_m"]:
        w, sl, sr, he = width_at_stage(s, z, i_t, hf)
        key = f"W_hand_{stage_label(hf)}"
        out[key] = w
        out[key + "_sL"] = sl
        out[key + "_sR"] = sr
        if he:
            flags.append(f"stage_{hf:g}_hits_transect_end")
    if np.isfinite(bf["h_bankfull"]):
        _, sl, sr, _ = width_at_stage(s, z, i_t, bf["h_bankfull"])
        out["W_hand_bankfull_sL"], out["W_hand_bankfull_sR"] = sl, sr
    else:
        out["W_hand_bankfull_sL"] = out["W_hand_bankfull_sR"] = np.nan

    out.update(bank_break_width(s, z, i_t, cfg))
    if not np.isfinite(out["W_bankbreak"]):
        flags.append("no_bank_break")
    out.update(valley_floor_width(s, z, i_t, cfg))
    if out["valley_hits_end"]:
        flags.append("valley_hits_transect_end")

    cens, bound = censor(out["W_hand_bankfull"], pixel_size, cfg["censor_multiple"])
    out["censored"] = cens
    out["censor_bound"] = bound
    for key in [f"W_hand_{stage_label(hf)}" for hf in cfg["fixed_stages_m"]] + ["W_bankbreak", "W_valley"]:
        out[key + "_censored"] = censor(out[key], pixel_size, cfg["censor_multiple"])[0]

    sc, flat_w = shape_class(s, z, i_t, cfg)
    out["shape_class"] = sc
    out["flat_span_m"] = flat_w
    out["qc_flags"] = ";".join(flags)
    return out
