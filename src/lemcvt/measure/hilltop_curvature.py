"""Hilltop curvature C_ht and D0 = U / C_ht.

Hilltop pixels (explicit definition)
    A pixel is a hilltop pixel when it (i) receives no D8 flow (a divide cell,
    TopoAnalysis ``divides``), (ii) is adjacent (8-neighbourhood) to a pixel that
    drains to a different channel-network basin (so it sits on the drainage divide
    between two catchments of the channel network at threshold ``A_c``, after Hurst
    et al. 2012 who take hilltops as the divides between basins of a channel
    network), (iii) is not within ``edge_margin`` pixels of the tile edge, and
    (iv) has a local slope below ``max_slope`` (Hurst et al. 2012 exclude steep
    hilltops, |grad z| > 0.4, where the linear-diffusion assumption fails).

Curvature
    Total curvature C = d2z/dx2 + d2z/dy2 from a least-squares quadratic surface
    z = a x^2 + b y^2 + c xy + d x + e y + f fitted over a (2w+1)^2 window (Evans
    1980; Hurst et al. 2012): C = 2(a + b). For a fixed window this is a linear
    filter, so C_w is a convolution of z with a kernel; the kernel is built from the
    pseudo-inverse of the design matrix. Convex-up hilltops have C < 0; the reported
    C_ht is the median of -C over hilltop pixels (positive magnitude, the convention
    of D0 = U/C_ht).

Scale selection
    The curvature of a lidar hilltop depends on the fitting window because of
    small-scale roughness (and, on AP, TIN facets). C_ht(L) is computed for windows
    of length L = (2w+1) dx from 3 m to about 61 m. The chosen scale L* is the
    smallest L at which the median hilltop curvature has stopped changing: the
    relative change of C_ht between successive windows is below ``plateau_tol`` for
    that step and the next one. The same rule is applied at both sites; the curve
    is reported so the choice can be judged.

Uncertainty
    Bootstrap over connected hilltop segments (8-connected components of the hilltop
    mask), which respects the spatial correlation of neighbouring hilltop pixels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.ndimage import label
from scipy.signal import fftconvolve

from lemcvt.hydro.d8 import d8_divides, d8_receiver_arrays

DEFAULT_HALF_WIDTHS = (1, 2, 3, 4, 5, 7, 10, 12, 15, 20, 25, 30)


def curvature_kernel(w: int, dx: float, dy: float | None = None) -> np.ndarray:
    """Convolution kernel giving C = 2(a + b) of the least-squares quadratic over a (2w+1)^2 window."""
    dy = dx if dy is None else dy
    ys, xs = np.mgrid[-w:w + 1, -w:w + 1]
    X = (xs * dx).ravel()
    Y = (ys * dy).ravel()
    G = np.column_stack([X * X, Y * Y, X * Y, X, Y, np.ones_like(X)])
    P = np.linalg.pinv(G)                 # (6, n)
    weights = 2.0 * (P[0] + P[1])         # C = 2a + 2b
    k = weights.reshape(2 * w + 1, 2 * w + 1)
    return k[::-1, ::-1]                  # convolution flips the kernel


def total_curvature(z: np.ndarray, w: int, dx: float, dy: float | None = None) -> np.ndarray:
    """Total curvature grid for window half-width ``w`` (NaN within ``w`` of the edge)."""
    zz = np.nan_to_num(z, nan=float(np.nanmean(z)))
    C = fftconvolve(zz, curvature_kernel(w, dx, dy), mode="same")
    C[:w, :] = np.nan
    C[-w:, :] = np.nan
    C[:, :w] = np.nan
    C[:, -w:] = np.nan
    C[~np.isfinite(z)] = np.nan
    return C


def basin_labels(area: np.ndarray, fd: np.ndarray, channel_mask: np.ndarray) -> np.ndarray:
    """Label every pixel by the channel pixel it first reaches (0 = reaches none)."""
    ny, nx = area.shape
    r_to, c_to, good = d8_receiver_arrays(fd)
    labels = np.zeros((ny, nx), dtype=np.int64)
    ch_rows, ch_cols = np.nonzero(channel_mask)
    labels[ch_rows, ch_cols] = np.arange(1, ch_rows.size + 1)
    # receivers before donors: area descending, equal areas in ascending flat index (numerics_changes.md, N4)
    area_f = np.asarray(area, dtype=np.float64).ravel()
    order = np.lexsort((np.arange(area_f.size), -area_f))
    rows, cols = np.unravel_index(order, (ny, nx))
    for r, c in zip(rows.tolist(), cols.tolist(), strict=True):
        if labels[r, c] == 0 and good[r, c]:
            labels[r, c] = labels[r_to[r, c], c_to[r, c]]
    # relabel channel pixels by their outlet basin: walk channel pixels downstream
    return labels


def hilltop_mask(z: np.ndarray, area: np.ndarray, fd: np.ndarray, A_c: float, dx: float, dy: float | None = None,
                 edge_margin: int = 2, max_slope: float | None = 0.4,
                 basin_of_channel: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Hilltop pixel mask and diagnostics (see module docstring)."""
    dy = dx if dy is None else dy
    ny, nx = z.shape
    divides = d8_divides(fd).astype(bool)
    channel = np.isfinite(area) & (area >= A_c)
    lab = basin_labels(area, fd, channel)
    if basin_of_channel is not None:
        # map channel-pixel labels to outlet-basin ids so the divide is between networks' basins
        lab = np.where(lab > 0, basin_of_channel[lab - 1] + 1, 0) if basin_of_channel.size else lab
    # neighbour with a different (non-zero) label
    diff = np.zeros((ny, nx), dtype=bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            sh = np.zeros_like(lab)
            rs = slice(max(dr, 0), ny + min(dr, 0))
            cs = slice(max(dc, 0), nx + min(dc, 0))
            rs2 = slice(max(-dr, 0), ny + min(-dr, 0))
            cs2 = slice(max(-dc, 0), nx + min(-dc, 0))
            sh[rs, cs] = lab[rs2, cs2]
            diff |= (sh != lab) & (sh > 0) & (lab > 0)
    mask = divides & diff & ~channel
    mask[:edge_margin, :] = False
    mask[-edge_margin:, :] = False
    mask[:, :edge_margin] = False
    mask[:, -edge_margin:] = False
    diag = {"n_divide": int(divides.sum()), "n_divide_between_basins": int((divides & diff).sum())}
    if max_slope is not None:
        gy, gx = np.gradient(z, dy, dx)
        s = np.hypot(gx, gy)
        diag["n_removed_steep"] = int((mask & (s > max_slope)).sum())
        mask &= s <= max_slope
    diag["n_hilltop"] = int(mask.sum())
    return mask, diag


@dataclass
class CurvatureResult:
    A_c: float
    half_widths: list[int]
    window_lengths_m: list[float]
    C_ht_median: list[float]
    C_ht_q16: list[float]
    C_ht_q84: list[float]
    frac_convex: list[float]
    w_star: int
    L_star_m: float
    C_ht: float
    C_ht_ci68: tuple[float, float] | None
    C_ht_ci95: tuple[float, float] | None
    D0: float
    D0_ci68: tuple[float, float] | None
    D0_ci95: tuple[float, float] | None
    n_hilltop: int
    n_segments: int
    diagnostics: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def select_scale(L: np.ndarray, C: np.ndarray, plateau_tol: float = 0.05) -> tuple[int, str]:
    """Smallest index i with |C[i+1]-C[i]|/|C[i]| < tol and |C[i+2]-C[i+1]|/|C[i+1]| < tol."""
    rel = np.abs(np.diff(C)) / np.abs(C[:-1])
    for i in range(len(rel) - 1):
        if rel[i] < plateau_tol and rel[i + 1] < plateau_tol:
            return i, ""
    return int(np.nanargmin(rel)), "no plateau within the window range; smallest change taken"


def measure_hilltop_curvature(z: np.ndarray, area: np.ndarray, fd: np.ndarray, A_c: float, U: float, dx: float,
                              dy: float | None = None, half_widths=DEFAULT_HALF_WIDTHS, max_slope: float | None = 0.4,
                              plateau_tol: float = 0.05, n_boot: int = 200, seed: int = 0,
                              basin_of_channel: np.ndarray | None = None) -> CurvatureResult:
    dy = dx if dy is None else dy
    mask, diag = hilltop_mask(z, area, fd, A_c, dx, dy, max_slope=max_slope, basin_of_channel=basin_of_channel)
    if mask.sum() < 20:
        raise ValueError(f"only {int(mask.sum())} hilltop pixels at A_c={A_c:g}")
    seg, n_seg = label(mask, structure=np.ones((3, 3)))
    hw = list(half_widths)
    L = np.array([(2 * w + 1) * dx for w in hw])
    med, q16, q84, frac = [], [], [], []
    per_w_values = []
    for w in hw:
        C = total_curvature(z, w, dx, dy)
        v = -C[mask]                      # positive on convex hilltops
        v_ok = v[np.isfinite(v)]
        per_w_values.append((-C, ))
        med.append(float(np.median(v_ok)))
        q16.append(float(np.percentile(v_ok, 16)))
        q84.append(float(np.percentile(v_ok, 84)))
        frac.append(float(np.mean(v_ok > 0)))
    i_star, note = select_scale(L, np.array(med), plateau_tol)
    w_star = hw[i_star]
    C_grid = per_w_values[i_star][0]
    vals = C_grid[mask]
    seg_ids = seg[mask]
    ok = np.isfinite(vals)
    vals, seg_ids = vals[ok], seg_ids[ok]
    C_ht = float(np.median(vals))
    notes = [note] if note else []
    ci68 = ci95 = d68 = d95 = None
    if n_boot > 0 and n_seg >= 5:
        rng = np.random.default_rng(seed)
        uniq = np.unique(seg_ids)
        bs = np.empty(n_boot)
        for i in range(n_boot):
            pick = rng.choice(uniq, uniq.size, replace=True)
            counts = np.bincount(pick, minlength=seg_ids.max() + 1)
            wts = counts[seg_ids]
            bs[i] = _weighted_median(vals, wts)
        ci68 = (float(np.percentile(bs, 16)), float(np.percentile(bs, 84)))
        ci95 = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
        d68 = (U / ci68[1], U / ci68[0])
        d95 = (U / ci95[1], U / ci95[0])
    if C_ht <= 0:
        notes.append("median hilltop curvature is not convex; D0 undefined")
    return CurvatureResult(
        A_c=A_c, half_widths=hw, window_lengths_m=L.tolist(), C_ht_median=med, C_ht_q16=q16, C_ht_q84=q84,
        frac_convex=frac, w_star=w_star, L_star_m=float(L[i_star]), C_ht=C_ht, C_ht_ci68=ci68, C_ht_ci95=ci95,
        D0=float(U / C_ht) if C_ht > 0 else np.nan, D0_ci68=d68, D0_ci95=d95, n_hilltop=int(ok.sum()),
        n_segments=int(n_seg), diagnostics=diag, notes=notes,
    )


def _weighted_median(v: np.ndarray, w: np.ndarray) -> float:
    order = np.argsort(v, kind="stable")   # ties cannot change the returned value; stable for determinism (N5)
    cw = np.cumsum(w[order])
    if cw[-1] == 0:
        return np.nan
    return float(v[order][np.searchsorted(cw, 0.5 * cw[-1])])
