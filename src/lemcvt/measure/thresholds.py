"""Choosing the drainage-area threshold for channel pixels.

Primary choice: the slope-area rollover (Montgomery & Foufoula-Georgiou 1993;
Tarolli & Dalla Fontana 2009). Binned (median) local slope against log drainage
area rises on hillslopes and falls on channels; the area of maximum binned median
slope marks the transition. Slopes are D8 steepest-descent slopes of the filled
grid (``(z - z_receiver) / link length``), binned in 0.1 decades of area, with a
3-bin running median to suppress the TIN-facet noise of the AP tiles. The result
and the whole curve are reported; theta and k_s are also measured on a fixed
ladder of thresholds so the sensitivity is visible.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from lemcvt.hydro.d8 import d8_receiver_arrays

THRESHOLD_LADDER = (1e3, 2e3, 5e3, 1e4, 2e4, 5e4)


@dataclass
class Rollover:
    A_rollover: float
    log10A_centers: list[float]
    median_slope: list[float]
    smoothed_slope: list[float]
    counts: list[int]
    note: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def d8_slope(z: np.ndarray, fd: np.ndarray, dx: float, dy: float | None = None) -> np.ndarray:
    dy = dx if dy is None else dy
    r_to, c_to, good = d8_receiver_arrays(fd)
    ny, nx = z.shape
    rows, cols = np.indices((ny, nx))
    S = np.full(z.shape, np.nan)
    rr, cc = rows[good], cols[good]
    dr, dc = r_to[good] - rr, c_to[good] - cc
    L = np.where((dr != 0) & (dc != 0), np.hypot(dx, dy), np.where(dr != 0, dy, dx))
    S[rr, cc] = (z[rr, cc] - z[r_to[good], c_to[good]]) / L
    return S


def slope_area_rollover(z: np.ndarray, area: np.ndarray, fd: np.ndarray, dx: float, dy: float | None = None,
                        bin_width_decades: float = 0.1, min_count: int = 50, a_min: float = 10.0,
                        a_max: float = 1e6) -> Rollover:
    S = d8_slope(z, fd, dx, dy)
    ok = np.isfinite(S) & np.isfinite(area) & (area >= a_min) & (area <= a_max)
    la = np.log10(area[ok])
    s = S[ok]
    edges = np.arange(np.log10(a_min), np.log10(a_max) + bin_width_decades, bin_width_decades)
    centers, med, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        sel = (la >= lo) & (la < hi)
        n = int(sel.sum())
        if n < min_count:
            continue
        centers.append(0.5 * (lo + hi))
        med.append(float(np.median(s[sel])))
        counts.append(n)
    med_a = np.array(med)
    if med_a.size < 5:
        return Rollover(np.nan, centers, med, med, counts, note="too few populated bins")
    sm = med_a.copy()
    sm[1:-1] = np.median(np.vstack([med_a[:-2], med_a[1:-1], med_a[2:]]), axis=0)
    j = int(np.argmax(sm))
    note = "" if 0 < j < sm.size - 1 else "rollover at the edge of the area range"
    return Rollover(float(10 ** centers[j]), centers, med, sm.tolist(), counts, note)
