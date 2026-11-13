"""Regime outputs: W/delta maps, regime-III channel-length fraction vs grid
spacing, and the regime-boundary table A* = (delta / c)^(1/b)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SWEEP_DX_MAX_M, SWEEP_DX_MIN_M, SWEEP_N_DX, acrit_m2
from .network import receiver_grids
from .tiles import Tile


def sweep_dx() -> np.ndarray:
    """delta = 1 m plus the 25 log-spaced values of the existing 5-100 m sweep."""
    return np.concatenate([[1.0], np.logspace(np.log10(SWEEP_DX_MIN_M), np.log10(SWEEP_DX_MAX_M), SWEEP_N_DX)])


def channel_pixels(tile: Tile) -> dict:
    """Channel pixels (A >= the tile's A_crit) with their D8 step length (channel length per pixel)."""
    a_crit = acrit_m2()[(tile.site, tile.tile)]
    mask = tile.area >= a_crit
    rows, cols = np.nonzero(mask)
    ri, rj = receiver_grids(tile.fd)
    diag = (ri[rows, cols] >= 0) & (ri[rows, cols] != rows) & (rj[rows, cols] != cols)
    length = np.where(diag, np.sqrt(2.0), 1.0) * tile.pixel_size
    return dict(site=tile.site, tile=tile.tile, a_crit=a_crit, rows=rows, cols=cols,
                area=tile.area[rows, cols], length=length, pixel_size=tile.pixel_size)


def width_of(area: np.ndarray, a: float, b: float) -> np.ndarray:
    return 10 ** (a + b * np.log10(area))


def regime_fraction(chan: list[dict], a: float, b: float, dxs: np.ndarray) -> np.ndarray:
    """Fraction of channel length (pooled over the given tiles) with W(A) > delta, for each delta."""
    area = np.concatenate([c["area"] for c in chan])
    length = np.concatenate([c["length"] for c in chan])
    W = width_of(area, a, b)
    tot = length.sum()
    return np.array([(length[W > dx]).sum() / tot for dx in dxs])


def regime_fraction_table(site: str, chan: list[dict], fit: dict, dxs: np.ndarray, pcts=(2.5, 97.5)) -> pd.DataFrame:
    """Point estimate and bootstrap band of the regime-III fraction vs delta for one site/variant."""
    a, b = fit["tobit"]["a"], fit["tobit"]["b"]
    rows = []
    est = regime_fraction(chan, a, b, dxs) if np.isfinite(a) else np.full(len(dxs), np.nan)
    boot = fit.get("_boot_ab", np.zeros((0, 2)))
    if len(boot):
        bf = np.array([regime_fraction(chan, aa, bb, dxs) for aa, bb in boot])
        lo, hi = np.percentile(bf, pcts, axis=0)
    else:
        lo = hi = np.full(len(dxs), np.nan)
    for k, dx in enumerate(dxs):
        rows.append(dict(site=site, variant=fit["label"], dx_m=float(dx), fraction=float(est[k]),
                         fraction_lo=float(lo[k]), fraction_hi=float(hi[k])))
    return pd.DataFrame(rows)


def regime_boundary_table(site: str, fit: dict, dxs: np.ndarray, pcts=(2.5, 97.5)) -> pd.DataFrame:
    """A* = (delta / c)^(1/b) for each sweep delta, with bootstrap CI."""
    a, b, c = fit["tobit"]["a"], fit["tobit"]["b"], fit["tobit"]["c"]
    boot = fit.get("_boot_ab", np.zeros((0, 2)))
    rows = []
    for dx in dxs:
        A_star = (dx / c) ** (1.0 / b) if (np.isfinite(a) and b > 0) else np.nan
        if len(boot):
            cb = 10 ** boot[:, 0]
            bb = boot[:, 1]
            ok = bb > 0
            As = (dx / cb[ok]) ** (1.0 / bb[ok])
            lo, hi = np.percentile(As, pcts) if ok.sum() else (np.nan, np.nan)
        else:
            lo = hi = np.nan
        rows.append(dict(site=site, variant=fit["label"], dx_m=float(dx), c=c, b=b,
                         A_star_m2=float(A_star), A_star_lo=float(lo), A_star_hi=float(hi)))
    return pd.DataFrame(rows)
