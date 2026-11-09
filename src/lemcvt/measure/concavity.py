"""Concavity theta and steepness k_s from chi-elevation, with a basin bootstrap.

Primary objective (``linear``): for each theta on a grid, fit ``z - z_out = k_s chi``
through the origin in every basin and sum the squared residuals; theta* minimises
the total. This is the steady-state relation the models assume (z = z_b + (U/K) chi
with m = theta, n = 1), so k_s at theta* is exactly the quantity the models call
U/K. Reported alongside it:

* ``disorder`` (Hergarten et al. 2016; Mudd et al. 2018): sort each basin's channel
  pixels by chi and compute ``(sum |dz| - (z_max - z_min)) / (z_max - z_min)``;
  collinear tributaries minimise it. Cross-check for theta only.
* ``ks_pooled``: at theta*, one k_s for the tile from the pooled through-origin
  regression over all basins (the value used for K0); per-basin k_s as a distribution.
* bootstrap: basins are resampled with replacement (``n_boot`` draws); theta* and
  k_s are recomputed from the per-basin SSR(theta) curves and cross products, which
  makes the bootstrap cheap. Intervals are 16-84 % and 2.5-97.5 % percentiles.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from lemcvt.measure.chi import ChannelNetwork, chi_for_thetas, relative_elevation

DEFAULT_THETAS = np.round(np.arange(0.10, 0.901, 0.01), 3)


@dataclass
class ChiFit:
    A_c: float
    thetas: list[float]
    ssr_total: list[float]
    disorder_total: list[float]
    theta_star: float
    theta_star_disorder: float
    ks_pooled: float
    ks_per_basin_median: float
    ks_per_basin_q16: float
    ks_per_basin_q84: float
    r2_pooled: float
    n_pixels: int
    n_basins: int
    n_basins_used: int
    n_edge_outlets: int
    theta_ci68: tuple[float, float] | None = None
    theta_ci95: tuple[float, float] | None = None
    ks_ci68: tuple[float, float] | None = None
    ks_ci95: tuple[float, float] | None = None
    n_boot: int = 0
    rule: str = "upstream"
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def _basin_groups(net: ChannelNetwork, min_pixels: int):
    """Indices of channel pixels per basin (excluding the outlet pixel itself, which has chi = 0)."""
    groups = []
    for o in net.outlet_index:
        idx = np.nonzero(net.basin == o)[0]
        idx = idx[idx != o]
        if idx.size >= min_pixels:
            groups.append(idx)
    return groups


def _disorder(chi: np.ndarray, z: np.ndarray) -> float:
    if chi.size < 3:
        return np.nan
    order = np.argsort(chi, kind="stable")
    zs = z[order]
    rng = zs.max() - zs.min()
    if rng <= 0:
        return np.nan
    return float((np.abs(np.diff(zs)).sum() - rng) / rng)


def fit_chi(net: ChannelNetwork, thetas: np.ndarray = DEFAULT_THETAS, min_pixels_per_basin: int = 10,
            n_boot: int = 200, seed: int = 0, rule: str = "upstream") -> ChiFit:
    thetas = np.asarray(thetas, dtype=float)
    chi = chi_for_thetas(net, thetas, rule=rule)     # (n, k)
    zrel = relative_elevation(net)
    groups = _basin_groups(net, min_pixels_per_basin)
    nb = len(groups)
    notes: list[str] = []
    if nb == 0:
        raise ValueError(f"no basin with >= {min_pixels_per_basin} channel pixels at A_c={net.A_c:g}")
    k = thetas.size
    sxx = np.zeros((nb, k))
    sxy = np.zeros((nb, k))
    syy = np.zeros(nb)
    dis = np.zeros((nb, k))
    for b, idx in enumerate(groups):
        x = chi[idx]                 # (m, k)
        y = zrel[idx]                # (m,)
        sxx[b] = (x * x).sum(axis=0)
        sxy[b] = (x * y[:, None]).sum(axis=0)
        syy[b] = float((y * y).sum())
        for j in range(k):
            dis[b, j] = _disorder(x[:, j], y)
    # per-basin through-origin fits
    with np.errstate(invalid="ignore", divide="ignore"):
        ks_b = sxy / sxx                                           # (nb, k)
        ssr_b = syy[:, None] - sxy * sxy / sxx                     # residual SS per basin per theta
    ssr_total = np.nansum(ssr_b, axis=0)
    j_star = int(np.nanargmin(ssr_total))
    theta_star = float(thetas[j_star])
    dis_total = np.nanmean(dis, axis=0)
    j_dis = int(np.nanargmin(dis_total))

    ks_pooled = float(sxy[:, j_star].sum() / sxx[:, j_star].sum())
    ss_res = float(ssr_total[j_star])
    ss_tot = float(syy.sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    ksb = ks_b[:, j_star]
    ksb = ksb[np.isfinite(ksb)]

    fit = ChiFit(
        A_c=net.A_c, thetas=thetas.tolist(), ssr_total=ssr_total.tolist(), disorder_total=dis_total.tolist(),
        theta_star=theta_star, theta_star_disorder=float(thetas[j_dis]), ks_pooled=ks_pooled,
        ks_per_basin_median=float(np.median(ksb)) if ksb.size else np.nan,
        ks_per_basin_q16=float(np.percentile(ksb, 16)) if ksb.size else np.nan,
        ks_per_basin_q84=float(np.percentile(ksb, 84)) if ksb.size else np.nan,
        r2_pooled=float(r2), n_pixels=int(sum(len(g) for g in groups)), n_basins=net.n_basins, n_basins_used=nb,
        n_edge_outlets=net.n_edge_outlets, rule=rule, notes=notes,
    )
    if theta_star in (thetas[0], thetas[-1]):
        notes.append("theta* at the edge of the theta grid")
    if nb < 3:
        notes.append(f"only {nb} basins with >= {min_pixels_per_basin} pixels; bootstrap not meaningful")
    if n_boot > 0 and nb >= 3:
        rng = np.random.default_rng(seed)
        th_bs = np.empty(n_boot)
        ks_bs = np.empty(n_boot)
        for i in range(n_boot):
            w = np.bincount(rng.integers(0, nb, nb), minlength=nb).astype(float)
            tot = np.nansum(ssr_b * w[:, None], axis=0)
            jj = int(np.nanargmin(tot))
            th_bs[i] = thetas[jj]
            ks_bs[i] = (sxy[:, jj] * w).sum() / (sxx[:, jj] * w).sum()
        fit.theta_ci68 = (float(np.percentile(th_bs, 16)), float(np.percentile(th_bs, 84)))
        fit.theta_ci95 = (float(np.percentile(th_bs, 2.5)), float(np.percentile(th_bs, 97.5)))
        fit.ks_ci68 = (float(np.percentile(ks_bs, 16)), float(np.percentile(ks_bs, 84)))
        fit.ks_ci95 = (float(np.percentile(ks_bs, 2.5)), float(np.percentile(ks_bs, 97.5)))
        fit.n_boot = n_boot
    return fit


def ks_at_theta(net: ChannelNetwork, theta: float, min_pixels_per_basin: int = 10,
                rule: str = "upstream") -> tuple[float, float]:
    """Pooled through-origin k_s and R2 at a fixed theta (e.g. the v1 or regional theta)."""
    chi = chi_for_thetas(net, np.array([theta]), rule=rule)[:, 0]
    zrel = relative_elevation(net)
    idx = np.concatenate(_basin_groups(net, min_pixels_per_basin))
    x, y = chi[idx], zrel[idx]
    ks = float((x * y).sum() / (x * x).sum())
    ss_tot = float((y * y).sum())
    r2 = 1.0 - float(((y - ks * x) ** 2).sum()) / ss_tot if ss_tot > 0 else np.nan
    return ks, r2
