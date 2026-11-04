"""Method P: Pelletier (2013) steps 1-3 in Python.

1. Optimal Wiener filter: radially averaged power spectrum; |H|^2 = power-law
   fit to the signal-dominated low frequencies; |N|^2 = mean power just below
   the Nyquist frequency; transfer function |H|^2 / (|H|^2 + |N|^2) (Pelletier
   2013, eq. 4). No free parameter.
2. Contour (tangential) curvature, Mitasova & Hofierka (1993) form (Pelletier
   eq. 1); positive = convergent.
3. Heads: the filtered DEM is hydrologically conditioned (priority-flood with
   epsilon gradients); pixels are visited from the highest to the lowest; a
   pixel is a head if its curvature exceeds the threshold and no higher
   8-neighbour is already downstream of a head ("flag all pixels located
   downstream along any downslope flow pathway", Pelletier 2013 par. 17).
   ``component_top_heads`` gives the alternative definition in the directive
   (highest pixel of each connected component of the curvature mask).

Nothing here uses drainage area.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


# ---------------------------------------------------------------------------
# 1. optimal Wiener filter
# ---------------------------------------------------------------------------

def _detrend_plane(z: np.ndarray):
    ny, nx = z.shape
    jj, ii = np.meshgrid(np.arange(nx), np.arange(ny))
    A = np.column_stack([np.ones(z.size), ii.ravel(), jj.ravel()])
    coef, *_ = np.linalg.lstsq(A, z.ravel(), rcond=None)
    plane = (A @ coef).reshape(z.shape)
    return z - plane, plane


def radial_power_spectrum(z: np.ndarray, dx: float, n_bins: int):
    """Radially averaged power spectrum S(nu) of a (detrended) grid; nu in cycles per metre."""
    F = np.fft.fft2(z)
    P = np.abs(F) ** 2 / z.size
    fy = np.fft.fftfreq(z.shape[0], d=dx)
    fx = np.fft.fftfreq(z.shape[1], d=dx)
    nu = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    nyq = 0.5 / dx
    edges = np.linspace(0, nyq, n_bins + 1)
    idx = np.clip(np.digitize(nu, edges) - 1, 0, n_bins - 1)
    valid = nu <= nyq
    S = np.bincount(idx[valid], weights=P[valid], minlength=n_bins)
    cnt = np.bincount(idx[valid], minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        S = S / cnt
    centres = 0.5 * (edges[:-1] + edges[1:])
    return centres, S, nu, nyq


def wiener_model(centres, S, nyq, cfg) -> dict:
    """Fit |N|^2 (constant, high frequencies) and |H|^2 (power law, signal-dominated low frequencies)."""
    ok = np.isfinite(S) & (S > 0)
    lo, hi = cfg["wiener_noise_band"]
    nb = ok & (centres >= lo * nyq) & (centres <= hi * nyq)
    N2 = float(np.mean(S[nb]))
    skip = cfg["wiener_skip_lowest_bins"]
    sb = ok & (np.arange(len(S)) >= skip) & (S > cfg["wiener_signal_min_ratio"] * N2) & (centres > 0)
    if sb.sum() < 5:                       # fallback: lowest tenth of the spectrum
        sb = ok & (np.arange(len(S)) >= skip) & (centres <= 0.1 * nyq) & (centres > 0)
    beta, loga = np.polyfit(np.log10(centres[sb]), np.log10(S[sb]), 1)
    return dict(N2=N2, a=10 ** loga, beta=float(beta), n_signal_bins=int(sb.sum()),
                signal_max_nu=float(centres[sb].max()), nyquist=float(nyq))


def wiener_filter(z: np.ndarray, dx: float, cfg: dict) -> tuple[np.ndarray, dict]:
    """Optimal Wiener filter of a DEM (Pelletier 2013). Returns (filtered DEM, spectrum model)."""
    pad = int(cfg["wiener_pad_px"])
    zp = np.pad(z, pad, mode="reflect")
    zd, plane = _detrend_plane(zp)
    centres, S, nu, nyq = radial_power_spectrum(zd, dx, cfg["wiener_n_radial_bins"])
    m = wiener_model(centres, S, nyq, cfg)
    with np.errstate(divide="ignore"):
        H2 = m["a"] * np.power(nu, m["beta"])      # beta < 0: infinite at nu = 0 -> phi = 1
    phi = np.where(nu > 0, H2 / (H2 + m["N2"]), 1.0)
    zf = np.real(np.fft.ifft2(np.fft.fft2(zd) * phi)) + plane
    out = zf[pad:-pad, pad:-pad]
    m["rms_change_m"] = float(np.sqrt(np.mean((out - z) ** 2)))
    return out, m


# ---------------------------------------------------------------------------
# 2. contour curvature
# ---------------------------------------------------------------------------

def contour_curvature(z: np.ndarray, dx: float) -> np.ndarray:
    """Mitasova & Hofierka (1993) tangential curvature (Pelletier 2013 eq. 1); positive = convergent.

    Derivatives by central differences. Rows increase southward, so h_y is
    taken along the row axis; the formula is symmetric in the axis sign.
    """
    hy, hx = np.gradient(z, dx, dx)
    hyy, hyx = np.gradient(hy, dx, dx)
    hxy, hxx = np.gradient(hx, dx, dx)
    g2 = hx**2 + hy**2
    with np.errstate(divide="ignore", invalid="ignore"):
        k = (hxx * hy**2 - 2 * hxy * hx * hy + hyy * hx**2) / (g2 * np.sqrt(1 + g2))
    k[g2 == 0] = 0.0
    return k


# ---------------------------------------------------------------------------
# 3. heads
# ---------------------------------------------------------------------------

def condition(z: np.ndarray, epsilon: bool = True) -> np.ndarray:
    """Priority-flood depression filling (epsilon gradients) of the filtered DEM."""
    import richdem as rd
    arr = rd.rdarray(np.ascontiguousarray(z, dtype=np.float64), no_data=-9999.0)
    filled = rd.FillDepressions(arr, epsilon=epsilon, in_place=False)
    return np.array(filled, dtype=np.float64)


def pelletier_heads(zc: np.ndarray, kappa: np.ndarray, kt: float) -> tuple[np.ndarray, np.ndarray]:
    """Pelletier (2013) valley heads on a conditioned DEM ``zc`` with curvature ``kappa``.

    Visit pixels from the highest to the lowest. A pixel is *downstream* if any
    strictly higher 8-neighbour is a head or downstream (that covers every
    descending flow pathway, not only steepest descent). A pixel is a head if
    kappa > kt and it is not downstream. Returns (rows, cols) of the heads.
    """
    ny, nx = zc.shape
    order = np.argsort(zc, axis=None)[::-1]
    flagged = np.zeros(zc.size, dtype=bool)
    head = np.zeros(zc.size, dtype=bool)
    zf = zc.ravel()
    kf = (kappa > kt).ravel()
    offs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for p in order:
        i, j = divmod(int(p), nx)
        zp = zf[p]
        down = False
        for a, b in offs:
            i2, j2 = i + a, j + b
            if 0 <= i2 < ny and 0 <= j2 < nx:
                q = i2 * nx + j2
                if zf[q] > zp and (flagged[q] or head[q]):
                    down = True
                    break
        if down:
            flagged[p] = True
        elif kf[p]:
            head[p] = True
    rows, cols = np.nonzero(head.reshape(ny, nx))
    return rows, cols


def component_top_heads(zc: np.ndarray, kappa: np.ndarray, kt: float) -> tuple[np.ndarray, np.ndarray]:
    """Alternative (directive wording): the highest pixel of each 8-connected component of kappa > kt."""
    lab, n = ndimage.label(kappa > kt, structure=np.ones((3, 3)))
    if n == 0:
        return np.array([], int), np.array([], int)
    pos = ndimage.maximum_position(zc, lab, index=np.arange(1, n + 1))
    pos = np.array(pos, dtype=int).reshape(-1, 2)
    return pos[:, 0], pos[:, 1]


def extract(z: np.ndarray, dx: float, kt: float, cfg: dict) -> dict:
    """Full Method P on one tile. Returns dict with the filtered DEM, curvature, conditioned DEM and heads."""
    zf, model = wiener_filter(z, dx, cfg)
    kappa = contour_curvature(zf, dx)
    zc = condition(zf, cfg["fill_epsilon"])
    r, c = pelletier_heads(zc, kappa, kt)
    r2, c2 = component_top_heads(zc, kappa, kt)
    return dict(z_filtered=zf, kappa=kappa, z_conditioned=zc, wiener=model,
                heads_rc=(r, c), component_heads_rc=(r2, c2), kt=kt,
                n_valley_pixels=int((kappa > kt).sum()))
