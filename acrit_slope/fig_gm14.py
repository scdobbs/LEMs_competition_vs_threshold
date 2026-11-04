"""A_crit against median head slope for GM tiles 1-4 only (the tiles whose
partition heads look like hollow heads in the QC maps).

    python -m acrit_slope.fig_gm14

Reads outputs/acrit_slope/heads.csv, refits OLS and RMA on the four tiles with
the two-level bootstrap, writes figures/fig7_gm1to4_acrit_vs_slope.png and
fits_gm1to4.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from width_analysis.tiles import REPO_ROOT  # noqa: E402

from .config import A_CRIT_M2, CONFIG  # noqa: E402
from .figures import C_MUTED, C_SITE, C_TEXT, M_SITE, _style  # noqa: E402
from .fits import ci, fit_ols, fit_rma, two_level_bootstrap  # noqa: E402
from .run import _json_safe  # noqa: E402

KEYS = [("GM", 1), ("GM", 2), ("GM", 3), ("GM", 4)]


def main(argv=None):
    out_dir = REPO_ROOT / CONFIG["output_dir"]
    heads = pd.read_csv(out_dir / "heads.csv")
    Ld = f"S_up_{CONFIG['L_default_m']:g}"
    hs = {k: heads[(heads.site == k[0]) & (heads.tile == k[1]) & heads.kept][Ld].dropna().to_numpy() for k in KEYS}
    logA = {k: float(np.log10(A_CRIT_M2[k])) for k in KEYS}
    site = {k: k[0] for k in KEYS}
    med = np.array([np.median(hs[k]) for k in KEYS])
    q25 = np.array([np.percentile(hs[k], 25) for k in KEYS])
    q75 = np.array([np.percentile(hs[k], 75) for k in KEYS])
    x, y = np.log10(med), np.array([logA[k] for k in KEYS])
    rng = np.random.default_rng(CONFIG["random_seed"] + 2)
    fits = {"tiles": [f"{k[0]}{k[1]}" for k in KEYS], "n_heads": {f"{k[0]}{k[1]}": int(len(hs[k])) for k in KEYS},
            "median_S_up_10": dict(zip([f"{k[0]}{k[1]}" for k in KEYS], med.tolist())), "A_crit_m2": {f"{k[0]}{k[1]}": A_CRIT_M2[k] for k in KEYS}}
    boots = {}
    for m, fn in (("OLS", fit_ols), ("RMA", fit_rma)):
        f = fn(x, y)
        b = two_level_bootstrap(KEYS, logA, site, hs, m, CONFIG["bootstrap_n"], rng)
        boots[m] = b
        cib = ci(b["b"], CONFIG["ci_percentiles"])
        fits[m] = dict(b=f["b"], a=f["a"], ci_b=cib, r2=f["r2"], includes_minus2=bool(cib[0] <= -2 <= cib[1]),
                       boot_n_ok=b["n_ok"], boot_n_skipped=b["n_skipped"], label="GM tiles 1-4 only (n = 4): low power")
    with open(out_dir / "fits_gm1to4.json", "w") as fh:
        json.dump(_json_safe(fits), fh, indent=1)

    fig, ax = plt.subplots(figsize=(6.4, 5.0), facecolor="white")
    xg = np.logspace(np.log10(med.min()) - 0.12, np.log10(med.max()) + 0.12, 100)
    lx = np.log10(xg)
    for m, color, ls in (("OLS", C_TEXT, "-"), ("RMA", C_SITE["AP"], (0, (5, 2)))):
        f = fits[m]
        ax.plot(xg, 10 ** (f["a"] + f["b"] * lx), color=color, lw=1.8, ls=ls,
                label=f"{m}: b = {f['b']:.2f} [{f['ci_b'][0]:.2f}, {f['ci_b'][1]:.2f}], r² = {f['r2']:.2f}")
    b = boots["OLS"]
    if b["a"] is not None and len(b["a"]):
        mm = min(len(b["a"]), len(b["b"]))
        pred = b["a"][:mm, None] + b["b"][:mm, None] * lx[None, :]
        lo, hi = np.percentile(pred, CONFIG["ci_percentiles"], axis=0)
        ax.fill_between(xg, 10 ** lo, 10 ** hi, color=C_TEXT, alpha=0.08, lw=0, label="OLS 95 % bootstrap band")
    cx, cy = x.mean(), y.mean()
    ax.plot(xg, 10 ** (cy + CONFIG["reference_exponent"] * (lx - cx)), color=C_MUTED, lw=1.4, ls=(0, (4, 3)),
            label="A ∝ S⁻² through the centroid")
    ax.errorbar(med, 10 ** y, xerr=[med - q25, q75 - med], fmt="none", ecolor=C_SITE["GM"], elinewidth=1.2, capsize=2.5)
    ax.plot(med, 10 ** y, M_SITE["GM"], color=C_SITE["GM"], ms=9, mec="white", mew=1.0, ls="none",
            label="GM tiles 1–4 (horizontal bar: IQR of head slope)")
    for xi, yi, k in zip(med, 10 ** y, KEYS):
        ax.annotate(f"{k[0]}{k[1]} (n = {len(hs[k])})", (xi, yi), xytext=(7, 5), textcoords="offset points", fontsize=8, color=C_TEXT)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"median head slope, {Ld.replace('_', ' ')} m path (m/m)")
    ax.set_ylabel("optimized A_crit (m²)")
    ax.set_title("GM tiles 1–4 only: optimized A_crit against slope at the partition heads", fontsize=10, loc="left", color=C_TEXT)
    _style(ax)
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout()
    out = out_dir / "figures" / "fig7_gm1to4_acrit_vs_slope.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    for m in ("OLS", "RMA"):
        print(m, {k: fits[m][k] for k in ("b", "ci_b", "r2", "includes_minus2", "boot_n_ok", "boot_n_skipped")})
    print("wrote", out)


if __name__ == "__main__":
    main()
